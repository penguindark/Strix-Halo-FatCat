#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
"""Analyze a FatCat expert-hotness trace (FCT1) and report tier budgets.

Usage: analyze_trace.py full-gpu.trace [--gb 8 16 24 32 48 64]

Trace format: header "FCT1" u32 n_layers u32 n_experts, then f64 weighted
[n_layers][n_experts], then u64 raw counts [n_layers][n_experts].

Expert byte costs are for Mistral-Large-4 UD-IQ1_S (measured strides):
IQ1_S matrix = 7.37 MB, IQ2_XXS matrix = 9.73 MB; per expert = gate+up+down.
blk.1-6 and blk.71 are all IQ2_XXS; blk.7-70 use IQ1_S gate/up + IQ2_XXS down.
"""
import heapq
import struct
import sys

import numpy as np

IQ1S_MB, IQ2MB = 7.37, 9.73


def expert_mb(layer: int) -> float:
    # layer here is the 0-based block index as stored in the trace
    if 1 <= layer <= 6 or layer == 71:
        return 3 * IQ2MB
    return 2 * IQ1S_MB + IQ2MB


def load(path):
    buf = open(path, "rb").read()
    assert buf[:4] == b"FCT1", "not an FCT1 trace"
    nl, ne = struct.unpack("<II", buf[4:12])
    off = 12
    w = np.frombuffer(buf, "<f8", count=nl * ne, offset=off).reshape(nl, ne)
    off += nl * ne * 8
    raw = np.frombuffer(buf, "<u8", count=nl * ne, offset=off).reshape(nl, ne)
    return w, raw


def hitrate_for_budget(w, moe, ne, gb):
    """Best-case warm-tier hit rate: greedily keep the highest weight/byte
    experts per layer (prefix of the sorted hotness list) until the budget is
    spent. Assumes independent draws from the empirical distribution, i.e. a
    lower bound for an LRU policy with temporal locality."""
    taken = {L: 0 for L in moe}
    srt = {L: np.sort(w[L])[::-1] for L in moe}
    heap = [(-srt[L][0] / expert_mb(L), L) for L in moe if srt[L].sum() > 0]
    heapq.heapify(heap)
    budget, hits = gb * 1024.0, 0.0
    while heap:
        negd, L = heapq.heappop(heap)
        k = taken[L]
        if k >= ne:
            continue
        eb = expert_mb(L)
        if eb > budget:
            break
        budget -= eb
        taken[L] = k + 1
        hits += srt[L][k]
        if k + 1 < ne:
            heapq.heappush(heap, (-srt[L][k + 1] / eb, L))
    return hits / w[moe].sum(), taken


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # analyze_trace.py --compare A.trace B.trace [GB ...]
        # Domain comparison: coverage curves + best-case hit rates per domain,
        # e.g. is CODE routing concentrated enough to pin a domain expert set?
        paths = [p for p in sys.argv[2:] if not p.isdigit()]
        gbs = [int(x) for x in sys.argv[2:] if x.isdigit()] or [64, 86, 96]
        for path in paths:
            w, raw = load(path)
            nl, ne = w.shape
            moe = [L for L in range(nl) if raw[L].sum() > 0]
            tot = sum(int((raw[L] > 0).sum()) for L in moe) / len(moe)
            acts = int(raw.sum())
            print(f"== {path}: {acts} routed activations, "
                  f"{tot:.1f} distinct experts/layer (avg)")
            for name, frac in (("cov50", .5), ("cov80", .8), ("cov95", .95)):
                c = []
                for L in moe:
                    s = np.sort(w[L])[::-1]
                    c.append(int(np.searchsorted(np.cumsum(s) / s.sum(), frac) + 1))
                pin_gb = sum(expert_mb(L) * int(np.median(c)) for L in moe) / 1024
                print(f"  {name}: min/med/max {min(c)}/{int(np.median(c))}/{max(c)}"
                      f"  med pin-set ~= {pin_gb:.1f} GB")
            for gb in gbs:
                hr, taken = hitrate_for_budget(w, moe, ne, gb)
                print(f"  warm {gb:3d} GB -> best-case hit {hr * 100:5.1f}%")
        return
    path = sys.argv[1]
    gbs = [int(x) for x in (sys.argv[3:] if len(sys.argv) > 2 else
                            [8, 16, 24, 32, 48, 64, 86, 96])]
    w, raw = load(path)
    nl, ne = w.shape
    moe = [L for L in range(nl) if raw[L].sum() > 0]
    print(f"layers={nl} experts={ne} moe_layers={len(moe)} "
          f"total_routed={sum(expert_mb(L) * ne for L in moe) / 1024:.1f} GB")

    def cov(L, frac):
        s = np.sort(w[L])[::-1]
        return int(np.searchsorted(np.cumsum(s) / s.sum(), frac) + 1)

    for name, frac in (("cov50", .5), ("cov80", .8), ("cov95", .95),
                       ("cov99", .99)):
        c = [cov(L, frac) for L in moe]
        print(f"{name}: min/med/max experts = {min(c)}/{int(np.median(c))}/{max(c)}")

    for gb in gbs:
        hr, taken = hitrate_for_budget(w, moe, ne, gb)
        cnts = np.array(list(taken.values()))
        print(f"warm {gb:3d} GB -> hit {hr * 100:5.1f}%  "
              f"experts/layer min/med/max {cnts.min()}/{int(np.median(cnts))}/{cnts.max()}")


if __name__ == "__main__":
    main()
