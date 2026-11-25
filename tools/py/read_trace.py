#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
"""read_trace.py — inspect/validate FCTS v1/v2 sequence traces (tools/trace_seq.h).

Usage:
  read_trace.py TRACE.fcts            # summary report
  read_trace.py TRACE.fcts --ids      # dump all records (token layer ids weights...)

The record list is also importable: `load(path) -> (meta, records)` where
records is a list of (token, layer, [expert_ids], [weights]) — v1 traces
carry no weights, so weights is an empty list there. Input for stream-sim
and skip-sim.
"""
import struct
import sys
from collections import Counter


def load(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"FCTS":
            raise ValueError(f"not an FCTS trace: {magic!r}")
        ver, n_layers, n_experts, topk = struct.unpack("<IIII", f.read(16))
        if ver not in (1, 2):
            raise ValueError(f"unsupported FCTS version {ver}")
        rec_bytes = 8 + 2 * topk + (4 * topk if ver == 2 else 0)
        meta = {"version": ver, "n_layers": n_layers,
                "n_experts": n_experts, "topk": topk}
        records = []
        while True:
            buf = f.read(rec_bytes)
            if len(buf) < rec_bytes:
                break
            tok, layer = struct.unpack("<II", buf[:8])
            ids = list(struct.unpack(f"<{topk}H", buf[8:8 + 2 * topk]))
            if ver == 2:
                wts = list(struct.unpack(f"<{topk}f", buf[8 + 2 * topk:]))
            else:
                wts = []
            records.append((tok, layer, ids, wts))
    return meta, records


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    meta, records = load(sys.argv[1])
    if "--ids" in sys.argv:
        for tok, layer, ids, wts in records:
            print(tok, layer, *ids, *wts)
        return
    tokens = sorted({t for t, _, _, _ in records})
    layers = sorted({l for _, l, _, _ in records})
    per_layer = Counter(l for _, l, _, _ in records)
    print(f"FCTS v{meta['version']}: n_layers={meta['n_layers']} "
          f"n_experts={meta['n_experts']} topk={meta['topk']}")
    print(f"records={len(records)}  tokens={len(tokens)} "
          f"(0..{tokens[-1] if tokens else '-'})  moe_layers={len(layers)}")
    expect = len(tokens) * len(layers)
    ok = "OK" if len(records) == expect else "MISMATCH"
    print(f"completeness: {len(records)} == {len(tokens)}x{len(layers)}"
          f"={expect} {ok}")
    # hotness: share of activations held by the top 10% experts per layer
    shares = []
    for l in layers:
        c = Counter()
        for t, ll, ids, _ in records:
            if ll == l:
                c.update(ids)
        tot = sum(c.values())
        top = sum(v for _, v in c.most_common(max(1, meta["n_experts"] // 10)))
        shares.append(top / tot)
    if shares:
        print(f"top-10%-expert activation share: mean={sum(shares)/len(shares):.2f} "
              f"min={min(shares):.2f} max={max(shares):.2f} "
              f"(uniform would be 0.10 -> skew justifies a hot-core cache)")


if __name__ == "__main__":
    main()
