#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
"""skip-sim — H3 cache-aware tail-skip simulation over FCTS v2 traces.

Replays per-token router decisions (ids + normalized weights) against a
simulated warm tier (S3-FIFO, mirroring the engine), and for each skip
threshold tau answers: how many cold-miss expert fetches disappear, and
how much router weight mass is dropped (the quality price).

Policy simulated (identical to the engine's FATCAT_SKIP path):
  expert k (k>0) is dropped iff it is a tier MISS and w_k < tau * w_1.
  Dropped experts are removed from the tier access sequence (never
  fetched -> never resident), which also changes future hit rates.

Usage:
  skip-sim.py TRACE.fcts [TRACE2.fcts ...] [--tier=64] [--taus=0.05,0.1,0.15,0.2,0.3]

Requires FCTS v2 (weights). Column meanings:
  cold GB/tok   streamed expert bytes per token WITHOUT skipping
  saved %       fraction of those bytes not fetched with the threshold
  drop w/tok    mean router weight mass dropped per token (all layers)
  drop %        drop w/tok relative to total routed weight (topk mass ~1/layer)
"""
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "py"))
from read_trace import load  # noqa: E402

EXPERT_BYTES = 23.5e6   # avg expert (3 IQ1_S matrices), measured


class S3FIFO:
    """Engine-mirror S3-FIFO: small 10% + main 90% + ghost ring."""

    def __init__(self, cap):
        self.small_cap = max(1, cap // 10)
        self.main_cap = cap - self.small_cap
        self.small, self.main, self.ghost = OrderedDict(), OrderedDict(), OrderedDict()
        self.cap = cap

    def access(self, k):
        """Touch k as resident-consumed; returns True if it was resident."""
        if k in self.main:
            self.main[k] = 1
            return True
        if k in self.small:
            self.small[k] = 1
            return True
        if k in self.ghost:
            del self.ghost[k]
            self.main[k] = 0
        else:
            self.small[k] = 0
        while len(self.small) > self.small_cap:
            k2, f = self.small.popitem(last=False)
            if f:
                self.main[k2] = 0
            else:
                self.ghost[k2] = 0
                while len(self.ghost) > self.cap:
                    self.ghost.popitem(last=False)
        while len(self.main) > self.main_cap:
            k2, f = self.main.popitem(last=False)
            if f:
                self.main[k2] = 0
                self.main.move_to_end(k2)
            else:
                self.ghost[k2] = 0
                while len(self.ghost) > self.cap:
                    self.ghost.popitem(last=False)
        while len(self.small) + len(self.main) > self.cap:
            k2, f = self.small.popitem(last=False)
            self.main[k2] = f
        return False


def run(records, cap, taus):
    """Returns {tau: dict} stats; tau=0.0 baseline (no skipping)."""
    stats = {t: dict(cold=0, cold_bytes=0.0, drop_w=0.0, hits=0, total=0,
                     skipped=0) for t in taus}
    tiers = {t: S3FIFO(cap) for t in taus}
    ntok = len({r[0] for r in records}) or 1
    for tok, layer, ids, wts in records:
        for tau in taus:
            tier = tiers[tau]
            s = stats[tau]
            w1 = wts[0] if wts else 1.0
            s["total"] += len(ids)
            for k, e in enumerate(ids):
                key = (layer, e)
                resident = key in tier.main or key in tier.small
                if resident:
                    tier.access(key)
                    s["hits"] += 1
                    continue
                # cold miss: would fetch EXPERT_BYTES unless skipped
                if tau > 0 and k > 0 and wts and wts[k] < tau * w1:
                    s["skipped"] += 1
                    s["drop_w"] += wts[k]
                    continue
                tier.access(key)
                s["cold"] += 1
                s["cold_bytes"] += EXPERT_BYTES
    for t, s in stats.items():
        s["gb_tok"] = s["cold_bytes"] / 1e9 / ntok
    base = stats[min(taus)]["gb_tok"] if 0.0 in taus else None
    return stats, ntok, base


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    tier_gb, taus = 64, [0.05, 0.10, 0.15, 0.20, 0.30]
    for a in sys.argv[1:]:
        if a.startswith("--tier="):
            tier_gb = float(a.split("=")[1])
        if a.startswith("--taus="):
            taus = [float(x) for x in a.split("=")[1].split(",")]
    if not paths:
        print(__doc__)
        sys.exit(2)
    taus = sorted(set(taus) | {0.0})
    for path in paths:
        meta, records = load(path)
        if meta["version"] < 2 or not records[0][3]:
            print(f"{path}: FCTS v2 with weights required (got v{meta['version']})")
            continue
        cap = max(8, int(tier_gb * 1.073e9 / EXPERT_BYTES))
        ntok = len({r[0] for r in records})
        print(f"\n=== {os.path.basename(path)}: {len(records)} records, "
              f"{ntok} tokens, tier {tier_gb:g} GB ({cap} expert slots) ===")
        stats, _, _ = run(records, cap, taus)
        base_gb = stats[0.0]["gb_tok"]
        hr = 1 - stats[0.0]["cold"] / max(1, stats[0.0]["total"])
        print(f"baseline: hit {100*hr:.1f}%  cold {base_gb:.2f} GB/tok")
        print(f"{'tau':>5} {'saved%':>7} {'cold GB/tok':>12} {'drop w/tok':>11} "
              f"{'drop %':>7} {'skip/tok':>9}")
        nlayers = len({r[1] for r in records}) or 1
        for t in taus:
            if t == 0.0:
                continue
            s = stats[t]
            saved = 100 * (1 - s["gb_tok"] / base_gb) if base_gb else 0
            dw = s["drop_w"] / ntok
            print(f"{t:5.2f} {saved:6.1f}% {s['gb_tok']:12.2f} {dw:11.4f} "
                  f"{100*dw/nlayers:6.2f}% {s['skipped']/ntok:9.1f}")


if __name__ == "__main__":
    main()
