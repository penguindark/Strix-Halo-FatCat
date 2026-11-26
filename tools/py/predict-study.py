#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
"""predict-study.py — offline study: can a learned inter-layer predictor beat
the stale-router expert prefetch guess (65.3% engine-measured)?

Method (matches the planned online implementation):
  - Phase 1 (prefill tokens): LEARN an inter-layer affinity table
    W[(prev_layer, e)][next_layer, f] += 1 for co-routed token slots,
    plus per-layer expert frequency.
  - Phase 2 (decode tokens): PREDICT layer L's experts from layer L-1's
    actual ids via argmax_f (W[(L-1,e)][f] + alpha*freq_L[f]), compare to
    actual. No leakage: phase-2 outcomes never feed the table.

Baselines: stale-router (predict = actual ids of L-1, the shipped method),
global frequency, and the affinity predictor at several alphas.

Metrics per predictor:
  engine-hit : fraction of predicted top-4 present in actual top-4
               (stale-router should reproduce the engine's ~65%)
  cover@2    : fraction of ACTUAL top-4 ids covered by predicted top-2
               (what FATCAT_SPECP=2 actually prefetches)
  cover@2w   : same but against the actual top-2 by router weight (the
               heavy slices that matter most for stall avoidance)

Usage: predict-study.py TRACE.fcts [--prefill N]
"""
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from read_trace import load


def main():
    path = sys.argv[1]
    meta, recs = load(path)
    topk = meta["topk"]
    by_tok = defaultdict(dict)
    for tok, layer, ids, wts in recs:
        by_tok[tok][layer] = ids
    toks = sorted(by_tok)
    n_pref = int(sys.argv[sys.argv.index("--prefill") + 1]) \
        if "--prefill" in sys.argv else 473
    pre, dec = toks[:n_pref], toks[n_pref:]

    # ---- phase 1: learn affinity + frequency from prefill only ----
    W = defaultdict(lambda: defaultdict(float))   # (prev_layer, e) -> {f: count}
    freq = defaultdict(lambda: defaultdict(float))
    for tok in pre:
        d = by_tok[tok]
        ls = sorted(d)
        for L in ls:
            for e in d[L]:
                freq[L][e] += 1
        for a, b in zip(ls, ls[1:]):
            for e in d[a]:
                we = W[(a, e)]
                for f in d[b]:
                    we[f] += 1

    def top_by(score, k):
        return [x for x, _ in sorted(score.items(), key=lambda kv: -kv[1])[:k]]

    pairs = sum(len(by_tok[t]) - 1 for t in dec)
    stats = {}

    def score_pred(name, pred_fn):
        h4 = c2 = c2n = c2w = c2wn = 0
        for tok in dec:
            d = by_tok[tok]
            ls = sorted(d)
            for a, b in zip(ls, ls[1:]):
                actual4 = set(d[b])
                actual_w2 = set(d[b][:2])
                p = pred_fn(d[a], b)
                h4 += sum(1 for x in p[:4] if x in actual4)
                c2 += len(actual4 & set(p[:2])); c2n += len(actual4)
                c2w += len(actual_w2 & set(p[:2])); c2wn += len(actual_w2)
        stats[name] = (h4 / (pairs * topk), c2 / c2n, c2w / c2wn)

    score_pred("stale-router", lambda prev, b: list(prev))
    score_pred("global-freq", lambda prev, b: top_by(freq[b], 4))

    # affinity predictor: needs (a, b), separate scorer
    def score_aff(al):
        h4 = c2 = c2n = c2w = c2wn = 0
        for tok in dec:
            d = by_tok[tok]
            ls = sorted(d)
            for a, b in zip(ls, ls[1:]):
                actual4 = set(d[b])
                actual_w2 = set(d[b][:2])
                score = defaultdict(float)
                for e in d[a]:
                    for f, c in W[(a, e)].items():
                        score[f] += c
                if al:
                    mx = max(freq[b].values())
                    for f, c in freq[b].items():
                        score[f] += al * c / mx
                p = top_by(score, 4)
                h4 += sum(1 for x in p[:4] if x in actual4)
                c2 += len(actual4 & set(p[:2])); c2n += len(actual4)
                c2w += len(actual_w2 & set(p[:2])); c2wn += len(actual_w2)
        stats[f"affinity a={al}"] = (h4 / (pairs * topk), c2 / c2n, c2w / c2wn)

    for al in (0.0, 0.1, 1.0):
        score_aff(al)

    print(f"trace: {len(toks)} tokens ({len(pre)} prefill, {len(dec)} decode), "
          f"{meta['n_layers']} MoE layers, topk={topk}, pairs={pairs}")
    print(f"{'predictor':22s} {'engine-hit':>10s} {'cover@2':>8s} {'cover@2w':>9s}")
    for name, (h, c, cw) in stats.items():
        print(f"{name:22s} {h:10.1%} {c:8.1%} {cw:9.1%}")


if __name__ == "__main__":
    main()
