# Expert Streaming Policy

Status: **measured** (P1 complete 2026-11-02) — numbers below come from the
full-model GPU profile: 949 tokens, 71 MoE layers × 128 experts, UD-IQ1_S.
Raw data: `/home/user/data/prompts/full-gpu.trace` (FCT1) + `profile-gpu.log`;
reproduce with `tools/analyze_trace.py`. Authoritative status: `../project.md`.

## Measured hotness (949-token profile corpus)

- Routed-expert payload: **221.3 GB** total (expert = gate+up+down;
  24.5 MB on blk.7–70, 29.2 MB on blk.1–6/71).
- The distribution is **much flatter than hoped**: a median layer touches
  126/128 distinct experts in 949 tokens. Coverage per layer (median):
  50% of traffic needs **11** experts, 80% needs **37**, 95% needs **74**,
  99% needs **100**.
- Best-case warm-tier hit rate (greedy weight/byte allocation, independent-draw
  lower bound — real LRU gains extra from temporal locality):

  | warm budget | hit rate | experts/layer (med) |
  |---|---|---|
  | 16 GB | 45.8% | 9  |
  | 32 GB | 61.6% | 19 |
  | 48 GB | 71.9% | 28 |
  | 64 GB | 79.4% | 36 |

- Consequence: **no static pin set reaches high hit rates**; the engine must
  rely on (a) per-layer LFU+recency tiers, (b) aggressive prefetch that hides
  the ~7–9 ms NVMe read per expert behind compute, and (c) longer-context
  hot-set reinforcement (the 949-token corpus is a pessimistic short sample).
- Per-layer allocation should be proportional to traffic/expert-bytes
  (the greedy sweep above already does this: hot layers like blk.5 get fewer
  slots, flat layers more, automatically).

## Tier model
- **Hot**: pinned for GPU access (GTT carve-out or `hipHostRegister`/userptr).
  Budget set by BIOS GTT size and `FATCAT` config.
- **Warm**: mapped in RAM (page cache or explicit hugepage arena).
- **Cold**: on NVMe, read via io_uring into a warm-tier slot.

## Policy (initial proposal)
- Popularity score per expert: exponential recency decay + total count
  (LFU+LRU hybrid), updated per router hit.
- Promotion: cold→warm on first hit (async), warm→hot when score exceeds the
  hot-tier threshold.
- Eviction: lowest score first; never evict an expert in flight.
- Prefetch: router logits of layer L can hint layer L+1 candidates; also
  prefetch the top cold candidates per layer during idle I/O windows.
- Warm-up: optional offline pass over representative prompts to pre-populate
  tiers before serving.

## Policy (current, Session 33–36)
Trace replay (`tools/stream-sim` over FCTS per-token sequences) replaced the
initial LFU+recency proposal:
- **S3-FIFO is the DEFAULT** (`FATCAT_POLICY=lfu` restores the legacy
  policy): small FIFO (10% of slots) filters one-hit wonders, main FIFO
  (90%) holds promoted residents via a freq bit, ghost table remembers
  evicted keys so re-referenced keys re-enter main. Evidence: trace replay
  on mixed977 (30.4/42.1/59.0/75.0% vs 17.2/34.3/54.5/74.1% for LFU at
  8/16/32/64 GB) and a standalone GPU engine A/B at an 18 GB tier
  (Session 36: hit 77.6% vs 74.6%, -8.5% evictions, -8.5% wall time,
  logits bit-identical 1000/1000). Victims are queue heads — cheaper than
  the sampled scoring it replaces.
- Caveat: at a 1 GB tier (far below the working set) LFU was ~1pt ahead;
  the opt-out exists for such regimes.
- The LFU+recency fallback underperformed even plain LRU on measured traces
  (frequency stickiness hurts when the working set cycles).
- A keeping-quota hot core (protected region for the hot decile) was measured
  and **rejected**: no gain over flat policies on these traces.

## Open questions (P1 answers these)
- ✅ Answered: hotness skew is low (see table above; median cov95 = 74/128).
- ✅ Answered: per-layer budget ∝ traffic/bytes via greedy sweep
  (`tools/analyze_trace.py`).
- ✅ Answered: stacked expert tensors are uniform-stride contiguous — one
  `pread` per matrix (3 per expert) suffices, no repack needed (validated by
  the profiler's streaming path: 6.36 TB read at ~3.3 GB/s avg).
- ✅ Answered (P5+, Session 32): per-token expert sequences are now traceable
  — FCTS v1 format (`tools/trace_seq.h`, `--trace-seq` on logit-dump and
  fatcat-run, validator/loader `tools/py/read_trace.py`). Recorded traces:
  `/home/user/data/fatcat-traces/{p5.fcts, mixed977.fcts}`. Sequence-domain
  skew: top-10% experts carry mean 41–46% of activations per layer (25–66%
  range) — 4x uniform. Per-token sequence replay (stream-sim) DONE
  (Session 33): it selected S3-FIFO and rejected the keeping-quota idea.
