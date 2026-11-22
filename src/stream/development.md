# src/stream — development.md

## Goal
The heart of FatCat: the tiered expert streaming manager.

- **Hot tier**: experts pinned in the GPU-resident region (GTT carve-out or
  pinned pages) for direct iGPU access.
- **Warm tier**: experts resident in system RAM (unified LPDDR5X).
- **Cold tier**: experts on NVMe, streamed on demand via `io_uring`.

Popularity counters (LFU + recency) drive promotion/eviction; prefetching
overlaps SSD reads with compute. On Strix Halo RAM and VRAM are the same
physical memory, so tiers are a *policy* (pinning + scheduling), not copies
between devices.

## Status
**P3 v1 complete (2026-11-17).** Warm tier implemented and integrated:
`stream.c` slot arena (10 MB slots, `FATCAT_WARM_MB` budget, default 32768
for standalone use), LFU+recency eviction (`count/(1+age/64)`), open-addressing
key table, use-count refs (`get`/`release`) so in-use slices are never evicted,
condvar coalescing of concurrent same-key gets, per-token recency tick.
`ml4.c` fetches each layer's 12 expert slices through the tier (parallel
preads) and fadvise(DONTNEED)s the file range on miss. Unit tests:
`tests/test_stream.c` ALL PASS. E2E: warm-tier logits bit-identical to the
mmap path. Measured hit rates: 13.1% @8 GB (below per-token working set),
23.5% @16 GB over 512 tokens — policy tuning + async prefetch are the gap to
the 45.8% static bound in `docs/streaming-policy.md`.

**P3 v2 complete (2026-11-17):** io_uring fill worker (`uring.c`, raw
syscalls; `FATCAT_URING=0` -> serial-pread fallback) +
`fatcat_stream_prefetch()`. ml4.c prefetches each layer's 12 slices right
after its router, overlapping NVMe with the shared-expert FFN. 8-token e2e
with a 2 GB tier: 44.9% of requests non-blocking, wall 44.7 s vs 62 s
no-tier (-28%), logits bit-identical.

**P3 v2.1 (2026-11-17):** deferred per-slice waits — `expert_slice()` waits
per matrix inside the GEMV loop, so expert k's compute overlaps the layer's
remaining in-flight reads; worker adds 200 us burst coalescing for deep
io_uring batches. Critical perf finding: libgomp spin-waiting costs ~3x wall
under fine-grained I/O waits; `OMP_WAIT_POLICY=passive` must be set before
process load (logit-dump self-re-execs to default it; library embedders
must export it themselves). Result: 18 s vs 62 s baseline (3.4x) at a 2 GB
tier, user CPU 24 min -> 1.5 min, logits bit-identical.

**P3 1A router speculation (2026-11-17):** the next MoE layer's router runs
on the pre-FFN hidden state and its top-4 slices are prefetched one layer
early (issued after the current layer's critical reads; `FATCAT_SPEC=0`
disables). Measured 74.3% guess accuracy; tier hit 82.4% @2 GB. At this
tier size the engine is NVMe-bandwidth-bound, so the speculation win shows
at larger budgets — measure with `tools/budget-sweep.sh` (standalone).

**Memory reality check:** process baseline RSS is ~16 GB (mmap'd shared
weights) ON TOP of the tier — budget the tier with that in mind.

**Policy: S3-FIFO is the DEFAULT (Session 36).** `FATCAT_POLICY=lfu`
restores the legacy LFU/recency sampling. Flip decided by the standalone GPU
A/B at an 18 GB tier (report ab-policy-20261119-1313.txt): s3fifo hit 77.6%
vs lfu 74.6%, evictions -8.5%, wall -8.5%, logits bit-identical (1000/1000).
`fatcat_stream_policy()` reports the active name; logit-dump prints it in
the warm-tier stats line.
**Memory discipline (hard rule):** the box hosts the assistant LLM (~68 GB
RSS). Agent-run tests must use `FATCAT_WARM_MB<=6144`,
`FATCAT_MEM_LIMIT_MB<=15000`, short corpora; full-budget runs are the user's,
run standalone. `logit-dump` prints peak RSS.

## Scope
- In: expert residency table, hit-rate statistics, promotion/eviction policy,
  io_uring read queue, prefetch hints from router logits, warm-up/cold-start
  strategy, per-layer budget configuration.
- Out: GPU copy scheduling details (P4, `src/backend/gpu/`).

## Design notes
- **Hard RAM budget**: the tier manager enforces a configurable cap
  (`FATCAT_RAM_BUDGET`) on all resident bytes; at low budgets it degrades
  gracefully (more cold hits, slower but correct). Budget-sweep curves
  (tok/s and hit-rate vs budget) are a core deliverable — see
  `../../docs/performance.md` §3bis.
- **Slice testing**: all early tests run on small model slices extracted by
  `tools/gguf-slice` (shared-machine constraint).
- Performance doctrine: `../../docs/performance.md` (§3 cache design). A cold
  miss costs 8–19 ms (~10–20% of the token budget): misses must be prefetched
  or eliminated, never on the critical path.
- One expert ≈ 57 MB @4-bit → with 128 GB RAM only a fraction of the 9,088
  routed experts fit warm; the profiler (`tools/`) must measure real hotness
  before policies are tuned.
- Streaming granularity: whole expert (all 3 matrices contiguous in file) —
  requires GGUF tensors laid out per-expert contiguously, page-aligned reads.

## Next steps
1. Optional: more/longer traces to map where s3fifo vs lfu crossover lies
   (lfu ~1pt ahead at 1 GB; s3fifo wins from ~8 GB up).
2. Budget-sweep hit-rate curves at realistic tier sizes (standalone runs).
3. Optional: per-layer policy/budget configuration.

## Shrink (Session 35)
- `fatcat_stream_shrink(st, keep_bytes)`: emergency reclaim for the
  `FATCAT_MEM_BUDGET_GB` hard cap — evicts resident slots (payload munmap'd)
  in policy order until used fits, skipping in-flight/pinned/referenced
  slots. ml4 calls it at KV growth to make room before realloc'ing the KV.
