# project.md — FatCat Master Development Tracker

> Author: Dario Deledda (MIT License, see `LICENSE`).

> FatCat: a streaming LLM inference engine specialized for
> **mistralai/Mistral-Large-4-1T-A52B** on **AMD Strix Halo (gfx1151)**.

Last updated: 2026-11-01

## 1. Vision

Run a ~1T-parameter MoE (≈515 GB at 4-bit) on a single Strix Halo machine by
treating memory as a three-tier hierarchy for the routed experts:

| Tier | Location | Role |
|---|---|---|
| Hot | GPU-resident region (GTT carve-out / pinned pages) | Most-used experts, always available to the iGPU |
| Warm | System RAM (unified LPDDR5X) | Less-used experts, fast host access |
| Cold | NVMe SSD | All remaining experts, streamed on demand via `io_uring` |

On Strix Halo RAM and "VRAM" are the same physical unified memory, so the tiering
is a **policy** (pinning, prefetch scheduling, eviction) rather than physical
device copies. The real bandwidth wall is NVMe → RAM and the ~256 GB/s RAM
ceiling.

## 2. Target model facts (from upstream `config.json`, see `Manuals/model/`)

- 72 layers: layer 0 dense, layers 1–71 MoE.
- MoE: **128 routed experts + 1 shared expert, top-4 routing** per token.
- `hidden_size=6144`, `moe_intermediate_size=6144` → each expert ≈ 113M params
  (gate/up/down) ≈ **~57 MB at 4-bit**.
- 71 × 128 = 9,088 routed experts ≈ **~515 GB total at 4-bit**.
- Attention: MLA-style (`kv_lora_rank=512`) plus a sparse **indexer**
  (DSA-style, `index_topk=2048`) — reduces KV bandwidth, important on Strix Halo.
- Native weights FP8 (128×128 block), YaRN RoPE, 1M context, vocab 204,800.
- Multimodal (Pixtral vision tower) — **out of initial scope** (Phase P6).

## 3. Locked decisions

| Decision | Choice | Date |
|---|---|---|
| Language | C (C23), HIP only where unavoidable | 2026-11-01 |
| Platform | Strix Halo (gfx1151) only | 2026-11-01 |
| Build system | CMake | 2026-11-01 |
| Weight input | **GGUF** (reuse existing quants, parse natively) | 2026-11-01 |
| GPU stack | ROCm 10.1 (TheRock gfx1151) at `/home/user/rocm-10.1/install`; rocBLAS/hipBLASLt first, custom HIP kernels later | 2026-11-01 |
| Scope v1 | Text-only (vision tower deferred to P6) | 2026-11-01 |
| Documentation language | English only (see `AGENT.md`) | 2026-11-01 |
| Model footprint | **GGUF quants** of Mistral-Large-4, smallest footprint with best quality that works in **128 GB RAM** with **256K context**; routed experts streamed from NVMe (model ≈ 260–600 GB depending on quant) | 2026-11-01 |
| GGUF repack | Per-layer/tier **mixed quantization** via GGUF modification (hot experts higher bits, cold lower, expert-contiguous layout) is planned but **deferred to a later step** | 2026-11-01 |
| Dev workflow | Strix Halo also hosts the assistant LLM → development and tests start on **small slices of the model** (few layers, small GGUF extracts) to validate concepts without exhausting shared RAM | 2026-11-01 |
| RAM budget | The engine must support a **configurable RAM budget** (hard cap on resident bytes): it must stay functional — even if much slower — at very low memory budgets, to test low-memory token generation | 2026-11-01 |
| First test quant | **`unsloth/Mistral-Large-4-1T-A52B-GGUF` → `UD-IQ1_S`** (~1.7 bpw, 233 GB, 6 splits): smallest footprint to validate all streaming/compute concepts end-to-end; quality is low by design (mechanism testing, not output quality). Downloading to `/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/` | 2026-11-01 |
| Performance | Maximum tokens/s is the primary objective; custom low-level matrix kernels (AVX-512 + HIP) are in scope wherever libraries fall short — see `docs/performance.md` | 2026-11-01 |

## 3bis. Performance objective (governing)

The engine must generate the **maximum possible tokens/s** on Strix Halo.
Full doctrine: `docs/performance.md`. Key numbers (4-bit weights):

- Decode is bandwidth-bound: ~26 GB of active weights per token → theoretical
  ceiling **~9.8 tok/s** @256 GB/s, realistic target **≥ 6 tok/s** (T2).
- One cold expert miss from NVMe costs 8–19 ms (~10–20% of the token budget):
  misses must be prefetched or eliminated, never on the critical path.
- Smart expert cache: LFU+recency hybrid policy, per-layer budgets from real
  traces, router-lookahead prefetch via io_uring, warm-up pass.
- Every optimization must be backed by measurements (`benchmarks/`,
  `development_log.md` entries record tok/s / GB/s numbers).

## 4. Roadmap

- **P0 — Scaffolding** ✅: repo structure, tracking docs, `Manuals/`, CMake stub.
- **P1 — GGUF layer** ✅: GGUF v3 reader (mmap), tensor index, expert-usage
  profiler measured hotness on the full Mistral-Large-4 GGUF (949 tokens,
  GPU-accelerated; see `docs/streaming-policy.md`).
- **P2 — Correct forward pass (CPU)** ✅: P2a engine core (`ba90337`),
  P2c numerical correctness (`5227706` — proven correct vs golden f64
  forward, more accurate than llama.cpp), P2b AVX-512 kernels (fused
  dequant-GEMV Q8_0/Q6_K/IQ1_S/IQ2_XXS, 4.7–29× single-thread).
- **P3 — Streaming engine** ✅ complete: warm tier (`src/stream/`: LFU+recency
  slot arena, refs) + io_uring prefetch + deferred per-slice waits + passive
  OMP + router-speculation lookahead (74.3% guess accuracy). 8-token e2e with
  a 2 GB tier: **~17 s vs 62 s baseline (3.6x)**, 82.4% tier hit rate,
  bit-identical logits. `tools/budget-sweep.sh` measures the budget curve
  (run standalone). Library users must export `OMP_WAIT_POLICY=passive`.
- **P4 — GPU backend** ✅ v3: wired into the engine; zero-copy warm tier;
  async pipelined GEMV (2 syncs/layer); optimized HIP kernels (vectorized
  dequant-GEMV + device-x staging): IQ1_S 4->51-74, Q8_0 11->139-217,
  Q6_K 21->97-142 GB/s (read ceiling 236). GPU best 22.9 s vs CPU 17-19 s
  on the shared box; verdict unchanged: APU GPU shares system-RAM
  bandwidth with the CPU, CPU AVX-512 stays mainline. GPU optional
  (`FATCAT_GPU=0`/`FATCAT_GPU_PIPE=0`). Re-measure on an idle GPU.
- **P5 — Sparse attention + bandwidth tuning** ✅: v1 ✅ — DSA lightning
  indexer implemented (44 full layers, ikv cache, top-k selection, sparse
  MQA; `FATCAT_IDX`/`FATCAT_IDX_TOPK` knobs); validated at the real
  top_k=2048 on a 2100-token run: bit-identical to dense below the
  threshold, sparse activates exactly at 2048 with cos 0.994..1.000 vs
  dense and stable memory. Scorer is bandwidth-bound (AVX-512 transposed
  probe gave no win; reverted). v2 ✅ — real f16 MLA KV storage
  (`FATCAT_KV_F16=1`, opt-in): halves KV 38 GB→19 GB at 256K, cos
  0.992..0.998 vs f32, f32 path bit-identical. v3 ✅ — opt-in transparent
  hugepages (`FATCAT_KV_HUGEPAGE=1`, numerically transparent). v4 ✅ —
  KV pre-allocation (`FATCAT_KV_TARGET=<tokens>`): allocates KV+indexer
  caches at a fixed capacity instead of lazy doubling (which overshot up
  to 2x: ~57 GB vs 38.7 GB at 256K f32); numerically transparent. GTT/
  NUMA/THP guidance documented in `Manuals/io/memory-tuning.md` (Strix
  Halo is single-NUMA: no binding needed). Deferred to idle-box runs:
  hugepage + f16 perf validation at 256K.
- **P6 — Multimodal (optional)** ⬜: Pixtral vision tower.

## 5. Current status

- **Repo**: local git repository initialized in this folder (`main` branch);
  `build/` and `.venv/` are gitignored. Commit after each meaningful step.
- **Python convention**: analysis/reference scripts run from `.venv/`
  (`python3 -m venv .venv && .venv/bin/pip install numpy gguf tokenizers`).
  `gguf-py` doubles as the golden reference for dequantization correctness.
- **Phase: P1 (GGUF layer) — ✅ complete.**
  - ✅ GGUF v2/v3 reader (`src/io/gguf.c`): mmap, bounds-checked, all KV types,
    split GGUF (llama.cpp semantics), quant size table. Unit tests pass.
  - ✅ `tools/gguf-inspect` + full IQ1_S download (233 GB, 6 parts) validated;
    real expert layout measured (stacked tensors, avg expert ≈ 23.5 MB,
    19 GB always-resident).
  - ✅ `tools/gguf-slice` — layers 0–2 + embeddings + lm_head extracted to
    `/home/user/data/models/fatcat-slices/ML4-IQ1_S-layers0-2.gguf`
    (77 tensors, 10.61 GB, self-validating writer).
  - ✅ Scalar dequant kernels F32/F16/Q8_0/Q6_K/IQ2_XXS/IQ1_S bit-exact vs
    gguf-py golden data (`test_dequant` PASS).
  - ✅ Full ML4 forward (`src/model/ml4.c`): absorbed MLA, LeRoPE+YaRN, SSMax,
    dense+MoE FFN, NoPE layers, KV sharing, per-tensor expert strides.
    Verified vs llama.cpp on full model: pearson corr 0.997, top-1000
    overlap 936/1000 (1-token test).
  - ✅ GPU backend (HIP, gfx1151): fused dequant+GEMV kernels for
    Q8_0/Q6_K/IQ1_S/IQ2_XXS/F32 (`src/gpu/`), fixed weights uploaded once,
    experts streamed via 16 LRU-tagged 10 MB staging slots; CPU fallback
    per-tensor on upload failure. Kernels validated ~1e-7 relerr (`gputest`).
  - ✅ Expert-usage profiler: full 949-token standalone run healthy — 0.49
    tok/s, memory flat ~18 GB, upload_fails=0, rc=0. Hotness analyzed
    (`tools/analyze_trace.py`) → budgets in `docs/streaming-policy.md`.
- **P2 complete. P3 v1 complete** (warm tier + refs + tests). Next: **P3 v2** —
  io_uring async prefetch overlapping compute, router-lookahead hints, policy
  tuning toward the 45.8% static bound in `docs/streaming-policy.md`.
  **Agent test memory rule: hard cap — run every engine test under
  `tools/mem-run 12288 ...` (watchdog kills if the process's *anonymous*
  memory exceeds 12 GB; resident file pages are reclaimable cache and are
  excluded); `FATCAT_WARM_MB<=1024`, `FATCAT_MEM_LIMIT_MB<=8000`,
  `FATCAT_KV_TARGET` set to the corpus length, short corpora only**
  (host LLM needs the RAM).

### Recent progress (Sessions 28–35, 2026-11-19)

- P5 v4 ✅: KV pre-allocation (`FATCAT_KV_TARGET`) + system memory-tuning
  doc (`Manuals/io/memory-tuning.md`).
- Test infrastructure: `tools/mem-run` hard anonymous-memory watchdog
  (enforces the 12 GB agent test cap); `fatcat-run` prints peak RSS.
- E2E generation coherence check ✅ (Session 29); GPU + full P5 stack
  (indexer + f16 KV) validated together ✅ (Session 30).
- `tools/fatcat-chat` ✅ (Session 31): interactive chat CLI
  (`--system/--context[-file]`, `/help /clear /status /exit`, per-response
  tok/s + ctx% stats, persistent KV via `fatcat-run --serve`).
- Hot-cache Phase A ✅ (Session 32): FCTS v1 per-token expert sequence
  traces (`--trace-seq`, `tools/py/read_trace.py`); recorded traces at
  `/home/user/data/fatcat-traces/`; measured hotness skew (top-10% experts
  carry 41–46% of activations).
- Hot-cache Phase B ✅ (Session 33): `tools/stream-sim` policy replay —
  S3-FIFO wins (+77% relative hit @ 8 GB tier); keeping-quota rejected by
  data; current LFU default underperforms plain LRU on traces.
- Hot-cache Phase C ✅ (Sessions 34/36): S3-FIFO implemented in `stream.c`;
  won the standalone GPU A/B at 18 GB (77.6% vs 74.6% hit, -8.5% wall) →
  **default policy is now s3fifo** (`FATCAT_POLICY=lfu` = opt-out).
- `FATCAT_MEM_BUDGET_GB` hard cap ✅ (Session 35): open-time budget
  planner (KV clamp + warm clamp) + runtime warm-tier shrink at KV growth
  + growth refusal; `fatcat_stream_shrink()`; engine `mem_budget_mb`.
- Speed workstream Phase D1 ✅ (Session 38): batched prefill
  (`forward_batched()`, **DEFAULT ON chunk 256**; `FATCAT_BATCH=0` restores
  the sequential path): layer-outer chunks, per-expert token grouping, one
  slice load per expert per chunk, union prefetch. Logits bit-identical to
  the sequential path (maxdiff 0.0, CPU + GPU tiers). Standalone GPU A/B @
  18 GB: 1.0 → 3.41 tok/s (chunk 512), tier hit 78.3% → 100%. Next: D2
  GPU staging fix (scope shrank: zero-copy wrap now handles tier slots),
  batched GEMM kernels, D0 fabric-clock hold — `docs/prefill-speed-study.md`.
- Batched GEMM kernels ✅ (Session 38b): CPU AVX-512 + GPU batched row
  kernels (one weight decode serves ≤8 tokens), bit-identical; CPU e2e
  +18–29%, GPU A/B NEUTRAL → the GPU wall is ~700 per-token dense-op
  round-trips, not expert weight handling. D0 fabric-clock hold tool ✅
  (`tools/fabric-clock.sh`).
- Speed workstream Phase D4 stage 1 ✅ (Session 39): batched the dense
  per-token ops in `forward_batched()` — wq/wkv_a/wo attention
  projections, shared-expert FFN, layer-0 dense FFN now one GEMM per
  weight per chunk (new Q8_0/Q6_K batched kernels CPU+GPU, bit-identical
  by construction); attention middle (RoPE, k-absorption, KV publish,
  indexer, MQA scan, SSMax) stays sequential via `attention_core()`.
  Router stays per-token (F32). Round-trips/token ~700 → ~4 at chunk
  256. CPU e2e bit-identical (policy100 + mixed500, maxdiff 0.0). GPU
  bit-identical after building kernels.hip with `-ffp-contract=off`
  (gemv/gemm FMA-contraction drift had broken bit-identity; explicit
  fmaf pinning MISCOMPILES on this ROCm — do not reintroduce). GPU
  A/B @ 18 GB: 3.41 → ~3.5 tok/s (+2–6%: the remaining wall is the
  expert-side per-expert staging/launch/sync cycles, not dense ops).
  Stage 2 (indexer batching) optional; bigger prefill lever = async
  expert staging.
- Speed workstream Phase D3 ❌ NEGATIVE (Session 40): prompt-lookup
  speculative decode implemented (`FATCAT_SPEC=1`, default OFF) and proven
  lossless (BYTE-IDENTICAL sequences on 7 corpus-runs incl. GPU, via new
  `fatcat_ml4_forward_rows()` + `fatcat_ml4_rewind()`), but it LOSES on
  every venue: GPU @ 18 GB −45…−70%, CPU −13…−23%. Root cause: decode is
  expert-weight-streaming-bound, not round-trip-bound — m verification
  tokens touch ~m×topk distinct experts per layer (no union reuse at
  small m), and rejected tokens burn expert bytes too, so speculation
  increases streamed bytes per emitted token. Generalized lesson (with
  D4): in a streaming-MoE engine, anything that computes discardable
  tokens is a loss; the dominant cost is per-token expert bytes. Code
  kept (correct, gated off); `tools/spec-decode-test.sh` reproduces.
- Speed workstream Phase D5 ✅ (Sessions 40–42): decode 0.55 → **1.34–1.37
  tok/s** @ 64 GB tier (+143%), prefill ~2.4 tok/s. Components: D5a
  profiler (`FATCAT_PROF=1`, phase + routed split), D5b GPU spin-wait +
  queued attention projections, D5d indexer skip when kv_cap ≤ top-k,
  D5e GPU absorb + batched wv kernels (attn 1.25 → 0.105 s/tok; NoPE
  row-stride bug caught by the `FATCAT_GPU_ATT=2` in-engine verify mode),
  D5f byte war: decode is NVMe-bandwidth-bound (measured 4.6–6 GB/s
  device reads ≈ whole token time), so the levers were bytes — warm tier
  64 GB (curve still climbing: 0.69/1.07/1.34 tok/s at 18/40/64 GB;
  standalone tier guidance retargeted to the 110 GB budget: up to
  ~96 GB warm at short context, ~84 GB at 256K KV),
  expert speculation prefetch OFF by default (`FATCAT_SPECP=0`, was 74%
  of ring NVMe bytes at depth 4), hot-weight pinning measured flat
  (`FATCAT_PIN=1` opt-in only). Decode-only stats window + per-process
  io probe (`decode nvme:` line) distinguish engine vs system bytes.
- External optimization-proposal audit ✅ CLOSED (Sessions 43–43f): six
  externally-proposed levers tested against measurements, five rejected
  with data — H1 unified O_DIRECT pool (GGUF data_start is 32B-aligned:
  O_DIRECT impossible; 94 GB tier math ignores GPU GTT, S42f), H2
  multi-token spec-decode (oracle benchmark `--spec-oracle` +
  `tools/spec-oracle.sh`: batched verify costs R(K)>K bytes AND time at
  verify-sized chunks — no union reuse below ~256 tok chunks, S43b), H3
  cache-aware cold-tail skip (FCTS v2 router weights + `tools/skip-sim.py`
  on a real trace: <1% byte savings at any sane tau, `FATCAT_SKIP` stays
  measurement-only, S43d), H4 per-expert GGUF repack (bandwidth-bound,
  not queue-depth-bound — priority-queue A/B was flat), H5 GPU/IO overlap
  (already implemented: union prefetch + async GEMV pipe). **H6 accepted
  (S43e/f): prefill anti-thrash `FATCAT_NOPROMO=2` is the new DEFAULT —
  prefill tier misses promote only on a 2nd reference within the same
  prefill: decode +5.9%, prefill +9.0% in the standalone A/B.** Net of
  the audit: ~+5% decode for free.
- Pista A — the 1.2 GB/tok "non-ring" device-read mystery ✅ SOLVED
  (Session 44): NOT staging double-reads, NOT mmap refaults (majflt
  counter added, ~0), NOT a hidden read site (`tools/io_sampler.py`
  attributes every byte per-thread: the fill worker alone reads at line
  rate). Root cause: this box runs `read_ahead_kb=4096`, so the kernel
  dragged up to 4 MB of dead bytes after every random ~8 MB ring read
  (15.7k reads/tok-window x 4 MB == the 63 GB remainder exactly). Fix:
  the stream ring now reads through a dedicated `POSIX_FADV_RANDOM` fd
  (`FATCAT_RAND=0` restores); device bytes now match ring bytes to 0.1%.
  Decode speed UNCHANGED (0.86 vs 0.87 tok/s): the NVMe had headroom all
  along — the real decode limiter is PREFETCH LATENCY, not bandwidth:
  every tier miss (`late-prefetch == miss` in the stats) stalls the
  producer waiting for its fill. Next lever: Pista B (earlier/deeper
  expert prefetch coverage).
- Pista B — speculative expert prefetch ✅ ACCEPTED (Sessions 45–45c):
  the next-layer router guess (65% accurate) now PREFETCHES by default,
  issued end-of-layer so it rides behind the current critical reads and
  lands during the GEMVs (`FATCAT_SPECE=1` early-issue variant measured
  slower — guesses must never queue ahead of critical). New default
  `FATCAT_SPECP=2`: standalone A/B +18.6% decode (0.86 -> 1.02 tok/s on
  the 474-tok benchmark) for +14% ring bytes; depth 4 costs 30% more
  bytes and measures slower (tier pollution from wrong guesses).
  Confirmed off-benchmark (Session 48, `tools/chat-ab.sh`): +16.3%
  decode on a realistic 630-tok multi-turn chat with a 256-tok
  generation; the OLD arm also re-showed the readahead waste (+49%
  device bytes over ring) on new content.
- Specter-style proxy routing predictor ❌ REJECTED by offline study
  (Session 46, `tools/py/predict-study.py`, no GPU needed): a learned
  inter-layer affinity table (train on prefill routing, predict next
  layer's experts from the previous layer's ids) scores 23-25% engine-hit
  on two real traces at any train/eval split — 2.6x WORSE than the
  shipped guess (65.3%). The shipped predictor reuses the REAL router on
  a nearly-fresh hidden state (only the current FFN delta is missing),
  so token-specific routing content is already captured; pure structural
  co-occurrence cannot substitute for it. No implementation pursued.

### Next implementations (queued backlog, in priority order)

1. ~~**Hard memory-limit parameter**~~ ✅ **DONE (Session 35)**:
   `FATCAT_MEM_BUDGET_GB` / `_MB` caps the engine's OWN memory
   (warm tier + KV/indexer + 1 GB fixed reserve). Planner at open clamps
   the KV target to fit (KV wins) and gives the warm tier the remainder;
   at runtime KV growth first shrinks the warm tier
   (`fatcat_stream_shrink`, policy-ordered victims, busy slots untouched)
   and refuses growth beyond the cap. Deviation from the original design:
   deterministic accounting at the single KV-growth point instead of
   periodic smaps polling (KV capacity only changes there; the warm tier
   is bounded by its planned budget) — simpler and race-free.
   Engine config field `mem_budget_mb` added. Validated: warm clamp,
   KV-target clamp (100000 -> 6403 tok @ 2 GB), refusal path,
   logits bit-identical vs unbudgeted run.
2. ~~Warm-tier keeping quota~~ — **SUPERSEDED by Phase B sim (Session 33)**:
   quota-partition showed no gain over flat LFU on measured traces
   (hot decile spans more objects than any realistic keep region).
   Replaced by: **S3-FIFO — now the DEFAULT policy (Session 36)**:
   implemented in `stream.c` (Session 34), validated bit-identical +
   sanitizer-clean, and won the standalone GPU A/B at an 18 GB tier
   (77.6% vs 74.6% hit, -8.5% evictions, -8.5% wall). `FATCAT_POLICY=lfu`
   restores the legacy policy for tiny tiers.
3. `tools/stream-sim` ✅ DONE (Session 33): trace-replay harness shipped;
   results in development_log Session 33. Remaining: record more/longer
   traces to harden the policy decision before flipping the default.
4. Speed-work external audit ✅ CLOSED (Sessions 43–43f, see the audit
   bullet above): H6 shipped as default; H1–H5 rejected/already-done with
   data. Remaining optional speed items, in order: (a) confirm H6's +5.9%
   with a second A/B or a long-prompt variant (variance caveat S43f);
   (b) D4 stage 2 — indexer batching in the batched path (small, only if
   prefill is ever made latency-critical). Everything else is at the
   NVMe-bandwidth plateau: no software lever left at 64 GB.

## 6. Repository layout

```
AGENT.md              Project rules (language, C23, tracking conventions)
project.md            This tracker
development_log.md    Append-only development log
Manuals/              Platform/library documentation (see Manuals/README.md)
docs/                 Design documents (architecture, performance, streaming, build)
include/fatcat/       Public C headers
src/core/             Engine loop, tokenizer, sampling
src/model/            Layers, MLA+indexer attention, MoE router
src/quant/            GGUF K-quant dequant/dispatch kernels
src/stream/           Expert tier manager (pin/LFU/promote/evict, prefetch)
src/io/               GGUF reader, mmap, io_uring async reads
src/backend/cpu/      AVX-512/VNNI/BF16 kernels
src/backend/gpu/      gfx1151 backend (rocBLAS/hipBLASLt, later HIP kernels)
tests/                Unit + numerical correctness tests
tools/                GGUF inspector, expert-usage profiler, stream simulator
benchmarks/           tokens/s, expert hit-rate, SSD bandwidth benchmarks
third_party/          Vendored dependencies (each with a note in development.md)
```

Every code/script/test folder has a `development.md` with its detailed status.

## 7. Decision log

- 2026-11-01 — Project created; decisions of §3 locked after discussion.
- 2026-11-01 — `AMD strix halo.7z` (AMD 57930 PPR 3.00 + 58550 docs) moved into
  `Manuals/strix-halo/`.
- 2026-11-01 — Performance doctrine adopted (`docs/performance.md`): maximum
  tokens/s is the primary objective; development of custom low-level matrix
  kernels (AVX-512 CPU, HIP GPU) is explicitly in scope.
