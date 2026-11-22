# Performance Strategy — Maximum tokens/s on Strix Halo

Status: active doctrine (P0). Authoritative status: `../project.md`.
Goal: **maximum generated tokens/s** via (a) fully optimized compute and
(b) the fastest, smartest expert cache. Custom low-level matrix kernels are
explicitly **in scope** wherever libraries fall short.

## 1. Physical ceilings (computed from model facts, 4-bit weights)

| Quantity | Value |
|---|---|
| Expert size (gate+up+down) @4-bit | **56.6 MB** |
| All routed experts @4-bit | ~503 GB |
| Active weights touched per token (52B params @4-bit) | ~26 GB |
| Decode ceiling @256 GB/s (theoretical, batch=1) | **~9.8 tok/s** |
| Decode ceiling @200 GB/s (realistic sustained) | **~7.7 tok/s** |
| Cold expert miss from NVMe (3–7 GB/s) | **8–19 ms** |

Consequences:
- Decode is **memory-bandwidth-bound**: every token must stream ~26 GB of
  active weights. No kernel can beat the RAM ceiling; the game is (1) never
  touch bytes we don't need, (2) keep active bytes in the fastest tier,
  (3) overlap the rest.
- A single cold miss (~19 ms) eats ~20% of the ~100 ms token budget:
  **misses must be prefetched or eliminated**, never on the critical path.
- Prefill is compute-bound and much less sensitive to streaming (few forward
  passes, high expert reuse within the prompt).

Targets (batch=1 decode, to be refined after P1 traces):
- **T0 (P2)**: correct forward pass, measure actual tok/s with all-RAM weights.
- **T1 (P3)**: ≥ 90% warm-tier hit-rate on typical prompts after warm-up.
- **T2 (P4/P5)**: ≥ 80% of the 200 GB/s ceiling (~6 tok/s) end-to-end.

## 2. Compute optimization strategy

### 2.1 CPU (16× Zen 5, AVX-512: BW/VNNI/BF16)
- **Fused dequant-GEMV** is the critical kernel for decode: dequantize GGUF
  K-quants directly inside the dot-product loop (no dequant-to-memory pass).
- Custom **hand-written AVX-512 kernels** are in scope: Q4_K/Q5_K/Q6_K/Q8_K
  GEMV, RMSNorm, RoPE, softmax, router top-4 (small-k selection network).
- Batch the 4 selected experts per layer into one grouped GEMV pass to amortize
  activation traffic and improve TLB behavior.
- Thread pool: 16 threads pinned to cores (SMT-aware), static row partitioning
  for GEMV; avoid false sharing; hugepages for weight arenas.
- Activations in BF16 where numerically safe (Zen 5 AVX512-BF16), FP32
  accumulate.
- Bandwidth discipline: every kernel must approach memcpy speed for its access
  pattern; validated with micro-benchmarks in `benchmarks/`.

### 2.2 GPU (gfx1151, ROCm 10.1)
- Start with hipBLASLt/rocBLAS for GEMMs (prefill, batched expert GEMMs).
- **Custom HIP kernels** in scope where libraries lose: fused dequant-GEMM for
  GGUF block formats, fused router+gather, attention kernels for the indexer.
- Unified memory: pin hot experts once (`hipHostRegister`/userptr), never copy
  per token; use async copy engines for cold→warm staging overlap.
- Router + small ops on CPU, expert GEMMs on GPU; overlap layer L compute with
  layer L+1 expert prefetching (double-buffered expert slots).

### 2.3 Algorithmic multipliers
- **Sparse indexer (DSA)**: only 2048 KV entries scored per query — implement
  first, it directly cuts KV bandwidth.
- **Speculative decoding** (small draft model, e.g. Mistral-Small on the same
  box): accepted multi-token steps amortize expert loads over several tokens —
  the single biggest potential tok/s multiplier for a streaming MoE.
- Shared expert computed in parallel with routed experts (independent GEMMs).

## 3. Expert cache design (the "smart" part)

### 3.1 Tiers & budgets
- Hot (GPU-pinned region) / Warm (RAM arena, hugepages) / Cold (NVMe, GGUF).
- Budgets per tier configured per-layer: layers with flatter expert histograms
  get bigger budgets (allocation ∝ usage entropy — decided from P1 traces).
- Never-stream set: embeddings, lm_head, attention+indexer, routers, shared
  experts, dense layer 0.

### 3.2 Policy (LFU + recency hybrid)
- Score per expert: `score = α·recency_decay + β·count` updated on every router
  hit; promotion cold→warm on first hit (async), warm→hot above threshold.
- Eviction: lowest score first; in-flight experts are unevictable.
- **Router-lookahead prefetch**: after layer L routing, the top-ranked *not
  selected* experts of layer L+1 (from its router logits are not available
  yet, so use the previous token's routing as predictor — MoE expert usage is
  highly temporally local) are prefetched via io_uring into warm slots during
  the current token's compute.
- Warm-up pass: optional offline pass over representative prompts to pre-fill
  tiers before serving (cold-start elimination).

### 3.3 I/O
- io_uring ring per tier-manager thread, batched submits, 512 KB+ aligned reads
  (whole expert = contiguous in file → single read).
- Overlap target: NVMe bandwidth used 100% during compute, 0 ms added latency.

## 3bis. Configurable RAM budget (hard requirement)

The engine must run under an explicit, configurable resident-memory budget
(`FATCAT_RAM_BUDGET`, e.g. `--ram-budget 4G`), enforced by the tier manager as
a **hard cap** on all resident allocations (hot + warm expert slots, KV cache,
arenas). Semantics:

- Fixed costs (KV cache for the configured context, always-resident weights)
  must fit the budget or startup fails with a clear report of the minimum.
- When the budget is too small for the working set, the engine **degrades
  gracefully**: more experts stay cold, hit-rate drops, generation gets slower
  but stays correct — this is the intended mode for low-memory testing.
- Budget sweeps (e.g. 2/4/8/16/32/64 GB) are a standard benchmark: tok/s and
  hit-rate vs budget is the engine's primary characterization curve.

## 3ter. Shared-machine development workflow

Strix Halo also hosts the assistant LLM, so all development tests must be
lightweight:
- Early validation runs on **model slices**: `tools/gguf-slice` extracts a
  few layers (e.g. layer 0 dense + layers 1–3 MoE + final norm + lm_head)
  into a small GGUF for numerical and streaming tests.
- Every test/benchmark declares its memory budget and must run within a few
  GB; full-model runs are explicit, scheduled operations, never defaults.

## 4. Measurement plan (no optimization without numbers)
- `benchmarks/`: kernel micro-benchmarks (GB/s achieved vs 256 GB/s ceiling),
  io_uring bandwidth, expert hit-rate vs hot/warm budget sweeps, end-to-end
  tok/s per phase.
- `tools/expert-profiler` (P1): real hotness traces → feed policy tuning and
  per-layer budgets in `streaming-policy.md`.
- Every phase entry in `development_log.md` must record measured tok/s or
  bandwidth numbers.

## 5. Kernel development roadmap
1. P2: scalar reference kernels (correctness) → AVX-512 fused dequant-GEMV.
2. P3: streaming overlap — no kernel change, scheduling only.
3. P4: hipBLASLt GEMMs; profile; write custom HIP kernels only where profiling
   proves libraries are the bottleneck.
4. P5: fused attention/indexer kernels, final tuning pass.
