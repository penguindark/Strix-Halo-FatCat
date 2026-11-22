# src/model — development.md

## Goal
Mistral-Large-4 text model implementation: decoder layers, MLA attention with
the sparse indexer, MoE router (128 routed + 1 shared expert, top-4), dense
FFN (layer 0), RoPE (YaRN), RMSNorm, logits soft-capping. Also hosts the
reference dequant kernels (`dequant.c`, `quant_tables.c`).

## Status
**ML4 forward DONE + GPU-accelerated (P1, 2026-11-02).** `ml4.c`:

- Full text forward: absorbed MLA (kv_lora 512 + q 2×3072), LeRoPE + YaRN,
  SSMax attention, NoPE layers, KV sharing (`kv_source_layer`), dense FFN
  layer 0 + MoE (128 routed top-4 + shared expert), per-tensor expert
  strides, router softmax + weight renorm.
- Verified vs llama.cpp on the full model: pearson corr **0.997**, top-1000
  overlap 936/1000 (1-token); GPU path corr 0.9970 / 931/1000, argmax match.
- GPU integration: fixed weights uploaded once (`wt_upload`, mmap copy
  `MADV_DONTNEED`'d after upload); routed experts streamed per token via
  parallel `pread()` into a reusable 24×10 MB aligned pool (8 OMP threads),
  staged into 16 LRU-tagged GPU slots (tag = layer<<34|kind<<32|expert);
  per-tensor CPU fallback if GPU staging/gemv fails.
- Memory policy: watches system usage (MemTotal−MemAvailable), default cap
  `FATCAT_MEM_LIMIT_MB=80000`; past it each expert's page-cache range is
  fadvise-dropped after use. Measured flat ~18 GB on the full 949-token run.
- `dequant.c`/`quant_tables.c`: scalar reference dequant for F32/F16/Q8_0/
  Q6_K/IQ2_XXS/IQ1_S, bit-exact vs gguf-py golden data (`test_dequant` PASS).

## Scope
- In: architecture per `Manuals/model/config.json` (72 layers, hidden 6144,
  MLA `kv_lora_rank=512`, `index_topk=2048`, YaRN, vocab 204800).
- Out: vision tower (P6); the sparse indexer is not present in this model's
  attention path (plain MLA + sliding windows).

## Key facts
- Expert = gate/up/down of 6144×6144: 24.5 MB (IQ1_S gate/up + IQ2_XXS down,
  blk.7–70) or 29.2 MB (all IQ2_XXS, blk.1–6/71) per expert.
- Shared expert + router + attention + embeddings are *always resident*; only
  routed experts are streamed by `src/stream/`.
- 71 MoE layers × top-4 × 3 matrices ≈ 852 expert-slice reads/token
  (~6.8 GB/token) — the streaming path must stay parallel + async (P3).

## Next steps
1. AVX-512 GEMV kernels behind the same `wt_t` interface (`src/backend/cpu/`)
   — the only remaining P2 item (P2b).
2. ~~Replace synchronous expert pool with io_uring prefetch + router lookahead
   (P3).~~ ✅ done (P3).
3. **D1 batched prefill** (Session 38): `forward_batched()` in `ml4.c`,
   **DEFAULT ON at chunk 256** (`FATCAT_BATCH=<chunk tokens>`, 0 = off;
   `logit-dump --batch N`). Layer-outer chunked loop: attention per token,
   routers for the chunk, per-expert token grouping (linked lists), ONE
   expert-slice load per expert per chunk, union prefetch overlapped with
   the shared-expert FFNs. Per-token accumulation keeps top-k order →
   logits bit-identical to the sequential path (validated maxdiff 0.0:
   CPU 1 GB tier + standalone GPU 18 GB tier, chunks 64..512). Standalone
   GPU A/B @ 18 GB (mixed500): sequential 1.0 → chunk 128 2.94 → chunk
   512 3.41 tok/s, tier hit 78.3% → 100% (chunk union fits the tier,
   streaming eliminated). `fatcat_engine_prefill` passes the whole prompt
   in one call, so fatcat-chat prefill gets this automatically. v1 limits:
   no router speculation, no `FATCAT_DUMP_HID` (falls back to sequential).
   Session 38b: the grouped expert pass now uses batched GEMM kernels
   (`fatcat_cpu_gemm` / `fatcat_gpu_gemm`) — one weight decode per expert
   per up-to-8 tokens, still bit-identical (per-token op order preserved);
   +29% CPU prefill at the 1 GB tier (1.23 -> 1.59 tok/s).

## Status
- P2c numerical correctness ✅ (Session 15): golden numpy-f64 layer-0 check
  shows fatcat delta relerr 4.2e-6 vs llama.cpp's 1.3e-2 — fatcat is the more
  accurate engine; residual logit-corr gap vs llama.cpp is llama.cpp kernel
  noise. Fixed there: logit softcap, `attention.layer_norm_epsilon` key,
  rope adjacent-pair (NORM) rotation, gemv chunked dequant for n_in>8192.
- Diagnostics: `FATCAT_DUMP_HID` residual-stream dump; `tools/tensor-dump`
  (bit-exact dequant → f32); `tools/llama-layer-dump` (llama.cpp per-layer).
- P5 DSA indexer v1 ✅ (Session 24): lightning indexer on the 44 "full"
  layers (LayerNorm'd q input, per-head SSMax `log(t+1)`, RoPE on first
  n_rot dims, `score(p)=Σ_h relu(q_h·k_p)·w_h` with `w` pre-scaled
  1/√(nh·dk), top-k over 0..t incl. current pos, shared layers reuse the
  previous selection; Hadamard rotation is score-invariant → skipped).
  Sparse MQA in `attention()`; `FATCAT_IDX=0` / `FATCAT_IDX_TOPK` knobs.
  Also fixed: RMS-norm eps key (`layer_norm_rms_epsilon`, 1e-5) vs indexer
  LayerNorm eps (`layer_norm_epsilon`, 1e-6); `fatcat_topk_select` sorted
  by index (test_topk vs brute force). CPU baseline: `eight_fc_p5b.bin`.
- P5 long-context validation ✅ (Session 25): 2100-token run at the real
  top_k=2048 — positions 0..2047 bit-identical to dense (selection inert
  while seq ≤ top_k), 2048..2099 sparse with cos 0.994..1.000 vs dense,
  top-1 47/52 (near-dense at threshold). Indexer-scorer AVX-512 transposed
  probe gave no speedup (bandwidth-bound) and was reverted.
- P5 f16 KV storage ✅ (Session 26): `FATCAT_KV_F16=1` now stores the MLA KV
  cache as f16 (uint16_t), halving it 38→19 GB at 256K; read path uses new
  F16C kernels `fatcat_cpu_dot_f32_f16` / `fatcat_cpu_kv_accum_f16`. cos
  0.992..0.998 vs f32; f32 path bit-identical. Opt-in (default f32).
- P5 hugepages ✅ (Session 27): `FATCAT_KV_HUGEPAGE=1` madvise(MADV_HUGEPAGE)
  on KV + indexer caches after growth; numerically transparent (bit-identical).
  Default off (compaction stalls under memory pressure).
- P5 KV pre-allocation ✅ (Session 28): `FATCAT_KV_TARGET=<tokens>` allocates
  the KV+indexer caches at a fixed capacity in one shot instead of lazy
  doubling (which overshot up to 2x the sequence: ~57 GB vs 38.7 GB at 256K
  f32 — an OOM edge next to the resident assistant). GGUF `context_length`
  (1M tokens here) is deliberately not used as the default. Numerically
  transparent (bit-identical); growth logs `kv: capacity N tokens, X GB`.
