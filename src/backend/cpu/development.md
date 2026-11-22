# src/backend/cpu — development.md

## Goal
CPU compute backend for Strix Halo's 16× Zen 5 cores: AVX-512 kernels for
fused dequant-GEMV, RMSNorm and dot products. Hand-written low-level kernels
are explicitly in scope — see `../../../docs/performance.md`.

## Status
**P2b ✅ (Session 16).** AVX-512 fused dequant-GEMV row kernels for
Q8_0 / Q6_K / IQ1_S / IQ2_XXS + F32, plus `dot_f32` and `rmsnorm`.
- `cpu.c` — public API + runtime dispatch (`__builtin_cpu_supports`),
  scalar fallback via `fatcat_dequant`, row-level OpenMP inside the kernels,
  `FATCAT_CPU_SCALAR=1` forces scalar for A/B.
- `avx512.c` — kernels; decoded values fold into the dot in 8–32 element
  register windows (no full-row f32 scratch; the old path streamed a 32 KB
  tmp per row through L1/L2).
- Exact-f32 design: weights dequantize to f32 and FMA (bit-exact decode
  values vs `src/model/dequant.c`; only accumulation order differs).
  No int8-VNNI activation quantization (that is llama.cpp's lossy trick —
  rejected in P2c).
- `ml4.c` `gemv`/`gemv_expert` CPU paths, `dot_f32`, `rmsnorm` all route here.
- Tests: `tests/test_cpu_kernels.c` (random-block vs dequant+dot reference,
  all PASS, relerr ~1e-6). Bench: `tools/cpu-bench`.

## Measured (Zen 5, warm cache; `tools/cpu-bench`)
| kernel | 1T scalar | 1T AVX | speedup | 32T AVX |
|---|---|---|---|---|
| Q8_0 (ffn_gate slice)   | 2.33 GB/s | 11.05 GB/s | 4.7×  | ~250 GB/s (L3) |
| Q6_K (attn_q / output)  | 1.69 GB/s | 12.79 GB/s | 7.6×  | ~220 GB/s (L3) |
| IQ1_S (expert gate)     | 0.28 GB/s | 1.45 GB/s  | 5.2×  | ~33 GB/s |
| IQ2_XXS (expert down)   | 0.08 GB/s | 2.36 GB/s  | 28.8× | ~58 GB/s |

End-to-end 8-token CPU forward: 84.4 s → 54.8 s wall (1.54×; remainder is
NVMe expert streaming + I/O). Logits bit-identical to the scalar path
(corr 1.000000 at all positions); GPU path unchanged.

Note: benchmark numbers are noisy when the box is busy (the assistant LLM
shares this machine) — take best-of-many, prefer single-thread ratios.

## Scope
- In: intrinsics kernels (AVX-512F/BW/VL/DQ), memory-bandwidth-friendly loops
  (the ~256 GB/s unified-memory ceiling dominates design).
- Out: aarch64 or non-Zen5 paths — Strix Halo only. RoPE/softmax/router SIMD
  not yet needed (not hot at decode batch 1).

## Next steps
1. (P3) Row-range gemv variants so the streaming scheduler can fuse expert
   prefetch with compute at sub-matrix granularity.
2. Possible: 2-row interleaved kernels to reuse x from L1 across rows;
   IQ1_S kernel tuning (currently 8-wide; could batch 4 lookups to 32-wide).
3. ~~batched GEMM~~ ✅ (Session 38): `fatcat_cpu_gemm()` — IQ1_S/IQ2_XXS
   batched kernels decode each quantized row ONCE and FMA into up to 8
   per-token accumulators (tile loop beyond 8); per-token op order matches
   the gemv kernels exactly, so results are BIT-identical (unit-tested via
   `test_cpu_kernels` memcmp for n_tok 1..9). Used by `forward_batched()`
   for experts with >=2 tokens; measured +29% prefill at the 1 GB tier.
