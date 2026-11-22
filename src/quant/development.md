# src/quant — development.md

## Goal
Quantization support: GGUF K-quant family (Q4_K/Q5_K/Q6_K/Q8_K and friends)
dequantization and fused dequant-GEMV/GEMM dispatch, plus FP8 block-quant
handling if native weights are used.

## Status
**Scalar reference kernels DONE — physically in `../model/` (2026-11-02).**
`src/model/dequant.c` + `quant_tables.c` implement F32/F16/Q8_0/Q6_K/
IQ2_XXS/IQ1_S dequant, bit-exact vs gguf-py golden data (`test_dequant`
PASS); `fatcat_dequant()` is the dispatch entry (`type → kernel`). This
folder remains the planned home for the expanded K-quant registry and the
AVX-512 dispatch tables when `src/backend/cpu/` kernels arrive (P2).

## Scope
- In: block dequant kernels (AVX-512 in `src/backend/cpu/`), quant format
  registry, per-tensor type dispatch.
- Out: custom quant formats (GGUF quants are the locked input format).

## Next steps
1. Reference scalar dequant implementations for each GGUF type (correctness).
2. Dispatch table `tensor_type → kernel`.
3. AVX-512/VNNI optimized kernels (P2), benchmarked in `benchmarks/`.
