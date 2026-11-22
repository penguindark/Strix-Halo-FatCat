# src/backend/gpu — development.md

## Goal
gfx1151 (RDNA 3.5) iGPU backend using the local ROCm 10.1 TheRock install
(`/home/user/rocm-10.1/install`). Start with library-level compute
(rocBLAS / hipBLASLt) driven from C via the HIP C API; custom HIP kernels
later where libraries fall short (fused dequant-GEMM for GGUF types).

## Status
**Scaffold — superseded by `src/gpu/` for now.** The live HIP backend (fused
dequant+GEMV kernels, validated 2026-11-02) lives in `../../gpu/` — see its
`development.md`. This folder remains the planned home of the library-level
(rocBLAS/hipBLASLt) backend for prefill GEMM in P4. CMake already detects
ROCm and links rocBLAS/hipBLASLt when found (`FATCAT_WITH_GPU`).

## Scope
- In: device/context management, GPU-pinned expert region (GTT carve-out or
  userptr), GEMM dispatch, custom HIP kernels where profiling proves
  libraries are the bottleneck (fused dequant-GEMM for GGUF block formats,
  fused router+gather, indexer attention) — see `../../../docs/performance.md`.
- Out: multi-GPU, non-gfx1151 targets.

## Environment notes
- ROCm 10.1 TheRock gfx1151: `/home/user/rocm-10.1/install`
  (older 7.14.1 at `/home/user/rocm`).
- Unified memory: prefer `hipHostRegister`/userptr on RAM-resident expert
  buffers instead of explicit copies where possible.

## Next steps
1. ~~Minimal HIP C-API smoke test~~ ✅ done (P4).
2. hipBLASLt GEMM benchmark vs CPU AVX-512 for expert-shaped matrices.
3. ~~batched GEMM~~ ✅ (Session 38): `fatcat_gpu_gemm()` + `k_gemm_iq1_s` /
   `k_gemm_iq2_xxs` in `kernels.hip` — one codebook decode per weight row
   reused across up to 8 tokens (host tiles larger batches); per-token op
   order mirrors the gemv kernels exactly -> BIT-identical. Sync staging
   path (grow-on-device X + mapped-host Y). Wired into `forward_batched()`
   via `expert_gemm()` (wrap -> stage -> CPU ladder). Standalone GPU A/B
   pending: `tools/gpu-gemm-test.sh`.
