# src/gpu — development.md

## Goal
gfx1151 (RDNA 3.5) iGPU backend using the local ROCm 10.1 TheRock install
(`/home/user/rocm-10.1/install`). Custom HIP fused dequant+GEMV kernels for
GGUF quant types; fixed weights uploaded once, routed experts streamed
through LRU-tagged staging slots. C wrapper (`gpu.c`) with a CPU-only stub
(`gpu_stub.c`) when HIP is unavailable; `FATCAT_GPU=0` disables at runtime.

## Status
**Fused-kernel backend DONE + validated (P1, 2026-11-02).**
**P4 v3 (2026-11-17): kernel optimization for gfx1151 (tools/gpu-kbench).**
- Read ceiling ~236 GB/s; kernels were 3.6-21 GB/s. Root causes: mapped-
  host x re-read by every row (fixed: device-x staging, sync path stages
  into hipMalloc g_x; async path uses stream-ordered ring
  fatcat_gpu_xstage(), ml4 stages xn once/layer + fa[k] per expert), and
  scalar lane-strided loads (fixed: float4 x, u64 codebook loads, IQ1_S
  one-thread-per-32-elem mapping, 16B alignment check in launcher).
- Now: IQ1_S 51-74, IQ2_XXS 57-86, Q8_0 139-217, Q6_K 97-142 GB/s.
**P4 v2 (2026-11-17): async GEMV pipelining.**
- Launch-only GEMV API (`fatcat_gpu_gemv_dev` + `fatcat_gpu_wait`); expert
  loop runs 2 device syncs per MoE layer (gate+up batch -> wait -> silu +
  down batch -> wait). Forward scratch is persistent and device-mapped,
  so expert GEMVs read x / write y with zero staging copies.
  `FATCAT_GPU_PIPE=0` forces the v1 sync path for A/B.
- Measured NEUTRAL on the shared box (min-of-6: 25.3 s pipelined vs
  25.6 s sync): batch-1 GEMV is bandwidth-bound and the APU GPU shares
  system-RAM bandwidth with the CPU; sync overhead was small, and the
  io_uring fill thread already overlapped NVMe during syncs. Keep for
  idle-GPU / larger-tier scenarios.
**P4 v1 (2026-11-17): wired into the ml4 engine + zero-copy warm tier.**
- Mapped-pinned x/y staging: one device sync per GEMV (was three).
- Warm-tier slots can be device-mapped pinned memory (stream allocator
  hook + unified-address probe): the GPU reads streamed experts in place,
  no GTT copies (68.3 -> 14.8 GB per 8-token run; wall 55.7 -> 28.0 s).
- Numerics in the P2c GPU band; top-1 matches the CPU path.
- GPU currently trails the optimized CPU path (28 s vs 17 s): per-GEMV
  launch+sync overhead (~14k calls) and GTT contention with the host LLM
  server sharing the iGPU. Next: async GEMV pipelining (batch gate+up,
  one sync per expert), overlap with io_uring prefetch, idle-GPU re-test.

- `kernels.hip`: 5 fused dequant+GEMV kernels (F32/Q8_0/Q6_K/IQ1_S/IQ2_XXS),
  warpSize-aware reductions (RDNA warp=64, shuffle mask `~0ull`).
  Validated per-type vs CPU at ~1.2e-7 norm-relerr (`tests/gputest.c`).
- `gpu.c`: device init (`hipDeviceGetAttribute`/`hipDeviceTotalMem` — the
  properties macro needs 4 args), fixed-weight upload, 16 × 10 MB LRU staging
  slots, per-call sticky-error clear before launch, upload failures print +
  count (`gpu stats: ... upload_fails=N`).
- Full-model run: all fixed weights uploaded (`upload_fails=0` standalone),
  6.36 TB staged across 808 975 slot copies at ~3.3 GB/s avg.

## Environment notes
- ROCm 10.1 TheRock gfx1151: `/home/user/rocm-10.1/install`
  (older 7.14.1 at `/home/user/rocm`). hipcc invoked via custom CMake rule.
- Unified memory: GPU allocations come from system RAM and are NOT counted
  in process RSS (they show in TTM/meminfo). `hipMalloc` can fail under
  memory pressure — every caller must handle per-tensor CPU fallback.
- Sticky HIP errors poison later calls: clear before each launch.

## Scope
- In: current fused GEMV set; later hipBLASLt GEMM for prefill, GPU-pinned
  hot-expert region (GTT carve-out/userptr), fused router+gather.
- Out: multi-GPU, non-gfx1151 targets. (`src/backend/gpu/` is the P4 scaffold
  folder for the library-level backend; live code is here.)

## Next steps
1. Prefill GEMM path (hipBLASLt or tiled custom kernel) — P2/P4.
2. Persistent hot-expert GPU region driven by `src/stream/` policy (P3/P4).
3. Micro-bench staging slot count/size vs NVMe queue depth (benchmarks/).
