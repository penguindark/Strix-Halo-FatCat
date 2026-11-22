# benchmarks — development.md

## Goal
Performance benchmarking: end-to-end tokens/s (prefill + decode), expert cache
hit-rates, NVMe streaming bandwidth, CPU/GPU kernel micro-benchmarks.

## Status
**First numbers from the P1 profiler (2026-11-02).** Full-model expert
profiling run (`/home/user/data/prompts/profile-gpu.log`):
- 949 tokens @ **0.49 tok/s** (GPU gemv + synchronous expert streaming),
  memory flat ~18 GB.
- Expert staging: 808 975 slot copies, 6 361.6 GB → **~3.3 GB/s sustained**
  with 8-thread parallel `pread()` (single-threaded mmap faults were ~1 GB/s).
- Warm-tier hit-rate vs budget (offline, `tools/analyze_trace.py`):
  16 GB→45.8%, 32 GB→61.6%, 48 GB→71.9%, 64 GB→79.4%.
Raw results belong in `results/` (gitignored); log + trace stay in
`/home/user/data/prompts/`.

## Scope
- In: kernel micro-benchmarks (dequant-GEMV/GEMM), io_uring read bandwidth,
  expert hit-rate vs hot-tier size sweeps, end-to-end prompt benchmarks.
  Results go in `benchmarks/results/` (gitignored).
- Out: model quality/eval scores.

## Next steps
1. NVMe sequential/4K-random read bandwidth script (baseline for P3 design).
2. Dequant-GEMV micro-benchmark harness (P2).
