# docs — development.md

## Goal
Design documents for FatCat: architecture overview, expert streaming policy,
GGUF layout requirements, build instructions. These are living design notes;
the authoritative status tracker remains `../project.md`.

## Status
**P0 + P1 filled (2026-11-02).** `streaming-policy.md` now carries the
measured hotness analysis from the full 949-token GPU profile (flat
distribution, cov50/80/95/99 = 11/37/74/100 experts per layer, warm-tier
budget sweep 16–64 GB → 46–79% hit rate). Others are living design notes.

## Documents
- `architecture.md` — engine architecture and data flow.
- `performance.md` — **performance doctrine**: ceilings, compute optimization,
  cache design, measurement plan, kernel roadmap.
- `streaming-policy.md` — tiered expert cache design (P3 input).
- `build.md` — build instructions (CMake, ROCm 10.1 TheRock, Strix Halo).

## Next steps
1. Document the GPU-pinned memory strategy before P4 (hot tier sizing from the
   budget sweep in `streaming-policy.md`).
2. Fill `architecture.md` with the CPU-control / GPU-compute split as P2 lands.
