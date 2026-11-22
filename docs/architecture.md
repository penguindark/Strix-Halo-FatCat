# Architecture Overview

Status: draft (P0). Authoritative status: `../project.md`.

## Big picture

FatCat runs Mistral-Large-4-1T-A52B (72 layers, 128+1 experts, top-4) with the
routed experts split across a three-tier hierarchy on Strix Halo:

```
NVMe (all 9,088 routed experts, GGUF)            ~515 GB @4-bit
   │  io_uring async streaming (src/io, src/stream)
   ▼
System RAM (warm tier, unified LPDDR5X)          up to ~120 GB usable
   │  pinning policy (no physical copy — unified memory)
   ▼
GPU-resident region (hot tier, GTT carve-out / pinned pages)
   ▼
gfx1151 iGPU (rocBLAS/hipBLASLt, src/backend/gpu)  + 16× Zen 5 (src/backend/cpu)
```

Always-resident (never streamed): embeddings, lm_head, attention (MLA +
indexer), routers, shared experts, dense layer 0.

## Data flow per token
1. Tokenizer → embeddings (`src/core`, `src/model`).
2. Per layer: MLA attention (sparse indexer selects 2048 KV entries) → MoE
   router picks 4/128 experts → `src/stream` resolves each expert's tier
   (hot/warm/cold; cold triggers io_uring read + possible eviction) → expert
   GEMMs on CPU or GPU.
3. Sampling loop back to `src/core`.

## Key constraints
- ~256 GB/s unified memory bandwidth: decode is bandwidth-bound; the indexer
  and MLA (small KV) are major allies.
- One streamed expert ≈ 57 MB @4-bit → at ~3 GB/s effective NVMe, a cold miss
  costs ~20 ms: prefetch and hot-tier sizing must make misses rare.
- Streaming granularity = whole expert, page-aligned, contiguous in file.
