# Prefill Speed Study — what llama.cpp-family engines exploit (Session 37)

Question: gufo (llama.cpp derivative for Strix Halo, github.com/mikealanni/gufo)
reports 630-4100 tok/s prefill for MoE models on the same silicon. FatCat
prefills Mistral-Large-4 at ~1 tok/s (GPU) / 0.4 (CPU). Why, and what transfers?

## Closest comparables (from gufo README, measured 2026-11)

| model | active params | weights | resident? | prefill tok/s |
|---|---|---|---|---|
| GLM-5.3 Flash | 14 B (321B MoE) | 81 GiB IQ2_XXS | fully in RAM | 630-740 |
| Laguna S 2.1 | ~10 B (118B MoE) | 68 GiB Q4_K_M | fully in RAM | 1145-1277 |
| Qwen3.6-35B-A3B | 3 B | ~20 GiB | fully in RAM | 2900-3300 |
| **FatCat Mistral-Large-4** | **52 B (1T MoE)** | **218 GB IQ1_S** | **18-64 GB tier, rest NVMe** | **~1** |

Decode sanity check: gufo GLM decode 18.9 tok/s x ~9 GB/token ~= 170 GB/s —
the LPDDR5X ceiling with everything in RAM. FatCat decode ~1 tok/s x 6.7 GB
= 6.7 GB/s effective: ~25x below the ceiling even accounting for NVMe misses.

## The four structural reasons they are faster

1. **Whole model resident in RAM.** Their smallest comparable is 81 GiB on a
   128 GB box: zero per-token NVMe traffic. FatCat streams 1.5-6.7 GB/token
   from NVMe (218 GB model cannot fit). This alone is the dominant term.
2. **Batched (chunked) prefill with grouped expert GEMMs.** llama.cpp feeds
   prompts in chunks of 512-8192 tokens; per layer, tokens are grouped by
   expert and each expert weight is loaded ONCE for all tokens routed to it
   (gufo: "grouped F16 matrix-core GEMMs"). Weight bytes/token drops by the
   reuse factor (hundreds), turning memory-bound GEMV into compute-bound
   GEMM. FatCat processes ONE token at a time: every token re-pays the full
   expert streaming + dequant cost. This is our biggest missing lever.
3. **Fabric clock held at max** (`pp_dpm_fclk=2000MHz`, manual DPM): gufo
   documents this as REQUIRED for full bandwidth on gfx1151. Free win we
   have not applied on this box.
4. **Speculative decoding for decode** (MTP/DFlash2/prompt-lookup): decode
   61->159 tok/s on predictable text. Prompt-lookup is lossless and needs no
   draft model. Orthogonal to prefill but big for perceived speed.

Secondary: their quants (Q4_K_XL, IQ2_XXS) have cheaper dequant paths than
IQ1_S ternary codebooks; IQ1_S trades ~2x fewer bytes for heavier unpack.

## What transfers to FatCat, ranked

1. **Batched prefill (Phase D1).** ✅ **IMPLEMENTED Session 38** as
   `forward_batched()` (`FATCAT_BATCH=<chunk>`, `logit-dump --batch N`):
   layer-outer chunk loop, per-expert token grouping, one slice load per
   expert per chunk, union prefetch. Bit-identical logits (maxdiff 0.0);
   ~3x at a 1 GB CPU tier (0.4 -> 1.3 tok/s), expert accesses/token -5x.
   Remaining for the full 10-30x estimate: batched GEMV->GEMM kernels
   (CPU + GPU), GPU async pipe, larger tiers (the 1 GB tier thrashes on
   the chunk's expert union: hit 41-57% vs 83% sequential).
   Original design: Chunk C tokens: run routers for the chunk
   layer-by-layer, union the needed experts per layer (at C>=256 the union
   approaches all 128/layer, i.e. prefill streams ~one full model pass per
   chunk), load each expert once, group tokens per expert, GEMM. NVMe-bound
   ceiling: 218 GB / C per token -> C=1024: ~15 tok/s; C=4096: ~30+ tok/s
   with warm tier absorbing the hot set. Est. 10-30x prefill speedup.
   STANDING RESULT (standalone GPU A/B @ 18 GB tier, mixed500, report
   perf-validation-20261120-0324): sequential 1.0 tok/s (hit 78.3%) vs
   chunk 128 2.94 (98.4%) vs chunk 512 3.41 (100.0%, zero misses);
   logits bit-identical (maxdiff 0.0). Default now ON at chunk 256. The
   remaining gap to the 10-30 estimate: the per-token attention path and
   (GPU) matrix-core utilization; batched GEMM kernels landed Session 38b
   (bit-identical, +29% CPU at 1 GB tier; GPU A/B via gpu-gemm-test.sh).
2. **GPU staging fix (Phase D2).** Session 36 smoke: `stage_hits=0`,
   ~7.4 GB/token copied RAM->GPU staging buffers. Each slice used once per
   token -> staging cache can never hit; the zero-copy wrap path is not
   catching tier slots. Small investigation, est. up to ~1.5x on GPU path.
3. **Fabric clock hold (Phase D0, trivial).** `power_dpm_force_performance_level
   = manual; pp_dpm_fclk = top` (see gufo tools/npu/fabric-clock.sh).
   Add to Manuals/io/memory-tuning.md; measure A/B.
4. **Prompt-lookup speculative decode (Phase D3).** Lossless, no draft model;
   big on repetitive/structured text, ~0 on prose. Decode-path work.
5. **Batch the dense per-token ops (Phase D4). ✅ DONE stage 1 (Session 39).**
   The GPU GEMM A/B (Session 38b)
   showed expert weight handling is NOT the GPU wall: chunk 512 stayed at
   ~3.4 tok/s with 8x fewer weight reads. The wall is ~700 synchronous
   memcpy+launch+sync GPU round-trips per token (wq/wkv_a/wo/router/
   shared-FFN: ~10 dense calls x 71 layers, one per token). D4 batches
   those over the chunk exactly like D1 batched the experts: one GEMM per
   weight per chunk, bit-identical by construction (per-token op order
   preserved). Stage 1 = router + shared FFN + dense FFN + wq/wkv_a/wo
   (attention middle stays sequential: RoPE/KV-writes/SSMax/indexer/scan
   are positional); stage 2 (optional) = the indexer gemvs. Est. round-
   trips/token 700 -> ~150; prefill 3.4 -> ~6-7 tok/s at 18 GB tier.
   **Outcome: +2–6%, not 2x** (chunk 512: 3.41 → 3.49–3.62 tok/s @ 18 GB;
   CPU policy100 +3.7%). The estimate was wrong: dense-op round-trips were
   NOT the dominant wall — with them batched, the time sits in the
   EXPERT-side per-(layer,expert) staging+launch+sync cycles (~330 per
   layer per chunk). Bit-identity needed `-ffp-contract=off` on
   kernels.hip: the batched kernel's hoisted quant decode let the compiler
   contract mul/add into FMAs differently than the gemv kernel (1-ULP
   drift → cos 0.9987 e2e). Explicit `__fmaf_rn` pinning MISCOMPILES on
   this ROCm build — do not reintroduce. Remaining prefill lever: async /
   multi-expert expert-side staging (bigger surgery).

## What does NOT transfer

- Their absolute numbers: 4x fewer active params and full RAM residency.
  Fair target for FatCat prefill with batching at 64 GB tier: 10-30 tok/s,
  not 600.
- NPU offload (+1-28%): dense projections only, marginal for a model whose
  time is in expert streaming; XDNA2 toolchain is a large detour.
- MTP/DFlash2 drafting: Mistral-Large-4 checkpoint has no MTP head; a DFlash
  drafter for this model does not exist. Prompt-lookup only.
