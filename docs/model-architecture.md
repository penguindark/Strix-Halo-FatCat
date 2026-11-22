# Mistral-Large-4-1T-A52B — Internal Architecture Analysis

Status: active (P0). Sources: `Manuals/model/` (upstream config.json, HF blog,
model card, vLLM day-0 blog, DSpark/MLA/DeepSeek-V3.2 papers).
Authoritative status: `../project.md`.

## 1. Top-level facts

| Item | Value |
|---|---|
| Total / active params | ~1T / ~49B (52B incl. embeddings + output head) |
| Layers / hidden / heads | 72 / 6144 / 64 (head_dim 64 qk_nope + 64 rope, v 128) |
| Vocab / max context | 204,800 / 1,048,576 (FatCat target: **256K**) |
| Output | logit soft-cap 30.0 |
| Checkpoints | FP8 (128×128 block), NVFP4; GGUF quants: IQ2_XXS, Q2_K, Q4_K_M, NVFP4, Q8_0, BF16 (ggml-org + Unsloth UD) |
| Vision | Pixtral encoder 1.6B, 30 layers — **deferred (P6)** |

## 2. Attention stack (per layer)

Hybrid **MLA + DSA sparse indexer** (DeepSeek-V3.2-style):

- **MLA**: no query LoRA; `kv_lora_rank=512`; latent 512 + rope 64 dims;
  `kv_a_proj_with_mqa` → `kv_b_proj` expansion to 64 heads.
- **DSA indexer**: 32 index heads × 128 dim, FP8 keys; selects **top-2048**
  key positions per query. 44 layers own an indexer; **28 layers reuse the
  top-k of the nearest earlier indexer layer** (`indexer_types: shared`).
- **Cross-layer KV sharing**: 9 layers (39, 43, …, 71) read **layer 35's**
  latent KV instead of owning one (`kv_source_layers`). Only 63 of 72 layers
  own KV memory.
- **Position**: LeRoPE = YaRN-scaled RoPE with learned per-frequency
  multipliers; **3:1 RoPE:NoPE pattern** (every 4th layer NoPE).
- **SSMax**: length-dependent query scaling on every attention layer and
  indexer.
- **KV size per token**: 47,136 B (upstream `fp8_ds_mla`: 41,328 MLA + 5,808
  indexer); plain FP8 E4M3 layout (AMD path): 42,096 B/token.
  → **256K context ≈ 10.5–11.5 GB KV** for FatCat.

## 3. MoE stack

- Layer 0: dense FFN (`intermediate_size=24576`).
- Layers 1–71: **128 routed experts** (`moe_intermediate_size=6144`),
  **softmax router, top-4**, + **1 shared expert** always active
  (5 active experts/token).
- Expert = gate+up+down = 3×6144×6144 ≈ 113M params ≈ **56.6 MB @4-bit**.
- 9,088 routed experts total ≈ 503 GB @4-bit.

## 4. DSpark speculative drafter (built into checkpoint)

The checkpoint ships a DSpark drafter (`mtp_dspark.*` tensors):
- Parallel (non-causal) drafter: **3 sliding-window MLA layers** (window
  2048, BF16 KV), depth 3, conditioned on target hidden taps at layers
  1, 18, 36, 54, 71.
- RNN Markov head (rank 512) makes draft-position logits depend on earlier
  drafts; position-specific mask embeddings.
- Measured: **~2.9 accepted tokens/step → ~2× output throughput** (vLLM,
  MI350: 2.7× lower per-token latency at c1). Lossless (rejection sampling).
- FatCat implication: implementing DSpark is the single biggest tok/s
  multiplier (§2.3 of `performance.md`) — it also amortizes expert streaming
  over ~3 tokens/step.

## 5. What this means for FatCat

- **Streaming set = routed experts only.** Always-resident: embeddings,
  lm_head, attention + indexers, routers, shared experts, dense layer 0,
  DSpark drafter ≈ **~22B params** (~11–18 GB depending on quant).
- The DSA indexer + MLA + KV sharing make KV small enough that 256K context
  fits in ~11 GB — bandwidth-friendly on Strix Halo.
- 28 shared-indexer layers and 9 KV-sharing layers reduce per-token work.
- Router is softmax top-4 (no group constraints: `n_group=topk_group=1`).

## 6. GGUF footprint & FatCat memory budget (128 GB RAM, 256K context)

Approximate full-model sizes (~1T params): Q4_K_M ≈ 600 GB, Q2_K ≈ 330 GB,
IQ2_XXS ≈ 260 GB. None fits in RAM → streaming is mandatory (by design).

Budget on 128 GB:

| Block | Estimate |
|---|---|
| KV cache @256K (FP8 latent + indexer) | ~11 GB |
| Always-resident weights (≈22B params, Q4–Q6 mix) | ~13–18 GB |
| Engine arenas, io_uring buffers, OS | ~8–10 GB |
| **Warm expert budget** | **~90–95 GB** |

→ warm capacity ≈ **1,600 experts @Q4 (≈18%)** or ≈ **3,300 @Q2_K (≈37%)**
of the 9,088 routed experts. P1 traces decide the quant mix.

**First test quant (locked 2026-11-01):** `UD-IQ1_S` (~1.7 bpw, **233 GB**,
6 split parts, at `/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/`).
Measured from the real file (1,602 tensors):

- Experts are **stacked tensors**: `blk.N.ffn_{gate,up,down}_exps.weight`
  with dims `6144×6144×128` (213 expert tensors = 71 layers × 3 matrices).
  Expert `e` sits at stride `e × 6144×6144` elements inside each tensor →
  streaming one expert = 3 contiguous slice-reads at computed offsets.
- Unsloth UD mixed quant per tensor: 43 tensors IQ2_XXS (9.73 MB/matrix/expert),
  170 tensors IQ1_S (7.37 MB) → **avg expert ≈ 23.5 MB** (214 GB routed total).
- Always-resident (Q6_K/Q8_0/F32): **19 GB** measured (attention, indexers,
  shared experts, dense layer 0, embeddings, routers).
- Warm budget (~90–95 GB) ⇒ **≈ 4,000 experts ≈ 44%** of 9,088 can stay warm.
- IQ1_S dequant is table-based and slow — acceptable for concept validation;
  quality is deliberately low at 1 bit (mechanism testing only).

**Development slice (built 2026-11-01):**
`/home/user/data/models/fatcat-slices/ML4-IQ1_S-layers0-2.gguf` — layers 0–2
(dense layer 0 + 2 MoE layers), embeddings, lm_head, tokenizer; 77 tensors,
10.61 GB, produced by `tools/gguf-slice` with corrected `block_count=3` and
truncated per-layer KV arrays. All early forward-pass/streaming tests run on
this slice (shared-machine RAM budget, see project.md §3).

**Deferred step (user decision 2026-11-01):** repack/modify the GGUF to apply
**per-layer / per-tier mixed quantization** (e.g. higher bits for hot experts
and attention, lower bits for cold experts, expert-contiguous page-aligned
layout for single-read streaming). Tooling will live in `tools/`.
