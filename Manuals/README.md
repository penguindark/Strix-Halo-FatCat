# Manuals — Platform & Library Documentation

Central store for platform and library documentation used by FatCat development.
Documents are downloaded from the web or manually copied here. **Do not edit the
documents themselves**; when adding a document, record it below with source URL,
retrieval date and license notes.

> **Redistribution note:** most documents here (AMD PPR, arXiv papers, vendor
> posts) are third-party copyrighted material kept for local reference only.
> They are listed in `.gitignore` and must NOT be committed or republished.
> The entries below are the canonical sources to fetch them from.

## Index

### strix-halo/ — AMD Strix Halo platform docs
| File | Description | Source |
|---|---|---|
| `57930-A0-PUB_3.00.pdf` / `.html` | AMD Family 1Ah Model 50/51 (Strix Halo) Processor Programming Reference (PPR) 3.00 | AMD product documentation (provided as `AMD strix halo.7z`) |
| `AMD 57930-A0-PUB_3.00.zip` | Original AMD distribution archive of the PPR | AMD |
| `AMD 58550-0.01.pdf` | AMD doc 58550 rev 0.01 (Strix Halo related publication) | AMD |
| `extracted/ppr_text.txt`, `extracted/page_index.txt`, `extracted/section_map.tsv` | Text extraction + page/section index of the PPR for fast searching | Derived locally from the PPR PDF |
| `README.txt`, `New Text Document.txt` | Notes shipped with the original archive | — |

### gpu/ — ROCm / gfx1151
Placeholder. ROCm 10.1 (TheRock gfx1151) is installed locally at
`/home/user/rocm-10.1/install` (headers in `include/`, libs in `lib/`).
Relevant API references (HIP, rocBLAS, hipBLASLt) to be copied here as needed.

### cpu/ — Zen 5 / AVX-512
Placeholder. To hold AMD Zen 5 optimization guides (software optimization,
AVX-512/VNNI/BF16 references).

### model/ — Target model references
| File | Description | Source |
|---|---|---|
| `config.json` | Upstream `mistralai/Mistral-Large-4-1T-A52B` model config | https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/resolve/main/config.json |
| `mistral-large4-model-card.md` | Official model card (usage, variants, license) | https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/raw/main/README.md |
| `mistral-large4-hf-blog.md` | HF release blog: architecture (MLA+DSA, 3:1 RoPE/NoPE, KV sharing, LeRope), GGUF quant list, reference perf | https://raw.githubusercontent.com/huggingface/blog/main/mistral-large4.md |
| `vllm-day0-mistral-large4.md` | vLLM day-0 blog: full architecture table, KV bytes/token, DSpark drafter details, AMD/ROCm implementation (AITER, MXFP4 requant, sparse MLA) | https://raw.githubusercontent.com/vllm-project/vllm-project.github.io/main/_posts/2026-10-29-mistral-large-4.md |
| `dspark-arxiv-2607.05147.html` | DSpark paper: confidence-scheduled speculative decoding (built-in drafter) | https://arxiv.org/abs/2607.05147 |
| `mla-deepseek-v2-arxiv-2405.04434.html` | DeepSeek-V2 paper: MLA (Multi-head Latent Attention) | https://arxiv.org/abs/2405.04434 |
| `deepseek-v32-arxiv-2512.02556.html` | DeepSeek-V3.2 paper: DSA sparse attention (indexer) | https://arxiv.org/abs/2512.02556 |


### io/ — Streaming & memory subsystem notes
| File | Description | Source |
|---|---|---|
| `memory-tuning.md` | 256K memory budget table, GTT/NUMA/THP guidance and validation procedure for Strix Halo | Derived locally (P5 sessions 25–28) |

## Rules
- English only for any notes added here (see `AGENT.md`).
- One subfolder per topic; keep this README as the authoritative index.
