# FatCat

Copyright (c) 2026 Dario Deledda — released under the MIT License (see
`LICENSE`).

A streaming LLM inference engine written in C23, specialized for
**mistralai/Mistral-Large-4-1T-A52B** (1T-param MoE, ~52B active) on **AMD
Strix Halo** (gfx1151, e.g. Ryzen AI Max+ 395, 128 GB unified memory).

The model (~233 GB at IQ1_S) does not fit in RAM, so FatCat treats memory as
a three-tier hierarchy and streams routed experts from NVMe on demand:

| Tier | Location | Role |
|---|---|---|
| Warm | system RAM (`FATCAT_WARM_MB`, S3-FIFO cache) | hot experts, zero-copy to the iGPU (HSA userptr) |
| Cold | NVMe SSD | everything else, streamed via `io_uring` |
| Compute | iGPU (gfx1151) + Zen 5 CPU (AVX-512) | HIP kernels + CPU reference kernels |

Decode is NVMe-bandwidth-bound by design; the tuned recipe reaches
**~1.4 tok/s decode** end-to-end. See `Manuals/io/memory-tuning.md` for the
measured operating points and `development_log.md` for the full history.

## Requirements

**Hardware**

- AMD Strix Halo (gfx1151) with 128 GB unified memory, x86_64.
  Other platforms are unsupported and untested (the build warns).
- NVMe SSD with ≥ 300 GB free (the IQ1_S model is 233 GB in 6 GGUF splits).
- ≥ 100 GB of *free* RAM at run time for the tuned recipe (see RAM rules).

**Software**

- Linux kernel ≥ 6.1 (`io_uring`; falls back to `pread` if unavailable).
- GCC ≥ 14 (C23 + `-march=znver5`; developed with GCC 15).
- CMake ≥ 3.28.
- ROCm 10.1 with gfx1151 support (TheRock build recommended) for the GPU
  backend: needs `hipcc`, `rocBLAS`, `hipBLASLt`. Without ROCm the build
  silently falls back to CPU-only (correct but ~50× slower — testing only).
- Python ≥ 3.11 with `tokenizers` and `huggingface_hub` (chat + tokenizer
  tools only).

## 1. Install dependencies (Ubuntu 25.x example)

```bash
sudo apt install gcc cmake libgomp1
# ROCm: install TheRock/ROCm 10.1 for gfx1151 wherever you like, e.g.
#   ~/rocm-10.1/install   (headers in include/, libs in lib/, bin/hipcc)
```

## 2. Build

```bash
git clone <this-repo> fatcat && cd fatcat
cmake -B build -DROCM_PATH=$HOME/rocm-10.1/install
cmake --build build -j
```

Notes:

- `-DROCM_PATH` defaults to `/home/user/rocm-10.1/install`; point it at your
  ROCm prefix. If ROCm is not found the configure step prints
  `ROCm not found ... building CPU-only`.
- The HIP kernels (`src/gpu/kernels.hip`) are compiled at build time with
  `hipcc -O3 -ffp-contract=off --offload-arch=gfx1151`. Do **not** remove
  `-ffp-contract=off`: GPU/CPU bit-identity between gemv and gemm kernels
  depends on it (see CMakeLists comment).
- Useful binaries land in `build/`: `fatcat-run` (engine + server mode) and
  the test/validation tools.

## 3. Download the model

The reference quant is `unsloth/Mistral-Large-4-1T-A52B-GGUF`, `UD-IQ1_S`
split (~233 GB, 6 files). Use the venv from step 4 or any `hf` CLI:

```bash
python3 -m venv .venv && .venv/bin/pip install huggingface_hub tokenizers
.venv/bin/hf download unsloth/Mistral-Large-4-1T-A52B-GGUF \
    --include "UD-IQ1_S/*" \
    --local-dir /home/user/data/models
```

Point `MODEL` at the **first shard** (`...-00001-of-00006.gguf`); the engine
finds the rest. The tokenizer (`tokenizer.json`, Mistral-Large-4 tekken,
HF ids == GGUF ids) is already in the repo at `Manuals/model/tokenizer.json`.

Sanity-check the download:

```bash
./build/gguf-inspect "$MODEL"        # tensors, quant types, metadata
```

## 4. Run

### Interactive chat (the easy path)

```bash
tools/chat-gpu.sh                       # 32K context, tuned defaults
tools/chat-gpu.sh --reasoning medium    # more thinking, slower replies
CTX=8192 tools/chat-gpu.sh              # smaller context
```

`chat-gpu.sh` bakes in the measured max-speed recipe (see below) and drives
`build/fatcat-run --serve` through `tools/py/chat.py` (streaming replies,
thinking rendered dim, `/reset` `/info` `/max` `/sys` `/quit` commands).

### Batch generation

```bash
.venv/bin/python tools/py/tokenize.py Manuals/model/tokenizer.json < prompt.txt > /tmp/p.tokens
FATCAT_GPU=1 FATCAT_WARM_MB=65536 FATCAT_MEM_LIMIT_MB=110000 \
FATCAT_KV_F16=1 FATCAT_KV_TARGET=4096 \
./build/fatcat-run "$MODEL" --tokens /tmp/p.tokens -n 64
```

### Server protocol

`./build/fatcat-run "$MODEL" --serve` reads line commands on stdin
(`P` prefill, `G` generate, `I` info, `R` reset, `Q` quit) and keeps the KV
across commands — see the header of `tools/fatcat-run.c`.

## 5. Tuning & RAM rules (read before first run)

| env | tuned value | meaning |
|---|---|---|
| `FATCAT_GPU` | `1` | GPU backend (`0`/`--cpu`: CPU reference, testing only) |
| `FATCAT_WARM_MB` | `65536` | warm expert tier; the perf curve flattens at 64 GB |
| `FATCAT_MEM_LIMIT_MB` | `110000` | hard budget guard. **Never `0`** — an uncapped page cache once OOM-killed the desktop (Session 40c) |
| `FATCAT_KV_F16` | `1` | f16 KV (validated cos=1.0) |
| `FATCAT_KV_TARGET` | per use | pre-allocated KV tokens (chat: `--ctx` sets it) |
| `FATCAT_KV_HUGEPAGE` | `1` | transparent hugepages on KV/indexer caches |
| `FATCAT_IDX` | `1` | DSA sparse attention (keeps decode flat as context grows) |
| `FATCAT_SPECP` | `0` (default) | speculative expert prefetch: costs ~1 GB/token of NVMe, off |
| `FATCAT_NOPROMO` | `2` (default) | prefill anti-thrash: prompt experts join the warm tier only on a 2nd reference (+6% decode, +9% prefill measured) |
| `FATCAT_PROF` | `1` for runs | per-phase timing breakdown on stderr |

- The budget counts tier + KV + reserve only. On Strix Halo `hipMalloc`
  **is** system RAM (GTT, ~6–8 GB) and the GGUF page cache grows on its own:
  keep `tier ≤ free_RAM − KV − ~9 GB`. At 128 GB: 64 GB tier is the sweet
  spot; 86 GB adds ~1% decode; **96 GB swaps** — don't.
- Expect the first 2–3 chat turns to be ~30% slower while the warm tier
  fills (~64 GB of experts at ~5 GB/s NVMe).

## 6. Validation & tooling

- `tools/perf-validation.sh` — engine GPU-vs-CPU logit validation.
- `tools/decode-tier-sweep.sh` — warm-tier vs decode-throughput sweep.
- `tools/gguf-inspect`, `tools/gguf-slice`, `tools/expert-profiler` (+
  `tools/analyze_trace.py`) — model inspection and expert-routing analysis.
- `tools/mem-run <MB> <cmd>` — run a command under a memory cap (used by
  the CPU smoke tests).

## License

Copyright (c) 2026 **Dario Deledda**. Code in this repository is released
under the **MIT License** (see `LICENSE`). Notes:

- **Model weights are not part of this project.** You download them
  yourself (step 3); they remain under Mistral's and the quant publisher's
  respective licenses.
- `Manuals/model/tokenizer.json`, `tekken.json` and `chat_template.jinja`
  are Mistral artifacts (Apache-2.0) redistributed for convenience.
- Some reference documents under `Manuals/` (AMD Strix Halo PPR, arXiv
  papers, vendor posts) are **not redistributable** and are deliberately
  excluded from the repository; `Manuals/README.md` indexes them with
  their original sources.

## Repository layout

```
src/            engine: core/ (engine.c), model/ (ml4.c model core),
                stream/ (warm tier, io_uring), gpu/ (HIP kernels), io/ (GGUF)
include/fatcat/ public headers
tools/          CLI tools, chat driver (py/chat.py), validation scripts
Manuals/        platform/model documentation (tokenizer.json lives here)
docs/           design notes (streaming policy, ...)
project.md      master development tracker
development_log.md  append-only session log
```
