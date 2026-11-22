#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
# chat-gpu.sh — interactive chat with the tuned standalone recipe.
#
# The assistant LLM MUST be stopped first (this box has 128 GB shared
# with it; the engine needs ~78 GB: 64 warm tier + KV + GPU GTT).
#
# Usage:  tools/chat-gpu.sh                 # 32K context, tuned defaults
#         CTX=8192 tools/chat-gpu.sh        # smaller context
#         tools/chat-gpu.sh --reasoning medium   # extra flags -> chat.py
#
# Speed levers baked in (all measured, see Manuals/io/memory-tuning.md):
#   WARM_MB=65536   tier sweet spot: curve flattens at 64 GB (S42f)
#   MEM_LIMIT=110000 budget guard; NEVER 0 (OOM incident, S40c)
#   KV_F16=1        half the KV bytes; engine validated at cos=1.0
#   KV_HUGEPAGE=1   THP on the big KV/indexer caches (TLB misses)
#   IDX=1           DSA sparse attention: decode stays flat as the
#                   conversation grows toward the 32K cap
#   SPECP=0         (engine default) speculative expert prefetch costs
#                   ~1 GB/token of NVMe the critical reads need
#   reasoning=low   (chat.py default) model card: low = chat replies
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf}

export FATCAT_GPU=1
export FATCAT_WARM_MB=${WARM_MB:-65536}
export FATCAT_MEM_LIMIT_MB=${MEM_LIMIT_MB:-110000}
export FATCAT_KV_F16=1
export FATCAT_KV_HUGEPAGE=1
export FATCAT_IDX=1

exec .venv/bin/python tools/py/chat.py "$MODEL" --ctx "${CTX:-32000}" "$@"
