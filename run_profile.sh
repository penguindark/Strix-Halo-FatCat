#!/bin/bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
# FatCat expert-profiler — full-model run (949 tokens, GPU).
# Shut down the assistant LLM first so the profiler gets all the RAM.
#
# Usage:  ./run_profile.sh
#
# Health checklist (also in RUN.md):
#   selftest: ALL KERNELS OK        <- kernels correct (aborts rc=3 otherwise)
#   health=ok on every progress line
#   upload_fails=0 in "gpu stats"   <- all fixed weights on GPU
#   DONE rc=0                       <- run complete and healthy
# Memory is self-capped: experts stream via pread() into a 240 MB pool;
# page-cache release kicks in past 80 GB system usage (override:
# FATCAT_MEM_LIMIT_MB=<MB>, 0 disables). Progress line shows mem=<MB> used.
# Ctrl-C is safe: the trace is checkpointed every 64 tokens.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf
TOKENS=/home/user/data/prompts/profile.tokens
TRACE=/home/user/data/prompts/full-gpu.trace
LOG=/home/user/data/prompts/profile-gpu.log

if [ ! -x build/expert-profiler ]; then
    echo "build/expert-profiler missing — run: cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

stdbuf -oL ./build/expert-profiler "$MODEL" \
    --tokens "$TOKENS" \
    --trace  "$TRACE" \
    2>&1 | tee "$LOG"
