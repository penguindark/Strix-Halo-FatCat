#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
# spec-oracle.sh — H2 upper-bound benchmark: cost of batched verification
# forwards (M=K) vs single-token decode, under identical warm-tier state.
#
# The assistant LLM MUST be stopped first (needs the tuned standalone
# recipe: ~78 GB of the 128 GB unified memory).
#
# Usage:
#   tools/spec-oracle.sh TOKENS_FILE [N] [KLIST]
#     TOKENS_FILE  whitespace token ids (tools/py/tokenize.py output)
#     N            reference tokens to generate/replay (default 24)
#     KLIST        comma-separated verify widths (default 2,3,4,6)
#
# Example:
#   .venv/bin/python tools/py/tokenize.py Manuals/model/tokenizer.json \
#       < some.txt > /home/user/data/fatcat-traces/oracle-p.tokens
#   tools/spec-oracle.sh /home/user/data/fatcat-traces/oracle-p.tokens 24 2,3,4,6
#
# Reading the table: R = T_rows(K)/T_fwd(1) is the verification cost
# ratio; E(a,K) = (1-a^(K+1))/(1-a) tokens accepted per forward at
# draft acceptance a; tok/s columns = E/T_rows(K), i.e. the ORACLE
# ceiling of lossless spec-decode for that K and acceptance. Compare
# against the single-forward baseline line. If tok/s at a=0.7 is below
# ~1.6x baseline, a draft model is not worth building.
#
# Env overrides mirror chat-gpu.sh: WARM_MB, MEM_LIMIT_MB, CTX.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf}
TOK=${1:?usage: spec-oracle.sh TOKENS_FILE [N] [KLIST]}
N=${2:-24}
KL=${3:-2,3,4,6}

export FATCAT_GPU=1
export FATCAT_WARM_MB=${WARM_MB:-65536}
export FATCAT_MEM_LIMIT_MB=${MEM_LIMIT_MB:-110000}
export FATCAT_KV_F16=1
export FATCAT_KV_HUGEPAGE=1
export FATCAT_IDX=1
# KV must hold prompt + N + verify slack; prompt size is visible in the file
NP=$(wc -w < "$TOK")
export FATCAT_KV_TARGET=$(( NP > ${CTX:-4096} ? NP + N + 64 : ${CTX:-4096} ))
export FATCAT_PROF=${FATCAT_PROF:-1}

exec ./build/fatcat-run "$MODEL" --tokens "$TOK" -n "$N" --spec-oracle "$KL"
