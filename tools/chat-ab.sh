#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
# chat-ab.sh — external-validity A/B for the Pista A+B decode levers, on a
# realistic multi-turn chat prompt (NOT the 474-tok benchmark) and a longer
# generation (-n 256). Confirms the +19% decode gain holds off-benchmark.
#
# Runs the SAME prompt twice, fresh process each (fresh warm tier, like the
# benchmark methodology):
#   NEW = current defaults (FATCAT_SPECP=2, FATCAT_RAND=1)
#   OLD = pre-Pista behaviour (FATCAT_SPECP=0 FATCAT_RAND=0)
#
# STOP THE ASSISTANT LLM FIRST: this box shares 128 GB with it and the GPU
# measurement is invalid while it runs. ~16 min total (2 runs).
#
# Usage: tools/chat-ab.sh [MODEL.gguf]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf}"
TOK=/home/user/data/fatcat-traces/chat-real.tokens
OUT=/home/user/data/fatcat-traces
N=${N:-256}

# regenerate the prompt if missing
if [[ ! -s $TOK ]]; then
  .venv/bin/python tools/py/chat-prompt.py tools/data/chat-real.json > "$TOK"
fi
echo "prompt: $(wc -w < "$TOK") tokens, generating -n $N"

run() {  # run LABEL [extra env assignments...]
  local label=$1; shift
  echo "=== $label ==="
  env FATCAT_GPU=1 FATCAT_WARM_MB=65536 FATCAT_MEM_LIMIT_MB=110000 \
      FATCAT_KV_F16=1 FATCAT_KV_HUGEPAGE=1 FATCAT_IDX=1 FATCAT_KV_TARGET=4096 \
      "$@" \
      ./build/fatcat-run "$MODEL" --tokens "$TOK" -n "$N" \
    2>&1 | grep -E "warm bytes|decode nvme|prefill:|decode:" \
    | tee "$OUT/chat-ab-$label.txt"
}

run new
run old FATCAT_SPECP=0 FATCAT_RAND=0

echo
echo "=== summary (decode tok/s) ==="
for label in new old; do
  line=$(grep -h "^prefill:" "$OUT/chat-ab-$label.txt" || true)
  printf "%-4s  %s\n" "$label" "$line"
done
echo "expect: NEW decode > OLD by ~15-20% if the Pista A+B levers generalize"
echo "        to real chat content and a 256-tok generation."
