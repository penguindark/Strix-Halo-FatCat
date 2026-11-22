#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
# ab-policy.sh — warm-tier eviction policy A/B (lfu vs s3fifo), GPU path.
#
# Runs logit-dump once per policy over the same token corpus, then compares
# (a) warm-tier hit rates and (b) logits bit-identity (the policy must be
# numerically transparent). Everything is appended to a report file.
#
# Usage (from anywhere):
#   tools/ab-policy.sh [MODEL] [TOKENS] [WARM_MB]
# Defaults: Mistral-Large-4 IQ1_S part1, mixed1000 corpus, 18 GB tier.
#
# RAM note: runs WITHOUT a watchdog — meant for the standalone box (assistant
# LLM stopped). For a runaway guard prefix the inner command with
# `./tools/mem-run 61440` (kills the engine above 60 GB anonymous).
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL=${1:-/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf}
TOK=${2:-/home/user/data/fatcat-traces/mixed1000.tokens}
WARM=${3:-18432}

OUT=/home/user/data/fatcat-traces
STAMP=$(date +%Y%m%d-%H%M)
REPORT=$OUT/ab-policy-$STAMP.txt
NTOK=$(wc -w < "$TOK")

{
  echo "=== FatCat policy A/B — $(date) ==="
  echo "model: $MODEL"
  echo "corpus: $TOK ($NTOK tokens)  warm tier: $WARM MB  GPU: on  f16 KV"
  echo

  for pol in lfu s3fifo; do
    echo "-- policy=$pol --"
    t0=$(date +%s)
    FATCAT_POLICY=$pol FATCAT_GPU=1 FATCAT_WARM_MB=$WARM \
    FATCAT_MEM_LIMIT_MB=1 FATCAT_KV_F16=1 FATCAT_KV_TARGET=$((NTOK + 100)) \
      ./build/logit-dump "$MODEL" --tokens "$TOK" \
        --out "$OUT/ablg-$STAMP-$pol.bin" > "$OUT/ablog-$STAMP-$pol.txt" 2>&1
    rc=$?
    t1=$(date +%s)
    echo "rc=$rc  wall=$((t1 - t0))s  approx $((NTOK / (t1 - t0 + 1))) tok/s"
    grep -E 'warm tier' "$OUT/ablog-$STAMP-$pol.txt" | tail -1
    echo
  done

  echo "-- logits comparison (expect bit-identical) --"
  .venv/bin/python tools/py/cmp_logits.py \
    "$OUT/ablg-$STAMP-lfu.bin" "$OUT/ablg-$STAMP-s3fifo.bin" 2>&1 | tail -1
  echo
  echo "verdict: compare the two hit% lines above; the policy with the"
  echo "higher hit% at tiers below the working-set size wins."
  echo "report: $REPORT"
} 2>&1 | tee "$REPORT"
