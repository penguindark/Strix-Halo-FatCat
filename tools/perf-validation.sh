#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
# perf-validation.sh — standalone performance validation day.
# RUN WITH THE ASSISTANT LLM STOPPED (needs the full 128 GB + GPU pool).
#
# Parts (default: all four):
#   1  256K capacity ladder  - KV prealloc @263168 f16+hugepages, 48 GB tier
#   2  warm-tier budget sweep - 8/16/32/64 GB, s3fifo (docs/performance.md 3bis)
#   3  GPU vs CPU A/B         - 18 GB tier, s3fifo (refreshes the stale P4 verdict)
#   4  D1 batched prefill     - sequential vs chunk 128/512, GPU, 18 GB tier
#
# Usage:  tools/perf-validation.sh [parts...]      e.g. tools/perf-validation.sh 2 3
# Env overrides: MODEL=..., TOK=... (token corpus file)
# Report: /home/user/data/fatcat-traces/perf-validation-<stamp>.txt
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL=${MODEL:-/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf}
TOK=${TOK:-/home/user/data/fatcat-traces/mixed1000.tokens}
OUT=/home/user/data/fatcat-traces
STAMP=$(date +%Y%m%d-%H%M)
REPORT=$OUT/perf-validation-$STAMP.txt
PARTS="${*:-1 2 3 4}"
NTOK=$(wc -w < "$TOK")

run() {   # run <logkey> <label> <env assignments as one string> [extra cli args]
  local log=$1 label=$2 envs=$3 extra=${4:-}
  echo "-- $label --"
  local t0 t1 rc
  t0=$(date +%s)
  eval "env $envs ./build/logit-dump \"\$MODEL\" --tokens \"\$TOK\" $extra \
      --out \"$OUT/pvlg-$STAMP-$log.bin\"" \
      > "$OUT/pvlog-$STAMP-$log.txt" 2>&1
  rc=$?; t1=$(date +%s)
  echo "rc=$rc  wall=$((t1 - t0))s  approx $((NTOK / (t1 - t0 + 1))) tok/s"
  grep -E 'batched prefill|warm tier \[|kv: capacity|peak RSS' "$OUT/pvlog-$STAMP-$log.txt" | tail -3
  echo
}

{
  echo "=== FatCat performance validation — $(date) ==="
  echo "model:  $MODEL"
  echo "corpus: $TOK ($NTOK tokens)"
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  echo "MemAvailable now: ${avail} GB   (want >= 100 GB — stop the assistant first!)"
  echo

  if [[ " $PARTS " == *" 1 "* ]]; then
    echo "=== PART 1: 256K capacity ladder ==="
    echo "Validates the 262K KV allocation path (f16 + hugepages + indexer),"
    echo "memory ceiling with a 64 GB warm tier, and per-token timing trend."
    echo "NOTE: a FULL 262K-token prefill would take days (expert streaming is"
    echo "~1-2 s/token); this proves capacity, not throughput-at-length."
    # 80 GB budget: KV f16 ~19 GB + indexer ~5.9 GB + reserve ~1 GB + warm 48 GB ~= 74 GB
    run p1_256k "256K capacity: KV_TARGET=263168 f16 hugepage idx, warm 48 GB" \
      "FATCAT_KV_TARGET=263168 FATCAT_KV_F16=1 FATCAT_KV_HUGEPAGE=1 FATCAT_IDX=1 FATCAT_GPU=1 FATCAT_WARM_MB=49152 FATCAT_MEM_LIMIT_MB=1"
  fi

  if [[ " $PARTS " == *" 2 "* ]]; then
    echo "=== PART 2: warm-tier budget sweep (s3fifo, GPU) ==="
    for w in 8192 16384 32768 65536; do
      run "p2_warm$w" "budget sweep warm=${w} MB" \
        "FATCAT_GPU=1 FATCAT_WARM_MB=$w FATCAT_MEM_LIMIT_MB=1 FATCAT_KV_F16=1 FATCAT_KV_TARGET=$((NTOK + 100))"
    done
    echo "expected shape: hit% rises steeply 8->32 GB then flattens as the"
    echo "working set fits; compare against the Session 33 sim curve."
    echo
  fi

  if [[ " $PARTS " == *" 3 "* ]]; then
    echo "=== PART 3: GPU vs CPU A/B (s3fifo, 18 GB tier) ==="
    run p3_gpu "GPU ON" \
      "FATCAT_GPU=1 FATCAT_WARM_MB=18432 FATCAT_MEM_LIMIT_MB=1 FATCAT_KV_F16=1 FATCAT_KV_TARGET=$((NTOK + 100))"
    run p3_cpu "GPU OFF" \
      "FATCAT_GPU=0 FATCAT_WARM_MB=18432 FATCAT_MEM_LIMIT_MB=1 FATCAT_KV_F16=1 FATCAT_KV_TARGET=$((NTOK + 100))"
    echo "P4 verdict (GPU trails CPU) predates zero-copy tier + s3fifo;"
    echo "Session 36 hinted GPU ~2.5x faster. This settles it."
    echo
  fi

  if [[ " $PARTS " == *" 4 "* ]]; then
    echo "=== PART 4: D1 batched prefill (GPU, 18 GB tier) ==="
    echo "Sequential baseline vs FATCAT_BATCH chunks; logits of the batched"
    echo "runs are the final row only - compare with:"
    echo "  tail -c 819200 pvlg-$STAMP-p4_seq.bin > /tmp/p4_last.bin"
    echo "  python3 tools/py/cmp_logits.py /tmp/p4_last.bin pvlg-$STAMP-p4_batch128.bin 1"
    run p4_seq "sequential (FATCAT_BATCH off)" \
      "FATCAT_GPU=1 FATCAT_WARM_MB=18432 FATCAT_MEM_LIMIT_MB=1 FATCAT_KV_F16=1 FATCAT_KV_TARGET=$((NTOK + 100))"
    for B in 128 512; do
      run "p4_batch$B" "batched chunk=$B" \
        "FATCAT_GPU=1 FATCAT_WARM_MB=18432 FATCAT_MEM_LIMIT_MB=1 FATCAT_KV_F16=1 FATCAT_KV_TARGET=$((NTOK + 100))" \
        "--batch $B"
    done
    echo "CPU @ 1 GB tier measured 3x (Session 38); at 18 GB the chunk expert"
    echo "union fits -> expect a larger win. Chunk 512 tests amortization vs"
    echo "tier pressure at GPU speed."
    echo
  fi

  echo "report:  $REPORT"
  echo "logs:    $OUT/pvlog-$STAMP-*.txt   (full per-run stderr)"
} 2>&1 | tee "$REPORT"
