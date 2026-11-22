#!/usr/bin/env bash
# gpu-gemm-test.sh — quick standalone GPU check of the D1 batched GEMM path.
# RUN WITH THE ASSISTANT LLM STOPPED (hipMalloc fails otherwise -> CPU fallback).
#
# Measures batched-prefill tok/s on GPU @ 18 GB tier (chunk 128/512) and
# verifies the final logits are BIT-identical to the sequential GPU baseline
# captured in perf-validation-20261120-0324 (pvlg-...-p4_seq.bin).
# Reference numbers from that run: seq ~1.0 tok/s, chunk128 2.94 (GEMV),
# chunk512 3.41 (GEMV). The batched GEMM kernels should beat those.
#
# Usage: tools/gpu-gemm-test.sh
# Env:   MODEL=..., TOK=... (default mixed500)
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL=${MODEL:-/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf}
TOK=${TOK:-/home/user/data/fatcat-traces/mixed500.tokens}
OUT=/home/user/data/fatcat-traces
STAMP=$(date +%Y%m%d-%H%M)
REPORT=$OUT/gpu-gemm-test-$STAMP.txt
REF=$OUT/gg-seq-$STAMP.bin    # fresh sequential baseline from THIS build

{
  echo "=== GPU batched-GEMM quick test — $(date) ==="
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  echo "MemAvailable: ${avail} GB  (want >= 100 GB — assistant stopped?)"
  echo
  echo "-- kernel-level: GPU gemm vs gemv bit-identity (Q6_K/Q8_0) --"
  ./build/gputest "$MODEL" more 2>&1 | tail -7
  echo
  echo "-- sequential baseline (this build, for bit-identity ref) --"
  t0=$(date +%s)
  FATCAT_GPU=1 FATCAT_BATCH=0 FATCAT_WARM_MB=18432 FATCAT_MEM_LIMIT_MB=1 FATCAT_KV_F16=1 \
  FATCAT_KV_TARGET=600 ./build/logit-dump "$MODEL" --tokens "$TOK" \
      --out "$REF" > "$OUT/gg-seq-$STAMP.log" 2>&1
  rc=$?; t1=$(date +%s)
  echo "rc=$rc wall=$((t1 - t0))s"
  grep -E 'prefill|warm tier \[' "$OUT/gg-seq-$STAMP.log" | tail -2
  echo
  for B in 128 512; do
    echo "-- batch $B (GEMM kernels) --"
    t0=$(date +%s)
    FATCAT_GPU=1 FATCAT_WARM_MB=18432 FATCAT_MEM_LIMIT_MB=1 FATCAT_KV_F16=1 \
    FATCAT_KV_TARGET=600 ./build/logit-dump "$MODEL" --tokens "$TOK" \
        --batch "$B" --out "$OUT/gg-b$B-$STAMP.bin" > "$OUT/gg-b$B-$STAMP.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "rc=$rc wall=$((t1 - t0))s"
    grep -E 'batched prefill|warm tier \[' "$OUT/gg-b$B-$STAMP.log" | tail -2
    if [ -f "$REF" ]; then
      tail -c 819200 "$REF" > /tmp/gg-ref-last.bin
      tail -c 819200 "$OUT/gg-b$B-$STAMP.bin" > /tmp/gg-out-last.bin
      echo "bit-identity vs sequential GPU baseline:"
      python3 tools/py/cmp_logits.py /tmp/gg-ref-last.bin /tmp/gg-out-last.bin | tail -1
    else
      echo "(baseline $REF not found - skipping bit-identity check)"
    fi
    echo
  done
  echo "-- default config (chunk 256) vs sequential — policy100 --"
  for MODE in seq b256; do
    BARG=(FATCAT_BATCH=0)
    [ "$MODE" = b256 ] && BARG=(FATCAT_BATCH=256)
    t0=$(date +%s)
    env FATCAT_GPU=1 "${BARG[@]}" FATCAT_WARM_MB=18432 FATCAT_MEM_LIMIT_MB=1 FATCAT_KV_F16=1 \
    FATCAT_KV_TARGET=234 ./build/logit-dump "$MODEL" \
        --tokens /home/user/data/fatcat-traces/policy100.tokens \
        --out /tmp/gg-p100-$MODE.bin > "$OUT/gg-p100-$MODE-$STAMP.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "p100 $MODE: rc=$rc wall=$((t1 - t0))s"
    grep -E 'prefill|warm tier \[' "$OUT/gg-p100-$MODE-$STAMP.log" | tail -2
  done
  echo
  echo "report: $REPORT"
  echo "logs:   $OUT/gg-b{128,512}-$STAMP.log"
} 2>&1 | tee "$REPORT"
