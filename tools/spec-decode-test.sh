#!/usr/bin/env bash
# spec-decode-test.sh — standalone A/B of D3 prompt-lookup speculative decode.
# RUN WITH THE ASSISTANT LLM STOPPED (hipMalloc fails otherwise -> CPU fallback).
#
# Greedy generation on policy100 (48 tokens) at the 18 GB warm tier:
#   spec=0 (plain greedy) vs spec=1 (prompt-lookup, draft 8)
#   plus draft-size sweep and a repetitive-input case if REP_TOK is set.
# Correctness bar: the generated token sequences MUST be identical
# (speculation is lossless for greedy sampling).
#
# Usage: tools/spec-decode-test.sh
# Env:   MODEL=..., TOK=... (default policy100), N=48
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL=${MODEL:-/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf}
TOK=${TOK:-/home/user/data/fatcat-traces/policy100.tokens}
N=${N:-48}
OUT=/home/user/data/fatcat-traces
STAMP=$(date +%Y%m%d-%H%M)
REPORT=$OUT/spec-decode-test-$STAMP.txt

{
  echo "=== D3 speculative decode A/B — $(date) ==="
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  echo "MemAvailable: ${avail} GB  (want >= 100 GB — assistant stopped?)"
  echo

  run() {  # name TOK extra-env...
    local name=$1 tok=$2; shift 2
    echo "-- $name --"
    local t0 t1 rc
    t0=$(date +%s)
    env FATCAT_GPU=1 FATCAT_WARM_MB=18432 FATCAT_MEM_LIMIT_MB=1 FATCAT_KV_F16=1 \
        FATCAT_KV_TARGET=600 "$@" \
        ./build/fatcat-run "$MODEL" --tokens "$tok" -n "$N" \
        > "$OUT/sd-$name-$STAMP.tok" 2> "$OUT/sd-$name-$STAMP.log"
    rc=$?; t1=$(date +%s)
    echo "rc=$rc wall=$((t1 - t0))s"
    grep -E 'decode:|spec:' "$OUT/sd-$name-$STAMP.log" | tail -2
    echo
  }

  RL=/home/user/data/fatcat-traces/replist.tokens
  run base      "$TOK" FATCAT_SPEC=0
  run spec8     "$TOK" FATCAT_SPEC=1
  run base-rep  "$RL"  FATCAT_SPEC=0
  run spec8-rep "$RL"  FATCAT_SPEC=1
  run spec16-rep "$RL" FATCAT_SPEC=1 FATCAT_SPEC_DRAFT=16

  echo "token-sequence identity (all must be IDENTICAL):"
  for v in spec8; do
    if diff -q "$OUT/sd-base-$STAMP.tok" "$OUT/sd-$v-$STAMP.tok" >/dev/null; then
      echo "  base vs $v: IDENTICAL"
    else
      echo "  base vs $v: *** MISMATCH ***"
    fi
  done
  for v in spec8-rep spec16-rep; do
    if diff -q "$OUT/sd-base-rep-$STAMP.tok" "$OUT/sd-$v-$STAMP.tok" >/dev/null; then
      echo "  base-rep vs $v: IDENTICAL"
    else
      echo "  base-rep vs $v: *** MISMATCH ***"
    fi
  done
  echo
  echo "report: $REPORT"
} | tee "$REPORT"
