#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
# decode-tier-sweep.sh — read of decode speed vs warm-tier size (~5 min per tier).
# RUN WITH THE ASSISTANT LLM STOPPED (hipMalloc fails otherwise -> CPU fallback).
#
# Decode is expert-bytes-bound, so the warm-tier hit rate IS the decode
# speed curve. This sweep measures steady-state decode tok/s + tier hit%
# at three tier sizes on the same short prompt/generation, enough to see
# whether the curve is still climbing (cache work wins) or flat (streaming
# constant wins -> async staging).
#
# Usage: tools/decode-tier-sweep.sh          (~8 min total)
# Env:   MODEL=..., TOK=... (default policy100), N=24, TIERS="18432 40960 65536"
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL=${MODEL:-/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf}
TOK=${TOK:-/home/user/data/fatcat-traces/policy100.tokens}
N=${N:-24}
TIERS=${TIERS:-"18432 40960 65536 86016"}
OUT=/home/user/data/fatcat-traces
STAMP=$(date +%Y%m%d-%H%M)
REPORT=$OUT/decode-tier-sweep-$STAMP.txt

{
  echo "=== decode tier sweep — $(date) ==="
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  echo "MemAvailable: ${avail} GB  (want >= 100 GB — assistant stopped?)"
  echo "prompt=$TOK  gen=$N tok  tiers=$TIERS"
  echo
  for T in $TIERS; do
    echo "-- tier ${T} MB --"
    t0=$(date +%s)
    # MEM_LIMIT here is only a runaway guardrail: with the assistant stopped,
    # system usage never approaches 110 GB, so drop_experts stays OFF and the
    # warm tier actually caches (MEM_LIMIT_MB=1 would force drop_experts and
    # invalidate the whole sweep — learned the hard way, Session 40c).
    FATCAT_GPU=1 FATCAT_PROF=1 FATCAT_WARM_MB=$T FATCAT_MEM_LIMIT_MB=110000 FATCAT_KV_F16=1 \
    FATCAT_KV_TARGET=600 ./build/fatcat-run "$MODEL" \
        --tokens "$TOK" -n "$N" > /tmp/dts-$T.tok 2> "$OUT/dts-$T-$STAMP.log"
    rc=$?; t1=$(date +%s)
    echo "rc=$rc wall=$((t1 - t0))s"
    grep -E 'decode:|warm tier \[' "$OUT/dts-$T-$STAMP.log" | tail -2
    echo
  done
  echo "NOTE: first tier runs with a cold NVMe page cache; later tiers"
  echo "inherit it, so treat tier-1 numbers as a mild worst case."
  echo "report: $REPORT"
} | tee "$REPORT"
