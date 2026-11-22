#!/usr/bin/env bash
# d5-validate.sh — D5d/e acceptance gate (attention GPU kernels + indexer skip).
# RUN WITH THE ASSISTANT LLM STOPPED (needs the GPU pool).
#
# 1. kernel-level bit-identity: gpu-absorb-test (batched wv == gpu gemv,
#    absorb kernel == CPU dequant+dot bit-for-bit)
# 2. e2e: GPU decode logits with the new attention kernels (default) vs
#    with them off (FATCAT_GPU_ATT=0) on policy100. Acceptance = the
#    established GPU-vs-CPU accumulation-noise band (mean cos >= ~0.98,
#    top-1 preserved; P4/P5 precedent, devlog Session ~P5). Bit-identity
#    is required only vs the existing GPU kernels (gate 1), which the new
#    kernels meet exactly; wv rides the GPU gemv kernel whose ULP noise
#    vs CPU is the same class as every other GPU gemv in the engine.
#
# Usage: tools/d5-validate.sh
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL=${MODEL:-/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf}
TOK=${TOK:-/home/user/data/fatcat-traces/policy100.tokens}
OUT=/home/user/data/fatcat-traces
STAMP=$(date +%Y%m%d-%H%M)
REPORT=$OUT/d5-validate-$STAMP.txt
ENV_BASE="FATCAT_GPU=1 FATCAT_WARM_MB=18432 FATCAT_MEM_LIMIT_MB=110000 FATCAT_KV_F16=1 FATCAT_KV_TARGET=600"

{
  echo "=== D5d/e validation — $(date) ==="
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  echo "MemAvailable: ${avail} GB  (assistant stopped?)"
  echo
  echo "-- 1. kernel bit-identity (gpu-absorb-test) --"
  ./build/gpu-absorb-test
  echo
  for MODE in att0 att1; do
    EXTRA=FATCAT_GPU_ATT=0; [ "$MODE" = att1 ] && EXTRA=FATCAT_GPU_ATT=1
    echo "-- 2. logit-dump GPU $MODE (policy100) --"
    t0=$(date +%s)
    env $ENV_BASE $EXTRA ./build/logit-dump "$MODEL" --tokens "$TOK" \
        --out "$OUT/d5lg-$STAMP-$MODE.bin" > "$OUT/d5lg-$STAMP-$MODE.log" 2>&1
    rc=$?; t1=$(date +%s)
    echo "rc=$rc wall=$((t1 - t0))s"
    grep -E 'prefill|decode|health' "$OUT/d5lg-$STAMP-$MODE.log" | tail -2
    echo
  done
  echo "-- cos(att1, att0) on last position --"
  tail -c 819200 "$OUT/d5lg-$STAMP-att0.bin" > /tmp/d5-a0.bin
  tail -c 819200 "$OUT/d5lg-$STAMP-att1.bin" > /tmp/d5-a1.bin
  python3 tools/py/cmp_logits.py /tmp/d5-a0.bin /tmp/d5-a1.bin | tail -1
} 2>&1 | tee "$REPORT"
echo "report: $REPORT"
