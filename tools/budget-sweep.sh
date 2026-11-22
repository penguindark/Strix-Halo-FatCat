#!/usr/bin/env bash
# budget-sweep.sh — tok/s and warm-tier hit rate vs tier budget.
# Core P3 deliverable (docs/performance.md §3bis). RUN STANDALONE: each
# config's peak RSS is ~16 GB process baseline + the tier budget, so the
# script checks MemAvailable and skips budgets that would not fit.
#
# usage: tools/budget-sweep.sh MODEL.gguf TOKENS_FILE [BUDGET_MB ...]
#        (default budgets: 8192 16384 32768 49152)
set -uo pipefail

MODEL=${1:?usage: budget-sweep.sh MODEL TOKENS [BUDGET_MB...]}
TOK=${2:?usage: budget-sweep.sh MODEL TOKENS [BUDGET_MB...]}
shift 2
BUDGETS=("$@"); ((${#BUDGETS[@]})) || BUDGETS=(8192 16384 32768 49152)

cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/home/user/rocm-10.1/install/lib}
export FATCAT_GPU=${FATCAT_GPU:-0}
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive}
N=$(wc -w < "$TOK")

printf '%-10s %8s %9s %8s %10s %12s\n' budget wall_s tok/s hit% nonblock% peakRSS_MB
for mb in "${BUDGETS[@]}"; do
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    need=$((mb + 20000))
    if (( avail < need )); then
        echo "$mb MB SKIPPED: ${avail} MB available, need ~${need} MB"
        continue
    fi
    t0=$SECONDS
    out=$(FATCAT_WARM_MB="$mb" FATCAT_MEM_LIMIT_MB=100000 \
        ./build/logit-dump "$MODEL" --tokens "$TOK" --out "/tmp/sweep_${mb}.bin" 2>&1)
    el=$((SECONDS - t0))
    echo "$out" | grep -E 'warm tier|peak RSS' | tr '\n' ' '
    awk -v n="$N" -v e="$el" 'BEGIN{printf "wall=%ds %.3f tok/s\n", e, n/e}'
done
