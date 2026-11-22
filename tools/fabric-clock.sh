#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
# fabric-clock.sh — hold the AMD GPU fabric clock at max (D0, Session 38).
#
# On Strix Halo (gfx1151) the fabric/memory-side clock (fclk) drops when the
# GPU is idle-ish, which caps unified-memory bandwidth — the resource FatCat's
# streaming GEMV/GEMM kernels live on. llama.cpp-family forks (gufo) document
# holding it as REQUIRED for full bandwidth. FatCat runs are bandwidth-bound,
# so this is a free win for GPU prefill/decode.
#
# Usage (root):
#   sudo tools/fabric-clock.sh status   # show current level + fclk table
#   sudo tools/fabric-clock.sh hold     # manual DPM + fclk pinned to top step
#   sudo tools/fabric-clock.sh reset    # back to auto (recommended when done)
#
# A/B protocol: measure with tools/gpu-gemm-test.sh before/after `hold`.
# Note: settings do not survive reboot.
set -u
DEV=/sys/class/drm/card1/device
LVL=$DEV/power_dpm_force_performance_level
FCLK=$DEV/pp_dpm_fclk

case "${1:-status}" in
  status)
    echo "performance level: $(cat "$LVL")"
    echo "fclk table:"; cat "$FCLK"
    ;;
  hold)
    [ -w "$LVL" ] && [ -w "$FCLK" ] || { echo "need root (cannot write sysfs)"; exit 1; }
    top=$(awk '{gsub(/Mhz/,""); if ($1+0>m) {m=$1+0; s=$0}} END {print s}' "$FCLK")
    echo manual > "$LVL"
    echo "${top%% *}" > "$FCLK"
    echo "held fclk at: $(grep '\*' "$FCLK" || cat "$FCLK" | tail -1)"
    echo "level=$(cat "$LVL")"
    ;;
  reset)
    echo auto > "$LVL"
    echo "level=$(cat "$LVL")"
    ;;
  *) echo "usage: $0 {status|hold|reset}"; exit 2 ;;
esac
