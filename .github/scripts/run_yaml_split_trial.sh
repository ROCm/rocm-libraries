#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set -euo pipefail
python3 .github/scripts/benchmark_yaml_split.py --root "$BENCH_ROOT" run \
  --strategy "$1" --workers "$2" --trial "$3" --target gfx1250-strict \
  --client "$BENCH_CLIENT" --compiler "$ROCM_PATH/bin/amdclang++" --timeout 600 \
  2>&1 \
  | sed -E -e 's/ in [0-9]+(\.[0-9]+)?s\b//g' -e 's/ \([0-9]+:[0-9]{2}:[0-9]{2}\)//g' \
      -e 's/ \([0-9][^()]* seconds\)//g' -e 's/ \([0-9]+(\.[0-9]+)? secs elapsed\)//g' \
      -e 's/(End -|_Final) [0-9]+(\.[0-9]+)?s\b/\1/g' -e 's/ done \([0-9]+(\.[0-9]+)?s\)/ done/g' \
  | awk -F, -v OFS=, "$HW_SCRUB" \
  | tee "$GITHUB_WORKSPACE/tensilelite_test_results/$3.log"
