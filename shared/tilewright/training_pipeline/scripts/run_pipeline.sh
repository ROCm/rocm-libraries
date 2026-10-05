#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Run the training pipeline for one config:
#
#   scripts/run_pipeline.sh <config.yaml> --mode full [run_orchestrator.py args...]
#
# Environment (configs refer to the TILEWRIGHT_* paths as ${VAR}):
#   TILEWRIGHT_BENCH_ROOT   rocm-libraries checkout whose hipBLASLt build the run
#                           benchmarks. Default: the checkout holding this script.
#   TILEWRIGHT_DEPLOY_ROOT  checkout that receives the trained weights.
#                           Default: the checkout holding this script.
#   TILEWRIGHT_DATASETS     directory of stage08 datasets, for configs that list any.
#   PYTHON                  interpreter with requirements.txt and the tilewright
#                           module installed. Default: python3.
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PIPELINE_DIR/../../.." && pwd)"
CFG="${1:?usage: run_pipeline.sh <config.yaml> [run_orchestrator.py args...]}"
shift

export TILEWRIGHT_BENCH_ROOT="${TILEWRIGHT_BENCH_ROOT:-$REPO_ROOT}"
export TILEWRIGHT_DEPLOY_ROOT="${TILEWRIGHT_DEPLOY_ROOT:-$REPO_ROOT}"

# hipBLASLt loads HIPBLASLT_TENSILE_LIBPATH verbatim instead of the library
# subtree that matches the device's ASIC revision.
unset HIPBLASLT_TENSILE_LIBPATH

commit_of() {
  git -C "$1" rev-parse --short HEAD 2>/dev/null || echo "not a git checkout"
}
echo "config      : $CFG"
echo "bench root  : $TILEWRIGHT_BENCH_ROOT ($(commit_of "$TILEWRIGHT_BENCH_ROOT"))"
echo "deploy root : $TILEWRIGHT_DEPLOY_ROOT ($(commit_of "$TILEWRIGHT_DEPLOY_ROOT"))"

exec "${PYTHON:-python3}" "$PIPELINE_DIR/run_orchestrator.py" --config "$CFG" "$@"
