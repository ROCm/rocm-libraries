#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Run the training pipeline for one config:
#
#   scripts/run_pipeline.sh <config.yaml> --mode full [run_orchestrator.py args...]
#
# Environment (configs refer to the TILEWRIGHT_* paths as ${VAR}):
#   TILEWRIGHT_BENCH_ROOT   rocm-libraries checkout whose hipBLASLt build the run
#                           benchmarks. Default: the rocm-libraries checkout
#                           holding this script (shared/tilewright/training_pipeline);
#                           elsewhere it stays unset, and the orchestrator names it
#                           when the config refers to it.
#   TILEWRIGHT_DEPLOY_ROOT  checkout that receives the trained weights. Default as
#                           for TILEWRIGHT_BENCH_ROOT.
#   TILEWRIGHT_DATASETS     directory of stage08 datasets, for configs that list any.
#   TILEWRIGHT_AUDIT_ROOT   directory of audit sets, for configs that name one.
#   PYTHON                  interpreter with requirements.txt and the tilewright
#                           module installed. Default: python3.
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:?usage: run_pipeline.sh <config.yaml> [run_orchestrator.py args...]}"
shift

REPO_ROOT="$(cd "$PIPELINE_DIR/../../.." && pwd)"
if [[ "$PIPELINE_DIR" == "$REPO_ROOT/shared/tilewright/training_pipeline" ]]; then
  export TILEWRIGHT_BENCH_ROOT="${TILEWRIGHT_BENCH_ROOT:-$REPO_ROOT}"
  export TILEWRIGHT_DEPLOY_ROOT="${TILEWRIGHT_DEPLOY_ROOT:-$REPO_ROOT}"
fi

# hipBLASLt loads HIPBLASLT_TENSILE_LIBPATH verbatim instead of the library
# subtree that matches the device's ASIC revision.
unset HIPBLASLT_TENSILE_LIBPATH

commit_of() {
  git -C "$1" rev-parse --short HEAD 2>/dev/null || echo "not a git checkout"
}
root_of() {
  if [[ -n "${!1:-}" ]]; then
    echo "${!1} ($(commit_of "${!1}"))"
  else
    echo "unset"
  fi
}
echo "config      : $CFG"
echo "bench root  : $(root_of TILEWRIGHT_BENCH_ROOT)"
echo "deploy root : $(root_of TILEWRIGHT_DEPLOY_ROOT)"

exec "${PYTHON:-python3}" "$PIPELINE_DIR/run_orchestrator.py" --config "$CFG" "$@"
