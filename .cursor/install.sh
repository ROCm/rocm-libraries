#!/usr/bin/env bash
#
# Idempotent Cloud Agent bootstrap for the ROCm Libraries super-repo.
#
# Scope: this super-repo consolidates AMD's ROCm GPU math/communication
# libraries. A full superbuild (`cmake --preset release:all`) requires the ROCm
# toolchain (amdclang/hipcc) and AMD GPU hardware, neither of which exists in a
# CPU-only Cloud Agent VM. This script therefore prepares the development
# tooling that is meaningful without a GPU:
#   * Tensile (shared/tensile) - the pure-Python GEMM/tensor-contraction kernel
#     generator, whose large unit-test suite runs host-only.
#   * pre-commit - the repository-wide lint/format gate used across projects.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${ROCM_LIBS_VENV:-$HOME/.venvs/rocm-libraries}"

echo "==> Installing system packages"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip

echo "==> Creating Python virtual environment at $VENV"
if [ ! -x "$VENV/bin/python3" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python3 -m pip install --upgrade pip

echo "==> Installing Tensile (shared/tensile) and test tooling"
pip install -r shared/tensile/requirements.txt
pip install pytest pytest-xdist pytest-cov filelock joblib
pip install -e shared/tensile

echo "==> Installing repository lint/format tooling"
pip install pre-commit

echo "==> Environment ready. Activate with: source $VENV/bin/activate"
