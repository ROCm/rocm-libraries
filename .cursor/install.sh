#!/usr/bin/env bash
# Idempotent bootstrap for the rocm-libraries Cloud Agent development environment.
#
# The full ROCm superbuild (see CONTRIBUTING.md) requires the ROCm toolchain
# (amdclang/HIP) and AMD GPUs, which are not present on a CPU-only agent VM.
# This script prepares the reproducible, CPU-runnable development experience:
# the repository-wide pre-commit quality gate and the Python-based components
# (e.g. Tensile) whose logic tests run without a GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="$(command -v python3)"
echo "==> Using $("$PY" --version) at $PY"

echo "==> Installing Python dev tooling (pre-commit, pytest) into the user site"
"$PY" -m pip install --user --upgrade pip
"$PY" -m pip install --user pre-commit pytest

# Tensile is a pure-Python GEMM/tensor-contraction code generator. Its unit
# tests exercise generator logic on CPU; installing its runtime dependencies
# lets those tests run in this environment.
if [ -f shared/tensile/requirements.txt ]; then
  echo "==> Installing Tensile runtime dependencies"
  "$PY" -m pip install --user -r shared/tensile/requirements.txt
fi

# Pre-fetch and build every pre-commit hook environment (black, clang-format,
# bandit, cmake-lint, and the local Python hooks) so the quality gate runs
# offline and fast for the agent. Cached under ~/.cache/pre-commit.
echo "==> Pre-installing pre-commit hook environments"
"$PY" -m pre_commit install-hooks

echo "==> Development environment ready"
