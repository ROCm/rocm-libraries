#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/preflight-selftest.sh"
"$HERE/mutmut-verify-selftest.sh"
python3 -m pytest -q \
  --confcutdir="$HERE" \
  "$HERE/test_pyproject_mutmut.py"
