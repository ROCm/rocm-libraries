#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# This file is invoked explicitly; it is not part of the normal test suite.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/preflight-selftest.sh"
"$HERE/mutmut-verify-selftest.sh"
python3 -m pytest -q \
  --confcutdir="$HERE" \
  "$HERE/pyproject_mutmut_selftest.py"
