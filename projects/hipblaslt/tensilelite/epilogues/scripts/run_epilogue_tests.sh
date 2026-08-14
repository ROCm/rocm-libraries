#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
set -euo pipefail

export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}

# Build client
invoke build-client --build-dir build_tmp --gpu-targets gfx950

# GPU epilogue tests — must collect from Tensile/Tests/common/ so that
# conftest.py's pytest_generate_tests hook parametrizes test_config.py.
pytest -v \
    --prebuilt-client build_tmp/tensilelite/client/tensilelite-client \
    --gpu-targets gfx950 \
    Tensile/Tests/common/ \
    -k "epilogues_k1" \
    "$@"
