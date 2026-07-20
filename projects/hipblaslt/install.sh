#!/usr/bin/env bash

# ########################################################################
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

# Compatibility wrapper — delegates to 'invoke build' (tasks.py).
# This script will be removed in a future release. Use 'invoke build' directly.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
VENV_DIR="${BUILD_DIR}/venv"

# ---------------------------------------------------------------------------
# Set up Python virtual environment (required on Ubuntu which blocks global pip)
# ---------------------------------------------------------------------------
setup_venv() {
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        source "${VENV_DIR}/bin/activate"
        return
    fi

    echo "Creating Python virtual environment in ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip wheel invoke
    if [[ -f "${SCRIPT_DIR}/tensilelite/requirements.txt" ]]; then
        pip install -r "${SCRIPT_DIR}/tensilelite/requirements.txt"
    fi
}

# Activate or create venv
setup_venv

# Map install.sh flags to invoke arguments
INVOKE_ARGS=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "hipBLASLt build helper (compatibility wrapper for 'invoke build')"
            echo ""
            echo "Usage: ./install.sh [options]"
            echo ""
            echo "  -h, --help                Show this help"
            echo "  -i, --install             Install after build"
            echo "  -d, --dependencies        Install build dependencies"
            echo "  -c, --clients             Build library clients"
            echo "  -g, --debug               Debug build"
            echo "  -k, --relwithdebinfo      RelWithDebInfo build"
            echo "  -a, --architecture <arch>  GPU target(s)"
            echo "  -n, --client-only         Build without Tensile"
            echo "  -j <N>                    Parallel jobs (caps both cmake and Tensile kernel generation)"
            echo "  -l, --logic <path>        Tensile logic path"
            echo "  --static                  Static library"
            echo "  --address-sanitizer       AddressSanitizer"
            echo "  --codecoverage            Code coverage"
            echo "  --gprof                   GNU gprof profiling"
            echo "  --skip_rocroller          Skip rocRoller"
            echo "  --keep-build-tmp          Keep build artifacts"
            echo "  --no-compress             Don't compress asm objects"
            echo "  --logic-yaml-filter <f>   Logic filter"
            echo ""
            echo "Prefer using 'invoke build' directly. Run 'invoke --help build' for all options."
            exit 0
            ;;
        -i|--install)       INVOKE_ARGS="$INVOKE_ARGS --install-pkg" ;;
        -d|--dependencies)  INVOKE_ARGS="$INVOKE_ARGS --install-deps" ;;
        -c|--clients)       INVOKE_ARGS="$INVOKE_ARGS --clients" ;;
        -g|--debug)         INVOKE_ARGS="$INVOKE_ARGS --debug" ;;
        -k|--relwithdebinfo) INVOKE_ARGS="$INVOKE_ARGS --relwithdebinfo" ;;
        -a|--architecture)  shift; INVOKE_ARGS="$INVOKE_ARGS --architecture='$1'" ;;
        -n|--client-only)   INVOKE_ARGS="$INVOKE_ARGS --no-tensile" ;;
        -j)                 shift; INVOKE_ARGS="$INVOKE_ARGS --jobs $1 --tensile-threads $1" ;;
        -l|--logic)         shift; INVOKE_ARGS="$INVOKE_ARGS --tensile-logic=$1" ;;
        --static)           INVOKE_ARGS="$INVOKE_ARGS --static" ;;
        --address-sanitizer) INVOKE_ARGS="$INVOKE_ARGS --address-sanitizer" ;;
        --codecoverage)     INVOKE_ARGS="$INVOKE_ARGS --codecoverage" ;;
        --gprof)            INVOKE_ARGS="$INVOKE_ARGS --gprof" ;;
        --skip_rocroller)   INVOKE_ARGS="$INVOKE_ARGS --skip-rocroller" ;;
        --keep-build-tmp)   INVOKE_ARGS="$INVOKE_ARGS --keep-build-tmp" ;;
        --no-compress)      INVOKE_ARGS="$INVOKE_ARGS --no-compress" ;;
        --logic-yaml-filter) shift; INVOKE_ARGS="$INVOKE_ARGS --logic-filter=$1" ;;
        --use-system-packages) INVOKE_ARGS="$INVOKE_ARGS --use-system-packages" ;;
        --tensile-verbose)  shift; INVOKE_ARGS="$INVOKE_ARGS --tensile-verbose=$1" ;;
        *)
            echo "Warning: unknown flag '$1' — passing through"
            EXTRA_ARGS="$EXTRA_ARGS $1"
            ;;
    esac
    shift
done

echo "NOTE: install.sh is deprecated. Use 'invoke build$INVOKE_ARGS' directly."
cd "$SCRIPT_DIR"
eval invoke build $INVOKE_ARGS $EXTRA_ARGS
