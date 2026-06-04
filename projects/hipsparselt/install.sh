#!/usr/bin/env bash

# ########################################################################
# Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# CTION WITH THE SOFTWARE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

# Compatibility wrapper — delegates to 'invoke build' (tasks.py), which runs
# install-host.sh with PATH adjusted so the venv Python (with PyYAML) is used
# by CMake for client test generation.
#
# Prefer: invoke build --install-deps --clients ...
# Legacy implementation: ./install-host.sh ...

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
VENV_DIR="${BUILD_DIR}/venv"

# ---------------------------------------------------------------------------
# Python virtual environment (Ubuntu and minimal CI images often lack PyYAML
# on the system interpreter CMake would otherwise pick.)
# ---------------------------------------------------------------------------
setup_venv() {
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "${VENV_DIR}/bin/activate"
        return
    fi

    echo "Creating Python virtual environment in ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip wheel
    pip install -r "${SCRIPT_DIR}/requirements-build.txt"
}

setup_venv

# Expand bundled short flags (e.g. -cdi → -c -d -i) so legacy CI/Docker and
# install-host-style invocations match the one-flag-at-a-time parser below.
# Only letters that are standalone short options in this wrapper (no bundled
# value); flags that take values (-a, -l, …) must remain separate tokens.
declare -a _hipsparselt_norm=()
while (($# > 0)); do
    _a="$1"
    if [[ "$_a" =~ ^-([cdigrknh]+)$ ]] && (( ${#BASH_REMATCH[1]} >= 2 )); then
        _bundle="${BASH_REMATCH[1]}"
        for ((_i = 0; _i < ${#_bundle}; _i++)); do
            _hipsparselt_norm+=("-${_bundle:_i:1}")
        done
    else
        _hipsparselt_norm+=("$_a")
    fi
    shift
done
set -- "${_hipsparselt_norm[@]}"
unset _a _bundle _i _hipsparselt_norm

INVOKE_ARGS=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "hipSPARSELt build helper (compatibility wrapper for 'invoke build')"
            echo ""
            echo "Usage: ./install.sh [options]"
            echo ""
            echo "  -h, --help                 Show this help"
            echo "  Combined short flags such as -cdi are expanded to -c -d -i (legacy Docker/CI)."
            echo "  -i, --install              Install after build (package + distro install)"
            echo "  -d, --dependencies         Install build dependencies"
            echo "  -c, --clients              Build library clients / tests / benchmarks"
            echo "  -r, --relocatable          Relocatable ROCm package"
            echo "  -g, --debug                Debug build"
            echo "  -k, --relwithdebinfo       RelWithDebInfo build"
            echo "  --cuda|--use-cuda          CUDA backend"
            echo "  --static                   Static library"
            echo "  --address-sanitizer        AddressSanitizer"
            echo "  --codecoverage             Code coverage"
            echo "  -a, --architecture <arch>  GPU target(s) for GPU_TARGETS"
            echo "  --cpu_ref_lib <lib>        blis or lapack"
            echo "  --prefix <dir>             Install prefix directory name (legacy)"
            echo "  -l, --logic <path>         TensileLite logic path"
            echo "  -o, --cov <n>              Tensile coverage"
            echo "  -f, --fork <name>          Tensile fork"
            echo "  -b, --branch <tag>         Tensile tag/branch"
            echo "  -t, --test_local_path <p>  Tensile local test path"
            echo "  -n, --no_tensile|--no-tensile  Legacy no-tensile (may fail)"
            echo "  --merge-files              Tensile merge files"
            echo "  -no-merge-files            Disable Tensile merge files"
            echo "  -u, --use-custom-version <v>  Custom Tensile version"
            echo "  --msgpack                  Tensile msgpack backend (default)"
            echo "  --no-msgpack               Tensile YAML backend"
            echo "  --build_dir <dir>          Build directory name"
            echo "  --keep-build-tmp           Keep Tensile build tmp"
            echo "  --disable-hipsparselt-marker"
            echo "  --enable-tensile-marker"
            echo ""
            echo "Prefer 'invoke build' directly. See: invoke --help build"
            exit 0
            ;;
        -i|--install)          INVOKE_ARGS="$INVOKE_ARGS --install-pkg" ;;
        -d|--dependencies)     INVOKE_ARGS="$INVOKE_ARGS --install-deps" ;;
        -c|--clients)          INVOKE_ARGS="$INVOKE_ARGS --clients" ;;
        -r|--relocatable)      INVOKE_ARGS="$INVOKE_ARGS --relocatable" ;;
        -g|--debug)            INVOKE_ARGS="$INVOKE_ARGS --debug" ;;
        -k|--relwithdebinfo)   INVOKE_ARGS="$INVOKE_ARGS --relwithdebinfo" ;;
        --cuda|--use-cuda)     INVOKE_ARGS="$INVOKE_ARGS --cuda" ;;
        --static)              INVOKE_ARGS="$INVOKE_ARGS --static" ;;
        --address-sanitizer)   INVOKE_ARGS="$INVOKE_ARGS --address-sanitizer" ;;
        --codecoverage)        INVOKE_ARGS="$INVOKE_ARGS --codecoverage" ;;
        -a|--architecture)     shift; INVOKE_ARGS="$INVOKE_ARGS --architecture='$1'" ;;
        --cpu_ref_lib)         shift; INVOKE_ARGS="$INVOKE_ARGS --cpu-ref-lib='$1'" ;;
        --prefix)              shift; INVOKE_ARGS="$INVOKE_ARGS --install-prefix='$1'" ;;
        -l|--logic)            shift; INVOKE_ARGS="$INVOKE_ARGS --tensile-logic='$1'" ;;
        -o|--cov)              shift; INVOKE_ARGS="$INVOKE_ARGS --tensile-cov='$1'" ;;
        -f|--fork)             shift; INVOKE_ARGS="$INVOKE_ARGS --tensile-fork='$1'" ;;
        -b|--branch)           shift; INVOKE_ARGS="$INVOKE_ARGS --tensile-tag='$1'" ;;
        -t|--test_local_path)  shift; INVOKE_ARGS="$INVOKE_ARGS --tensile-test-local-path='$1'" ;;
        -n|--no_tensile|--no-tensile) INVOKE_ARGS="$INVOKE_ARGS --no-tensile" ;;
        --merge-files)         INVOKE_ARGS="$INVOKE_ARGS --merge-files" ;;
        -no-merge-files)       INVOKE_ARGS="$INVOKE_ARGS --no-merge-files" ;;
        -u|--use-custom-version) shift; INVOKE_ARGS="$INVOKE_ARGS --tensile-version='$1'" ;;
        --msgpack)             INVOKE_ARGS="$INVOKE_ARGS --msgpack" ;;
        --no-msgpack)          INVOKE_ARGS="$INVOKE_ARGS --no-msgpack" ;;
        --build_dir)           shift; INVOKE_ARGS="$INVOKE_ARGS --build-dir='$1'" ;;
        --keep-build-tmp)      INVOKE_ARGS="$INVOKE_ARGS --keep-build-tmp" ;;
        --disable-hipsparselt-marker) INVOKE_ARGS="$INVOKE_ARGS --disable-hipsparselt-marker" ;;
        --enable-tensile-marker) INVOKE_ARGS="$INVOKE_ARGS --enable-tensile-marker" ;;
        *)
            echo "Warning: unknown flag '$1' — passing through to invoke build"
            EXTRA_ARGS="$EXTRA_ARGS $1"
            ;;
    esac
    shift
done

echo "NOTE: ./install.sh is a thin wrapper. Prefer: invoke build${INVOKE_ARGS}"
cd "$SCRIPT_DIR"
# shellcheck disable=SC2086
eval invoke build ${INVOKE_ARGS} ${EXTRA_ARGS}
