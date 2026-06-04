#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPDNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$HIPDNN_ROOT/../.." && pwd)"

BUILD_DIR="$HIPDNN_ROOT/build"
INSTALL_DIR="/opt/rocm"
DNN_BENCH_WORKSPACE="${DNN_BENCH_WORKSPACE:-/workspace}"
mkdir -p "$DNN_BENCH_WORKSPACE"
export DNN_BENCH_WORKSPACE
VENV_DIR="$DNN_BENCH_WORKSPACE/.venv"
MIOPEN_PROVIDER_DIR="$WORKSPACE_ROOT/dnn-providers/miopen-provider"
MIOPEN_BUILD_DIR="$MIOPEN_PROVIDER_DIR/build"

FORCE_BUILD=0
AUTO_YES=0
SKIP_TORCH_INSTALL=0
REUSE_VENV=0
TORCH_MODE="${DNN_BENCH_TORCH_MODE:-rocm}"
ROCM_PREFIX="${DNN_BENCH_ROCM_PREFIX:-}"
GPU_ARCH_OVERRIDE="${DNN_BENCH_GPU_ARCH:-}"
TORCH_INDEX_URL="${DNN_BENCH_TORCH_INDEX_URL:-}"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "  --torch-mode <rocm|cpu|none>"
    echo "                       Select the PyTorch install and hipDNN binding prefix"
    echo "                       discovery flow. Default: $TORCH_MODE"
    echo "                         rocm: install ROCm torch nightly and build bindings"
    echo "                               against hipDNN from the venv ROCm SDK wheels."
    echo "                         cpu:  install CPU-only torch and build bindings"
    echo "                               against --rocm-prefix or /opt/rocm."
    echo "                         none: leave torch untouched and only install this"
    echo "                               package plus hipDNN bindings."
    echo "  --skip-torch-install Do not install torch; use the venv's existing torch."
    echo "  --reuse-venv         Reuse an existing $VENV_DIR instead of deleting it."
    echo "  --torch-index-url <url>"
    echo "                       Override the pip index URL used for torch."
    echo "  --gpu-arch <gfx*>    Override GPU architecture detection for ROCm torch"
    echo "                       nightly selection."
    echo "  --rocm-prefix <path> Explicit ROCm/hipDNN prefix for binding/provider"
    echo "                       builds. Takes precedence over venv discovery."
    echo "  --install-dir <path> Legacy alias for --rocm-prefix; also the install"
    echo "                       prefix used by --force-build. Default: $INSTALL_DIR"
    echo "  --force-build        Build hipDNN and the MIOpen provider from source,"
    echo "                       overwriting artifacts under --install-dir."
    echo "  -y                   Skip confirmation prompts."
    echo ""
    echo "  The installed plugin will be at:"
    echo "    <selected-prefix>/lib/hipdnn_plugins/engines/"
    echo "  Pass that path to --plugin-path when benchmarking."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-build) FORCE_BUILD=1 ;;
        --install-dir)
            shift
            INSTALL_DIR="$1"
            ROCM_PREFIX="$1"
            ;;
        --rocm-prefix)
            shift
            ROCM_PREFIX="$1"
            ;;
        --torch-mode)
            shift
            TORCH_MODE="$1"
            ;;
        --skip-torch-install) SKIP_TORCH_INSTALL=1 ;;
        --reuse-venv) REUSE_VENV=1 ;;
        --torch-index-url)
            shift
            TORCH_INDEX_URL="$1"
            ;;
        --gpu-arch)
            shift
            GPU_ARCH_OVERRIDE="$1"
            ;;
        -y) AUTO_YES=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
    shift
done

case "$TORCH_MODE" in
    rocm|cpu|none) ;;
    *)
        echo "ERROR: --torch-mode must be one of: rocm, cpu, none" >&2
        exit 1
        ;;
esac

hipdnn_config_path() {
    local prefix="$1"
    echo "$prefix/lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"
}

hipdnn_backend_config_path() {
    local prefix="$1"
    echo "$prefix/lib/cmake/hipdnn_backend/hipdnn_backendConfig.cmake"
}

prefix_has_hipdnn() {
    local prefix="$1"
    [ -f "$(hipdnn_config_path "$prefix")" ] && [ -f "$(hipdnn_backend_config_path "$prefix")" ]
}

discover_rocm_wheel_prefix() {
    python - <<'PY'
from pathlib import Path
import site
import sys

roots = []
try:
    roots.extend(Path(p) for p in site.getsitepackages())
except Exception:
    pass
try:
    roots.append(Path(site.getusersitepackages()))
except Exception:
    pass

matches = []
for root in roots:
    if not root.is_dir():
        continue
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("_rocm_sdk_libraries_"):
            continue
        if (
            child.joinpath("lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake").is_file()
            and child.joinpath("lib/cmake/hipdnn_backend/hipdnn_backendConfig.cmake").is_file()
        ):
            matches.append(child)

if matches:
    print(sorted(matches)[0])
    sys.exit(0)
sys.exit(1)
PY
}

detect_gpu_arch() {
    local arch
    if [ -n "$GPU_ARCH_OVERRIDE" ]; then
        echo "$GPU_ARCH_OVERRIDE"
        return
    fi
    if command -v rocm_agent_enumerator &>/dev/null; then
        arch=$(rocm_agent_enumerator | grep -m1 'gfx9' || true)
    elif command -v rocminfo &>/dev/null; then
        arch=$(rocminfo | grep -oP 'gfx\d+' | head -1 || true)
    fi
    echo "${arch:-}"
}

install_torch() {
    if [ "$SKIP_TORCH_INSTALL" -eq 1 ] || [ "$TORCH_MODE" = "none" ]; then
        echo "Skipping torch install."
        return
    fi

    case "$TORCH_MODE" in
        cpu)
            local index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
            echo "Installing CPU-only PyTorch from $index_url"
            pip install torch --index-url "$index_url"
            ;;
        rocm)
            local index_url="$TORCH_INDEX_URL"
            if [ -z "$index_url" ]; then
                local gpu_arch index_arch
                gpu_arch=$(detect_gpu_arch)
                case "$gpu_arch" in
                    gfx90*) index_arch="gfx90X" ;;
                    gfx94*) index_arch="gfx94X" ;;
                    *)
                        echo "ERROR: Unsupported GPU architecture '${gpu_arch:-none}'." >&2
                        echo "Supported: gfx90a (MI200/MI210/MI250), gfx942 (MI300X/MI300A)" >&2
                        echo "Pass --gpu-arch or --torch-index-url to override detection." >&2
                        exit 1
                        ;;
                esac
                index_url="https://rocm.nightlies.amd.com/v2-staging/${index_arch}-dcgpu/"
                echo "Detected GPU: $gpu_arch"
            fi
            echo "Installing ROCm PyTorch from $index_url"
            pip install --pre torch --index-url "$index_url"
            ;;
    esac
}

select_binding_prefix() {
    if [ -n "$ROCM_PREFIX" ]; then
        echo "$ROCM_PREFIX"
        return
    fi

    if [ "$TORCH_MODE" = "rocm" ]; then
        local wheel_prefix
        if wheel_prefix=$(discover_rocm_wheel_prefix); then
            echo "$wheel_prefix"
            return
        fi
        if [ "$FORCE_BUILD" -eq 1 ]; then
            echo "$INSTALL_DIR"
            return
        fi
        echo "ERROR: no hipDNN CMake configs found in venv ROCm SDK wheels." >&2
        echo "Use a ROCm torch wheel that includes hipDNN, pass --rocm-prefix explicitly, or pass --force-build." >&2
        exit 1
    fi

    echo "$INSTALL_DIR"
}

maybe_install_amdsmi() {
    local prefix="$1"
    local amdsmi_dir="$prefix/share/amd_smi"
    if ! python -c "import amdsmi" >/dev/null 2>&1; then
        if [ -f "$amdsmi_dir/setup.py" ] || [ -f "$amdsmi_dir/pyproject.toml" ]; then
            echo "Installing amdsmi Python bindings from $amdsmi_dir..."
            if ! pip install "$amdsmi_dir"; then
                echo "Warning: amdsmi install failed; GPU SMI snapshot will be disabled." >&2
            fi
        else
            echo "Warning: amdsmi not found at $amdsmi_dir; GPU SMI snapshot will be disabled." >&2
        fi
    fi
}

build_hipdnn() {
    echo "Building and installing hipDNN to $INSTALL_DIR..."
    cmake -S "$HIPDNN_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DHIPDNN_SKIP_TESTS=ON
    cmake --build "$BUILD_DIR"
    cmake --install "$BUILD_DIR"
}

build_miopen_provider() {
    local prefix="$1"
    if [ ! -d "$MIOPEN_PROVIDER_DIR" ]; then
        echo "Error: miopen-provider not found at $MIOPEN_PROVIDER_DIR" >&2
        exit 1
    fi
    echo "Building and installing MIOpen provider to $prefix..."
    rm -rf "$MIOPEN_BUILD_DIR"
    cmake -S "$MIOPEN_PROVIDER_DIR" -B "$MIOPEN_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DCMAKE_PREFIX_PATH="$prefix" \
        -DMIOPENPROVIDER_SKIP_TESTS=ON
    cmake --build "$MIOPEN_BUILD_DIR"
    cmake --install "$MIOPEN_BUILD_DIR"
    echo ""
    echo "MIOpen plugin installed to: $prefix/lib/hipdnn_plugins/engines/"
}

if [ "$FORCE_BUILD" -eq 1 ] && [ "$AUTO_YES" -eq 0 ]; then
    read -r -p "This will build and install hipDNN to $INSTALL_DIR. Continue? [Y/n] " confirm
    case "$confirm" in
        [nN]) echo "Aborted."; exit 0 ;;
    esac
fi

# 1. Create or activate venv
if [ -d "$VENV_DIR" ]; then
    if [ "$REUSE_VENV" -eq 1 ]; then
        echo "Reusing existing virtual environment at $VENV_DIR..."
    else
        echo "Removing existing virtual environment at $VENV_DIR..."
        rm -rf "$VENV_DIR"
    fi
fi
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Redirect Python's bytecode cache away from the network home directory.
# The source tree lives on a network filesystem; without this, every import
# writes/reads .pyc files over the network. Must be injected into the venv
# activate script so it's set before the interpreter starts (setting it in
# Python code is too late for that process's own imports).
ACTIVATE_LOCAL="$VENV_DIR/bin/activate.local"
if [ ! -f "$ACTIVATE_LOCAL" ] || ! grep -q PYTHONPYCACHEPREFIX "$ACTIVATE_LOCAL"; then
    {
        echo "export PYTHONPYCACHEPREFIX=$DNN_BENCH_WORKSPACE/pycache"
        echo "export DNN_BENCH_WORKSPACE=$DNN_BENCH_WORKSPACE"
    } >> "$ACTIVATE_LOCAL"
fi
if ! grep -q "activate.local" "$VENV_DIR/bin/activate"; then
    # shellcheck disable=SC2016
    echo 'source "$(dirname "${BASH_SOURCE[0]}")/activate.local" 2>/dev/null || true' \
        >> "$VENV_DIR/bin/activate"
fi
export PYTHONPYCACHEPREFIX="$DNN_BENCH_WORKSPACE/pycache"

# 2. Install torch, then editable-install the benchmark package. pyproject.toml
# intentionally omits torch so pip never replaces the selected torch wheel.
install_torch
pip install -e "$SCRIPT_DIR"

# 3. Select the hipDNN/ROCm prefix used by Python bindings and provider builds.
BINDING_PREFIX=$(select_binding_prefix)
echo "Using hipDNN/ROCm prefix: $BINDING_PREFIX"

if [ "$FORCE_BUILD" -eq 1 ]; then
    build_hipdnn
    BINDING_PREFIX="$INSTALL_DIR"
elif ! prefix_has_hipdnn "$BINDING_PREFIX"; then
    echo "ERROR: hipDNN CMake configs were not found under $BINDING_PREFIX." >&2
    echo "Expected:" >&2
    echo "  $(hipdnn_config_path "$BINDING_PREFIX")" >&2
    echo "  $(hipdnn_backend_config_path "$BINDING_PREFIX")" >&2
    echo "Install ROCm/hipDNN artifacts there, use --rocm-prefix, or pass --force-build." >&2
    exit 1
fi

# 4. Install amdsmi Python bindings if present in the selected ROCm install.
# amdsmi is not on PyPI — it ships under <prefix>/share/amd_smi/. The always-on
# GPU snapshot in metrics/gpu_smi.py degrades gracefully if amdsmi is absent.
maybe_install_amdsmi "$BINDING_PREFIX"

# 5. Build/install the MIOpen provider if the selected prefix does not already
# contain a provider shared library. This keeps the ROCm torch wheel flow
# self-contained: torch supplies ROCm + hipDNN, and setup.sh adds the local
# provider plugin.
PLUGIN_DIR="$BINDING_PREFIX/lib/hipdnn_plugins/engines"
if [ "$FORCE_BUILD" -eq 1 ] || ! compgen -G "$PLUGIN_DIR/*.so" >/dev/null; then
    build_miopen_provider "$BINDING_PREFIX"
fi

# 6. Install hipDNN Python bindings.
# Wipe any stale cmake build cache (can reference deleted pip temp envs).
rm -rf "$HIPDNN_ROOT/python/build"
CMAKE_PREFIX_PATH="$BINDING_PREFIX" \
    pip install -e "$HIPDNN_ROOT/python"

# 7. Patch the ROCm PyTorch wheel's bundled libhipdnn_backend.so when the user
# explicitly rebuilt hipDNN elsewhere. In normal ROCm torch mode BINDING_PREFIX
# is the wheel package itself, so source and destination are identical/no-op.
WHEEL_BACKEND=$(find "$VENV_DIR" -path '*/_rocm_sdk_libraries_*/lib/libhipdnn_backend.so' 2>/dev/null | head -1)
SOURCE_BACKEND="$BINDING_PREFIX/lib/libhipdnn_backend.so"
if [ -n "$WHEEL_BACKEND" ] && [ -f "$SOURCE_BACKEND" ] && [ "$WHEEL_BACKEND" != "$SOURCE_BACKEND" ]; then
    echo "Patching PyTorch wheel's bundled libhipdnn_backend.so..."
    cp "$SOURCE_BACKEND" "$WHEEL_BACKEND"
fi

echo ""
echo "Setup complete. Activate the virtual environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run benchmarks with:"
echo "  python -m dnn_benchmarking --graph <graph.json> \\"
echo "    --plugin-path $BINDING_PREFIX/lib/hipdnn_plugins/engines"
