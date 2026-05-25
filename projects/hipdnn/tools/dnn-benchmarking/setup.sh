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
usage() {
    echo "Usage: $0 [--force-build] [--install-dir <path>] [-y]"
    echo ""
    echo "  --force-build        Force rebuild of hipDNN and the MIOpen provider,"
    echo "                           overwriting existing artifacts."
    echo "  --install-dir <path> Install prefix for hipDNN and the MIOpen provider."
    echo "                           Default: $INSTALL_DIR"
    echo "  -y                   Skip confirmation prompts."
    echo ""
    echo "  The installed plugin will be at:"
    echo "    <install-dir>/lib/hipdnn_plugins/engines/"
    echo "  Pass that path to --plugin-path when benchmarking."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-build) FORCE_BUILD=1 ;;
        --install-dir) shift; INSTALL_DIR="$1" ;;
        -y) AUTO_YES=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
    shift
done

HIPDNN_CONFIG="$INSTALL_DIR/lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"
if { [ "$FORCE_BUILD" -eq 1 ] || [ ! -f "$HIPDNN_CONFIG" ]; } && [ "$AUTO_YES" -eq 0 ]; then
    read -r -p "This will install hipDNN to $INSTALL_DIR. Continue? [Y/n] " confirm
    case "$confirm" in
        [nN]) echo "Aborted."; exit 0 ;;
    esac
fi

# 1. Create or activate venv
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment at $VENV_DIR..."
    rm -rf "$VENV_DIR"
fi
echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
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

# 2. Detect GPU architecture and install ROCm PyTorch from the matching nightly index.
detect_gpu_arch() {
    local arch
    if command -v rocm_agent_enumerator &>/dev/null; then
        arch=$(rocm_agent_enumerator | grep -m1 'gfx9')
    elif command -v rocminfo &>/dev/null; then
        arch=$(rocminfo | grep -oP 'gfx\d+' | head -1)
    fi
    echo "${arch:-}"
}

GPU_ARCH=$(detect_gpu_arch)
case "$GPU_ARCH" in
    gfx90*) INDEX_ARCH="gfx90X" ;;
    gfx94*) INDEX_ARCH="gfx94X" ;;
    *)
        echo "ERROR: Unsupported GPU architecture '${GPU_ARCH:-none}'."
        echo "Supported: gfx90a (MI200/MI210/MI250), gfx942 (MI300X/MI300A)"
        exit 1 ;;
esac

INDEX_URL="https://rocm.nightlies.amd.com/v2-staging/${INDEX_ARCH}-dcgpu/"
echo "Detected GPU: $GPU_ARCH → installing PyTorch from $INDEX_URL"

# Install ROCm torch first from its dedicated index. Then editable-install the
# package; pyproject.toml omits torch (so pip won't touch the already-installed
# ROCm build) and lists the rest (numpy, pytest, pytest-cov) which resolve
# cleanly from PyPI.
pip install --pre torch --index-url "$INDEX_URL"
pip install -e "$SCRIPT_DIR"

# 3. Build and install hipDNN + MIOpen
# The installed cmake configs use install-tree paths; pointing CMAKE_PREFIX_PATH at
# the raw build dir causes "non-existent path" errors in hipdnn_data_sdkConfig.cmake.
if [ "$FORCE_BUILD" -eq 1 ] || [ ! -f "$HIPDNN_CONFIG" ]; then
    echo "Building and installing hipDNN..."
    cmake -S "$HIPDNN_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DHIPDNN_SKIP_TESTS=ON
    cmake --build "$BUILD_DIR"
    cmake --install "$BUILD_DIR"

    if [ ! -d "$MIOPEN_PROVIDER_DIR" ]; then
        echo "Error: miopen-provider not found at $MIOPEN_PROVIDER_DIR"
        exit 1
    fi
    echo "Building and installing MIOpen provider..."
    rm -rf "$MIOPEN_BUILD_DIR"
    cmake -S "$MIOPEN_PROVIDER_DIR" -B "$MIOPEN_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
        -DMIOPENPROVIDER_SKIP_TESTS=ON
    cmake --build "$MIOPEN_BUILD_DIR"
    cmake --install "$MIOPEN_BUILD_DIR"
    echo ""
    echo "MIOpen plugin installed to: $INSTALL_DIR/lib/hipdnn_plugins/engines/"
fi

# 5. Install hipdnn Python bindings
# Wipe any stale cmake build cache (can reference deleted pip temp envs).
rm -rf "$HIPDNN_ROOT/python/build"
CMAKE_PREFIX_PATH="$INSTALL_DIR" \
    pip install -e "$HIPDNN_ROOT/python"

# 6. Patch the ROCm PyTorch wheel's bundled libhipdnn_backend.so
# The rocm_sdk wheel preloads its own copy of libhipdnn_backend.so with
# RTLD_GLOBAL before hipdnn_frontend can load the system copy. Replace
# the wheel's stale copy with the freshly built one so both torch and
# hipdnn_frontend use the same library.
WHEEL_BACKEND=$(find "$VENV_DIR" -path '*/_rocm_sdk_libraries_*/lib/libhipdnn_backend.so' 2>/dev/null | head -1)
if [ -n "$WHEEL_BACKEND" ] && [ -f "$INSTALL_DIR/lib/libhipdnn_backend.so" ]; then
    echo "Patching PyTorch wheel's bundled libhipdnn_backend.so..."
    cp "$INSTALL_DIR/lib/libhipdnn_backend.so" "$WHEEL_BACKEND"
fi

echo ""
echo "Setup complete. Activate the virtual environment with:"
echo "  source $VENV_DIR/bin/activate"
if [ "$FORCE_BUILD" -eq 1 ]; then
    echo ""
    echo "Run benchmarks with:"
    echo "  python -m dnn_benchmarking --graph <graph.json> \\"
    echo "    --plugin-path $INSTALL_DIR/lib/hipdnn_plugins/engines"
fi
