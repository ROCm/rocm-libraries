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
HIP_KERNEL_PROVIDER_DIR="$WORKSPACE_ROOT/dnn-providers/hip-kernel-provider"
HIP_KERNEL_BUILD_DIR="$HIP_KERNEL_PROVIDER_DIR/build"
HIPBLASLT_PROVIDER_DIR="$WORKSPACE_ROOT/dnn-providers/hipblaslt-provider"
HIPBLASLT_BUILD_DIR="$HIPBLASLT_PROVIDER_DIR/build"

FORCE_BUILD=0
AUTO_YES=0
usage() {
    echo "Usage: $0 [--force-build] [--install-dir <path>] [-y]"
    echo ""
    echo "  --force-build        Force rebuild of hipDNN and provider plugins,"
    echo "                           overwriting existing artifacts."
    echo "  --install-dir <path> Install prefix for hipDNN and provider plugins."
    echo "                           Default: $INSTALL_DIR"
    echo "  -y                   Skip confirmation prompts."
    echo ""
    echo "  The installed plugins will be at:"
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
PLUGIN_DIR="$INSTALL_DIR/lib/hipdnn_plugins/engines"
MIOPEN_PLUGIN="$PLUGIN_DIR/libmiopen_plugin.so"
HIP_KERNEL_PLUGIN="$PLUGIN_DIR/libhip_kernel_provider.so"
HIPBLASLT_PLUGIN="$PLUGIN_DIR/libhipblaslt_plugin.so"

needs_install() {
    [ "$FORCE_BUILD" -eq 1 ] || [ ! -f "$1" ]
}

if { needs_install "$HIPDNN_CONFIG" || needs_install "$MIOPEN_PLUGIN" \
    || needs_install "$HIP_KERNEL_PLUGIN" || needs_install "$HIPBLASLT_PLUGIN"; } \
    && [ "$AUTO_YES" -eq 0 ]; then
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

# 2b. Install amdsmi Python bindings if present in the ROCm install.
# amdsmi is not on PyPI — it ships under /opt/rocm/share/amd_smi/. The
# always-on GPU snapshot in metrics/gpu_smi.py uses it; if absent the
# snapshot fields stay None (warn-once), so this install is best-effort.
AMDSMI_DIR="$INSTALL_DIR/share/amd_smi"
if ! python -c "import amdsmi" >/dev/null 2>&1; then
    if [ -f "$AMDSMI_DIR/setup.py" ] || [ -f "$AMDSMI_DIR/pyproject.toml" ]; then
        echo "Installing amdsmi Python bindings from $AMDSMI_DIR..."
        if ! pip install "$AMDSMI_DIR"; then
            echo "Warning: amdsmi install failed; GPU SMI snapshot will be disabled." >&2
        fi
    else
        echo "Warning: amdsmi not found at $AMDSMI_DIR; GPU SMI snapshot will be disabled." >&2
    fi
fi

# 3. Build and install hipDNN + provider plugins
# The installed cmake configs use install-tree paths; pointing CMAKE_PREFIX_PATH at
# the raw build dir causes "non-existent path" errors in hipdnn_data_sdkConfig.cmake.
require_provider_dir() {
    local name="$1"
    local dir="$2"
    if [ ! -d "$dir" ]; then
        echo "Error: $name not found at $dir"
        exit 1
    fi
}

BUILT_HIPDNN=0
if needs_install "$HIPDNN_CONFIG"; then
    echo "Building and installing hipDNN..."
    cmake -S "$HIPDNN_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DHIPDNN_SKIP_TESTS=ON \
        -DHIPDNN_ENABLE_SDPA=ON \
        -DENABLE_CLANG_FORMAT=OFF \
        -DENABLE_CLANG_TIDY=OFF
    cmake --build "$BUILD_DIR"
    cmake --install "$BUILD_DIR"
    BUILT_HIPDNN=1
fi

if [ "$BUILT_HIPDNN" -eq 1 ] || needs_install "$MIOPEN_PLUGIN"; then
    require_provider_dir "miopen-provider" "$MIOPEN_PROVIDER_DIR"
    echo "Building and installing MIOpen provider..."
    rm -rf "$MIOPEN_BUILD_DIR"
    cmake -S "$MIOPEN_PROVIDER_DIR" -B "$MIOPEN_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
        -DMIOPENPROVIDER_SKIP_TESTS=ON \
        -DENABLE_CLANG_FORMAT=OFF \
        -DENABLE_CLANG_TIDY=OFF
    cmake --build "$MIOPEN_BUILD_DIR"
    cmake --install "$MIOPEN_BUILD_DIR"
fi

if [ "$BUILT_HIPDNN" -eq 1 ] || needs_install "$HIPBLASLT_PLUGIN"; then
    require_provider_dir "hipblaslt-provider" "$HIPBLASLT_PROVIDER_DIR"
    echo "Building and installing hipBLASLt provider..."
    rm -rf "$HIPBLASLT_BUILD_DIR"
    cmake -S "$HIPBLASLT_PROVIDER_DIR" -B "$HIPBLASLT_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
        -DHIPDNN_SKIP_TESTS=ON \
        -DENABLE_CLANG_FORMAT=OFF \
        -DENABLE_CLANG_TIDY=OFF
    cmake --build "$HIPBLASLT_BUILD_DIR"
    cmake --install "$HIPBLASLT_BUILD_DIR"
fi

if [ "$BUILT_HIPDNN" -eq 1 ] || needs_install "$HIP_KERNEL_PLUGIN"; then
    require_provider_dir "hip-kernel-provider" "$HIP_KERNEL_PROVIDER_DIR"
    echo "Building and installing hip-kernel-provider..."
    rm -rf "$HIP_KERNEL_BUILD_DIR"
    cmake -S "$HIP_KERNEL_PROVIDER_DIR" -B "$HIP_KERNEL_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
        -DHIPKERNELPROVIDER_ENABLE_TESTS=OFF \
        -DENABLE_CLANG_FORMAT=OFF \
        -DENABLE_CLANG_TIDY=OFF \
        -DENABLE_ASM_SDPA_ENGINE=ON
    cmake --build "$HIP_KERNEL_BUILD_DIR"
    cmake --install "$HIP_KERNEL_BUILD_DIR"
fi

echo ""
echo "hipDNN plugins installed to: $PLUGIN_DIR/"

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
