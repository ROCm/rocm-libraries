#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPDNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$HIPDNN_ROOT/../.." && pwd)"
BUILD_DIR="$HIPDNN_ROOT/build"
INSTALL_DIR="$BUILD_DIR/install"
VENV_DIR="$SCRIPT_DIR/.venv"
MIOPEN_PROVIDER_DIR="$WORKSPACE_ROOT/dnn-providers/miopen-provider"
MIOPEN_BUILD_DIR="$MIOPEN_PROVIDER_DIR/build"

FORCE_BUILD=0
usage() {
    echo "Usage: $0 [--force-build]"
    echo ""
    echo "  --force-build   Force rebuild of hipDNN and the MIOpen provider,"
    echo "                      overwriting existing artifacts. The installed plugin"
    echo "                      will be at:"
    echo "                        $INSTALL_DIR/lib/hipdnn_plugins/engines/"
    echo "                      Pass that path to --plugin-path when benchmarking."
}

for arg in "$@"; do
    case "$arg" in
        --force-build) FORCE_BUILD=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $arg"; usage; exit 1 ;;
    esac
done

# 1. Create or activate venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 2. Install requirements and package
# Install ROCm torch first via its dedicated index, then install the package with
# --no-deps so pip never resolves torch from PyPI (which would pull the CUDA build).
# Dev extras (pytest, pytest-cov) are installed separately as they have no GPU deps.
pip install -r "$SCRIPT_DIR/requirements-rocm.txt"
pip install -e "$SCRIPT_DIR" --no-deps

# 3. Build and install hipdnn
# The installed cmake configs use install-tree paths; pointing CMAKE_PREFIX_PATH at
# the raw build dir causes "non-existent path" errors in hipdnn_data_sdkConfig.cmake.
HIPDNN_CONFIG="$INSTALL_DIR/lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"
if [ "$FORCE_BUILD" -eq 1 ] || [ ! -f "$HIPDNN_CONFIG" ]; then
    echo "Building and installing hipDNN..."
    cmake -G Ninja -S "$HIPDNN_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DHIPDNN_SKIP_TESTS=ON
    ninja -C "$BUILD_DIR"
    cmake --install "$BUILD_DIR"
fi

# 4. Build and install MIOpen provider (only with --force-build)
if [ "$FORCE_BUILD" -eq 1 ]; then
    if [ ! -d "$MIOPEN_PROVIDER_DIR" ]; then
        echo "Error: miopen-provider not found at $MIOPEN_PROVIDER_DIR"
        exit 1
    fi
    echo "Building and installing MIOpen provider..."
    rm -rf "$MIOPEN_BUILD_DIR"
    cmake -G Ninja -S "$MIOPEN_PROVIDER_DIR" -B "$MIOPEN_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_PREFIX_PATH="$INSTALL_DIR"
    ninja -C "$MIOPEN_BUILD_DIR"
    cmake --install "$MIOPEN_BUILD_DIR"
    echo ""
    echo "MIOpen plugin installed to: $INSTALL_DIR/lib/hipdnn_plugins/engines/"
fi

# 5. Install hipdnn Python bindings
# Wipe any stale cmake build cache (can reference deleted pip temp envs).
rm -rf "$HIPDNN_ROOT/python/build"
CMAKE_PREFIX_PATH="$INSTALL_DIR" \
    pip install -e "$HIPDNN_ROOT/python"

echo ""
echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
if [ "$FORCE_BUILD" -eq 1 ]; then
    echo ""
    echo "Run benchmarks with:"
    echo "  python -m dnn_benchmarking --graph <graph.json> \\"
    echo "    --plugin-path $INSTALL_DIR/lib/hipdnn_plugins/engines"
fi
