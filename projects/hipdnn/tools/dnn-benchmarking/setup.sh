#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPDNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$HIPDNN_ROOT/../.." && pwd)"
INSTALL_DIR="/opt/rocm"
VENV_DIR="$SCRIPT_DIR/.venv"


FORCE_BUILD=0
usage() {
    echo "Usage: $0 [--force-build] [--install-dir <path>]"
    echo ""
    echo "  --force-build        Force rebuild of hipDNN and the MIOpen provider,"
    echo "                           overwriting existing artifacts."
    echo "  --install-dir <path> Install prefix for hipDNN and the MIOpen provider."
    echo "                           Default: $INSTALL_DIR"
    echo ""
    echo "  The installed plugin will be at:"
    echo "    <install-dir>/lib/hipdnn_plugins/engines/"
    echo "  Pass that path to --plugin-path when benchmarking."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-build) FORCE_BUILD=1 ;;
        --install-dir) shift; INSTALL_DIR="$1" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
    shift
done

# 1. Create or activate venv
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
    echo 'export PYTHONPYCACHEPREFIX=/tmp/pycache' >> "$ACTIVATE_LOCAL"
fi
if ! grep -q "activate.local" "$VENV_DIR/bin/activate"; then
    # shellcheck disable=SC2016
    echo 'source "$(dirname "${BASH_SOURCE[0]}")/activate.local" 2>/dev/null || true' \
        >> "$VENV_DIR/bin/activate"
fi
export PYTHONPYCACHEPREFIX=/tmp/pycache

# 2. Install requirements and package.
# Install ROCm torch first from its dedicated index. Then editable-install the
# package; pyproject.toml omits torch (so pip won't touch the already-installed
# ROCm build) and lists the rest (numpy, pytest, pytest-cov) which resolve
# cleanly from PyPI.
pip install -r "$SCRIPT_DIR/requirements-rocm.txt"
pip install -e "$SCRIPT_DIR"

# 3. Build and install hipDNN + providers
# The installed cmake configs use install-tree paths; pointing CMAKE_PREFIX_PATH at
# the raw build dir causes "non-existent path" errors in hipdnn_data_sdkConfig.cmake.
HIPDNN_CONFIG="$INSTALL_DIR/lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake"
if [ "$FORCE_BUILD" -eq 1 ] || [ ! -f "$HIPDNN_CONFIG" ]; then
    echo "Building and installing hipDNN and providers..."
    cd "$WORKSPACE_ROOT"
    cmake --preset hipdnn-providers
    cmake --build build
    cmake --install build
    cd "$SCRIPT_DIR"
fi

# 5. Install hipdnn Python bindings
# Wipe any stale cmake build cache (can reference deleted pip temp envs).
rm -rf "$HIPDNN_ROOT/python/build"
CMAKE_PREFIX_PATH="$INSTALL_DIR" \
    pip install -e "$HIPDNN_ROOT/python"

echo ""
echo "Setup complete. Activate the virtual environment with:"
echo "  source $VENV_DIR/bin/activate"
if [ "$FORCE_BUILD" -eq 1 ]; then
    echo ""
    echo "Plugins installed to: $INSTALL_DIR/lib/hipdnn_plugins/engines/"
    echo "Run benchmarks with:"
    echo "  python -m dnn_benchmarking --graph <graph.json> \\"
    echo "    --plugin-path $INSTALL_DIR/lib/hipdnn_plugins/engines"
fi
