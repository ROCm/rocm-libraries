#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPDNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$HIPDNN_ROOT/build"
VENV_DIR="$SCRIPT_DIR/.venv"

# 1. Create or activate venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 2. Install requirements
pip install -r "$SCRIPT_DIR/requirements-rocm.txt"
pip install -e "$SCRIPT_DIR[dev]"

# 3. Build hipdnn if needed
if [ ! -f "$BUILD_DIR/lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake" ]; then
    echo "Building hipdnn..."
    cmake -G Ninja -S "$HIPDNN_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHIPDNN_SKIP_TESTS=ON
    ninja -C "$BUILD_DIR"
fi

# 4. Install hipdnn Python bindings
CMAKE_PREFIX_PATH="$BUILD_DIR/lib/cmake" \
    pip install -e "$HIPDNN_ROOT/python"

echo ""
echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
