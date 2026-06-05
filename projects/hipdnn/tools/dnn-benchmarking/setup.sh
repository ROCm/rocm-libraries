#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPDNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$HIPDNN_ROOT/../.." && pwd)"

BUILD_DIR="$HIPDNN_ROOT/build"
DEFAULT_ROCM_PREFIX="/opt/rocm"
DNN_BENCH_WORKSPACE="${DNN_BENCH_WORKSPACE:-/workspace}"
VENV_DIR="$DNN_BENCH_WORKSPACE/.venv"
MIOPEN_PROVIDER_DIR="$WORKSPACE_ROOT/dnn-providers/miopen-provider"
MIOPEN_BUILD_DIR="$MIOPEN_PROVIDER_DIR/build"

FORCE_BUILD=0
AUTO_YES=0
REUSE_VENV=0
TORCH_MODE="${DNN_BENCH_TORCH_MODE:-rocm}"
ROCM_PREFIX="${DNN_BENCH_ROCM_PREFIX:-}"
GPU_ARCH_OVERRIDE="${DNN_BENCH_GPU_ARCH:-}"
TORCH_INDEX_URL="${DNN_BENCH_TORCH_INDEX_URL:-}"
RESOLVED_TORCH_INDEX_URL=""

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "  --torch-mode <rocm|cpu|existing|none>"
    echo "                       Select how torch is provided. Default: $TORCH_MODE"
    echo "                         rocm: install ROCm torch nightly, use hipDNN"
    echo "                               from the torch wheel's bundled ROCm SDK"
    echo "                               libraries, and install SDK devel only if"
    echo "                               needed to build the provider. Does not"
    echo "                               require a system ROCm install."
    echo "                         cpu:  install CPU-only torch and build bindings"
    echo "                               against installed ROCm/hipDNN."
    echo "                         existing:"
    echo "                               reuse torch already present in $VENV_DIR."
    echo "                               ROCm torch uses its bundled SDK libraries;"
    echo "                               CPU/non-ROCm torch uses installed ROCm/hipDNN."
    echo "                         none: leave torch uninstalled and build bindings"
    echo "                               against installed ROCm/hipDNN."
    echo "  --reuse-venv         Reuse an existing $VENV_DIR instead of deleting it."
    echo "  --workspace <path>  Workspace root for the venv, Python bytecode cache,"
    echo "                       and runtime benchmark caches. Default: $DNN_BENCH_WORKSPACE"
    echo "                       The virtual environment is <path>/.venv."
    echo "  --torch-index-url <url>"
    echo "                       Override the pip index URL used for torch."
    echo "  --gpu-arch <gfx*>    Override GPU architecture detection for ROCm torch"
    echo "                       nightly selection. Supported: gfx90a, gfx942, gfx950."
    echo "  --rocm-prefix <path> Explicit ROCm/hipDNN prefix for binding/provider"
    echo "                       builds. Takes precedence over venv discovery."
    echo "  --force-build        Build hipDNN and the MIOpen provider from source,"
    echo "                       overwriting artifacts under the selected ROCm prefix."
    echo "  -y                   Skip confirmation prompts."
    echo ""
    echo "  The installed plugin will be at:"
    echo "    <selected-prefix>/lib/hipdnn_plugins/engines/"
    echo "  Pass that path to --plugin-path when benchmarking."
}

require_arg() {
    local option="$1"
    local value="${2:-}"
    if [ -z "$value" ] || [[ "$value" == -* ]]; then
        echo "ERROR: $option requires a value." >&2
        usage
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-build) FORCE_BUILD=1 ;;
        --rocm-prefix)
            require_arg "$1" "${2:-}"
            shift
            ROCM_PREFIX="$1"
            ;;
        --torch-mode)
            require_arg "$1" "${2:-}"
            shift
            TORCH_MODE="$1"
            ;;
        --reuse-venv) REUSE_VENV=1 ;;
        --workspace)
            require_arg "$1" "${2:-}"
            shift
            DNN_BENCH_WORKSPACE="$1"
            VENV_DIR="$DNN_BENCH_WORKSPACE/.venv"
            ;;
        --torch-index-url)
            require_arg "$1" "${2:-}"
            shift
            TORCH_INDEX_URL="$1"
            ;;
        --gpu-arch)
            require_arg "$1" "${2:-}"
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
    rocm|cpu|existing|none) ;;
    *)
        echo "ERROR: --torch-mode must be one of: rocm, cpu, existing, none" >&2
        exit 1
        ;;
esac

if [ "$TORCH_MODE" = "existing" ]; then
    REUSE_VENV=1
fi
mkdir -p "$DNN_BENCH_WORKSPACE"
export DNN_BENCH_WORKSPACE


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

find_rocm_wheel_prefix() {
    local kind="$1"
    python - "$kind" <<'PY'
from pathlib import Path
import site
import sys

kind = sys.argv[1]

roots = []
try:
    roots.extend(Path(p) for p in site.getsitepackages())
except Exception:
    pass
try:
    roots.append(Path(site.getusersitepackages()))
except Exception:
    pass

matches = {}
for root in roots:
    if not root.is_dir():
        continue
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if kind == "libraries":
            if not child.name.startswith("_rocm_sdk_libraries_"):
                continue
            if not (
                child.joinpath("lib/cmake/hipdnn_frontend/hipdnn_frontendConfig.cmake").is_file()
                and child.joinpath("lib/cmake/hipdnn_backend/hipdnn_backendConfig.cmake").is_file()
            ):
                continue
        elif kind == "devel":
            if not child.name.startswith("_rocm_sdk_devel"):
                continue
            if not (
                child.joinpath("lib/llvm/bin/clang").is_file()
                and child.joinpath("lib/llvm/bin/clang++").is_file()
            ):
                continue
        else:
            print(f"ERROR: unknown ROCm wheel prefix kind: {kind}", file=sys.stderr)
            sys.exit(2)
        matches[child.resolve()] = child

if len(matches) == 1:
    print(next(iter(matches.values())))
    sys.exit(0)
if len(matches) > 1:
    print(f"ERROR: multiple usable ROCm SDK {kind} prefixes found:", file=sys.stderr)
    for path in sorted(matches.values()):
        print(f"  {path}", file=sys.stderr)
    print("Use a clean workspace/venv so setup cannot mix ROCm SDK packages.", file=sys.stderr)
    sys.exit(2)
sys.exit(1)
PY
}

discover_rocm_wheel_libraries_prefix() {
    find_rocm_wheel_prefix libraries
}

discover_rocm_wheel_devel_prefix() {
    find_rocm_wheel_prefix devel
}

require_rocm_wheel_libraries_prefix() {
    local prefix status
    if prefix=$(discover_rocm_wheel_libraries_prefix); then
        echo "$prefix"
        return
    fi
    status=$?
    if [ "$status" -ne 1 ]; then
        exit 1
    fi
    echo "ERROR: no hipDNN CMake configs found in venv ROCm SDK library wheels." >&2
    echo "Expected exactly one _rocm_sdk_libraries_* package with hipDNN frontend/backend configs." >&2
    echo "Use a ROCm torch wheel that includes hipDNN, pass --rocm-prefix explicitly, or pass --force-build." >&2
    exit 1
}

ensure_rocm_wheel_devel_prefix() {
    local index_url="$1"
    local prefix status
    if prefix=$(discover_rocm_wheel_devel_prefix); then
        echo "$prefix"
        return
    fi
    status=$?
    if [ "$status" -ne 1 ]; then
        exit 1
    fi
    if [ -z "$index_url" ]; then
        echo "ERROR: no ROCm SDK devel wheel found in this venv." >&2
        echo "Expected exactly one _rocm_sdk_devel package with lib/llvm/bin/clang++." >&2
        echo "Install rocm-sdk-devel from the same ROCm torch index, or pass --rocm-prefix." >&2
        exit 1
    fi

    echo "ROCm SDK devel wheel not found; installing rocm-sdk-devel from $index_url..." >&2
    pip install --pre rocm-sdk-devel --index-url "$index_url" >&2

    if prefix=$(discover_rocm_wheel_devel_prefix); then
        echo "$prefix"
        return
    fi
    status=$?
    if [ "$status" -ne 1 ]; then
        exit 1
    fi
    echo "ERROR: rocm-sdk-devel installed, but no usable _rocm_sdk_devel prefix was found." >&2
    echo "Expected lib/llvm/bin/clang and lib/llvm/bin/clang++ under _rocm_sdk_devel." >&2
    exit 1
}

detect_gpu_arch_from_kfd() {
    local props key value
    for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -r "$props" ] || continue
        while read -r key value _; do
            [ "$key" = "gfx_target_version" ] || continue
            case "${value:-0}" in
                90010) echo "gfx90a"; return 0 ;;
                90402) echo "gfx942"; return 0 ;;
                90500) echo "gfx950"; return 0 ;;
            esac
        done < "$props"
    done
    return 1
}

detect_gpu_arch() {
    local arch
    if [ -n "$GPU_ARCH_OVERRIDE" ]; then
        echo "$GPU_ARCH_OVERRIDE"
        return
    fi
    if command -v rocm_agent_enumerator &>/dev/null; then
        arch=$(rocm_agent_enumerator | grep -m1 'gfx9' || true)
        if [ -n "$arch" ]; then
            echo "$arch"
            return
        fi
    fi
    if command -v rocminfo &>/dev/null; then
        arch=$(rocminfo | grep -oP 'gfx\d+[a-z0-9]*' | head -1 || true)
        if [ -n "$arch" ]; then
            echo "$arch"
            return
        fi
    fi
    if arch=$(detect_gpu_arch_from_kfd); then
        echo "$arch"
        return
    fi
    echo ""
}

torch_is_importable() {
    python - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
PY
}

torch_is_rocm_wheel() {
    python - <<'PY' >/dev/null 2>&1
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if getattr(torch.version, "hip", None) else 1)
PY
}

resolve_installed_rocm_prefix() {
    if [ -n "$ROCM_PREFIX" ]; then
        echo "$ROCM_PREFIX"
        return
    fi
    if [ -n "${ROCM_PATH:-}" ]; then
        echo "$ROCM_PATH"
        return
    fi
    echo "$DEFAULT_ROCM_PREFIX"
}

install_torch() {
    case "$TORCH_MODE" in
        none)
            echo "Leaving torch uninstalled."
            return
            ;;
        existing)
            if ! torch_is_importable; then
                echo "ERROR: --torch-mode existing requires torch to already be installed in $VENV_DIR." >&2
                echo "Use --torch-mode rocm or --torch-mode cpu to install torch automatically." >&2
                exit 1
            fi
            echo "Using existing PyTorch in $VENV_DIR."
            return
            ;;
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
                    gfx90a) index_arch="gfx90X" ;;
                    gfx942) index_arch="gfx94X" ;;
                    gfx950) index_arch="gfx950" ;;
                    *)
                        echo "ERROR: Unsupported GPU architecture '${gpu_arch:-none}'." >&2
                        echo "Supported: gfx90a (MI200/MI210/MI250), gfx942 (MI300X/MI300A), gfx950 (MI350)" >&2
                        echo "Pass --gpu-arch or --torch-index-url to override detection." >&2
                        exit 1
                        ;;
                esac
                index_url="https://rocm.nightlies.amd.com/v2-staging/${index_arch}-dcgpu/"
                echo "Detected GPU: $gpu_arch"
            fi
            RESOLVED_TORCH_INDEX_URL="$index_url"
            echo "Installing ROCm PyTorch from $index_url"
            pip install --pre torch --index-url "$index_url"
            ;;
    esac
}

select_binding_prefix() {
    if [ "$FORCE_BUILD" -eq 1 ] || [ -n "$ROCM_PREFIX" ]; then
        resolve_installed_rocm_prefix
        return
    fi

    case "$TORCH_MODE" in
        rocm)
            require_rocm_wheel_libraries_prefix
            ;;
        existing)
            if torch_is_rocm_wheel; then
                require_rocm_wheel_libraries_prefix
                return
            fi
            resolve_installed_rocm_prefix
            ;;
        cpu|none)
            resolve_installed_rocm_prefix
            ;;
    esac
}

select_provider_toolchain_prefix() {
    if [ "$FORCE_BUILD" -eq 1 ] || [ -n "$ROCM_PREFIX" ]; then
        resolve_installed_rocm_prefix
        return
    fi

    case "$TORCH_MODE" in
        rocm)
            ensure_rocm_wheel_devel_prefix "$RESOLVED_TORCH_INDEX_URL"
            ;;
        existing)
            if torch_is_rocm_wheel; then
                ensure_rocm_wheel_devel_prefix ""
                return
            fi
            resolve_installed_rocm_prefix
            ;;
        cpu|none)
            resolve_installed_rocm_prefix
            ;;
    esac
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
    local prefix
    prefix=$(resolve_installed_rocm_prefix)
    echo "Building and installing hipDNN to $prefix..."
    cmake -S "$HIPDNN_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DHIPDNN_SKIP_TESTS=ON
    cmake --build "$BUILD_DIR"
    cmake --install "$BUILD_DIR"
}

build_miopen_provider() {
    local install_prefix="$1"
    local toolchain_prefix="$2"
    local cmake_prefix_path="$install_prefix"
    local cmake_program_path="$toolchain_prefix/bin;$toolchain_prefix/lib/llvm/bin"

    if [ "$toolchain_prefix" != "$install_prefix" ]; then
        cmake_prefix_path="$install_prefix;$toolchain_prefix"
    fi

    if [ ! -d "$MIOPEN_PROVIDER_DIR" ]; then
        echo "Error: miopen-provider not found at $MIOPEN_PROVIDER_DIR" >&2
        exit 1
    fi
    echo "Building and installing MIOpen provider to $install_prefix..."
    echo "Using ROCm compiler/devel prefix: $toolchain_prefix"
    rm -rf "$MIOPEN_BUILD_DIR"
    cmake -S "$MIOPEN_PROVIDER_DIR" -B "$MIOPEN_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$install_prefix" \
        -DCMAKE_PREFIX_PATH="$cmake_prefix_path" \
        -DCMAKE_PROGRAM_PATH="$cmake_program_path" \
        -DROCM_PATH="$toolchain_prefix" \
        -DMIOPENPROVIDER_SKIP_TESTS=ON
    cmake --build "$MIOPEN_BUILD_DIR"
    cmake --install "$MIOPEN_BUILD_DIR"
    echo ""
    echo "MIOpen plugin installed to: $install_prefix/lib/hipdnn_plugins/engines/"
}

FORCE_BUILD_PREFIX=$(resolve_installed_rocm_prefix)

if [ "$FORCE_BUILD" -eq 1 ] && [ "$AUTO_YES" -eq 0 ]; then
    read -r -p "This will build and install hipDNN to $FORCE_BUILD_PREFIX. Continue? [Y/n] " confirm
    case "$confirm" in
        [nN]) echo "Aborted."; exit 0 ;;
    esac
fi

# 1. Create or activate venv
if [ "$TORCH_MODE" = "existing" ] && [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: --torch-mode existing requires an existing virtual environment at $VENV_DIR." >&2
    echo "Use --torch-mode rocm or --torch-mode cpu to create one and install torch automatically." >&2
    exit 1
fi
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

echo "Torch mode: $TORCH_MODE"

# 2. Install torch, then editable-install the benchmark package. pyproject.toml
# intentionally omits torch so pip never replaces the selected torch wheel.
install_torch
pip install -e "$SCRIPT_DIR"

# 3. Select the hipDNN/ROCm prefix used by Python bindings and provider builds.
BINDING_PREFIX=$(select_binding_prefix)
echo "Using hipDNN/ROCm prefix: $BINDING_PREFIX"

if [ "$FORCE_BUILD" -eq 1 ]; then
    build_hipdnn
    BINDING_PREFIX=$(resolve_installed_rocm_prefix)
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
# contain the specific MIOpen provider plugin. This keeps the ROCm torch wheel
# flow self-contained: torch supplies ROCm + hipDNN, and setup.sh adds the local
# provider plugin when needed.
PLUGIN_DIR="$BINDING_PREFIX/lib/hipdnn_plugins/engines"
MIOPEN_PLUGIN="$PLUGIN_DIR/libmiopen_plugin.so"
if [ "$FORCE_BUILD" -eq 1 ] || [ ! -f "$MIOPEN_PLUGIN" ]; then
    PROVIDER_TOOLCHAIN_PREFIX=$(select_provider_toolchain_prefix)
    build_miopen_provider "$BINDING_PREFIX" "$PROVIDER_TOOLCHAIN_PREFIX"
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
