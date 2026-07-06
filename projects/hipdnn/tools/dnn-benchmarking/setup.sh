#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPDNN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$HIPDNN_ROOT/../.." && pwd)"

DEFAULT_DNN_BENCH_WORKSPACE="$SCRIPT_DIR/.workspace"
if [ -d /workspace ] && [ -w /workspace ]; then
    DEFAULT_DNN_BENCH_WORKSPACE="/workspace"
fi

BUILD_DIR="$HIPDNN_ROOT/build"
DEFAULT_ROCM_PREFIX="/opt/rocm"
DNN_BENCH_WORKSPACE="${DNN_BENCH_WORKSPACE:-$DEFAULT_DNN_BENCH_WORKSPACE}"
VENV_DIR="$DNN_BENCH_WORKSPACE/.venv"
MIOPEN_PROVIDER_DIR="$WORKSPACE_ROOT/dnn-providers/miopen-provider"
MIOPEN_BUILD_DIR="$MIOPEN_PROVIDER_DIR/build"
HIP_KERNEL_PROVIDER_DIR="$WORKSPACE_ROOT/dnn-providers/hip-kernel-provider"
HIP_KERNEL_BUILD_DIR="$HIP_KERNEL_PROVIDER_DIR/build"
HIPBLASLT_PROVIDER_DIR="$WORKSPACE_ROOT/dnn-providers/hipblaslt-provider"
HIPBLASLT_BUILD_DIR="$HIPBLASLT_PROVIDER_DIR/build"

FORCE_BUILD=0
AUTO_YES=0
REUSE_VENV=0
TORCH_MODE="${DNN_BENCH_TORCH_MODE:-rocm}"
ROCM_PREFIX="${DNN_BENCH_ROCM_PREFIX:-}"
GPU_ARCH_OVERRIDE="${DNN_BENCH_GPU_ARCH:-}"
TORCH_INDEX_URL="${DNN_BENCH_TORCH_INDEX_URL:-}"
RESOLVED_TORCH_INDEX_URL=""
INSTALLED_TORCH_MODE="missing"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "  Requires Python 3.12 or newer."
    echo ""
    echo "  --torch-mode <rocm|cuda|cpu|existing|none>"
    echo "                       Select how torch is provided. Default: $TORCH_MODE"
    echo "                         rocm: install ROCm torch nightly, use ROCm"
    echo "                               libraries/toolchain from the torch wheel's"
    echo "                               bundled ROCm SDK packages, and build local"
    echo "                               hipDNN/provider artifacts when absent."
    echo "                         cuda: install CUDA torch from PyPI (or"
    echo "                               --torch-index-url) for the PyTorch"
    echo "                               execution backend only. hipDNN bindings,"
    echo "                               engine plugins, and ROCm setup are skipped."
    echo "                         cpu:  install CPU-only torch and build bindings"
    echo "                               against installed ROCm/hipDNN."
    echo "                         existing:"
    echo "                               reuse torch already present in $VENV_DIR."
    echo "                               ROCm torch uses its bundled SDK libraries;"
    echo "                               CUDA torch skips hipDNN/ROCm setup;"
    echo "                               CPU torch uses installed ROCm/hipDNN."
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
    echo "  --force-build        Build hipDNN and provider plugins from source,"
    echo "                       overwriting artifacts under the selected ROCm prefix."
    echo "  -y                   Skip confirmation prompts."
    echo "  The selected ROCm prefix is exported as ROCM_PATH and its"
    echo "  lib directory is prepended to LD_LIBRARY_PATH by the venv"
    echo "  activation script. dnn-benchmarking infers plugins from:"
    echo "    \$ROCM_PATH/lib/hipdnn_plugins/engines/"
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

require_python_version() {
    python3 - <<'PY'
import sys

required = (3, 12)
if sys.version_info < required:
    version = ".".join(str(part) for part in sys.version_info[:3])
    raise SystemExit(
        f"ERROR: setup.sh requires Python >= 3.12, but python3 is {version}. "
        "Run setup with a Python 3.12+ environment."
    )
PY
}

prepend_ld_library_path() {
    local lib_dir="$1"
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$lib_dir:"*) ;;
        *) export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}

write_activation_local() {
    local rocm_prefix="$1"
    local lib_dir="$2"
    {
        printf 'export PYTHONPYCACHEPREFIX=%q\n' "$DNN_BENCH_WORKSPACE/pycache"
        printf 'export DNN_BENCH_WORKSPACE=%q\n' "$DNN_BENCH_WORKSPACE"
        printf 'export ROCM_PATH=%q\n' "$rocm_prefix"
        printf 'case ":%s:" in\n' '${LD_LIBRARY_PATH:-}'
        printf '    *:%s:*) ;;\n' "$lib_dir"
        printf '    *) export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ;;\n' "$lib_dir"
        printf 'esac\n'
    } > "$ACTIVATE_LOCAL"
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
    rocm|cuda|cpu|existing|none) ;;
    *)
        echo "ERROR: --torch-mode must be one of: rocm, cuda, cpu, existing, none" >&2
        exit 1
        ;;
esac

if [ "$TORCH_MODE" = "cuda" ] && [ "$FORCE_BUILD" -eq 1 ]; then
    echo "ERROR: --force-build is not supported with --torch-mode cuda;" >&2
    echo "hipDNN and provider plugins require a ROCm toolchain." >&2
    exit 1
fi

if [ "$TORCH_MODE" = "existing" ]; then
    REUSE_VENV=1
fi

require_python_version
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
import sys
import sysconfig

kind = sys.argv[1]
venv_root = Path(sys.prefix).resolve()

roots = []
for key in ("purelib", "platlib"):
    value = sysconfig.get_path(key)
    if value:
        path = Path(value).resolve()
        if path == venv_root or venv_root in path.parents:
            roots.append(path)

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
            lib_dir = child.joinpath("lib")
            if not lib_dir.is_dir() or not any(lib_dir.glob("libMIOpen.so*")):
                continue
        elif kind == "devel":
            if not child.name.startswith("_rocm_sdk_devel"):
                continue
            if not (
                child.joinpath("lib/llvm/bin/clang").is_file()
                and child.joinpath("lib/llvm/bin/clang++").is_file()
                and child.joinpath("lib/cmake/hip/hip-config.cmake").is_file()
            ):
                continue
        elif kind == "core":
            if not child.name.startswith("_rocm_sdk_core"):
                continue
            amd_smi_dir = child.joinpath("share/amd_smi")
            if not (
                child.joinpath("lib").is_dir()
                and any(child.joinpath("lib").glob("libamd_smi.so*"))
                and amd_smi_dir.is_dir()
                and (
                    amd_smi_dir.joinpath("setup.py").is_file()
                    or amd_smi_dir.joinpath("pyproject.toml").is_file()
                )
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

discover_rocm_wheel_core_prefix() {
    find_rocm_wheel_prefix core
}
expand_rocm_sdk_devel() {
    python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("rocm_sdk_devel") is None:
    sys.exit(1)

subprocess.run([sys.executable, "-m", "rocm_sdk", "init"], check=True)
PY
}

require_rocm_wheel_libraries_prefix() {
    local prefix status
    if prefix=$(discover_rocm_wheel_libraries_prefix); then
        echo "$prefix"
        return
    else
        status=$?
    fi
    if [ "$status" -ne 1 ]; then
        exit 1
    fi
    echo "ERROR: no usable ROCm SDK libraries package found in this venv." >&2
    echo "Expected exactly one _rocm_sdk_libraries_* package containing MIOpen libraries." >&2
    echo "Use a ROCm torch wheel that includes ROCm SDK libraries, pass --rocm-prefix explicitly, or pass --force-build." >&2
    exit 1
}

ensure_rocm_wheel_devel_prefix() {
    local index_url="$1"
    local prefix status
    if prefix=$(discover_rocm_wheel_devel_prefix); then
        echo "$prefix"
        return
    else
        status=$?
    fi
    if [ "$status" -ne 1 ]; then
        exit 1
    fi

    if expand_rocm_sdk_devel >&2; then
        if prefix=$(discover_rocm_wheel_devel_prefix); then
            echo "$prefix"
            return
        else
            status=$?
        fi
        if [ "$status" -ne 1 ]; then
            exit 1
        fi
    fi

    if [ -z "$index_url" ]; then
        echo "ERROR: no ROCm SDK compiler/toolchain prefix found in this venv." >&2
        echo "Expected exactly one _rocm_sdk_devel package with lib/llvm/bin/clang++ and hip CMake configs." >&2
        echo "Install rocm-sdk-devel from the same ROCm torch index, or pass --rocm-prefix." >&2
        exit 1
    fi

    echo "ROCm SDK compiler/toolchain prefix not found; installing rocm-sdk-devel from $index_url..." >&2
    pip install --pre rocm-sdk-devel --index-url "$index_url" >&2

    if ! expand_rocm_sdk_devel >&2; then
        echo "ERROR: rocm-sdk-devel installed, but its devel payload could not be expanded." >&2
        exit 1
    fi

    if prefix=$(discover_rocm_wheel_devel_prefix); then
        echo "$prefix"
        return
    else
        status=$?
    fi
    if [ "$status" -ne 1 ]; then
        exit 1
    fi
    echo "ERROR: rocm-sdk-devel installed, but no usable ROCm SDK compiler/toolchain prefix was found." >&2
    echo "Expected lib/llvm/bin/clang, lib/llvm/bin/clang++, and hip CMake configs under _rocm_sdk_devel." >&2
    exit 1
}
amdsmi_importable() {
    python - <<'PY'
import sys

try:
    import amdsmi  # noqa: F401
except Exception:
    sys.exit(1)
PY
}

maybe_install_amdsmi() {
    if amdsmi_importable; then
        return
    fi

    local prefix candidate status seen
    local -a candidates=()

    if prefix=$(discover_rocm_wheel_core_prefix); then
        candidate="$prefix/share/amd_smi"
        candidates+=("$candidate")
    else
        status=$?
        if [ "$status" -ne 1 ]; then
            echo "Warning: ROCm SDK core discovery failed; skipping SDK amdsmi candidate." >&2
        fi
    fi

    for prefix in "$@"; do
        if [ -z "$prefix" ]; then
            continue
        fi
        candidate="$prefix/share/amd_smi"
        if [ -d "$candidate" ] && { [ -f "$candidate/setup.py" ] || [ -f "$candidate/pyproject.toml" ]; }; then
            candidates+=("$candidate")
        fi
    done

    seen="|"
    for candidate in "${candidates[@]}"; do
        case "$seen" in
            *"|$candidate|"*) continue ;;
        esac
        seen="$seen$candidate|"

        if [ ! -d "$candidate" ] || { [ ! -f "$candidate/setup.py" ] && [ ! -f "$candidate/pyproject.toml" ]; }; then
            continue
        fi

        echo "Installing amdsmi Python bindings from $candidate..."
        if pip install -e "$candidate" && amdsmi_importable; then
            return
        fi
        echo "Warning: amdsmi install from $candidate failed; trying next candidate." >&2
    done

    echo "Warning: amdsmi Python bindings were not installed; GPU SMI snapshot will be disabled." >&2
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
    echo ""
}

get_torch_mode() {
    # `import torch` from a ROCm wheel with no visible GPU can print SDK probe
    # warnings to stdout, and this value is captured via command substitution.
    # Emit the mode on its own final line (the leading newline guards against a
    # warning that lacks a trailing newline) and read only that last line.
    python - <<'PY' | tail -n1
try:
    import torch
except Exception:
    mode = "missing"
else:
    if getattr(torch.version, "hip", None):
        mode = "rocm"
    elif getattr(torch.version, "cuda", None):
        mode = "cuda"
    else:
        mode = "cpu"
print("\n" + mode)
PY
}

require_torch_mode() {
    local expected="$1"
    local status
    status="${INSTALLED_TORCH_MODE:-$(get_torch_mode)}"
    if [ "$status" != "$expected" ]; then
        echo "ERROR: --torch-mode $expected requested, but $VENV_DIR contains torch mode '$status'." >&2
        echo "Use a clean workspace or remove the existing virtual environment before changing torch modes." >&2
        exit 1
    fi
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
            if [ "$INSTALLED_TORCH_MODE" = "missing" ]; then
                echo "ERROR: --torch-mode existing requires torch to already be installed in $VENV_DIR." >&2
                echo "Use --torch-mode rocm or --torch-mode cpu to install torch automatically." >&2
                exit 1
            fi
            echo "Using existing PyTorch in $VENV_DIR."
            return
            ;;
        cpu)
            local index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
            if [ "$INSTALLED_TORCH_MODE" != "missing" ]; then
                require_torch_mode cpu
                echo "Using existing CPU-only PyTorch in $VENV_DIR."
                return
            fi
            echo "Installing CPU-only PyTorch from $index_url"
            pip install torch --index-url "$index_url"
            INSTALLED_TORCH_MODE=$(get_torch_mode)
            require_torch_mode cpu
            ;;
        cuda)
            if [ "$INSTALLED_TORCH_MODE" != "missing" ]; then
                require_torch_mode cuda
                echo "Using existing CUDA PyTorch in $VENV_DIR."
                return
            fi
            if [ -n "$TORCH_INDEX_URL" ]; then
                echo "Installing CUDA PyTorch from $TORCH_INDEX_URL"
                pip install torch --index-url "$TORCH_INDEX_URL"
            else
                echo "Installing CUDA PyTorch from PyPI"
                pip install torch
            fi
            INSTALLED_TORCH_MODE=$(get_torch_mode)
            require_torch_mode cuda
            ;;
        rocm)
            local index_url="$TORCH_INDEX_URL"
            if [ -z "$index_url" ]; then
                local gpu_arch index_bucket
                gpu_arch=$(detect_gpu_arch)
                # ROCm nightly bucket per GPU arch. gfx90a's current torch +
                # ROCm SDK builds live in the bare "gfx90a" bucket; the older
                # "gfx90X-dcgpu" family bucket is frozen at a release that
                # predates several SDK libraries (e.g. hipdnn). gfx942/gfx950
                # are still served by their "-dcgpu" family buckets.
                case "$gpu_arch" in
                    gfx90a) index_bucket="gfx90a" ;;
                    gfx942) index_bucket="gfx94X-dcgpu" ;;
                    gfx950) index_bucket="gfx950-dcgpu" ;;
                    *)
                        echo "ERROR: Unsupported GPU architecture '${gpu_arch:-none}'." >&2
                        echo "Supported: gfx90a (MI200/MI210/MI250), gfx942 (MI300X/MI300A), gfx950 (MI350)" >&2
                        echo "Pass --gpu-arch or --torch-index-url to override detection." >&2
                        exit 1
                        ;;
                esac
                index_url="https://rocm.nightlies.amd.com/v2-staging/${index_bucket}/"
                echo "Detected GPU: $gpu_arch"
            fi
            RESOLVED_TORCH_INDEX_URL="$index_url"
            if [ "$INSTALLED_TORCH_MODE" != "missing" ]; then
                require_torch_mode rocm
                echo "Using existing ROCm PyTorch in $VENV_DIR."
                return
            fi
            echo "Installing ROCm PyTorch from $index_url"
            pip install --pre torch --index-url "$index_url"
            INSTALLED_TORCH_MODE=$(get_torch_mode)
            require_torch_mode rocm
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
            if [ "$INSTALLED_TORCH_MODE" = "rocm" ]; then
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
            if [ "$INSTALLED_TORCH_MODE" = "rocm" ]; then
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


build_hipdnn() {
    local install_prefix="$1"
    local toolchain_prefix="$2"
    local cmake_prefix_path="$install_prefix"
    local cmake_program_path="$toolchain_prefix/bin;$toolchain_prefix/lib/llvm/bin"

    if [ "$toolchain_prefix" != "$install_prefix" ]; then
        cmake_prefix_path="$install_prefix;$toolchain_prefix"
    fi

    echo "Building and installing hipDNN to $install_prefix..."
    echo "Using ROCm compiler/devel prefix: $toolchain_prefix"
    rm -rf "$BUILD_DIR"
    cmake -S "$HIPDNN_ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        "${HIPDNN_HIP_ARCH_ARGS[@]}" \
        -DCMAKE_INSTALL_PREFIX="$install_prefix" \
        -DCMAKE_PREFIX_PATH="$cmake_prefix_path" \
        -DCMAKE_PROGRAM_PATH="$cmake_program_path" \
        -DROCM_PATH="$toolchain_prefix" \
        -DHIPDNN_SKIP_TESTS=ON \
        -DHIPDNN_ENABLE_SDPA=ON \
        -DENABLE_CLANG_FORMAT=OFF \
        -DENABLE_CLANG_TIDY=OFF
    cmake --build "$BUILD_DIR"
    cmake --install "$BUILD_DIR"
}

build_provider() {
    local name="$1"
    local provider_dir="$2"
    local build_dir="$3"
    local install_prefix="$4"
    local toolchain_prefix="$5"
    shift 5

    local cmake_prefix_path="$install_prefix"
    local cmake_program_path="$toolchain_prefix/bin;$toolchain_prefix/lib/llvm/bin"

    if [ "$toolchain_prefix" != "$install_prefix" ]; then
        cmake_prefix_path="$install_prefix;$toolchain_prefix"
    fi

    if [ ! -d "$provider_dir" ]; then
        echo "Warning: $name not found at $provider_dir" >&2
        return 1
    fi

    echo "Building and installing $name to $install_prefix..."
    echo "Using ROCm compiler/devel prefix: $toolchain_prefix"
    rm -rf "$build_dir"
    cmake -S "$provider_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        "${HIPDNN_HIP_ARCH_ARGS[@]}" \
        -DCMAKE_INSTALL_PREFIX="$install_prefix" \
        -DCMAKE_PREFIX_PATH="$cmake_prefix_path" \
        -DCMAKE_PROGRAM_PATH="$cmake_program_path" \
        -DROCM_PATH="$toolchain_prefix" \
        -DENABLE_CLANG_FORMAT=OFF \
        -DENABLE_CLANG_TIDY=OFF \
        "$@" &&
        cmake --build "$build_dir" &&
        cmake --install "$build_dir"
}

build_miopen_provider() {
    build_provider \
        "MIOpen provider" \
        "$MIOPEN_PROVIDER_DIR" \
        "$MIOPEN_BUILD_DIR" \
        "$1" \
        "$2" \
        -DMIOPENPROVIDER_SKIP_TESTS=ON || return $?
    echo ""
    echo "MIOpen plugin installed to: $1/lib/hipdnn_plugins/engines/"
}

build_hipblaslt_provider() {
    build_provider \
        "hipBLASLt provider" \
        "$HIPBLASLT_PROVIDER_DIR" \
        "$HIPBLASLT_BUILD_DIR" \
        "$1" \
        "$2" \
        -DHIPDNN_SKIP_TESTS=ON || return $?
}

build_hip_kernel_provider() {
    build_provider \
        "hip-kernel-provider" \
        "$HIP_KERNEL_PROVIDER_DIR" \
        "$HIP_KERNEL_BUILD_DIR" \
        "$1" \
        "$2" \
        -DHIPKERNELPROVIDER_ENABLE_TESTS=OFF \
        -DENABLE_ASM_SDPA_ENGINE=ON || return $?
}

try_build_optional_provider() {
    local name="$1"
    shift

    if "$@"; then
        return 0
    fi

    echo "Warning: $name plugin build failed; continuing with any available providers." >&2
    return 1
}

has_engine_plugins() {
    local plugin_dir="$1"
    local plugins=()

    if [ ! -d "$plugin_dir" ]; then
        return 1
    fi

    plugins=("$plugin_dir"/*.so)
    [ -e "${plugins[0]}" ]
}

warn_no_native_engine_plugins() {
    local plugin_dir="$1"

    echo "Warning: no native hipDNN engine plugins were found in $plugin_dir." >&2
    echo "Setup will still finish, but default hipDNN benchmark runs need engine plugins." >&2
    echo "Pass --plugin-path or config plugin_path to use custom provider plugins." >&2
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
{
    printf 'export PYTHONPYCACHEPREFIX=%q\n' "$DNN_BENCH_WORKSPACE/pycache"
    printf 'export DNN_BENCH_WORKSPACE=%q\n' "$DNN_BENCH_WORKSPACE"
} > "$ACTIVATE_LOCAL"
INSTALLED_TORCH_MODE=$(get_torch_mode)
# An existing venv with CUDA torch reaches the CUDA skip path below even when
# --torch-mode existing was passed; building hipDNN there would silently be
# skipped, so reject the build request as early as possible.
if [ "$INSTALLED_TORCH_MODE" = "cuda" ] && [ "$FORCE_BUILD" -eq 1 ]; then
    echo "ERROR: --force-build is not supported with an existing CUDA torch venv;" >&2
    echo "building hipDNN requires a ROCm toolchain. Remove $VENV_DIR or use a ROCm torch mode." >&2
    exit 1
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

# CUDA torch supports only the PyTorch execution backend: no hipDNN Python
# bindings, engine plugins, amdsmi, or ROCm prefix are installed or required.
if [ "$INSTALLED_TORCH_MODE" = "cuda" ] || [ "$TORCH_MODE" = "cuda" ]; then
    echo ""
    echo "CUDA torch selected: skipping hipDNN/provider builds, hipDNN Python"
    echo "bindings, and ROCm environment setup."
    echo ""
    echo "Setup complete. Activate the virtual environment with:"
    echo "  source $VENV_DIR/bin/activate"
    echo ""
    echo "Run PyTorch-backend benchmarks with:"
    echo "  python -m dnn_benchmarking --graph <graph.json> --backend pytorch"
    exit 0
fi

# Resolve the GPU arch once and hand it to the HIP device-code builds. The
# wheel-bundled ROCm SDK ships no rocm_agent_enumerator/offload-arch on PATH and
# the build may run with no GPU, so HIP cannot autodetect the offload arch; pass
# it explicitly via hipDNN's documented GPU_TARGETS instead of letting HIP fall
# back to a default target list. --gpu-arch (or detection on a configured host)
# is the single source of truth -- no external GPU_TARGETS or
# ROCM_SDK_TARGET_FAMILY is required.
RESOLVED_GPU_ARCH=$(detect_gpu_arch)
HIPDNN_HIP_ARCH_ARGS=()
if [ -n "$RESOLVED_GPU_ARCH" ]; then
    HIPDNN_HIP_ARCH_ARGS=(-DGPU_TARGETS="$RESOLVED_GPU_ARCH" -DAMDGPU_TARGETS="$RESOLVED_GPU_ARCH")
    # Belt-and-suspenders for any torch C++/HIP extension compile (none today:
    # the Python bindings are nanobind host code linking hip::host).
    export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$RESOLVED_GPU_ARCH}"
fi

# 3. Select the hipDNN/ROCm prefix used by Python bindings and provider builds.
BINDING_PREFIX=$(select_binding_prefix)
echo "Using hipDNN/ROCm prefix: $BINDING_PREFIX"

PROVIDER_TOOLCHAIN_PREFIX=""
if [ "$FORCE_BUILD" -eq 1 ] || ! prefix_has_hipdnn "$BINDING_PREFIX"; then
    PROVIDER_TOOLCHAIN_PREFIX=$(select_provider_toolchain_prefix)
fi

BUILT_HIPDNN=0
if [ "$FORCE_BUILD" -eq 1 ]; then
    build_hipdnn "$BINDING_PREFIX" "$PROVIDER_TOOLCHAIN_PREFIX"
    BUILT_HIPDNN=1
elif ! prefix_has_hipdnn "$BINDING_PREFIX"; then
    if [ "$TORCH_MODE" = "rocm" ] || { [ "$TORCH_MODE" = "existing" ] && [ "$INSTALLED_TORCH_MODE" = "rocm" ]; }; then
        build_hipdnn "$BINDING_PREFIX" "$PROVIDER_TOOLCHAIN_PREFIX"
        BUILT_HIPDNN=1
    else
        echo "ERROR: hipDNN CMake configs were not found under $BINDING_PREFIX." >&2
        echo "Expected:" >&2
        echo "  $(hipdnn_config_path "$BINDING_PREFIX")" >&2
        echo "  $(hipdnn_backend_config_path "$BINDING_PREFIX")" >&2
        echo "Install ROCm/hipDNN artifacts there, use --rocm-prefix, or pass --force-build." >&2
        exit 1
    fi
fi


# 4. Build/install provider plugins if the selected prefix does not already
# contain the specific plugin artifacts. This keeps the ROCm torch wheel flow
# self-contained: torch supplies ROCm libraries, setup.sh adds local hipDNN and
# provider artifacts when needed.
PLUGIN_DIR="$BINDING_PREFIX/lib/hipdnn_plugins/engines"
MIOPEN_PLUGIN="$PLUGIN_DIR/libmiopen_plugin.so"
HIPBLASLT_PLUGIN="$PLUGIN_DIR/libhipblaslt_plugin.so"
HIP_KERNEL_PLUGIN="$PLUGIN_DIR/libhip_kernel_provider.so"
if [ "$FORCE_BUILD" -eq 1 ] || [ "$BUILT_HIPDNN" -eq 1 ] || \
    [ ! -f "$MIOPEN_PLUGIN" ] || [ ! -f "$HIPBLASLT_PLUGIN" ] || \
    [ ! -f "$HIP_KERNEL_PLUGIN" ]; then
    if [ -z "$PROVIDER_TOOLCHAIN_PREFIX" ]; then
        PROVIDER_TOOLCHAIN_PREFIX=$(select_provider_toolchain_prefix)
    fi
fi

PROVIDER_BUILD_FAILED=0
if [ "$FORCE_BUILD" -eq 1 ] || [ "$BUILT_HIPDNN" -eq 1 ] || [ ! -f "$MIOPEN_PLUGIN" ]; then
    try_build_optional_provider \
        "MIOpen provider" \
        build_miopen_provider "$BINDING_PREFIX" "$PROVIDER_TOOLCHAIN_PREFIX" ||
        PROVIDER_BUILD_FAILED=1
fi
if [ "$FORCE_BUILD" -eq 1 ] || [ "$BUILT_HIPDNN" -eq 1 ] || [ ! -f "$HIPBLASLT_PLUGIN" ]; then
    try_build_optional_provider \
        "hipBLASLt provider" \
        build_hipblaslt_provider "$BINDING_PREFIX" "$PROVIDER_TOOLCHAIN_PREFIX" ||
        PROVIDER_BUILD_FAILED=1
fi
if [ "$FORCE_BUILD" -eq 1 ] || [ "$BUILT_HIPDNN" -eq 1 ] || [ ! -f "$HIP_KERNEL_PLUGIN" ]; then
    try_build_optional_provider \
        "hip-kernel-provider" \
        build_hip_kernel_provider "$BINDING_PREFIX" "$PROVIDER_TOOLCHAIN_PREFIX" ||
        PROVIDER_BUILD_FAILED=1
fi

NATIVE_ENGINE_PLUGINS_AVAILABLE=1
if ! has_engine_plugins "$PLUGIN_DIR"; then
    NATIVE_ENGINE_PLUGINS_AVAILABLE=0
    warn_no_native_engine_plugins "$PLUGIN_DIR"
fi

if [ "$PROVIDER_BUILD_FAILED" -ne 0 ]; then
    echo "Warning: one or more provider plugins failed to build." >&2
    echo "Continuing with available or user-specified plugins." >&2
fi

echo ""
if [ "$NATIVE_ENGINE_PLUGINS_AVAILABLE" -eq 1 ]; then
    echo "hipDNN plugins available at: $PLUGIN_DIR/"
else
    echo "hipDNN plugin search path: $PLUGIN_DIR/ (no .so files found)"
fi
ROCM_PATH="$BINDING_PREFIX"
export ROCM_PATH
prepend_ld_library_path "$BINDING_PREFIX/lib"
write_activation_local "$ROCM_PATH" "$BINDING_PREFIX/lib"


# 6. Install hipDNN Python bindings.
# Wipe any stale cmake build cache (can reference deleted pip temp envs).
if [ -z "$PROVIDER_TOOLCHAIN_PREFIX" ]; then
    PROVIDER_TOOLCHAIN_PREFIX=$(select_provider_toolchain_prefix)
fi
maybe_install_amdsmi "$BINDING_PREFIX" "$PROVIDER_TOOLCHAIN_PREFIX" "${ROCM_PREFIX:-}" "$DEFAULT_ROCM_PREFIX"

PY_BINDING_CMAKE_PREFIX_PATH="$BINDING_PREFIX"
if [ "$PROVIDER_TOOLCHAIN_PREFIX" != "$BINDING_PREFIX" ]; then
    PY_BINDING_CMAKE_PREFIX_PATH="$BINDING_PREFIX;$PROVIDER_TOOLCHAIN_PREFIX"
fi
rm -rf "$HIPDNN_ROOT/python/build"
ROCM_PATH="$PROVIDER_TOOLCHAIN_PREFIX" \
    CMAKE_PREFIX_PATH="$PY_BINDING_CMAKE_PREFIX_PATH" \
    pip install -e "$HIPDNN_ROOT/python"


echo ""
echo "Setup complete. Activate the virtual environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run benchmarks with:"
echo "  python -m dnn_benchmarking --graph <graph.json>"
echo ""
echo "The activation script sets:"
echo "  ROCM_PATH=$ROCM_PATH"
echo "  LD_LIBRARY_PATH=$BINDING_PREFIX/lib:\${LD_LIBRARY_PATH}"
echo "dnn-benchmarking infers plugins from \$ROCM_PATH/lib/hipdnn_plugins/engines."
echo "Pass --plugin-path explicitly only when overriding the setup-installed plugins."
