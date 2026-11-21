#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Build one per-family hipdnn-vllm wheel:
#   1. hipDNN from a rocm-libraries checkout (HIPDNN_SRC): the backend, the
#      frontend headers and every engine provider (rocKE-enabled
#      hip-kernel-provider, miopen-provider, hipblaslt-provider), installed to
#      a staging prefix;
#   2. the backend copied to hipdnn_vllm/backend/<family>/ and the engine
#      directory (providers plus their arch content) to
#      hipdnn_vllm/providers/<family>/ in a per-family copy of this tool
#      directory, plus _build_info.json;
#   3. the wheel (setup.py builds _C against the venv's torch and the staged
#      hipDNN frontend headers, in runtime-load mode: _C dlopens the bundled
#      backend by explicit path and never links a libhipdnn_backend).
#
# Run it on a compute node (sbatch), NEVER on a login node: it compiles.
#
# usage: build_wheel.sh <family>
#   family: a key of [families] in BUILD_PINS.toml (gfx94X, gfx950, gfx110X-all,
#           gfx1151, gfx120X-all)
# environment:
#   VENV      pinned vLLM venv (python with torch + rocm[devel]); required
#   WORK      scratch root for builds; required
#   OUT       directory for the finished wheel; required
#   KPACK_PYTHON_DIR  rocm_kpack python dir for the provider's kpack packing; required
#   TOOLS_VENV  venv with cmake, ninja, pytest, build, numpy (created if missing)
#   JOBS      parallel build jobs (default: nproc)
#   HIPDNN_SRC  rocm-libraries checkout hipDNN and the providers are built from
#             (default: the checkout holding this tool directory)
set -euo pipefail

FAMILY=${1:?usage: build_wheel.sh <family>}
: "${VENV:?set VENV}" "${WORK:?set WORK}" "${OUT:?set OUT}" "${KPACK_PYTHON_DIR:?set KPACK_PYTHON_DIR}"
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC=$(cd "${HIPDNN_SRC:-$TOOL_DIR/../../../..}" && pwd)
JOBS=${JOBS:-$(nproc)}
PY=$VENV/bin/python
TOOLS_VENV=${TOOLS_VENV:-$WORK/tools-venv}

echo "host=$(hostname) nproc=$(nproc) family=$FAMILY src=$SRC date=$(date -Is)"
test -f "$SRC/dnn-providers/hip-kernel-provider/CMakeLists.txt" || { echo "not a rocm-libraries checkout: $SRC" >&2; exit 1; }

# Line 1: the family's gfx targets; then one provider cmake flag per line.
mapfile -t PINNED < <("$PY" - "$TOOL_DIR/BUILD_PINS.toml" "$FAMILY" <<'EOF'
import sys, tomllib
pins = tomllib.load(open(sys.argv[1], "rb"))
fams = pins["families"]
if sys.argv[2] not in fams:
    sys.exit(f"unknown family {sys.argv[2]!r}; known: {', '.join(fams)}")
print(";".join(fams[sys.argv[2]]))
print("\n".join(pins["provider"]["cmake_flags"]))
EOF
)
test "${#PINNED[@]}" -ge 2 || { echo "could not read $TOOL_DIR/BUILD_PINS.toml for $FAMILY" >&2; exit 1; }
TARGETS=${PINNED[0]}
PROVIDER_FLAGS=("${PINNED[@]:1}")
echo "gpu targets: $TARGETS"
echo "provider flags: ${PROVIDER_FLAGS[*]}"

if [ ! -x "$TOOLS_VENV/bin/python" ]; then
  "$PY" -m venv "$TOOLS_VENV"
fi
# Build tools, rocm_kpack's runtime deps (the provider's kpack packing step) and
# dvc at the version hipdnn-superbuild-ci uses. dvc 3.62.0 imports a name that
# pathspec 1.x removed, so pathspec stays below 1.
"$TOOLS_VENV/bin/pip" install -q cmake ninja pytest build numpy "zstandard>=0.20.0" msgpack \
  "dvc[s3]==3.62.0" "pathspec<1"
SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])")
SITE_REAL=$(readlink -f "$SITE")
# pip wheel runs with --no-build-isolation, so pyproject's build requirement is
# not enforced: check the venv's setuptools takes the SPDX `license` string.
"$PY" -c 'import setuptools, sys; sys.exit(int(setuptools.__version__.split(".")[0]) < 77)' \
  || { echo "setuptools >= 77 required in $VENV (pyproject.toml uses an SPDX license string)" >&2; exit 1; }
ROCM_SDK_PATH=$("$VENV/bin/rocm-sdk" path --root)
test -d "$ROCM_SDK_PATH/include" || { echo "rocm-sdk-devel not expanded at $ROCM_SDK_PATH (run rocm-sdk init)" >&2; exit 1; }
COMGR_LIB=$(ls "$SITE/_rocm_sdk_core/lib/libamd_comgr.so".* | head -n1)
export PATH="$TOOLS_VENV/bin:$ROCM_SDK_PATH/bin:$PATH"
export ROCM_PATH=$ROCM_SDK_PATH ROCKE_COMGR_LIB=$COMGR_LIB HIPKERNELPROVIDER_ROCM_KPACK_DIR=$KPACK_PYTHON_DIR
RTLIB="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_libraries/lib"
# Steps that write into the shared source tree run under this per-checkout lock,
# so families built at once from one checkout do not race.
LOCK=$(git -C "$SRC" rev-parse --absolute-git-dir)/hipdnn_vllm-build.lock

# --- 0. rocKE descriptors ------------------------------------------------------
# The shipped rocKE ingestor bundles are DVC outputs. Without them the provider
# configures with only a warning and packs no rocKE content, so pull them the way
# hipdnn-superbuild-ci does (anonymous S3) and configure with
# REQUIRE_DVC_DESCRIPTORS=ON (BUILD_PINS.toml) so a missing bundle fails.
flock "$LOCK" bash -c 'cd "$1" && dvc pull --recursive dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors' _ "$SRC"

# --- 1. hipDNN backend, frontend and providers -----------------------------------
B=$WORK/hipdnn-$FAMILY/build
STAGE=$WORK/hipdnn-$FAMILY/stage
# Configure every time, so a flag change in BUILD_PINS.toml reaches an existing build tree.
(cd "$SRC" && cmake --preset hipdnn-dev-all -GNinja -B "$B" \
  -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hip-kernel-provider;miopen-provider;hipblaslt-provider" \
  -DCMAKE_INSTALL_PREFIX="$STAGE" \
  -DROCM_PATH="$ROCM_SDK_PATH" \
  -DCMAKE_PROGRAM_PATH="$ROCM_SDK_PATH/lib/llvm/bin" \
  -DGPU_TARGETS="$TARGETS" \
  -DENABLE_CLANG_FORMAT=OFF -DENABLE_CLANG_TIDY=OFF \
  "${PROVIDER_FLAGS[@]}" \
  -DHIPKERNELPROVIDER_ROCKE_COMGR_LIB="$COMGR_LIB" \
  -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR="$KPACK_PYTHON_DIR" \
  -DPython_EXECUTABLE="$TOOLS_VENV/bin/python" \
  -DPython3_EXECUTABLE="$TOOLS_VENV/bin/python")
echo "hipdnn build start $(date -Is)"
# rocKE's Python steps (rocke-pyenv's editable install and the platform wheel)
# run setuptools inside the source tree: they write platform/build/ and
# python/rocke.egg-info/ there. Two families built at once from one checkout
# race on those directories (one build deletes the bdist tree the other is
# writing), so those targets build under the lock first and the rest of the
# build then runs unlocked.
ROCKE_TARGETS=$(ninja -C "$B" -t targets all | sed -n 's/^\(rocke-pyenv\|rocke-wheels\): phony$/\1/p' | sort -u)
test -n "$ROCKE_TARGETS" || { echo "no rocke-pyenv/rocke-wheels targets in $B" >&2; exit 1; }
echo "building" $ROCKE_TARGETS "under $LOCK"
# shellcheck disable=SC2086  # target names, one per word
flock "$LOCK" cmake --build "$B" -j "$JOBS" --target $ROCKE_TARGETS
cmake --build "$B" -j "$JOBS"
rm -rf "$STAGE"
cmake --install "$B" --prefix "$STAGE" >/dev/null
ENGINES=$STAGE/lib/hipdnn_plugins/engines
BACKEND=$(readlink -f "$STAGE/lib/libhipdnn_backend.so")
for f in libhip_kernel_provider.so libmiopen_plugin.so libhipblaslt_plugin.so; do
  test -f "$ENGINES/$f" || { echo "no $f in $ENGINES" >&2; ls -R "$STAGE/lib" | head -50; exit 1; }
done
test -f "$BACKEND" || { echo "no libhipdnn_backend.so in $STAGE/lib" >&2; exit 1; }
echo "hipdnn build done $(date -Is)"

# The backend and every provider must resolve their dependencies from the pip
# ROCm install alone (miopen/hipblaslt against the SDK's libMIOpen/libhipblaslt),
# and no provider may bind a libhipdnn_backend by SONAME: the bundled backend is
# loaded by path, so a NEEDED entry could bind to torch's copy instead.
# ldd also resolves through RUNPATH, ld.so.cache and a system ROCm, so "no
# 'not found'" proves nothing: every resolved path is checked. ROCm libraries
# must come from the pip install (or the bundled engine directory); anything
# else may only come from the system library directories.
under_bundle() {
  case $1 in "$SITE"/* | "$SITE_REAL"/* | "$ENGINES"/*) return 0 ;; esac
  return 1
}
check_deps() {
  local so=$1 bad=0 line name path real
  while IFS= read -r line; do
    if [[ $line == *"not found"* ]]; then
      echo "$so: unresolved dependency: $line" >&2; bad=1; continue
    fi
    [[ $line == *" => /"* ]] || continue
    name=${line%% =>*}; name=${name//[[:space:]]/}
    path=${line#*=> }; path=${path%% (*}
    real=$(readlink -f "$path")
    if under_bundle "$path" || under_bundle "$real"; then
      continue
    fi
    if [[ $name =~ ^lib(amd|hsa|hip|roc|MIOpen|rccl) ]]; then
      echo "$so: $name resolves outside the pip ROCm install: $real" >&2; bad=1; continue
    fi
    case $real in
      /lib/* | /lib64/* | /usr/lib/* | /usr/lib64/*) ;;
      *) echo "$so: $name resolves outside the pip ROCm install and the system library dirs: $real" >&2; bad=1 ;;
    esac
  done < <(LD_LIBRARY_PATH="$RTLIB" ldd "$so")
  return $bad
}
for so in "$BACKEND" "$ENGINES"/lib*.so; do
  check_deps "$so" || { echo "$so has dependencies from outside the wheel's runtime" >&2; exit 1; }
  if readelf -d "$so" | grep -E "NEEDED.*libhipdnn_backend"; then
    echo "$so links libhipdnn_backend by SONAME" >&2; exit 1
  fi
  echo "deps ok: $(basename "$so")"
done
# The two SDK-backed providers must actually load the SDK library they wrap.
for pair in libmiopen_plugin.so:libMIOpen libhipblaslt_plugin.so:libhipblaslt; do
  so=${pair%%:*}; lib=${pair#*:}
  LD_LIBRARY_PATH="$RTLIB" ldd "$ENGINES/$so" | grep -E "^[[:space:]]*$lib[.-]" \
    || { echo "$so does not depend on $lib" >&2; exit 1; }
done

# --- 2. per-family package tree ----------------------------------------------
PKG=$WORK/pkg-$FAMILY
rm -rf "$PKG"
mkdir -p "$PKG"
(cd "$TOOL_DIR" && tar --exclude=build --exclude='*.egg-info' --exclude=__pycache__ \
   --exclude='hipdnn_vllm/providers' --exclude='hipdnn_vllm/backend' --exclude='hipdnn_vllm/_build_info.json' \
   --exclude='hipdnn_vllm/_C*.so' -cf - .) \
  | tar -xf - -C "$PKG"
mkdir -p "$PKG/hipdnn_vllm/providers/$FAMILY"
# Production content only, even if a build tree configured with tests staged test content.
(cd "$ENGINES" && tar --exclude=./test_arch_content -cf - .) | tar -xf - -C "$PKG/hipdnn_vllm/providers/$FAMILY"
mkdir -p "$PKG/hipdnn_vllm/backend/$FAMILY"
# The frontend appends the platform file name (libhipdnn_backend.so) to the
# directory _bootstrap points it at, so the real file is stored under that name.
cp "$BACKEND" "$PKG/hipdnn_vllm/backend/$FAMILY/libhipdnn_backend.so"
BACKEND_VERSION=$("$PY" -c 'import ctypes, sys
fn = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_LOCAL).hipdnnVersionString_ext
fn.restype = ctypes.c_char_p
print(fn().decode())' "$PKG/hipdnn_vllm/backend/$FAMILY/libhipdnn_backend.so")
echo "bundled backend version: $BACKEND_VERSION"

COMMIT=$(git -C "$SRC" rev-parse HEAD)
DIRTY=$(git -C "$SRC" status --porcelain -- dnn-providers projects/hipdnn | wc -l)
PLUGIN_COMMIT=$(git -C "$TOOL_DIR" rev-parse HEAD)
PLUGIN_DIRTY=$(git -C "$TOOL_DIR" status --porcelain -- . | wc -l)
"$PY" - "$PKG" "$FAMILY" "$TARGETS" "$COMMIT" "$DIRTY" "$BACKEND_VERSION" "$PLUGIN_COMMIT" "$PLUGIN_DIRTY" <<'EOF'
import json, os, sys, tomllib, datetime
pkg, family, targets, commit, dirty, backend_version, plugin_commit, plugin_dirty = sys.argv[1:9]
pins = tomllib.load(open(os.path.join(pkg, "BUILD_PINS.toml"), "rb"))
info = {
    "family": family,
    "gfx_targets": targets.split(";"),
    "hipdnn_commit": commit,
    "hipdnn_tree_dirty_files": int(dirty),
    "plugin_commit": plugin_commit,
    "plugin_tree_dirty_files": int(plugin_dirty),
    "backend_version": backend_version,
    "provider_cmake_flags": pins["provider"]["cmake_flags"],
    "pins": pins["pins"],
    "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(os.path.join(pkg, "hipdnn_vllm", "_build_info.json"), "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
EOF

# --- 3. wheel ------------------------------------------------------------------
mkdir -p "$OUT"
export CXX=$ROCM_SDK_PATH/lib/llvm/bin/clang++ CC=$ROCM_SDK_PATH/lib/llvm/bin/clang
export ROCM_SDK_PATH MAX_JOBS=$JOBS HIPDNN_VLLM_HIPDNN_PREFIX=$STAGE
(cd "$PKG" && HIPDNN_VLLM_FAMILY=$FAMILY "$PY" -m pip wheel --no-build-isolation --no-deps -w "$OUT" .)
ls -la "$OUT"
echo "WHEEL_OK $FAMILY $(date -Is)"
