#!/usr/bin/env bash
# Build the embedded-MicroPython static library the ck-dsl-provider plugin links.
#
# Pipeline (all out-of-source, under $OUT_DIR):
#   1. clone MicroPython @ pin + build mpy-cross
#   2. build_bundle.py   : transform ck_dsl + ck_dsl_provider -> embed bundle
#   3. build_frozen.py   : capture the conv codegen closure -> frozen_src
#   4. gen.mk (embed.mk) : generate micropython_embed/ package + frozen_content.c
#   5. compile every embed .c + frozen_content.c + the provider's port/comgr/modcomgr
#      + extmod/modre.c into libckdsl_micropython.a  (-fPIC, hidden visibility)
#
# Outputs the static lib at $OUT_DIR/libckdsl_micropython.a and the include dirs
# the C++ bridge/interpreter compile against ($OUT_DIR/micropython_embed[/port],
# $OUT_DIR/build-embed, and the provider src/micropython/). Driven by CMake but
# also runnable standalone for debugging.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDER_ROOT="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$PROVIDER_ROOT/../.." && pwd)"

MPY_COMMIT="${MPY_COMMIT:-44a569b637}"
PROVIDER_C_DIR="${PROVIDER_C_DIR:-$PROVIDER_ROOT/src/micropython}"
CK_DSL_SRC="${CK_DSL_SRC:-$REPO_ROOT/projects/composablekernel/python/ck_dsl}"
CK_DSL_PROVIDER_SRC="${CK_DSL_PROVIDER_SRC:-$PROVIDER_ROOT/python/ck_dsl_provider}"
OUT_DIR="${OUT_DIR:-$PROVIDER_ROOT/build-micropython}"
MPY_DIR="${MPY_DIR:-$OUT_DIR/micropython}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
PYTHON="${PYTHON:-python3}"
CC="${CC:-cc}"
AR="${AR:-ar}"
JOBS="$(nproc)"

echo "== ck-dsl embed build"
echo "   OUT_DIR     = $OUT_DIR"
echo "   CK_DSL_SRC  = $CK_DSL_SRC"
echo "   ROCM_PATH   = $ROCM_PATH"
mkdir -p "$OUT_DIR"

# --- 1. MicroPython source + mpy-cross -------------------------------------
if [ ! -d "$MPY_DIR/.git" ]; then
    echo "== cloning MicroPython @ $MPY_COMMIT"
    git clone --quiet https://github.com/micropython/micropython "$MPY_DIR"
    git -C "$MPY_DIR" checkout --quiet "$MPY_COMMIT"
fi
echo "== building mpy-cross"
make -C "$MPY_DIR/mpy-cross" -j"$JOBS" >/dev/null

# --- 2. Transform ck_dsl + ck_dsl_provider into the embed bundle -----------
BUNDLE_DIR="$OUT_DIR/ckbundle"
echo "== build_bundle.py"
CK_DSL_SRC="$CK_DSL_SRC" CK_DSL_PROVIDER_SRC="$CK_DSL_PROVIDER_SRC" BUNDLE_DIR="$BUNDLE_DIR" \
    "$PYTHON" "$HERE/build_bundle.py"

# --- 3. Capture the codegen closure -> frozen_src --------------------------
FROZEN_DIR="$OUT_DIR/frozen_src"
echo "== build_frozen.py"
BUNDLE_DIR="$BUNDLE_DIR" SHIMS_DIR="$HERE/shims" FROZEN_DIR="$FROZEN_DIR" \
    "$PYTHON" "$HERE/build_frozen.py"

# --- 4. Embed package + frozen_content.c (out of source) -------------------
PKG_DIR="$OUT_DIR/micropython_embed"
BUILD_DIR="$OUT_DIR/build-embed"
MAKE_ARGS=(
    -C "$PROVIDER_C_DIR"
    -f "$HERE/gen.mk"
    MICROPYTHON_TOP="$MPY_DIR"
    FROZEN_MANIFEST="$HERE/manifest.py"
    CKDSL_C_DIR="$PROVIDER_C_DIR"
    BUILD="$BUILD_DIR"
    PACKAGE_DIR="$PKG_DIR"
)
echo "== generate embed package"
CKDSL_FROZEN_DIR="$FROZEN_DIR" make "${MAKE_ARGS[@]}" -j"$JOBS" micropython-embed-package >/dev/null
echo "== freeze modules (mpy-cross)"
CKDSL_FROZEN_DIR="$FROZEN_DIR" make "${MAKE_ARGS[@]}" -j"$JOBS" "$BUILD_DIR/frozen_content.c" >/dev/null

# --- 5. Compile the static library -----------------------------------------
OBJ_DIR="$OUT_DIR/obj"
rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"

CFLAGS_COMMON=(
    -I"$PROVIDER_C_DIR" -I"$PKG_DIR" -I"$PKG_DIR/port" -I"$BUILD_DIR" -I"$MPY_DIR"
    -fPIC -fvisibility=hidden -Og -fno-common -Wall
    -DMICROPY_MODULE_FROZEN_MPY=1 -DMICROPY_MODULE_FROZEN_STR=1
)

# Source set: embed package + frozen content + provider C + extmod/modre.
mapfile -t SRCS < <(find "$PKG_DIR" -name '*.c' | sort)
SRCS+=("$BUILD_DIR/frozen_content.c")
SRCS+=("$PROVIDER_C_DIR/embed_port.c" "$PROVIDER_C_DIR/modcomgr.c" "$PROVIDER_C_DIR/comgr_compile.c")
SRCS+=("$MPY_DIR/extmod/modre.c")

echo "== compiling ${#SRCS[@]} embed sources"
OBJS=()
i=0
for src in "${SRCS[@]}"; do
    obj="$OBJ_DIR/$(printf '%03d' "$i")_$(basename "${src%.c}").o"
    extra=()
    # comgr_compile.c is the only TU that needs the ROCm headers (amd_comgr.h).
    if [ "$(basename "$src")" = "comgr_compile.c" ]; then
        extra=(-I"$ROCM_PATH/include")
    fi
    "$CC" "${CFLAGS_COMMON[@]}" "${extra[@]}" -c "$src" -o "$obj"
    OBJS+=("$obj")
    i=$((i + 1))
done

LIB="$OUT_DIR/libckdsl_micropython.a"
rm -f "$LIB"
"$AR" rcs "$LIB" "${OBJS[@]}"
echo "== done: $LIB ($(du -h "$LIB" | cut -f1))"
