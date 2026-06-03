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
# Distribution mode: frozen (bake modules into the .a) | py (.py on disk) |
# mpy (.mpy on disk). py/mpy build an on-disk module tree instead of freezing.
CKDSL_MODE="${CKDSL_MODE:-frozen}"

echo "== ck-dsl embed build"
echo "   OUT_DIR     = $OUT_DIR"
echo "   CK_DSL_SRC  = $CK_DSL_SRC"
echo "   ROCM_PATH   = $ROCM_PATH"
echo "   MODE        = $CKDSL_MODE"
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
# Regenerate genhdr from scratch: MicroPython's qstr scan is incremental and does
# not re-run when only QSTR_GEN_FLAGS change, so a mode switch (frozen <-> on-disk)
# would otherwise reuse a stale qstr pool missing the other mode's qstrs.
rm -rf "$BUILD_DIR"

# The embed package (py/*.c + genhdr) is needed in every mode. In on-disk modes
# the qstr scan must also see CKDSL_ON_DISK so the generated qstr pool includes
# the on-disk-only qstrs (e.g. MP_QSTR__mpy, referenced by modsys.c when
# MICROPY_PERSISTENT_CODE_LOAD is enabled) -- QSTR_GEN_FLAGS feeds the scan.
# QSTR_GEN_FLAGS defaults to -DNO_QSTR (breaks the qstr chicken-and-egg); keep it
# and add CKDSL_ON_DISK so the scan sees the on-disk config.
QSTR_EXTRA=()
if [ "$CKDSL_MODE" != "frozen" ]; then
    QSTR_EXTRA=("QSTR_GEN_FLAGS=-DNO_QSTR -DCKDSL_ON_DISK=1")
fi
echo "== generate embed package"
CKDSL_FROZEN_DIR="$FROZEN_DIR" make "${MAKE_ARGS[@]}" "${QSTR_EXTRA[@]}" -j"$JOBS" \
    micropython-embed-package >/dev/null

# Mode-specific module delivery + compile flags.
#  frozen : freeze the closure into frozen_content.c (compiled into the .a).
#  py/mpy : leave the closure on disk (frozen_src, or .mpy-compiled) for the
#           interpreter to load via sys.path; the .a carries no modules.
CFLAGS_COMMON=(
    -I"$PROVIDER_C_DIR" -I"$PKG_DIR" -I"$PKG_DIR/port" -I"$BUILD_DIR" -I"$MPY_DIR"
    -fPIC -fvisibility=hidden -Og -fno-common -Wall
)
mapfile -t SRCS < <(find "$PKG_DIR" -name '*.c' | sort)
SRCS+=("$PROVIDER_C_DIR/embed_port.c" "$PROVIDER_C_DIR/modcomgr.c" "$PROVIDER_C_DIR/comgr_compile.c")
SRCS+=("$MPY_DIR/extmod/modre.c")

if [ "$CKDSL_MODE" = "frozen" ]; then
    echo "== freeze modules (mpy-cross)"
    CKDSL_FROZEN_DIR="$FROZEN_DIR" make "${MAKE_ARGS[@]}" -j"$JOBS" "$BUILD_DIR/frozen_content.c" >/dev/null
    CFLAGS_COMMON+=(-DMICROPY_MODULE_FROZEN_MPY=1 -DMICROPY_MODULE_FROZEN_STR=1)
    SRCS+=("$BUILD_DIR/frozen_content.c")
else
    CFLAGS_COMMON+=(-DCKDSL_ON_DISK=1)
    if [ "$CKDSL_MODE" = "mpy" ]; then
        echo "== compiling on-disk modules to .mpy"
        MPY_CROSS="$MPY_DIR/mpy-cross/build/mpy-cross"
        MPY_OUT="$OUT_DIR/frozen_src_mpy"
        rm -rf "$MPY_OUT"
        while IFS= read -r py; do
            rel="${py#"$FROZEN_DIR"/}"
            dst="$MPY_OUT/${rel%.py}.mpy"
            mkdir -p "$(dirname "$dst")"
            "$MPY_CROSS" -o "$dst" "$py"
        done < <(find "$FROZEN_DIR" -name '*.py')
        echo "   on-disk .mpy tree: $MPY_OUT"
    else
        echo "   on-disk .py tree:  $FROZEN_DIR"
    fi
fi

OBJ_DIR="$OUT_DIR/obj"
rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"

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
