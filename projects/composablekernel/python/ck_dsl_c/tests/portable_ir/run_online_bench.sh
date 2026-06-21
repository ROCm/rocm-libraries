#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_online_bench.sh -- build a shared libckc + the comgr tool, then run the
# online portable-IR compile-timeline benchmark (ck_dsl.portable_ir.bench_online).
# Splits each compile into build / serialize / py_lower / c_build / c_lower /
# comgr so the handoff cost is attributed against the (dominant) backend compile.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"
PYROOT="$(cd "$CKC/.." && pwd)"
OUT="${TMPDIR:-/tmp}/ckc_online_bench"
ARCH="${1:-gfx950}"
ROCM="${ROCM_PATH:-/opt/rocm}"
mkdir -p "$OUT"
export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

echo ">> building shared libckc.so (incl online.c) + comgr tool"
cc -std=c99 -O2 -fPIC -shared -I "$CKC/include" "$CKC"/src/*.c -lm -o "$OUT/libckc.so" 2>/dev/null || {
    echo "libckc.so build FAILED"; exit 1; }
if cc -std=c99 -O2 -I "$ROCM/include" "$HERE/comgr_compile_ll.c" -L"$ROCM/lib" -lamd_comgr -o "$OUT/comgr" 2>/dev/null; then
    export COMGR="$OUT/comgr"
else
    echo "   (comgr tool unavailable -- comgr timing will be n/a)"
fi
export CKC_LIB="$OUT/libckc.so" ARCH
echo ""
python3 -m ck_dsl.portable_ir.bench_online
