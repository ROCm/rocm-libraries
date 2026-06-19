#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_parity.sh -- build the C emitter, then for each kernel emit the .ll from
# both the C engine and the Python reference and sha256-compare them.
#
# PASS = all four kernels byte-identical. Prints a per-kernel verdict + a diff
# hunk for any mismatch.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"          # ck_dsl_c root
PYROOT="$(cd "$CKC/.." && pwd)"           # python/ (holds ck_dsl package)
OUT="${TMPDIR:-/tmp}/ckc_parity"
mkdir -p "$OUT"

BIN="$OUT/emit_c"
echo ">> compiling C emitter"
# Engine sources are C++20 (.cpp); the emitter is built as C++20 alongside them.
c++ -std=c++20 -I "$CKC/include" $(find "$CKC/src" -name '*.cpp') "$HERE/emit.c" -o "$BIN" -lm || {
    echo "C emitter compile FAILED"; exit 1; }

export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

rc=0
for k in scalar memory forloop vector; do
    "$BIN" "$k" > "$OUT/c_$k.ll" 2> "$OUT/c_$k.err"
    python3 "$HERE/emit.py" "$k" > "$OUT/py_$k.ll" 2> "$OUT/py_$k.err"
    cs=$(sha256sum "$OUT/c_$k.ll"  | cut -d' ' -f1)
    ps=$(sha256sum "$OUT/py_$k.ll" | cut -d' ' -f1)
    if [ "$cs" = "$ps" ]; then
        echo "PASS  $k  $cs"
    else
        echo "FAIL  $k  C=$cs PY=$ps"
        diff -u "$OUT/py_$k.ll" "$OUT/c_$k.ll" | head -40
        [ -s "$OUT/c_$k.err" ]  && echo "C_ERR:  $(cat "$OUT/c_$k.err")"
        [ -s "$OUT/py_$k.err" ] && echo "PY_ERR: $(tail -3 "$OUT/py_$k.err")"
        rc=1
    fi
done
exit $rc
