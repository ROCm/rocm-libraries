#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_gemm_parity.sh -- build the C universal-GEMM emitter, then for each of the
# 7 sampled configs emit the .ll from both the C engine (gemm_emit.c) and the
# Python reference (gemm_emit.py) and sha256-compare them.
#
# PASS = all configs byte-identical. Prints a per-config verdict + the first
# diff hunk for any mismatch.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"          # ck_dsl_c root
PYROOT="$(cd "$CKC/.." && pwd)"           # python/ (holds ck_dsl package)
OUT="${TMPDIR:-/tmp}/ckc_gemm_parity"
mkdir -p "$OUT"

BIN="$OUT/gemm_emit_c"
echo ">> compiling C universal-GEMM emitter"
cc -std=c99 -I "$CKC/include" "$CKC"/src/*.c "$HERE/gemm_emit.c" -o "$BIN" -lm || {
    echo "C emitter compile FAILED"; exit 1; }

export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

NAMES=(test1 test2 test3 test4 test5 test6 test7)
rc=0
for i in 0 1 2 3 4 5 6; do
    n="${NAMES[$i]}"
    "$BIN" "$i" > "$OUT/c_$n.ll" 2> "$OUT/c_$n.err"
    crc=$?
    python3 "$HERE/gemm_emit.py" "$i" > "$OUT/py_$n.ll" 2> "$OUT/py_$n.err"
    prc=$?

    # A config may be rejected by the validity gate on BOTH sides; that is still
    # parity (identical empty .ll + matching nonzero exit). Only a divergence in
    # exit status, or differing emitted .ll, is a failure.
    if [ $crc -ne 0 ] || [ $prc -ne 0 ]; then
        if [ $crc -ne 0 ] && [ $prc -ne 0 ] \
           && ! [ -s "$OUT/c_$n.ll" ] && ! [ -s "$OUT/py_$n.ll" ]; then
            echo "PASS  [$i] $n  (both rejected, empty .ll)"
            [ -s "$OUT/py_$n.err" ] && echo "       reason: $(tail -1 "$OUT/py_$n.err")"
            continue
        fi
        echo "FAIL  [$i] $n  exit-status divergence  C_rc=$crc PY_rc=$prc"
        [ -s "$OUT/c_$n.err" ]  && echo "  C_ERR:  $(cat "$OUT/c_$n.err")"
        [ -s "$OUT/py_$n.err" ] && echo "  PY_ERR: $(tail -3 "$OUT/py_$n.err")"
        rc=1
        continue
    fi
    cs=$(sha256sum "$OUT/c_$n.ll"  | cut -d' ' -f1)
    ps=$(sha256sum "$OUT/py_$n.ll" | cut -d' ' -f1)
    if [ "$cs" = "$ps" ]; then
        echo "PASS  [$i] $n  $cs"
    else
        echo "FAIL  [$i] $n  C=$cs PY=$ps"
        diff -u "$OUT/py_$n.ll" "$OUT/c_$n.ll" | head -40
        rc=1
    fi
done
[ $rc -eq 0 ] && echo ">> ALL PARITY CHECKS PASSED" || echo ">> PARITY FAILURES PRESENT"
exit $rc
