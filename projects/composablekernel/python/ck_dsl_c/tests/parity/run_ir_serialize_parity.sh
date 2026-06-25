#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_ir_serialize_parity.sh -- build the C ck.dsl.ir/v1 emitter, then for each
# of the 7 sampled universal-GEMM configs serialize the IR from both the C
# engine (ir_serialize_emit.c) and the Python reference (ir_serialize_emit.py)
# and sha256-compare them.
#
# PASS = all configs byte-identical. "Both rejected / empty output" counts as
# PASS like the gemm runner. Prints a per-config verdict + the first diff hunk
# for any mismatch.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"          # ck_dsl_c root
PYROOT="$(cd "$CKC/.." && pwd)"           # python/ (holds ck_dsl package)
OUT="${TMPDIR:-/tmp}/ckc_ir_serialize_parity"
mkdir -p "$OUT"

BIN="$OUT/ir_serialize_emit_c"

# First try the gemm-runner style flat compile of all sources. If it fails with
# a duplicate-symbol link error (some helper buckets duplicate-define peers),
# fall back to linking against the CMake static archive libckc_core.a.
echo ">> compiling C ck.dsl.ir/v1 emitter (flat source build)"
# Engine sources are C++20 (.cpp); the emitter is built as C++20 alongside them.
if c++ -std=c++20 -I "$CKC/include" $(find "$CKC/src" -name '*.cpp') \
      "$HERE/ir_serialize_emit.c" -o "$BIN" -lm 2> "$OUT/cc.err"; then
    echo "   flat build OK"
else
    echo "   flat build FAILED (likely duplicate symbols); linking against archive"
    ARCHIVE="${CKC_ARCHIVE:-/tmp/ckc_build_ws1/libckc_core.a}"
    if [ ! -f "$ARCHIVE" ]; then
        echo "   building archive via CMake at /tmp/ckc_build_ws1"
        cmake -S "$CKC" -B /tmp/ckc_build_ws1 -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
        cmake --build /tmp/ckc_build_ws1 -j"$(nproc)" >/dev/null 2>&1
        ARCHIVE=/tmp/ckc_build_ws1/libckc_core.a
    fi
    if c++ -std=c++20 -I "$CKC/include" "$HERE/ir_serialize_emit.c" \
          "$ARCHIVE" -o "$BIN" -lm 2> "$OUT/cc2.err"; then
        echo "   archive build OK ($ARCHIVE)"
    else
        echo "C emitter compile FAILED"; cat "$OUT/cc2.err"; exit 1
    fi
fi

export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

NAMES=(test1 test2 test3 test4 test5 test6 test7)
rc=0
for i in 0 1 2 3 4 5 6; do
    n="${NAMES[$i]}"
    "$BIN" "$i" > "$OUT/c_$n.ir" 2> "$OUT/c_$n.err"
    crc=$?
    python3 "$HERE/ir_serialize_emit.py" "$i" > "$OUT/py_$n.ir" 2> "$OUT/py_$n.err"
    prc=$?

    if [ $crc -ne 0 ] || [ $prc -ne 0 ]; then
        if [ $crc -ne 0 ] && [ $prc -ne 0 ] \
           && ! [ -s "$OUT/c_$n.ir" ] && ! [ -s "$OUT/py_$n.ir" ]; then
            echo "PASS  [$i] $n  (both rejected, empty output)"
            [ -s "$OUT/py_$n.err" ] && echo "       reason: $(tail -1 "$OUT/py_$n.err")"
            continue
        fi
        echo "FAIL  [$i] $n  exit-status divergence  C_rc=$crc PY_rc=$prc"
        [ -s "$OUT/c_$n.err" ]  && echo "  C_ERR:  $(cat "$OUT/c_$n.err")"
        [ -s "$OUT/py_$n.err" ] && echo "  PY_ERR: $(tail -3 "$OUT/py_$n.err")"
        rc=1
        continue
    fi
    cs=$(sha256sum "$OUT/c_$n.ir"  | cut -d' ' -f1)
    ps=$(sha256sum "$OUT/py_$n.ir" | cut -d' ' -f1)
    if [ "$cs" = "$ps" ]; then
        echo "PASS  [$i] $n  $cs"
    else
        echo "FAIL  [$i] $n  C=$cs PY=$ps"
        diff -u "$OUT/py_$n.ir" "$OUT/c_$n.ir" | head -40
        rc=1
    fi
done
[ $rc -eq 0 ] && echo ">> ALL PARITY CHECKS PASSED" || echo ">> PARITY FAILURES PRESENT"
exit $rc
