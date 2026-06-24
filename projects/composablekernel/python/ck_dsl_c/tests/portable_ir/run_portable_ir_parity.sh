#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_portable_ir_parity.sh -- prove the Python frontend -> portable IR ->
# C backend boundary is lossless.
#
# For each parity kernel:
#   1. Python lowers it directly             -> py_<k>.ll   (reference)
#   2. Python exports it to portable IR JSON -> <k>.ir.json
#   3. The C roundtrip driver imports the JSON and lowers it -> cjson_<k>.ll
#   4. sha256(py) vs sha256(cjson) must match.
#
# PASS = all kernels byte-identical between the Python lowering and the
# C-from-portable-IR lowering.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKC="$(cd "$HERE/../.." && pwd)"          # ck_dsl_c root
PYROOT="$(cd "$CKC/.." && pwd)"           # python/ (holds ck_dsl package)
PARITY="$CKC/tests/parity"
OUT="${TMPDIR:-/tmp}/ckc_portable_ir"
ARCH="${1:-gfx950}"
mkdir -p "$OUT"

BIN="$OUT/roundtrip"
echo ">> compiling C roundtrip driver"
cc -std=c99 -I "$CKC/include" "$CKC"/src/portable_ir/*.c "$HERE/roundtrip.c" -o "$BIN" -lm || {
    echo "C roundtrip compile FAILED"; exit 1; }

export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"

rc=0
for k in scalar memory forloop vector; do
    python3 "$PARITY/emit.py" "$k" > "$OUT/py_$k.ll" 2> "$OUT/py_$k.err"
    python3 "$HERE/export_parity.py" "$k" > "$OUT/$k.ir.json" 2> "$OUT/exp_$k.err"
    "$BIN" "$OUT/$k.ir.json" "$ARCH" > "$OUT/cjson_$k.ll" 2> "$OUT/cjson_$k.err"
    ps=$(sha256sum "$OUT/py_$k.ll"    | cut -d' ' -f1)
    cs=$(sha256sum "$OUT/cjson_$k.ll" | cut -d' ' -f1)
    if [ "$ps" = "$cs" ]; then
        echo "PASS  $k  $cs"
    else
        echo "FAIL  $k  PY=$ps  CJSON=$cs"
        diff -u "$OUT/py_$k.ll" "$OUT/cjson_$k.ll" | head -40
        [ -s "$OUT/cjson_$k.err" ] && echo "CJSON_ERR: $(cat "$OUT/cjson_$k.err")"
        [ -s "$OUT/exp_$k.err" ]   && echo "EXPORT_ERR: $(tail -3 "$OUT/exp_$k.err")"
        rc=1
    fi
done
exit $rc
