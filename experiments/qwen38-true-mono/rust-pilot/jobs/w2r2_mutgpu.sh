#!/bin/bash
# R2 hardware mutation check (rocke node, container mode): the direct pilot numerics (r2_gpu.py numerics) must FAIL with
# every mutant library of w2r2_mut.sh (and pass with the real one, as the control). Each run in a fresh process under
# a timeout: a mutant poll can spin forever.
#   q38_job.sh --mode container --stage w2-r2-rust --stage w2-r2-tools --stage w2-r2-bin --stage w2-r2-mut -- \
#     bash $Q38_STAGES/w2-r2-tools/jobs/w2r2_mutgpu.sh
set -uo pipefail
RUST=$Q38_STAGES/w2-r2-rust
TOOLS=$Q38_STAGES/w2-r2-tools
O=$Q38_OUT
mkdir -p "$O/logs"
EXTRA=$Q38_JOB/pysite
python -c 'import msgpack' 2> /dev/null || python -m pip install -q --target "$EXTRA" msgpack > /dev/null 2>&1
export PYTHONPATH=$RUST/platform/python:$RUST/library:$EXTRA ROCKE_BACKEND=python PYTHONDONTWRITEBYTECODE=1
unset ROCKE_CODEOBJ_BACKEND
rc=0
run() { # name lib what -> numerics rc
    local name=$1 lib=$2 what=$3
    ROCKE_CODEOBJ_DIRECT_LIB=$lib timeout 900 python "$TOOLS/r2_gpu.py" numerics --out "$O/$name" --what $what > "$O/logs/$name.txt" 2>&1
}
run control "$Q38_STAGES/w2-r2-bin/so/librocke_codeobj_ffi.so" "head toy"
c=$?
echo "== control rc=$c (must be 0): $(grep -aE 'NUMERICS' "$O/logs/control.txt" | tail -1)"
[ $c = 0 ] || rc=1
for lib in "$Q38_STAGES"/w2-r2-mut/mut_*.so; do
    n=$(basename "$lib" .so)
    case $n in
        *cvt_byte* | *permlane16*) what=head ;;
        *) what="head toy" ;;
    esac
    run "$n" "$lib" "$what"
    r=$?
    v=KILLED; [ $r = 0 ] && { v=SURVIVED; rc=1; }
    echo "== $n rc=$r $v: $(grep -aE 'NUMERICS|DeviceHang|Error|error' "$O/logs/$n.txt" | tail -1 | cut -c1-200)"
done
echo "== W2R2MUT rc=$rc"
exit $rc
