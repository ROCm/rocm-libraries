#!/bin/bash
# Verifier job (rocke node, q38_job.sh --mode venv, 1 GPU; compile only): knob-off byte identity of the R2 rust tree
# vs the mono tree (R1's q38_identity setup + the R2 hybrid head), an A-vs-A mono control, and a must-fail mutant
# (R2 irops knobs forced on: exactly the head and toy best_spec labels must differ).
#   q38_job.sh --mode venv --stage w2-vrust-rust --stage w2-vrust-mono --stage w2-vrust-tools -- \
#     bash $Q38_STAGES/w2-vrust-tools/jobs/vrust_identity.sh
set -uo pipefail
RUST=$Q38_STAGES/w2-vrust-rust
MONO=$Q38_STAGES/w2-vrust-mono
TOOLS=$Q38_STAGES/w2-vrust-tools
O=$Q38_OUT
mkdir -p "$O/logs"
export ROCKE_BACKEND=python PYTHONDONTWRITEBYTECODE=1
unset ROCKE_CODEOBJ_BACKEND ROCKE_DIRECT_LEVERS
EXTRA=$Q38_JOB/pysite
python -c 'import msgpack' 2> /dev/null || python -m pip install -q --target "$EXTRA" msgpack > "$O/logs/pip_msgpack.txt" 2>&1
echo "== node $(hostname) comgr=${ROCKE_COMGR_LIB:-?} flavor=${ROCKE_LLVM_FLAVOR:-auto}"
rc=0
step() { # name cmd...
    local name=$1; shift
    "$@" > "$O/logs/$name.txt" 2>&1; local r=$?
    echo "== step $name rc=$r: $(grep -aE 'IDENTITY|MUTANT|Error|error' "$O/logs/$name.txt" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
    return $r
}
rpath() { echo "$1/platform/python:$1/library:$EXTRA"; }
P=$RUST/library/kernels/gfx950/qwen38_mono/direct_pilot.py
for t in mono_a mono_b rust; do
    tree=$MONO; [ $t = rust ] && tree=$RUST
    step dump_$t env PYTHONPATH=$(rpath "$tree") python "$TOOLS/v_identity.py" dump --pilot "$P" --out "$O/identity_$t.json" || rc=1
done
step dump_rust_mut env PYTHONPATH=$(rpath "$RUST") python "$TOOLS/v_identity.py" dump --pilot "$P" --out "$O/identity_rust_mut.json" --mutant irops || rc=1
step identity_control python "$TOOLS/v_identity.py" compare "$O/identity_mono_a.json" "$O/identity_mono_b.json" || rc=1
step identity python "$TOOLS/v_identity.py" compare "$O/identity_mono_a.json" "$O/identity_rust.json" || rc=1
step mutant python "$TOOLS/v_identity.py" compare "$O/identity_mono_a.json" "$O/identity_rust_mut.json" \
    --expect-differ head_logits,head_fused_argmax,head_nt2,head_hybrid,toy_best_d4w8_l2,toy_best_d8w0 || rc=1
cat "$O/logs/identity.txt" "$O/logs/mutant.txt"
echo "== VRUST rc=$rc"
exit $rc
