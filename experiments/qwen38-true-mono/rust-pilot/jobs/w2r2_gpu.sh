#!/bin/bash
# R2 GPU job (rocke node, q38_job.sh --mode container: the hybrid's ROCm 10 COMGR): the direct compiler's pilot gates
# and, when every gate passes, the interleaved timing; numerics always before timing, in the same job.
#   q38_job.sh --mode container --stage w2-r2-rust --stage w2-r2-tools --stage w2-r2-bin \
#     [--env W2R2_TIME=0] [--env W2R2_ROUNDS=N] [--env W2R2_ARMS=head:NAME=LABEL/BACKEND/LEVERS,...] -- \
#     bash $Q38_STAGES/w2-r2-tools/jobs/w2r2_gpu.sh
set -uo pipefail
RUST=$Q38_STAGES/w2-r2-rust
TOOLS=$Q38_STAGES/w2-r2-tools
BIN=$Q38_STAGES/w2-r2-bin
O=$Q38_OUT
mkdir -p "$O/logs" "$O/obj"
export ROCKE_CODEOBJ_DIRECT_LIB=$BIN/so/librocke_codeobj_ffi.so ROCKE_BACKEND=python PYTHONDONTWRITEBYTECODE=1
unset ROCKE_CODEOBJ_BACKEND
LLVM=/work/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin
EXTRA=$Q38_JOB/pysite
if ! python -c 'import msgpack' 2> /dev/null; then
    python -m pip install -q --target "$EXTRA" msgpack > "$O/logs/pip_msgpack.txt" 2>&1
fi
export PYTHONPATH=$RUST/platform/python:$RUST/library:$EXTRA
echo "== node $(hostname) direct lib $(sha256sum "$ROCKE_CODEOBJ_DIRECT_LIB" | cut -c1-16) flavor=${ROCKE_LLVM_FLAVOR:-auto}"
python -c 'import msgpack; print("== msgpack", msgpack.version)' 2> /dev/null || echo "== msgpack ABSENT"
rc=0
step() { # name cmd...
    local name=$1; shift
    local t0=$SECONDS
    "$@" > "$O/logs/$name.txt" 2>&1; local r=$?
    echo "== step $name rc=$r ($((SECONDS - t0)) s): $(grep -aE 'passed|failed|NUMERICS|offending|OK|FAIL|error' "$O/logs/$name.txt" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
    [ $r = 0 ] || rc=1
    return $r
}
# 1. tests with the direct library on gfx950 (strict: no skips)
cd "$RUST/library" || exit 126
step q38_pilot env Q38_DIRECT_STRICT=1 python -m pytest -p no:cacheprovider -q -rs tests/test_qwen38_direct_pilot.py
step q38_related python -m pytest -p no:cacheprovider -q -rs tests/test_mailbox_ir.py tests/test_mx_decode_gemv.py \
    tests/test_qwen38_layer_framework.py tests/test_qwen38_stage_adapt.py
cd "$RUST/platform" || exit 126
step py_direct python -m pytest -p no:cacheprovider -q -rs tests/runtime/test_code_object.py tests/runtime/test_code_object_elf.py \
    tests/runtime/test_direct_gemm.py tests/runtime/test_direct_targets.py tests/runtime/test_direct_session.py
step py_core python -m pytest -p no:cacheprovider -q -rs tests/core/test_persistent_mem_ops.py tests/test_mailbox_helpers.py \
    tests/core/test_compile_target_selection.py tests/core/test_ir_serialize.py
# 2. objects: COMGR and direct of every pilot, direct at forced VGPR budgets
cd "$O" || exit 126
step build python "$TOOLS/r2_build.py" --out "$O/obj" --backends comgr direct --allow-refusal \
    --spill-fracs 0.5 0.75 --target-compile "$BIN/bin/target_compile"
for h in obj/*.hsaco; do "$LLVM/llvm-objdump" -d --mcpu=gfx950 "$h" > "${h%.hsaco}.s" 2> /dev/null; done
# 3. exec-join ISA lint of every object; hazard_corpus on the COMGR objects (the recognizer never waits less)
step lint python "$TOOLS/r2_lint.py" obj/*.s
# hazard_corpus cannot tell a hazard s_nop from padding or an inline-asm statement's s_nop: every site where the
# recognizer waits less than COMGR is classified (r2_hazard_classify.py); an unexplained site fails the gate.
hargs=()
for s in obj/*.comgr.s; do
    b=$(basename "$s" .s)
    "$BIN/bin/hazard_corpus" "logs/hazard_$b.tsv" "$s" > "logs/hazard_$b.txt" 2>&1
    echo "== hazard $b: $(tail -n 2 "logs/hazard_$b.txt" | tr '\n' ' ' | cut -c1-250)"
    hargs+=("logs/hazard_$b.tsv" "$s")
done
step hazard_classify python "$TOOLS/r2_hazard_classify.py" "${hargs[@]}"
# 4. numerics on hardware (bitwise vs COMGR, forced spills, references)
SPILL=$O/spill
mkdir -p "$SPILL"
python - "$O/obj" "$SPILL" << 'EOF'
import os, shutil, sys
src, dst = sys.argv[1:]
m = {"head_hybrid_irops": "head_fused_irops", "toy_best_d4w8_l2_irops": "toy_d4w8l2_irops", "toy_best_d8w0_irops": "toy_d8w0_irops"}
for f in os.listdir(src):
    for a, b in m.items():
        if f.startswith(a + ".direct_v"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, b + f[len(a):]))
print(sorted(os.listdir(dst)))
EOF
ARMS=$(echo "${W2R2_ARMS:-}" | tr ',' ' ')
step numerics python "$TOOLS/r2_gpu.py" numerics --out "$O" --spill-dir "$SPILL" --arm $ARMS
num=$?
# 5. timing (only after every gate passed)
if [ "${W2R2_TIME:-1}" = 1 ] && [ $rc = 0 ] && [ $num = 0 ]; then
    echo "== neighbours on $(hostname -s): $(squeue -h -w "$(hostname -s)" -o '%u:%j:%b' 2> /dev/null | tr '
' ' ' | cut -c1-400)"
    (while true; do amd-smi metric -g 0 --clock --power --json 2> /dev/null | tr -d '\n'; echo; sleep 2; done) > "$O/logs/amdsmi.jsonl" &
    smi=$!
    step time python "$TOOLS/r2_gpu.py" time --out "$O" --rounds "${W2R2_ROUNDS:-20}" --arm $ARMS
    kill $smi 2> /dev/null
    cat "$O/logs/time.txt" | grep -E "^head|^toy" | cut -c1-600
else
    echo "== timing skipped (W2R2_TIME=${W2R2_TIME:-1} gates_rc=$rc numerics_rc=$num)"
fi
rm -rf "$O/obj"/*.hsaco "$O/fresh_"*.hsaco
echo "== W2R2GPU rc=$rc"
exit $rc
