#!/bin/bash
# R1 node job (rocke partition, q38_job.sh --mode venv, 1 GPU): Python direct tests, the Q38 pilot
# tests with the direct library, knob-off byte identity against the mono tree, hazard_corpus on the
# COMGR objects of the Q38 builds.
#   q38_job.sh --mode venv --stage w2-r1-rust --stage w2-r1-mono --stage w2-r1-tools \
#     --env W2R1_SO=<dir with librocke_codeobj_ffi.so> --env W2R1_BIN=<dir with hazard_corpus> -- \
#     bash $Q38_STAGES/w2-r1-tools/jobs/w2r1_gpu.sh
set -uo pipefail
RUST=$Q38_STAGES/w2-r1-rust
MONO=$Q38_STAGES/w2-r1-mono
TOOLS=$Q38_STAGES/w2-r1-tools
O=$Q38_OUT
mkdir -p "$O/logs" "$O/hsaco" "$O/disasm"
tl=$(python -c 'import importlib.util,os; print(os.path.join(os.path.dirname(importlib.util.find_spec("torch").origin),"lib"))')
export ROCKE_HIP_LIB=${ROCKE_HIP_LIB:-$(ls $tl/libamdhip64.so* 2> /dev/null | head -1)}
export ROCKE_BACKEND=python PYTHONDONTWRITEBYTECODE=1
export ROCKE_CODEOBJ_DIRECT_LIB=$W2R1_SO/librocke_codeobj_ffi.so
unset ROCKE_CODEOBJ_BACKEND
echo "== node $(hostname) hip=$ROCKE_HIP_LIB comgr=$ROCKE_COMGR_LIB flavor=$ROCKE_LLVM_FLAVOR"
echo "== direct lib $(sha256sum "$ROCKE_CODEOBJ_DIRECT_LIB")"
# msgpack (platform/pyproject.toml dev dependency; code_object_elf reads the AMDGPU metadata note) is not in
# the shared venv: install it job-locally, never into the venv.
if ! python -c 'import msgpack' 2> /dev/null; then
    python -m pip install -q --target "$Q38_JOB/pysite" msgpack > "$O/logs/pip_msgpack.txt" 2>&1
fi
EXTRA=$Q38_JOB/pysite
PYTHONPATH=$EXTRA python -c 'import msgpack; print("== msgpack", msgpack.version)' 2> /dev/null || echo "== msgpack ABSENT"
rc=0
step() { # name cmd...
    local name=$1; shift
    local t0=$SECONDS
    "$@" > "$O/logs/$name.txt" 2>&1; local r=$?
    echo "== step $name rc=$r ($((SECONDS - t0)) s): $(grep -aE 'passed|failed|IDENTITY|error|Error' "$O/logs/$name.txt" | tail -n 2 | tr '\n' ' ')"
    [ $r = 0 ] || rc=1
    return $r
}
rpath() { echo "$1/platform/python:$1/library:$EXTRA"; }

# 1. Python direct runtime + tool tests (rust tree, platform layer).
cd "$RUST/platform" || exit 126
step py_direct env PYTHONPATH=python:$EXTRA python -m pytest -p no:cacheprovider -q -rs \
    tests/runtime/test_code_object.py tests/runtime/test_code_object_elf.py tests/runtime/test_direct_gemm.py \
    tests/runtime/test_direct_targets.py tests/runtime/test_direct_session.py
step py_core env PYTHONPATH=python:$EXTRA python -m pytest -p no:cacheprovider -q -rs \
    tests/core/test_compile_target_selection.py tests/core/test_ir_serialize.py tests/core/test_xf32_bf16x3_lowering.py
step py_tools env PYTHONPATH=python:$EXTRA python -m pytest -p no:cacheprovider -q -rs \
    tests/tools/test_isa_tables.py tests/tools/test_isa_xcheck.py tests/tools/test_isa_burs.py

# 2. Q38 pilot tests (strict: no skips) and the touched framework's own tests.
cd "$RUST/library" || exit 126
step q38_pilot env PYTHONPATH=$(rpath "$RUST") Q38_DIRECT_STRICT=1 python -m pytest -p no:cacheprovider -q -rs \
    tests/test_qwen38_direct_pilot.py
step q38_framework env PYTHONPATH=$(rpath "$RUST") python -m pytest -p no:cacheprovider -q -rs \
    tests/test_qwen38_layer_framework.py tests/test_qwen38_stage_adapt.py

# 3. Knob-off byte identity: mono twice (A-vs-A control, separate processes), then the rust tree.
P=$RUST/library/kernels/gfx950/qwen38_mono/direct_pilot.py
for t in mono_a mono_b rust; do
    tree=$MONO; [ $t = rust ] && tree=$RUST
    step dump_$t env PYTHONPATH=$(rpath "$tree") python "$TOOLS/q38_identity.py" dump --pilot "$P" --out "$O/identity_$t.json"
done
step identity_control python "$TOOLS/q38_identity.py" compare "$O/identity_mono_a.json" "$O/identity_mono_b.json"
step identity python "$TOOLS/q38_identity.py" compare "$O/identity_mono_a.json" "$O/identity_rust.json"

# 4. hazard_corpus on the COMGR objects of the Q38 builds (diagnostic: unread instructions are
#    the R2 ISA-table work; a site where ours waits less than COMGR would be a recognizer bug).
cd "$O" || exit 126
env PYTHONPATH=$(rpath "$RUST") python - "$P" << 'EOF' > "$O/logs/hsaco_dump.txt" 2>&1
import importlib.util, sys
spec = importlib.util.spec_from_file_location("dp", sys.argv[1]); dp = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp)
from rocke.helpers.compile import compile_kernel
for label, build in dp.representative_builds():
    art = compile_kernel(build(), arch="gfx950", backend="python")
    open(f"hsaco/{label}.hsaco", "wb").write(art.hsaco)
    print(label, len(art.hsaco))
EOF
OBJDUMP=$(command -v llvm-objdump || echo /opt/rocm/llvm/bin/llvm-objdump)
for h in hsaco/*.hsaco; do
    b=$(basename "$h" .hsaco)
    "$OBJDUMP" -d --mcpu=gfx950 "$h" > "disasm/$b.s" 2> /dev/null
    "$W2R1_BIN/hazard_corpus" "logs/hazard_$b.tsv" "disasm/$b.s" > "logs/hazard_$b.txt" 2>&1
    echo "== hazard $b rc=$? $(tail -n 3 "logs/hazard_$b.txt" | tr '\n' ' ' | cut -c1-300)"
done
rm -rf "$O/hsaco"
echo "== W2R1 rc=$rc"
exit $rc
