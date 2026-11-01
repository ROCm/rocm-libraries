#!/bin/bash
# R2 reference job (rocke node, q38_job.sh --mode container: the hybrid's ROCm 10 COMGR): COMGR objects of the R2
# pilots (asm and irops forms), their disassembly and kernel descriptors, and the asm-vs-irops .text identity.
#   q38_job.sh --mode container --stage w2-r2-rust --stage w2-r2-tools -- bash $Q38_STAGES/w2-r2-tools/jobs/w2r2_ref.sh
set -uo pipefail
RUST=$Q38_STAGES/w2-r2-rust
TOOLS=$Q38_STAGES/w2-r2-tools
O=$Q38_OUT
mkdir -p "$O/obj" "$O/logs"
export PYTHONPATH=$RUST/platform/python:$RUST/library ROCKE_BACKEND=python PYTHONDONTWRITEBYTECODE=1
LLVM=$(python -c 'import _rocm_sdk_devel,os;print(os.path.join(os.path.dirname(_rocm_sdk_devel.__file__),"lib","llvm","bin"))' 2> /dev/null)
[ -x "$LLVM/llvm-objdump" ] || LLVM=/work/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin
echo "== node $(hostname) llvm=$LLVM flavor=${ROCKE_LLVM_FLAVOR:-auto}"
python "$TOOLS/r2_build.py" --out "$O/obj" --backends comgr > "$O/logs/build.txt" 2>&1
echo "== build rc=$? $(tail -n 3 "$O/logs/build.txt" | tr '\n' ' ')"
cd "$O/obj" || exit 126
for h in *.comgr.hsaco; do
    b=${h%.hsaco}
    "$LLVM/llvm-objdump" -d --mcpu=gfx950 "$h" > "$b.s" 2> /dev/null
    "$LLVM/llvm-readelf" --notes -s "$h" > "$b.notes" 2> /dev/null
    "$LLVM/llvm-objdump" -s -j .rodata "$h" > "$b.rodata" 2> /dev/null
    "$LLVM/llvm-objcopy" -O binary --only-section=.text "$h" "$b.text" 2> /dev/null
done
for p in head_hybrid toy_best_d4w8_l2 toy_best_d8w0; do
    if cmp -s "$p.comgr.text" "${p}_irops.comgr.text"; then echo "== text $p vs irops: IDENTICAL"; else echo "== text $p vs irops: differ"; fi
done
grep -h "kernarg_preload\|\.sgpr_count\|\.vgpr_count\|user_sgpr" *.notes | sort | uniq -c | head -40
echo "== W2R2REF done"
