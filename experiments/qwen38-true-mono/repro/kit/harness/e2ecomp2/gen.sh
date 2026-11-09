#!/usr/bin/env bash
# e2ecomp2: emit the NUMA-aware rocKE stage functions into $1 (inside the q38 container, rocKE stage on PYTHONPATH):
#   $1/b3   B3 + B3 prologue, stage_b3 --knobs tuned_numa (TUNED_SPECS_NUMA: knob numa=1, sidecars "numa1")
#   $1/moe  fused M3 (+ empty M4 / M5), stage_moe_fused --knobs best (BEST_KNOBS: numa=1, sidecars "numa1")
# and a sha256 manifest of every file. Both sets address the mailboxes of ATOM's numa=1 lever: run them with
# ATOM_Q38_SCHED=best (or numa=1); ATOM keeps the lever for builds that link only numa1 functions.
set -uo pipefail
FN=${1:?out dir}
R=/job/stages/${Q38_ROCKE:-rocke-e2ecomp2}
mkdir -p "$FN/b3" "$FN/moe"
export ROCKE_BACKEND=python
rc=0
(cd "$R/library" && ROCKE_LLVM_FLAVOR=llvm23 python -m kernels.gfx950.qwen38_mono.stage_b3 --out "$FN/b3" --knobs tuned_numa) \
  > "$FN/b3_gen.log" 2>&1 || { echo "GEN FAIL b3"; tail -20 "$FN/b3_gen.log"; rc=1; }
(cd "$R/library" && ROCKE_LLVM_FLAVOR=llvm23 python -m kernels.gfx950.qwen38_mono.stage_moe_fused --out "$FN/moe" \
  --variants r1,r2,r3,r4,r2s,r3s,r4s --knobs best) > "$FN/moe_gen.json" 2> "$FN/moe_gen.err" \
  || { echo "GEN FAIL moe"; tail -20 "$FN/moe_gen.err"; rc=1; }
(cd "$FN" && find b3 moe -type f | sort | xargs sha256sum > manifest.sha256)
echo "gen: b3: $(ls "$FN/b3" | grep -c '\.ll$') .ll ($(ls "$FN/b3" | grep '\.ll$' | tr '\n' ' '))"
echo "gen: moe: $(ls "$FN/moe" | grep -c '\.ll$') .ll"
echo "gen: sidecar layouts: $(cat "$FN"/b3/*.json "$FN"/moe/*.json | grep -o '"mailboxes": "[a-z0-9]*"' | sort | uniq -c | tr -s ' ')"
exit $rc
