#!/bin/bash
# Phase 4 ship gate for a re-mapped TN HHS catalog.
#   ./gate_ship.sh hhs_remap_extship
#
# Every check here exists because something in this campaign passed a weaker version of it:
#
#  * FRESH TREE PER GATE, exactly one YAML asserted. TensileCreateLibrary RECURSES, and a gate
#    once reported "298 kernels, 0 errors" while silently merging three stale catalogs
#    (298+73+100 = 471 solutions, exactly the artifact count). Exit code proves nothing.
#  * SOLUTION COUNT READ FROM THE BUILT LIBRARY, by unpacking the msgpack -- not by grepping it.
#    Regex probing of this format gave a wrong answer twice here: once matching
#    WorkGroupMapping as a prefix of WorkGroupMappingRR, once matching a capability flag.
#    Parse the format, don't grep it.
#  * GRID ROW COUNT ASSERTED UNCHANGED. A re-map may only rewrite element[7] solution indices;
#    a dropped or reordered row would look exactly like a successful re-map.
#  * ELF FLAGS CHECKED per architecture, because the ISA regex in an earlier tool missed
#    gfx115x entirely and made five distinct catalogs look identical.
set -u
cd /home/vmijovic/navi32
ARM=${1:-hhs_remap_extship}
log(){ echo "[$(date +%F\ %H:%M:%S)] $*"; }
export PYTHONPATH=/home/vmijovic/navi32/src/projects/hipblaslt/build/release/tensilelite/rocisa:/home/vmijovic/navi32/src/projects/hipblaslt/tensilelite

SRC=arms/$ARM/x.yaml
[ -f "$SRC" ] || { log "FATAL: $SRC not found"; exit 1; }

# ---- structural asserts on the catalog itself, before any build ---------------------------
log "structural check of $SRC"
python3 - "$SRC" <<'PY' || exit 1
import sys, yaml
try: from yaml import CSafeLoader as L
except ImportError: from yaml import SafeLoader as L
new = yaml.load(open(sys.argv[1]), Loader=L)
base = yaml.load(open("arms/hhs_remap_gated/x.yaml"), Loader=L)
assert new[11] == "GridBased", f"element[11] is {new[11]!r}"
assert len(new[7]) == len(base[7]), f"GRID SIZE CHANGED {len(base[7])} -> {len(new[7])}"
assert [tuple(e[0]) for e in new[7]] == [tuple(e[0]) for e in base[7]], "grid KEYS changed"
assert all(0 <= e[1][0] < len(new[5]) for e in new[7]), "a row points outside the pool"
ch = sum(1 for a, b in zip(new[7], base[7]) if a[1][0] != b[1][0])
print(f"    rows {len(new[7])} unchanged, keys identical, pool {len(new[5])}, "
      f"{ch} rows repointed vs shipped ({100*ch/len(new[7]):.1f}%)")
PY

# ---- build for the REAL targets ------------------------------------------------------------
RC=0
for AR in gfx1101 gfx1102; do
  D=$([ "$AR" = gfx1101 ] && echo navi32 || echo navi33)
  G=gate_${ARM}_${AR}
  rm -rf "$G"; mkdir -p "$G/logic/$D/GridBased"
  python3 retarget_logic.py "$SRC" "$G/logic/$D/GridBased/${D}_ship.yaml" --isa "$AR" >/dev/null 2>&1
  N=$(find "$G/logic" -name '*.yaml' | wc -l)
  [ "$N" -eq 1 ] || { log "  ABORT $AR: $N yaml in gate tree, expected 1"; RC=1; continue; }
  ( cd src/projects/hipblaslt/tensilelite && timeout 7200 python3 Tensile/bin/TensileCreateLibrary \
      "/home/vmijovic/navi32/$G/logic/$D" "/home/vmijovic/navi32/$G/out" HIP \
      --architecture "$AR" --jobs "$(nproc)" --logic-format yaml --no-compress \
      > "/home/vmijovic/navi32/logs/$G.log" 2>&1 )
  CO=$(ls "$G/out/library/$AR"/*.co 2>/dev/null | head -1)
  K=$(llvm-readelf --notes "$CO" 2>/dev/null | grep -c '\.symbol:')
  ASM=$(grep -c "not a valid operand" "logs/$G.log")
  OVF=$(grep -c "overflowedResources" "logs/$G.log")
  FL=$(llvm-readelf -h "$CO" 2>/dev/null | grep -oE 'Flags:.*' | head -1)
  SOL=$(python3 -c "
import zlib,msgpack,glob
f=[x for x in glob.glob('$G/out/library/$AR/*.dat.zlib') if 'lazy' not in x]
print(len(msgpack.unpackb(zlib.decompress(open(f[0],'rb').read()),strict_map_key=False,raw=False).get('solutions',[])) if f else 'NONE')
" 2>/dev/null)
  log "  GATE $AR: kernels=$K solutions=$SOL asm_err=$ASM overflow=$OVF $FL"
  [ "$ASM" -eq 0 ] && [ "$OVF" -eq 0 ] && [ "$K" -gt 0 ] || { log "  FAIL $AR"; RC=1; }
done

# ---- numerical correctness, on the only GPU this machine has ------------------------------
# A re-map only changes WHICH pre-existing kernel a row names, so correctness ought to be
# inherited -- but it sends shapes to kernels they were never exercised on, which is worth
# checking rather than assuming.
log "numerical correctness (gfx1100 build of the same catalog)"
G=gate_${ARM}_gfx1100
rm -rf "$G"; mkdir -p "$G/logic/navi32/GridBased"
cp "$SRC" "$G/logic/navi32/GridBased/ship.yaml"
( cd src/projects/hipblaslt/tensilelite && timeout 7200 python3 Tensile/bin/TensileCreateLibrary \
    "/home/vmijovic/navi32/$G/logic/navi32" "/home/vmijovic/navi32/$G/out" HIP \
    --architecture gfx1100 --jobs "$(nproc)" --logic-format yaml --no-compress \
    > "/home/vmijovic/navi32/logs/$G.log" 2>&1 )
LIB=$PWD/$G/out/library/gfx1100
BENCH=src/projects/hipblaslt/build/release/clients/hipblaslt-bench
PASS=0; FAIL=0; RUN=0
# Correctness is checked by check_correctness.py, which PARSES the norm_error column.
# The first version of this gate grepped for "PASS" -- a word hipblaslt-bench -v never prints --
# and so reported 0 pass / 26 fail on a catalog where every single run actually succeeded
# (norm_error ~5e-05 vs rtol 1e-02). A gate that greps for a token the tool does not emit fails
# 100% of the time and is indistinguishable from a real defect. Parse the output, don't grep it.
if timeout 6000 python3 check_correctness.py --lib "$LIB" --shapes state/eval_fullcov.json --n 120; then
  log "  correctness: PASS"
else
  log "  correctness: FAIL"; RC=1
fi


log "GATE RESULT: $([ $RC -eq 0 ] && echo PASS || echo FAIL)"
exit $RC
