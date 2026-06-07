#!/usr/bin/env bash
# P6 mutation validation — STRICTLY SERIAL, trap-protected, in the campaign worktree
# (the tl-char container is bound to this worktree, so per-mutant throwaway worktrees are
# not visible in-container; serial-with-guaranteed-revert is the safe realization of P6).
# Each mutant: assert-clean -> apply one-line edit -> run coverage-selected subset in-container
# -> classify killed/survived -> REVERT -> assert-clean. Never committed. Tree clean at end.
set -u
ROOT=/home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage
PROJ=/work/projects/hipblaslt/tensilelite
SRC=$ROOT/projects/hipblaslt/tensilelite
CON=tl-char
OUT=$ROOT/work/tensilelite-characterization/coverage/p6
mkdir -p "$OUT"
cd "$SRC"

# Target source files (for the trap + final clean assertion). config_helpers.py is a
# pre-existing uncommitted change (NOT ours) and is excluded from the clean check.
TARGETS=(
  Tensile/TensileMergeLibrary.py
  Tensile/Activation.py
  Tensile/SolutionStructs/Solution.py
  Tensile/BenchmarkProblems.py
  Tensile/Components/StreamK.py
)
revert_all() { for f in "${TARGETS[@]}"; do git checkout -- "$f" 2>/dev/null; done; }
trap 'revert_all' EXIT

# mutant: id | file | python-search | python-replace | test-subset (in-container paths, space-sep)
CHAR=Tensile/Tests/unit/characterization
MUTANTS=(
  "m1_mergelib_count|Tensile/TensileMergeLibrary.py|    return newSizes, len(newSizes)|    return newSizes, 0|$CHAR/TensileMergeLibrary"
  "m2_mergelib_trim|Tensile/TensileMergeLibrary.py|        size = size[:-4] if len(size) >= 8 else size|        size = size[:-4] if len(size) >= 999 else size|$CHAR/TensileMergeLibrary"
  "m3_activation_relu|Tensile/Activation.py|            module.add(VMaxF32(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprIn), src1=0, comment=\"x = max(0, x)\" ))|            module.add(VMaxF32(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprIn), src1=1, comment=\"x = max(0, x)\" ))|$CHAR/Activation"
  "m4_sol_autolrvw|Tensile/SolutionStructs/Solution.py|          autoLRVW = False|          autoLRVW = True|$CHAR/SolutionDerivation $CHAR/SolutionArms $CHAR/SolutionEdges"
  "m5_benchcache|Tensile/BenchmarkProblems.py|    return all(cacheData[f] == getattr(benchmarkStep, attr) for f, attr in _CACHE_FIELDS.items())|    return not all(cacheData[f] == getattr(benchmarkStep, attr) for f, attr in _CACHE_FIELDS.items())|$CHAR/BenchmarkProblems"
  "m7_streamk_fixup|Tensile/Components/StreamK.py|        if kernel[\"StreamKFixupTreeReduction\"] == 1:|        if kernel[\"StreamKFixupTreeReduction\"] == 999:|$CHAR/_codegen/test_r7_streamk_grid_char.py $CHAR/_codegen/test_r2_streamk_char.py"
)

echo "P6 mutation validation — $(git rev-parse --short HEAD)" | tee "$OUT/mutation-report.txt"
echo "============================================================" | tee -a "$OUT/mutation-report.txt"
printf "%-20s %-9s %s\n" "MUTANT" "STATUS" "DETAIL" | tee -a "$OUT/mutation-report.txt"

apply_mutant() { # file search replace -> prints OK/FAIL
  python3 - "$1" "$2" "$3" <<'PY'
import sys
f,search,replace=sys.argv[1],sys.argv[2],sys.argv[3]
s=open(f).read()
n=s.count(search)
if n!=1:
    print("FAIL: search matched %d times (need 1)"%n); sys.exit(3)
open(f,"w").write(s.replace(search,replace,1))
print("OK")
PY
}

for spec in "${MUTANTS[@]}"; do
  IFS='|' read -r id file search replace subset <<<"$spec"
  base=$(basename "$file")
  # assert clean before
  if ! git diff --quiet -- "$file"; then echo "$id ABORT: $file dirty before apply" | tee -a "$OUT/mutation-report.txt"; continue; fi
  ap=$(apply_mutant "$file" "$search" "$replace")
  if [[ "$ap" != OK ]]; then printf "%-20s %-9s %s\n" "$id" "SKIP" "$ap ($base)" | tee -a "$OUT/mutation-report.txt"; git checkout -- "$file"; continue; fi
  # run the coverage-selected subset in-container (no --cov; we only need pass/fail)
  log="$OUT/${id}.log"
  docker exec -e PYTHONPATH=$PROJ -w $PROJ $CON pytest -p no:cacheprovider -m unit -q $subset > "$log" 2>&1
  rc=$?
  summary=$(grep -vE "RegisterPool|Warning" "$log" | grep -E "passed|failed|error" | tail -1)
  if [[ $rc -ne 0 ]]; then status=KILLED; else status=SURVIVED; fi
  printf "%-20s %-9s %s\n" "$id" "$status" "$summary" | tee -a "$OUT/mutation-report.txt"
  # REVERT + assert clean
  git checkout -- "$file"
  if ! git diff --quiet -- "$file"; then echo "$id LEAK: $file still dirty after revert!" | tee -a "$OUT/mutation-report.txt"; fi
done

echo "============================================================" | tee -a "$OUT/mutation-report.txt"
# FINAL clean assertion: no source .py modified by mutation (exclude pre-existing config_helpers.py)
leak=$(git status --porcelain -- 'Tensile/*.py' 'Tensile/**/*.py' | grep -vE "config_helpers.py" | grep -E "^ ?M" || true)
if [[ -n "$leak" ]]; then echo "MUTATION LEAK DETECTED:" | tee -a "$OUT/mutation-report.txt"; echo "$leak" | tee -a "$OUT/mutation-report.txt";
else echo "CLEAN: campaign worktree has no mutation leak (only pre-existing config_helpers.py modified)." | tee -a "$OUT/mutation-report.txt"; fi
echo "P6_DONE rc-summary above"
