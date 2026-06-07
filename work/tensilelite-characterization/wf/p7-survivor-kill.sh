#!/usr/bin/env bash
# P7 survivor-kill verification — STRICTLY SERIAL, trap-protected, in the campaign
# worktree. Proves the two assertion-strengthening tests added for the P6 survivors
# now KILL their mutants: for each mutant, the NEW test must FAIL while mutated and
# PASS after revert. Mirrors wf/p6-mutation.sh (serial-with-guaranteed-revert; the
# tl-char container is bound to this one worktree). Never committed; tree clean at end.
set -u
ROOT=/home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage
PROJ=/work/projects/hipblaslt/tensilelite
SRC=$ROOT/projects/hipblaslt/tensilelite
CON=tl-char
OUT=$ROOT/work/tensilelite-characterization/coverage/p7
mkdir -p "$OUT"
cd "$SRC"

TARGETS=(Tensile/Activation.py Tensile/SolutionStructs/Solution.py)
revert_all() { for f in "${TARGETS[@]}"; do git checkout -- "$f" 2>/dev/null; done; }
trap 'revert_all' EXIT

CHAR=Tensile/Tests/unit/characterization
# id | file | search | replace | new-test (must now FAIL under the mutant)
MUTANTS=(
  "m3_activation_relu|Tensile/Activation.py|            module.add(VMaxF32(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprIn), src1=0, comment=\"x = max(0, x)\" ))|            module.add(VMaxF32(dst=self.vgprPrefix(vgprOut), src0=self.vgprPrefix(vgprIn), src1=1, comment=\"x = max(0, x)\" ))|$CHAR/Activation/test_r4_activation2_char.py::test_relu_clamp_floor_is_zero"
  "m4_sol_autolrvw|Tensile/SolutionStructs/Solution.py|          autoLRVW = False|          autoLRVW = True|$CHAR/SolutionDerivation/test_r5_autolrvw_char.py::test_gfx950_mx_fp8_explicit_wide_lrvw_preserved"
)

REPORT="$OUT/survivor-kill-report.txt"
echo "P7 survivor-kill verification — $(git rev-parse --short HEAD)" | tee "$REPORT"
echo "============================================================" | tee -a "$REPORT"
printf "%-20s %-9s %s\n" "MUTANT" "VERDICT" "DETAIL" | tee -a "$REPORT"

apply_mutant() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys
f,search,replace=sys.argv[1],sys.argv[2],sys.argv[3]
s=open(f).read(); n=s.count(search)
if n!=1:
    print("FAIL: search matched %d times (need 1)"%n); sys.exit(3)
open(f,"w").write(s.replace(search,replace,1)); print("OK")
PY
}

overall_ok=1
for spec in "${MUTANTS[@]}"; do
  IFS='|' read -r id file search replace test <<<"$spec"
  base=$(basename "$file")
  if ! git diff --quiet -- "$file"; then echo "$id ABORT: $file dirty before apply" | tee -a "$REPORT"; overall_ok=0; continue; fi

  # 1) baseline: new test PASSES on clean source
  blog="$OUT/${id}.baseline.log"
  docker exec -e PYTHONPATH=$PROJ -w $PROJ $CON pytest -p no:cacheprovider -m unit -q "$test" > "$blog" 2>&1
  base_rc=$?

  # 2) apply mutant
  ap=$(apply_mutant "$file" "$search" "$replace")
  if [[ "$ap" != OK ]]; then printf "%-20s %-9s %s\n" "$id" "SKIP" "$ap ($base)" | tee -a "$REPORT"; git checkout -- "$file"; overall_ok=0; continue; fi

  # 3) mutated: new test must FAIL
  mlog="$OUT/${id}.mutated.log"
  docker exec -e PYTHONPATH=$PROJ -w $PROJ $CON pytest -p no:cacheprovider -m unit -q "$test" > "$mlog" 2>&1
  mut_rc=$?

  # 4) revert + assert clean
  git checkout -- "$file"
  clean_after=ok; git diff --quiet -- "$file" || clean_after=LEAK

  msum=$(grep -E "passed|failed|error" "$mlog" | tail -1)
  if [[ $base_rc -eq 0 && $mut_rc -ne 0 && "$clean_after" == ok ]]; then
    verdict=KILLED
  else
    verdict=BAD; overall_ok=0
  fi
  printf "%-20s %-9s %s\n" "$id" "$verdict" "base_rc=$base_rc mut_rc=$mut_rc revert=$clean_after | $msum" | tee -a "$REPORT"
done

echo "============================================================" | tee -a "$REPORT"
# Leak check: only NON-TEST source (the mutated files) must be clean. The two
# campaign test files we add assertions to legitimately show as modified and are
# excluded, as is the pre-existing config_helpers.py change.
leak=$(git status --porcelain -- 'Tensile/*.py' 'Tensile/**/*.py' \
       | grep -vE "config_helpers.py|/Tests/" | grep -E "^ ?M" || true)
if [[ -n "$leak" ]]; then echo "MUTATION LEAK DETECTED:" | tee -a "$REPORT"; echo "$leak" | tee -a "$REPORT"; overall_ok=0;
else echo "CLEAN: no non-test source leak (mutated files reverted)." | tee -a "$REPORT"; fi
[[ $overall_ok -eq 1 ]] && echo "P7_RESULT: ALL KILLED" | tee -a "$REPORT" || echo "P7_RESULT: FAILURE (see above)" | tee -a "$REPORT"
