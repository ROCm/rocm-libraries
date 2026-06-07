#!/usr/bin/env bash
# mutmut-verify.sh — GENERIC, manifest-driven, STRICTLY SERIAL survivor kill-proof.
#
# Replaces the hardcoded wf/p7-survivor-kill.sh (which pinned two mutants/files and
# applied search/replace strings). This runner is generic over ANY mutmut survivor:
# it materializes each mutant either via `mutmut apply <id>` or by applying a
# normalized unified diff, runs ONE pytest node clean then mutated, and proves the
# KILL (test PASSES clean, FAILS mutated), reverting the source after each.
#
# It is the single serial actor for mutation APPLICATION (see PLAN concurrency
# rule): never run two of these, and never apply mutants concurrently — apply/run/
# revert touches the shared worktree. A trap reverts every target file on exit.
#
# Usage:
#   wf/mutmut-verify.sh --container tl-mut --manifest <rows.tsv> --out <dir> \
#                       [--src projects/hipblaslt/tensilelite] [--root <worktree>]
#
# Manifest: TSV with a header line, then one row per mutant:
#   mutant_id <TAB> file <TAB> apply_method <TAB> test_node <TAB> \
#       expect_clean_rc <TAB> expect_mutant_rc_nonzero <TAB> revert_assert
#     file          : path relative to --src, e.g. Tensile/Common/Utilities.py
#     apply_method  : "mutmut_apply"  -> docker exec ... mutmut apply <mutant_id>
#                     "diff:<abspath>" -> git -C <src> apply <abspath>   (host side)
#     test_node     : pytest node relative to the in-container project dir, e.g.
#                     Tensile/Tests/unit/characterization/CommonUtilities/test_mut_X_char.py::test_Y
#     expect_clean_rc          : usually 0 (test must PASS on clean source)
#     expect_mutant_rc_nonzero : true|false (test must FAIL under the mutant)
#     revert_assert            : true|false (assert file clean after revert)
#
# Output: <out>/kill_matrix.tsv  (one row per manifest row) + <out>/verify-report.txt
# Verdict KILLED iff: base_rc == expect_clean_rc AND (mut_rc!=0) == expect_mutant_rc_nonzero
#                     AND revert == ok. Otherwise BAD (with a reason).

set -u

# ----------------------------------------------------------------- args
CON="" ; MANIFEST="" ; OUT="" ; SRC_REL="projects/hipblaslt/tensilelite" ; ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --container) CON="$2"; shift 2;;
    --manifest)  MANIFEST="$2"; shift 2;;
    --out)       OUT="$2"; shift 2;;
    --src)       SRC_REL="$2"; shift 2;;
    --root)      ROOT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -z "$CON" || -z "$MANIFEST" || -z "$OUT" ]] && {
  echo "usage: $0 --container <name> --manifest <tsv> --out <dir> [--src <rel>] [--root <worktree>]" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }

# Derive worktree root from git if not given (so the script is location-independent).
if [[ -z "$ROOT" ]]; then ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; fi
[[ -n "$ROOT" ]] || { echo "could not determine --root (not in a git worktree?)" >&2; exit 2; }
SRC="$ROOT/$SRC_REL"             # host-side source dir (git target)
PROJ="/work/$SRC_REL"            # in-container project dir (bind-mounted == SRC)
[[ -d "$SRC" ]] || { echo "src dir not found: $SRC" >&2; exit 2; }
docker inspect "$CON" >/dev/null 2>&1 || { echo "container not found: $CON" >&2; exit 2; }

mkdir -p "$OUT"
KM="$OUT/kill_matrix.tsv"
REPORT="$OUT/verify-report.txt"

# ----------------------------------------------------------------- trap revert
# Collect every target file from the manifest so the trap can restore them all,
# even on crash/ctrl-C. Skip the header row (col1 == "mutant_id").
mapfile -t TARGETS < <(awk -F'\t' 'NR>1 && $1!="mutant_id" && $2!="" {print $2}' "$MANIFEST" | sort -u)
revert_all() { for f in "${TARGETS[@]:-}"; do [[ -n "$f" ]] && git -C "$SRC" checkout -- "$f" 2>/dev/null; done; }
trap 'revert_all' EXIT

# ----------------------------------------------------------------- helpers
run_node() { # $1=test_node -> echoes rc
  docker exec -e PYTHONPATH="$PROJ" -w "$PROJ" "$CON" \
    pytest -p no:cacheprovider -m unit -q "$1" >/dev/null 2>&1
  echo $?
}
apply_mutant() { # $1=method $2=mutant_id $3=file -> rc (0 ok)
  local method="$1" mid="$2" file="$3"
  case "$method" in
    mutmut_apply)
      docker exec -w "$PROJ" "$CON" mutmut apply "$mid" >/dev/null 2>&1 ;;
    diff:*)
      git -C "$SRC" apply "${method#diff:}" >/dev/null 2>&1 ;;
    *) return 90 ;;   # unknown method
  esac
}

# ----------------------------------------------------------------- header
echo "mutmut-verify — $(git -C "$SRC" rev-parse --short HEAD) — $(date -u +%FT%TZ)" | tee "$REPORT"
echo "container=$CON  src=$SRC  proj=$PROJ" | tee -a "$REPORT"
echo "============================================================" | tee -a "$REPORT"
printf "mutant_id\tfile\tbase_rc\tmut_rc\trevert\tverdict\tdetail\n" > "$KM"
printf "%-28s %-8s %s\n" "MUTANT" "VERDICT" "DETAIL" | tee -a "$REPORT"

overall_ok=1
# ----------------------------------------------------------------- main loop (SERIAL)
# Read manifest, skipping header. IFS=tab.
{
  read -r _hdr   # discard header
  while IFS=$'\t' read -r mid file method node exp_clean exp_mut_nz rev_assert; do
    [[ -z "${mid:-}" ]] && continue
    detail=""; base_rc=-1; mut_rc=-1; revert="-"; verdict="BAD"

    # 0) assert clean before
    if ! git -C "$SRC" diff --quiet -- "$file"; then
      detail="dirty-before-apply"; verdict="BAD"; overall_ok=0
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "$verdict" "$detail" >> "$KM"
      printf "%-28s %-8s %s\n" "$mid" "$verdict" "$detail" | tee -a "$REPORT"; continue
    fi

    # 1) baseline: node on CLEAN source (must PASS == expect_clean_rc)
    base_rc=$(run_node "$node")

    # 2) materialize mutant
    if ! apply_mutant "$method" "$mid" "$file"; then
      git -C "$SRC" checkout -- "$file" 2>/dev/null
      detail="apply-failed ($method)"; verdict="BAD"; overall_ok=0
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "$verdict" "$detail" >> "$KM"
      printf "%-28s %-8s %s\n" "$mid" "$verdict" "$detail" | tee -a "$REPORT"; continue
    fi

    # 3) mutated: node must FAIL
    mut_rc=$(run_node "$node")

    # 4) revert + assert clean
    git -C "$SRC" checkout -- "$file" 2>/dev/null
    if git -C "$SRC" diff --quiet -- "$file"; then revert="ok"; else revert="LEAK"; overall_ok=0; fi

    # 5) classify
    mut_nz="false"; [[ "$mut_rc" -ne 0 ]] && mut_nz="true"
    if [[ "$base_rc" -eq "${exp_clean:-0}" && "$mut_nz" == "${exp_mut_nz:-true}" && "$revert" == "ok" ]]; then
      verdict="KILLED"; detail="base_rc=$base_rc mut_rc=$mut_rc"
    else
      verdict="BAD"; detail="base_rc=$base_rc mut_rc=$mut_rc revert=$revert (expected clean=${exp_clean:-0} mut_nz=${exp_mut_nz:-true})"; overall_ok=0
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "$verdict" "$detail" >> "$KM"
    printf "%-28s %-8s %s\n" "$mid" "$verdict" "$detail" | tee -a "$REPORT"
  done
} < "$MANIFEST"

echo "============================================================" | tee -a "$REPORT"
# Final leak assertion: no tracked source .py modified beyond the pre-existing
# config_helpers.py (add-only test files are NEW/untracked, so they don't show as ' M').
leak=$(git -C "$SRC" status --porcelain -- 'Tensile/*.py' 'Tensile/**/*.py' \
       | grep -vE 'config_helpers.py' | grep -E '^ ?M' || true)
if [[ -n "$leak" ]]; then echo "LEAK DETECTED:" | tee -a "$REPORT"; echo "$leak" | tee -a "$REPORT"; overall_ok=0;
else echo "CLEAN: no mutated-source leak." | tee -a "$REPORT"; fi

[[ $overall_ok -eq 1 ]] && echo "RESULT: ALL KILLED" | tee -a "$REPORT" || echo "RESULT: FAILURE (see above)" | tee -a "$REPORT"
echo "kill_matrix: $KM"
[[ $overall_ok -eq 1 ]]
