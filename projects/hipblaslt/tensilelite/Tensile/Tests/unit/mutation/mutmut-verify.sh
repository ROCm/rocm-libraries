#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# mutmut-verify.sh — GENERIC, manifest-driven, STRICTLY SERIAL survivor kill-proof.
#
# This runner is generic over ANY mutmut survivor:
# it materializes each mutant either via `mutmut apply <id>` or by applying a
# normalized unified diff, runs ONE pytest node clean then mutated, and proves the
# KILL (test PASSES clean, FAILS mutated), reverting the source after each.
#
# Run only one instance at a time and never apply mutants concurrently: apply/run/
# revert touches the shared worktree. A trap reverts the active target on exit.
#
# Usage:
#   mutmut-verify.sh --container tl-mut --manifest <rows.tsv> --out <dir> \
#                       [--src projects/hipblaslt/tensilelite] [--root <worktree>]
#
# Manifest: TSV with a header line, then one row per mutant:
#   mutant_id <TAB> file <TAB> apply_method <TAB> test_node <TAB> \
#       expect_clean_rc <TAB> expect_mutant_rc_nonzero
#     file          : path relative to --src, e.g. Tensile/Common/Utilities.py
#     apply_method  : "mutmut_apply"  -> docker exec ... mutmut apply <mutant_id>
#                     "diff:<abspath>" -> git -C <src> apply <abspath>   (host side)
#     test_node     : pytest node relative to the in-container project dir, e.g.
#                     Tensile/Tests/unit/characterization/CommonUtilities/test_mut_X_char.py::test_Y
#     expect_clean_rc          : usually 0 (test must PASS on clean source)
#     expect_mutant_rc_nonzero : true|false. true (normal kill): mutant node must
#                                FAIL with pytest ASSERTION-FAILURE rc==1. false:
#                                node must still pass rc==0.
#
# Output: <out>/kill_matrix.tsv  (one row per manifest row) + <out>/verify-report.txt
# Verdict KILLED iff base_rc == expect_clean_rc AND revert == ok
#   AND (when expect_mutant_rc_nonzero=true) the mutant node fails with rc==1.
#   rc==0 => not killed (survived). rc in {2,3,4,5,...} => INCONCLUSIVE
#   (collection/usage/internal/interrupt error), NOT a kill. Any non-KILLED row
#   makes the run FAILURE. (Previously any non-zero rc was wrongly counted a kill.)

set -u

# ----------------------------------------------------------------- pure classify
# classify_verdict — strict, pure, side-effect-free kill classification.
#   args: base_rc exp_clean mut_rc want_fail(true|false) revert(ok|LEAK)
#   echoes: "<verdict>\t<detail>"  where verdict in {KILLED, BAD, INCONCLUSIVE}
# KILLED requires the mutant node to FAIL with pytest ASSERTION-FAILURE rc==1
# (when want_fail=true). rc==0 => survived (not killed). rc in {2,3,4,5,...} =>
# INCONCLUSIVE (collection/usage/internal/interrupt error), NEVER a kill.
classify_verdict() {
  local base_rc="$1" exp_clean="$2" mut_rc="$3" want_fail="$4" revert="$5"
  # numeric guard: rc/expected fields must be integers, else BAD (fail-closed;
  # avoids a set -u arithmetic abort on a malformed manifest column).
  local _int='^-?[0-9]+$'
  if ! [[ "$base_rc" =~ $_int && "$exp_clean" =~ $_int && "$mut_rc" =~ $_int ]]; then
    printf 'BAD\tnon-numeric rc field (base_rc=%s exp_clean=%s mut_rc=%s)' "$base_rc" "$exp_clean" "$mut_rc"; return
  fi
  if [[ "$base_rc" -ne "$exp_clean" ]]; then
    printf 'BAD\tbase_rc=%s != expected clean %s' "$base_rc" "$exp_clean"; return
  fi
  if [[ "$revert" != "ok" ]]; then
    printf 'BAD\trevert=%s base_rc=%s mut_rc=%s' "$revert" "$base_rc" "$mut_rc"; return
  fi
  if [[ "$want_fail" == "true" ]]; then
    if [[ "$mut_rc" -eq 1 ]]; then
      printf 'KILLED\tbase_rc=%s mut_rc=1' "$base_rc"
    elif [[ "$mut_rc" -eq 0 ]]; then
      printf 'BAD\tsurvived: mut_rc=0 (test passed under mutant)'
    else
      printf 'INCONCLUSIVE\tmut_rc=%s (collection/usage/internal error, not an assertion failure)' "$mut_rc"
    fi
  else
    # expected-pass mode (rare; expect_mutant_rc_nonzero=false): NOT a kill.
    # KILLED is reserved strictly for mut_rc==1, so an expected
    # pass gets a distinct non-kill success token (OK), never KILLED.
    if [[ "$mut_rc" -eq 0 ]]; then
      printf 'OK\tbase_rc=%s mut_rc=0 (expected pass; not a kill)' "$base_rc"
    else
      printf 'BAD\tmut_rc=%s (expected 0)' "$mut_rc"
    fi
  fi
}

# When sourced for unit selftests (MUTMUT_VERIFY_LIB_ONLY set), stop here so the
# pure helpers are available without running the CLI/main.
if [[ -n "${MUTMUT_VERIFY_LIB_ONLY:-}" ]]; then return 0 2>/dev/null || exit 0; fi

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
ROOT="$(readlink -f -- "$ROOT")"
SRC="$ROOT/$SRC_REL"             # host-side source dir (git target)
PROJ="/work/$SRC_REL"            # in-container project dir (bind-mounted == SRC)
[[ -d "$SRC" ]] || { echo "src dir not found: $SRC" >&2; exit 2; }
docker inspect "$CON" >/dev/null 2>&1 || { echo "container not found: $CON" >&2; exit 2; }

# Applying and restoring a mutant must operate on the same tree. In particular,
# --root must not select one worktree while the container's /work bind mount
# points at another one.
IFS=$'\t' read -r CONTAINER_ROOT CONTAINER_ROOT_RW < <(
  docker inspect "$CON" \
    --format '{{range .Mounts}}{{if eq .Destination "/work"}}{{printf "%s\t%t" .Source .RW}}{{end}}{{end}}'
)
[[ -n "$CONTAINER_ROOT" ]] || {
  echo "container $CON has no bind mount at /work; recreate it with --mount type=bind,source=$ROOT,target=/work" >&2
  exit 2
}
CONTAINER_ROOT="$(readlink -f -- "$CONTAINER_ROOT")"
[[ "$CONTAINER_ROOT" == "$ROOT" ]] || {
  echo "container /work mount mismatch: container uses $CONTAINER_ROOT, but --root resolves to $ROOT" >&2
  exit 2
}
[[ "$CONTAINER_ROOT_RW" == "true" ]] || {
  echo "container /work mount is read-only; mutation verification requires a read-write bind mount" >&2
  exit 2
}

mkdir -p "$OUT"
KM="$OUT/kill_matrix.tsv"
REPORT="$OUT/verify-report.txt"

# ----------------------------------------------------------------- manifest safety
EXPECTED_HEADER=$'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero'
IFS= read -r MANIFEST_HEADER < "$MANIFEST"
[[ "$MANIFEST_HEADER" == "$EXPECTED_HEADER" ]] || {
  echo "invalid manifest header; expected: $EXPECTED_HEADER" >&2
  exit 2
}

# Reject every unsafe target before installing the cleanup trap. The trap only
# restores the file currently being mutated, which was proven clean beforehand;
# it can therefore never discard a pre-existing edit from another manifest row.
mapfile -t TARGETS < <(awk -F'\t' 'NR>1 && $2!="" {print $2}' "$MANIFEST" | sort -u)
for f in "${TARGETS[@]}"; do
  case "$f" in
    /*|..|../*|*/../*|*/..) echo "unsafe manifest file path: $f" >&2; exit 2 ;;
  esac
  git -C "$SRC" ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || {
    echo "manifest file is not tracked under $SRC: $f" >&2; exit 2; }
  git -C "$SRC" diff --quiet HEAD -- "$f" || {
    echo "manifest target is dirty before verification: $f" >&2; exit 2; }
done

ACTIVE_FILE=""
revert_active() {
  [[ -z "$ACTIVE_FILE" ]] && return 0
  git -C "$SRC" checkout -- "$ACTIVE_FILE" 2>/dev/null || return 1
  ACTIVE_FILE=""
}
trap 'revert_active' EXIT

# ----------------------------------------------------------------- helpers
run_node() { # $1=test_node -> echoes rc
  docker exec -e PYTHONPATH="$PROJ" -w "$PROJ" "$CON" \
    pytest -p no:cacheprovider -m unit -q "$1" >/dev/null 2>&1 </dev/null
  echo $?
}
apply_mutant() { # $1=method $2=mutant_id $3=file -> rc (0 ok)
  local method="$1" mid="$2" file="$3"
  case "$method" in
    mutmut_apply)
      docker exec -w "$PROJ" "$CON" mutmut apply "$mid" >/dev/null 2>&1 </dev/null ;;
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
  while IFS=$'\t' read -r mid file method node exp_clean exp_mut_nz extra; do
    [[ -z "${mid:-}" ]] && continue
    detail=""; base_rc=-1; mut_rc=-1; revert="-"; verdict="BAD"

    if [[ -n "${extra:-}" ]]; then
      detail="unexpected-extra-manifest-column"; verdict="BAD"; overall_ok=0
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "$verdict" "$detail" >> "$KM"
      printf "%-28s %-8s %s\n" "$mid" "$verdict" "$detail" | tee -a "$REPORT"; continue
    fi
    if [[ "${exp_mut_nz:-true}" != "true" && "${exp_mut_nz:-true}" != "false" ]]; then
      detail="expect_mutant_rc_nonzero-must-be-true-or-false"; verdict="BAD"; overall_ok=0
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "$verdict" "$detail" >> "$KM"
      printf "%-28s %-8s %s\n" "$mid" "$verdict" "$detail" | tee -a "$REPORT"; continue
    fi

    # 0) assert clean before
    if ! git -C "$SRC" diff --quiet HEAD -- "$file"; then
      detail="dirty-before-apply"; verdict="BAD"; overall_ok=0
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "$verdict" "$detail" >> "$KM"
      printf "%-28s %-8s %s\n" "$mid" "$verdict" "$detail" | tee -a "$REPORT"; continue
    fi

    # 1) baseline: node on CLEAN source (must PASS == expect_clean_rc)
    base_rc=$(run_node "$node")

    # 2) materialize mutant
    ACTIVE_FILE="$file"
    if ! apply_mutant "$method" "$mid" "$file"; then
      revert_active || true
      detail="apply-failed ($method)"; verdict="BAD"; overall_ok=0
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "$verdict" "$detail" >> "$KM"
      printf "%-28s %-8s %s\n" "$mid" "$verdict" "$detail" | tee -a "$REPORT"; continue
    fi

    # 3) mutated: node must FAIL
    mut_rc=$(run_node "$node")

    # 4) revert + assert clean
    revert_active || true
    if git -C "$SRC" diff --quiet HEAD -- "$file"; then revert="ok"; else revert="LEAK"; overall_ok=0; fi

    # 5) classify (STRICT: KILLED requires assertion-failure rc==1, not any non-zero)
    IFS=$'\t' read -r verdict detail < <(classify_verdict "$base_rc" "${exp_clean:-0}" "$mut_rc" "${exp_mut_nz:-true}" "$revert")
    # KILLED (mut_rc==1) and OK (expected-pass mode) are the non-failing verdicts;
    # BAD and INCONCLUSIVE both fail the run.
    [[ "$verdict" == "KILLED" || "$verdict" == "OK" ]] || overall_ok=0
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
