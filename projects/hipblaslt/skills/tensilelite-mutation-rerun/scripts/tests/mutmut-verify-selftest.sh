#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Hermetic self-tests for mutmut-verify.sh. These tests use temporary Git
# repositories and a fake docker executable; they require neither Docker nor
# mutmut, pytest, ROCm, or a GPU.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="$HERE/../mutmut-verify.sh"
REAL_GIT="$(command -v git)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/mutmut-verify-selftest.XXXXXXXX")"
trap 'rm -rf -- "$TMP"' EXIT

tests=0
failures=0

pass() {
  ((tests += 1))
  printf 'ok %d - %s\n' "$tests" "$1"
}

fail() {
  ((tests += 1))
  ((failures += 1))
  printf 'not ok %d - %s\n' "$tests" "$1"
}

expect_eq() {
  local description="$1" expected="$2" actual="$3"
  if [[ "$actual" == "$expected" ]]; then pass "$description"; else
    fail "$description (expected '$expected', got '$actual')"
  fi
}

expect_contains() {
  local description="$1" file="$2" pattern="$3"
  if grep -Fq -- "$pattern" "$file"; then pass "$description"; else
    fail "$description (missing '$pattern' in $file)"
  fi
}

expect_not_contains() {
  local description="$1" file="$2" pattern="$3"
  if grep -Fq -- "$pattern" "$file"; then
    fail "$description (unexpected '$pattern' in $file)"
  else
    pass "$description"
  fi
}

expect_nonzero() {
  local description="$1" rc="$2"
  if [[ "$rc" -ne 0 ]]; then pass "$description"; else fail "$description (rc=0)"; fi
}

make_fake_commands() {
  mkdir -p "$TMP/bin"
  # The fake mutator is selected by mutant_id. Its writes model mutmut apply,
  # including bad metadata and partially successful commands.
  printf '%s\n' '#!/usr/bin/env bash' \
    'set -u' \
    'subcommand="${1:-}"; shift || true' \
    'case "$subcommand" in' \
    '  inspect)' \
    '    [[ "${FAKE_INSPECT_FAIL:-0}" == 1 ]] && exit 1' \
    '    formatted=0' \
    '    for arg in "$@"; do [[ "$arg" == "--format" ]] && formatted=1; done' \
    '    if [[ $formatted -eq 1 ]]; then' \
    '      case "${FAKE_MOUNT_MODE:-ok}" in' \
    '        missing) ;;' \
    '        mismatch) printf "%s\ttrue\n" "$FAKE_MISMATCH_ROOT" ;;' \
    '        readonly) printf "%s\tfalse\n" "$FAKE_ROOT" ;;' \
    '        *) printf "%s\ttrue\n" "$FAKE_ROOT" ;;' \
    '      esac' \
    '    fi' \
    '    exit 0' \
    '    ;;' \
    '  exec)' \
    '    while [[ $# -gt 0 ]]; do' \
    '      case "$1" in' \
    '        -e|-w) shift 2 ;;' \
    '        *) shift; break ;;' \
    '      esac' \
    '    done' \
    '    command="${1:-}"; shift || true' \
    '    if [[ "$command" == rm ]]; then' \
    '      command rm "$@" || exit $?' \
    '      rm_count=0' \
    '      [[ -f "$FAKE_RM_COUNT" ]] && rm_count="$(<"$FAKE_RM_COUNT")"' \
    '      ((rm_count += 1))' \
    '      printf "%s\n" "$rm_count" > "$FAKE_RM_COUNT"' \
    '      if [[ "${FAKE_RM_BLOCK_ON:-0}" -eq "$rm_count" ]]; then : > "$FAKE_RM_READY"; exec sleep 30; fi' \
    '      exit 0' \
    '    fi' \
    '    if [[ "$command" == sh && "${1:-}" == -c ]]; then' \
    '      state_file="${4:-}"' \
    '      marker="${5:-}"' \
    '      auth="${6:-}"' \
    '      if [[ "${6:-}" == __terminate__ ]]; then' \
    '        auth="$marker"' \
    '        [[ -r "$state_file" ]] || exit 42' \
    '        IFS=: read -r state seen_auth pid command_rc < "$state_file"' \
    '        [[ "$seen_auth" == "$auth" ]] || exit 49' \
    '        if [[ "$state" == running && "$pid" =~ ^[0-9]+$ ]]; then' \
    '          kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true' \
    '          for _ in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.01; done' \
    '          kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true' \
    '        elif [[ "$state" != done && "$state" != terminated ]]; then' \
    '          exit 44' \
    '        fi' \
    '        rm -f -- "$state_file"' \
    '        exit 0' \
    '      fi' \
    '      tool="${7:-}"' \
    '      count=0' \
    '      if [[ "$tool" == pytest ]]; then' \
    '        [[ -f "$FAKE_NODE_COUNT" ]] && count="$(<"$FAKE_NODE_COUNT")"' \
    '        ((count += 1))' \
    '        printf "%s\n" "$count" > "$FAKE_NODE_COUNT"' \
    '      fi' \
    '      if [[ "${FAKE_LAUNCH_FAIL_ON:-}" == "$tool" && "${FAKE_LAUNCH_FAIL_NODE:-0}" -eq "$count" ]]; then' \
    '        printf "launch-failed:%s:1\n" "$auth" > "$state_file"' \
    '        printf "%sINFRA:1\n" "$marker"' \
    '        exit 0' \
    '      fi' \
    '      printf "running:%s:%s\n" "$auth" "$$" > "$state_file"' \
    '      if [[ "$tool" == pytest ]]; then' \
    '        if [[ "${FAKE_SECOND_UNCONFIRMED:-0}" == 1 && $count -eq 2 ]]; then' \
    '          rm -f -- "$state_file"' \
    '          printf "unconfirmed transport failure\n"' \
    '          exit 1' \
    '        fi' \
    '        if [[ "${FAKE_SECOND_EXEC_FAIL:-0}" == 1 && $count -eq 2 ]]; then' \
    '          printf "done:%s:%s:1\n" "$auth" "$$" > "$state_file"' \
    '          printf "docker transport failed\n"' \
    '          exit 1' \
    '        fi' \
    '        if [[ "${FAKE_SECOND_MISSING_MARKER:-0}" == 1 && $count -eq 2 ]]; then' \
    '          printf "done:%s:%s:1\n" "$auth" "$$" > "$state_file"' \
    '          printf "wrapper ended without marker\n"' \
    '          exit 0' \
    '        fi' \
    '        clean=0' \
    '        "$REAL_GIT" -C "$FAKE_ROOT" diff --quiet -- . && clean=1' \
    '        if [[ $clean -eq 1 ]]; then' \
    '          case "${FAKE_CLEAN_ACTION:-none}" in' \
    '            write) printf "clean-test write\n" >> "$FAKE_SRC/other.py" ;;' \
    '            assume-write) "$REAL_GIT" -C "$FAKE_ROOT" update-index --assume-unchanged project/other.py; printf "hidden clean-test write\n" >> "$FAKE_SRC/other.py" ;;' \
    '            skip-write) "$REAL_GIT" -C "$FAKE_ROOT" update-index --skip-worktree project/other.py; printf "hidden clean-test write\n" >> "$FAKE_SRC/other.py" ;;' \
    '            hardlink-target) rm -f -- "$FAKE_SRC/target.py"; ln "$FAKE_HARDLINK_SENTINEL" "$FAKE_SRC/target.py" ;;' \
    '            signal-write) printf "clean-test write\n" >> "$FAKE_SRC/other.py"; kill -TERM "$PPID"; exit 143 ;;' \
    '            mutate-inputs) printf "replaced manifest\n" > "$FAKE_ORIGINAL_MANIFEST"; [[ -z "${FAKE_ORIGINAL_PATCH:-}" ]] || printf "replaced patch\n" > "$FAKE_ORIGINAL_PATCH" ;;' \
    '            corrupt-report) rm -f -- "$FAKE_REPORT"; ln -s /dev/full "$FAKE_REPORT" ;;' \
    '            corrupt-matrix) rm -f -- "$FAKE_MATRIX"; ln "$FAKE_ORIGINAL_MANIFEST" "$FAKE_MATRIX" ;;' \
    '            truncate-report) printf "overwritten in place\n" > "$FAKE_REPORT" ;;' \
    '          esac' \
    '          rc="${FAKE_BASE_RC:-0}"' \
    '        else' \
    '          case "${FAKE_MUTANT_ACTION:-none}" in' \
    '            write) printf "mutant-test write\n" >> "$FAKE_SRC/other.py" ;;' \
    '            same-target) printf "mutant-test same-target write\n" >> "$FAKE_SRC/target.py" ;;' \
    '            assume-write) "$REAL_GIT" -C "$FAKE_ROOT" update-index --assume-unchanged project/other.py; printf "hidden mutant-test write\n" >> "$FAKE_SRC/other.py" ;;' \
    '            skip-write) "$REAL_GIT" -C "$FAKE_ROOT" update-index --skip-worktree project/other.py; printf "hidden mutant-test write\n" >> "$FAKE_SRC/other.py" ;;' \
    '          esac' \
    '          rc="${FAKE_MUTANT_RC:-1}"' \
    '        fi' \
    '      elif [[ "$tool" == mutmut && "${8:-}" == apply ]]; then' \
    '        mutant="${9:-}"' \
    '        printf "%s\n" "$mutant" >> "$FAKE_APPLY_LOG"' \
    '        rc=0' \
    '        case "$mutant" in' \
    '          good|good2) printf "mutated\n" >> "$FAKE_SRC/target.py" ;;' \
    '          wrong) printf "mutated\n" >> "$FAKE_SRC/other.py" ;;' \
    '          outside) printf "mutated\n" >> "$FAKE_ROOT/outside.txt" ;;' \
    '          multi) printf "mutated\n" >> "$FAKE_SRC/target.py"; printf "mutated\n" >> "$FAKE_SRC/other.py" ;;' \
    '          new) printf "mutated\n" >> "$FAKE_SRC/target.py"; printf "new\n" > "$FAKE_SRC/new.py" ;;' \
    '          delete) rm -f -- "$FAKE_SRC/target.py" ;;' \
    '          rename) mv -- "$FAKE_SRC/target.py" "$FAKE_SRC/renamed.py" ;;' \
    '          partial) printf "mutated\n" >> "$FAKE_SRC/target.py"; printf "mutated\n" >> "$FAKE_SRC/other.py"; rc=17 ;;' \
    '          noop) ;;' \
    '          signal) printf "mutated\n" >> "$FAKE_SRC/target.py"; kill -TERM "$PPID"; exit 143 ;;' \
    '          block) printf "mutated\n" >> "$FAKE_SRC/target.py"; : > "$FAKE_BLOCK_READY"; exec sleep 30 ;;' \
    '          linger) printf "mutated\n" >> "$FAKE_SRC/target.py"; setsid sh -c '\''sleep 1; printf "delayed write\n" >> "$1"; sleep 30'\'' sh "$FAKE_SRC/target.py" & runner=$!; printf "running:%s:%s\n" "$auth" "$runner" > "$state_file"; : > "$FAKE_BLOCK_READY"; wait "$runner"; rc=$? ;;' \
    '          *) rc=19 ;;' \
    '        esac' \
    '      else' \
    '        exit 127' \
    '      fi' \
    '      printf "done:%s:%s:%s\n" "$auth" "$$" "$rc" > "$state_file"' \
    '      printf "fake command output rc=%s\n" "$rc"' \
    '      printf "%s%s\n" "$marker" "$rc"' \
    '      exit 0' \
    '    fi' \
    '    exit 127' \
    '    ;;' \
    'esac' \
    'exit 127' > "$TMP/bin/docker"

  # Delegates every Git operation except for an optional one-shot restore
  # failure. This exercises both the reported LEAK and the EXIT-trap retry.
  printf '%s\n' '#!/usr/bin/env bash' \
    'set -u' \
    'if [[ "${FAKE_GIT_FAIL_RESTORE:-0}" == 1 && " $* " == *" restore "* && ! -e "$FAKE_RESTORE_FAILED" ]]; then' \
    '  : > "$FAKE_RESTORE_FAILED"' \
    '  exit 1' \
    'fi' \
    'exec "$REAL_GIT" "$@"' > "$TMP/bin/git"
  chmod +x "$TMP/bin/docker" "$TMP/bin/git"
}

make_fake_commands

setup_case() {
  CASE_DIR="$(mktemp -d "$TMP/case.XXXXXXXX")"
  REPO="$CASE_DIR/repo"
  SRC="$REPO/project"
  MANIFEST="$CASE_DIR/manifest.tsv"
  OUT="$CASE_DIR/out"
  STDOUT="$CASE_DIR/stdout"
  STDERR="$CASE_DIR/stderr"
  APPLY_LOG="$CASE_DIR/apply.log"
  NODE_COUNT="$CASE_DIR/node.count"
  RESTORE_FAILED="$CASE_DIR/restore.failed"
  BLOCK_READY="$CASE_DIR/block.ready"
  RM_COUNT="$CASE_DIR/rm.count"
  RM_READY="$CASE_DIR/rm.ready"
  HARDLINK_SENTINEL="$CASE_DIR/runtime-hardlink-sentinel.py"
  MISMATCH_ROOT="$CASE_DIR/other-root"
  mkdir -p "$SRC/cache" "$MISMATCH_ROOT"
  printf 'original target\n' > "$SRC/target.py"
  printf 'original other\n' > "$SRC/other.py"
  printf 'original outside\n' > "$REPO/outside.txt"
  printf 'preserve me\n' > "$SRC/cache/keep.txt"
  "$REAL_GIT" -C "$REPO" init -q
  "$REAL_GIT" -C "$REPO" config user.email selftest@example.invalid
  "$REAL_GIT" -C "$REPO" config user.name 'Verifier Selftest'
  "$REAL_GIT" -C "$REPO" add project/target.py project/other.py outside.txt
  "$REAL_GIT" -C "$REPO" commit -qm baseline
  RUN_RC=0
  VERIFY_PID=""
  ROOT_ARG="$REPO"
  SRC_ARG=project
  FAKE_BASE_RC=0
  FAKE_MUTANT_RC=1
  FAKE_MOUNT_MODE=ok
  FAKE_INSPECT_FAIL=0
  FAKE_GIT_FAIL_RESTORE=0
  FAKE_SECOND_EXEC_FAIL=0
  FAKE_SECOND_MISSING_MARKER=0
  FAKE_SECOND_UNCONFIRMED=0
  FAKE_LAUNCH_FAIL_ON=""
  FAKE_LAUNCH_FAIL_NODE=0
  FAKE_CLEAN_ACTION=none
  FAKE_MUTANT_ACTION=none
  FAKE_RM_BLOCK_ON=0
  ORIGINAL_PATCH=""
  RUN_TMPDIR="${TMPDIR:-/tmp}"
}

write_manifest() {
  local mutant="$1" file="${2:-target.py}" clean="${3:-0}" want_fail="${4:-true}"
  printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
  printf '%s\t%s\tmutmut_apply\ttest_node\t%s\t%s\n' \
    "$mutant" "$file" "$clean" "$want_fail" >> "$MANIFEST"
}

start_verifier() {
  env \
    "PATH=$TMP/bin:$PATH" \
    "REAL_GIT=$REAL_GIT" \
    "FAKE_ROOT=$REPO" \
    "FAKE_MISMATCH_ROOT=$MISMATCH_ROOT" \
    "FAKE_SRC=$SRC" \
    "FAKE_APPLY_LOG=$APPLY_LOG" \
    "FAKE_NODE_COUNT=$NODE_COUNT" \
    "FAKE_BLOCK_READY=$BLOCK_READY" \
    "FAKE_RM_COUNT=$RM_COUNT" \
    "FAKE_RM_READY=$RM_READY" \
    "FAKE_HARDLINK_SENTINEL=$HARDLINK_SENTINEL" \
    "FAKE_BASE_RC=$FAKE_BASE_RC" \
    "FAKE_MUTANT_RC=$FAKE_MUTANT_RC" \
    "FAKE_SECOND_EXEC_FAIL=$FAKE_SECOND_EXEC_FAIL" \
    "FAKE_SECOND_MISSING_MARKER=$FAKE_SECOND_MISSING_MARKER" \
    "FAKE_SECOND_UNCONFIRMED=$FAKE_SECOND_UNCONFIRMED" \
    "FAKE_LAUNCH_FAIL_ON=$FAKE_LAUNCH_FAIL_ON" \
    "FAKE_LAUNCH_FAIL_NODE=$FAKE_LAUNCH_FAIL_NODE" \
    "FAKE_CLEAN_ACTION=$FAKE_CLEAN_ACTION" \
    "FAKE_MUTANT_ACTION=$FAKE_MUTANT_ACTION" \
    "FAKE_RM_BLOCK_ON=$FAKE_RM_BLOCK_ON" \
    "FAKE_ORIGINAL_MANIFEST=$MANIFEST" \
    "FAKE_ORIGINAL_PATCH=$ORIGINAL_PATCH" \
    "FAKE_REPORT=$OUT/verify-report.txt" \
    "FAKE_MATRIX=$OUT/kill_matrix.tsv" \
    "FAKE_MOUNT_MODE=$FAKE_MOUNT_MODE" \
    "FAKE_INSPECT_FAIL=$FAKE_INSPECT_FAIL" \
    "FAKE_GIT_FAIL_RESTORE=$FAKE_GIT_FAIL_RESTORE" \
    "FAKE_RESTORE_FAILED=$RESTORE_FAILED" \
    "TMPDIR=$RUN_TMPDIR" \
    bash "$SUT" --container tl-mut --manifest "$MANIFEST" --out "$OUT" \
      --root "$ROOT_ARG" --src "$SRC_ARG" > "$STDOUT" 2> "$STDERR" &
  VERIFY_PID=$!
}

run_verifier() {
  start_verifier
  wait "$VERIFY_PID"
  RUN_RC=$?
  VERIFY_PID=""
}

tracked_tree_is_clean() {
  "$REAL_GIT" -C "$SRC" diff --quiet -- . \
    && "$REAL_GIT" -C "$SRC" diff --cached --quiet HEAD -- .
}

index_flags_are_safe() {
  ! "$REAL_GIT" -C "$REPO" ls-files -v | grep -Eq '^[a-zS] '
}

baseline_is_preserved() {
  tracked_tree_is_clean \
    && index_flags_are_safe \
    && [[ "$(<"$SRC/target.py")" == "original target" ]] \
    && [[ "$(<"$SRC/other.py")" == "original other" ]] \
    && [[ "$(<"$REPO/outside.txt")" == "original outside" ]] \
    && [[ "$(<"$SRC/cache/keep.txt")" == "preserve me" ]] \
    && [[ ! -e "$SRC/new.py" ]] \
    && [[ ! -e "$SRC/renamed.py" ]]
}

# Pure classification cannot be tricked by a configured failing baseline.
verdict="$(MUTMUT_VERIFY_LIB_ONLY=1 bash -c 'source "$1"; classify_verdict 1 1 1 true ok' _ "$SUT" | cut -f1)"
expect_eq "matching nonzero clean status is not a kill" "BAD" "$verdict"

# A normal assertion failure is a proven kill and leaves the full baseline.
setup_case
write_manifest good
run_verifier
expect_eq "valid kill exits zero" 0 "$RUN_RC"
expect_contains "valid kill is counted" "$OUT/verify-report.txt" 'RESULT: ALL KILLED (1)'
expect_contains "container path is derived from canonical source" "$OUT/verify-report.txt" 'proj=/work/project'
expect_contains "valid kill matrix verdict" "$OUT/kill_matrix.tsv" $'\tKILLED\t'
expect_contains "clean pytest log is preserved" "$OUT/row-000001-clean.log" 'fake command output rc=0'
expect_contains "mutant pytest log is preserved" "$OUT/row-000001-mutant.log" 'fake command output rc=1'
baseline_is_preserved && pass "valid kill restores baseline" || fail "valid kill restores baseline"

# The documented output path is inside the worktree. Verifier-owned reports are
# captured before the transaction baseline and are not mistaken for mutation.
setup_case
write_manifest good
OUT="$REPO/work/verify"
STDOUT="$CASE_DIR/stdout"
STDERR="$CASE_DIR/stderr"
run_verifier
expect_eq "in-worktree output does not pollute mutation delta" 0 "$RUN_RC"
expect_contains "in-worktree report records kill" "$OUT/verify-report.txt" 'RESULT: ALL KILLED (1)'
baseline_is_preserved && pass "in-worktree output run restores source baseline" \
  || fail "in-worktree output run restores source baseline"

# Multiple rows are all counted; an empty or partially consumed manifest cannot
# accidentally produce an all-killed summary.
setup_case
write_manifest good
printf 'good2\ttarget.py\tmutmut_apply\ttest_node_2\t0\ttrue\n' >> "$MANIFEST"
run_verifier
expect_eq "two valid kills exit zero" 0 "$RUN_RC"
expect_contains "all rows are counted" "$OUT/verify-report.txt" 'RESULT: ALL KILLED (2)'
expect_eq "kill matrix has two data rows" 3 "$(wc -l < "$OUT/kill_matrix.tsv")"
baseline_is_preserved && pass "multi-row run restores baseline" || fail "multi-row run restores baseline"

# Expected-pass rows succeed without being called kills.
setup_case
write_manifest good target.py 0 false
FAKE_MUTANT_RC=0
run_verifier
expect_eq "expected-pass row exits zero" 0 "$RUN_RC"
expect_contains "expected-pass summary is truthful" "$OUT/verify-report.txt" 'RESULT: SUCCESS (KILLED=0 EXPECTED_PASS=1)'
expect_not_contains "expected-pass summary does not claim all killed" "$OUT/verify-report.txt" 'ALL KILLED'
baseline_is_preserved && pass "expected-pass row restores baseline" || fail "expected-pass row restores baseline"

# Pytest infrastructure statuses remain inconclusive, never kills.
setup_case
write_manifest good
FAKE_MUTANT_RC=2
run_verifier
expect_nonzero "mutant collection error fails verification" "$RUN_RC"
expect_contains "collection error is inconclusive" "$OUT/kill_matrix.tsv" $'\tINCONCLUSIVE\t'
expect_contains "inconclusive counter is truthful" "$OUT/verify-report.txt" 'INCONCLUSIVE=1'
baseline_is_preserved && pass "inconclusive row restores baseline" || fail "inconclusive row restores baseline"

# A mutant that leaves the test passing is BAD, not successful evidence.
setup_case
write_manifest good
FAKE_MUTANT_RC=0
run_verifier
expect_nonzero "surviving mutant fails verification" "$RUN_RC"
expect_contains "survivor is BAD" "$OUT/kill_matrix.tsv" $'\tBAD\tsurvived:'
expect_contains "bad counter is truthful" "$OUT/verify-report.txt" 'BAD=1'
baseline_is_preserved && pass "survivor row restores baseline" || fail "survivor row restores baseline"

# A clean failure stops before mutation even when the manifest requests a kill.
setup_case
write_manifest good
FAKE_BASE_RC=1
run_verifier
expect_nonzero "failing clean test rejects row" "$RUN_RC"
expect_contains "clean failure is BAD" "$OUT/kill_matrix.tsv" $'\tBAD\tclean test did not pass'
[[ ! -e "$APPLY_LOG" ]] && pass "clean failure never applies mutant" || fail "clean failure never applies mutant"
baseline_is_preserved && pass "clean failure preserves baseline" || fail "clean failure preserves baseline"

# Clean tests run inside a transaction too. Side effects are restored and reject
# the row regardless of whether pytest itself reports pass or failure.
for clean_rc in 0 1; do
  setup_case
  write_manifest good
  FAKE_BASE_RC="$clean_rc"
  FAKE_CLEAN_ACTION=write
  run_verifier
  expect_nonzero "clean rc=$clean_rc side effect fails verification" "$RUN_RC"
  expect_contains "clean rc=$clean_rc side effect is diagnosed" "$OUT/kill_matrix.tsv" 'clean test changed worktree paths'
  [[ ! -e "$APPLY_LOG" ]] && pass "clean rc=$clean_rc side effect prevents mutation" \
    || fail "clean rc=$clean_rc side effect prevents mutation"
  baseline_is_preserved && pass "clean rc=$clean_rc side effect is restored" \
    || fail "clean rc=$clean_rc side effect is restored"
done

setup_case
write_manifest good
FAKE_CLEAN_ACTION=assume-write
run_verifier
expect_nonzero "clean test cannot hide a write with assume-unchanged" "$RUN_RC"
expect_contains "clean assume-unchanged injection is diagnosed" "$OUT/kill_matrix.tsv" 'unsafe_index_flags=project/other.py'
[[ ! -e "$APPLY_LOG" ]] && pass "hidden clean write prevents mutation" || fail "hidden clean write prevents mutation"
baseline_is_preserved && pass "hidden clean write and flag are restored" \
  || fail "hidden clean write and flag are restored"

setup_case
write_manifest good
printf 'original target\n' > "$HARDLINK_SENTINEL"
FAKE_CLEAN_ACTION=hardlink-target
run_verifier
expect_nonzero "clean test cannot replace target with a hardlink" "$RUN_RC"
expect_contains "runtime target hardlink is diagnosed" "$OUT/kill_matrix.tsv" 'unsafe_targets=project/target.py'
expect_eq "runtime hardlink sentinel is not overwritten during restore" 'original target' "$(<"$HARDLINK_SENTINEL")"
baseline_is_preserved && pass "runtime target hardlink is broken and target restored" \
  || fail "runtime target hardlink is broken and target restored"

setup_case
write_manifest good
FAKE_CLEAN_ACTION=signal-write
run_verifier
expect_nonzero "signal during clean test interrupts verification" "$RUN_RC"
baseline_is_preserved && pass "signal during clean test restores side effects" \
  || fail "signal during clean test restores side effects"

# Docker status alone is not accepted as pytest status. The wrapper must emit
# its per-invocation completion marker; transport failure is inconclusive.
setup_case
write_manifest good
FAKE_SECOND_EXEC_FAIL=1
run_verifier
expect_nonzero "second docker exec failure is not a kill" "$RUN_RC"
expect_contains "docker rc=1 is inconclusive" "$OUT/kill_matrix.tsv" $'\tINCONCLUSIVE\tdocker exec did not return a trusted completion marker (rc=1)'
expect_not_contains "transport failure never reports KILLED" "$OUT/kill_matrix.tsv" $'\tKILLED\t'
expect_contains "failed docker output is preserved" "$OUT/row-000001-mutant.log" 'docker transport failed'
baseline_is_preserved && pass "docker exec failure restores mutant" || fail "docker exec failure restores mutant"

setup_case
write_manifest good
FAKE_SECOND_MISSING_MARKER=1
run_verifier
expect_nonzero "missing pytest completion marker is not a kill" "$RUN_RC"
expect_contains "missing marker is inconclusive" "$OUT/kill_matrix.tsv" $'\tINCONCLUSIVE\tdocker exec did not return a trusted completion marker (rc=0)'
expect_not_contains "missing marker never reports KILLED" "$OUT/kill_matrix.tsv" $'\tKILLED\t'
expect_contains "missing-marker output is preserved" "$OUT/row-000001-mutant.log" 'wrapper ended without marker'
baseline_is_preserved && pass "missing marker restores mutant" || fail "missing marker restores mutant"

# A launcher rc=1 before the authenticated inner state is written is
# infrastructure, not pytest assertion failure and therefore never a kill.
setup_case
write_manifest good
FAKE_LAUNCH_FAIL_ON=pytest
FAKE_LAUNCH_FAIL_NODE=2
run_verifier
expect_nonzero "setsid launcher rc=1 is not a kill" "$RUN_RC"
expect_contains "launcher failure is inconclusive" "$OUT/kill_matrix.tsv" $'\tINCONCLUSIVE\tcontainer command launcher failed before authenticated start (rc=1)'
expect_not_contains "launcher failure never reports KILLED" "$OUT/kill_matrix.tsv" $'\tKILLED\t'
[[ -e "$APPLY_LOG" ]] && pass "launcher failure occurs after mutant application" \
  || fail "launcher failure occurs after mutant application"
baseline_is_preserved && pass "launcher failure restores mutant" || fail "launcher failure restores mutant"

setup_case
write_manifest good
FAKE_SECOND_UNCONFIRMED=1
run_verifier
expect_eq "unconfirmed container termination exits with safety status" 75 "$RUN_RC"
expect_contains "unconfirmed process prevents restoration" "$STDERR" 'refusing restoration'
if tracked_tree_is_clean; then
  fail "unconfirmed process state is not raced by restoration"
else
  pass "unconfirmed process state is not raced by restoration"
fi

setup_case
write_manifest good
FAKE_MUTANT_ACTION=write
run_verifier
expect_nonzero "mutant-test side effect fails verification" "$RUN_RC"
expect_contains "mutant-test side effect is diagnosed" "$OUT/kill_matrix.tsv" 'mutant test changed unexpected paths'
baseline_is_preserved && pass "mutant-test side effect is restored" \
  || fail "mutant-test side effect is restored"

setup_case
write_manifest good
FAKE_MUTANT_ACTION=same-target
run_verifier
expect_nonzero "same-target mutant-test write fails verification" "$RUN_RC"
expect_contains "same-target write is caught by fingerprint" "$OUT/kill_matrix.tsv" 'mutant test modified declared target after application'
expect_not_contains "same-target write never reports KILLED" "$OUT/kill_matrix.tsv" $'\tKILLED\t'
baseline_is_preserved && pass "same-target mutant-test write is restored" \
  || fail "same-target mutant-test write is restored"

setup_case
write_manifest good
FAKE_MUTANT_ACTION=skip-write
run_verifier
expect_nonzero "mutant test cannot hide another-file write with skip-worktree" "$RUN_RC"
expect_contains "mutant skip-worktree injection is diagnosed" "$OUT/kill_matrix.tsv" 'unsafe_index_flags=project/other.py'
expect_not_contains "hidden mutant write never reports KILLED" "$OUT/kill_matrix.tsv" $'\tKILLED\t'
if baseline_is_preserved; then
  pass "hidden mutant write and flag are restored"
else
  "$REAL_GIT" -C "$REPO" status --short
  "$REAL_GIT" -C "$REPO" ls-files -v | grep -E 'project/(target|other)'
  sed 's/^/# matrix: /' "$OUT/kill_matrix.tsv"
  fail "hidden mutant write and flag are restored"
fi

# Manifest validation is complete and occurs before mutation.
setup_case
printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
run_verifier
expect_nonzero "header-only manifest is rejected" "$RUN_RC"
expect_contains "header-only diagnostic" "$STDERR" 'at least one data row is required'

setup_case
write_manifest good target.py 1 true
run_verifier
expect_nonzero "nonzero expected clean status is rejected" "$RUN_RC"
expect_contains "clean-status diagnostic" "$STDERR" 'expect_clean_rc must be 0'
[[ ! -e "$APPLY_LOG" ]] && pass "invalid clean status never applies mutant" || fail "invalid clean status never applies mutant"

setup_case
write_manifest good
printf 'good\tother.py\tmutmut_apply\ttest_node\t0\ttrue\n' >> "$MANIFEST"
run_verifier
expect_nonzero "duplicate mutant id is rejected" "$RUN_RC"
expect_contains "duplicate diagnostic" "$STDERR" 'duplicate mutant_id good'

setup_case
printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
printf 'good\ttarget.py\tmutmut_apply\ttest_node\t0\n' >> "$MANIFEST"
run_verifier
expect_nonzero "short manifest row is rejected" "$RUN_RC"
expect_contains "short-row diagnostic" "$STDERR" 'expected 6 columns, got 5'

setup_case
printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
printf 'good\ttarget.py\tmutmut_apply\ttest_node\t0\ttrue' >> "$MANIFEST"
run_verifier
expect_eq "final manifest row needs no trailing newline" 0 "$RUN_RC"
expect_contains "unterminated final row is counted" "$OUT/verify-report.txt" 'RESULT: ALL KILLED (1)'
baseline_is_preserved && pass "unterminated final row restores baseline" || fail "unterminated final row restores baseline"

setup_case
printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
printf 'good\t\tmutmut_apply\ttest_node\t0\ttrue\n' >> "$MANIFEST"
run_verifier
expect_nonzero "empty manifest field is rejected" "$RUN_RC"
expect_contains "empty-field diagnostic" "$STDERR" 'column 2 is empty'

setup_case
printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
printf 'good\ttarget.py\tcommand\ttest_node\t0\ttrue\n' >> "$MANIFEST"
run_verifier
expect_nonzero "unknown apply method is rejected" "$RUN_RC"
expect_contains "apply-method diagnostic" "$STDERR" 'unsupported apply_method command'

setup_case
printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
printf 'good\ttarget.py\tmutmut_apply\ttest_node\t0\tmaybe\n' >> "$MANIFEST"
run_verifier
expect_nonzero "invalid mutant expectation is rejected" "$RUN_RC"
expect_contains "mutant-expectation diagnostic" "$STDERR" 'must be true or false'

setup_case
write_manifest good cache/keep.txt
run_verifier
expect_nonzero "untracked manifest target is rejected" "$RUN_RC"
expect_contains "untracked-target diagnostic" "$STDERR" 'not one exact tracked path'

setup_case
write_manifest good ../target.py
run_verifier
expect_nonzero "traversing manifest target is rejected" "$RUN_RC"
expect_contains "unsafe-target diagnostic" "$STDERR" 'unsafe manifest file path'

# --src is canonicalized before deriving the container path and cannot traverse,
# use an absolute path, escape through a symlink, or select a nested repository.
setup_case
write_manifest good
SRC_ARG='../project'
run_verifier
expect_nonzero "traversing --src is rejected" "$RUN_RC"
expect_contains "traversing --src diagnostic" "$STDERR" '--src must be a normalized path'

setup_case
write_manifest good
SRC_ARG="$SRC"
run_verifier
expect_nonzero "absolute --src is rejected" "$RUN_RC"
expect_contains "absolute --src diagnostic" "$STDERR" '--src must be a normalized path'

setup_case
write_manifest good
ESCAPED_SRC="$CASE_DIR/escaped-source"
mkdir -p "$ESCAPED_SRC"
ln -s "$ESCAPED_SRC" "$REPO/escape"
SRC_ARG=escape
run_verifier
expect_nonzero "symlink escape from --root is rejected" "$RUN_RC"
expect_contains "symlink escape diagnostic" "$STDERR" '--src escapes --root'

setup_case
write_manifest good
mkdir -p "$SRC/nested"
"$REAL_GIT" -C "$SRC/nested" init -q
SRC_ARG=project/nested
run_verifier
expect_nonzero "nested Git worktree is rejected" "$RUN_RC"
expect_contains "nested Git-root diagnostic" "$STDERR" 'belongs to a different Git worktree'

setup_case
write_manifest good
ln -s project "$REPO/project-alias"
SRC_ARG=project-alias
run_verifier
expect_eq "in-root source symlink is canonicalized safely" 0 "$RUN_RC"
expect_contains "canonical source drives container path" "$OUT/verify-report.txt" 'proj=/work/project'
baseline_is_preserved && pass "canonical source run restores baseline" || fail "canonical source run restores baseline"

setup_case
printf 'external\n' > "$CASE_DIR/external.py"
ln -s "$CASE_DIR/external.py" "$SRC/link.py"
"$REAL_GIT" -C "$REPO" add project/link.py
"$REAL_GIT" -C "$REPO" commit -qm 'tracked symlink fixture'
write_manifest good link.py
run_verifier
expect_nonzero "symbolic-link manifest target is rejected" "$RUN_RC"
expect_contains "symbolic-link target diagnostic" "$STDERR" 'tracked source path has unsafe hardlink, symlink, or file type'
expect_eq "external symlink target remains untouched" external "$(<"$CASE_DIR/external.py")"

setup_case
write_manifest good
EXTERNAL_SENTINEL="$CASE_DIR/external-sentinel.py"
printf 'original target\n' > "$EXTERNAL_SENTINEL"
rm -f -- "$SRC/target.py"
ln "$EXTERNAL_SENTINEL" "$SRC/target.py"
run_verifier
expect_nonzero "target hardlinked outside repository is rejected" "$RUN_RC"
expect_contains "external hardlink diagnostic" "$STDERR" 'tracked source path has unsafe hardlink, symlink, or file type'
expect_eq "external hardlink sentinel is preserved" 'original target' "$(<"$EXTERNAL_SENTINEL")"
[[ ! -e "$APPLY_LOG" ]] && pass "external hardlink prevents mutation" || fail "external hardlink prevents mutation"

setup_case
write_manifest good
IN_REPO_SENTINEL="$SRC/cache/hardlink-sentinel.py"
printf 'original target\n' > "$IN_REPO_SENTINEL"
rm -f -- "$SRC/target.py"
ln "$IN_REPO_SENTINEL" "$SRC/target.py"
run_verifier
expect_nonzero "target hardlinked to in-repo untracked file is rejected" "$RUN_RC"
expect_contains "in-repo hardlink diagnostic" "$STDERR" 'tracked source path has unsafe hardlink, symlink, or file type'
expect_eq "in-repo hardlink sentinel is preserved" 'original target' "$(<"$IN_REPO_SENTINEL")"
[[ ! -e "$APPLY_LOG" ]] && pass "in-repo hardlink prevents mutation" || fail "in-repo hardlink prevents mutation"

setup_case
write_manifest wrong
EXTERNAL_SENTINEL="$CASE_DIR/wrong-target-external-sentinel.py"
printf 'original other\n' > "$EXTERNAL_SENTINEL"
rm -f -- "$SRC/other.py"
ln "$EXTERNAL_SENTINEL" "$SRC/other.py"
run_verifier
expect_nonzero "wrong-target external hardlink is rejected before docker" "$RUN_RC"
expect_contains "wrong-target hardlink diagnostic" "$STDERR" 'tracked source path has unsafe hardlink, symlink, or file type'
expect_eq "wrong-target hardlink sentinel is preserved" 'original other' "$(<"$EXTERNAL_SENTINEL")"
[[ ! -e "$APPLY_LOG" ]] && pass "wrong-target hardlink prevents application" || fail "wrong-target hardlink prevents application"

setup_case
write_manifest wrong
EXTERNAL_SENTINEL="$CASE_DIR/wrong-target-external-sentinel.py"
printf 'original other\n' > "$EXTERNAL_SENTINEL"
rm -f -- "$SRC/other.py"
ln -s "$EXTERNAL_SENTINEL" "$SRC/other.py"
"$REAL_GIT" -C "$REPO" add project/other.py
"$REAL_GIT" -C "$REPO" commit -qm 'external symlink fixture'
run_verifier
expect_nonzero "wrong-target external symlink is rejected before docker" "$RUN_RC"
expect_contains "wrong-target symlink diagnostic" "$STDERR" 'tracked source path has unsafe hardlink, symlink, or file type'
expect_eq "wrong-target symlink sentinel is preserved" 'original other' "$(<"$EXTERNAL_SENTINEL")"
[[ ! -e "$APPLY_LOG" ]] && pass "wrong-target symlink prevents application" || fail "wrong-target symlink prevents application"

setup_case
write_manifest good
printf 'agent instructions\n' > "$SRC/AGENTS.md"
ln -s AGENTS.md "$SRC/CLAUDE.md"
"$REAL_GIT" -C "$REPO" add project/AGENTS.md project/CLAUDE.md
"$REAL_GIT" -C "$REPO" commit -qm 'safe in-tree symlink fixture'
run_verifier
expect_eq "tracked in-tree symlink to tracked content is allowed" 0 "$RUN_RC"
baseline_is_preserved && pass "safe in-tree symlink run restores baseline" \
  || fail "safe in-tree symlink run restores baseline"

# Any pre-existing tracked edit is rejected and preserved.
setup_case
write_manifest good
printf 'user edit\n' >> "$SRC/other.py"
before_dirty="$(sha256sum "$SRC/other.py" | cut -d' ' -f1)"
run_verifier
expect_nonzero "dirty tracked source is rejected" "$RUN_RC"
expect_contains "dirty-source diagnostic" "$STDERR" 'tracked source has unstaged changes'
expect_eq "dirty source bytes are preserved" "$before_dirty" "$(sha256sum "$SRC/other.py" | cut -d' ' -f1)"
[[ ! -e "$APPLY_LOG" ]] && pass "dirty source never applies mutant" || fail "dirty source never applies mutant"

setup_case
write_manifest good
printf 'staged user edit\n' >> "$SRC/other.py"
"$REAL_GIT" -C "$SRC" add other.py
before_dirty="$(sha256sum "$SRC/other.py" | cut -d' ' -f1)"
run_verifier
expect_nonzero "staged tracked source is rejected" "$RUN_RC"
expect_contains "staged-source diagnostic" "$STDERR" 'tracked source has staged changes'
expect_eq "staged source bytes are preserved" "$before_dirty" "$(sha256sum "$SRC/other.py" | cut -d' ' -f1)"
[[ ! -e "$APPLY_LOG" ]] && pass "staged source never applies mutant" || fail "staged source never applies mutant"

# Index flags that suppress ordinary dirty detection are rejected explicitly.
setup_case
write_manifest good
"$REAL_GIT" -C "$REPO" update-index --assume-unchanged project/other.py
run_verifier
expect_nonzero "assume-unchanged entry is rejected" "$RUN_RC"
expect_contains "assume-unchanged diagnostic" "$STDERR" 'skip-worktree or assume-unchanged'

setup_case
write_manifest good
"$REAL_GIT" -C "$REPO" update-index --skip-worktree project/other.py
run_verifier
expect_nonzero "skip-worktree entry is rejected" "$RUN_RC"
expect_contains "skip-worktree diagnostic" "$STDERR" 'skip-worktree or assume-unchanged'

# Actual changes, rather than the manifest claim, determine validation/cleanup.
for mutation in wrong outside multi new rename partial noop; do
  setup_case
  write_manifest "$mutation"
  run_verifier
  expect_nonzero "$mutation application is rejected" "$RUN_RC"
  case "$mutation" in
    partial) expect_contains "partial apply diagnostic" "$OUT/kill_matrix.tsv" 'apply failed rc=17' ;;
    *) expect_contains "$mutation path diagnostic" "$OUT/kill_matrix.tsv" 'application changed unexpected paths' ;;
  esac
  baseline_is_preserved && pass "$mutation application restores complete baseline" \
    || fail "$mutation application restores complete baseline"
done

# Deleting the declared tracked target is a valid one-path application and is
# restored before the kill is accepted.
setup_case
write_manifest delete
run_verifier
expect_eq "declared deletion can prove a kill" 0 "$RUN_RC"
baseline_is_preserved && pass "deleted target is restored" || fail "deleted target is restored"

# The host-side diff backend uses the same changed-path transaction.
setup_case
PATCH_FILE="$CASE_DIR/target.diff"
printf 'mutated\n' >> "$SRC/target.py"
"$REAL_GIT" -C "$REPO" diff -- project/target.py > "$PATCH_FILE"
[[ -s "$PATCH_FILE" ]] && pass "diff fixture contains a change" || fail "diff fixture contains a change"
"$REAL_GIT" -C "$SRC" restore -- target.py
printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
printf 'diff-good\ttarget.py\tdiff:%s\ttest_node\t0\ttrue\n' "$PATCH_FILE" >> "$MANIFEST"
run_verifier
expect_eq "single-path diff can prove a kill" 0 "$RUN_RC"
baseline_is_preserved && pass "diff backend restores baseline" || fail "diff backend restores baseline"

# Diff inputs are rejected before execution unless they touch exactly the one
# declared tracked path.
setup_case
PATCH_FILE="$CASE_DIR/multi.diff"
printf 'mutated\n' >> "$SRC/target.py"
printf 'mutated\n' >> "$SRC/other.py"
"$REAL_GIT" -C "$REPO" diff -- project/target.py project/other.py > "$PATCH_FILE"
"$REAL_GIT" -C "$REPO" restore -- project/target.py project/other.py
printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
printf 'diff-multi\ttarget.py\tdiff:%s\ttest_node\t0\ttrue\n' "$PATCH_FILE" >> "$MANIFEST"
run_verifier
expect_nonzero "multi-path diff is rejected before execution" "$RUN_RC"
expect_contains "multi-path diff diagnostic" "$STDERR" 'must touch exactly declared path'
[[ ! -e "$OUT/verify-report.txt" ]] && pass "invalid diff writes no report" || fail "invalid diff writes no report"
baseline_is_preserved && pass "diff prevalidation leaves baseline unchanged" || fail "diff prevalidation leaves baseline unchanged"

setup_case
PATCH_FILE="$CASE_DIR/untracked.diff"
printf '%s\n' \
  'diff --git a/project/cache/keep.txt b/project/cache/keep.txt' \
  '--- a/project/cache/keep.txt' \
  '+++ b/project/cache/keep.txt' \
  '@@ -1 +1 @@' \
  '-preserve me' \
  '+changed' > "$PATCH_FILE"
printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
printf 'diff-untracked\ttarget.py\tdiff:%s\ttest_node\t0\ttrue\n' "$PATCH_FILE" >> "$MANIFEST"
run_verifier
expect_nonzero "diff targeting existing untracked file is rejected" "$RUN_RC"
expect_contains "untracked diff target diagnostic" "$STDERR" 'must touch exactly declared path'
expect_eq "untracked file is untouched" 'preserve me' "$(<"$SRC/cache/keep.txt")"

# The original manifest and patch may change after validation; execution uses
# the protected copies captured before the first docker command.
setup_case
PATCH_FILE="$CASE_DIR/snapshotted.diff"
printf 'mutated\n' >> "$SRC/target.py"
"$REAL_GIT" -C "$REPO" diff -- project/target.py > "$PATCH_FILE"
"$REAL_GIT" -C "$REPO" restore -- project/target.py
printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
printf 'diff-snapshot\ttarget.py\tdiff:%s\ttest_node\t0\ttrue\n' "$PATCH_FILE" >> "$MANIFEST"
ORIGINAL_PATCH="$PATCH_FILE"
FAKE_CLEAN_ACTION=mutate-inputs
run_verifier
expect_eq "post-validation input mutation cannot change execution" 0 "$RUN_RC"
expect_contains "snapshotted row is still counted" "$OUT/verify-report.txt" 'RESULT: ALL KILLED (1)'
expect_eq "original manifest was actually replaced" 'replaced manifest' "$(<"$MANIFEST")"
expect_eq "original patch was actually replaced" 'replaced patch' "$(<"$PATCH_FILE")"
baseline_is_preserved && pass "immutable-input run restores baseline" || fail "immutable-input run restores baseline"

# A signal delivered after a partial write invokes the EXIT cleanup transaction.
setup_case
write_manifest signal
run_verifier
expect_nonzero "signal interrupts verification" "$RUN_RC"
baseline_is_preserved && pass "signal trap restores baseline" || fail "signal trap restores baseline"

# A transient restore failure is reported, and the EXIT trap retries cleanup.
setup_case
write_manifest good
FAKE_GIT_FAIL_RESTORE=1
run_verifier
expect_nonzero "restore failure fails verification" "$RUN_RC"
[[ -e "$RESTORE_FAILED" ]] && pass "restore failure was injected" || fail "restore failure was injected"
baseline_is_preserved && pass "EXIT trap retries failed restoration" || fail "EXIT trap retries failed restoration"

# A blocking child receives TERM and cannot prevent bounded cleanup.
setup_case
write_manifest block
start_verifier
for _ in $(seq 1 100); do
  [[ -e "$BLOCK_READY" ]] && break
  sleep 0.02
done
if [[ -e "$BLOCK_READY" ]]; then
  start_seconds="$(date +%s)"
  kill -TERM "$VERIFY_PID"
  wait "$VERIFY_PID"
  RUN_RC=$?
  elapsed_seconds=$(( $(date +%s) - start_seconds ))
  expect_nonzero "TERM stops blocking external command" "$RUN_RC"
  [[ $elapsed_seconds -lt 5 ]] && pass "TERM cleanup is bounded" \
    || fail "TERM cleanup is bounded (elapsed=${elapsed_seconds}s)"
  baseline_is_preserved && pass "TERM of blocking child restores baseline" \
    || fail "TERM of blocking child restores baseline"
else
  fail "blocking fake reached mutation point"
  kill -KILL "$VERIFY_PID" 2>/dev/null || true
  wait "$VERIFY_PID" 2>/dev/null || true
fi
VERIFY_PID=""

# Once a trusted completion marker is received, deleting its state file is only
# housekeeping. A signal while that cleanup client is blocked must still restore
# the already-completed mutation transaction.
setup_case
write_manifest good
FAKE_RM_BLOCK_ON=2
start_verifier
for _ in $(seq 1 100); do
  [[ -e "$RM_READY" ]] && break
  sleep 0.02
done
if [[ -e "$RM_READY" ]]; then
  kill -TERM "$VERIFY_PID"
  wait "$VERIFY_PID"
  RUN_RC=$?
  VERIFY_PID=""
  expect_nonzero "TERM interrupts completed-state cleanup client" "$RUN_RC"
  baseline_is_preserved && pass "trusted completion permits restoration after state removal" \
    || fail "trusted completion permits restoration after state removal"
else
  fail "state-cleanup fake reached blocking point"
  kill -KILL "$VERIFY_PID" 2>/dev/null || true
  wait "$VERIFY_PID" 2>/dev/null || true
  VERIFY_PID=""
fi

# Killing the local docker client is insufficient when an in-container process
# survives it. The recorded process group is terminated and verified first, so
# a delayed writer cannot race host restoration.
setup_case
write_manifest linger
start_verifier
for _ in $(seq 1 100); do
  [[ -e "$BLOCK_READY" ]] && break
  sleep 0.02
done
if [[ -e "$BLOCK_READY" ]]; then
  kill -TERM "$VERIFY_PID"
  wait "$VERIFY_PID"
  RUN_RC=$?
  VERIFY_PID=""
  expect_nonzero "TERM interrupts local client with surviving container child" "$RUN_RC"
  sleep 1.2
  baseline_is_preserved && pass "container process group is dead before restoration" \
    || fail "container process group is dead before restoration"
else
  fail "delayed-writer fake reached mutation point"
  kill -KILL "$VERIFY_PID" 2>/dev/null || true
  wait "$VERIFY_PID" 2>/dev/null || true
  VERIFY_PID=""
fi

# State cannot be placed inside the repository even when TMPDIR requests it.
setup_case
write_manifest good
mkdir -p "$REPO/tmp"
RUN_TMPDIR="$REPO/tmp"
run_verifier
expect_eq "TMPDIR inside worktree falls back to protected state" 0 "$RUN_RC"
if find "$REPO/tmp" -mindepth 1 -print -quit | grep -q .; then
  fail "no verifier state remains under in-worktree TMPDIR"
else
  pass "no verifier state remains under in-worktree TMPDIR"
fi
baseline_is_preserved && pass "in-worktree TMPDIR run restores baseline" \
  || fail "in-worktree TMPDIR run restores baseline"

# Output files may not alias the manifest or any other inode, and symlinked
# output directories are rejected before truncation.
setup_case
mkdir -p "$OUT"
MANIFEST="$OUT/kill_matrix.tsv"
write_manifest good
run_verifier
expect_nonzero "manifest/output direct alias is rejected" "$RUN_RC"
expect_contains "direct output alias diagnostic" "$STDERR" 'unsafe output destination'

setup_case
write_manifest good
mkdir -p "$OUT"
ln -s "$MANIFEST" "$OUT/verify-report.txt"
run_verifier
expect_nonzero "manifest/output symlink alias is rejected" "$RUN_RC"
expect_contains "symlink output alias diagnostic" "$STDERR" 'unsafe output destination'

setup_case
write_manifest good
mkdir -p "$OUT"
ln "$MANIFEST" "$OUT/kill_matrix.tsv"
run_verifier
expect_nonzero "manifest/output hardlink alias is rejected" "$RUN_RC"
expect_contains "hardlink output alias diagnostic" "$STDERR" 'unsafe output destination'

setup_case
write_manifest good
REAL_OUT="$CASE_DIR/real-output"
mkdir -p "$REAL_OUT"
ln -s "$REAL_OUT" "$CASE_DIR/output-link"
OUT="$CASE_DIR/output-link"
run_verifier
expect_nonzero "symlinked output directory is rejected" "$RUN_RC"
expect_contains "symlinked output directory diagnostic" "$STDERR" 'output path contains a symlink'

# Every original diff source is also a protected control input. No output may
# directly name it or reach its inode through a symlink or hardlink.
for alias_kind in direct symlink hardlink; do
  setup_case
  mkdir -p "$OUT"
  PATCH_FILE="$CASE_DIR/source.diff"
  [[ "$alias_kind" == direct ]] && PATCH_FILE="$OUT/kill_matrix.tsv"
  printf 'mutated\n' >> "$SRC/target.py"
  "$REAL_GIT" -C "$REPO" diff -- project/target.py > "$PATCH_FILE"
  "$REAL_GIT" -C "$REPO" restore -- project/target.py
  patch_hash="$(sha256sum "$PATCH_FILE" | cut -d' ' -f1)"
  printf 'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero\n' > "$MANIFEST"
  printf 'diff-alias\ttarget.py\tdiff:%s\ttest_node\t0\ttrue\n' "$PATCH_FILE" >> "$MANIFEST"
  case "$alias_kind" in
    symlink) ln -s "$PATCH_FILE" "$OUT/kill_matrix.tsv" ;;
    hardlink) ln "$PATCH_FILE" "$OUT/kill_matrix.tsv" ;;
  esac
  ORIGINAL_PATCH="$PATCH_FILE"
  run_verifier
  expect_nonzero "patch/output $alias_kind alias is rejected" "$RUN_RC"
  expect_contains "patch/output $alias_kind diagnostic" "$STDERR" 'unsafe output destination'
  expect_eq "patch bytes survive $alias_kind output alias" "$patch_hash" \
    "$(sha256sum "$PATCH_FILE" | cut -d' ' -f1)"
done

# Runtime replacement of report/matrix files is detected before another write;
# neither case is allowed to turn a successful row into accepted evidence.
for corrupt in corrupt-report corrupt-matrix; do
  setup_case
  write_manifest good
  FAKE_CLEAN_ACTION="$corrupt"
  run_verifier
  expect_nonzero "$corrupt fails closed" "$RUN_RC"
  expect_contains "$corrupt diagnostic" "$STDERR" 'output file became unsafe or unwritable'
  baseline_is_preserved && pass "$corrupt leaves source baseline intact" \
    || fail "$corrupt leaves source baseline intact"
done

setup_case
write_manifest good
mkdir -p "$OUT"
printf 'pre-existing report\n' > "$OUT/verify-report.txt"
FAKE_CLEAN_ACTION=truncate-report
run_verifier
expect_nonzero "in-place output truncation fails closed" "$RUN_RC"
expect_contains "in-place output fingerprint diagnostic" "$STDERR" 'output file became unsafe or unwritable'
expect_contains "tampered destination is not overwritten by success" "$OUT/verify-report.txt" 'overwritten in place'
expect_not_contains "tampered destination contains no published success" "$OUT/verify-report.txt" 'RESULT: ALL KILLED'
baseline_is_preserved && pass "in-place output tampering leaves source baseline intact" \
  || fail "in-place output tampering leaves source baseline intact"

# Mount/container failures stop before any mutation.
for mode in missing mismatch readonly; do
  setup_case
  write_manifest good
  FAKE_MOUNT_MODE="$mode"
  run_verifier
  expect_nonzero "$mode mount is rejected" "$RUN_RC"
  [[ ! -e "$APPLY_LOG" ]] && pass "$mode mount never applies mutant" || fail "$mode mount never applies mutant"
done

setup_case
write_manifest good
FAKE_INSPECT_FAIL=1
run_verifier
expect_nonzero "missing container is rejected" "$RUN_RC"
expect_contains "missing-container diagnostic" "$STDERR" 'container not found'

printf '1..%d\n' "$tests"
if [[ $failures -eq 0 ]]; then
  printf 'ALL MUTMUT VERIFIER SELFTESTS PASSED\n'
  exit 0
fi
printf '%d MUTMUT VERIFIER SELFTEST(S) FAILED\n' "$failures" >&2
exit 1
