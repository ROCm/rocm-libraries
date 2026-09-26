#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# mutmut-verify.sh — optional, manifest-driven survivor verification.
#
# Each row is handled as two fail-closed transactions: run its test against clean
# source, then apply exactly one reviewed mutation and run the same test again.
# Every changed tracked path and every newly created, non-ignored path is
# discovered from Git state and restored. The complete tracked worktree must be
# clean before verification and must match that baseline after every command.
#
# Usage:
#   mutmut-verify.sh --container tl-mut --manifest <rows.tsv> --out <dir> \
#       [--src projects/hipblaslt/tensilelite] [--root <worktree>]
#
# Manifest: TSV with one header and one or more data rows:
#   mutant_id <TAB> file <TAB> apply_method <TAB> test_node <TAB> \
#       expect_clean_rc <TAB> expect_mutant_rc_nonzero
#     file          : tracked path relative to --src
#     apply_method  : "mutmut_apply" or "diff:<absolute-patch-path>"
#     test_node     : pytest node relative to the in-container project directory
#     expect_clean_rc          : must be 0; a failing clean test is never kill proof
#     expect_mutant_rc_nonzero : true means pytest assertion-failure rc 1 is a kill;
#                                false means rc 0 is an expected pass, not a kill
#
# Output: <out>/kill_matrix.tsv, <out>/verify-report.txt, and per-row pytest logs.

set -u

# ----------------------------------------------------------------- pure classify
classify_verdict() {
  local base_rc="$1" exp_clean="$2" mut_rc="$3" want_fail="$4" revert="$5"
  local _int='^-?[0-9]+$'

  if ! [[ "$base_rc" =~ $_int && "$exp_clean" =~ $_int && "$mut_rc" =~ $_int ]]; then
    printf 'BAD\tnon-numeric rc field (base_rc=%s exp_clean=%s mut_rc=%s)' \
      "$base_rc" "$exp_clean" "$mut_rc"
    return
  fi
  if [[ "$exp_clean" -ne 0 ]]; then
    printf 'BAD\texpect_clean_rc=%s is unsupported; clean tests must pass with rc=0' "$exp_clean"
    return
  fi
  if [[ "$base_rc" -ne 0 ]]; then
    printf 'BAD\tclean test did not pass (base_rc=%s)' "$base_rc"
    return
  fi
  if [[ "$revert" != "ok" ]]; then
    printf 'BAD\trevert=%s base_rc=%s mut_rc=%s' "$revert" "$base_rc" "$mut_rc"
    return
  fi
  if [[ "$want_fail" != "true" && "$want_fail" != "false" ]]; then
    printf 'BAD\texpect_mutant_rc_nonzero must be true or false'
    return
  fi

  if [[ "$want_fail" == "true" ]]; then
    if [[ "$mut_rc" -eq 1 ]]; then
      printf 'KILLED\tbase_rc=0 mut_rc=1'
    elif [[ "$mut_rc" -eq 0 ]]; then
      printf 'BAD\tsurvived: mut_rc=0 (test passed under mutant)'
    else
      printf 'INCONCLUSIVE\tmut_rc=%s (collection/usage/internal error, not an assertion failure)' "$mut_rc"
    fi
  elif [[ "$mut_rc" -eq 0 ]]; then
    printf 'OK\tbase_rc=0 mut_rc=0 (expected pass; not a kill)'
  else
    printf 'BAD\tmut_rc=%s (expected 0)' "$mut_rc"
  fi
}

# Library mode exposes the pure classifier without running the CLI.
if [[ -n "${MUTMUT_VERIFY_LIB_ONLY:-}" ]]; then return 0 2>/dev/null || exit 0; fi

die() { printf 'mutmut-verify: ERROR: %s\n' "$*" >&2; exit 2; }

# ----------------------------------------------------------------- args
CON=""; MANIFEST=""; OUT=""; SRC_REL="projects/hipblaslt/tensilelite"; ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --container) [[ $# -ge 2 ]] || die "$1 requires a value"; CON="$2"; shift 2 ;;
    --manifest)  [[ $# -ge 2 ]] || die "$1 requires a value"; MANIFEST="$2"; shift 2 ;;
    --out)       [[ $# -ge 2 ]] || die "$1 requires a value"; OUT="$2"; shift 2 ;;
    --src)       [[ $# -ge 2 ]] || die "$1 requires a value"; SRC_REL="$2"; shift 2 ;;
    --root)      [[ $# -ge 2 ]] || die "$1 requires a value"; ROOT="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$CON" && -n "$MANIFEST" && -n "$OUT" ]] || {
  die "usage: $0 --container <name> --manifest <tsv> --out <dir> [--src <rel>] [--root <worktree>]"; }
[[ -f "$MANIFEST" ]] || die "manifest not found: $MANIFEST"

# Resolve --root as the actual Git worktree root, then confine --src beneath it.
if [[ -z "$ROOT" ]]; then ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; fi
[[ -n "$ROOT" ]] || die "could not determine --root (not in a git worktree?)"
ROOT="$(readlink -f -- "$ROOT")" || die "could not canonicalize --root"
[[ -d "$ROOT" ]] || die "root directory not found: $ROOT"
GIT_ROOT="$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null)" \
  || die "--root is not in a Git worktree: $ROOT"
GIT_ROOT="$(readlink -f -- "$GIT_ROOT")" || die "could not canonicalize Git worktree root"
[[ "$GIT_ROOT" == "$ROOT" ]] || die "--root must be the Git worktree root: $GIT_ROOT"

case "$SRC_REL" in
  /*|.|./*|..|../*|*/./*|*/.|*/../*|*/..) die "--src must be a normalized path below --root: $SRC_REL" ;;
esac
SRC="$(readlink -f -- "$ROOT/$SRC_REL")" || die "could not canonicalize --src: $SRC_REL"
[[ -d "$SRC" ]] || die "src dir not found: $SRC"
case "$SRC" in
  "$ROOT"/*) ;;
  *) die "--src escapes --root after canonicalization: $SRC_REL -> $SRC" ;;
esac
SRC_GIT_ROOT="$(git -C "$SRC" rev-parse --show-toplevel 2>/dev/null)" \
  || die "src is not in a Git worktree: $SRC"
SRC_GIT_ROOT="$(readlink -f -- "$SRC_GIT_ROOT")" || die "could not canonicalize source Git root"
[[ "$SRC_GIT_ROOT" == "$ROOT" ]] || die "--src belongs to a different Git worktree: $SRC_GIT_ROOT"
SRC_FROM_ROOT="${SRC#"$ROOT"/}"
[[ -n "$SRC_FROM_ROOT" && "$SRC_FROM_ROOT" != "$SRC" ]] \
  || die "--src must name a project directory below --root"
PROJ="/work/$SRC_FROM_ROOT"

# Keep immutable control data outside the worktree, even when TMPDIR points into
# it. /var/tmp is a final fallback for the unusual case where --root is /tmp.
make_state_dir() {
  local base candidate canonical
  for base in "${TMPDIR:-}" /tmp /var/tmp; do
    [[ -n "$base" && -d "$base" && -w "$base" ]] || continue
    candidate="$(mktemp -d "$base/mutmut-verify.XXXXXXXX" 2>/dev/null)" || continue
    canonical="$(readlink -f -- "$candidate")" || { rm -rf -- "$candidate"; continue; }
    case "$canonical" in
      "$ROOT"|"$ROOT"/*) rm -rf -- "$candidate" ;;
      *) printf '%s\n' "$canonical"; return 0 ;;
    esac
  done
  return 1
}

STATE_DIR="$(make_state_dir)" || die "could not create protected state outside the Git worktree"
early_cleanup() { rm -rf -- "$STATE_DIR"; }
trap early_cleanup EXIT

ACTIVE_CHILD_PID=""
ACTIVE_CONTAINER_STATE=""
ACTIVE_CONTAINER_AUTH=""
CONTAINER_SAFE_TO_RESTORE=1
run_external() {
  local rc
  "$@" &
  ACTIVE_CHILD_PID=$!
  wait "$ACTIVE_CHILD_PID"
  rc=$?
  ACTIVE_CHILD_PID=""
  return "$rc"
}

terminate_container_command() {
  local state_file="$ACTIVE_CONTAINER_STATE" expected_auth="$ACTIVE_CONTAINER_AUTH" killer rc i
  [[ -n "$state_file" ]] || return 0

  docker exec "$CON" sh -c '
    state_file=$1
    expected_auth=$2
    [ -r "$state_file" ] || exit 42
    IFS=: read -r state auth pgid command_rc < "$state_file" || exit 43
    [ "$auth" = "$expected_auth" ] || exit 49
    case "$state" in
      done|terminated) rm -f -- "$state_file"; exit 0 ;;
      running) ;;
      *) exit 44 ;;
    esac
    case "$pgid" in *[!0-9]*|"") exit 45 ;; esac
    kill -TERM "-$pgid" 2>/dev/null || true
    i=0
    while kill -0 "-$pgid" 2>/dev/null && [ "$i" -lt 40 ]; do
      sleep 0.05
      i=$((i + 1))
    done
    if kill -0 "-$pgid" 2>/dev/null; then
      kill -KILL "-$pgid" 2>/dev/null || true
      i=0
      while kill -0 "-$pgid" 2>/dev/null && [ "$i" -lt 20 ]; do
        sleep 0.05
        i=$((i + 1))
      done
    fi
    kill -0 "-$pgid" 2>/dev/null && exit 46
    printf "terminated:%s:%s\n" "$auth" "$pgid" > "$state_file" || exit 47
    rm -f -- "$state_file" || exit 48
  ' sh "$state_file" "$expected_auth" __terminate__ > "$STATE_DIR/container-terminate.log" 2>&1 &
  killer=$!
  ACTIVE_CHILD_PID="$killer"
  for ((i = 0; i < 120; i++)); do
    kill -0 "$killer" 2>/dev/null || break
    sleep 0.05
  done
  if kill -0 "$killer" 2>/dev/null; then
    kill -KILL "$killer" 2>/dev/null || true
    wait "$killer" 2>/dev/null || true
    ACTIVE_CHILD_PID=""
    return 1
  fi
  wait "$killer"
  rc=$?
  ACTIVE_CHILD_PID=""
  [[ $rc -eq 0 ]] || return 1
  ACTIVE_CONTAINER_STATE=""
  ACTIVE_CONTAINER_AUTH=""
  CONTAINER_SAFE_TO_RESTORE=1
  return 0
}

on_signal() {
  local signal="$1" exit_code="$2" child="$ACTIVE_CHILD_PID" i
  trap - INT HUP TERM
  if [[ -n "$child" ]]; then
    kill -s "$signal" "$child" 2>/dev/null || true
    for ((i = 0; i < 40; i++)); do
      kill -0 "$child" 2>/dev/null || break
      sleep 0.05
    done
    if kill -0 "$child" 2>/dev/null; then kill -KILL "$child" 2>/dev/null || true; fi
    wait "$child" 2>/dev/null || true
    ACTIVE_CHILD_PID=""
  fi
  if [[ $CONTAINER_SAFE_TO_RESTORE -ne 1 && -n "$ACTIVE_CONTAINER_STATE" ]] \
    && ! terminate_container_command; then
    CONTAINER_SAFE_TO_RESTORE=0
    printf 'mutmut-verify: ERROR: could not confirm in-container process termination; refusing restoration\n' >&2
    exit 75
  fi
  exit "$exit_code"
}
trap 'on_signal INT 130' INT
trap 'on_signal HUP 129' HUP
trap 'on_signal TERM 143' TERM

MANIFEST_CANON="$(readlink -f -- "$MANIFEST")" || die "could not canonicalize manifest"
[[ -f "$MANIFEST_CANON" && ! -d "$MANIFEST_CANON" ]] || die "manifest is not a regular file"
MANIFEST_SNAPSHOT="$STATE_DIR/manifest.tsv"
cp -- "$MANIFEST_CANON" "$MANIFEST_SNAPSHOT" || die "could not snapshot manifest"
chmod 400 "$MANIFEST_SNAPSHOT" || die "could not protect manifest snapshot"

# ----------------------------------------------------------------- manifest safety
EXPECTED_HEADER=$'mutant_id\tfile\tapply_method\ttest_node\texpect_clean_rc\texpect_mutant_rc_nonzero'
IFS= read -r MANIFEST_HEADER < "$MANIFEST_SNAPSHOT"
[[ "$MANIFEST_HEADER" == "$EXPECTED_HEADER" ]] || {
  die "invalid manifest header; expected: $EXPECTED_HEADER"; }

MANIFEST_COUNT_FILE="$STATE_DIR/manifest-count"
if ! awk -F'\t' -v count_file="$MANIFEST_COUNT_FILE" '
  NR == 1 { next }
  {
    rows++
    if (NF != 6) {
      printf "invalid manifest row %d: expected 6 columns, got %d\n", NR, NF > "/dev/stderr"
      bad = 1
      next
    }
    for (i = 1; i <= 6; i++) {
      if ($i == "") {
        printf "invalid manifest row %d: column %d is empty\n", NR, i > "/dev/stderr"
        bad = 1
      }
    }
    if (seen[$1]++) {
      printf "invalid manifest row %d: duplicate mutant_id %s\n", NR, $1 > "/dev/stderr"
      bad = 1
    }
    if ($3 != "mutmut_apply" && $3 !~ /^diff:\/.+/) {
      printf "invalid manifest row %d: unsupported apply_method %s\n", NR, $3 > "/dev/stderr"
      bad = 1
    }
    if ($5 != "0") {
      printf "invalid manifest row %d: expect_clean_rc must be 0\n", NR > "/dev/stderr"
      bad = 1
    }
    if ($6 != "true" && $6 != "false") {
      printf "invalid manifest row %d: expect_mutant_rc_nonzero must be true or false\n", NR > "/dev/stderr"
      bad = 1
    }
  }
  END {
    print rows + 0 > count_file
    if (rows == 0) {
      print "invalid manifest: at least one data row is required" > "/dev/stderr"
      bad = 1
    }
    exit bad
  }
' "$MANIFEST_SNAPSHOT"; then
  exit 2
fi
IFS= read -r MANIFEST_ROW_COUNT < "$MANIFEST_COUNT_FILE"
[[ "$MANIFEST_ROW_COUNT" =~ ^[1-9][0-9]*$ ]] || die "invalid manifest row count"

# Index flags can hide modified tracked files from ordinary diff commands. The
# baseline contains none; later scans therefore treat every such flag as a
# transaction side effect.
declare -a SCANNED_UNSAFE_INDEX_PATHS=()
scan_unsafe_index_flags() {
  local index_entry index_tag index_path
  SCANNED_UNSAFE_INDEX_PATHS=()
  git -C "$GIT_ROOT" ls-files -v -z > "$STATE_DIR/index-flags" || return 1
  while IFS= read -r -d '' index_entry; do
    index_tag="${index_entry:0:1}"
    index_path="${index_entry:2}"
    case "$index_tag" in
      S|[a-z]) SCANNED_UNSAFE_INDEX_PATHS+=("$index_path") ;;
    esac
  done < "$STATE_DIR/index-flags"
}

scan_unsafe_index_flags || die "could not inspect Git index flags"
if [[ ${#SCANNED_UNSAFE_INDEX_PATHS[@]} -gt 0 ]]; then
  die "tracked path uses skip-worktree or assume-unchanged: ${SCANNED_UNSAFE_INDEX_PATHS[0]}"
fi

BASELINE_HEAD="$(git -C "$GIT_ROOT" rev-parse --verify HEAD 2>/dev/null)" \
  || die "could not resolve source HEAD"
git -C "$GIT_ROOT" diff --quiet -- . || die "tracked source has unstaged changes before verification"
git -C "$GIT_ROOT" diff --cached --quiet "$BASELINE_HEAD" -- . \
  || die "tracked source has staged changes before verification"

# Validate storage aliases for every tracked path under --src, not just manifest
# targets. In-tree symlinks are allowed only when their fully resolved target is
# regular, tracked by this same worktree, and itself single-link.
declare -a SCANNED_UNSAFE_STORAGE_PATHS=()
scan_source_storage() {
  local allow_missing="${1:-0}" path full resolved resolved_rel links
  SCANNED_UNSAFE_STORAGE_PATHS=()
  git -C "$GIT_ROOT" ls-files -z -- "$SRC_FROM_ROOT/" > "$STATE_DIR/source-tracked" \
    || return 1
  while IFS= read -r -d '' path; do
    full="$GIT_ROOT/$path"
    if [[ -L "$full" ]]; then
      resolved="$(readlink -f -- "$full" 2>/dev/null)" || {
        SCANNED_UNSAFE_STORAGE_PATHS+=("$path"); continue; }
      case "$resolved" in
        "$GIT_ROOT"/*) ;;
        *) SCANNED_UNSAFE_STORAGE_PATHS+=("$path"); continue ;;
      esac
      resolved_rel="${resolved#"$GIT_ROOT"/}"
      mapfile -d '' -t RESOLVED_TRACKED_PATHS < <(
        git -C "$GIT_ROOT" ls-files -z -- "$resolved_rel")
      if [[ ${#RESOLVED_TRACKED_PATHS[@]} -ne 1 \
        || "${RESOLVED_TRACKED_PATHS[0]}" != "$resolved_rel" \
        || ! -f "$resolved" || -L "$resolved" ]]; then
        SCANNED_UNSAFE_STORAGE_PATHS+=("$path")
        continue
      fi
      links="$(stat -c '%h' -- "$resolved" 2>/dev/null)" || links=0
      [[ "$links" -eq 1 ]] || SCANNED_UNSAFE_STORAGE_PATHS+=("$path")
    elif [[ -e "$full" ]]; then
      if [[ ! -f "$full" ]]; then
        SCANNED_UNSAFE_STORAGE_PATHS+=("$path")
        continue
      fi
      links="$(stat -c '%h' -- "$full" 2>/dev/null)" || links=0
      [[ "$links" -eq 1 ]] || SCANNED_UNSAFE_STORAGE_PATHS+=("$path")
    elif [[ "$allow_missing" -ne 1 ]]; then
      SCANNED_UNSAFE_STORAGE_PATHS+=("$path")
    fi
  done < "$STATE_DIR/source-tracked"
}

scan_source_storage 0 || die "could not inspect tracked source storage"
if [[ ${#SCANNED_UNSAFE_STORAGE_PATHS[@]} -gt 0 ]]; then
  die "tracked source path has unsafe hardlink, symlink, or file type: ${SCANNED_UNSAFE_STORAGE_PATHS[0]}"
fi

# Validate exact tracked targets, then snapshot and preflight every external diff.
declare -a PATCH_SNAPSHOTS=()
declare -a PATCH_SOURCE_CANONS=()
manifest_index=0
{
  read -r _header
  while IFS=$'\t' read -r mid file method node exp_clean exp_mut_nz \
    || [[ -n "${mid:-}${file:-}${method:-}${node:-}${exp_clean:-}${exp_mut_nz:-}" ]]; do
    ((manifest_index += 1))
    case "$file" in
      /*|.|./*|..|../*|*/./*|*/.|*/../*|*/..) die "unsafe manifest file path: $file" ;;
    esac
    expected_path="${SRC_FROM_ROOT}/${file}"
    mapfile -d '' -t MATCHED_PATHS < <(git -C "$GIT_ROOT" ls-files -z -- "$expected_path")
    [[ ${#MATCHED_PATHS[@]} -eq 1 && "${MATCHED_PATHS[0]}" == "$expected_path" ]] || {
      die "manifest file is not one exact tracked path under $SRC: $file"; }
    [[ -f "$SRC/$file" && ! -L "$SRC/$file" ]] \
      || die "manifest target must be a regular non-symlink file: $file"
    target_links="$(stat -c '%h' -- "$SRC/$file" 2>/dev/null)" \
      || die "could not inspect manifest target: $file"
    [[ "$target_links" -eq 1 ]] || die "manifest target must have exactly one hard link: $file"
    if [[ "$method" == diff:* ]]; then
      patch_source="${method#diff:}"
      [[ -f "$patch_source" ]] || die "manifest patch not found: $patch_source"
      patch_source_canon="$(readlink -f -- "$patch_source")" \
        || die "could not canonicalize patch for $mid: $patch_source"
      [[ -f "$patch_source_canon" ]] || die "manifest patch is not a regular file: $patch_source"
      PATCH_SOURCE_CANONS+=("$patch_source_canon")
      patch_snapshot="$STATE_DIR/patch-$(printf '%06d' "$manifest_index").diff"
      cp -- "$patch_source_canon" "$patch_snapshot" || die "could not snapshot patch for $mid"
      chmod 400 "$patch_snapshot" || die "could not protect patch snapshot for $mid"
      run_external git -C "$GIT_ROOT" apply --check -- "$patch_snapshot" >/dev/null 2>&1 \
        || die "patch does not apply cleanly for $mid: $patch_source"
      run_external git -C "$GIT_ROOT" apply --numstat -z -- "$patch_snapshot" \
        > "$STATE_DIR/patch-paths" 2>/dev/null || die "could not inspect patch paths for $mid"
      declare -a patch_paths=()
      while IFS= read -r -d '' numstat_entry; do
        [[ "$numstat_entry" == *$'\t'*$'\t'* ]] \
          || die "patch has an unsupported path encoding for $mid"
        patch_path="${numstat_entry#*$'\t'}"
        patch_path="${patch_path#*$'\t'}"
        [[ -n "$patch_path" ]] || die "patch rename/copy is not supported for $mid"
        patch_paths+=("$patch_path")
      done < "$STATE_DIR/patch-paths"
      [[ ${#patch_paths[@]} -eq 1 && "${patch_paths[0]}" == "$expected_path" ]] || {
        die "patch for $mid must touch exactly declared path $expected_path"; }
      PATCH_SNAPSHOTS[$manifest_index]="$patch_snapshot"
    fi
  done
} < "$MANIFEST_SNAPSHOT"
[[ $manifest_index -eq $MANIFEST_ROW_COUNT ]] || die "manifest changed while creating immutable snapshot"

# Docker inspection happens only after all mutable input files have snapshots.
command -v docker >/dev/null 2>&1 || die "docker is unavailable (binary not found)"
run_external docker inspect "$CON" >/dev/null 2>&1 || die "container not found: $CON"
run_external docker inspect "$CON" \
  --format '{{range .Mounts}}{{if eq .Destination "/work"}}{{printf "%s\t%t" .Source .RW}}{{end}}{{end}}' \
  > "$STATE_DIR/container-mount" 2>/dev/null || die "could not inspect container mount: $CON"
IFS=$'\t' read -r CONTAINER_ROOT CONTAINER_ROOT_RW < "$STATE_DIR/container-mount"
[[ -n "$CONTAINER_ROOT" ]] || {
  die "container $CON has no bind mount at /work; recreate it with --mount type=bind,source=$ROOT,target=/work"; }
CONTAINER_ROOT="$(readlink -f -- "$CONTAINER_ROOT")" || die "could not canonicalize container /work source"
[[ "$CONTAINER_ROOT" == "$ROOT" ]] || {
  die "container /work mount mismatch: container uses $CONTAINER_ROOT, but --root resolves to $ROOT"; }
[[ "$CONTAINER_ROOT_RW" == "true" ]] || {
  die "container /work mount is read-only; mutation verification requires a read-write bind mount"; }

# ----------------------------------------------------------------- output safety
path_has_symlink_component() {
  local path="$1"
  [[ "$path" == /* ]] || path="$(pwd -P)/$path"
  while [[ "$path" != "/" && -n "$path" ]]; do
    [[ -L "$path" ]] && return 0
    path="${path%/*}"
    [[ -n "$path" ]] || path="/"
  done
  return 1
}

output_is_tracked() {
  local path="$1" rel
  case "$path" in
    "$GIT_ROOT"/*)
      rel="${path#"$GIT_ROOT"/}"
      git -C "$GIT_ROOT" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1
      ;;
    *) return 1 ;;
  esac
}

validate_output_destination() {
  local path="$1" links patch_source_canon
  path_has_symlink_component "$path" && return 1
  if [[ -e "$path" ]]; then
    [[ -f "$path" && ! -L "$path" ]] || return 1
    links="$(stat -c '%h' -- "$path" 2>/dev/null)" || return 1
    [[ "$links" -eq 1 ]] || return 1
    [[ ! "$path" -ef "$MANIFEST_CANON" ]] || return 1
    for patch_source_canon in "${PATCH_SOURCE_CANONS[@]}"; do
      [[ ! "$path" -ef "$patch_source_canon" ]] || return 1
    done
  fi
  output_is_tracked "$path" && return 1
  return 0
}

output_destination_fingerprint() {
  local path="$1" metadata digest
  if [[ ! -e "$path" && ! -L "$path" ]]; then
    printf 'absent\n'
    return 0
  fi
  validate_output_destination "$path" || return 1
  metadata="$(stat -c '%d:%i:%h:%f:%s' -- "$path" 2>/dev/null)" || return 1
  digest="$(sha256sum -- "$path" 2>/dev/null | awk '{print $1}')" || return 1
  printf 'present:%s:%s\n' "$metadata" "$digest"
}

path_has_symlink_component "$OUT" && die "output path contains a symlink: $OUT"
OUT_ABS="$(readlink -m -- "$OUT")" || die "could not canonicalize output path"
[[ "$OUT_ABS" != "$MANIFEST_CANON" ]] || die "output directory aliases manifest"
mkdir -p "$OUT_ABS" || die "cannot create output directory: $OUT_ABS"
path_has_symlink_component "$OUT_ABS" && die "output path contains a symlink: $OUT_ABS"

KM_FINAL="$OUT_ABS/kill_matrix.tsv"
REPORT_FINAL="$OUT_ABS/verify-report.txt"
LIVE_OUTPUT_DIR="$STATE_DIR/output"
mkdir -p "$LIVE_OUTPUT_DIR" || die "cannot create protected output directory"
KM="$LIVE_OUTPUT_DIR/kill_matrix.tsv"
REPORT="$LIVE_OUTPUT_DIR/verify-report.txt"
declare -a OUTPUT_FILES=("$KM_FINAL")
declare -a LIVE_OUTPUT_FILES=("$KM")
for ((i = 1; i <= MANIFEST_ROW_COUNT; i++)); do
  OUTPUT_FILES+=("$OUT_ABS/row-$(printf '%06d' "$i")-clean.log")
  OUTPUT_FILES+=("$OUT_ABS/row-$(printf '%06d' "$i")-mutant.log")
  LIVE_OUTPUT_FILES+=("$LIVE_OUTPUT_DIR/row-$(printf '%06d' "$i")-clean.log")
  LIVE_OUTPUT_FILES+=("$LIVE_OUTPUT_DIR/row-$(printf '%06d' "$i")-mutant.log")
done
# Publish the human-readable success/failure report last so it is also the
# completion marker for the complete artifact set.
OUTPUT_FILES+=("$REPORT_FINAL")
LIVE_OUTPUT_FILES+=("$REPORT")
declare -a OUTPUT_BASE_FINGERPRINTS=()
for output_file in "${OUTPUT_FILES[@]}"; do
  validate_output_destination "$output_file" \
    || die "unsafe output destination (symlink, hardlink, alias, or tracked file): $output_file"
  OUTPUT_BASE_FINGERPRINTS+=("$(output_destination_fingerprint "$output_file")") \
    || die "could not fingerprint output destination: $output_file"
done
for output_file in "${LIVE_OUTPUT_FILES[@]}"; do
  : > "$output_file" || die "cannot initialize protected output file: $output_file"
done

# Existing untracked contents are outside the restoration contract; their names
# are kept so cleanup removes only paths newly introduced by a transaction.
declare -A BASELINE_UNTRACKED=()
git -C "$GIT_ROOT" ls-files --others --exclude-standard -z -- . \
  > "$STATE_DIR/baseline-untracked" || die "could not capture untracked-path baseline"
while IFS= read -r -d '' path; do BASELINE_UNTRACKED["$path"]=1; done \
  < "$STATE_DIR/baseline-untracked"

# ----------------------------------------------------------------- transaction helpers
declare -a CHANGED_TRACKED=()
declare -a NEW_UNTRACKED=()
declare -a ACTUAL_CHANGED=()
declare -a UNSAFE_INDEX_PATHS=()
declare -a UNSAFE_TARGET_PATHS=()
INDEX_FLAGS_TAMPERED=0
TARGET_LINKS_TAMPERED=0
TRANSACTION_ACTIVE=0

collect_changes() {
  local path
  declare -A seen=()
  declare -A seen_tracked=()
  CHANGED_TRACKED=()
  NEW_UNTRACKED=()
  ACTUAL_CHANGED=()
  UNSAFE_INDEX_PATHS=()
  UNSAFE_TARGET_PATHS=()
  INDEX_FLAGS_TAMPERED=0
  TARGET_LINKS_TAMPERED=0

  # Clear flags introduced after the safe baseline before asking Git for the
  # actual delta; otherwise both diff and restore can silently omit the path.
  scan_unsafe_index_flags || return 1
  if [[ ${#SCANNED_UNSAFE_INDEX_PATHS[@]} -gt 0 ]]; then
    INDEX_FLAGS_TAMPERED=1
    UNSAFE_INDEX_PATHS=("${SCANNED_UNSAFE_INDEX_PATHS[@]}")
    for path in "${UNSAFE_INDEX_PATHS[@]}"; do
      git -C "$GIT_ROOT" update-index --no-skip-worktree -- "$path" >/dev/null 2>&1 \
        || return 1
      git -C "$GIT_ROOT" update-index --no-assume-unchanged -- "$path" >/dev/null 2>&1 \
        || return 1
      if [[ -z "${seen[$path]+x}" ]]; then ACTUAL_CHANGED+=("$path"); seen["$path"]=1; fi
    done
    scan_unsafe_index_flags || return 1
    [[ ${#SCANNED_UNSAFE_INDEX_PATHS[@]} -eq 0 ]] || return 1
  fi

  scan_source_storage 1 || return 1
  for path in "${SCANNED_UNSAFE_STORAGE_PATHS[@]}"; do
    # A deleted target is represented by Git's tracked delta and is safe to
    # restore. Existing targets must remain regular, non-symlink, single-link
    # files so restoration cannot write through an alias.
    TARGET_LINKS_TAMPERED=1
    UNSAFE_TARGET_PATHS+=("$path")
    CHANGED_TRACKED+=("$path")
    seen_tracked["$path"]=1
    if [[ -z "${seen[$path]+x}" ]]; then ACTUAL_CHANGED+=("$path"); seen["$path"]=1; fi
  done

  git -C "$GIT_ROOT" diff --name-only --no-renames -z "$BASELINE_HEAD" -- . \
    > "$STATE_DIR/tracked" || return 1
  while IFS= read -r -d '' path; do
    if [[ -z "${seen_tracked[$path]+x}" ]]; then CHANGED_TRACKED+=("$path"); seen_tracked["$path"]=1; fi
    if [[ -z "${seen[$path]+x}" ]]; then ACTUAL_CHANGED+=("$path"); seen["$path"]=1; fi
  done < "$STATE_DIR/tracked"

  git -C "$GIT_ROOT" ls-files --others --exclude-standard -z -- . \
    > "$STATE_DIR/untracked" || return 1
  while IFS= read -r -d '' path; do
    if [[ -z "${BASELINE_UNTRACKED[$path]+x}" ]]; then
      NEW_UNTRACKED+=("$path")
      if [[ -z "${seen[$path]+x}" ]]; then ACTUAL_CHANGED+=("$path"); seen["$path"]=1; fi
    fi
  done < "$STATE_DIR/untracked"
}

baseline_matches() {
  local path current_head
  declare -A current_untracked=()

  current_head="$(git -C "$GIT_ROOT" rev-parse --verify HEAD 2>/dev/null)" || return 1
  [[ "$current_head" == "$BASELINE_HEAD" ]] || return 1
  scan_unsafe_index_flags || return 1
  [[ ${#SCANNED_UNSAFE_INDEX_PATHS[@]} -eq 0 ]] || return 1
  scan_source_storage 0 || return 1
  [[ ${#SCANNED_UNSAFE_STORAGE_PATHS[@]} -eq 0 ]] || return 1
  git -C "$GIT_ROOT" diff --quiet -- . || return 1
  git -C "$GIT_ROOT" diff --cached --quiet "$BASELINE_HEAD" -- . || return 1
  git -C "$GIT_ROOT" ls-files --others --exclude-standard -z -- . \
    > "$STATE_DIR/current-untracked" || return 1
  while IFS= read -r -d '' path; do current_untracked["$path"]=1; done \
    < "$STATE_DIR/current-untracked"
  [[ ${#current_untracked[@]} -eq ${#BASELINE_UNTRACKED[@]} ]] || return 1
  for path in "${!BASELINE_UNTRACKED[@]}"; do
    [[ -n "${current_untracked[$path]+x}" ]] || return 1
  done
}

describe_changes() {
  local path first=1
  if [[ ${#ACTUAL_CHANGED[@]} -eq 0 ]]; then printf '<none>'; return; fi
  for path in "${ACTUAL_CHANGED[@]}"; do
    [[ $first -eq 1 ]] || printf ','
    printf '%q' "$path"
    first=0
  done
}

describe_unsafe_index_flags() {
  local path first=1
  if [[ ${#UNSAFE_INDEX_PATHS[@]} -eq 0 ]]; then printf '<none>'; return; fi
  for path in "${UNSAFE_INDEX_PATHS[@]}"; do
    [[ $first -eq 1 ]] || printf ','
    printf '%q' "$path"
    first=0
  done
}

describe_unsafe_targets() {
  local path first=1
  if [[ ${#UNSAFE_TARGET_PATHS[@]} -eq 0 ]]; then printf '<none>'; return; fi
  for path in "${UNSAFE_TARGET_PATHS[@]}"; do
    [[ $first -eq 1 ]] || printf ','
    printf '%q' "$path"
    first=0
  done
}

restore_transaction() {
  local path target_links restore_rc=0
  [[ $TRANSACTION_ACTIVE -eq 1 ]] || return 0
  [[ $CONTAINER_SAFE_TO_RESTORE -eq 1 ]] || return 1
  collect_changes || return 1
  if [[ ${#CHANGED_TRACKED[@]} -gt 0 ]]; then
    for path in "${CHANGED_TRACKED[@]}"; do
      if [[ -e "$GIT_ROOT/$path" || -L "$GIT_ROOT/$path" ]]; then
        target_links="$(stat -c '%h' -- "$GIT_ROOT/$path" 2>/dev/null)" || target_links=0
        if [[ -L "$GIT_ROOT/$path" || ! -f "$GIT_ROOT/$path" || "$target_links" -ne 1 ]]; then
          if [[ -d "$GIT_ROOT/$path" && ! -L "$GIT_ROOT/$path" ]]; then
            restore_rc=1
          else
            rm -f -- "$GIT_ROOT/$path" || restore_rc=1
          fi
        fi
      fi
    done
    git -C "$GIT_ROOT" restore --source="$BASELINE_HEAD" --staged --worktree -- \
      "${CHANGED_TRACKED[@]}" >/dev/null 2>&1 || restore_rc=1
  fi
  for path in "${NEW_UNTRACKED[@]}"; do
    if [[ -d "$GIT_ROOT/$path" && ! -L "$GIT_ROOT/$path" ]]; then
      restore_rc=1
    else
      rm -f -- "$GIT_ROOT/$path" || restore_rc=1
    fi
  done
  if [[ $restore_rc -eq 0 ]] && baseline_matches; then
    TRANSACTION_ACTIVE=0
    return 0
  fi
  return 1
}

on_exit() {
  local rc=$?
  trap - EXIT INT HUP TERM
  if [[ $TRANSACTION_ACTIVE -eq 1 ]]; then
    if [[ $CONTAINER_SAFE_TO_RESTORE -ne 1 ]]; then
      printf 'mutmut-verify: ERROR: restoration withheld because an in-container process may still be running\n' >&2
      rc=75
    elif ! restore_transaction; then
      printf 'mutmut-verify: ERROR: failed to restore mutation transaction during exit\n' >&2
      rc=70
    fi
  fi
  if [[ -n "${PUBLISH_TEMP:-}" ]]; then rm -f -- "$PUBLISH_TEMP" 2>/dev/null || true; fi
  rm -rf -- "$STATE_DIR"
  exit "$rc"
}
trap on_exit EXIT

# ----------------------------------------------------------------- output helpers
output_failure() {
  printf 'mutmut-verify: ERROR: output file became unsafe or unwritable: %s\n' "$1" >&2
  exit 74
}

ensure_live_output() {
  local path="$1" links
  case "$path" in "$LIVE_OUTPUT_DIR"/*) ;; *) return 1 ;; esac
  [[ -f "$path" && ! -L "$path" ]] || return 1
  links="$(stat -c '%h' -- "$path" 2>/dev/null)" || return 1
  [[ "$links" -eq 1 ]] || return 1
  return 0
}

report_line() {
  local line="$1"
  ensure_live_output "$REPORT" || output_failure "$REPORT"
  printf '%s\n' "$line" >> "$REPORT" || output_failure "$REPORT"
}

matrix_header() {
  ensure_live_output "$KM" || output_failure "$KM"
  printf 'mutant_id\tfile\tbase_rc\tmut_rc\trevert\tverdict\tdetail\n' > "$KM" \
    || output_failure "$KM"
}

PUBLISH_TEMP=""
publish_outputs() {
  local i live final current_fingerprint
  # Validate the complete destination set before replacing any member.
  for ((i = 0; i < ${#OUTPUT_FILES[@]}; i++)); do
    final="${OUTPUT_FILES[$i]}"
    current_fingerprint="$(output_destination_fingerprint "$final")" \
      || output_failure "$final"
    [[ "$current_fingerprint" == "${OUTPUT_BASE_FINGERPRINTS[$i]}" ]] \
      || output_failure "$final"
  done
  for ((i = 0; i < ${#OUTPUT_FILES[@]}; i++)); do
    live="${LIVE_OUTPUT_FILES[$i]}"
    final="${OUTPUT_FILES[$i]}"
    ensure_live_output "$live" || output_failure "$final"
    PUBLISH_TEMP="$(mktemp "$OUT_ABS/.mutmut-verify-publish.XXXXXXXX")" \
      || output_failure "$final"
    cp -- "$live" "$PUBLISH_TEMP" || output_failure "$final"
    chmod 644 "$PUBLISH_TEMP" || output_failure "$final"
    mv -fT -- "$PUBLISH_TEMP" "$final" || output_failure "$final"
    PUBLISH_TEMP=""
  done
}

# ----------------------------------------------------------------- execution helpers
CONTAINER_COMMAND_RC=-1
CONTAINER_COMMAND_ERROR=""

run_container_command() {
  local log="$1" add_pythonpath="$2" marker token docker_rc last_line parsed completed_state
  shift 2
  local -a docker_args=(docker exec)
  CONTAINER_COMMAND_RC=-1
  CONTAINER_COMMAND_ERROR=""
  token="mutmut_verify_${$}_${RANDOM}_${RANDOM}"
  ACTIVE_CONTAINER_STATE="/tmp/${token}.state"
  ACTIVE_CONTAINER_AUTH="$token"
  marker="__MUTMUT_VERIFY_COMMAND_RC_${token}__="
  CONTAINER_SAFE_TO_RESTORE=0
  if [[ "$add_pythonpath" -eq 1 ]]; then docker_args+=(-e "PYTHONPATH=$PROJ"); fi
  docker_args+=(-w "$PROJ" "$CON" sh -c '
    state_file=$1
    marker=$2
    auth=$3
    shift 3
    state_tmp="${state_file}.tmp.$$"
    if ! command -v setsid >/dev/null 2>&1; then
      printf "launch-failed:%s:127\n" "$auth" > "$state_tmp" && mv -f -- "$state_tmp" "$state_file"
      printf "\n%sINFRA:127\n" "$marker"
      exit 0
    fi
    setsid -w sh -c '\''
      state_file=$1
      auth=$2
      shift 2
      state_tmp="${state_file}.tmp.$$"
      printf "running:%s:%s\n" "$auth" "$$" > "$state_tmp" || exit 125
      mv -f -- "$state_tmp" "$state_file" || exit 125
      exec "$@"
    '\'' sh "$state_file" "$auth" "$@" &
    runner=$!
    wait "$runner"
    rc=$?
    state_tmp="${state_file}.tmp.$$"
    if IFS=: read -r state seen_auth pgid < "$state_file" 2>/dev/null \
      && [ "$state" = running ] && [ "$seen_auth" = "$auth" ] \
      && case "$pgid" in *[!0-9]*|"") false ;; *) true ;; esac; then
      if printf "done:%s:%s:%s\n" "$auth" "$pgid" "$rc" > "$state_tmp" \
        && mv -f -- "$state_tmp" "$state_file"; then
        printf "\n%s%s\n" "$marker" "$rc"
      else
        printf "\n%sINFRA:125\n" "$marker"
      fi
    else
      printf "launch-failed:%s:%s\n" "$auth" "$rc" > "$state_tmp" \
        && mv -f -- "$state_tmp" "$state_file"
      printf "\n%sINFRA:%s\n" "$marker" "$rc"
    fi
    exit 0
  ' sh "$ACTIVE_CONTAINER_STATE" "$marker" "$token" "$@")

  run_external "${docker_args[@]}" > "$log" 2>&1
  docker_rc=$?
  if [[ $docker_rc -eq 0 ]]; then
    last_line="$(tail -n 1 -- "$log" 2>/dev/null)" || last_line=""
    last_line="${last_line%$'\r'}"
    if [[ "$last_line" == "$marker"* ]]; then
      parsed="${last_line#"$marker"}"
      if [[ "$parsed" =~ ^[0-9]+$ ]]; then
        CONTAINER_COMMAND_RC="$parsed"
        CONTAINER_SAFE_TO_RESTORE=1
        # The trusted marker proves the command group finished. State cleanup is
        # best effort; a unique token prevents stale files being reused.
        completed_state="$ACTIVE_CONTAINER_STATE"
        ACTIVE_CONTAINER_STATE=""
        ACTIVE_CONTAINER_AUTH=""
        run_external docker exec "$CON" rm -f -- "$completed_state" \
          >/dev/null 2>&1 || true
        return 0
      elif [[ "$parsed" =~ ^INFRA:([0-9]+)$ ]]; then
        CONTAINER_COMMAND_ERROR="container command launcher failed before authenticated start (rc=${BASH_REMATCH[1]})"
        CONTAINER_SAFE_TO_RESTORE=1
        completed_state="$ACTIVE_CONTAINER_STATE"
        ACTIVE_CONTAINER_STATE=""
        ACTIVE_CONTAINER_AUTH=""
        run_external docker exec "$CON" rm -f -- "$completed_state" \
          >/dev/null 2>&1 || true
        return 1
      fi
    fi
  fi

  CONTAINER_COMMAND_ERROR="docker exec did not return a trusted completion marker (rc=$docker_rc)"
  if terminate_container_command; then
    CONTAINER_SAFE_TO_RESTORE=1
  else
    CONTAINER_SAFE_TO_RESTORE=0
  fi
  return 1
}

NODE_RC=-1
NODE_INFRA=""
run_node() {
  local node="$1" log="$2" published_log="$3"
  NODE_RC=-1
  NODE_INFRA=""
  ensure_live_output "$log" || { NODE_INFRA="unsafe protected pytest log for $published_log"; return 1; }
  : > "$log" || { NODE_INFRA="could not write protected pytest log for $published_log"; return 1; }
  if ! run_container_command "$log" 1 pytest -p no:cacheprovider -m unit -q "$node"; then
    NODE_INFRA="$CONTAINER_COMMAND_ERROR; log=$published_log"
    return 1
  fi
  NODE_RC="$CONTAINER_COMMAND_RC"
  return 0
}

apply_mutant() {
  local method="$1" mid="$2" row_index="$3"
  case "$method" in
    mutmut_apply)
      if run_container_command "$STATE_DIR/apply-command.log" 0 mutmut apply "$mid"; then
        return "$CONTAINER_COMMAND_RC"
      fi
      return 125 ;;
    diff:*)
      run_external git -C "$GIT_ROOT" apply "${PATCH_SNAPSHOTS[$row_index]}" \
        >/dev/null 2>&1 </dev/null ;;
    *) return 90 ;;
  esac
}

# Fingerprint both the worktree-to-HEAD and index-to-HEAD deltas for the declared
# target. The second snapshot catches a mutant test that edits (or stages) the
# already-approved path without introducing an additional changed pathname.
fingerprint_target_delta() {
  local path="$1" snapshot="$STATE_DIR/target-fingerprint-input"
  if [[ -e "$GIT_ROOT/$path" || -L "$GIT_ROOT/$path" ]]; then
    stat -c 'PATH:%d:%i:%h:%f:%s' -- "$GIT_ROOT/$path" > "$snapshot" || return 1
  else
    printf 'PATH:absent\n' > "$snapshot" || return 1
  fi
  git -C "$GIT_ROOT" diff --binary --full-index --no-ext-diff --no-textconv \
    "$BASELINE_HEAD" -- "$path" >> "$snapshot" || return 1
  printf '\0INDEX\0' >> "$snapshot" || return 1
  git -C "$GIT_ROOT" diff --cached --binary --full-index --no-ext-diff --no-textconv \
    "$BASELINE_HEAD" -- "$path" >> "$snapshot" || return 1
  git -C "$GIT_ROOT" hash-object "$snapshot" 2>/dev/null
}

# ----------------------------------------------------------------- report setup
ensure_live_output "$REPORT" || output_failure "$REPORT_FINAL"
: > "$REPORT" || output_failure "$REPORT"
matrix_header
report_line "mutmut-verify — $(git -C "$SRC" rev-parse --short HEAD) — $(date -u +%FT%TZ)"
report_line "container=$CON  src=$SRC  proj=$PROJ"
report_line "============================================================"
report_line "$(printf '%-28s %-12s %s' MUTANT VERDICT DETAIL)"

overall_ok=1
row_count=0
killed_count=0
expected_pass_count=0
bad_count=0
inconclusive_count=0

record_result() {
  local mid="$1" file="$2" base_rc="$3" mut_rc="$4" revert="$5" verdict="$6" detail="$7" line
  ((row_count += 1))
  case "$verdict" in
    KILLED)       ((killed_count += 1)) ;;
    OK)           ((expected_pass_count += 1)) ;;
    INCONCLUSIVE) ((inconclusive_count += 1)); overall_ok=0 ;;
    *)            ((bad_count += 1)); overall_ok=0 ;;
  esac
  detail="${detail//$'\t'/ }"
  detail="${detail//$'\n'/ }"
  ensure_live_output "$KM" || output_failure "$KM_FINAL"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "$verdict" "$detail" >> "$KM" \
    || output_failure "$KM"
  line="$(printf '%-28s %-12s %s' "$mid" "$verdict" "$detail")"
  report_line "$line"
}

# Main loop consumes only the protected snapshot and must account for every row.
manifest_index=0
{
  read -r _header
  while IFS=$'\t' read -r mid file method node exp_clean exp_mut_nz \
    || [[ -n "${mid:-}${file:-}${method:-}${node:-}${exp_clean:-}${exp_mut_nz:-}" ]]; do
    ((manifest_index += 1))
    detail=""; base_rc=-1; mut_rc=-1; revert="not-run"; verdict="BAD"
    expected_path="${SRC_FROM_ROOT}/${file}"
    clean_log_final="$OUT_ABS/row-$(printf '%06d' "$manifest_index")-clean.log"
    mutant_log_final="$OUT_ABS/row-$(printf '%06d' "$manifest_index")-mutant.log"
    clean_log="$LIVE_OUTPUT_DIR/row-$(printf '%06d' "$manifest_index")-clean.log"
    mutant_log="$LIVE_OUTPUT_DIR/row-$(printf '%06d' "$manifest_index")-mutant.log"

    if ! baseline_matches; then
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "LEAK" "BAD" \
        "source no longer matches the captured baseline before row"
      break
    fi

    # The clean test is itself a transaction: plugins and fixtures can write to
    # tracked source, and an interrupted clean run must be restored too.
    TRANSACTION_ACTIVE=1
    if ! run_node "$node" "$clean_log" "$clean_log_final"; then
      infra_detail="$NODE_INFRA"
      if [[ $CONTAINER_SAFE_TO_RESTORE -ne 1 ]]; then
        printf 'mutmut-verify: ERROR: clean-test container process could not be quiesced; refusing restoration\n' >&2
        exit 75
      fi
      if restore_transaction; then revert="ok"; else revert="LEAK"; fi
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "INCONCLUSIVE" "$infra_detail"
      [[ "$revert" == "ok" ]] || break
      continue
    fi
    base_rc="$NODE_RC"
    if ! collect_changes; then
      restore_transaction || true
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "LEAK" "BAD" \
        "could not discover paths changed by clean test; log=$clean_log_final"
      break
    fi
    changed_desc="$(describe_changes)"
    flag_desc="$(describe_unsafe_index_flags)"
    unsafe_target_desc="$(describe_unsafe_targets)"
    if [[ ${#ACTUAL_CHANGED[@]} -ne 0 ]]; then
      if restore_transaction; then revert="ok"; else revert="LEAK"; fi
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "BAD" \
        "clean test changed worktree paths: $changed_desc; unsafe_index_flags=$flag_desc unsafe_targets=$unsafe_target_desc; log=$clean_log_final"
      [[ "$revert" == "ok" ]] || break
      continue
    fi
    if restore_transaction; then revert="ok"; else revert="LEAK"; fi
    if [[ "$revert" != "ok" ]]; then
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "BAD" \
        "failed to close clean-test transaction; log=$clean_log_final"
      break
    fi
    if [[ "$base_rc" -ne 0 ]]; then
      IFS=$'\t' read -r verdict detail < <(
        classify_verdict "$base_rc" "$exp_clean" "$mut_rc" "$exp_mut_nz" "$revert")
      detail="$detail; log=$clean_log_final"
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "$verdict" "$detail"
      continue
    fi

    TRANSACTION_ACTIVE=1
    apply_mutant "$method" "$mid" "$manifest_index"
    apply_rc=$?
    if [[ $CONTAINER_SAFE_TO_RESTORE -ne 1 ]]; then
      printf 'mutmut-verify: ERROR: mutation command could not be quiesced; refusing restoration\n' >&2
      exit 75
    fi
    if ! collect_changes; then
      restore_transaction || true
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "LEAK" "BAD" \
        "could not discover paths changed by application"
      break
    fi
    changed_desc="$(describe_changes)"
    flag_desc="$(describe_unsafe_index_flags)"
    unsafe_target_desc="$(describe_unsafe_targets)"
    if [[ $apply_rc -ne 0 ]]; then
      if restore_transaction; then revert="ok"; else revert="LEAK"; fi
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "BAD" \
        "apply failed rc=$apply_rc; changed=$changed_desc; unsafe_index_flags=$flag_desc unsafe_targets=$unsafe_target_desc"
      [[ "$revert" == "ok" ]] || break
      continue
    fi
    if [[ $INDEX_FLAGS_TAMPERED -ne 0 || $TARGET_LINKS_TAMPERED -ne 0 \
      || ${#ACTUAL_CHANGED[@]} -ne 1 \
      || "${ACTUAL_CHANGED[0]:-}" != "$expected_path" ]]; then
      if restore_transaction; then revert="ok"; else revert="LEAK"; fi
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "BAD" \
        "application changed unexpected paths or metadata: declared=$expected_path actual=$changed_desc unsafe_index_flags=$flag_desc unsafe_targets=$unsafe_target_desc"
      [[ "$revert" == "ok" ]] || break
      continue
    fi
    if ! applied_fingerprint="$(fingerprint_target_delta "$expected_path")"; then
      restore_transaction || true
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "LEAK" "BAD" \
        "could not fingerprint applied target: $expected_path"
      break
    fi

    if ! run_node "$node" "$mutant_log" "$mutant_log_final"; then
      infra_detail="$NODE_INFRA"
      if [[ $CONTAINER_SAFE_TO_RESTORE -ne 1 ]]; then
        printf 'mutmut-verify: ERROR: mutant-test container process could not be quiesced; refusing restoration\n' >&2
        exit 75
      fi
      if restore_transaction; then revert="ok"; else revert="LEAK"; fi
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "INCONCLUSIVE" "$infra_detail"
      [[ "$revert" == "ok" ]] || break
      continue
    fi
    mut_rc="$NODE_RC"
    if ! collect_changes; then
      restore_transaction || true
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "LEAK" "BAD" \
        "could not discover paths changed by mutant test; log=$mutant_log_final"
      break
    fi
    changed_desc="$(describe_changes)"
    flag_desc="$(describe_unsafe_index_flags)"
    unsafe_target_desc="$(describe_unsafe_targets)"
    if [[ $INDEX_FLAGS_TAMPERED -ne 0 || $TARGET_LINKS_TAMPERED -ne 0 \
      || ${#ACTUAL_CHANGED[@]} -ne 1 \
      || "${ACTUAL_CHANGED[0]:-}" != "$expected_path" ]]; then
      if restore_transaction; then revert="ok"; else revert="LEAK"; fi
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "BAD" \
        "mutant test changed unexpected paths or metadata: declared=$expected_path actual=$changed_desc unsafe_index_flags=$flag_desc unsafe_targets=$unsafe_target_desc; log=$mutant_log_final"
      [[ "$revert" == "ok" ]] || break
      continue
    fi
    if ! post_test_fingerprint="$(fingerprint_target_delta "$expected_path")"; then
      restore_transaction || true
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "LEAK" "BAD" \
        "could not fingerprint target after mutant test: $expected_path; log=$mutant_log_final"
      break
    fi
    if [[ "$post_test_fingerprint" != "$applied_fingerprint" ]]; then
      if restore_transaction; then revert="ok"; else revert="LEAK"; fi
      record_result "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "BAD" \
        "mutant test modified declared target after application: $expected_path; log=$mutant_log_final"
      [[ "$revert" == "ok" ]] || break
      continue
    fi
    if restore_transaction; then revert="ok"; else revert="LEAK"; fi
    IFS=$'\t' read -r verdict detail < <(
      classify_verdict "$base_rc" "$exp_clean" "$mut_rc" "$exp_mut_nz" "$revert")
    detail="$detail; clean_log=$clean_log_final mutant_log=$mutant_log_final"
    record_result "$mid" "$file" "$base_rc" "$mut_rc" "$revert" "$verdict" "$detail"
    [[ "$revert" == "ok" ]] || break
  done
} < "$MANIFEST_SNAPSHOT"

if [[ $manifest_index -ne $MANIFEST_ROW_COUNT || $row_count -ne $MANIFEST_ROW_COUNT ]]; then
  overall_ok=0
  report_line "MANIFEST ERROR: expected $MANIFEST_ROW_COUNT rows, processed $manifest_index, recorded $row_count."
fi

report_line "============================================================"
if baseline_matches; then
  report_line "CLEAN: tracked worktree and untracked path set match the pre-run baseline."
else
  report_line "LEAK DETECTED: source does not match the complete pre-run baseline."
  overall_ok=0
fi

if [[ $overall_ok -eq 1 && $expected_pass_count -eq 0 && $killed_count -eq $row_count ]]; then
  report_line "RESULT: ALL KILLED ($killed_count)"
elif [[ $overall_ok -eq 1 ]]; then
  report_line "RESULT: SUCCESS (KILLED=$killed_count EXPECTED_PASS=$expected_pass_count)"
else
  report_line "RESULT: FAILURE (KILLED=$killed_count EXPECTED_PASS=$expected_pass_count BAD=$bad_count INCONCLUSIVE=$inconclusive_count)"
fi
report_line "kill_matrix: $KM_FINAL"

# Never publish evidence while source may still be changing or restoration is
# incomplete. Destination fingerprints also detect in-place tampering that does
# not change file type or link count.
if [[ $CONTAINER_SAFE_TO_RESTORE -ne 1 || $TRANSACTION_ACTIVE -ne 0 ]] || ! baseline_matches; then
  printf 'mutmut-verify: ERROR: refusing to publish before the worktree is safely restored\n' >&2
  exit 70
fi
publish_outputs
cat -- "$REPORT_FINAL" || true
[[ $overall_ok -eq 1 ]]
