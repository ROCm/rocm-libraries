#!/usr/bin/env bash
# pyproject-mutmut.sh — Issue 3: back up / rewrite / restore / gate the
# [tool.mutmut] slice config in projects/hipblaslt/tensilelite/pyproject.toml.
#
# pyproject.toml is a TRACKED file. Each slice rewrites `only_mutate` and
# `pytest_add_cli_args_test_selection`; the plan's done-criteria require the tree
# to end clean (no dirty config) unless a new allowlist is DELIBERATELY committed.
# This helper is the single serial config actor: never run it concurrently with
# `mutmut run`.
#
# Restore is byte-exact by construction: `backup` copies the whole file, `restore`
# copies it back — restoration never depends on reversing the `set` formatter.
# `set` rewrites ONLY the two target arrays inside the [tool.mutmut] table and
# preserves every other byte (source_paths, do_not_mutate, also_copy,
# mutate_only_covered_lines=false, comments) unchanged.
#
# Host TOML note: this host's python is 3.8 (no tomllib/tomlkit/tomli_w), so `set`
# uses explicit, selftested stdlib line-based rewriting scoped to the
# [tool.mutmut] table (see tests/pyproject-mutmut-selftest.sh).
#
# Commands:
#   backup                     snapshot pyproject.toml -> backup location
#   set    --only-mutate ...   rewrite only_mutate / pytest_add_cli_args_test_selection
#          --test-selection ...   (repeatable and/or comma-separated; regenerates the
#                                  arrays deterministically so re-running is idempotent)
#   restore                    copy the backup back over pyproject.toml (byte-exact)
#   assert-clean [--allow-allowlist]
#                              exit 0 iff pyproject.toml == HEAD; non-zero if dirty,
#                              UNLESS --allow-allowlist (deliberate committed allowlist)
#
# Common flags: --src <dir> (default projects/hipblaslt/tensilelite),
#               --backup <path> (default work/.../mutprod/pyproject.toml.bak),
#               -h|--help
#
# Safety: never runs mutmut/tests; no push; no GitHub issues; single serial actor.

set -u

die() { printf 'pyproject-mutmut: ERROR: %s\n' "$*" >&2; exit 1; }

SRC="projects/hipblaslt/tensilelite"
BACKUP=""
declare -a ONLY_MUTATE=()
declare -a TEST_SELECTION=()
ALLOW_ALLOWLIST=0

show_help() { sed -n '2,$p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; }
[[ $# -ge 1 ]] || die "a command is required (backup|set|restore|assert-clean); see --help"
case "$1" in -h|--help) show_help; exit 0 ;; esac
CMD="$1"; shift

# split a possibly comma-separated value into the named array
append_csv() {
  local __arr="$1" __val="$2" part
  local IFS=','
  for part in $__val; do
    [[ -n "$part" ]] && eval "$__arr+=(\"\$part\")"
  done
}

need_val() { [[ $# -ge 2 ]] || die "$1 requires a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)             need_val "$@"; SRC="$2"; shift 2 ;;
    --backup)          need_val "$@"; BACKUP="$2"; shift 2 ;;
    --only-mutate)     need_val "$@"; append_csv ONLY_MUTATE "$2"; shift 2 ;;
    --test-selection)  need_val "$@"; append_csv TEST_SELECTION "$2"; shift 2 ;;
    --allow-allowlist) ALLOW_ALLOWLIST=1; shift ;;
    -h|--help)         show_help; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# ------------------------------------------------------------- resolve root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
cd "$ROOT" || die "cannot cd to repo root: $ROOT"

FILE="$SRC/pyproject.toml"
[[ -n "$BACKUP" ]] || BACKUP="work/mutation/pyproject.toml.bak"

# ------------------------------------------------------------------ commands
cmd_backup() {
  [[ -f "$FILE" ]] || die "pyproject.toml not found: $FILE"
  mkdir -p "$(dirname "$BACKUP")" || die "cannot create backup dir"
  cp -p "$FILE" "$BACKUP" || die "backup copy failed"
  printf 'pyproject-mutmut: backup %s -> %s\n' "$FILE" "$BACKUP"
}

cmd_restore() {
  [[ -f "$BACKUP" ]] || die "no backup to restore: $BACKUP (run 'backup' first)"
  cp -p "$BACKUP" "$FILE" || die "restore copy failed"
  printf 'pyproject-mutmut: restore %s -> %s\n' "$BACKUP" "$FILE"
}

cmd_set() {
  [[ -f "$FILE" ]] || die "pyproject.toml not found: $FILE"
  [[ ${#ONLY_MUTATE[@]} -gt 0 || ${#TEST_SELECTION[@]} -gt 0 ]] \
    || die "set needs at least one of --only-mutate / --test-selection"
  local om ts
  om="$(printf '%s\n' "${ONLY_MUTATE[@]:-}")"
  ts="$(printf '%s\n' "${TEST_SELECTION[@]:-}")"
  PM_FILE="$FILE" \
  PM_HAS_OM="$([[ ${#ONLY_MUTATE[@]} -gt 0 ]] && echo 1 || echo 0)" \
  PM_HAS_TS="$([[ ${#TEST_SELECTION[@]} -gt 0 ]] && echo 1 || echo 0)" \
  PM_ONLY_MUTATE="$om" \
  PM_TEST_SELECTION="$ts" \
  python3 - <<'PY'
import os, sys, json

path = os.environ["PM_FILE"]
with open(path, "r") as fh:
    lines = fh.readlines()  # keeps line endings

def bail(msg):
    sys.stderr.write("pyproject-mutmut: ERROR: %s\n" % msg)
    raise SystemExit(1)

# locate [tool.mutmut] table: header at column 0, table ends at next col-0 '['.
start = None
for i, ln in enumerate(lines):
    if ln.strip() == "[tool.mutmut]":
        start = i
        break
if start is None:
    bail("[tool.mutmut] table not found in %s" % path)
end = len(lines)
for j in range(start + 1, len(lines)):
    if lines[j].startswith("["):
        end = j
        break

def render_array(key, values):
    out = ["%s = [\n" % key]
    for v in values:
        out.append("    %s,\n" % json.dumps(v))
    out.append("]\n")
    return out

def bracket_delta(s):
    # net '['-minus-']' on a line, ignoring brackets inside "..."/'...' strings
    # and after an unquoted '#' comment. Used to find the array span robustly so
    # an inline comment or a ']' inside a quoted value cannot truncate it.
    depth = 0
    i = 0
    n = len(s)
    q = None
    while i < n:
        c = s[i]
        if q is not None:
            if q == '"' and c == "\\":
                i += 2
                continue
            if c == q:
                q = None
        else:
            if c == '"' or c == "'":
                q = c
            elif c == "#":
                break
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
        i += 1
    return depth

def replace_key(lines, start, end, key, values):
    # find 'key = ...' inside [start,end)
    k = None
    for i in range(start + 1, end):
        s = lines[i].lstrip()
        if s.startswith(key) and s[len(key):].lstrip().startswith("="):
            k = i
            break
    if k is None:
        bail("key '%s' not found inside [tool.mutmut]" % key)
    # array span = from the opener line until running bracket depth returns to 0
    # (handles single-line, multi-line, inline comments, and ']' inside values).
    depth = 0
    m = None
    for j in range(k, end):
        depth += bracket_delta(lines[j])
        if depth <= 0:
            m = j
            break
    if m is None:
        bail("unterminated array for key '%s'" % key)
    new = render_array(key, values)
    return lines[:k] + new + lines[m + 1:], (m - k + 1), len(new)

def parse_env_list(name):
    raw = os.environ.get(name, "")
    return [x for x in raw.splitlines() if x.strip()]

changed = []
if os.environ.get("PM_HAS_OM") == "1":
    vals = parse_env_list("PM_ONLY_MUTATE")
    lines, _old, _new = replace_key(lines, start, end, "only_mutate", vals)
    # table bounds may have shifted; recompute end for the next replacement
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("["):
            end = j; break
    else:
        end = len(lines)
    changed.append("only_mutate")
if os.environ.get("PM_HAS_TS") == "1":
    vals = parse_env_list("PM_TEST_SELECTION")
    lines, _old, _new = replace_key(lines, start, end, "pytest_add_cli_args_test_selection", vals)
    changed.append("pytest_add_cli_args_test_selection")

with open(path, "w") as fh:
    fh.writelines(lines)
sys.stderr.write("pyproject-mutmut: set rewrote %s\n" % ", ".join(changed))
PY
  local rc=$?
  [[ $rc -eq 0 ]] || die "set failed (python rc=$rc)"
  printf 'pyproject-mutmut: set OK in %s\n' "$FILE"
}

cmd_assert_clean() {
  [[ -f "$FILE" ]] || die "pyproject.toml not found: $FILE"
  # Compare against HEAD (not the index): catches staged AND unstaged deviations
  # from the committed baseline, so a `git add`ed slice config cannot slip past.
  if git -C "$SRC" diff --quiet HEAD -- pyproject.toml 2>/dev/null; then
    printf 'pyproject-mutmut: assert-clean OK (pyproject.toml == HEAD)\n'
    return 0
  fi
  if [[ "$ALLOW_ALLOWLIST" -eq 1 ]]; then
    printf 'pyproject-mutmut: assert-clean: pyproject.toml differs from HEAD, allowed by --allow-allowlist (deliberate allowlist commit)\n'
    return 0
  fi
  printf 'pyproject-mutmut: ERROR: pyproject.toml is dirty vs HEAD (restore or pass --allow-allowlist)\n' >&2
  git -C "$SRC" --no-pager diff HEAD -- pyproject.toml >&2 || true
  return 1
}

case "$CMD" in
  backup)       cmd_backup ;;
  set)          cmd_set ;;
  restore)      cmd_restore ;;
  assert-clean) cmd_assert_clean ;;
  *) die "unknown command: $CMD (expected backup|set|restore|assert-clean)" ;;
esac
