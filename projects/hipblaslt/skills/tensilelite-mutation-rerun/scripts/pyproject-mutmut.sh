#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# pyproject-mutmut.sh — safely back up, rewrite, restore, and check the
# [tool.mutmut] slice config in projects/hipblaslt/tensilelite/pyproject.toml.
#
# pyproject.toml is a TRACKED file. Each slice rewrites `only_mutate` and
# `pytest_add_cli_args_test_selection`; the safety contract requires the tree to
# end clean (no dirty config) unless a new allowlist is deliberately committed.
# This helper is the single serial config actor: never run it concurrently with
# `mutmut run`.
#
# Restore is byte-exact by construction: `backup` copies the whole file, `restore`
# copies it back — restoration never depends on reversing the `set` formatter.
# A `<backup>.meta.json` sidecar binds the backup to the canonical source path,
# source file, Git HEAD, original content hash/mode, and latest generated content
# hash/mode. `set` and `restore` refuse stale or unrelated state instead of
# overwriting it.
# `set` rewrites ONLY the two target arrays inside the [tool.mutmut] table and
# preserves every other byte (source_paths, do_not_mutate, also_copy,
# mutate_only_covered_lines=false, comments) unchanged.
#
# Host TOML note: this host's python is 3.8 (no tomllib/tomlkit/tomli_w), so `set`
# uses explicit, selftested stdlib line-based rewriting scoped to the
# [tool.mutmut] table while preserving all unrelated bytes.
#
# Commands:
#   backup                     start a new backup transaction; refuses to
#                              overwrite an existing backup or metadata sidecar
#   set    --only-mutate ...   rewrite only_mutate / pytest_add_cli_args_test_selection
#          --test-selection ...   (repeatable and/or comma-separated; regenerates the
#                                  arrays deterministically so re-running is idempotent)
#   restore                    atomically restore the backup after validating
#                              transaction ownership, then remove backup + metadata
#   assert-clean [--allow-allowlist]
#                              exit 0 iff pyproject.toml == HEAD; non-zero if dirty,
#                              UNLESS --allow-allowlist (deliberate committed allowlist)
#
# Common flags: --src <dir> (default projects/hipblaslt/tensilelite),
#               --backup <path> (default work/.../mutprod/pyproject.toml.bak),
#               -h|--help
#
# Safety: never runs mutmut/tests; no push; no GitHub issues; single serial actor.
# Writes use temporary files in the destination directory followed by os.replace.

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
  local array_name="$1" value="$2" part
  local -a parts=()
  IFS=',' read -r -a parts <<< "$value"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] || continue
    case "$array_name" in
      ONLY_MUTATE)    ONLY_MUTATE+=("$part") ;;
      TEST_SELECTION) TEST_SELECTION+=("$part") ;;
      *) die "internal unknown list: $array_name" ;;
    esac
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
[[ -n "$ROOT" ]] || ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$ROOT" || die "cannot cd to repo root: $ROOT"

[[ -d "$SRC" ]] || die "source directory not found: $SRC"
SRC="$(cd "$SRC" && pwd -P)" || die "cannot canonicalize source directory: $SRC"
FILE="$SRC/pyproject.toml"
[[ -f "$FILE" ]] || die "pyproject.toml not found: $FILE"
[[ ! -L "$FILE" ]] || die "pyproject.toml must not be a symbolic link: $FILE"

[[ -n "$BACKUP" ]] || BACKUP="work/mutation/pyproject.toml.bak"
BACKUP="$(python3 - "$BACKUP" <<'PY'
import os
import sys
print(os.path.realpath(os.path.abspath(sys.argv[1])))
PY
)" || die "cannot canonicalize backup path"
META="$BACKUP.meta.json"
[[ "$BACKUP" != "$FILE" && "$META" != "$FILE" ]] \
  || die "backup and metadata paths must differ from pyproject.toml"

git -C "$SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "source directory is not in a Git worktree: $SRC"
git -C "$SRC" ls-files --error-unmatch -- pyproject.toml >/dev/null 2>&1 \
  || die "pyproject.toml is not tracked under: $SRC"

# Both flags can make `git diff` omit working-tree changes. Check them through
# the index tags before backup, set, restore, or assert-clean can trust Git.
INDEX_TAG="$(git -C "$SRC" ls-files -t -- pyproject.toml)" \
  || die "could not inspect pyproject.toml index flags"
[[ "${INDEX_TAG:0:1}" != "S" ]] \
  || die "pyproject.toml has the unsafe skip-worktree index flag"
INDEX_VTAG="$(git -C "$SRC" ls-files -v -- pyproject.toml)" \
  || die "could not inspect pyproject.toml index flags"
[[ ! "${INDEX_VTAG:0:1}" =~ [a-z] ]] \
  || die "pyproject.toml has the unsafe assume-unchanged index flag"

HEAD_OID="$(git -C "$SRC" rev-parse --verify 'HEAD^{commit}' 2>/dev/null)" \
  || die "could not resolve Git HEAD for: $SRC"

# backup/set/restore are implemented together so they share one validation and
# atomic-write implementation. A pending hash in the metadata is a tiny journal:
# after an interrupted set, the next command can distinguish the old generated
# file, the intended new file, and an unrelated edit without guessing.
run_transaction() {
  local om ts rc
  om="$(printf '%s\n' "${ONLY_MUTATE[@]:-}")"
  ts="$(printf '%s\n' "${TEST_SELECTION[@]:-}")"
  PM_COMMAND="$CMD" \
  PM_SOURCE_DIR="$SRC" \
  PM_FILE="$FILE" \
  PM_BACKUP="$BACKUP" \
  PM_META="$META" \
  PM_HEAD_OID="$HEAD_OID" \
  PM_HAS_OM="$([[ ${#ONLY_MUTATE[@]} -gt 0 ]] && echo 1 || echo 0)" \
  PM_HAS_TS="$([[ ${#TEST_SELECTION[@]} -gt 0 ]] && echo 1 || echo 0)" \
  PM_ONLY_MUTATE="$om" \
  PM_TEST_SELECTION="$ts" \
  python3 - <<'PY'
import hashlib
import json
import os
import re
import signal
import stat
import sys
import tempfile

SCHEMA = "pyproject-mutmut-backup/1"
command = os.environ["PM_COMMAND"]
source_dir = os.path.realpath(os.environ["PM_SOURCE_DIR"])
source_file = os.path.realpath(os.environ["PM_FILE"])
backup_file = os.path.realpath(os.environ["PM_BACKUP"])
metadata_file = os.path.realpath(os.environ["PM_META"])
head_oid = os.environ["PM_HEAD_OID"]


class TransactionError(Exception):
    pass


def fail(message):
    raise TransactionError(message)


def interrupted(signum, _frame):
    raise TransactionError("interrupted by signal %d" % signum)


for signal_name in ("SIGHUP", "SIGINT", "SIGTERM"):
    if hasattr(signal, signal_name):
        signal.signal(getattr(signal, signal_name), interrupted)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path, label):
    try:
        with open(path, "rb") as stream:
            return stream.read()
    except OSError as exc:
        fail("cannot read %s %s: %s" % (label, path, exc))


def fsync_directory(path):
    directory = os.path.dirname(path) or "."
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_temp(path, data, mode):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s.tmp." % os.path.basename(path), dir=directory
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        return temporary
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_replace(path, data, mode):
    temporary = prepare_temp(path, data, mode)
    try:
        os.replace(temporary, path)
        fsync_directory(path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def create_exclusive(path, data, mode):
    """Install complete bytes without ever replacing an existing destination."""
    temporary = prepare_temp(path, data, mode)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            fail("backup transaction already exists: %s" % path)
        fsync_directory(path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def metadata_bytes(metadata):
    return (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_metadata(metadata):
    atomic_replace(metadata_file, metadata_bytes(metadata), 0o600)


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_metadata():
    if not os.path.isfile(metadata_file):
        fail("backup metadata is missing: %s" % metadata_file)
    try:
        with open(metadata_file, "r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, ValueError) as exc:
        fail("invalid backup metadata %s: %s" % (metadata_file, exc))
    if not isinstance(metadata, dict) or metadata.get("schema") != SCHEMA:
        fail("unsupported backup metadata schema in %s" % metadata_file)

    expected = {
        "source_dir": source_dir,
        "source_file": source_file,
        "backup_file": backup_file,
        "head_oid": head_oid,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            fail(
                "backup metadata %s mismatch: recorded %r, current %r"
                % (key, metadata.get(key), value)
            )
    for key in ("original_sha256", "generated_sha256"):
        if not valid_hash(metadata.get(key)):
            fail("backup metadata %s is missing or invalid" % key)
    pending = metadata.get("pending_generated_sha256")
    if pending is not None and not valid_hash(pending):
        fail("backup metadata pending_generated_sha256 is invalid")
    for key in ("original_mode", "generated_mode"):
        value = metadata.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > 0o7777
        ):
            fail("backup metadata %s is missing or invalid" % key)
    return metadata


def load_session(allow_restored_source=False):
    if not os.path.isfile(backup_file) or not os.path.isfile(metadata_file):
        fail(
            "incomplete or missing backup transaction: expected %s and %s"
            % (backup_file, metadata_file)
        )
    metadata = load_metadata()
    pending = metadata.get("pending_generated_sha256")

    backup_data = read_bytes(backup_file, "backup")
    if sha256_bytes(backup_data) != metadata["original_sha256"]:
        fail("backup content hash does not match backup metadata")
    backup_mode = stat.S_IMODE(os.stat(backup_file).st_mode)
    if backup_mode != metadata["original_mode"]:
        fail(
            "backup mode does not match backup metadata "
            "(expected %04o, got %04o)"
            % (metadata["original_mode"], backup_mode)
        )
    current_data = read_bytes(source_file, "source file")
    current_hash = sha256_bytes(current_data)
    current_mode = stat.S_IMODE(os.stat(source_file).st_mode)

    # Reconcile the two safe outcomes of an interrupted atomic set. Anything
    # else is an unrelated edit and must not be overwritten.
    if pending is not None:
        if current_hash == pending:
            if current_mode != metadata["generated_mode"]:
                fail("source mode changed during the pending set transaction")
            metadata["generated_sha256"] = pending
        elif (
            current_hash != metadata["generated_sha256"]
            or current_mode != metadata["generated_mode"]
        ):
            fail("source state matches neither side of the pending set transaction")
        metadata["pending_generated_sha256"] = None
        write_metadata(metadata)

    if (
        current_hash != metadata["generated_sha256"]
        or current_mode != metadata["generated_mode"]
    ):
        restored_source = (
            allow_restored_source
            and current_hash == metadata["original_sha256"]
            and current_mode == metadata["original_mode"]
        )
        if not restored_source:
            if current_hash == metadata["generated_sha256"]:
                fail(
                    "source mode changed outside this backup transaction "
                    "(expected %04o, got %04o)"
                    % (metadata["generated_mode"], current_mode)
                )
            fail(
                "source content changed outside this backup transaction "
                "(expected generated SHA-256 %s, got %s)"
                % (metadata["generated_sha256"], current_hash)
            )
    return metadata, backup_data, current_data


def finalize_metadata_only_restore():
    """Finish cleanup after restore removed the backup but not its metadata."""
    metadata = load_metadata()
    if metadata.get("pending_generated_sha256") is not None:
        fail("metadata-only restore state contains a pending set transaction")
    current_data = read_bytes(source_file, "source file")
    current_hash = sha256_bytes(current_data)
    current_mode = stat.S_IMODE(os.stat(source_file).st_mode)
    if (
        current_hash != metadata["original_sha256"]
        or current_mode != metadata["original_mode"]
    ):
        fail(
            "metadata-only restore state is not the recorded original file "
            "(expected SHA-256 %s mode %04o, got SHA-256 %s mode %04o)"
            % (
                metadata["original_sha256"],
                metadata["original_mode"],
                current_hash,
                current_mode,
            )
        )
    os.unlink(metadata_file)
    fsync_directory(metadata_file)
    print("pyproject-mutmut: restore already complete for %s" % source_file)
    print("pyproject-mutmut: cleared backup transaction")


def render_updated_config(data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail("pyproject.toml is not valid UTF-8: %s" % exc)
    lines = text.splitlines(keepends=True)
    newline = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"

    start = None
    for index, line in enumerate(lines):
        if line.strip() == "[tool.mutmut]":
            start = index
            break
    if start is None:
        fail("[tool.mutmut] table not found in %s" % source_file)
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("["):
            end = index
            break

    def render_array(key, values):
        output = ["%s = [%s" % (key, newline)]
        for value in values:
            output.append(
                "    %s,%s" % (json.dumps(value, ensure_ascii=False), newline)
            )
        output.append("]%s" % newline)
        return output

    def bracket_delta(value):
        depth = 0
        index = 0
        quote = None
        while index < len(value):
            char = value[index]
            if quote is not None:
                if quote == '"' and char == "\\":
                    index += 2
                    continue
                if char == quote:
                    quote = None
            else:
                if char in ('"', "'"):
                    quote = char
                elif char == "#":
                    break
                elif char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
            index += 1
        return depth

    def replace_key(current_lines, table_start, table_end, key, values):
        key_line = None
        for index in range(table_start + 1, table_end):
            stripped = current_lines[index].lstrip()
            if stripped.startswith(key) and stripped[len(key):].lstrip().startswith("="):
                key_line = index
                break
        if key_line is None:
            fail("key '%s' not found inside [tool.mutmut]" % key)
        depth = 0
        last_line = None
        for index in range(key_line, table_end):
            depth += bracket_delta(current_lines[index])
            if depth <= 0:
                last_line = index
                break
        if last_line is None:
            fail("unterminated array for key '%s'" % key)
        replacement = render_array(key, values)
        return current_lines[:key_line] + replacement + current_lines[last_line + 1:]

    def env_list(name):
        return [value for value in os.environ.get(name, "").splitlines() if value.strip()]

    changed = []
    if os.environ.get("PM_HAS_OM") == "1":
        lines = replace_key(lines, start, end, "only_mutate", env_list("PM_ONLY_MUTATE"))
        for index in range(start + 1, len(lines)):
            if lines[index].startswith("["):
                end = index
                break
        else:
            end = len(lines)
        changed.append("only_mutate")
    if os.environ.get("PM_HAS_TS") == "1":
        lines = replace_key(
            lines,
            start,
            end,
            "pytest_add_cli_args_test_selection",
            env_list("PM_TEST_SELECTION"),
        )
        changed.append("pytest_add_cli_args_test_selection")
    return "".join(lines).encode("utf-8"), changed


def backup():
    if os.path.lexists(backup_file) or os.path.lexists(metadata_file):
        fail(
            "backup transaction already exists; restore or remove both %s and %s"
            % (backup_file, metadata_file)
        )
    original = read_bytes(source_file, "source file")
    digest = sha256_bytes(original)
    mode = stat.S_IMODE(os.stat(source_file).st_mode)
    metadata = {
        "schema": SCHEMA,
        "source_dir": source_dir,
        "source_file": source_file,
        "backup_file": backup_file,
        "head_oid": head_oid,
        "original_sha256": digest,
        "generated_sha256": digest,
        "pending_generated_sha256": None,
        "original_mode": mode,
        "generated_mode": mode,
    }
    backup_created = False
    metadata_created = False
    try:
        create_exclusive(backup_file, original, mode)
        backup_created = True
        create_exclusive(metadata_file, metadata_bytes(metadata), 0o600)
        metadata_created = True
    except BaseException:
        if metadata_created:
            try:
                os.unlink(metadata_file)
            except OSError:
                pass
        if backup_created:
            try:
                os.unlink(backup_file)
            except OSError:
                pass
        raise
    print("pyproject-mutmut: backup %s -> %s" % (source_file, backup_file))
    print("pyproject-mutmut: metadata %s" % metadata_file)


def set_config():
    metadata, _backup_data, current = load_session()
    updated, changed = render_updated_config(current)
    updated_hash = sha256_bytes(updated)
    previous = dict(metadata)
    pending = dict(metadata)
    pending["pending_generated_sha256"] = updated_hash
    write_metadata(pending)
    try:
        atomic_replace(source_file, updated, metadata["generated_mode"])
        final = dict(pending)
        final["generated_sha256"] = updated_hash
        final["pending_generated_sha256"] = None
        write_metadata(final)
    except BaseException:
        # Bring the journal to whichever complete file is visible. If even this
        # repair fails, the next invocation performs the same reconciliation.
        try:
            visible_hash = sha256_bytes(read_bytes(source_file, "source file"))
            if visible_hash == previous["generated_sha256"]:
                write_metadata(previous)
            elif visible_hash == updated_hash:
                final = dict(pending)
                final["generated_sha256"] = updated_hash
                final["pending_generated_sha256"] = None
                write_metadata(final)
        except BaseException:
            pass
        raise
    sys.stderr.write("pyproject-mutmut: set rewrote %s\n" % ", ".join(changed))
    print("pyproject-mutmut: set OK in %s" % source_file)


def restore():
    if os.path.lexists(metadata_file) and not os.path.lexists(backup_file):
        finalize_metadata_only_restore()
        return
    metadata, backup_data, _current = load_session(allow_restored_source=True)
    atomic_replace(source_file, backup_data, metadata["original_mode"])
    if sha256_bytes(read_bytes(source_file, "restored source file")) != metadata["original_sha256"]:
        fail("restored pyproject.toml does not match the recorded original hash")
    if stat.S_IMODE(os.stat(source_file).st_mode) != metadata["original_mode"]:
        fail("restored pyproject.toml does not match the recorded original mode")
    # Remove the backup first. If interrupted between these two unlinks, the
    # metadata-only state can be authenticated and finalized by a restore retry.
    # The reverse order would leave an unverifiable backup-only orphan.
    os.unlink(backup_file)
    os.unlink(metadata_file)
    fsync_directory(backup_file)
    if os.path.dirname(metadata_file) != os.path.dirname(backup_file):
        fsync_directory(metadata_file)
    print("pyproject-mutmut: restore %s -> %s" % (backup_file, source_file))
    print("pyproject-mutmut: cleared backup transaction")


try:
    if command == "backup":
        backup()
    elif command == "set":
        set_config()
    elif command == "restore":
        restore()
    else:
        fail("internal unsupported transaction command: %s" % command)
except TransactionError as exc:
    sys.stderr.write("pyproject-mutmut: ERROR: %s\n" % exc)
    sys.exit(1)
except OSError as exc:
    sys.stderr.write("pyproject-mutmut: ERROR: transaction I/O failed: %s\n" % exc)
    sys.exit(1)
PY
  rc=$?
  return "$rc"
}

# ------------------------------------------------------------------ commands
cmd_backup() {
  git -C "$SRC" diff --quiet HEAD -- pyproject.toml 2>/dev/null \
    || die "refusing to back up pyproject.toml with staged or unstaged changes"
  run_transaction
}

cmd_restore() {
  run_transaction
}

cmd_set() {
  [[ ${#ONLY_MUTATE[@]} -gt 0 || ${#TEST_SELECTION[@]} -gt 0 ]] \
    || die "set needs at least one of --only-mutate / --test-selection"
  run_transaction
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
