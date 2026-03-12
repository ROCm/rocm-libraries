#!/usr/bin/env python3
"""
State management helper for the documentation agent.

Usage:
    python doc_agent_state.py init
    python doc_agent_state.py get-work
    python doc_agent_state.py mark-visited --dir DIR --covered FILE1,FILE2
    python doc_agent_state.py finish-run
    python doc_agent_state.py show

State model
-----------
The state file tracks every directory under the configured targets that
contains documentable source files.  Each directory entry holds:

    last_visited          — ISO timestamp of the last run that worked on
                            this directory (None if never visited).
    files_covered         — source filenames discussed in at least one doc.
    runs_since_last_visit — how many runs have passed since this directory
                            was last worked on (reset to 0 on visit).
    visited_this_run      — flag set by mark-visited, cleared by finish-run.
                            Used to determine which directories to increment.

Uncovered files are not stored in state.  Instead, ``get-work`` computes
them on the fly as ``all_files - files_covered`` using the current
filesystem scan.  This avoids stale entries for deleted or renamed files.

Lifecycle of a directory across runs:

    1. ``init`` scans the filesystem and creates an entry for every
       directory with documentable files.

    2. ``get-work`` re-scans the filesystem each run:
       - New directories are added to state.
       - Deleted directories are removed from state.
       It then selects up to two directories to work on (one reactive,
       one proactive) based on priority queues.

       Reactive changes are tracked at the **directory** level, not per
       file.  ``git diff`` identifies which files changed, and those
       changes are grouped by parent directory.  Only the directory
       names enter the reactive queue — there is no per-file backlog.

       Because only two directories can be worked on per run, reactive
       directories that aren't picked are saved to ``pending_reactive``
       so they carry over to the next run instead of being lost when
       ``last_commit`` advances.  Note that for carried-over
       directories, the file-level diff is no longer available (the
       stored commit hash has advanced past those changes).  The agent
       must compare existing docs against current source code to
       identify what needs updating.

    3. ``mark-visited`` is called after the agent documents a directory.
       It updates files_covered and resets runs_since_last_visit to 0.

    4. ``finish-run`` increments runs_since_last_visit for all
       directories NOT visited this run and records the current commit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

SCRIPT_DIR: Path = Path(__file__).resolve().parent
STATE_FILE: Path = SCRIPT_DIR / ".doc-agent-state.json"
TARGETS_FILE: Path = SCRIPT_DIR / "targets.json"
DOCUMENTABLE_EXTENSIONS: set[str] = {".py", ".cpp", ".h", ".hpp", ".yaml", ".yml", ".cmake", ".sh"}
SKIP_DIRS: set[str] = {".*", "docs", "__pycache__", "build", "dist", "*.egg-info"}
SKIP_FILES: set[str] = {".*"}


def load_targets() -> tuple[list[str], list[dict[str, str]]]:
    """Load target directories and exclude rules from the targets config file.

    Returns a tuple of (targets, excludes) where targets is a list of directory
    paths relative to the repo root and excludes is a list of dicts, each with
    ``"dir"`` (a directory prefix) and ``"pattern"`` (an fnmatch pattern for
    filenames).  Any file whose directory starts with ``dir`` and whose name
    matches ``pattern`` is excluded from scanning.
    """
    if not TARGETS_FILE.exists():
        print(f"Error: targets file not found at {TARGETS_FILE}", file=sys.stderr)
        sys.exit(1)
    with open(TARGETS_FILE) as f:
        config = json.load(f)
    repo_root = get_repo_root()
    targets: list[str] = []
    for t in config["targets"]:
        target_path = repo_root / t
        if not target_path.is_dir():
            print(f"Warning: target directory '{t}' does not exist, skipping.", file=sys.stderr)
            continue
        targets.append(t)
    excludes: list[dict[str, str]] = config.get("exclude", [])
    return targets, excludes


def load_state() -> dict[str, Any] | None:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def save_state(state: dict[str, Any]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State written to {STATE_FILE}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=SCRIPT_DIR
    )
    if result.returncode != 0:
        print("Error: not inside a git repository", file=sys.stderr)
        sys.exit(1)
    return Path(result.stdout.strip())


def git_rev_parse_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=SCRIPT_DIR
    )
    if result.returncode != 0:
        print(f"Error running git rev-parse HEAD: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def git_diff_files(last_commit: str, targets: list[str]) -> list[str]:
    repo_root = get_repo_root()
    cmd = ["git", "diff", "--name-only", f"{last_commit}..HEAD", "--"] + targets
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]


def should_skip_dir(dirname: str) -> bool:
    """Return True if a directory should be excluded from scanning.

    A directory is skipped if it matches any pattern in SKIP_DIRS.
    The ".*" entry in SKIP_DIRS handles all hidden directories.
    """
    return any(fnmatch(dirname, pat) for pat in SKIP_DIRS)


def is_documentable(filename: str) -> bool:
    """Return True if a file should be considered for documentation.

    A file is skipped if it matches any pattern in SKIP_FILES (e.g. ".*"
    for hidden files).  Otherwise it must have an extension listed in
    DOCUMENTABLE_EXTENSIONS.
    """
    if any(fnmatch(filename, pat) for pat in SKIP_FILES):
        return False
    return Path(filename).suffix in DOCUMENTABLE_EXTENSIONS


def is_excluded(dir_path: str, filename: str, excludes: list[dict[str, str]]) -> bool:
    """Return True if a file should be excluded from scanning.

    Each exclude rule has a ``"dir"`` prefix and a ``"pattern"`` fnmatch glob.
    A file is excluded when its directory starts with the rule's dir and its
    filename matches the rule's pattern.
    """
    for exc in excludes:
        if dir_path.startswith(exc["dir"]) and fnmatch(filename, exc["pattern"]):
            return True
    return False


def scan_directories(targets: list[str], excludes: list[dict[str, str]] | None = None) -> dict[str, list[str]]:
    """Scan all target roots and return a dict of dir_path -> list of documentable filenames."""
    if excludes is None:
        excludes = []
    repo_root = get_repo_root()
    directories: dict[str, list[str]] = {}
    for target in targets:
        code_root = repo_root / target
        for root, dirs, files in os.walk(code_root):
            # In-place modification of dirs prunes os.walk's traversal —
            # subdirectories removed here will not be visited.
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]
            rel_path = str(Path(root).relative_to(repo_root))
            documentable = [
                f for f in files
                if is_documentable(f) and not is_excluded(rel_path, f, excludes)
            ]
            if documentable:
                directories[rel_path] = sorted(documentable)
    return directories


def has_docs_dir(dir_path: str) -> bool:
    repo_root = get_repo_root()
    docs_path = repo_root / dir_path / "docs"
    return docs_path.is_dir()


def cmd_init(_args: argparse.Namespace) -> None:
    """Initialize the state file by scanning the directory tree."""
    if STATE_FILE.exists():
        print("State file already exists. Use 'show' to inspect it.", file=sys.stderr)
        sys.exit(1)

    targets, excludes = load_targets()
    scanned = scan_directories(targets, excludes)
    directories = {}
    for dir_path, files in scanned.items():
        directories[dir_path] = {
            "last_visited": None,
            "files_covered": [],
            "runs_since_last_visit": 0,
            "visited_this_run": False,
        }

    state = {
        "last_commit": git_rev_parse_head(),
        "last_run": now_iso(),
        "run_count": 0,
        "directories": directories,
    }
    save_state(state)
    print(f"Initialized state with {len(directories)} directories across {len(targets)} target(s).")


def cmd_get_work(_args: argparse.Namespace) -> None:
    """Determine the two work items for this run and print them.

    Work is selected from two priority queues:

    **Reactive queue** — directories where source files have changed since
    the last run (git diff).  These need their existing documentation
    updated to reflect the code changes.  Sorted by number of changed
    files (most changes first).

    **Proactive queue** — directories that need *new* documentation work,
    independent of recent code changes.  Prioritised in this order:
      1. Directories with no ``docs/`` directory yet (brand-new docs).
      2. Directories whose docs exist but have uncovered source files.
      3. Directories whose docs are fully covered but haven't been
         reviewed for staleness recently (sorted by
         ``runs_since_last_visit``, most stale first).

    The two slots are filled as follows:
      * slot1 = top of the reactive queue (if non-empty), otherwise top
        of the proactive queue.
      * slot2 = first directory with missing documentation (no docs or
        partial docs), then reactive, then staleness review — skipping
        slot1 to avoid duplicates.
    """
    state = load_state()
    if state is None:
        print("No state file found. Run 'init' first.", file=sys.stderr)
        sys.exit(1)

    targets, excludes = load_targets()

    # Refresh directory list: add new dirs, remove deleted ones
    scanned = scan_directories(targets, excludes)
    for dir_path in scanned:
        if dir_path not in state["directories"]:
            state["directories"][dir_path] = {
                "last_visited": None,
                "files_covered": [],
                "runs_since_last_visit": 0,
                "visited_this_run": False,
            }
    removed = [d for d in state["directories"] if d not in scanned]
    for d in removed:
        del state["directories"][d]

    # ── Reactive queue ──────────────────────────────────────────────
    # Directories whose source files changed since last run, plus any
    # carried over from previous runs that weren't worked on yet.
    reactive_set: set[str] = set(state.get("pending_reactive", []))
    last_commit = state.get("last_commit")
    if last_commit:
        changed_files = git_diff_files(last_commit, targets)
        dir_change_count: dict[str, int] = {}
        for f in changed_files:
            parent = str(Path(f).parent)
            if parent in state["directories"]:
                dir_change_count[parent] = dir_change_count.get(parent, 0) + 1
                reactive_set.add(parent)
        # Sort by change count (dirs from pending_reactive with no new
        # changes get a count of 0, so they appear after dirs with fresh changes).
        reactive_queue: list[str] = sorted(
            reactive_set,
            key=lambda d: dir_change_count.get(d, 0),
            reverse=True,
        )
    else:
        reactive_queue = sorted(reactive_set)

    # ── Proactive queue ─────────────────────────────────────────────
    # Directories that need new documentation work regardless of
    # recent code changes.  Priority: no docs > partial docs > stale.
    no_docs: list[str] = []
    partial_docs: list[str] = []
    stalest: list[str] = []

    for dir_path, entry in state["directories"].items():
        all_files = set(scanned.get(dir_path, []))
        uncovered = all_files - set(entry["files_covered"])
        if not has_docs_dir(dir_path):
            no_docs.append(dir_path)
        elif uncovered:
            partial_docs.append(dir_path)
        else:
            stalest.append(dir_path)

    stalest.sort(key=lambda d: state["directories"][d]["runs_since_last_visit"],
                 reverse=True)

    # High-priority proactive: directories with missing docs (no_docs + partial_docs).
    # Low-priority proactive: staleness reviews (stalest).
    # Reactive work (code changed) is more important than staleness reviews
    # but less important than genuinely missing documentation.
    high_priority_proactive: list[str] = no_docs + partial_docs

    # ── Fill the two work slots ─────────────────────────────────────
    slot1: str | None = None
    slot1_source: str | None = None
    slot2: str | None = None
    slot2_source: str | None = None

    # slot1: prefer reactive work; fall back to proactive.
    if reactive_queue:
        slot1 = reactive_queue[0]
        slot1_source = "reactive"
    elif high_priority_proactive:
        slot1 = high_priority_proactive[0]
        slot1_source = "proactive"
    elif stalest:
        slot1 = stalest[0]
        slot1_source = "proactive"

    # slot2: prefer high-priority proactive, then reactive, then stalest.
    # Skip slot1 to avoid duplicates.
    for candidate in high_priority_proactive:
        if candidate != slot1:
            slot2 = candidate
            slot2_source = "proactive"
            break
    if slot2 is None:
        for candidate in reactive_queue:
            if candidate != slot1:
                slot2 = candidate
                slot2_source = "reactive"
                break
    if slot2 is None:
        for candidate in stalest:
            if candidate != slot1:
                slot2 = candidate
                slot2_source = "proactive"
                break

    # ── Build output ────────────────────────────────────────────────
    output: dict[str, Any] = {
        "slot1": None,
        "slot2": None,
        "reactive_queue_size": len(reactive_queue),
        "proactive_queue_size": len(high_priority_proactive) + len(stalest),
    }

    if slot1:
        entry = state["directories"][slot1]
        all_files = scanned.get(slot1, [])
        output["slot1"] = {
            "directory": slot1,
            "source": slot1_source,
            "has_docs": has_docs_dir(slot1),
            "files_covered": entry["files_covered"],
            "files_uncovered": sorted(set(all_files) - set(entry["files_covered"])),
            "all_files": all_files,
            "runs_since_last_visit": entry["runs_since_last_visit"],
        }

    if slot2:
        entry = state["directories"][slot2]
        all_files = scanned.get(slot2, [])
        output["slot2"] = {
            "directory": slot2,
            "source": slot2_source,
            "has_docs": has_docs_dir(slot2),
            "files_covered": entry["files_covered"],
            "files_uncovered": sorted(set(all_files) - set(entry["files_covered"])),
            "all_files": all_files,
            "runs_since_last_visit": entry["runs_since_last_visit"],
        }

    # Persist reactive directories that weren't picked up this run.
    # These carry over so they aren't lost when last_commit advances.
    picked = {slot1, slot2}
    state["pending_reactive"] = [d for d in reactive_queue if d not in picked]

    # Save any directory list updates we made
    save_state(state)

    print(json.dumps(output, indent=2))


def cmd_mark_visited(args: argparse.Namespace) -> None:
    """Mark a directory as visited and update its covered source file list."""
    state = load_state()
    if state is None:
        print("No state file found. Run 'init' first.", file=sys.stderr)
        sys.exit(1)

    dir_path = args.dir
    if dir_path not in state["directories"]:
        print(f"Directory '{dir_path}' not found in state.", file=sys.stderr)
        sys.exit(1)

    covered = [f.strip() for f in args.covered.split(",") if f.strip()] if args.covered else []

    entry = state["directories"][dir_path]
    # Add newly covered source files (avoid duplicates)
    existing_covered = set(entry["files_covered"])
    for f in covered:
        existing_covered.add(f)
    entry["files_covered"] = sorted(existing_covered)

    entry["last_visited"] = now_iso()
    entry["runs_since_last_visit"] = 0
    entry["visited_this_run"] = True

    save_state(state)
    print(f"Marked '{dir_path}' as visited. Covered: {len(entry['files_covered'])}")


def cmd_finish_run(_args: argparse.Namespace) -> None:
    """End-of-run bookkeeping: increment counters, update commit hash."""
    state = load_state()
    if state is None:
        print("No state file found. Run 'init' first.", file=sys.stderr)
        sys.exit(1)

    visited_dirs: set[str] = set()
    for dir_path, entry in state["directories"].items():
        if entry.get("visited_this_run"):
            entry["visited_this_run"] = False
            visited_dirs.add(dir_path)
        else:
            entry["runs_since_last_visit"] += 1

    # Remove visited directories from pending_reactive
    state["pending_reactive"] = [
        d for d in state.get("pending_reactive", []) if d not in visited_dirs
    ]

    state["last_commit"] = git_rev_parse_head()
    state["last_run"] = now_iso()
    state["run_count"] = state.get("run_count", 0) + 1

    save_state(state)
    print(f"Run #{state['run_count']} completed.")


def cmd_show(_args: argparse.Namespace) -> None:
    """Print the current state file contents."""
    state = load_state()
    if state is None:
        print("No state file found.")
        return
    print(json.dumps(state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Documentation agent state manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize state file by scanning directory tree")
    subparsers.add_parser("get-work", help="Get the two work items for this run")

    mark_parser = subparsers.add_parser("mark-visited", help="Mark a directory as visited")
    mark_parser.add_argument("--dir", required=True, help="Directory path (relative to repo root)")
    mark_parser.add_argument("--covered", default="", help="Comma-separated source files now covered by documentation")

    subparsers.add_parser("finish-run", help="End-of-run bookkeeping")
    subparsers.add_parser("show", help="Print current state")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "get-work": cmd_get_work,
        "mark-visited": cmd_mark_visited,
        "finish-run": cmd_finish_run,
        "show": cmd_show,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()


# ── Unit Tests ─────────────────────────────────────────────────────────
# Run with: pytest doc_agent_state.py -v

try:
    import pytest
except ImportError:
    pass


def test_should_skip_dir_hidden():
    assert should_skip_dir(".git") is True
    assert should_skip_dir(".hidden") is True


def test_should_skip_dir_named():
    assert should_skip_dir("docs") is True
    assert should_skip_dir("__pycache__") is True
    assert should_skip_dir("build") is True
    assert should_skip_dir("dist") is True


def test_should_skip_dir_egg_info():
    assert should_skip_dir("foo.egg-info") is True


def test_should_skip_dir_allowed():
    assert should_skip_dir("src") is False
    assert should_skip_dir("library") is False
    assert should_skip_dir("tests") is False


def test_is_documentable_valid_extensions():
    for ext in DOCUMENTABLE_EXTENSIONS:
        assert is_documentable(f"file{ext}") is True, f"Expected {ext} to be documentable"


def test_is_documentable_invalid_extensions():
    assert is_documentable("file.txt") is False
    assert is_documentable("file.json") is False
    assert is_documentable("file.md") is False
    assert is_documentable("file.o") is False


def test_is_documentable_hidden_files():
    assert is_documentable(".hidden.py") is False
    assert is_documentable(".gitignore") is False


def test_is_documentable_no_extension():
    assert is_documentable("Makefile") is False


def test_now_iso_format():
    result = now_iso()
    # Should be parseable as ISO format and have timezone info
    dt = datetime.fromisoformat(result)
    assert dt.tzinfo is not None


def test_load_state_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.STATE_FILE", tmp_path / "nonexistent.json")
    assert load_state() is None


def test_load_state_existing_file(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    data = {"last_commit": "abc123", "run_count": 1}
    state_file.write_text(json.dumps(data))
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    result = load_state()
    assert result == data


def test_save_state_creates_file(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    data = {"last_commit": "abc123", "run_count": 1}
    save_state(data)
    assert state_file.exists()
    assert json.loads(state_file.read_text()) == data


def test_save_state_overwrites(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state_file.write_text('{"old": true}')
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    new_data = {"new": True}
    save_state(new_data)
    assert json.loads(state_file.read_text()) == new_data


def test_load_targets_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", tmp_path / "nonexistent.json")
    with pytest.raises(SystemExit):
        load_targets()


def test_load_targets_valid(tmp_path, monkeypatch):
    # Create a targets file and matching directories
    targets_file = tmp_path / "targets.json"
    targets_file.write_text(json.dumps({"targets": ["subdir_a", "subdir_b"]}))
    (tmp_path / "subdir_a").mkdir()
    (tmp_path / "subdir_b").mkdir()
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    targets, excludes = load_targets()
    assert targets == ["subdir_a", "subdir_b"]
    assert excludes == []


def test_load_targets_skips_missing_dirs(tmp_path, monkeypatch):
    targets_file = tmp_path / "targets.json"
    targets_file.write_text(json.dumps({"targets": ["exists", "missing"]}))
    (tmp_path / "exists").mkdir()
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    targets, excludes = load_targets()
    assert targets == ["exists"]
    assert excludes == []


def test_load_targets_with_excludes(tmp_path, monkeypatch):
    targets_file = tmp_path / "targets.json"
    config = {
        "targets": ["src"],
        "exclude": [{"dir": "src/tests", "pattern": "*.yaml"}],
    }
    targets_file.write_text(json.dumps(config))
    (tmp_path / "src").mkdir()
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    targets, excludes = load_targets()
    assert targets == ["src"]
    assert excludes == [{"dir": "src/tests", "pattern": "*.yaml"}]


def test_load_targets_with_multiple_excludes(tmp_path, monkeypatch):
    targets_file = tmp_path / "targets.json"
    config = {
        "targets": ["src"],
        "exclude": [
            {"dir": "src/tests/common", "pattern": "*.yaml"},
            {"dir": "src/docs", "pattern": "*"},
        ],
    }
    targets_file.write_text(json.dumps(config))
    (tmp_path / "src").mkdir()
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    targets, excludes = load_targets()
    assert targets == ["src"]
    assert len(excludes) == 2
    assert excludes[0] == {"dir": "src/tests/common", "pattern": "*.yaml"}
    assert excludes[1] == {"dir": "src/docs", "pattern": "*"}


def test_get_repo_root_success(monkeypatch):
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="/fake/root\n", stderr="")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_result)
    assert get_repo_root() == Path("/fake/root")


def test_get_repo_root_failure(monkeypatch):
    fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="not a repo")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_result)
    with pytest.raises(SystemExit):
        get_repo_root()


def test_git_rev_parse_head_success(monkeypatch):
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123def\n", stderr="")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_result)
    assert git_rev_parse_head() == "abc123def"


def test_git_rev_parse_head_failure(monkeypatch):
    fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_result)
    with pytest.raises(SystemExit):
        git_rev_parse_head()


def test_git_diff_files_success(monkeypatch):
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="src/foo.py\nsrc/bar.cpp\n", stderr=""
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_result)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: Path("/fake"))
    result = git_diff_files("abc123", ["src"])
    assert result == ["src/foo.py", "src/bar.cpp"]


def test_git_diff_files_failure(monkeypatch):
    fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_result)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: Path("/fake"))
    result = git_diff_files("abc123", ["src"])
    assert result == []


def test_git_diff_files_empty_output(monkeypatch):
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_result)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: Path("/fake"))
    result = git_diff_files("abc123", ["src"])
    assert result == []


def test_scan_directories(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    # Create a target with some files
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.cpp").write_text("int main(){}")
    (src / "util.h").write_text("// header")
    (src / "readme.txt").write_text("not documentable")
    (src / ".hidden.py").write_text("hidden")
    # Create a hidden subdir that should be skipped
    hidden = src / ".secret"
    hidden.mkdir()
    (hidden / "secret.py").write_text("secret")

    result = scan_directories(["src"])
    assert "src" in result
    assert sorted(result["src"]) == ["main.cpp", "util.h"]
    assert ".secret" not in str(result)


def test_scan_directories_nested(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    src = tmp_path / "src"
    sub = src / "sub"
    sub.mkdir(parents=True)
    (sub / "lib.py").write_text("pass")
    result = scan_directories(["src"])
    assert "src/sub" in result
    assert result["src/sub"] == ["lib.py"]


def test_scan_directories_skips_build(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    src = tmp_path / "src"
    build = src / "build"
    build.mkdir(parents=True)
    (build / "output.cpp").write_text("generated")
    (src / "real.cpp").write_text("real code")
    result = scan_directories(["src"])
    assert "src/build" not in result
    assert "src" in result


def test_is_excluded_matches():
    excludes = [{"dir": "src/tests/common", "pattern": "*.yaml"}]
    assert is_excluded("src/tests/common", "foo.yaml", excludes) is True


def test_is_excluded_subdir_matches():
    excludes = [{"dir": "src/tests", "pattern": "*.yaml"}]
    assert is_excluded("src/tests/common/deep", "foo.yaml", excludes) is True


def test_is_excluded_no_match_wrong_dir():
    excludes = [{"dir": "src/tests/common", "pattern": "*.yaml"}]
    assert is_excluded("src/lib", "foo.yaml", excludes) is False


def test_is_excluded_no_match_wrong_pattern():
    excludes = [{"dir": "src/tests/common", "pattern": "*.yaml"}]
    assert is_excluded("src/tests/common", "main.cpp", excludes) is False


def test_is_excluded_empty_excludes():
    assert is_excluded("src/tests", "anything.yaml", []) is False


def test_is_excluded_multiple_rules():
    excludes = [
        {"dir": "src/tests", "pattern": "*.yaml"},
        {"dir": "src/build", "pattern": "*.cpp"},
    ]
    assert is_excluded("src/tests/common", "foo.yaml", excludes) is True
    assert is_excluded("src/build/gen", "out.cpp", excludes) is True
    assert is_excluded("src/lib", "real.cpp", excludes) is False


def test_scan_directories_with_excludes(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    # Create target with mixed files
    tests = tmp_path / "src" / "tests" / "common"
    tests.mkdir(parents=True)
    (tests / "test1.yaml").write_text("test")
    (tests / "test2.yaml").write_text("test")
    (tests / "helper.py").write_text("code")  # should NOT be excluded
    (tmp_path / "src" / "main.cpp").write_text("code")

    excludes = [{"dir": "src/tests/common", "pattern": "*.yaml"}]
    result = scan_directories(["src"], excludes)
    # main.cpp should be in src/
    assert "src" in result
    assert "main.cpp" in result["src"]
    # helper.py should remain, yamls should be excluded
    assert "src/tests/common" in result
    assert result["src/tests/common"] == ["helper.py"]


def test_scan_directories_with_multiple_excludes(tmp_path, monkeypatch):
    """Multiple exclude rules each filter their respective directories."""
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    # First excluded dir: yaml tests
    tests = tmp_path / "src" / "tests" / "common"
    tests.mkdir(parents=True)
    (tests / "test1.yaml").write_text("test")
    (tests / "helper.py").write_text("code")
    # Second excluded dir: published docs (exclude everything)
    docs = tmp_path / "src" / "docs"
    docs.mkdir(parents=True)
    (docs / "guide.md").write_text("guide")  # not documentable anyway
    (docs / "build.cmake").write_text("cmake")  # documentable but excluded
    (docs / "setup.sh").write_text("script")  # documentable but excluded
    # Non-excluded dir
    lib = tmp_path / "src" / "lib"
    lib.mkdir(parents=True)
    (lib / "core.cpp").write_text("code")
    (lib / "config.yaml").write_text("config")

    excludes = [
        {"dir": "src/tests/common", "pattern": "*.yaml"},
        {"dir": "src/docs", "pattern": "*"},
    ]
    result = scan_directories(["src"], excludes)
    # tests/common: yaml excluded, helper.py remains
    assert "src/tests/common" in result
    assert result["src/tests/common"] == ["helper.py"]
    # docs: everything excluded, directory should be absent
    assert "src/docs" not in result
    # lib: nothing excluded
    assert "src/lib" in result
    assert "core.cpp" in result["src/lib"]
    assert "config.yaml" in result["src/lib"]


def test_scan_directories_excludes_drops_empty_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    # Directory with only excluded files should be omitted
    tests = tmp_path / "src" / "tests" / "common"
    tests.mkdir(parents=True)
    (tests / "test1.yaml").write_text("test")
    (tests / "test2.yaml").write_text("test")

    excludes = [{"dir": "src/tests/common", "pattern": "*.yaml"}]
    result = scan_directories(["src"], excludes)
    assert "src/tests/common" not in result


def test_has_docs_dir_true(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    (tmp_path / "mydir" / "docs").mkdir(parents=True)
    assert has_docs_dir("mydir") is True


def test_has_docs_dir_false(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    (tmp_path / "mydir").mkdir()
    assert has_docs_dir("mydir") is False


def test_cmd_init_creates_state(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    monkeypatch.setattr("doc_agent_state.git_rev_parse_head", lambda: "fakecommit")

    # Create target dir with a file
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.cpp").write_text("code")
    targets_file = tmp_path / "targets.json"
    targets_file.write_text(json.dumps({"targets": ["src"]}))
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)

    cmd_init(argparse.Namespace())
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state["last_commit"] == "fakecommit"
    assert state["run_count"] == 0
    assert "src" in state["directories"]


def test_cmd_init_with_excludes(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    monkeypatch.setattr("doc_agent_state.git_rev_parse_head", lambda: "fakecommit")

    # Create target dir with mixed files
    tests = tmp_path / "src" / "tests"
    tests.mkdir(parents=True)
    (tests / "test.yaml").write_text("test")
    (tmp_path / "src" / "main.cpp").write_text("code")
    targets_file = tmp_path / "targets.json"
    targets_file.write_text(json.dumps({
        "targets": ["src"],
        "exclude": [{"dir": "src/tests", "pattern": "*.yaml"}],
    }))
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)

    cmd_init(argparse.Namespace())
    state = json.loads(state_file.read_text())
    # src/ should be present (has main.cpp), src/tests should not (only excluded yamls)
    assert "src" in state["directories"]
    assert "src/tests" not in state["directories"]


def test_cmd_init_refuses_if_state_exists(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state_file.write_text("{}")
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    with pytest.raises(SystemExit):
        cmd_init(argparse.Namespace())


def test_cmd_show_no_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("doc_agent_state.STATE_FILE", tmp_path / "nonexistent.json")
    cmd_show(argparse.Namespace())
    assert "No state file found" in capsys.readouterr().out


def test_cmd_show_with_state(tmp_path, monkeypatch, capsys):
    state_file = tmp_path / "state.json"
    data = {"run_count": 5}
    state_file.write_text(json.dumps(data))
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    cmd_show(argparse.Namespace())
    output = capsys.readouterr().out
    assert '"run_count": 5' in output


def test_cmd_mark_visited_updates_state(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state = {
        "last_commit": "abc",
        "directories": {
            "src": {
                "last_visited": None,
                "files_covered": ["old.py"],
                "runs_since_last_visit": 3,
            }
        },
    }
    state_file.write_text(json.dumps(state))
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)

    args = argparse.Namespace(dir="src", covered="a.py,new.py")
    cmd_mark_visited(args)

    updated = json.loads(state_file.read_text())
    entry = updated["directories"]["src"]
    assert "a.py" in entry["files_covered"]
    assert "new.py" in entry["files_covered"]
    assert "old.py" in entry["files_covered"]  # existing covered preserved
    assert "files_uncovered" not in entry
    assert entry["runs_since_last_visit"] == 0
    assert entry["last_visited"] is not None
    assert entry["visited_this_run"] is True


def test_cmd_mark_visited_no_state(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.STATE_FILE", tmp_path / "nonexistent.json")
    args = argparse.Namespace(dir="src", covered="")
    with pytest.raises(SystemExit):
        cmd_mark_visited(args)


def test_cmd_mark_visited_unknown_dir(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"directories": {}}))
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    args = argparse.Namespace(dir="nonexistent", covered="")
    with pytest.raises(SystemExit):
        cmd_mark_visited(args)


def test_cmd_finish_run_increments_counters(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state = {
        "last_commit": "old",
        "last_run": "2024-01-01",
        "run_count": 2,
        "directories": {
            "visited": {
                "last_visited": "2024-01-01",
                "files_covered": [],
                "runs_since_last_visit": 0,
                "visited_this_run": True,
            },
            "not_visited": {
                "last_visited": "2023-01-01",
                "files_covered": [],
                "runs_since_last_visit": 5,
                "visited_this_run": False,
            },
            "never_visited": {
                "last_visited": None,
                "files_covered": [],
                "runs_since_last_visit": 0,
                "visited_this_run": False,
            },
        },
    }
    state_file.write_text(json.dumps(state))
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    monkeypatch.setattr("doc_agent_state.git_rev_parse_head", lambda: "newcommit")

    cmd_finish_run(argparse.Namespace())

    updated = json.loads(state_file.read_text())
    assert updated["run_count"] == 3
    assert updated["last_commit"] == "newcommit"
    # "visited" had visited_this_run=True → stays at 0, flag cleared
    assert updated["directories"]["visited"]["runs_since_last_visit"] == 0
    assert updated["directories"]["visited"]["visited_this_run"] is False
    # "not_visited" had visited_this_run=False → incremented to 6
    assert updated["directories"]["not_visited"]["runs_since_last_visit"] == 6
    # "never_visited" had visited_this_run=False → incremented to 1
    assert updated["directories"]["never_visited"]["runs_since_last_visit"] == 1


def test_cmd_finish_run_no_state(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.STATE_FILE", tmp_path / "nonexistent.json")
    with pytest.raises(SystemExit):
        cmd_finish_run(argparse.Namespace())


def test_cmd_get_work_no_state(tmp_path, monkeypatch):
    monkeypatch.setattr("doc_agent_state.STATE_FILE", tmp_path / "nonexistent.json")
    with pytest.raises(SystemExit):
        cmd_get_work(argparse.Namespace())


def test_cmd_get_work_proactive_no_docs_first(tmp_path, monkeypatch, capsys):
    """Directories without docs/ should appear before those with docs."""
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    monkeypatch.setattr("doc_agent_state.git_diff_files", lambda *a: [])

    # Create two target dirs: one with docs, one without
    (tmp_path / "has_docs" / "docs").mkdir(parents=True)
    (tmp_path / "has_docs" / "file.cpp").write_text("code")
    (tmp_path / "no_docs").mkdir()
    (tmp_path / "no_docs" / "file.py").write_text("code")

    targets_file = tmp_path / "targets.json"
    targets_file.write_text(json.dumps({"targets": ["has_docs", "no_docs"]}))
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)

    state = {
        "last_commit": "abc",
        "directories": {
            "has_docs": {
                "last_visited": None,
                "files_covered": [],
                "runs_since_last_visit": 10,
            },
            "no_docs": {
                "last_visited": None,
                "files_covered": [],
                "runs_since_last_visit": 0,
            },
        },
    }
    state_file.write_text(json.dumps(state))

    cmd_get_work(argparse.Namespace())
    output = json.loads(capsys.readouterr().out.split("\n", 1)[-1])  # skip "State written" line
    assert output["slot1"]["directory"] == "no_docs"
    assert output["slot1"]["source"] == "proactive"


def test_cmd_get_work_computes_uncovered_on_the_fly(tmp_path, monkeypatch, capsys):
    """files_uncovered in output should be computed from all_files - files_covered."""
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)
    monkeypatch.setattr("doc_agent_state.git_diff_files", lambda *a: [])

    src = tmp_path / "src"
    src.mkdir()
    (src / "covered.py").write_text("code")
    (src / "uncovered.py").write_text("code")
    (src / "also_uncovered.cpp").write_text("code")

    targets_file = tmp_path / "targets.json"
    targets_file.write_text(json.dumps({"targets": ["src"]}))
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)

    state = {
        "last_commit": "abc",
        "directories": {
            "src": {
                "last_visited": "2024-01-01",
                "files_covered": ["covered.py"],
                "runs_since_last_visit": 0,
            },
        },
    }
    state_file.write_text(json.dumps(state))

    cmd_get_work(argparse.Namespace())
    output = json.loads(capsys.readouterr().out.split("\n", 1)[-1])

    # Uncovered should be computed: all_files - files_covered
    slot = output["slot1"] or output["slot2"]
    assert sorted(slot["files_uncovered"]) == ["also_uncovered.cpp", "uncovered.py"]
    assert slot["files_covered"] == ["covered.py"]
    assert sorted(slot["all_files"]) == ["also_uncovered.cpp", "covered.py", "uncovered.py"]

    # State should NOT contain files_uncovered
    updated = json.loads(state_file.read_text())
    assert "files_uncovered" not in updated["directories"]["src"]


def test_cmd_get_work_slot2_falls_back_to_reactive(tmp_path, monkeypatch, capsys):
    """When proactive queue is empty, slot2 should fill from reactive queue."""
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)

    # Two directories, both fully documented (proactive queue will be empty)
    for name in ("dir_a", "dir_b"):
        d = tmp_path / name
        (d / "docs").mkdir(parents=True)
        (d / "file.py").write_text("code")

    targets_file = tmp_path / "targets.json"
    targets_file.write_text(json.dumps({"targets": ["dir_a", "dir_b"]}))
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)

    # Both directories have all files covered → empty proactive queue
    # Both have reactive changes
    monkeypatch.setattr("doc_agent_state.git_diff_files",
                        lambda *a: ["dir_a/file.py", "dir_b/file.py"])

    state = {
        "last_commit": "abc",
        "directories": {
            "dir_a": {
                "last_visited": "2024-01-01",
                "files_covered": ["file.py"],
                "runs_since_last_visit": 0,
                "visited_this_run": False,
            },
            "dir_b": {
                "last_visited": "2024-01-01",
                "files_covered": ["file.py"],
                "runs_since_last_visit": 0,
                "visited_this_run": False,
            },
        },
    }
    state_file.write_text(json.dumps(state))

    cmd_get_work(argparse.Namespace())
    output = json.loads(capsys.readouterr().out.split("\n", 1)[-1])

    assert output["slot1"] is not None
    assert output["slot1"]["source"] == "reactive"
    assert output["slot2"] is not None
    assert output["slot2"]["source"] == "reactive"
    assert output["slot1"]["directory"] != output["slot2"]["directory"]


def test_cmd_get_work_saves_pending_reactive(tmp_path, monkeypatch, capsys):
    """Reactive directories not picked for slots are saved to pending_reactive."""
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)

    # Three directories, all fully documented
    for name in ("dir_a", "dir_b", "dir_c"):
        d = tmp_path / name
        (d / "docs").mkdir(parents=True)
        (d / "file.py").write_text("code")

    targets_file = tmp_path / "targets.json"
    targets_file.write_text(json.dumps({"targets": ["dir_a", "dir_b", "dir_c"]}))
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)

    # All three have reactive changes — only 2 can be picked
    monkeypatch.setattr("doc_agent_state.git_diff_files",
                        lambda *a: ["dir_a/file.py", "dir_b/file.py", "dir_c/file.py"])

    state = {
        "last_commit": "abc",
        "directories": {
            name: {
                "last_visited": "2024-01-01",
                "files_covered": ["file.py"],
                "runs_since_last_visit": 0,
                "visited_this_run": False,
            }
            for name in ("dir_a", "dir_b", "dir_c")
        },
    }
    state_file.write_text(json.dumps(state))

    cmd_get_work(argparse.Namespace())
    capsys.readouterr()  # consume output

    updated = json.loads(state_file.read_text())
    # Two dirs were picked for slots, one should be in pending_reactive
    assert len(updated["pending_reactive"]) == 1
    assert updated["pending_reactive"][0] in ("dir_a", "dir_b", "dir_c")


def test_cmd_get_work_merges_pending_reactive(tmp_path, monkeypatch, capsys):
    """pending_reactive from previous run is merged into the reactive queue."""
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    monkeypatch.setattr("doc_agent_state.get_repo_root", lambda: tmp_path)

    # One directory carried over from last run
    d = tmp_path / "dir_a"
    (d / "docs").mkdir(parents=True)
    (d / "file.py").write_text("code")

    targets_file = tmp_path / "targets.json"
    targets_file.write_text(json.dumps({"targets": ["dir_a"]}))
    monkeypatch.setattr("doc_agent_state.TARGETS_FILE", targets_file)

    # No new reactive changes this run
    monkeypatch.setattr("doc_agent_state.git_diff_files", lambda *a: [])

    state = {
        "last_commit": "abc",
        "pending_reactive": ["dir_a"],
        "directories": {
            "dir_a": {
                "last_visited": "2024-01-01",
                "files_covered": ["file.py"],
                "runs_since_last_visit": 0,
                "visited_this_run": False,
            },
        },
    }
    state_file.write_text(json.dumps(state))

    cmd_get_work(argparse.Namespace())
    output = json.loads(capsys.readouterr().out.split("\n", 1)[-1])

    # dir_a should be picked up from pending_reactive
    assert output["slot1"] is not None
    assert output["slot1"]["directory"] == "dir_a"
    assert output["slot1"]["source"] == "reactive"


def test_cmd_finish_run_clears_visited_from_pending_reactive(tmp_path, monkeypatch):
    """finish-run removes visited directories from pending_reactive."""
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("doc_agent_state.STATE_FILE", state_file)
    monkeypatch.setattr("doc_agent_state.git_rev_parse_head", lambda: "newcommit")

    state = {
        "last_commit": "old",
        "last_run": "2024-01-01",
        "run_count": 1,
        "pending_reactive": ["dir_a", "dir_b"],
        "directories": {
            "dir_a": {
                "last_visited": "2024-01-01",
                "files_covered": [],
                "runs_since_last_visit": 0,
                "visited_this_run": True,
            },
            "dir_b": {
                "last_visited": None,
                "files_covered": [],
                "runs_since_last_visit": 0,
                "visited_this_run": False,
            },
        },
    }
    state_file.write_text(json.dumps(state))

    cmd_finish_run(argparse.Namespace())

    updated = json.loads(state_file.read_text())
    # dir_a was visited → removed from pending_reactive
    # dir_b was not visited → stays
    assert updated["pending_reactive"] == ["dir_b"]
