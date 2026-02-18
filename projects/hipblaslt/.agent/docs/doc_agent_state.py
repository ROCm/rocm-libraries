#!/usr/bin/env python3
"""
State management helper for the documentation agent.

Usage:
    python doc_agent_state.py init
    python doc_agent_state.py get-work
    python doc_agent_state.py mark-visited --dir DIR --covered FILE1,FILE2 --uncovered FILE3,FILE4
    python doc_agent_state.py finish-run
    python doc_agent_state.py show
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STATE_FILE = SCRIPT_DIR / ".doc-agent-state.json"
TARGETS_FILE = SCRIPT_DIR / "targets.json"
DOCUMENTABLE_EXTENSIONS = {".py", ".cpp", ".h", ".hpp", ".yaml", ".yml", ".cmake", ".sh"}
SKIP_DIRS = {"docs", "__pycache__", ".git", "build", "dist", ".eggs", "*.egg-info"}


def load_targets():
    if not TARGETS_FILE.exists():
        print(f"Error: targets file not found at {TARGETS_FILE}", file=sys.stderr)
        sys.exit(1)
    with open(TARGETS_FILE) as f:
        config = json.load(f)
    repo_root = get_repo_root()
    targets = []
    for t in config["targets"]:
        target_path = repo_root / t
        if not target_path.is_dir():
            print(f"Warning: target directory '{t}' does not exist, skipping.", file=sys.stderr)
            continue
        targets.append(t)
    return targets


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State written to {STATE_FILE}")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_repo_root():
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=SCRIPT_DIR
    )
    if result.returncode != 0:
        print("Error: not inside a git repository", file=sys.stderr)
        sys.exit(1)
    return Path(result.stdout.strip())


def git_rev_parse_head():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=SCRIPT_DIR
    )
    if result.returncode != 0:
        print(f"Error running git rev-parse HEAD: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def git_diff_files(last_commit, targets):
    repo_root = get_repo_root()
    cmd = ["git", "diff", "--name-only", f"{last_commit}..HEAD", "--"] + targets
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]


def should_skip_dir(dirname):
    if dirname.startswith("."):
        return True
    return dirname in SKIP_DIRS


def is_documentable(filename):
    if filename.startswith("."):
        return False
    return Path(filename).suffix in DOCUMENTABLE_EXTENSIONS


def scan_directories(targets):
    """Scan all target roots and return a dict of dir_path -> list of documentable filenames."""
    repo_root = get_repo_root()
    directories = {}
    for target in targets:
        code_root = repo_root / target
        for root, dirs, files in os.walk(code_root):
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]
            documentable = [f for f in files if is_documentable(f)]
            if documentable:
                rel_path = str(Path(root).relative_to(repo_root))
                directories[rel_path] = sorted(documentable)
    return directories


def has_docs_dir(dir_path):
    repo_root = get_repo_root()
    docs_path = repo_root / dir_path / "docs"
    return docs_path.is_dir()


def cmd_init(_args):
    """Initialize the state file by scanning the directory tree."""
    if STATE_FILE.exists():
        print("State file already exists. Use 'show' to inspect it.", file=sys.stderr)
        sys.exit(1)

    targets = load_targets()
    scanned = scan_directories(targets)
    directories = {}
    for dir_path, files in scanned.items():
        directories[dir_path] = {
            "last_visited": None,
            "files_covered": [],
            "files_uncovered": [],
            "runs_since_last_visit": 0,
        }

    state = {
        "last_commit": git_rev_parse_head(),
        "last_run": now_iso(),
        "run_count": 0,
        "directories": directories,
    }
    save_state(state)
    print(f"Initialized state with {len(directories)} directories across {len(targets)} target(s).")


def cmd_get_work(_args):
    """Determine the two work items for this run and print them."""
    state = load_state()
    if state is None:
        print("No state file found. Run 'init' first.", file=sys.stderr)
        sys.exit(1)

    targets = load_targets()

    # Refresh directory list: add new dirs, remove deleted ones
    scanned = scan_directories(targets)
    for dir_path in scanned:
        if dir_path not in state["directories"]:
            state["directories"][dir_path] = {
                "last_visited": None,
                "files_covered": [],
                "files_uncovered": [],
                "runs_since_last_visit": 0,
            }
    removed = [d for d in state["directories"] if d not in scanned]
    for d in removed:
        del state["directories"][d]

    # Build reactive queue
    reactive_queue = []
    last_commit = state.get("last_commit")
    if last_commit:
        changed_files = git_diff_files(last_commit, targets)
        dir_change_count = {}
        for f in changed_files:
            parent = str(Path(f).parent)
            if parent in state["directories"]:
                dir_change_count[parent] = dir_change_count.get(parent, 0) + 1
        reactive_queue = sorted(dir_change_count.keys(),
                                key=lambda d: dir_change_count[d], reverse=True)

    # Build proactive queue
    no_docs = []
    partial_docs = []
    stalest = []

    for dir_path, entry in state["directories"].items():
        if not has_docs_dir(dir_path):
            no_docs.append(dir_path)
        elif entry["files_uncovered"]:
            partial_docs.append(dir_path)
        else:
            stalest.append(dir_path)

    # Sort stalest by runs_since_last_visit descending
    stalest.sort(key=lambda d: state["directories"][d]["runs_since_last_visit"],
                 reverse=True)

    proactive_queue = no_docs + partial_docs + stalest

    # Pick work items
    slot1 = None
    slot1_source = None
    slot2 = None
    slot2_source = None

    if reactive_queue:
        slot1 = reactive_queue[0]
        slot1_source = "reactive"
    elif proactive_queue:
        slot1 = proactive_queue[0]
        slot1_source = "proactive"

    for candidate in proactive_queue:
        if candidate != slot1:
            slot2 = candidate
            slot2_source = "proactive"
            break

    # Print results
    output = {
        "slot1": None,
        "slot2": None,
        "reactive_queue_size": len(reactive_queue),
        "proactive_queue_size": len(proactive_queue),
    }

    if slot1:
        entry = state["directories"][slot1]
        output["slot1"] = {
            "directory": slot1,
            "source": slot1_source,
            "has_docs": has_docs_dir(slot1),
            "files_covered": entry["files_covered"],
            "files_uncovered": entry["files_uncovered"],
            "all_files": scanned.get(slot1, []),
            "runs_since_last_visit": entry["runs_since_last_visit"],
        }

    if slot2:
        entry = state["directories"][slot2]
        output["slot2"] = {
            "directory": slot2,
            "source": slot2_source,
            "has_docs": has_docs_dir(slot2),
            "files_covered": entry["files_covered"],
            "files_uncovered": entry["files_uncovered"],
            "all_files": scanned.get(slot2, []),
            "runs_since_last_visit": entry["runs_since_last_visit"],
        }

    # Save any directory list updates we made
    save_state(state)

    print(json.dumps(output, indent=2))


def cmd_mark_visited(args):
    """Mark a directory as visited and update its covered/uncovered source file lists."""
    state = load_state()
    if state is None:
        print("No state file found. Run 'init' first.", file=sys.stderr)
        sys.exit(1)

    dir_path = args.dir
    if dir_path not in state["directories"]:
        print(f"Directory '{dir_path}' not found in state.", file=sys.stderr)
        sys.exit(1)

    covered = [f.strip() for f in args.covered.split(",") if f.strip()] if args.covered else []
    uncovered = [f.strip() for f in args.uncovered.split(",") if f.strip()] if args.uncovered else []

    entry = state["directories"][dir_path]
    # Add newly covered source files (avoid duplicates)
    existing_covered = set(entry["files_covered"])
    for f in covered:
        existing_covered.add(f)
    entry["files_covered"] = sorted(existing_covered)

    # Update uncovered: set to provided list (agent determines what's still uncovered)
    entry["files_uncovered"] = sorted(uncovered)

    entry["last_visited"] = now_iso()
    entry["runs_since_last_visit"] = 0

    save_state(state)
    print(f"Marked '{dir_path}' as visited. Covered: {len(entry['files_covered'])}, Uncovered: {len(entry['files_uncovered'])}")


def cmd_finish_run(_args):
    """End-of-run bookkeeping: increment counters, update commit hash."""
    state = load_state()
    if state is None:
        print("No state file found. Run 'init' first.", file=sys.stderr)
        sys.exit(1)

    # Increment runs_since_last_visit for all directories not visited this run
    # (directories visited this run already have runs_since_last_visit = 0
    #  from mark-visited)
    for entry in state["directories"].values():
        if entry["runs_since_last_visit"] != 0 or entry["last_visited"] is None:
            entry["runs_since_last_visit"] += 1

    state["last_commit"] = git_rev_parse_head()
    state["last_run"] = now_iso()
    state["run_count"] = state.get("run_count", 0) + 1

    save_state(state)
    print(f"Run #{state['run_count']} completed.")


def cmd_show(_args):
    """Print the current state file contents."""
    state = load_state()
    if state is None:
        print("No state file found.")
        return
    print(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Documentation agent state manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize state file by scanning directory tree")
    subparsers.add_parser("get-work", help="Get the two work items for this run")

    mark_parser = subparsers.add_parser("mark-visited", help="Mark a directory as visited")
    mark_parser.add_argument("--dir", required=True, help="Directory path (relative to repo root)")
    mark_parser.add_argument("--covered", default="", help="Comma-separated source files now covered by documentation")
    mark_parser.add_argument("--uncovered", default="", help="Comma-separated source files not yet covered by documentation")

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
