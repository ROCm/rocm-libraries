#!/usr/bin/env python3
"""
Selective Test Filter Tool

Given two git refs (branches or commit IDs), this tool:
- Identifies changed files between the refs
- Loads the enhanced dependency mapping JSON (from enhanced_ninja_parser.py)
- Maps changed files to affected test executables (optionally filtering for "test_" prefix)
- Exports the list of tests to run to tests-to-run.json

Usage:
  python selective_test_filter.py <depmap_json> <ref1> <ref2> [--all | --test-prefix] [--output <output_json>]

Arguments:
  <depmap_json>   Path to miopen_dapper_mapping.json
  <ref1>          Source git ref (branch or commit)
  <ref2>          Target git ref (branch or commit)

Options:
  --all           Include all executables (default)
  --test-prefix   Only include executables starting with "test_"
  --output        Output JSON file (default: tests-to-run.json)
  --folder        Relative path to comparing folder
"""

import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or later is required.")

import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def _miopen_repo_prefix():
    """Repo-root-relative path of the MIOpen project (e.g. 'projects/miopen/'), used to
    discard changed files outside MIOpen.

    Derived from this script's own location, NOT from the git cwd: `git diff --name-only`
    is always repo-root-relative, but git runs in different places per build layout (MICI:
    the build dir; TheRock: the source dir), so the cwd can't tell us the project prefix.
    The script always lives at <miopen>/script/dependency-parser/src/, so parents[3] is the
    MIOpen source dir in both layouts (src=0, dependency-parser=1, script=2, miopen=3).
    Returns '' if MIOpen is itself the repo root (nothing to discard), or the literal
    fallback if git can't be queried.
    """
    try:
        miopen_dir = Path(__file__).resolve().parents[3]
        return subprocess.run(
            ["git", "-C", str(miopen_dir), "rev-parse", "--show-prefix"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()  # e.g. "projects/miopen/", or "" at the repo root
    except Exception:
        return "projects/miopen/"


def get_changed_files(ref1, ref2, path_to_folder, source_dir="."):
    """Return the set of files changed between two git refs, or None if the refs or
    the diff cannot be resolved.

    source_dir is the project's git worktree; for an out-of-source build (TheRock)
    the caller passes the MIOpen source dir since the build dir is not a git repo.
    Returning None lets the caller fail open (run the entire category) rather than
    crash when the base ref / history is unavailable.
    """
    git = ["git", "-C", source_dir]
    try:
        base_commit = subprocess.run(
            git + ["show", "-s", '--format="%h  %ad  %s"', "--date=iso", f"{ref1}"],
            capture_output=True,
            text=True,
            check=True,
        )
        feat_commit = subprocess.run(
            git + ["show", "-s", '--format="%h  %ad  %s"', "--date=iso", f"{ref2}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"DAPPER: could not resolve refs {ref1}..{ref2} in {source_dir}: {e}")
        return None
    print(f"DAPPER branches: MERGE-BASE: {base_commit.stdout.strip()}")
    print(f"                    FEATURE: {feat_commit.stdout.strip()}")

    args = git + ["diff", "--name-only", ref1, ref2]
    if path_to_folder:
        args += ["--", path_to_folder]
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        # Discard changed files outside MIOpen: git diff spans the whole rocm-libraries
        # monorepo, but only MIOpen sources/tests can map to MIOpen fixtures. Dropping the
        # rest keeps cross-project changes (composablekernel, hiptensor, ...) from feeding
        # select/classify_fallback and spuriously forcing entire_category.
        prefix = _miopen_repo_prefix()
        if prefix:
            kept = [ln for ln in lines if ln.startswith(prefix)]
            dropped = len(lines) - len(kept)
            if dropped:
                print(
                    f"DAPPER: discarded {dropped} non-MIOpen changed file(s) "
                    f"outside {prefix}"
                )
            lines = kept
        # parts[2:] drops the 'projects/miopen/' prefix -> project-relative paths that match
        # the mapping keys (consistent with the prefix filtered above).
        files = set(str(Path(*Path(line).parts[2:])).strip() for line in lines)
        return files
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e}")
        return None


def load_depmap(depmap_json):
    """Load the dependency mapping JSON."""
    with open(depmap_json, "r") as f:
        depmap = json.load(f)
    # TODO: remove old format
    if "file_to_executables" in depmap:
        return depmap["file_to_executables"]
    return depmap


def load_fixturemap(fixturemap_json):
    """Load the dependency mapping JSON."""
    with open(fixturemap_json, "r") as f:
        fixturemap = json.load(f)
    for test, fixtures in fixturemap.items():
        for i, fixture in enumerate(fixtures):
            if not fixture.endswith("*"):
                fixtures[i] = fixture + "*"
    return fixturemap


def select_tests(file_to_executables, changed_files, filter_mode):
    """Return a set of test executables affected by changed files."""
    affected = set()
    for f in changed_files:
        if f in file_to_executables:
            for exe in file_to_executables[f]:
                if filter_mode == "all":
                    affected.add(exe)
                elif filter_mode == "test_prefix" and (
                    exe.startswith("bin/test_") or exe == "bin/miopen_gtest"
                ):
                    affected.add(exe)
    return sorted(affected)


def create_gtest_filter(tests_to_run, fixturemap):
    gtest_filter = ""
    if fixturemap:
        for file in tests_to_run:
            if file not in fixturemap:
                if file != "bin/miopen_gtest":
                    print(
                        f"Warning: binary {file} was marked for run, but no gtests were found"
                    )
            else:
                for test in fixturemap[file]:
                    gtest_filter += test + ":"
        gtest_filter = gtest_filter[:-1]
    else:
        gtest_filter = "*"
    return gtest_filter


def classify_fallback(
    changed_files, file_to_executables, fixturemap, compiled_sources, gtest_filter
):
    """Decide how the runner should behave when attribution is incomplete.

    - 'union'           : changes were attributed to fixtures -> run the subtractive
                          union (dapper's normal, time-saving mode).
    - 'entire_category' : >=1 changed file is part of the build graph but attributed to
                          NO fixtures (e.g. a common .cpp body the include graph cannot
                          reach). Run the entire category filter -- safe, never misses,
                          and never the near-empty minimal default. Bridges move such
                          files back to 'union'.
    - 'minimal'         : no test-relevant change -> minimal smoke default.
    """
    if not changed_files:
        return "minimal"
    saw_attributed = bool(gtest_filter)
    for f in changed_files:
        exes = file_to_executables.get(f, [])
        if any(e in fixturemap for e in exes):
            continue  # attributed to a fixture-bearing test
        # Unattributed. Treat as test-relevant if it is compiled anywhere in the
        # build, or is a dependency (e.g. header) of some compiled object.
        if f in compiled_sources or exes:
            return "entire_category"
    return "union" if saw_attributed else "minimal"


def _xml_timestamp(ts):
    return ts.split(".")[0] + "Z"


def _xml_time(t):
    return t + "s"


def _xml_testcase(case):
    a = case.attrib
    tc = {
        "name": a["name"],
        "file": a["file"],
        "line": int(a["line"]),
        "status": a["status"].upper(),
        "result": a["result"].upper(),
        "timestamp": _xml_timestamp(a["timestamp"]),
        "time": _xml_time(a["time"]),
        "classname": a["classname"],
    }
    if "value_param" in a:
        tc["value_param"] = a["value_param"]
    failure_elems = [c for c in case if c.tag == "failure"]
    if failure_elems:
        tc["failures"] = [
            {"failure": f.attrib["message"], "type": f.attrib.get("type", "")}
            for f in failure_elems
        ]
    # reorder to match gtest json field order
    ordered = {}
    for key in (
        "name",
        "value_param",
        "file",
        "line",
        "status",
        "result",
        "timestamp",
        "time",
        "classname",
        "failures",
    ):
        if key in tc:
            ordered[key] = tc[key]
    return ordered


def _xml_testsuite(suite):
    a = suite.attrib
    return {
        "name": a["name"],
        "tests": int(a["tests"]),
        "failures": int(a["failures"]),
        "disabled": int(a["disabled"]),
        "errors": int(a["errors"]),
        "timestamp": _xml_timestamp(a["timestamp"]),
        "time": _xml_time(a["time"]),
        "testsuite": [_xml_testcase(c) for c in suite if c.tag == "testcase"],
    }


def _xml_to_gtest_json(xml_path):
    root = ET.parse(xml_path).getroot()
    a = root.attrib
    return {
        "tests": int(a["tests"]),
        "failures": int(a["failures"]),
        "disabled": int(a["disabled"]),
        "errors": int(a["errors"]),
        "timestamp": _xml_timestamp(a["timestamp"]),
        "time": _xml_time(a["time"]),
        "name": a["name"],
        "testsuites": [_xml_testsuite(s) for s in root if s.tag == "testsuite"],
    }


def load_shards(shardsfile):
    """Read shard filenames from shardsfile; store xml paths as-is (conversion happens later)."""
    return [line for line in open(shardsfile).read().splitlines() if line.strip()]


def main():
    if "--audit" in sys.argv:
        if len(sys.argv) < 2:
            print("Usage: python selective_test_filter.py <depmap_json> --audit")
            sys.exit(1)
        depmap_json = sys.argv[1]
        if not os.path.exists(depmap_json):
            print(f"Dependency map JSON not found: {depmap_json}")
            sys.exit(1)
        file_to_executables = load_depmap(depmap_json)
        for f, exes in file_to_executables.items():
            print(f"{f}: {', '.join(exes)}")
        print(f"Total files: {len(file_to_executables)}")
        sys.exit(0)

    if "--optimize-build" in sys.argv:
        if len(sys.argv) < 3:
            print(
                "Usage: python selective_test_filter.py <depmap_json> --optimize-build <changed_file1> [<changed_file2> ...]"
            )
            sys.exit(1)
        depmap_json = sys.argv[1]
        changed_files = set(sys.argv[sys.argv.index("--optimize-build") + 1 :])
        if not os.path.exists(depmap_json):
            print(f"Dependency map JSON not found: {depmap_json}")
            sys.exit(1)
        file_to_executables = load_depmap(depmap_json)
        affected_executables = set()
        for f in changed_files:
            if f in file_to_executables:
                affected_executables.update(file_to_executables[f])
        print("Affected executables:")
        for exe in sorted(affected_executables):
            print(exe)
        print(f"Total affected executables: {len(affected_executables)}")
        sys.exit(0)

    if len(sys.argv) < 4:
        print(
            "Usage: python selective_test_filter.py <depmap_json> <ref1> <ref2> [--all | --test-prefix] [--output <output_json>] [--folder <path_to_folder>]"
        )
        sys.exit(1)

    depmap_json = sys.argv[1]
    ref1 = sys.argv[2]
    ref2 = sys.argv[3]
    filter_mode = "test_prefix"
    output_json = "tests-to-run.json"
    path_to_folder = ""
    fixturemap_json = ""
    shardsfile = ""

    if "--test-prefix" in sys.argv:
        filter_mode = "test_prefix"
    if "--all" in sys.argv:
        filter_mode = "all"
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_json = sys.argv[idx + 1]
    if "--folder" in sys.argv:
        idx = sys.argv.index("--folder")
        if idx + 1 < len(sys.argv):
            path_to_folder = sys.argv[idx + 1]
    if "--fixturemap" in sys.argv:
        idx = sys.argv.index("--fixturemap")
        if idx + 1 < len(sys.argv):
            fixturemap_json = sys.argv[idx + 1]
    if "--shardsfile" in sys.argv:
        idx = sys.argv.index("--shardsfile")
        if idx + 1 < len(sys.argv):
            shardsfile = sys.argv[idx + 1]
    source_dir = "."
    if "--source-dir" in sys.argv:
        idx = sys.argv.index("--source-dir")
        if idx + 1 < len(sys.argv):
            source_dir = sys.argv[idx + 1]

    if not os.path.exists(depmap_json):
        print(f"Dependency map JSON not found: {depmap_json}")
        sys.exit(1)

    changed_files = get_changed_files(ref1, ref2, path_to_folder, source_dir)

    # Load the mapping once: file->executables plus the compiled-source set used to
    # classify unattributed-but-compiled changes.
    with open(depmap_json, "r") as f:
        depmap_raw = json.load(f)
    file_to_executables = depmap_raw.get("file_to_executables", depmap_raw)
    compiled_sources = set(depmap_raw.get("compiled_sources", []))
    fixturemap = load_fixturemap(fixturemap_json) if fixturemap_json else None

    if changed_files is None:
        # Could not determine the diff (missing base ref / shallow history). Fail open:
        # run the entire category so nothing is silently skipped.
        print("DAPPER: changed files undetermined; using entire_category fallback.")
        tests = []
        gtest_filter = ""
        fallback_mode = "entire_category"
    elif not changed_files:
        print("No changed files detected.")
        tests = []
        gtest_filter = ""
        fallback_mode = "minimal"
    else:
        tests = select_tests(file_to_executables, changed_files, filter_mode)
        gtest_filter = create_gtest_filter(tests, fixturemap)
        if fixturemap:
            fallback_mode = classify_fallback(
                changed_files,
                file_to_executables,
                fixturemap,
                compiled_sources,
                gtest_filter,
            )
        else:
            # No fixture map -> gtest_filter is "*" (run all); union is the safe label.
            fallback_mode = "union"

    with open(output_json, "w") as f:
        json.dump(
            {
                "tests_to_run": tests,
                "dapper_filter": gtest_filter,
                "fallback_mode": fallback_mode,
                "changed_files": sorted(changed_files) if changed_files else [],
            },
            f,
            indent=2,
        )

    print(f"Exported {len(tests)} test fixtures to run to {output_json}")


if __name__ == "__main__":
    main()
