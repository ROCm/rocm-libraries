#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or later is required.")

"""
Unified CLI for Ninja Dependency Analysis and Selective Testing

Features:
- Dependency parsing (from build.ninja)
- Selective test filtering (between git refs)
- Code auditing (--audit)
- Build optimization (--optimize-build)
"""

import argparse
import importlib
import os
import subprocess
import time


# Bridge registry: name -> (module, callable). A "bridge" is an additive
# attribution pass that runs after the ninja-deps mapping and only unions extra
# edges into the parser's in-memory file->executables map (never modifying the
# base include graph). Modules live on the gap-fix branches
# (symbol -> src/symbol_graph, future runtime -> ...) and are imported lazily so
# the base branch works with no bridge selected.
BRIDGE_REGISTRY = {
    "symbol": ("src.symbol_graph", "apply"),
}

# Supersession: selecting the key drops the listed bridges (a superseding bridge
# makes the superseded one redundant). Empty until multiple bridges coexist.
BRIDGE_SUPERSEDES = {}


def resolve_bridges(bridges_arg):
    """Parse the --bridges list, dropping bridges superseded by a selected one."""
    selected = [b.strip() for b in (bridges_arg or "").split(",") if b.strip()]
    for superseding, disabled in BRIDGE_SUPERSEDES.items():
        if superseding in selected:
            for name in disabled:
                if name in selected:
                    selected.remove(name)
                    print(f"bridge '{name}' disabled by '{superseding}'")
    seen = set()
    return [b for b in selected if not (b in seen or seen.add(b))]


def apply_bridges(parser, bridges_arg):
    """Run each selected additive bridge over the in-memory mapping, with timing."""
    for name in resolve_bridges(bridges_arg):
        if name not in BRIDGE_REGISTRY:
            sys.exit(
                f"Unknown bridge '{name}'. Known bridges: {sorted(BRIDGE_REGISTRY)}"
            )
        module_name, func_name = BRIDGE_REGISTRY[name]
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            sys.exit(
                f"Bridge '{name}' is not available on this branch "
                f"(module {module_name} missing): {e}"
            )
        print(f"[bridge:{name}] running...")
        t0 = time.monotonic()
        getattr(module, func_name)(parser)
        print(f"[bridge:{name}] completed in {time.monotonic() - t0:.1f}s")


def run_dependency_parser(build_ninja, ninja, workspace_root, bridges):
    from src.enhanced_ninja_parser import build_mapping, export_mapping

    parser = build_mapping(build_ninja, ninja, workspace_root or "..")
    apply_bridges(parser, bridges)
    export_mapping(parser, os.path.dirname(build_ninja))


def run_selective_test_filter(args):
    from src.selective_test_filter import main as filter_main

    sys.argv = ["selective_test_filter.py"] + args
    filter_main()


def get_git_sha(command):
    try:
        commit_sha = (
            subprocess.check_output(command, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return commit_sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_origin_url(repo_path="."):
    """
    Returns the Git origin URL for the given repository path.
    :param repo_path: Path to the local Git repository (default: current directory)
    :return: Origin URL as a string, or None if not found
    """
    try:
        # Run the git command to get the origin URL
        result = subprocess.run(
            ["git", "-C", repo_path, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("Error: Not a valid Git repository or 'origin' remote not set.")
    except FileNotFoundError:
        print("Error: Git is not installed or not found in PATH.")
    return None


def write_shas_file(context, shas_file, base_ref="origin/develop", source_dir="."):
    """Write base (merge-base with base_ref) and feature (HEAD) SHAs.

    source_dir points at the project's git worktree. For an in-source build (CI)
    this is the default '.'; for an out-of-source build (TheRock) the build dir is
    not a git repo, so the caller passes the MIOpen source dir.
    """
    origin = get_git_origin_url(source_dir)
    print(f"{context}: origin={origin} base_ref={base_ref} source_dir={source_dir}")
    feature_sha = get_git_sha(["git", "-C", source_dir, "rev-parse", "HEAD"])
    base_sha = get_git_sha(["git", "-C", source_dir, "merge-base", "HEAD", base_ref])
    with open(shas_file, "w") as file:
        file.write(f"{base_sha}\n")
        file.write(f"{feature_sha}\n")
    print(f"{context}: {base_sha} <- {feature_sha}")


def read_shas_file(context, shas_file):
    with open(shas_file, "r") as file:
        base_sha = file.readline().strip()
        feature_sha = file.readline().strip()
    print(f"{context}: {base_sha} <- {feature_sha}")
    return (base_sha, feature_sha)


def _finalize_truthy(value):
    return value is not None and str(value).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _atomic_write(path, text):
    """Write text to path via a temp file + os.replace so a concurrent reader on the
    shared filesystem never observes a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def run_finalize_ctest(args):
    """TheRock builder step: burn each Dapper-enabled category's union filter into the
    install CTestTestfile, and retain the full category as a '<name>_original_suite'.

    For each Dapper-enabled category (yaml 'enable_dapper'), the existing '<name>_suite'
    keeps its name but its --gtest_filter is replaced with the subtractive union (honoring
    fallback_mode); a '<name>_original_suite' entry is added that keeps the full original
    filter. Both the original and union filters are recorded in the dapper JSON for
    reference (downloadable record). All computation happens here, at build time, in one
    process; the runner just runs ctest with the burned-in filters (no dapper code ships).

    Fails open: if the yaml or dapper JSON can't be read, the CTestTestfile is copied
    through unchanged so the full categories still run.
    """
    import json
    import re

    from src.dapper_union import resolve_filter

    def _passthrough(reason):
        print(f"finalize-ctest: {reason}; leaving CTestTestfile unmodified.")
        with open(args.ctest_in, "r") as fin:
            _atomic_write(args.ctest_out, fin.read())

    try:
        import yaml

        with open(args.yaml, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001 - fail open on any yaml problem
        _passthrough(f"cannot read yaml '{args.yaml}' ({e})")
        return

    dapper_cats = {
        name
        for name, info in (cfg.get("test_categories") or {}).items()
        if _finalize_truthy((info or {}).get("enable_dapper"))
    }
    if not dapper_cats:
        _passthrough("no Dapper-enabled categories in yaml")
        return

    try:
        with open(args.dapper_json, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        _passthrough(f"cannot read dapper json '{args.dapper_json}' ({e})")
        return
    dapper_filter = data.get("dapper_filter", "")
    fallback_mode = data.get("fallback_mode", "union")

    add_test_re = re.compile(r"^\s*add_test\((\S+)\s")
    setprops_re = re.compile(r"^\s*set_tests_properties\((\S+)\s")
    filter_re = re.compile(r"--gtest_filter=([^\s)]+)")

    def match_category(name):
        # Suite names are '<prefix>_<category>_suite'; match by category suffix so we do
        # not depend on the prefix. Prefer the longest matching category name.
        if not name.endswith("_suite"):
            return None
        base = name[: -len("_suite")]
        best = None
        for cat in dapper_cats:
            if (base == cat or base.endswith("_" + cat)) and (
                best is None or len(cat) > len(best)
            ):
                best = cat
        return best

    def original_name(name):
        return name[: -len("_suite")] + "_original_suite"

    with open(args.ctest_in, "r") as f:
        lines = f.readlines()

    rewritten = {}  # union-suite name -> original-suite name
    processed = set()  # category names finalized
    out = []
    for line in lines:
        m = add_test_re.match(line)
        if m:
            name = m.group(1)
            cat = match_category(name)
            fm = filter_re.search(line) if cat else None
            if cat and fm:
                original_filter = fm.group(1)
                union = resolve_filter(
                    dapper_filter, fallback_mode, cat, original_filter
                )
                name_orig = original_name(name)
                out.append(
                    line.replace(
                        f"--gtest_filter={original_filter}",
                        f"--gtest_filter={union}",
                        1,
                    )
                )
                out.append(
                    line.replace(f"add_test({name} ", f"add_test({name_orig} ", 1)
                )
                rewritten[name] = name_orig
                processed.add(cat)
                data[f"category_{cat}_filter"] = original_filter
                data[f"category_{cat}_union"] = union
                continue
        sm = setprops_re.match(line)
        if sm and sm.group(1) in rewritten:
            name = sm.group(1)
            out.append(line)  # properties for the union suite (name unchanged)
            out.append(
                line.replace(name, rewritten[name], 1)
            )  # ...and the _original suite
            continue
        out.append(line)

    _atomic_write(args.ctest_out, "".join(out))
    data["dapper_categories"] = sorted(processed)
    _atomic_write(args.dapper_json, json.dumps(data, indent=2))
    print(
        f"finalize-ctest: burned union into {len(rewritten)} dapper suite(s) "
        f"({', '.join(sorted(processed)) or 'none'}); wrote {args.ctest_out}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unified Ninja Dependency & Selective Testing Tool"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sha selection
    parser_shas = subparsers.add_parser(
        "shas",
        help="Retrieve sha for merge-base and feature branch and storing in miopen_gtest_shas.txt.",
    )
    parser_shas.add_argument(
        "--base-ref",
        default="origin/develop",
        help="Git ref to merge-base against for the impact diff (default origin/develop).",
    )
    parser_shas.add_argument(
        "--source-dir",
        default=".",
        help="Project git worktree (for out-of-source builds, e.g. TheRock).",
    )

    # Dependency parsing
    parser_parse = subparsers.add_parser(
        "parse", help="Parse build.ninja and generate dependency mapping"
    )
    parser_parse.add_argument("build_ninja", help="Path to build.ninja")
    parser_parse.add_argument(
        "--ninja", help="Path to ninja executable", default="ninja"
    )
    parser_parse.add_argument(
        "--workspace-root", help="Path to workspace root", default=None
    )
    parser_parse.add_argument(
        "--bridges",
        default="",
        help="Comma-separated additive attribution bridges to run after the "
        "ninja-deps mapping (e.g. 'symbol'). Empty = none.",
    )

    # Selective testing
    parser_test = subparsers.add_parser(
        "select", help="Selective test filtering between git refs"
    )
    parser_test.add_argument("depmap_json", help="Path to dependency mapping JSON")
    parser_test.add_argument(
        "--base-sha",
        help="git base sha",
        default="None",
    )
    parser_test.add_argument(
        "--feature-sha",
        help="git feature sha",
        default="None",
    )
    parser_test.add_argument(
        "--all", action="store_true", help="Include all executables"
    )
    parser_test.add_argument(
        "--test-prefix",
        action="store_true",
        help="Only include executables starting with 'test_'",
    )
    parser_test.add_argument(
        "--output", help="Output JSON file", default="miopen_dapper_tests.json"
    )
    parser_test.add_argument(
        "--fixturemap",
        help="Optional path to file containing the test <-> gtest fixture mapping",
        default="",
    )
    parser_test.add_argument(
        "--shardsfile",
        help="Optional path to file containing a list of gtest shard output files",
        default="",
    )
    parser_test.add_argument(
        "--source-dir",
        default=".",
        help="Project git worktree for the impact diff (out-of-source builds, e.g. TheRock).",
    )

    # Code auditing
    parser_audit = subparsers.add_parser(
        "audit", help="List all files and their dependent executables"
    )
    parser_audit.add_argument("depmap_json", help="Path to dependency mapping JSON")

    # Build optimization
    parser_opt = subparsers.add_parser(
        "optimize", help="List affected executables for changed files"
    )
    parser_opt.add_argument("depmap_json", help="Path to dependency mapping JSON")
    parser_opt.add_argument("changed_files", nargs="+", help="List of changed files")

    # TheRock: burn per-category union filters into the install CTestTestfile.
    parser_finalize = subparsers.add_parser(
        "finalize-ctest",
        help="Burn per-category Dapper union filters into the install CTestTestfile "
        "and add '<name>_original_suite' entries retaining the full filters (TheRock).",
    )
    parser_finalize.add_argument(
        "--ctest-in", required=True, help="Configure-generated install CTestTestfile"
    )
    parser_finalize.add_argument(
        "--ctest-out", required=True, help="Path to write the finalized CTestTestfile"
    )
    parser_finalize.add_argument(
        "--yaml", required=True, help="test_categories.yaml (for 'enable_dapper')"
    )
    parser_finalize.add_argument(
        "--dapper-json",
        required=True,
        help="miopen_dapper_tests.json (dapper_filter + fallback_mode; augmented in place)",
    )

    args = parser.parse_args()
    shas_file = "miopen_dapper_shas.txt"

    if args.command == "shas":
        write_shas_file("MAIN SHAS: ", shas_file, args.base_ref, args.source_dir)
    elif args.command == "parse":
        if not os.path.isfile(shas_file):
            write_shas_file("MAIN PARSE: ", shas_file)
        run_dependency_parser(
            args.build_ninja, args.ninja, args.workspace_root, args.bridges
        )
    elif args.command == "select":
        filter_args = [args.depmap_json]
        (base_sha, feature_sha) = read_shas_file("MAIN SELECT", shas_file)
        filter_args.append(base_sha)
        filter_args.append(feature_sha)
        if args.test_prefix:
            filter_args.append("--test-prefix")
        if args.all:
            filter_args.append("--all")
        if args.output:
            filter_args += ["--output", args.output]
        if args.fixturemap:
            filter_args += ["--fixturemap", args.fixturemap]
        if args.shardsfile:
            print(f"main: ADDED SHARDSFILE: {args.shardsfile}")
            filter_args += ["--shardsfile", args.shardsfile]
        if args.source_dir:
            filter_args += ["--source-dir", args.source_dir]
        run_selective_test_filter(filter_args)
    elif args.command == "audit":
        run_selective_test_filter([args.depmap_json, "--audit"])
    elif args.command == "optimize":
        run_selective_test_filter(
            [args.depmap_json, "--optimize-build"] + args.changed_files
        )
    elif args.command == "finalize-ctest":
        run_finalize_ctest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
