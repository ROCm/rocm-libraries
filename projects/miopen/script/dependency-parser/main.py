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
# (stem -> src/common_stem, symbol -> src/symbol_graph, future runtime -> ...)
# and are imported lazily so the base branch works with no bridge selected.
BRIDGE_REGISTRY = {
    "stem": ("src.common_stem", "apply"),
    "symbol": ("src.symbol_graph", "apply"),
}

# Supersession: selecting the key drops the listed bridges (a superseding bridge
# makes the superseded one redundant). The symbol bridge is correctness-dominant
# over the stem bridge, so selecting 'symbol' disables 'stem'.
BRIDGE_SUPERSEDES = {"symbol": ["stem"]}


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


def write_shas_file(context, shas_file):
    origin = get_git_origin_url()
    print(f"{context}: origin={origin}")
    feature_sha = get_git_sha(["git", "rev-parse", "HEAD"])
    base_sha = get_git_sha(["git", "merge-base", "HEAD", "origin/develop"])
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
        "--use-cached",
        action="store_true",
        help="Reuse the existing shas file if present (skip); error if it is missing.",
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
        "ninja-deps mapping (e.g. 'stem', 'symbol'). Empty = none. If 'symbol' "
        "is listed it supersedes 'stem'.",
    )
    parser_parse.add_argument(
        "--use-cached",
        action="store_true",
        help="Reuse the existing mapping JSON if present (skip parse); error if missing.",
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

    args = parser.parse_args()
    shas_file = "miopen_dapper_shas.txt"

    if args.command == "shas":
        if args.use_cached:
            if os.path.isfile(shas_file):
                print(f"dapper(cached): using existing {shas_file}")
                return
            sys.exit(
                f"dapper(cached): {shas_file} not found. MIOPEN_DAPPER_USE_CACHED reuses "
                "existing dapper inputs; reconfigure once with -DMIOPEN_DAPPER_USE_CACHED=OFF "
                "to generate them."
            )
        write_shas_file("MAIN SHAS: ", shas_file)
    elif args.command == "parse":
        mapping_json = os.path.join(
            os.path.dirname(args.build_ninja) or ".", "miopen_dapper_mapping.json"
        )
        if args.use_cached:
            if os.path.isfile(mapping_json):
                print(f"dapper(cached): using existing {mapping_json}")
                return
            sys.exit(
                f"dapper(cached): {mapping_json} not found. MIOPEN_DAPPER_USE_CACHED reuses "
                "existing dapper inputs; reconfigure once with -DMIOPEN_DAPPER_USE_CACHED=OFF "
                "to generate them."
            )
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
        run_selective_test_filter(filter_args)
    elif args.command == "audit":
        run_selective_test_filter([args.depmap_json, "--audit"])
    elif args.command == "optimize":
        run_selective_test_filter(
            [args.depmap_json, "--optimize-build"] + args.changed_files
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
