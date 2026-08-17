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
import os
import subprocess


def run_dependency_parser(args):
    from src.enhanced_ninja_parser import main as ninja_main

    sys.argv = ["enhanced_ninja_parser.py"] + args
    ninja_main()


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
        write_shas_file("MAIN SHAS: ", shas_file)
    elif args.command == "parse":
        if not os.path.isfile(shas_file):
            write_shas_file("MAIN PARSE: ", shas_file)
        parse_args = [args.build_ninja, args.ninja]
        if args.workspace_root:
            parse_args.append(args.workspace_root)
        run_dependency_parser(parse_args)
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
