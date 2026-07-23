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
import json
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


def _run_git(args):
    """Run a git command; return (returncode, stdout_stripped, stderr_stripped)."""
    try:
        p = subprocess.run(args, capture_output=True, text=True)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:  # noqa: BLE001
        return 1, "", f"{type(e).__name__}: {e}"


def _ci_base_candidates():
    """Authoritative base candidates from the CI environment. On GitHub Actions the PR base
    is github.event.pull_request.base.sha (in the event payload at $GITHUB_EVENT_PATH) and
    GITHUB_BASE_REF holds the target branch name. Returns [(why, value)], most authoritative
    first. Optional -- resolve_base_sha also falls back to a blobless develop fetch."""
    cands = []
    ev = os.environ.get("GITHUB_EVENT_PATH")
    if ev and os.path.isfile(ev):
        try:
            with open(ev) as f:
                base = (json.load(f).get("pull_request") or {}).get("base") or {}
            if base.get("sha"):
                cands.append(("event.base.sha", base["sha"]))
            if base.get("ref"):
                cands.append(("event.base.ref", base["ref"]))
        except Exception as e:  # noqa: BLE001
            print(f"DAPPER base: could not parse GITHUB_EVENT_PATH ({e})")
    if os.environ.get("GITHUB_BASE_REF"):
        cands.append(("GITHUB_BASE_REF", os.environ["GITHUB_BASE_REF"]))
    return cands


def resolve_base_sha(source_dir, base_ref, feature="HEAD"):
    """Base commit for the impact diff = merge-base(feature, target branch).

    `feature` is the tip we diff FROM -- the PR head, NOT the CI merge commit (diffing the
    merge would drag in develop's advancement); the caller passes the real PR head.

    - Native MIOpen-CI: the target (base_ref / origin/develop) is already a local ref, so
      merge-base resolves directly.
    - TheRock CI: origin/develop is not fetched and the checkout is shallow, but the PR base
      is in the event payload (pull_request.base.sha). Fetch it BLOBLESS (commits + trees,
      no file blobs -- all that merge-base and `git diff --name-only` need), deepening until
      it shares history with the feature.
    - Last resort: a blobless develop / base_ref fetch (self-service; nothing needs to be
      passed down from TheRock).

    Returns None only if nothing resolves (caller fails open to entire_category)."""
    g = ["git", "-C", source_dir]

    def try_mb(ref, why):
        if not _run_git(g + ["rev-parse", "--verify", "-q", f"{ref}^{{commit}}"])[1]:
            return None
        rc, mb, _ = _run_git(g + ["merge-base", feature, ref])
        if rc == 0 and mb:
            print(f"DAPPER base [{why}] {ref} -> merge-base {mb[:12]}")
            return mb
        return None

    # 1. Target branch already present locally (native MIOpen-CI).
    for ref in (base_ref, "origin/develop"):
        mb = try_mb(ref, "local")
        if mb:
            return mb

    # 2. Fetch the base BLOBLESS and retry: CI-authoritative base(s) first, then develop /
    #    base_ref as a last resort; escalate depth until feature and base share history.
    for why, target in _ci_base_candidates() + [
        ("develop", "develop"),
        ("base_ref", base_ref),
    ]:
        for depth in ("--depth=1", "--deepen=5000", "--unshallow"):
            rc, _, err = _run_git(
                g
                + ["fetch", "--no-tags", "--filter=blob:none", depth, "origin", target]
            )
            if rc != 0:
                continue
            mb = try_mb("FETCH_HEAD", f"{why}/{depth}")
            if mb:
                return mb

    print("DAPPER base: unresolved -> entire_category fallback")
    return None


def resolve_feature_sha(source_dir):
    """The tip to diff FROM. In CI we must diff the PR HEAD, not the checkout's HEAD:
    TheRock checks out refs/pull/N/merge (the PR head merged with *current* develop), so
    HEAD contains develop's advancement, not just the PR. The event payload gives the true
    PR head (pull_request.head.sha); fetch it blobless and use it. Falls back to HEAD for a
    normal (non-PR) build such as native MIOpen-CI."""
    g = ["git", "-C", source_dir]
    head = None
    ev = os.environ.get("GITHUB_EVENT_PATH")
    if ev and os.path.isfile(ev):
        try:
            with open(ev) as f:
                pr = json.load(f).get("pull_request") or {}
            head = (pr.get("head") or {}).get("sha")
        except Exception as e:  # noqa: BLE001
            print(f"    feature: could not parse GITHUB_EVENT_PATH ({e})")
    # refs/pull/N/head from GITHUB_REF=refs/pull/N/merge -- a reliably-fetchable ref for the
    # PR head even when the bare sha isn't advertised.
    gref = os.environ.get("GITHUB_REF", "")
    pr_head_ref = (
        gref[: -len("/merge")] + "/head"
        if gref.startswith("refs/pull/") and gref.endswith("/merge")
        else None
    )
    if head:

        def _have():
            return bool(
                _run_git(g + ["rev-parse", "--verify", "-q", f"{head}^{{commit}}"])[1]
            )

        if not _have():
            base_fetch = [
                "fetch",
                "--no-tags",
                "--filter=blob:none",
                "--depth=1",
                "origin",
            ]
            for what in ([head], [pr_head_ref] if pr_head_ref else None):
                if not what:
                    continue
                rc, _, err = _run_git(g + base_fetch + what)
                print(f"    feature: fetch {what} rc={rc} {err[:120]}")
                if _have():
                    break
        if _have():
            print(f"    feature: PR head = {head[:12]}")
            return head
        print("    feature: PR head not usable after fetch; falling back to HEAD")
    return get_git_sha(g + ["rev-parse", "HEAD"])


def write_shas_file(context, shas_file, base_ref="origin/develop", source_dir="."):
    """Write base and feature SHAs.

    source_dir points at the project's git worktree. For an in-source build (CI)
    this is the default '.'; for an out-of-source build (TheRock) the build dir is
    not a git repo, so the caller passes the MIOpen source dir.
    """
    origin = get_git_origin_url(source_dir)
    print(f"{context}: origin={origin} base_ref={base_ref} source_dir={source_dir}")
    feature_sha = resolve_feature_sha(source_dir)
    base_sha = resolve_base_sha(source_dir, base_ref, feature_sha or "HEAD")
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


def _category_filter(cfg, name):
    """Build a gtest filter (positives '-' excludes) for a yaml category, using BOTH
    test_patterns and exclude (plus exclude_linux, since TheRock finalize runs on the linux
    builder). Returns None if the category has no positive patterns."""
    info = (cfg.get("test_categories") or {}).get(name) or {}
    pos = [p for p in (info.get("test_patterns") or []) if p]
    if not pos:
        return None
    exc = [e for e in (info.get("exclude") or []) if e]
    exc += [e for e in (info.get("exclude_linux") or []) if e]
    f = ":".join(pos)
    if exc:
        f += "-" + ":".join(exc)
    return f


def run_finalize_ctest(args):
    """TheRock builder step: burn each Dapper-enabled category's union filter into the
    install CTestTestfile, and retain the full category as a '<name>_unfiltered_suite'.

    For each Dapper-enabled category (yaml 'enable_dapper'), the existing '<name>_suite'
    keeps its name but its --gtest_filter is replaced with the subtractive union (honoring
    fallback_mode); a '<name>_unfiltered_suite' entry is added that keeps the full original
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

    # On TheRock the 'minimal' fallback (nothing test-relevant changed) runs the 'quick'
    # category instead of the near-empty smoke default -- so a no-op change still exercises
    # a small real suite. Built from quick's test_patterns and exclude (finalize runs on the
    # linux builder, so include exclude_linux too).
    quick_filter = _category_filter(cfg, "quick")
    print(f"finalize-ctest: minimal fallback = 'quick' filter: {quick_filter}")

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

    def unfiltered_name(name):
        return name[: -len("_suite")] + "_unfiltered_suite"

    with open(args.ctest_in, "r") as f:
        lines = f.readlines()

    rewritten = {}  # union-suite name -> unfiltered-suite name
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
                    dapper_filter,
                    fallback_mode,
                    cat,
                    original_filter,
                    minimal_filter=quick_filter,
                )
                name_unfiltered = unfiltered_name(name)
                out.append(
                    line.replace(
                        f"--gtest_filter={original_filter}",
                        f"--gtest_filter={union}",
                        1,
                    )
                )
                out.append(
                    line.replace(f"add_test({name} ", f"add_test({name_unfiltered} ", 1)
                )
                rewritten[name] = name_unfiltered
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
            )  # ...and the _unfiltered suite
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
        "and add '<name>_unfiltered_suite' entries retaining the full filters (TheRock).",
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
