#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Combine several Tensile tuning runs into one library logic file, and merge it into the repo.

Each tuning run leaves one file in ``<run dir>/3_LibraryLogic/``. Runs that share a problem
type produce the same file name, so they are partial results of one logic file. They are
combined by chaining TensileLite's own ``merge_library.avoidRegressions``: the first run is
the base and every later run is merged in as an incremental result. The combined file is
written to the current working directory.

``tensilelite/Utilities/merge.py`` is the older list-format merger and raises ``KeyError: 5``
on the dict-format logic these runs produce, so ``tensilelite/merge_library.py`` is used instead.

With ``--update`` the combined file is then merged into the matching logic file already in the
repo. The repo file is chosen by comparing ProblemType rather than by file name, because the
tuning output uses the current naming (``..._Bias_HA_S_SAV_...``) while the files in the repo
were generated under an older convention (``..._BiasSB_HAS_SAV_...``) even though the problem
type is identical.

Usage:
    python3 merge_tuning_logic.py [RUN_DIR ...] [--update] [--update-dir DIR] [--dry-run]
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HIPBLASLT = os.path.abspath(os.path.join(HERE, "..", ".."))
TENSILELITE = os.path.join(HIPBLASLT, "tensilelite")

# ----------------------------------------------------------------------------------------
# Which repo logic directory --update merges into. Edit this line when tuning for a
# different architecture or CU count; --update-dir still overrides it for a one-off run.
# The CU count in the path matters: a logic file only reaches the matching CU-specific
# device library when its own CUCount field agrees, so keep the two in step.
# ----------------------------------------------------------------------------------------
UPDATE_DIR = os.path.join(
    HIPBLASLT,
    "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1250/gfx1250_96cu/Equality")

# Two ProblemTypes count as the same when every key they share agrees. Keys only one side has
# are schema drift between the version that wrote the repo's logic and the version that wrote
# the tuning output (FusedGemmA2A was renamed to FusedA2A; MXBlockFreeA/B and OutputPeerD were
# added later). merge_library.compareProblemType waives one-sided keys the same way, and runs
# again during the merge itself, so this only has to be good enough to pick the right file.
MIN_SHARED_KEYS = 50


def import_merge_library():
    """Import TensileLite's dict-format merger from the tree this script lives in."""
    if TENSILELITE not in sys.path:
        sys.path.insert(0, TENSILELITE)
    try:
        from tensilelite import merge_library
    except ImportError as exc:
        raise SystemExit(f"cannot import merge_library from {TENSILELITE}: {exc}")
    return merge_library


def collect_logic_files(targets):
    """Return the logic yaml of every tuning run named on the command line."""
    files = []
    for target in targets:
        if os.path.isfile(target):
            files.append(os.path.abspath(target))
            continue
        for pattern in ("3_LibraryLogic/*.yaml", "*.yaml"):
            found = sorted(glob.glob(os.path.join(target, pattern)))
            if found:
                files.extend(os.path.abspath(f) for f in found)
                break
        else:
            raise SystemExit(f"no library logic yaml under {target}")
    return files


def problem_type(merge_library, path):
    """Return the top-level ProblemType of a logic file."""
    _, data, _ = merge_library.loadData(path)
    kind = data.get("ProblemType")
    if not isinstance(kind, dict):
        raise SystemExit(f"{path}: no top-level ProblemType")
    return kind


def same_problem_type(a, b):
    """True when every key the two ProblemTypes share agrees. Returns the one-sided keys too."""
    shared = set(a) & set(b)
    if len(shared) < MIN_SHARED_KEYS:
        return False, []
    differing = [k for k in sorted(shared) if a[k] != b[k]]
    return not differing, sorted(set(a) ^ set(b))


def stage(path, name, parent):
    """Copy path into a fresh directory under parent, named name. Return that directory."""
    directory = tempfile.mkdtemp(dir=parent)
    shutil.copyfile(path, os.path.join(directory, name))
    return directory


def combine(merge_library, files, basename, force_merge, parent):
    """Chain avoidRegressions over the runs and return the combined file."""
    base_dir = stage(files[0], basename, parent)
    for extra in files[1:]:
        inc_dir = stage(extra, basename, parent)
        out_dir = tempfile.mkdtemp(dir=parent)
        merge_library.avoidRegressions(base_dir, inc_dir, out_dir, force_merge, True)
        base_dir = out_dir
    return os.path.join(base_dir, basename)


def find_repo_file(merge_library, update_dir, wanted):
    """Pick the logic file in update_dir whose ProblemType matches, ignoring file names."""
    candidates = sorted(glob.glob(os.path.join(update_dir, "*.yaml")))
    if not candidates:
        raise SystemExit(f"no logic yaml in {update_dir}")
    for path in candidates:
        try:
            match, drift = same_problem_type(wanted, problem_type(merge_library, path))
        except SystemExit:
            continue
        if match:
            if drift:
                print(f"    ProblemType matches on every shared key; "
                      f"only one side has: {', '.join(drift)}")
            return path
    print(f"    no ProblemType match among the {len(candidates)} file(s) in {update_dir}; "
          f"this logic is new to that directory")
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*",
                        help="tuning run directories (or logic yamls). "
                             "Default: every */3_LibraryLogic/ next to this script")
    parser.add_argument("--update", action="store_true",
                        help="also merge the result into the repo's logic file")
    parser.add_argument("--update-dir", default=UPDATE_DIR,
                        help=f"logic directory to merge into, overriding the UPDATE_DIR set at "
                             f"the top of this script (default: "
                             f"{os.path.relpath(UPDATE_DIR, HIPBLASLT)})")
    parser.add_argument("--update-file", default=None,
                        help="exact repo logic file to merge into, instead of matching ProblemType")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --update, report what would change without writing")
    parser.add_argument("--force-merge", default="none", choices=["none", "true", "false"],
                        help="merge previously known sizes unconditionally (merge_library default: true)")
    args = parser.parse_args(argv)

    merge_library = import_merge_library()
    force_merge = args.force_merge != "false"

    runs = args.runs or sorted(
        d for d in glob.glob(os.path.join(HERE, "*"))
        if os.path.isdir(os.path.join(d, "3_LibraryLogic")))
    if not runs:
        parser.error("no tuning runs given and none found next to this script")

    files = collect_logic_files(runs)
    names = {os.path.basename(f) for f in files}
    if len(names) != 1:
        parser.error("the runs produced different logic files, so they are not partial results "
                     "of one file; merge them separately: " + ", ".join(sorted(names)))
    basename = names.pop()

    print(f"combining {len(files)} tuning run(s) into {basename}")
    for path in files:
        print(f"    {path}")

    with tempfile.TemporaryDirectory() as parent:
        combined_tmp = combine(merge_library, files, basename, force_merge, parent)
        combined = os.path.abspath(os.path.join(os.getcwd(), basename))
        shutil.copyfile(combined_tmp, combined)
        print(f"\nwrote {combined}")

        if not args.update:
            print("(--update not given, repo logic left alone)")
            return 0

        target = args.update_file or find_repo_file(
            merge_library, args.update_dir, problem_type(merge_library, combined))

        if target is None:
            # Nothing in the directory covers this problem type, so there is nothing to merge
            # against: the file is simply added. The library keys off the ProblemType inside,
            # not the file name, so the tuning output's own name is fine.
            added = os.path.join(args.update_dir, basename)
            if os.path.exists(added):
                raise SystemExit(f"{added} already exists but its ProblemType did not match; "
                                 f"merge it explicitly with --update-file")
            if args.dry_run:
                print(f"--dry-run: would add {added} ({os.path.getsize(combined)} bytes)")
                return 0
            shutil.copyfile(combined, added)
            print(f"added {added}")
            return 0

        print(f"\nmerging into {target}")

        # avoidRegressions pairs files by name, so stage the result under the repo's name.
        staging = stage(combined, os.path.basename(target), parent)
        result = tempfile.mkdtemp(dir=parent)
        merge_library.avoidRegressions(os.path.dirname(target), staging, result,
                                       force_merge, True)
        merged = os.path.join(result, os.path.basename(target))
        if not os.path.exists(merged):
            raise SystemExit(f"merge produced no {os.path.basename(target)} in {result}")

        if args.dry_run:
            print(f"--dry-run: would overwrite {target} "
                  f"({os.path.getsize(target)} -> {os.path.getsize(merged)} bytes)")
            return 0

        backup = target + ".bak"
        shutil.copyfile(target, backup)
        shutil.copyfile(merged, target)
        print(f"updated {target}\nbackup  {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
