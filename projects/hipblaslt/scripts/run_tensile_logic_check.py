#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run the installed TensileLite logic validator without checkout path injection.

How to run (from hipblaslt project root):
  python scripts/run_tensile_logic_check.py [--changed-files=FILE] [LIBLOGIC_PATH]

--changed-files=FILE takes the changed paths, one per line, and checks only the
architectures of the changed logic YAMLs; a changed validator, known-bugs list
or this script, or an empty value, checks everything.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _hipblaslt_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "library").is_dir():
        raise SystemExit(f"Cannot find the hipBLASLt library directory below {root}")
    return root


def _parse_argv(root: Path, args=None):
    """Split script argv into library path (or default), the --changed-files value
    (None if not given), and options to pass to the TensileLite logic command
    (-j, -v, etc.)."""
    default_lib = root / "library"
    args = list(sys.argv[1:] if args is None else args)
    lib_path = None
    changed_files = None
    passthrough = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--changed-files="):
            changed_files = a.split("=", 1)[1]
            i += 1
        elif a == "--changed-files":
            raise SystemExit("Error: use --changed-files=FILE (an empty value checks everything)")
        elif a in ("-j", "--jobs", "-v", "--verbose") or a.startswith("--jobs="):
            passthrough.append(a)
            i += 1
            if i < len(args) and a in ("-j", "--jobs", "-v", "--verbose") and not args[i].startswith("-"):
                passthrough.append(args[i])
                i += 1
        elif a == "--check-all":
            i += 1  # we add it below
        elif a == "--known-bugs":
            passthrough.append(a)
            i += 1
            if i < len(args) and not args[i].startswith("-"):
                passthrough.append(args[i])
                i += 1
            else:
                raise SystemExit("Error: --known-bugs requires a file path")
        elif not a.startswith("-"):
            lib_path = Path(a)
            i += 1
            passthrough.extend(args[i:])
            break
        else:
            passthrough.append(a)
            i += 1
    return (lib_path if lib_path is not None else default_lib, changed_files, passthrough)


def _changed_architectures(root: Path, list_file: Path):
    """Return the architectures to check for the changed paths in list_file, or
    None to check everything.

    Paths are one per line, absolute or relative to the repository root (the
    form `git diff --name-only` prints). A change to the validators, the YAML
    loader, the known-bugs list or this script can affect every architecture,
    so it means everything. Otherwise each changed logic YAML contributes the
    architecture its header declares, and other paths contribute nothing. A
    deleted logic file is resolved through a YAML still in its directory, since a
    directory holds one architecture's logic; if the directory is gone too, the
    answer is everything.
    """
    from tensilelite.CustomYamlLoader import load_logic_gfx_arch

    repo_root = root.parents[1]
    lib_root = (root / "library").resolve()
    validators = [
        (root / "tensilelite" / "tensilelite" / "logic").resolve(),
        (root / "tensilelite" / "tensilelite" / "CustomYamlLoader.py").resolve(),
        Path(__file__).resolve(),
    ]
    archs = set()
    for line in list_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        path = (Path(line) if Path(line).is_absolute() else repo_root / line).resolve()
        if any(path == v or v in path.parents for v in validators):
            return None
        if path.suffix != ".yaml" or lib_root not in path.parents:
            continue
        if not path.is_file():
            siblings = sorted(path.parent.glob("*.yaml")) if path.parent.is_dir() else []
            if not siblings:
                return None
            path = siblings[0]
        arch = load_logic_gfx_arch(path)
        if not isinstance(arch, str):
            return None
        archs.add(arch)
    return archs


def main(argv=None) -> int:
    root = _hipblaslt_root()
    logic_path, changed_files, passthrough = _parse_argv(root, argv)
    if not logic_path.exists():
        raise SystemExit(f"Library logic path not found: {logic_path}")

    if changed_files:
        archs = _changed_architectures(root, Path(changed_files))
        if archs is None:
            print("The changes can affect every architecture; checking everything.")
        elif not archs:
            print("No library logic or validator files changed; nothing to check.")
            return 0
        else:
            print(f"Checking the architectures of the changed logic files: {', '.join(sorted(archs))}")
            passthrough = ["--architecture", ";".join(sorted(archs))] + passthrough

    has_known_bugs_option = any(
        arg in ("--known-bugs", "--use-bundled-known-bugs")
        or arg.startswith("--known-bugs=")
        for arg in passthrough
    )
    if not has_known_bugs_option:
        passthrough = ["--use-bundled-known-bugs"] + passthrough
    if "--check-all" not in passthrough:
        passthrough = [*passthrough, "--check-all"]
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "tensilelite",
            "logic",
            str(logic_path.resolve()),
            *passthrough,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
