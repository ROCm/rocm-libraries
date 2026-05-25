#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Install hipDNN AI skills into Codex, Claude, or an explicit target directory.

This script copies skills as snapshots and automatically detects when installed
skills differ from the source. If a skill has changed, it is reinstalled.

Examples:
    # See available skills
    python3 link-skills.py --list

    # Copy skills into Codex global scope
    python3 link-skills.py --codex hipdnn-superbuild hipdnn-superbuild-test

    # Copy skills into Claude global scope
    python3 link-skills.py --claude hipdnn-review pr-summary

    # Copy skills into an explicit target directory
    python3 link-skills.py --target /path/to/skills hipdnn-superbuild

    # Backward-compatible positional target form
    python3 link-skills.py /path/to/skills hipdnn-review
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path


IGNORED_COPY_NAMES = {"__pycache__", ".pytest_cache"}


def is_skill_dir(path: Path) -> bool:
    return path.is_dir() and (path / "SKILL.md").exists()


def available_skills(skills_dir: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(skills_dir.iterdir()) if is_skill_dir(p)}


def copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORED_COPY_NAMES}


def calculate_skill_sha(source: Path) -> str:
    """Calculate SHA256 of all files in a skill directory for change detection."""
    hasher = hashlib.sha256()
    for fpath in sorted(source.rglob("*")):
        if fpath.is_file():
            try:
                with open(fpath, "rb") as f:
                    hasher.update(f.read())
            except (OSError, IOError):
                pass
    return hasher.hexdigest()


def install_or_update_copy(source: Path, target: Path) -> str:
    """Install skill or update if source has changed."""
    source_sha = calculate_skill_sha(source)
    
    if not target.exists():
        # First install
        shutil.copytree(source, target, ignore=copy_ignore)
        return "installed"
    
    if target.is_symlink():
        return f"skipped (symlink exists at {target})"
    
    # Check if installed version differs from source
    target_sha = calculate_skill_sha(target)
    if source_sha == target_sha:
        return "up to date"
    
    # Source changed, reinstall
    shutil.rmtree(target)
    shutil.copytree(source, target, ignore=copy_ignore)
    return "updated"


def codex_target() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser().resolve() / "skills"
    return Path.home().resolve() / ".codex" / "skills"


def claude_target() -> Path:
    return Path.home().resolve() / ".claude" / "skills"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install named hipDNN skills as snapshots. Changes in the source are "
            "detected and installed automatically."
        )
    )
    host = parser.add_mutually_exclusive_group()
    host.add_argument("--codex", action="store_true", help="Install into Codex skills")
    host.add_argument(
        "--claude", action="store_true", help="Install into Claude skills"
    )
    parser.add_argument("--target", help="Explicit skills target directory")
    parser.add_argument("--list", action="store_true", help="List available skills")
    parser.add_argument(
        "items",
        nargs="*",
        help=(
            "Skill names. In backward-compatible mode, the first item is the "
            "target directory and remaining items are skill names."
        ),
    )
    return parser.parse_args(argv)


def resolve_target_and_requested(args: argparse.Namespace) -> tuple[Path | None, list[str]]:
    host_target_count = sum([bool(args.codex), bool(args.claude), bool(args.target)])
    if host_target_count > 1:
        raise ValueError("Choose only one of --codex, --claude, or --target.")

    if args.codex:
        return codex_target(), args.items
    if args.claude:
        return claude_target(), args.items
    if args.target:
        return Path(args.target).expanduser().resolve(), args.items

    if args.list and not args.items:
        return None, []

    if not args.items:
        raise ValueError(
            "Missing target directory. Use --codex, --claude, --target, or "
            "the positional target-directory form."
        )

    return Path(args.items[0]).expanduser().resolve(), args.items[1:]


def print_available(skills_dir: Path, skills: dict[str, Path]) -> None:
    print(f"Available skills in {skills_dir}:")
    for name in skills:
        print(f"  {name}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    skills_dir = (Path(__file__).parent / "skills").resolve()
    skills = available_skills(skills_dir)

    if not skills:
        print(f"No skill directories found in {skills_dir}", file=sys.stderr)
        return 1

    try:
        target_dir, requested = resolve_target_and_requested(args)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if args.list:
        print_available(skills_dir, skills)
        return 0

    unknown = [name for name in requested if name not in skills]
    if unknown:
        print("ERROR: unknown skill name(s): " + ", ".join(unknown), file=sys.stderr)
        print(file=sys.stderr)
        print_available(skills_dir, skills)
        return 1

    if not requested:
        print("ERROR: no skills requested.", file=sys.stderr)
        return 2

    if target_dir is None:
        print("ERROR: no target directory resolved.", file=sys.stderr)
        return 2

    target_dir.mkdir(parents=True, exist_ok=True)

    # Install mode: install or update based on SHA
    print(f"Source:  {skills_dir}")
    print(f"Target:  {target_dir}")
    print()

    errors = 0
    for name in requested:
        skill = skills[name]
        target = target_dir / skill.name
        try:
            status = install_or_update_copy(skill, target)
            print(f"  {skill.name:30s} {status}")
        except OSError as error:
            print(f"  {skill.name:30s} FAILED: {error}")
            errors += 1

    print()
    if errors:
        print(f"Done with {errors} error(s).")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
