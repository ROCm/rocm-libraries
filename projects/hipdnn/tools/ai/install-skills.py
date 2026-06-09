#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Install hipDNN AI skills into Codex, Claude, or an explicit target directory.

This script copies skills as snapshots and automatically detects when installed
skills differ from the source. If a skill has changed, it is reinstalled.

By default, all available skills are installed to both ~/.codex/skills and
~/.claude/skills. Specify individual skill names to install only those skills.

Examples:
    # See available skills
    python3 install-skills.py --list

    # Install all skills to both Codex and Claude (default)
    python3 install-skills.py

    # Install specific skills to both Codex and Claude
    python3 install-skills.py hipdnn-superbuild hipdnn-superbuild-test

    # Install all skills to Codex only
    python3 install-skills.py --codex

    # Install specific skills to Claude only
    python3 install-skills.py --claude hipdnn-review pr-summary

    # Install all skills to both Codex and Claude (explicit)
    python3 install-skills.py --codex --claude

    # Install to an explicit target directory
    python3 install-skills.py --target /path/to/skills

    # Backward-compatible positional target form
    python3 install-skills.py /path/to/skills hipdnn-review
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
            "Install hipDNN skills as snapshots. By default installs all available skills to both "
            "~/.codex/skills and ~/.claude/skills. Changes in the source are "
            "detected and installed automatically."
        )
    )
    parser.add_argument(
        "--codex", action="store_true", help="Install into Codex skills"
    )
    parser.add_argument(
        "--claude", action="store_true", help="Install into Claude skills"
    )
    parser.add_argument("--target", help="Explicit skills target directory")
    parser.add_argument("--list", action="store_true", help="List available skills")
    parser.add_argument(
        "items",
        nargs="*",
        help=(
            "Skill names (optional). If not specified, all available skills are installed. "
            "In backward-compatible mode, the first item is the target directory and "
            "remaining items are skill names."
        ),
    )
    return parser.parse_args(argv)


def resolve_targets_and_requested(
    args: argparse.Namespace, available: dict[str, Path]
) -> tuple[list[Path], list[str]]:
    """Resolve target directories and skill names.

    Args:
        args: Parsed arguments
        available: Dict of available skills {name: path}

    Returns:
        (list of target directories, list of requested skill names)
    """
    targets = []

    # Explicit targets take precedence
    if args.target:
        targets.append(Path(args.target).expanduser().resolve())
        # If no skills specified, use all available
        requested = args.items if args.items else list(available.keys())
        return targets, requested

    if args.codex or args.claude:
        if args.codex:
            targets.append(codex_target())
        if args.claude:
            targets.append(claude_target())
        # If no skills specified, use all available
        requested = args.items if args.items else list(available.keys())
        return targets, requested

    # Default: both Codex and Claude if no specific host is requested
    if args.list and not args.items:
        return [], []

    # Check if first item looks like a directory (backward-compatible positional form)
    if args.items:
        first_item_path = Path(args.items[0]).expanduser()
        if not any(c in args.items[0] for c in ["@", ":"]) and (
            "/" in args.items[0] or "\\" in args.items[0] or first_item_path.exists()
        ):
            # Looks like a directory path
            targets.append(first_item_path.resolve())
            return targets, args.items[1:]

    # Default to both Codex and Claude, and all available skills if none specified
    requested = args.items if args.items else list(available.keys())
    return [codex_target(), claude_target()], requested


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
        targets, requested = resolve_targets_and_requested(args, skills)
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

    if not targets:
        print("ERROR: no target directories resolved.", file=sys.stderr)
        return 2

    # Create all target directories
    for target_dir in targets:
        target_dir.mkdir(parents=True, exist_ok=True)

    # Install to all targets
    print(f"Source:  {skills_dir}")
    if len(targets) == 1:
        print(f"Target:  {targets[0]}")
    else:
        print(f"Targets: {', '.join(str(t) for t in targets)}")
    print()

    total_errors = 0
    for target_dir in targets:
        if len(targets) > 1:
            print(f"Installing to {target_dir}:")

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

        total_errors += errors
        if len(targets) > 1:
            print()

    if total_errors:
        print(f"Done with {total_errors} error(s).")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
