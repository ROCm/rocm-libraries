#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Create symlinks (Linux/macOS) or directory junctions (Windows) from a target
directory to specific skill folders in the sibling ``skills/`` directory.

The user must explicitly name every skill they want installed. There is no
"link everything" mode — this is by design so a skill is never added without
the user's consent.

Usage:
    python link-skills.py <target-directory> <skill-name> [<skill-name> ...]

    Pass ``--list`` (in place of skill names) to print the skills available
    in the sibling ``skills/`` directory and exit.

Examples:
    # See what's available
    python link-skills.py ~/.claude/skills --list

    # Link two specific skills into the user-global Claude scope
    python link-skills.py ~/.claude/skills hipdnn-superbuild hipdnn-superbuild-test

    # Link a single skill into a workspace scope
    python link-skills.py /path/to/workspace/.claude/skills pr-summary
"""

import sys
import platform
import subprocess
from pathlib import Path


def is_skill_dir(path: Path) -> bool:
    return path.is_dir() and (path / "SKILL.md").exists()


def resolve_link_target(link: Path) -> Path | None:
    try:
        return link.resolve()
    except OSError:
        return None


def create_junction_windows(source: Path, target: Path) -> None:
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(target), str(source)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise OSError(
            f"mklink /J failed: {result.stderr.strip() or result.stdout.strip()}"
        )


def create_link(source: Path, target: Path) -> str:
    if target.exists() or target.is_symlink():
        existing_target = resolve_link_target(target)
        if existing_target == source.resolve():
            return "skipped (already linked)"
        return f"skipped (exists, points to {existing_target})"

    if platform.system() == "Windows":
        create_junction_windows(source, target)
    else:
        target.symlink_to(source, target_is_directory=True)

    return "created"


def print_usage(program: str) -> None:
    print(f"Usage: {program} <target-directory> <skill-name> [<skill-name> ...]")
    print(f"       {program} <target-directory> --list")
    print()
    print("Creates symlinks/junctions for the named skills only.")
    print("Use --list to see available skills.")


def main() -> int:
    if len(sys.argv) < 3:
        print_usage(sys.argv[0])
        return 1

    target_dir = Path(sys.argv[1]).expanduser().resolve()
    requested = sys.argv[2:]
    skills_dir = (Path(__file__).parent / "skills").resolve()

    available = {p.name: p for p in sorted(skills_dir.iterdir()) if is_skill_dir(p)}

    if not available:
        print(f"No skill directories found in {skills_dir}")
        return 1

    if requested == ["--list"]:
        print(f"Available skills in {skills_dir}:")
        for name in available:
            print(f"  {name}")
        return 0

    unknown = [name for name in requested if name not in available]
    if unknown:
        print("Unknown skill name(s): " + ", ".join(unknown))
        print()
        print("Available skills:")
        for name in available:
            print(f"  {name}")
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source:  {skills_dir}")
    print(f"Target:  {target_dir}")
    print(f"Method:  {'junction' if platform.system() == 'Windows' else 'symlink'}")
    print()

    errors = 0
    for name in requested:
        skill = available[name]
        link_path = target_dir / skill.name
        try:
            status = create_link(skill, link_path)
            print(f"  {skill.name:30s} {status}")
        except OSError as e:
            print(f"  {skill.name:30s} FAILED: {e}")
            errors += 1

    print()
    if errors:
        print(f"Done with {errors} error(s).")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
