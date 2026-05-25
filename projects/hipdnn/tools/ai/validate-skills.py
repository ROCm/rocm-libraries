#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate committed hipDNN AI skill packaging."""

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"

REQUIRED_OPENAI_FIELDS = (
    "display_name",
    "short_description",
    "default_prompt",
)

FORBIDDEN_SKILL_TEXT = (
    "$ARGUMENTS",
    "$HELPERS",
    "HELPERS =",
    "skills/helpers",
)

FORBIDDEN_SKILL_PATTERNS = (re.compile(r"\bAskUserQuestion\b"),)

SLASH_SKILL_PATTERN = re.compile(r"(?<![\w:/])/(?:hipdnn|pr-summary)[A-Za-z0-9_-]*")

EXPECTED_SCRIPTS = {
    "hipdnn-superbuild": ("windows_rocm_setup.py",),
    "hipdnn-superbuild-test": (
        "cmake_run.py",
        "discover_test_targets.py",
        "windows_rocm_setup.py",
    ),
}

REQUIRED_CLAUDE_COMMAND_FIELDS = ("argument-hint", "allowed-tools")

DUPLICATE_SCRIPT_PAIRS = (
    (
        SKILLS_DIR / "hipdnn-superbuild" / "scripts" / "windows_rocm_setup.py",
        SKILLS_DIR / "hipdnn-superbuild-test" / "scripts" / "windows_rocm_setup.py",
    ),
)


def skill_dirs() -> list[Path]:
    return sorted(
        path
        for path in SKILLS_DIR.iterdir()
        if path.is_dir() and (path / "SKILL.md").exists()
    )


def parse_openai_yaml(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    in_interface = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line == "interface:":
            in_interface = True
            continue
        if not in_interface or not line.startswith("  "):
            continue
        key, separator, value = line.strip().partition(":")
        if separator:
            fields[key] = value.strip().strip('"')
    return fields


def is_ignored(path: Path) -> bool:
    relative = path.relative_to(ROOT.parent.parent.parent.parent)
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(relative)],
        cwd=ROOT.parent.parent.parent.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def validate_skill(skill: Path) -> list[str]:
    errors: list[str] = []
    skill_md = skill / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")

    for token in FORBIDDEN_SKILL_TEXT:
        if token in text:
            errors.append(f"{skill_md}: contains stale host-specific text: {token}")

    for pattern in FORBIDDEN_SKILL_PATTERNS:
        if pattern.search(text):
            errors.append(
                f"{skill_md}: contains stale host-specific text: {pattern.pattern}"
            )

    match = SLASH_SKILL_PATTERN.search(text)
    if match:
        errors.append(f"{skill_md}: contains slash-command reference: {match.group(0)}")

    if skill.name == "hipdnn-superbuild":
        if "cmake --preset <preset> -B <build-dir>" not in text:
            errors.append(
                f"{skill_md}: configure command must bind the selected <build-dir>"
            )
        if "same `-B <build-dir>` command" not in text:
            errors.append(
                f"{skill_md}: stale-cache retry must reuse the selected <build-dir>"
            )

    openai_yaml = skill / "agents" / "openai.yaml"
    if not openai_yaml.exists():
        errors.append(f"{skill}: missing agents/openai.yaml")
    else:
        fields = parse_openai_yaml(openai_yaml)
        for field in REQUIRED_OPENAI_FIELDS:
            if not fields.get(field):
                errors.append(f"{openai_yaml}: missing interface.{field}")

    for script in EXPECTED_SCRIPTS.get(skill.name, ()):
        script_path = skill / "scripts" / script
        if not script_path.exists():
            errors.append(f"{skill}: missing scripts/{script}")

    # Skills with Claude commands must include argument-hint and allowed-tools in SKILL.md
    claude_commands = {
        "hipdnn-review",
        "hipdnn-superbuild",
        "hipdnn-superbuild-test",
        "pr-summary",
    }
    if skill.name in claude_commands:
        for field in REQUIRED_CLAUDE_COMMAND_FIELDS:
            if f"{field}:" not in text:
                errors.append(f"{skill_md}: missing {field} in frontmatter")

    return errors


def main() -> int:
    if not SKILLS_DIR.exists():
        print(f"ERROR: skills directory not found: {SKILLS_DIR}", file=sys.stderr)
        return 1

    skills = skill_dirs()
    if not skills:
        print(f"ERROR: no skills found in {SKILLS_DIR}", file=sys.stderr)
        return 1

    errors: list[str] = []
    for skill in skills:
        errors.extend(validate_skill(skill))

    for left, right in DUPLICATE_SCRIPT_PAIRS:
        # Skip validation if either file is a symlink (symlinks auto-stay in sync)
        if left.is_symlink() or right.is_symlink():
            continue
        if left.exists() and right.exists() and left.read_bytes() != right.read_bytes():
            errors.append(f"{left} and {right} must stay byte-identical")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Validated {len(skills)} skill(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
