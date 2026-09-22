#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate committed hipDNN AI skill packaging."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"
REPO_ROOT = ROOT.parent.parent.parent.parent

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

# The lookbehind excludes '.' so a relative path into a sibling skill directory
# ("../hipdnn-ingestor-engine/RUNBOOK.md") is not read as a slash command.
SLASH_SKILL_PATTERN = re.compile(r"(?<![\w:/.])/(?:hipdnn|pr-summary)[A-Za-z0-9_-]*")

EXPECTED_SCRIPTS = {
    "hipdnn-superbuild": ("windows_rocm_setup.py",),
    "hipdnn-superbuild-test": (
        "cmake_run.py",
        "discover_test_targets.py",
        "windows_rocm_setup.py",
    ),
}

REQUIRED_CLAUDE_COMMAND_FIELDS = ("argument-hint", "allowed-tools")

# [text](target), with optional angle brackets and an optional quoted title.
MARKDOWN_LINK_PATTERN = re.compile(
    r"\[[^\]]*\]\(\s*<?([^)>\s]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)"
)

# Link targets that are not in-repo relative paths: any URL scheme (which also
# covers a Windows drive letter), a protocol-relative URL, a site-absolute path
# and a pure in-page fragment.
NON_RELATIVE_LINK_PATTERN = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//|/|#)")

BACKTICK_PATTERN = re.compile(r"`([^`\n]+)`")

# Extensions that make a backticked token unambiguously a file reference rather
# than a directory, command or symbol.
PROSE_PATH_EXTENSIONS = frozenset(
    {".ps1", ".py", ".cmake", ".json", ".yaml", ".yml", ".md", ".j2"}
)

# Any of these makes a token a template, glob or shell expression, not a path.
PATH_PLACEHOLDER_CHARS = "<>${}*|\"' \t"

# A "<name>" placeholder. The lookbehind keeps C++ generics ("vector<int>") and
# quoted strings out.
PLACEHOLDER_PATTERN = re.compile(r"(?<![A-Za-z0-9_\"'])<([A-Za-z][A-Za-z0-9_-]*)>")

# Fence languages whose contents are commands. Untagged fences count; fences
# tagged for a real language do not, so C++ generics never reach the
# placeholder check.
COMMAND_FENCE_LANGUAGES = frozenset(
    {
        "",
        "bash",
        "sh",
        "shell",
        "console",
        "zsh",
        "powershell",
        "ps1",
        "pwsh",
        "cmd",
        "bat",
        "text",
    }
)

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


def symlink_target(path: Path) -> Path | None:
    """Return what `path` links to, or None when it is ordinary content.

    A checkout without symlink support (Windows without the privilege, or
    core.symlinks=false) stores git's mode-120000 entry as a regular file whose
    whole content is the link target, so both forms are recognised and the
    byte-identity check does not fire on a pair that is a symlink upstream.
    Ordinary script content never round-trips as a single-line path resolving to
    an existing file.
    """
    if path.is_symlink():
        return (path.parent / path.readlink()).resolve()
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    candidate = content.strip()
    if not candidate or "\n" in candidate or len(candidate) > 260:
        return None
    resolved = (path.parent / candidate).resolve()
    return resolved if resolved.is_file() else None


def split_markdown(text: str) -> tuple[str, str]:
    """Split a markdown document into its prose half and its command half.

    Prose is everything outside a fenced code block; link, path and
    placeholder-definition checks run there so an illustrative snippet is never
    mistaken for a real reference. Commands are the contents of untagged or
    shell-tagged fences, where placeholder *use* is collected.
    """
    prose: list[str] = []
    commands: list[str] = []
    fence = ""
    language = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence:
            if stripped.startswith(fence):
                fence = ""
                language = ""
            elif language in COMMAND_FENCE_LANGUAGES:
                commands.append(line)
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            info = stripped[3:].strip()
            language = info.split()[0].lower() if info else ""
            continue
        prose.append(line)
    return "\n".join(prose), "\n".join(commands)


def markdown_files(skill: Path) -> list[Path]:
    return sorted(skill.rglob("*.md"))


def relative_within(path: Path, base: Path) -> Path | None:
    try:
        return path.relative_to(base)
    except ValueError:
        return None


def check_link_targets(skill: Path) -> list[str]:
    """Markdown links must still resolve from an *installed* skill root.

    install-skills.py copies exactly one skill directory and nothing above it,
    so a link reaching outside `skills/<this-skill>/` resolves in the checkout
    and dangles once installed; targets are resolved against the installed
    layout instead. The one permitted escape is a sibling skill directory under
    `skills/`, which a default install lays down side by side.
    """
    errors: list[str] = []
    skill_root = skill.resolve()
    for markdown in markdown_files(skill):
        prose, _ = split_markdown(markdown.read_text(encoding="utf-8"))
        for raw_target in MARKDOWN_LINK_PATTERN.findall(prose):
            target = raw_target.strip()
            if not target or NON_RELATIVE_LINK_PATTERN.match(target):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            resolved = (markdown.parent / path_part).resolve()
            sibling = relative_within(resolved, skill_root.parent)
            reachable = relative_within(resolved, skill_root) is not None or (
                sibling is not None and len(sibling.parts) >= 2
            )
            if reachable:
                # Surviving the install copy is not the same as resolving.
                if not resolved.exists():
                    errors.append(
                        f"{markdown}: link target '{target}' stays inside the "
                        f"installed skill root but no such file exists"
                    )
                continue
            errors.append(
                f"{markdown}: link target '{target}' escapes the installed skill "
                f"root '{skill_root.name}/'; only that one directory is copied on "
                "install, so the link resolves in the checkout but dangles once "
                "installed"
            )
    return errors


def check_prose_paths(skill: Path) -> list[str]:
    """Backticked repo-relative file paths named in prose must exist.

    The guards are deliberately narrow, because false positives would make the
    check worthless. The decisive one is the last: a token is a claim about a
    location only when its FIRST component names a real directory at the
    repository root or inside the skill. An ambiguous token is left alone.
    """
    errors: list[str] = []
    for markdown in markdown_files(skill):
        prose, _ = split_markdown(markdown.read_text(encoding="utf-8"))
        for raw_token in BACKTICK_PATTERN.findall(prose):
            token = raw_token.strip()
            if any(character in token for character in PATH_PLACEHOLDER_CHARS):
                continue
            if "://" in token or token.startswith(("/", "~", ".", "-")):
                continue
            parts = token.split("/")
            if len(parts) < 2 or ".." in parts or "" in parts:
                continue
            if Path(token).suffix not in PROSE_PATH_EXTENSIONS:
                continue
            if (markdown.parent / token).exists() or (skill / token).exists():
                continue
            if (REPO_ROOT / token).exists():
                continue
            anchored_at_repo = (REPO_ROOT / parts[0]).is_dir()
            anchored_in_skill = (skill / parts[0]).is_dir()
            if not (anchored_at_repo or anchored_in_skill):
                continue
            anchor = "the repository root" if anchored_at_repo else f"{skill.name}/"
            errors.append(
                f"{markdown}: references '{token}', which does not exist; "
                f"'{parts[0]}/' resolves against {anchor} but the rest of the "
                "path does not"
            )
    return errors


def check_placeholders(skill: Path) -> list[str]:
    """A '<name>' placeholder used in a command must be introduced in prose.

    Collection is restricted to command-shaped fences. The "defined" test is
    deliberately permissive: the bare token appearing anywhere outside a fence
    in the same file counts, with or without angle brackets, and frontmatter
    counts as prose, so an `argument-hint` introduction is enough.
    """
    errors: list[str] = []
    for markdown in markdown_files(skill):
        prose, commands = split_markdown(markdown.read_text(encoding="utf-8"))
        used = {match.group(1) for match in PLACEHOLDER_PATTERN.finditer(commands)}
        for token in sorted(used):
            defined = re.search(
                rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])", prose
            )
            if defined:
                continue
            errors.append(
                f"{markdown}: command blocks use placeholder '<{token}>' but the "
                "file never introduces it outside a code fence"
            )
    return errors


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
        "hipdnn-ingestor-engine",
        "hipdnn-pr-quality",
        "hipdnn-superbuild",
        "hipdnn-superbuild-test",
        "rfc-backlog",
        "rfc-review",
        "rfc-review-compatibility",
        "rfc-review-ops",
        "rfc-review-security",
    }
    if skill.name in claude_commands:
        for field in REQUIRED_CLAUDE_COMMAND_FIELDS:
            if f"{field}:" not in text:
                errors.append(f"{skill_md}: missing {field} in frontmatter")

    errors.extend(check_link_targets(skill))
    errors.extend(check_prose_paths(skill))
    errors.extend(check_placeholders(skill))

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
        # Skip validation if either file links to the other (links auto-stay in sync)
        if symlink_target(left) == right.resolve():
            continue
        if symlink_target(right) == left.resolve():
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
