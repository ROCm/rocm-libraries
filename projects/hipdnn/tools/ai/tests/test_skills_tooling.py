# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the hipDNN AI skill tooling (install-skills, validate-skills).

These scripts have hyphenated filenames, so they are loaded by path rather than
imported by module name.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_AI_DIR = Path(__file__).resolve().parent.parent
SKILLS_DIR = TOOLS_AI_DIR / "skills"


def _load(script_name: str, module_name: str) -> ModuleType:
    path = TOOLS_AI_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader, f"could not load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validate_mod() -> ModuleType:
    return _load("validate-skills.py", "hipdnn_validate_skills")


@pytest.fixture(scope="module")
def install_mod() -> ModuleType:
    return _load("install-skills.py", "hipdnn_install_skills")


def _write_skill(
    skill_dir: Path,
    *,
    frontmatter_extra: str = "",
    body: str = "Example skill body.\n",
    with_openai: bool = True,
) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = "---\nname: demo-skill\n" + frontmatter_extra + "---\n\n"
    (skill_dir / "SKILL.md").write_text(frontmatter + body, encoding="utf-8")
    if with_openai:
        agents = skill_dir / "agents"
        agents.mkdir(parents=True, exist_ok=True)
        (agents / "openai.yaml").write_text(
            "interface:\n"
            '  display_name: "Demo"\n'
            '  short_description: "Demo skill"\n'
            '  default_prompt: "Use $demo-skill to do a demo."\n',
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# validate-skills.py
# --------------------------------------------------------------------------- #


def test_committed_skills_validate_clean():
    """The real committed skill set must pass its own validator."""
    result = subprocess.run(
        [sys.executable, str(TOOLS_AI_DIR / "validate-skills.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Validated" in result.stdout


def test_validate_skill_accepts_minimal_valid_skill(validate_mod, tmp_path):
    skill = tmp_path / "demo-skill"
    _write_skill(skill)
    assert validate_mod.validate_skill(skill) == []


def test_validate_skill_flags_missing_openai_yaml(validate_mod, tmp_path):
    skill = tmp_path / "demo-skill"
    _write_skill(skill, with_openai=False)
    errors = validate_mod.validate_skill(skill)
    assert any("openai.yaml" in error for error in errors)


def test_validate_skill_flags_forbidden_text(validate_mod, tmp_path):
    skill = tmp_path / "demo-skill"
    _write_skill(skill, body="This references skills/helpers which is forbidden.\n")
    errors = validate_mod.validate_skill(skill)
    assert any("stale host-specific text" in error for error in errors)


def test_validate_skill_flags_slash_command_reference(validate_mod, tmp_path):
    skill = tmp_path / "demo-skill"
    _write_skill(skill, body="Invoke it with /hipdnn-demo in chat.\n")
    errors = validate_mod.validate_skill(skill)
    assert any("slash-command reference" in error for error in errors)


def test_validate_skill_requires_claude_frontmatter_fields(validate_mod, tmp_path):
    """A skill named in the claude_commands set needs argument-hint + allowed-tools."""
    skill = tmp_path / "hipdnn-pr-quality"
    _write_skill(skill)
    errors = validate_mod.validate_skill(skill)
    assert any("argument-hint" in error for error in errors)
    assert any("allowed-tools" in error for error in errors)


@pytest.mark.parametrize("name", ["pr-summary", "hipdnn-review"])
def test_deprecated_stub_announces_deprecation_and_redirects(name):
    """The retired skills must survive as stubs that flag themselves deprecated."""
    text = (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
    assert "deprecated" in text.lower()
    assert "hipdnn-pr-quality" in text


# --------------------------------------------------------------------------- #
# install-skills.py
# --------------------------------------------------------------------------- #


def test_available_skills_discovers_only_skill_dirs(install_mod, tmp_path):
    _write_skill(tmp_path / "alpha")
    _write_skill(tmp_path / "beta")
    (tmp_path / "not-a-skill").mkdir()  # no SKILL.md
    found = install_mod.available_skills(tmp_path)
    assert set(found) == {"alpha", "beta"}


def test_calculate_skill_sha_changes_with_content(install_mod, tmp_path):
    skill = tmp_path / "alpha"
    _write_skill(skill)
    before = install_mod.calculate_skill_sha(skill)
    (skill / "SKILL.md").write_text("changed content\n", encoding="utf-8")
    after = install_mod.calculate_skill_sha(skill)
    assert before != after


def test_install_or_update_copy_lifecycle(install_mod, tmp_path):
    source = tmp_path / "src" / "alpha"
    _write_skill(source)
    target = tmp_path / "dst" / "alpha"

    assert install_mod.install_or_update_copy(source, target) == "installed"
    assert (target / "SKILL.md").exists()
    assert install_mod.install_or_update_copy(source, target) == "up to date"

    (source / "SKILL.md").write_text("new body\n", encoding="utf-8")
    assert install_mod.install_or_update_copy(source, target) == "updated"
    assert (target / "SKILL.md").read_text(encoding="utf-8") == "new body\n"


def test_resolve_targets_explicit_target_dir(install_mod, tmp_path):
    _write_skill(tmp_path / "alpha")
    available = install_mod.available_skills(tmp_path)
    args = install_mod.parse_args(["--target", str(tmp_path / "out"), "alpha"])
    targets, requested = install_mod.resolve_targets_and_requested(args, available)
    assert targets == [(tmp_path / "out").resolve()]
    assert requested == ["alpha"]


def test_resolve_targets_explicit_target_defaults_to_all_skills(install_mod, tmp_path):
    _write_skill(tmp_path / "alpha")
    _write_skill(tmp_path / "beta")
    available = install_mod.available_skills(tmp_path)
    args = install_mod.parse_args(["--target", str(tmp_path / "out")])
    targets, requested = install_mod.resolve_targets_and_requested(args, available)
    assert targets == [(tmp_path / "out").resolve()]
    assert sorted(requested) == ["alpha", "beta"]


def test_resolve_targets_defaults_to_both_hosts(install_mod, tmp_path):
    _write_skill(tmp_path / "alpha")
    available = install_mod.available_skills(tmp_path)
    args = install_mod.parse_args([])
    targets, requested = install_mod.resolve_targets_and_requested(args, available)
    assert targets == [install_mod.codex_target(), install_mod.claude_target()]
    assert requested == ["alpha"]
