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


def test_validate_skill_allows_relative_link_to_a_sibling_skill(validate_mod, tmp_path):
    """A markdown link into a sibling skill directory is a path, not a command.

    Paired with the test above, which shares the '/hipdnn' text and must still be
    rejected, so this passing cannot mean the check stopped firing.
    """
    skill = tmp_path / "demo-skill"
    _write_skill(
        skill, body="See [the runbook](../hipdnn-demo/RUNBOOK.md) for steps.\n"
    )
    errors = validate_mod.validate_skill(skill)
    assert not any("slash-command reference" in error for error in errors)


def test_validate_skill_flags_link_escaping_the_installed_skill_root(
    validate_mod, tmp_path
):
    """A link above skills/<name>/ resolves in the checkout and dies on install:
    install-skills.py copies one skill directory and nothing above it."""
    skill = tmp_path / "demo-skill"
    _write_skill(
        skill, body="See [the generator](../../../IngestorGenerator/README.md).\n"
    )
    errors = validate_mod.validate_skill(skill)
    assert any("escapes the installed skill root" in error for error in errors)


def test_validate_skill_allows_link_into_a_sibling_skill_directory(
    validate_mod, tmp_path
):
    """Sibling skills install side by side, so ../<sibling>/ still resolves.
    Paired with the escape test above, which must still fail."""
    skill = tmp_path / "demo-skill"
    _write_skill(skill, body="See [the runbook](../other-skill/RUNBOOK.md#setup).\n")
    errors = validate_mod.validate_skill(skill)
    assert not any("escapes the installed skill root" in error for error in errors)


def test_validate_skill_flags_a_dangling_link_inside_the_skill(validate_mod, tmp_path):
    """A typo that stays inside skills/<name>/ passes the escape check, so the
    escape check alone cannot be what proves a link resolves."""
    skill = tmp_path / "demo-skill"
    _write_skill(skill, body="See [the runbook](./RUNBOOK.md) for steps.\n")
    errors = validate_mod.validate_skill(skill)
    assert any("no such file exists" in error for error in errors)


def test_validate_skill_allows_a_link_whose_target_exists(validate_mod, tmp_path):
    """Paired with the dangling test above: same link text, target exists."""
    skill = tmp_path / "demo-skill"
    _write_skill(skill, body="See [the runbook](./RUNBOOK.md) for steps.\n")
    (skill / "RUNBOOK.md").write_text("# Runbook\n", encoding="utf-8")
    errors = validate_mod.validate_skill(skill)
    assert not any("no such file exists" in error for error in errors)


def test_validate_skill_ignores_links_inside_a_fenced_block(validate_mod, tmp_path):
    """An illustrative link in a code fence is a sample, not a reference."""
    skill = tmp_path / "demo-skill"
    _write_skill(
        skill,
        body=(
            "Example markdown:\n\n"
            "```markdown\n[x](../../../elsewhere/README.md)\n```\n"
        ),
    )
    errors = validate_mod.validate_skill(skill)
    assert not any("escapes the installed skill root" in error for error in errors)


def test_validate_skill_flags_missing_repo_relative_path(validate_mod, tmp_path):
    """`tools/...` anchors at the repository root, so a missing tail is a defect."""
    skill = tmp_path / "demo-skill"
    _write_skill(
        skill, body="Run `tools/dnn-benchmarking/setup.ps1` before building.\n"
    )
    errors = validate_mod.validate_skill(skill)
    assert any("tools/dnn-benchmarking/setup.ps1" in error for error in errors)


def test_validate_skill_flags_missing_skill_relative_path(validate_mod, tmp_path):
    """`scripts/...` anchors inside the skill, so a missing tail is a defect too."""
    skill = tmp_path / "demo-skill"
    _write_skill(skill, body="Run `scripts/windows/windows_build_setup.ps1` first.\n")
    (skill / "scripts").mkdir(parents=True, exist_ok=True)
    errors = validate_mod.validate_skill(skill)
    assert any("windows_build_setup.ps1" in error for error in errors)


def test_validate_skill_allows_resolvable_and_ambiguous_paths(validate_mod, tmp_path):
    """Only an anchored miss is flagged -- the conservative half of the check.
    Paired with the two tests above, which share the token shape and must fail."""
    skill = tmp_path / "demo-skill"
    (skill / "scripts").mkdir(parents=True, exist_ok=True)
    (skill / "scripts" / "helper.py").write_text("pass\n", encoding="utf-8")
    _write_skill(
        skill,
        body=(
            # resolvable from the repository root
            "See `projects/hipdnn/tools/ai/validate-skills.py`.\n"
            # resolvable inside the skill
            "Run `scripts/helper.py`.\n"
            # a bare filename is prose, not a path claim
            "Then `setup.ps1` finishes the job.\n"
            # unanchored, so ambiguous: left alone
            "Also `descriptors/README.md` in the provider tree.\n"
            # templated, so not a literal path
            "And `<build-dir>/stamp/config.json` after configure.\n"
        ),
    )
    errors = validate_mod.validate_skill(skill)
    assert not any("does not exist" in error for error in errors), errors


def test_validate_skill_flags_undefined_command_placeholder(validate_mod, tmp_path):
    skill = tmp_path / "demo-skill"
    _write_skill(
        skill,
        body="Stage the DLL:\n\n```bash\n<PY> scripts/comgr_stage.py --verbose\n```\n",
    )
    errors = validate_mod.validate_skill(skill)
    assert any("placeholder '<PY>'" in error for error in errors)


def test_validate_skill_allows_placeholder_introduced_in_prose(validate_mod, tmp_path):
    """Paired with the test above: same command block, one prose sentence added."""
    skill = tmp_path / "demo-skill"
    _write_skill(
        skill,
        body=(
            "`<PY>` is the resolved Python interpreter for the active host.\n\n"
            "Stage the DLL:\n\n```bash\n<PY> scripts/comgr_stage.py --verbose\n```\n"
        ),
    )
    errors = validate_mod.validate_skill(skill)
    assert not any("placeholder" in error for error in errors), errors


def test_validate_skill_ignores_generics_in_a_language_fence(validate_mod, tmp_path):
    """C++ template arguments and include headers are not placeholders."""
    skill = tmp_path / "demo-skill"
    _write_skill(
        skill,
        body=(
            "Example:\n\n```cpp\n#include <algorithm>\nstd::vector<int> values;\n```\n"
        ),
    )
    errors = validate_mod.validate_skill(skill)
    assert not any("placeholder" in error for error in errors), errors


def test_validate_skill_requires_claude_frontmatter_fields(validate_mod, tmp_path):
    """A skill named in the claude_commands set needs argument-hint + allowed-tools."""
    skill = tmp_path / "hipdnn-pr-quality"
    _write_skill(skill)
    errors = validate_mod.validate_skill(skill)
    assert any("argument-hint" in error for error in errors)
    assert any("allowed-tools" in error for error in errors)


def test_symlink_target_recognises_a_git_materialized_link(validate_mod, tmp_path):
    """Windows checkouts store a mode-120000 entry as a file holding the target.
    The duplicate-script pair is a symlink upstream, so it must not read as a
    byte-identity violation."""
    real = tmp_path / "a" / "script.py"
    real.parent.mkdir(parents=True)
    real.write_text("print('hello')\n", encoding="utf-8")
    materialized = tmp_path / "b" / "script.py"
    materialized.parent.mkdir(parents=True)
    materialized.write_text("../a/script.py", encoding="utf-8")
    assert validate_mod.symlink_target(materialized) == real.resolve()


def test_symlink_target_ignores_ordinary_script_content(validate_mod, tmp_path):
    """Paired with the test above so a real divergence still reads as content."""
    real = tmp_path / "script.py"
    real.write_text("print('hello')\n", encoding="utf-8")
    assert validate_mod.symlink_target(real) is None

    dangling = tmp_path / "dangling.py"
    dangling.write_text("../nowhere/script.py", encoding="utf-8")
    assert validate_mod.symlink_target(dangling) is None


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
