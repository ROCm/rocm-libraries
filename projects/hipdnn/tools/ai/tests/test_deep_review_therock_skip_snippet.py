# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the TheRock lane's inline skip-ci classifier snippet (lanes/therock.md
section 1), including its fallback when `_is_path_unrelated_to_ci` is renamed or gone,
and for the section 2 tier helper drift check.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

THEROCK_MD = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "hipdnn-deep-review"
    / "lanes"
    / "therock.md"
)

DOC_PATH = "projects/hipdnn/docs/TESTING.md"
TOOLS_PATH = "projects/hipdnn/tools/ai/skills/x.py"
CPP_PATH = "projects/hipdnn/backend/src/HipdnnBackend.cpp"
FAKE_PIN = "deadbeef1234"
LOAD_CFG = (
    "def load_skip_ci_config(extension_config_path=None):\n"
    '    return (["projects/hipdnn/docs/*", "projects/hipdnn/tools/*", "*.md"], set())\n'
)


def _extract_snippet() -> str:
    text = THEROCK_MD.read_text(encoding="utf-8")
    section = re.search(
        r"^## 1\. Does TheRock CI run at all\?\n(.*?)^## ", text, re.M | re.S
    )
    assert section, "section 1 header not found in therock.md"
    m = re.search(r"python3 -c '(.*?)' \"\$C\" \"\$ROCK_REF\"", section.group(1), re.S)
    assert m, "inline python snippet not found in therock.md section 1"
    return m.group(1)


def _run(tmp_path, module_src):
    (tmp_path / "configure_ci_path_filters.py").write_text(module_src, encoding="utf-8")
    cfg = tmp_path / "skip-ci-config.toml"
    cfg.write_text("", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    return subprocess.run(
        [sys.executable, "-c", _extract_snippet(), str(cfg), FAKE_PIN],
        input=f"{DOC_PATH}\n{TOOLS_PATH}\n{CPP_PATH}\n",
        capture_output=True,
        text=True,
        env=env,
    )


def _classification(stdout):
    rows = re.findall(r"^(\[skip\]|\[ci\])\s+(.*)$", stdout, re.M)
    return {path: tag for tag, path in rows}


def test_present_helper_classifies_docs_vs_cpp_without_drift(tmp_path):
    src = LOAD_CFG + (
        "import fnmatch\n"
        "def _is_path_unrelated_to_ci(path, patterns):\n"
        "    return any(fnmatch.fnmatch(path, p) for p in patterns)\n"
    )
    r = _run(tmp_path, src)
    assert r.returncode == 0, r.stderr
    assert "HELPER-DRIFT" not in r.stderr
    assert _classification(r.stdout) == {
        DOC_PATH: "[skip]",
        TOOLS_PATH: "[skip]",
        CPP_PATH: "[ci]",
    }


def test_renamed_helper_is_found_and_reported_as_drift(tmp_path):
    src = LOAD_CFG + (
        "import fnmatch\n"
        "def _decoy_always_false(path, patterns):\n"
        "    return False\n"
        "def _classify_path_v2(path, patterns):\n"
        "    return any(fnmatch.fnmatch(path, p) for p in patterns)\n"
    )
    r = _run(tmp_path, src)
    assert r.returncode == 0, r.stderr
    assert (
        f"HELPER-DRIFT: _is_path_unrelated_to_ci not found in TheRock {FAKE_PIN}"
        in r.stderr
    )
    assert "_classify_path_v2" in r.stderr
    assert _classification(r.stdout) == {
        DOC_PATH: "[skip]",
        TOOLS_PATH: "[skip]",
        CPP_PATH: "[ci]",
    }


def test_no_usable_helper_stops_without_classifying(tmp_path):
    src = LOAD_CFG + (
        "import fnmatch\n"
        "def _decoy_always_true(path, patterns):\n"
        '    fnmatch.fnmatch(path, "*")\n'
        "    return True\n"
    )
    r = _run(tmp_path, src)
    assert r.returncode != 0
    assert "STOP:" in (r.stdout + r.stderr)
    assert not re.search(r"^\[skip\]|\[ci\]", r.stdout, re.M)


TIER_HELPERS = {
    "_determine_test_type": "configure_multi_arch_ci.py",
    "test_type_for_family": "amdgpu_family_matrix.py",
    "is_external_repo_ci_only_change": "configure_ci_path_filters.py",
}


def _extract_tier_drift_block() -> str:
    text = THEROCK_MD.read_text(encoding="utf-8")
    blocks = re.findall(r"^```bash\n(.*?)^```", text, re.M | re.S)
    found = [b for b in blocks if "_determine_test_type:" in b]
    assert len(found) == 1, "tier drift-check block not found in therock.md"
    return found[0]


def _run_tier_drift(tmp_path, helpers):
    tree = tmp_path / "TheRock"
    gha = tree / "build_tools" / "github_actions"
    gha.mkdir(parents=True)
    for fname in set(TIER_HELPERS.values()):
        names = [n for n, f in helpers.items() if f == fname]
        body = "".join(f"def {n}():\n    pass\n" for n in names)
        (gha / fname).write_text(body or "# empty\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    (scratch / "ROCK").mkdir(parents=True)
    (scratch / "ROCK" / "T").write_text(f"{tree}\n", encoding="utf-8")
    (scratch / "ROCK" / "ref").write_text(f"{FAKE_PIN}\n", encoding="utf-8")
    (tmp_path / "env.sh").write_text(
        f"SCRATCH={scratch}\nGREP=grep\n", encoding="utf-8"
    )
    return subprocess.run(
        ["bash", "-c", _extract_tier_drift_block()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_tier_drift_check_lists_every_helper_when_all_present(tmp_path):
    r = _run_tier_drift(tmp_path, TIER_HELPERS)
    assert "HELPER-DRIFT" not in r.stderr
    lines = [ln for ln in r.stdout.splitlines() if FAKE_PIN in ln]
    assert len(lines) == 1, r.stdout
    for name in TIER_HELPERS:
        assert re.search(rf"\b{name}\b", lines[0]), lines[0]


def test_tier_drift_check_reports_exactly_the_missing_helper(tmp_path):
    missing = "is_external_repo_ci_only_change"
    helpers = {n: f for n, f in TIER_HELPERS.items() if n != missing}
    r = _run_tier_drift(tmp_path, helpers)
    drift = re.findall(
        r"^HELPER-DRIFT: (\S+) not found in TheRock (\S+)$", r.stderr, re.M
    )
    assert drift == [(missing, FAKE_PIN)], r.stderr
    assert FAKE_PIN not in r.stdout
