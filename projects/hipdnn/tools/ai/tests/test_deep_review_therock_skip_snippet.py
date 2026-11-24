# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the TheRock lane's inline skip-ci classifier snippet (lanes/therock.md
section 1), including its fallback when `_is_path_unrelated_to_ci` is renamed or gone.
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
