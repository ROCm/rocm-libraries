# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The one compliance pattern file loads, finds its leaks, and refuses a broken file."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_patterns_module() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "tools" / "compliance_patterns.py"
    if not path.is_file():
        pytest.skip(
            "rocke/tools/compliance_patterns.py is absent from this tree",
            allow_module_level=True,
        )
    spec = importlib.util.spec_from_file_location(
        "compliance_patterns_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


patterns = _load_patterns_module()


# Seeded leaks are joined at run time so this file itself holds no compliance hit: the
# compliance hook and the scanner would refuse a literal one.
@pytest.mark.parametrize(
    ("text", "rule_id"),
    [
        (
            "tracked in https://example." + "atlassian.net/browse/PROJ-1",
            "issue-tracker-link",
        ),
        ("slides at team." + "sharepoint.com/sites/x", "internal-document-host"),
        ("opened vscode-webview:" + "//abc/index.html", "editor-local-link"),
        ("dashboard on perf.example." + "amd.com:8080", "internal-company-host"),
        ("ran as Slurm " + "job 4242", "cluster-job-id"),
        ("card on bus id " + "0x83", "device-bus-id"),
    ],
)
def test_each_rule_finds_its_leak(text: str, rule_id: str) -> None:
    hits = patterns.find_hits(f"intro\n{text}\n", patterns.load_rules())
    assert [(hit.rule_id, hit.line) for hit in hits] == [(rule_id, 2)]


@pytest.mark.parametrize(
    "text",
    [
        "docs: https://rocm.docs.amd.com/en/latest/ and https://www.amd.com",
        "tracked as AICK-1234 (plain-text ids are allowed)",
        "mail to someone@amd.com",
        "a public CI run: https://github.com/ROCm/example/actions/runs/1",
        "ran on gfx942",
    ],
)
def test_public_text_has_no_hits(text: str) -> None:
    assert patterns.find_hits(text, patterns.load_rules()) == []


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"rules": []}, "expected a top-level object with a non-empty 'rules' list"),
        (
            {"rules": [{"id": "x", "pattern": "a"}]},
            "expected an object with exactly the keys",
        ),
        (
            {
                "rules": [
                    {
                        "id": "x",
                        "pattern": "(",
                        "ignore_case": False,
                        "message": "m",
                        "allowed": [],
                    }
                ]
            },
            "rule='x', expected a valid regular expression",
        ),
        (
            {
                "rules": [
                    {
                        "id": "x",
                        "pattern": "a",
                        "ignore_case": False,
                        "message": "m",
                        "allowed": [],
                    }
                ]
                * 2
            },
            "expected unique ids",
        ),
    ],
)
def test_a_broken_pattern_file_is_refused(
    tmp_path: Path, document: dict, expected: str
) -> None:
    path = tmp_path / "patterns.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        patterns.load_rules(path)


def test_the_scanner_command_fails_on_a_leak_and_passes_clean_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "clean.md").write_text("gfx942 only\n", encoding="utf-8")
    command = [sys.executable, str(Path(patterns.__file__)), str(tmp_path)]
    clean = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert clean.returncode == 0, clean.stderr

    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "leak.md").write_text(
        "a\nhttps://x." + "atlassian.net/\n", encoding="utf-8"
    )
    leaked = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert leaked.returncode == 1
    assert "leak.md:2: [issue-tracker-link]" in leaked.stderr


def test_the_committed_agent_files_hold_no_compliance_hit() -> None:
    """The backstop: the real agent files and checking tools pass the scan."""
    rocke = Path(patterns.__file__).resolve().parents[1]
    agents = rocke / "agents"
    if not agents.is_dir():
        pytest.skip("rocke/agents is absent from this tree")
    found = patterns.scan_paths([agents, rocke / "tools"], patterns.load_rules())
    assert found == [], [hit.describe(str(path)) for path, hit in found]
