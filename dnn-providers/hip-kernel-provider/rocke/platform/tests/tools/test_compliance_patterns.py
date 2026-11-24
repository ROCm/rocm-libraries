# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The one compliance pattern file loads, finds its leaks, and refuses a broken file."""

from __future__ import annotations

import importlib.util
import json
import shutil
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
        ("measured on an M" + "I300X", "product-name"),
        ("the NUM_XCDS_M" + "I350X constant", "product-name"),
        ("a Rad" + "eon card", "product-name"),
        ("an AMD Inst" + "inct accelerator", "product-name"),
        ("the R" + "X 9070 board", "product-name"),
        # GPU codenames.
        ("a Na" + "vi 48 board", "chip-codename"),
        ("the Ald" + "ebaran die", "chip-codename"),
        ("an Arc" + "turus card", "chip-codename"),
        ("the Aqua Van" + "jaram package", "chip-codename"),
        ("a Sienna Cich" + "lid board", "chip-codename"),
        ("a Navy Floun" + "der board", "chip-codename"),
        ("a Dimgrey Cave" + "fish board", "chip-codename"),
        ("a Beige Go" + "by board", "chip-codename"),
        ("a Plum Bon" + "ito board", "chip-codename"),
        ("a Wheat N" + "as board", "chip-codename"),
        ("a Ve" + "ga 20 card", "chip-codename"),
        # APU and CPU codenames that are not ordinary words.
        ("a Str" + "ix Halo node", "chip-codename"),
        ("a Str" + "ix Point laptop", "chip-codename"),
        ("a Kra" + "ckan laptop", "chip-codename"),
        ("a Hawk Po" + "int laptop", "chip-codename"),
        ("a Dragon Ra" + "nge laptop", "chip-codename"),
        ("a Fire Ra" + "nge laptop", "chip-codename"),
        ("a Granite Ri" + "dge desktop", "chip-codename"),
        # Codenames that are also ordinary words: capitalized, after AMD or
        # before a hardware noun.
        ("a Pho" + "enix APU", "chip-codename"),
        ("a Gor" + "gon Point laptop", "chip-codename"),
        ("a Med" + "usa Point APU", "chip-codename"),
        ("a Rem" + "brandt laptop", "chip-codename"),
        ("a Rap" + "hael CPU", "chip-codename"),
        ("the Tur" + "in server", "chip-codename"),
        ("a Gen" + "oa node", "chip-codename"),
        ("Berg" + "amo cores", "chip-codename"),
        ("a Si" + "ena platform", "chip-codename"),
        ("AMD Ven" + "ice", "chip-codename"),
        ("a Pol" + "aris GPU", "chip-codename"),
        # Internal sites and node names.
        ("only the Mar" + "kham nodes share $HOME", "internal-site-name"),
        ("srun --nodelist=ct" + "r-node-b01-02", "internal-site-name"),
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
        "the 304-CU gfx942 part has 8 XCDs",
        "an instinctive answer; reorder-cost instincts",
        "clang built from https://github.com/Rad" + "eonOpenCompute/llvm-project",
        "Tensile token M" + "I16x16x1 packs one MFMA atom",
        "a semi-structured name: semi" + "300",
        # Ordinary words that are also codenames, outside a hardware context.
        "the old design rose like a phoenix from its ashes",
        "lower-case phoenix server and turin node",
        "a conference in Venice; the Polaris star; Medusa in the myth",
        "Raphael painted it; Genoa and Siena are cities",
        "a vegan meal; navigate the tree; the Turing machine",
        "fire ranges and a hawk pointer",
        "srun --nodelist=<node> on the <login-host>",
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


def test_the_tracked_rocke_tree_holds_no_compliance_hit() -> None:
    """Every git-tracked file of the rocke tree passes the scan, so a new leak, a product
    name included, fails this test wherever it is added. Untracked files (local notes,
    build trees) are not scanned."""
    rocke = Path(patterns.__file__).resolve().parents[1]
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    listed = subprocess.run(
        [git, "-C", str(rocke), "ls-files", "-z", "."],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if listed.returncode != 0:
        pytest.skip(f"rocke is not a git checkout: {listed.stderr.strip()}")
    tracked = [rocke / name for name in listed.stdout.split("\0") if name]
    present = [path for path in tracked if path.is_file()]
    assert present, "git ls-files listed no file under rocke"
    found = patterns.scan_paths(present, patterns.load_rules())
    assert found == [], [hit.describe(str(path)) for path, hit in found]
