# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The golden index and its tool (rocke/tools/golden.py).

The re-record command a failing golden prints is one the tool accepts, and the
index and the tool refuse a malformed entry or request.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import ModuleType

import pytest
from rocke.core.golden_file import rerecord_command

_TEST_FILE = Path(__file__).resolve()
# Source: rocke/platform/tests/tools/<this>; the tool is rocke/tools/golden.py.
_SOURCE_ROCKE = _TEST_FILE.parents[3]


def _tool_path() -> Path:
    """The tool in a source checkout, else its copy beside this installed test."""
    for candidate in (
        _SOURCE_ROCKE / "tools" / "golden.py",
        _TEST_FILE.with_name("golden.py"),
    ):
        if candidate.is_file():
            return candidate
    pytest.skip("rocke/tools/golden.py is absent from this platform-only tree")


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rocke_golden_tool", _tool_path())
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, as an import would: dataclasses look it up.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def index(tool) -> ModuleType:
    # The tool put its directory on sys.path to import the index beside it.
    return importlib.import_module("golden_index")


# --- the printed command is the tool's -------------------------------------------


# The source trees whose golden checks name their golden (Golden.files' roots).
_CHECKER_TREES = ("platform/tests", "platform/python/rocke", "library/tests")


def _golden_name_constants() -> dict[str, str]:
    """``<file>:<constant>`` -> value, for every module-level ``*GOLDEN_NAME``
    string constant in the source trees: the name a check prints in its
    re-record command."""
    found = {}
    for tree in _CHECKER_TREES:
        root = _SOURCE_ROCKE / tree
        if not root.is_dir():
            pytest.skip(f"{tree} is absent: the scan needs a source checkout")
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "GOLDEN_NAME" not in text:
                continue
            for node in ast.parse(text).body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.endswith("GOLDEN_NAME")
                    and isinstance(node.value, ast.Constant)
                ):
                    relative = path.relative_to(_SOURCE_ROCKE).as_posix()
                    found[f"{relative}:{node.targets[0].id}"] = node.value.value
    return found


def test_every_golden_name_a_check_prints_is_in_the_index(tool, index):
    constants = _golden_name_constants()
    known = {golden.name for golden in index.GOLDENS}
    assert {where: name for where, name in constants.items() if name not in known} == {}
    # And every golden is named by some check, so its failure prints a command.
    assert known - set(constants.values()) == set()


def test_every_golden_rerecord_command_parses_as_the_tool_runs_it(tool, index):
    parser = tool.build_parser()
    for golden in index.GOLDENS:
        families = ["first", "second"] if golden.takes_families else None
        words = shlex.split(rerecord_command(golden.name, families))
        assert words[:2] == ["python", "tools/golden.py"]
        args = parser.parse_args(words[2:])
        assert (args.command, args.names) == ("rerecord", [golden.name])
        assert args.only == ("first,second" if families else None)


def test_a_family_list_expands_into_each_record_step(index):
    differential = index.golden_named("differential")
    assert [step.arguments(["gemm", "moe"])[-2:] for step in differential.record] == [
        ["--only", "gemm,moe"],
        ["--only", "gemm,moe"],
    ]
    kda = index.golden_named("kda_gfx942")
    with pytest.raises(ValueError, match="writes its golden whole"):
        kda.record[0].arguments(["kda_gfx942"])


@pytest.mark.parametrize(
    "argv, message",
    [
        (
            ["rerecord", "kda_gfx942", "--only", "x"],
            "is re-recorded whole; drop --only",
        ),
        (["rerecord", "nope"], "unknown golden - name='nope', expected one of"),
        (["rerecord"], "rerecord takes golden names, --all or --conflicted"),
        (["rerecord", "conv", "--all"], "rerecord takes golden names, --all"),
        (["rerecord", "conv", "kda_gfx942", "--only", "x"], "exactly one golden"),
    ],
)
def test_rerecord_refuses_a_bad_request(tool, capsys, argv, message):
    with pytest.raises(SystemExit) as raised:
        tool.main(argv)
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_list_json_describes_every_golden(tool, index, capsys):
    assert tool.main(["list", "--json"]) == 0
    described = json.loads(capsys.readouterr().out)
    assert [entry["name"] for entry in described] == [g.name for g in index.GOLDENS]


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"name": "not an identifier"}, "expected an identifier"),
        ({"files": ("/abs/golden.json",)}, "relative to the rocke directory"),
        ({"checked_by": ("../outside.py",)}, "relative to the rocke directory"),
        ({"fingerprint": "sometimes"}, "expected one of"),
        ({"checked_by": ()}, "checked_by is empty"),
    ],
)
def test_an_index_entry_is_validated(index, fields, message):
    entry = {
        "name": "sample",
        "files": ("platform/tests/golden/sample.json",),
        "checked_by": ("platform/tests/test_sample.py",),
        "record": (index.RecordStep(cwd="platform", argv=("tests/test_sample.py",)),),
        "fingerprint": index.DOCUMENT,
    }
    with pytest.raises(ValueError, match=message):
        index.Golden(**{**entry, **fields})


def test_record_steps_must_agree_on_taking_families(index):
    whole = index.RecordStep(cwd="platform", argv=("a.py",))
    narrowed = index.RecordStep(
        cwd="platform", argv=("b.py",), only=("--only", "{families}")
    )
    with pytest.raises(ValueError, match="either every record step takes a family"):
        index.Golden(
            name="mixed",
            files=("platform/tests/golden/m.json",),
            checked_by=("platform/tests/test_m.py",),
            record=(whole, narrowed),
            fingerprint=index.DOCUMENT,
        )
    with pytest.raises(ValueError, match="expected an argument containing"):
        index.RecordStep(cwd="platform", argv=("a.py",), only=("--only",))


def test_gpu_goldens_are_skipped_by_name(tool, index, capsys):
    # --all and --conflicted both drop the goldens only a GPU can re-record.
    kept = tool._cpu_recordable(index.GOLDENS)
    assert [golden.name for golden in kept] == [
        golden.name for golden in index.GOLDENS if not golden.needs_gpu
    ]
    assert capsys.readouterr().out == (
        "[golden] examples: skipped, re-record it on a GPU\n"
    )
