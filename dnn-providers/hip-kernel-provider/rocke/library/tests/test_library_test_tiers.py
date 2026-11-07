# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts separating discovered unit tests from explicit resource gates."""

from __future__ import annotations

import fnmatch
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from rocke.tests.subprocess_launcher import (
    rocke_python_command,
    rocke_subprocess_env,
)

from . import run_all
from ._attention_bwd_counted_cases import (
    require_explicit_gate_entrypoint,
    selected_isa_arch,
)
from ._explicit_gate_policy import preflight_is_disabled, selection_is_partial
from .gates import conftest as gate_config
from .run_all import EXPLICIT_GATES, _gate_environment, _has_stage_output, _run

_GATE_DIR = Path(__file__).with_name("gates")
_EXPECTED_GATES = {
    "counted-structural": _GATE_DIR / "attention_bwd_counted_structural_gate.py",
    "counted-compiler": _GATE_DIR / "attention_bwd_counted_compiler_gate.py",
}


def test_explicit_gates_are_enrolled_but_not_default_pytest_modules():
    assert EXPLICIT_GATES == _EXPECTED_GATES
    for path in EXPLICIT_GATES.values():
        assert path.is_file()
        assert not fnmatch.fnmatch(path.name, "test_*.py")


@pytest.mark.parametrize(
    "arguments",
    [
        ["-k", "manifest"],
        ["-kmanifest"],
        ["-m=gpu"],
        ["-mgpu"],
        ["--collect-only"],
        ["--setup-plan"],
        ["--stepwise"],
        ["gate.py::test_one_row"],
    ],
)
def test_gate_selection_arguments_are_partial(arguments):
    assert selection_is_partial(arguments)
    assert not selection_is_partial(["-q", "-n", "4", "--durations=10"])


@pytest.mark.parametrize("gate", tuple(_EXPECTED_GATES))
@pytest.mark.parametrize("windows_path", (False, True))
def test_direct_gate_file_is_not_treated_as_partial_selection(gate, windows_path):
    path = str(_EXPECTED_GATES[gate])
    if windows_path:
        path = path.replace("/", "\\")
    complete = SimpleNamespace(
        invocation_params=SimpleNamespace(args=(path, "-v", "-rs", "--timeout=60")),
        option=SimpleNamespace(),
    )
    selected_node = SimpleNamespace(
        invocation_params=SimpleNamespace(args=(f"{path}::test_one_row",)),
        option=SimpleNamespace(),
    )

    assert not gate_config._selection_requested(complete)
    assert gate_config._selection_requested(selected_node)


@pytest.mark.parametrize(
    "arguments", [["--noconftest"], ["--confcutdir", "tests"], ["--confcutdir=tests"]]
)
def test_gate_preflight_suppression_is_never_allowed(arguments, monkeypatch):
    assert preflight_is_disabled(arguments)
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        [
            "run_all.py",
            "--gate",
            "counted-compiler",
            "--allow-partial-gate",
            "--",
            *arguments,
        ],
    )
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


def test_gate_module_requires_loaded_preflight_conftest():
    with pytest.raises(RuntimeError, match="conftest enabled"):
        require_explicit_gate_entrypoint({"ROCKE_EXPLICIT_GATE_RUN": "1"})


@pytest.mark.parametrize("addopts", ("-k selected_name", "--lf"))
def test_inherited_pytest_addopts_cannot_filter_direct_gate(addopts, tmp_path):
    gate = tmp_path / _EXPECTED_GATES["counted-structural"].name
    gate.write_text("def test_selected_name():\n    pass\n", encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "PYTEST_ADDOPTS": addopts,
            "ROCKE_EXPLICIT_GATE_RUN": "1",
            "ROCKE_EXPLICIT_GATE_PARTIAL": "0",
        }
    )
    result = subprocess.run(
        rocke_python_command(
            "-m", "pytest", "-q", "-p", "tests.gates.conftest", str(gate)
        ),
        cwd=str(_GATE_DIR.parent.parent),
        env=rocke_subprocess_env(environment),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "selection filters need explicit partial mode" in result.stdout + result.stderr
    )


def test_required_compiler_gate_cannot_inherit_optional_policy(monkeypatch):
    monkeypatch.setenv("ROCKE_COUNTED_ISA_OPTIONAL", "1")
    monkeypatch.setenv("ROCKE_BWD_ISA_ARCH", "gfx942")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k manifest")
    monkeypatch.setenv("ROCKE_EXPLICIT_GATE_PREFLIGHT", "1")
    monkeypatch.setenv("ROCKE_TEST_SHARD", "0/2")
    required = _gate_environment("counted-compiler", require_compiler=True)
    optional = _gate_environment("counted-compiler", require_compiler=False)
    partial = _gate_environment(
        "counted-compiler", require_compiler=True, gate_arch="gfx950"
    )
    assert "ROCKE_COUNTED_ISA_OPTIONAL" not in required
    assert optional["ROCKE_COUNTED_ISA_OPTIONAL"] == "1"
    assert required["ROCKE_BWD_NO_COMGR_PIN"] == "1"
    assert optional["ROCKE_BWD_NO_COMGR_PIN"] == "1"
    assert "ROCKE_BWD_ISA_ARCH" not in required
    assert "ROCKE_BWD_ISA_ARCH" not in optional
    assert "PYTEST_ADDOPTS" not in required
    assert "PYTEST_ADDOPTS" not in optional
    assert "ROCKE_EXPLICIT_GATE_PREFLIGHT" not in required
    assert "ROCKE_EXPLICIT_GATE_PREFLIGHT" not in optional
    assert "ROCKE_TEST_SHARD" not in required
    assert "ROCKE_TEST_SHARD" not in optional
    assert partial["ROCKE_TEST_SHARD"] == "0/2"
    assert partial["ROCKE_BWD_ISA_ARCH"] == "gfx950"


@pytest.mark.parametrize("flavor", ("llvm22", "llvm23", "llvm24"))
def test_compiler_preflight_accepts_qualified_active_flavor(flavor):
    assert (
        gate_config._qualification_gap(
            flavor,
            flavor,
            "comgr3.5/rocm10.3",
            frozenset(("gfx942", "gfx950")),
            ("gfx942", "gfx950"),
        )
        is None
    )


@pytest.mark.parametrize(
    "requested,active,comgr,targets,match",
    [
        ("llvm20", "llvm20", "comgr3.0/rocm7.1", frozenset(("gfx942",)), "unsupported"),
        (
            "llvm24",
            "llvm23",
            "comgr3.5/rocm10.3",
            frozenset(("gfx942",)),
            "does not match",
        ),
        (
            "llvm24",
            None,
            "comgr3.5/rocm10.3",
            frozenset(("gfx942",)),
            "could not query",
        ),
        (
            "llvm24",
            "llvm24",
            "comgr3.5/rocmunknown",
            frozenset(("gfx942",)),
            "provenance",
        ),
        ("llvm24", "llvm24", "comgr3.5/rocm10.3", None, "ISA targets"),
        ("llvm24", "llvm24", "comgr3.5/rocm10.3", frozenset(("gfx950",)), "gfx942"),
    ],
)
def test_compiler_preflight_rejects_unqualified_toolchain(
    requested, active, comgr, targets, match
):
    gap = gate_config._qualification_gap(requested, active, comgr, targets, ("gfx942",))
    assert match in gap


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({}, None),
        ({"ROCKE_BWD_ISA_ARCH": ""}, None),
        ({"ROCKE_BWD_ISA_ARCH": "gfx942"}, "gfx942"),
    ],
)
def test_isa_arch_selector_normalizes_empty_and_unset(environ, expected):
    assert selected_isa_arch(environ) == expected


def test_complete_gate_rejects_inherited_sharding(monkeypatch):
    path = _EXPECTED_GATES["counted-structural"]
    session = SimpleNamespace(
        config=SimpleNamespace(
            invocation_params=SimpleNamespace(args=(str(path),)),
            option=SimpleNamespace(),
        ),
        items=[SimpleNamespace(path=path)],
    )
    monkeypatch.setenv("ROCKE_TEST_SHARD", "0/2")
    monkeypatch.setenv("ROCKE_EXPLICIT_GATE_PARTIAL", "0")
    with pytest.raises(pytest.exit.Exception, match="sharding needs explicit partial"):
        gate_config.pytest_collection_finish(session)

    monkeypatch.setenv("ROCKE_EXPLICIT_GATE_PARTIAL", "1")
    gate_config.pytest_collection_finish(session)


def test_compiler_preflight_has_one_gate_level_outcome(monkeypatch):
    monkeypatch.setenv("ROCKE_COUNTED_ISA_OPTIONAL", "1")
    with pytest.raises(pytest.exit.Exception) as optional:
        gate_config._stop("missing toolchain")
    assert optional.value.returncode == 77

    monkeypatch.setenv("ROCKE_COUNTED_ISA_OPTIONAL", "0")
    with pytest.raises(pytest.exit.Exception) as required:
        gate_config._stop("missing toolchain")
    assert required.value.returncode == 1


@pytest.mark.parametrize(
    "arguments", (["--junitxml=report.xml"], ["--junit-xml", "report.xml"])
)
def test_junit_arguments_are_stage_owned(arguments):
    assert _has_stage_output(arguments)


def test_source_runner_rejects_shared_junit_output(monkeypatch):
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--", "--junitxml=report.xml"],
    )
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


def test_source_runner_reports_only_as_partial(monkeypatch, capsys):
    monkeypatch.setattr(
        run_all.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--only", "dense"])
    assert run_all.main() == 0
    assert "RESULT: PARTIAL" in capsys.readouterr().out


def test_source_runner_stops_before_resource_gates_after_unit_failure(monkeypatch):
    calls = []

    def fail_units(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(run_all.subprocess, "run", fail_units)
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py"])
    assert run_all.main() == 1
    assert len(calls) == 1


def test_source_runner_stops_before_compiler_gate_after_structural_failure(monkeypatch):
    calls = []
    results = iter((0, 1))

    def fail_structural(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=next(results))

    monkeypatch.setattr(run_all.subprocess, "run", fail_structural)
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py"])
    assert run_all.main() == 1
    assert len(calls) == 2
    assert calls[-1][-1] == str(EXPLICIT_GATES["counted-structural"])


def test_source_runner_keep_going_runs_named_stages_in_order(monkeypatch):
    calls = []
    results = iter((1, 0, 0))

    def run_all_stages(command, **kwargs):
        calls.append((command, kwargs.get("env")))
        return SimpleNamespace(returncode=next(results))

    monkeypatch.setattr(run_all.subprocess, "run", run_all_stages)
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--keep-going"])
    assert run_all.main() == 1

    unit, structural, compiler = calls
    assert unit[0][1:4] == ["-m", "pytest", str(Path(run_all.TESTS))]
    assert unit[1] is None
    assert structural[0][-1] == str(EXPLICIT_GATES["counted-structural"])
    assert structural[1]["ROCKE_EXPLICIT_GATE_RUN"] == "1"
    assert structural[1]["ROCKE_EXPLICIT_GATE_PARTIAL"] == "0"
    assert compiler[0][-1] == str(EXPLICIT_GATES["counted-compiler"])
    assert compiler[1]["ROCKE_COUNTED_ISA_OPTIONAL"] == "1"
    assert compiler[1]["ROCKE_BWD_NO_COMGR_PIN"] == "1"


def test_source_runner_reports_optional_gate_skip_once(monkeypatch):
    monkeypatch.setattr(
        "tests.run_all.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=77),
    )
    command = ["python", "gate.py"]
    optional = _run("optional compiler gate", command, allow_skip=True)
    required = _run("required compiler gate", command, allow_skip=False)
    assert optional.code == 0 and optional.skipped
    assert required.code == 77 and not required.skipped


def test_source_runner_reports_incomplete_when_optional_gate_skips(monkeypatch, capsys):
    results = iter((0, 0, 77))
    monkeypatch.setattr(
        run_all.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=next(results)),
    )
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py"])
    assert run_all.main() == 0
    assert "RESULT: INCOMPLETE" in capsys.readouterr().out
