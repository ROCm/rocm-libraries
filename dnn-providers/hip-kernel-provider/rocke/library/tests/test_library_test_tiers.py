# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts separating discovered unit tests from explicit resource gates."""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import _file_manifest_plugin as manifest_plugin
from . import run_all
from ._attention_bwd_counted_cases import (
    require_explicit_gate_entrypoint,
    selected_isa_arch,
)
from .gates import conftest as gate_config
from .run_all import (
    EXPLICIT_GATES,
    _gate_environment,
    _preflight_is_disabled,
    _run,
    _selection_is_partial,
)

_GATE_DIR = Path(__file__).with_name("gates")
_EXPECTED_GATES = {
    "counted-ir": _GATE_DIR / "attention_bwd_counted_ir_gate.py",
    "counted-isa": _GATE_DIR / "attention_bwd_counted_isa_gate.py",
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
    assert _selection_is_partial(arguments)
    assert not _selection_is_partial(["-q", "-n", "4", "--durations=10"])


@pytest.mark.parametrize(
    "arguments", [["--noconftest"], ["--confcutdir", "tests"], ["--confcutdir=tests"]]
)
def test_gate_preflight_suppression_is_never_allowed(arguments, monkeypatch):
    assert _preflight_is_disabled(arguments)
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        [
            "run_all.py",
            "--gate",
            "counted-isa",
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


@pytest.mark.parametrize("addopts", ("-k gate_manifest_is_stable", "--lf"))
def test_inherited_pytest_addopts_cannot_filter_direct_gate(addopts):
    environment = dict(os.environ)
    environment.update(
        {
            "PYTEST_ADDOPTS": addopts,
            "ROCKE_EXPLICIT_GATE_RUN": "1",
            "ROCKE_EXPLICIT_GATE_PARTIAL": "0",
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(_EXPECTED_GATES["counted-ir"])],
        cwd=str(_GATE_DIR.parent.parent),
        env=environment,
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
    monkeypatch.setenv("ROCKE_COUNTED_ISA_PARTIAL", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k manifest")
    monkeypatch.setenv("ROCKE_EXPLICIT_GATE_PREFLIGHT", "1")
    required = _gate_environment("counted-isa", require_compiler=True)
    optional = _gate_environment("counted-isa", require_compiler=False)
    partial = _gate_environment(
        "counted-isa", require_compiler=True, gate_arch="gfx950"
    )
    assert "ROCKE_COUNTED_ISA_OPTIONAL" not in required
    assert optional["ROCKE_COUNTED_ISA_OPTIONAL"] == "1"
    assert required["ROCKE_BWD_NO_COMGR_PIN"] == "1"
    assert optional["ROCKE_BWD_NO_COMGR_PIN"] == "1"
    assert "ROCKE_BWD_ISA_ARCH" not in required
    assert "ROCKE_BWD_ISA_ARCH" not in optional
    assert "ROCKE_COUNTED_ISA_PARTIAL" not in required
    assert "ROCKE_COUNTED_ISA_PARTIAL" not in optional
    assert "PYTEST_ADDOPTS" not in required
    assert "PYTEST_ADDOPTS" not in optional
    assert "ROCKE_EXPLICIT_GATE_PREFLIGHT" not in required
    assert "ROCKE_EXPLICIT_GATE_PREFLIGHT" not in optional
    assert partial["ROCKE_BWD_ISA_ARCH"] == "gfx950"
    assert partial["ROCKE_COUNTED_ISA_PARTIAL"] == "1"


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
    "arguments, marker",
    [
        (["--unit-only"], "not gpu and not exhaustive"),
        (["--unit-only", "--exhaustive"], "not gpu"),
        (["--unit-only", "--gpu"], "gpu or not exhaustive"),
        (["--gpu-only"], "gpu"),
        (["--gpu-only", "--exhaustive"], "gpu"),
    ],
)
def test_source_runner_keeps_exhaustive_matrices_out_of_the_affordable_lane(
    arguments, marker, monkeypatch
):
    calls = []

    def pass_stage(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_all.subprocess, "run", pass_stage)
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", *arguments, "--jobs", "0"])
    assert run_all.main() == 0
    assert len(calls) == 1
    assert calls[0][-2:] == ["-m", marker]


def test_source_runner_zero_jobs_uses_one_traditional_pytest_process(monkeypatch):
    calls = []

    def pass_stage(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_all.subprocess, "run", pass_stage)
    monkeypatch.setattr(
        run_all.sys, "argv", ["run_all.py", "--unit-only", "--jobs", "0"]
    )
    assert run_all.main() == 0
    assert len(calls) == 1
    assert calls[0][0:4] == [sys.executable, "-m", "pytest", str(run_all.TESTS)]


def test_source_runner_positive_jobs_use_fresh_file_processes(monkeypatch):
    calls = []

    def pass_files(**kwargs):
        calls.append(kwargs)
        return run_all._StageResult(0)

    monkeypatch.setattr(run_all, "_run_unit_file_stage", pass_files)
    monkeypatch.setattr(
        run_all.sys, "argv", ["run_all.py", "--unit-only", "--jobs", "3"]
    )
    assert run_all.main() == 0
    assert calls == [
        {
            "marker": "not gpu and not exhaustive",
            "only": "",
            "extra": [],
            "jobs": 3,
            "timeout": 60.0,
        }
    ]


@pytest.mark.parametrize(
    "arguments, expected_jobs, expected_timeout",
    [
        (["--unit-only"], min(8, os.cpu_count() or 1), 60.0),
        (["--gpu-only"], 1, 600.0),
        (["--unit-only", "--", "-m", "gpu"], 1, 600.0),
        (["--unit-only", "--exhaustive"], min(8, os.cpu_count() or 1), 600.0),
    ],
)
def test_source_runner_defaults_match_resource_class(
    arguments, expected_jobs, expected_timeout, monkeypatch
):
    calls = []

    def pass_files(**kwargs):
        calls.append(kwargs)
        return run_all._StageResult(0)

    monkeypatch.setattr(run_all, "_run_unit_file_stage", pass_files)
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", *arguments])
    assert run_all.main() == 0
    assert calls[0]["jobs"] == expected_jobs
    assert calls[0]["timeout"] == expected_timeout


def test_inherited_gpu_marker_uses_gpu_resource_defaults(monkeypatch):
    calls = []

    def pass_files(**kwargs):
        calls.append(kwargs)
        return run_all._StageResult(0)

    monkeypatch.setenv("PYTEST_ADDOPTS", "-m gpu")
    monkeypatch.setattr(run_all, "_run_unit_file_stage", pass_files)
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--unit-only"])
    assert run_all.main() == 0
    assert calls[0]["jobs"] == 1
    assert calls[0]["timeout"] == 600.0


def test_inherited_shared_output_requires_one_pytest_process(monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "--log-file=shared.log")
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--unit-only"])
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


def test_file_manifest_records_each_selected_file_once(tmp_path):
    output = tmp_path / "manifest.txt"
    config = SimpleNamespace(getoption=lambda option: output)
    session = SimpleNamespace(
        config=config,
        items=[
            SimpleNamespace(path=tmp_path / "a.py"),
            SimpleNamespace(path=tmp_path / "a.py"),
            SimpleNamespace(path=tmp_path / "b.py"),
        ],
    )
    manifest_plugin.pytest_collection_finish(session)
    assert output.read_text().splitlines() == [
        str(tmp_path / "a.py"),
        str(tmp_path / "b.py"),
    ]


def test_fresh_file_timeout_names_the_file(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 3)

    monkeypatch.setattr(run_all.subprocess, "run", timeout)
    test_file = tmp_path / "test_slow.py"
    log = tmp_path / "slow.log"
    result = run_all._run_unit_file(test_file, None, "", [], 3, log)
    assert result.code == 124
    assert result.timed_out
    assert result.path == test_file
    assert "TIMEOUT: exceeded 3s" in log.read_text()


def test_source_runner_rejects_nested_xdist(monkeypatch):
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--unit-only", "--jobs", "3", "--", "-n", "2"],
    )
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "pytest_args",
    [
        ["test_one.py::test_case"],
        ["tests/dispatch"],
        ["--junitxml=report.xml"],
        ["--log-file=combined.log"],
        ["--basetemp", "shared-temp"],
        ["--maxfail", "1"],
    ],
)
def test_file_workers_reject_arguments_with_suite_wide_semantics(
    pytest_args, monkeypatch
):
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--unit-only", "--jobs", "3", "--", *pytest_args],
    )
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


def test_source_runner_rejects_negative_jobs(monkeypatch):
    monkeypatch.setattr(
        run_all.sys, "argv", ["run_all.py", "--unit-only", "--jobs", "-1"]
    )
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


def test_source_runner_stops_before_resource_gates_after_unit_failure(monkeypatch):
    calls = []

    def fail_units(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(run_all.subprocess, "run", fail_units)
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--jobs", "0"])
    assert run_all.main() == 1
    assert len(calls) == 1


def test_source_runner_keep_going_runs_resource_gates_after_unit_failure(monkeypatch):
    calls = []
    results = iter((1, 0, 0))

    def run_all_stages(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=next(results))

    monkeypatch.setattr(run_all.subprocess, "run", run_all_stages)
    monkeypatch.setattr(
        run_all.sys, "argv", ["run_all.py", "--keep-going", "--jobs", "0"]
    )
    assert run_all.main() == 1
    assert len(calls) == 3


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
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--jobs", "0"])
    assert run_all.main() == 0
    assert "RESULT: INCOMPLETE" in capsys.readouterr().out
