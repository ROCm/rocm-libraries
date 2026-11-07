# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts separating discovered unit tests from explicit resource gates."""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import rocke
from rocke.tests.subprocess_launcher import (
    rocke_python_command,
    rocke_subprocess_env,
)

from . import _file_manifest_plugin as manifest_plugin
from . import _worker_process as worker_process
from . import run_all
from ._affordable_lane import EXHAUSTIVE_MODULES, mark_exhaustive_items
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


def _process_is_dead(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"
    except FileNotFoundError:
        return True


def _completed(returncode, kwargs):
    environment = kwargs.get("env") or {}
    if returncode == 0 and (output := environment.get("ROCKE_EXPLICIT_GATE_SENTINEL")):
        Path(output).write_text(json.dumps({"collected": 1, "exitstatus": 0}))
    return SimpleNamespace(returncode=returncode)


def test_source_runner_bootstraps_rocke_without_pythonpath():
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [run_all.sys.executable, "-E", str(Path(run_all.__file__)), "--help"],
        cwd=run_all.TESTS.parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rocKE library validation runner" in result.stdout


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
        ["-o", "python_functions=test_gate_manifest_is_stable"],
        ["-c", "custom.ini"],
        ["-p", "third_party_selector"],
        ["gate.py::test_one_row"],
    ],
)
def test_gate_selection_arguments_are_partial(arguments):
    assert selection_is_partial(arguments)
    assert not selection_is_partial(["-q", "-n", "4", "--durations=10"])
    assert not selection_is_partial(["-p", "no:cacheprovider"])
    assert not selection_is_partial(["-p", "pytest_timeout"])


@pytest.mark.parametrize("gate", tuple(_EXPECTED_GATES))
@pytest.mark.parametrize("windows_path", (False, True))
def test_direct_gate_file_is_not_treated_as_partial_selection(gate, windows_path):
    path = str(_EXPECTED_GATES[gate])
    if windows_path:
        path = path.replace("/", "\\")
    complete = SimpleNamespace(
        invocation_params=SimpleNamespace(args=(path, "-v", "-rs", "--timeout=60")),
        option=SimpleNamespace(),
        args=[path],
    )
    selected_node = SimpleNamespace(
        invocation_params=SimpleNamespace(args=(f"{path}::test_one_row",)),
        option=SimpleNamespace(),
        args=[f"{path}::test_one_row"],
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


def test_complete_gate_rejects_ini_overrides(monkeypatch):
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        [
            "run_all.py",
            "--gate",
            "counted-structural",
            "--",
            "-o",
            "python_functions=test_gate_manifest_is_stable",
        ],
    )
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


def test_gate_module_requires_loaded_preflight_conftest():
    with pytest.raises(RuntimeError, match="conftest enabled"):
        require_explicit_gate_entrypoint({"ROCKE_EXPLICIT_GATE_RUN": "1"})


@pytest.mark.worker_integration
@pytest.mark.parametrize(
    "addopts",
    (
        "-k selected_name",
        "--lf",
        "-o python_functions=test_selected_name",
    ),
)
def test_inherited_pytest_addopts_cannot_filter_direct_gate(addopts, tmp_path):
    gate = tmp_path / _EXPECTED_GATES["counted-structural"].name
    gate.write_text("def test_selected_name():\n    pass\n", encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "PYTEST_ADDOPTS": addopts,
            "ROCKE_EXPLICIT_GATE_RUN": "1",
            "ROCKE_EXPLICIT_GATE_PARTIAL": "0",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
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
    assert "PYTEST_PLUGINS" not in required
    assert "PYTEST_PLUGINS" not in optional
    assert required["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert optional["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
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
        ("llvm24", "llvm24", "", frozenset(("gfx942",)), "provenance"),
        ("llvm24", "llvm24", "rocm10.3", frozenset(("gfx942",)), "provenance"),
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
            args=[str(path)],
        ),
        items=[SimpleNamespace(path=path)],
    )
    monkeypatch.setenv("ROCKE_TEST_SHARD", "0/2")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("ROCKE_EXPLICIT_GATE_PARTIAL", "0")
    with pytest.raises(pytest.exit.Exception, match="sharding needs explicit partial"):
        gate_config.pytest_collection_finish(session)

    monkeypatch.setenv("ROCKE_EXPLICIT_GATE_PARTIAL", "1")
    gate_config.pytest_collection_finish(session)

    monkeypatch.delenv("ROCKE_TEST_SHARD")
    monkeypatch.setenv("ROCKE_EXPLICIT_GATE_PARTIAL", "0")
    session.config._rocke_gate_deselected = 1
    with pytest.raises(pytest.exit.Exception, match="item deselection"):
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


def test_affordable_lane_marks_only_declared_modules(tmp_path):
    for filename in EXHAUSTIVE_MODULES:
        assert Path(__file__).with_name(filename).is_file(), filename

    class Item:
        def __init__(self, name):
            self.path = tmp_path / name
            self.markers = []

        def add_marker(self, marker):
            self.markers.append(marker)

    exhaustive = Item(next(iter(EXHAUSTIVE_MODULES)))
    affordable = Item("test_fast_contract.py")
    mark_exhaustive_items([exhaustive, affordable])
    assert exhaustive.markers == ["exhaustive"]
    assert affordable.markers == []


def test_exhaustive_device_tests_remain_in_the_gpu_lane():
    expected = {
        "test_attention_bwd_agpr_probe.py": {
            "test_negative_control_fires_and_the_isa_floor_is_bitwise",
            "test_full_overlap_srcc_chain_needs_no_nop",
            "test_asm_cluster_is_bitwise_the_intrinsic_gemm",
            "test_zero_init_asm_cluster_is_bitwise_the_intrinsic_gemm",
        },
        "test_attention_bwd_xt_probe.py": {
            "test_probe_layout_correct_on_device",
            "test_conflict_counter_matches_predictor_on_device",
        },
    }
    for filename, functions in expected.items():
        tree = ast.parse(Path(__file__).with_name(filename).read_text(encoding="utf-8"))
        definitions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        for name in functions:
            decorators = definitions[name].decorator_list
            assert any(
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "gpu"
                and isinstance(decorator.value, ast.Attribute)
                and decorator.value.attr == "mark"
                for decorator in decorators
            ), f"{filename}::{name} is exhaustive but not gpu-marked"


@pytest.mark.parametrize(
    "arguments, marker",
    [
        (["--pytest-only"], "not gpu and not exhaustive and not worker_integration"),
        (["--pytest-only", "--exhaustive"], "not gpu"),
        (
            ["--pytest-only", "--gpu"],
            "gpu or (not exhaustive and not worker_integration)",
        ),
        (["--gpu-only"], "gpu"),
        (["--gpu-only", "--exhaustive"], "gpu"),
    ],
)
def test_affordable_lane_selects_the_expected_resource_class(
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


@pytest.mark.parametrize(
    "addopts, expected",
    (("-k smoke", ["-k", "smoke"]), ("--collect-only", ["--collect-only"])),
)
def test_inherited_pytest_selection_is_explicit_and_reports_partial(
    addopts, expected, monkeypatch, capsys
):
    calls = []
    monkeypatch.setenv("PYTEST_ADDOPTS", addopts)
    monkeypatch.setattr(
        run_all,
        "_run_affordable_file_stage",
        lambda **kwargs: calls.append(kwargs) or run_all._StageResult(0),
    )
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--pytest-only"])
    assert run_all.main() == 0
    assert calls[0]["extra"] == expected
    assert "PYTEST_ADDOPTS" not in run_all._pytest_environment()
    assert "RESULT: PARTIAL" in capsys.readouterr().out


@pytest.mark.parametrize(
    "option,value",
    (
        ("--rocke-reference-arch", "gfx942"),
        ("--rocke-reference-operation", "sdpa"),
        ("--rocke-reference-variant", "dense"),
        ("--gdn-batch", "1"),
        ("--gdn-spec-id", "example"),
        ("--rocke-seam-arches", "gfx942"),
    ),
)
def test_project_selection_options_report_partial(option, value, monkeypatch, capsys):
    monkeypatch.setattr(
        run_all,
        "_run_affordable_file_stage",
        lambda **kwargs: run_all._StageResult(0),
    )
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--pytest-only", "--", option, value],
    )
    assert run_all.main() == 0
    assert "RESULT: PARTIAL" in capsys.readouterr().out


def test_inherited_shard_requires_single_process(monkeypatch):
    monkeypatch.setenv("ROCKE_TEST_SHARD", "0/2")
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--pytest-only"])
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


def test_inherited_shard_reports_partial_in_single_process(monkeypatch, capsys):
    monkeypatch.setenv("ROCKE_TEST_SHARD", "0/2")
    monkeypatch.setattr(
        run_all.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--pytest-only", "--jobs", "0"],
    )
    assert run_all.main() == 0
    assert "RESULT: PARTIAL" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    (
        ["--pytest-only", "--", "-m", "gpu"],
        ["--gpu", "--", "--markexpr=not gpu"],
    ),
)
def test_forwarded_markers_cannot_replace_lane_selection(arguments, monkeypatch):
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", *arguments])
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


def test_positive_jobs_use_file_workers_and_accept_plugin_values(monkeypatch):
    calls = []

    def pass_files(**kwargs):
        calls.append(kwargs)
        return run_all._StageResult(0)

    monkeypatch.setattr(run_all, "_run_affordable_file_stage", pass_files)
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        [
            "run_all.py",
            "--pytest-only",
            "--jobs",
            "3",
            "--",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
    )
    assert run_all.main() == 0
    assert calls == [
        {
            "marker": "not gpu and not exhaustive and not worker_integration",
            "only": "",
            "extra": ["-q", "-p", "no:cacheprovider"],
            "jobs": 3,
            "timeout": 60.0,
            "lane_timeout": 300.0,
        }
    ]


def test_host_default_uses_effective_cpu_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(run_all, "available_cpus", lambda: 3)
    monkeypatch.setattr(
        run_all,
        "_run_affordable_file_stage",
        lambda **kwargs: calls.append(kwargs) or run_all._StageResult(0),
    )
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--pytest-only"])
    assert run_all.main() == 0
    assert calls[0]["jobs"] == 3


def test_gpu_selection_defaults_to_one_long_running_worker(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_all,
        "_run_affordable_file_stage",
        lambda **kwargs: calls.append(kwargs) or run_all._StageResult(0),
    )
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--gpu-only"])
    assert run_all.main() == 0
    assert calls[0]["jobs"] == 1
    assert calls[0]["timeout"] == 600.0


def test_file_manifest_records_files_and_rejects_positional_selection(tmp_path):
    output = tmp_path / "manifest.txt"
    root = tmp_path / "tests"
    root.mkdir()
    options = {
        "--rocke-selected-file-manifest": output,
        "--rocke-expected-test-root": root,
    }
    config = SimpleNamespace(
        getoption=lambda option: options[option],
        args=[str(root)],
    )
    session = SimpleNamespace(
        config=config,
        items=[
            SimpleNamespace(path=root / "a.py"),
            SimpleNamespace(path=root / "a.py"),
            SimpleNamespace(path=root / "b.py"),
        ],
    )
    manifest_plugin.pytest_collection_finish(session)
    assert output.read_text().splitlines() == [str(root / "a.py"), str(root / "b.py")]

    config.args.append(str(root / "a.py") + "::test_one")
    with pytest.raises(pytest.exit.Exception, match="positional pytest"):
        manifest_plugin.pytest_collection_finish(session)


def test_fresh_file_timeout_names_the_file_and_uses_bounded_runner(
    tmp_path, monkeypatch
):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 3)

    monkeypatch.setattr(run_all, "run_bounded_to_stream", timeout)
    test_file = tmp_path / "test_slow.py"
    log = tmp_path / "slow.log"
    result = run_all._run_pytest_file(test_file, None, "", [], 3, log)
    assert result.code == 124
    assert result.timed_out
    assert result.path == test_file
    assert "TIMEOUT: exceeded 3s" in log.read_text()


@pytest.mark.worker_integration
def test_affordable_collection_excludes_worker_integration(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_fast.py").write_text("def test_fast():\n    pass\n")
    (suite / "test_worker.py").write_text(
        "import pytest\n@pytest.mark.worker_integration\ndef test_worker():\n    pass\n"
    )
    monkeypatch.setattr(run_all, "TESTS", suite)
    code, files, diagnostic = run_all._collect_pytest_files(
        "not gpu and not exhaustive and not worker_integration",
        "",
        ["-q"],
        tmp_path / "manifest.txt",
        20,
    )
    assert code == 0, diagnostic
    assert [path.name for path in files] == ["test_fast.py"]


@pytest.mark.worker_integration
def test_file_lane_reports_all_files_and_fails_on_one_failure(
    tmp_path, monkeypatch, capsys
):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_pass.py").write_text("def test_pass():\n    pass\n")
    (suite / "test_fail.py").write_text(
        "def test_fail():\n    assert False, 'expected failure'\n"
    )
    monkeypatch.setattr(run_all, "TESTS", suite)
    result = run_all._run_affordable_file_stage(
        marker=None,
        only="",
        extra=["-q"],
        jobs=2,
        timeout=10,
        lane_timeout=30,
    )
    output = capsys.readouterr().out
    assert result.code == 1
    assert "test_pass.py" in output
    assert "test_fail.py" in output
    assert str(suite / "test_pass.py") not in output
    assert str(suite / "test_fail.py") not in output
    assert "1 passed, 1 failed" in output


@pytest.mark.worker_integration
def test_worker_child_imports_the_exact_parent_rocke_package(tmp_path):
    expected = Path(rocke.__file__).resolve()
    test_file = tmp_path / "test_package_identity.py"
    test_file.write_text(
        "from pathlib import Path\n"
        "import rocke\n"
        "def test_package_identity():\n"
        f"    assert Path(rocke.__file__).resolve() == Path({str(expected)!r})\n"
    )
    result = run_all._run_pytest_file(
        test_file, None, "", ["-q"], 10, tmp_path / "package-identity.log"
    )
    assert result.code == 0, result.log.read_text(errors="replace")


@pytest.mark.worker_integration
@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process-group contract")
def test_worker_normal_exit_cleans_up_owned_group_children(tmp_path):
    child_pid = tmp_path / "child.pid"
    test_file = tmp_path / "test_child.py"
    test_file.write_text(
        "import pathlib, subprocess, sys\n"
        "def test_child():\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"    pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
    )
    result = run_all._run_pytest_file(
        test_file, None, "", ["-q"], 10, tmp_path / "normal-exit.log"
    )
    assert result.code == 0
    pid = int(child_pid.read_text())
    for _ in range(100):
        if _process_is_dead(pid):
            break
        time.sleep(0.02)
    assert _process_is_dead(pid)


def test_affordable_lane_collection_timeout_starts_no_workers(monkeypatch, capsys):
    workers = []

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(run_all, "run_bounded_to_stream", timeout)
    monkeypatch.setattr(
        run_all, "_run_pytest_file", lambda *args, **kwargs: workers.append(args)
    )
    result = run_all._run_affordable_file_stage(
        marker=None,
        only="",
        extra=[],
        jobs=2,
        timeout=60,
        lane_timeout=0.1,
    )
    assert result.code == 124
    assert not workers
    assert "collection exceeded" in capsys.readouterr().out


@pytest.mark.worker_integration
@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_windows_job_close_kills_worker_descendants(tmp_path):
    child_pid = tmp_path / "child.pid"
    test_file = tmp_path / "test_child.py"
    test_file.write_text(
        "import pathlib, subprocess, sys\n"
        "def test_child():\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"    pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
    )
    result = run_all._run_pytest_file(
        test_file, None, "", ["-q"], 10, tmp_path / "windows-normal-exit.log"
    )
    assert result.code == 0
    pid = child_pid.read_text().strip()
    for _ in range(50):
        listing = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if pid not in listing:
            break
        time.sleep(0.02)
    assert pid not in listing


def test_affordable_lane_enforces_whole_stage_deadline(tmp_path, monkeypatch, capsys):
    files = [tmp_path / "a.py", tmp_path / "b.py"]
    monkeypatch.setattr(
        run_all,
        "_collect_pytest_files",
        lambda *args, **kwargs: (0, files, ""),
    )

    def slow_file(path, marker, only, extra, timeout, log):
        time.sleep(0.1)
        return run_all._FileResult(path, 0, 0.1, log)

    monkeypatch.setattr(run_all, "_run_pytest_file", slow_file)
    try:
        result = run_all._run_affordable_file_stage(
            marker=None,
            only="",
            extra=[],
            jobs=1,
            timeout=60,
            lane_timeout=0.02,
        )
    finally:
        worker_process._stopped = False
    assert result.code == 124
    assert "affordable-lane timeout" in capsys.readouterr().out


def test_available_cpus_honors_nested_v2_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(
        worker_process.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}
    )
    monkeypatch.setattr(worker_process.os, "cpu_count", lambda: 64)
    member = tmp_path / "team" / "job"
    member.mkdir(parents=True)
    (tmp_path / "cpu.max").write_text("max 100000")
    (member.parent / "cpu.max").write_text("300000 100000")
    (member / "cpu.max").write_text("150000 100000")
    proc = tmp_path / "self.cgroup"
    proc.write_text("0::/team/job\n")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"1 2 0:1 / {tmp_path} rw - cgroup2 cgroup rw\n")
    assert worker_process.available_cpus(tmp_path, proc, mountinfo) == 2


def test_available_cpus_honors_nested_v1_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(
        worker_process.os, "sched_getaffinity", lambda _pid: set(range(8))
    )
    monkeypatch.setattr(worker_process.os, "cpu_count", lambda: 64)
    mount = tmp_path / "cpu,cpuacct"
    member = mount / "batch" / "job"
    member.mkdir(parents=True)
    (member.parent / "cpu.cfs_quota_us").write_text("100000")
    (member.parent / "cpu.cfs_period_us").write_text("100000")
    (member / "cpu.cfs_quota_us").write_text("800000")
    (member / "cpu.cfs_period_us").write_text("100000")
    proc = tmp_path / "self.cgroup"
    proc.write_text("2:cpu,cpuacct:/batch/job\n")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"1 2 0:1 / {mount} rw - cgroup cgroup rw,cpu,cpuacct\n")
    assert worker_process.available_cpus(tmp_path, proc, mountinfo) == 1


@pytest.mark.parametrize("version", (1, 2))
def test_available_cpus_rebases_non_root_cgroup_mount(version, tmp_path, monkeypatch):
    monkeypatch.setattr(
        worker_process.os, "sched_getaffinity", lambda _pid: set(range(8))
    )
    monkeypatch.setattr(worker_process.os, "cpu_count", lambda: 64)
    mount = tmp_path / "mounted"
    member = mount / "team" / "job"
    member.mkdir(parents=True)
    if version == 2:
        (member / "cpu.max").write_text("150000 100000")
        cgroup_line = "0::/tenant.slice/team/job\n"
        mount_line = f"1 2 0:1 /tenant.slice {mount} rw - cgroup2 cgroup rw\n"
    else:
        (member / "cpu.cfs_quota_us").write_text("150000")
        (member / "cpu.cfs_period_us").write_text("100000")
        cgroup_line = "2:cpu,cpuacct:/tenant.slice/team/job\n"
        mount_line = (
            f"1 2 0:1 /tenant.slice {mount} rw - cgroup cgroup rw,cpu,cpuacct\n"
        )
    proc = tmp_path / "self.cgroup"
    proc.write_text(cgroup_line)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(mount_line)
    assert worker_process.available_cpus(tmp_path, proc, mountinfo) == 2


def test_available_cpus_honors_affinity_without_a_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_process.os, "sched_getaffinity", lambda _pid: {2, 3})
    monkeypatch.setattr(worker_process.os, "cpu_count", lambda: 64)
    assert (
        worker_process.available_cpus(
            tmp_path, tmp_path / "missing-cgroup", tmp_path / "missing-mountinfo"
        )
        == 2
    )


def test_sigterm_during_spawn_registers_then_kills_the_new_process(
    tmp_path, monkeypatch
):
    class Process:
        pid = 123

    process = Process()
    killed = []

    def spawn(*args, **kwargs):
        run_all._terminate_workers(signal.SIGTERM, None)
        return process

    monkeypatch.setattr(worker_process.subprocess, "Popen", spawn)
    monkeypatch.setattr(worker_process, "_kill_tree", killed.append)
    try:
        with (tmp_path / "spawn.log").open("w") as stream:
            with pytest.raises(KeyboardInterrupt, match="stopped during spawn"):
                worker_process.run_bounded_to_stream(
                    ["python"], timeout=1, cwd=tmp_path, stream=stream
                )
    finally:
        worker_process._stopped = False
        worker_process._spawning = 0
        worker_process._running.clear()
    assert killed == [process]


def test_windows_worker_is_assigned_before_wrapper_release(tmp_path, monkeypatch):
    events = []

    class Process:
        pid = 123
        _handle = 456

        def wait(self, timeout=None):
            events.append("wait")
            return 0

        def kill(self):
            events.append("kill")

    process = Process()

    def spawn(command, **kwargs):
        events.append("spawn")
        assert command[1].endswith("_windows_worker_wrapper.py")
        assert command[-1] == "payload"
        return process

    monkeypatch.setattr(worker_process, "_is_windows", lambda: True)
    monkeypatch.setattr(
        worker_process, "_create_windows_job", lambda: events.append("create") or 77
    )
    monkeypatch.setattr(
        worker_process,
        "_assign_windows_job",
        lambda job, child: events.append("assign") or True,
    )
    monkeypatch.setattr(
        worker_process, "_close_windows_job", lambda job: events.append("close")
    )
    monkeypatch.setattr(
        worker_process,
        "_release_windows_gate",
        lambda path: events.append("release"),
    )
    monkeypatch.setattr(worker_process.subprocess, "Popen", spawn)
    with (tmp_path / "windows.log").open("w") as stream:
        result = worker_process.run_bounded_to_stream(
            ["payload"], timeout=1, cwd=tmp_path, stream=stream
        )
    assert result.returncode == 0
    assert events == [
        "create",
        "spawn",
        "assign",
        "release",
        "wait",
        "close",
        "kill",
    ]


def test_windows_job_assignment_failure_fails_before_payload_release(
    tmp_path, monkeypatch
):
    class Process:
        pid = 123
        _handle = 456

        def kill(self):
            pass

    monkeypatch.setattr(worker_process, "_is_windows", lambda: True)
    monkeypatch.setattr(worker_process, "_create_windows_job", lambda: 77)
    monkeypatch.setattr(worker_process, "_assign_windows_job", lambda job, child: False)
    monkeypatch.setattr(worker_process, "_close_windows_job", lambda job: None)
    monkeypatch.setattr(
        worker_process.subprocess, "Popen", lambda *args, **kwargs: Process()
    )
    with (tmp_path / "windows-fail.log").open("w") as stream:
        with pytest.raises(OSError, match="could not assign"):
            worker_process.run_bounded_to_stream(
                ["payload"], timeout=1, cwd=tmp_path, stream=stream
            )


def test_windows_gate_release_failure_closes_the_owned_job(tmp_path, monkeypatch):
    class Process:
        pid = 123
        _handle = 456

        def kill(self):
            pass

    process = Process()
    closed = []
    monkeypatch.setattr(worker_process, "_is_windows", lambda: True)
    monkeypatch.setattr(worker_process, "_create_windows_job", lambda: 77)
    monkeypatch.setattr(worker_process, "_assign_windows_job", lambda job, child: True)
    monkeypatch.setattr(worker_process, "_close_windows_job", closed.append)
    monkeypatch.setattr(
        worker_process,
        "_release_windows_gate",
        lambda path: (_ for _ in ()).throw(OSError("release failed")),
    )
    monkeypatch.setattr(
        worker_process.subprocess, "Popen", lambda *args, **kwargs: process
    )
    with (tmp_path / "windows-release.log").open("w") as stream:
        with pytest.raises(OSError, match="release failed"):
            worker_process.run_bounded_to_stream(
                ["payload"], timeout=1, cwd=tmp_path, stream=stream
            )
    assert closed == [77]
    assert process not in worker_process._running
    assert process not in worker_process._windows_jobs


def test_windows_cleanup_closes_the_owned_job_object(monkeypatch):
    class Process:
        pid = 123

        def __init__(self):
            self.killed = False

        def kill(self):
            self.killed = True

    process = Process()
    closed = []
    monkeypatch.setattr(worker_process, "_is_windows", lambda: True)
    monkeypatch.setattr(worker_process, "_close_windows_job", closed.append)
    worker_process._windows_jobs[process] = 77
    worker_process._kill_tree(process)
    assert closed == [77]
    assert process.killed
    assert process not in worker_process._windows_jobs


@pytest.mark.worker_integration
@pytest.mark.skipif(
    not Path("/proc").is_dir(), reason="Linux process-liveness contract"
)
def test_bounded_worker_timeout_kills_owned_group_children(tmp_path):
    child_pid = tmp_path / "child.pid"
    log = tmp_path / "worker.log"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    with log.open("w", encoding="utf-8") as stream:
        with pytest.raises(subprocess.TimeoutExpired):
            worker_process.run_bounded_to_stream(
                [run_all.sys.executable, "-c", code],
                timeout=0.5,
                cwd=tmp_path,
                stream=stream,
            )
    pid = int(child_pid.read_text())
    for _ in range(50):
        if _process_is_dead(pid):
            break
        time.sleep(0.02)
    assert _process_is_dead(pid)


def test_gate_cannot_report_green_without_a_pytest_session(monkeypatch, capsys):
    monkeypatch.setattr(
        run_all.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--gate", "counted-structural", "--", "--help"],
    )
    assert run_all.main() == 1
    assert "FAIL(no executed pytest session)" in capsys.readouterr().out


@pytest.mark.worker_integration
@pytest.mark.skipif(
    not Path("/proc").is_dir(), reason="Linux process-liveness contract"
)
def test_runner_sigterm_kills_owned_group_children(tmp_path):
    child_pid = tmp_path / "child.pid"
    worker = tmp_path / "test_worker.py"
    worker.write_text(
        "import pathlib, subprocess, sys, time\n"
        "def test_worker():\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"    pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "    time.sleep(30)\n"
    )
    driver = (
        "import signal; from pathlib import Path; from tests import run_all; "
        "signal.signal(signal.SIGTERM, run_all._terminate_workers); "
        f"run_all._run_pytest_file(Path({str(worker)!r}), None, '', [], 30, "
        f"Path({str(tmp_path / 'worker.log')!r}))"
    )
    process = subprocess.Popen(
        rocke_python_command("-c", driver),
        cwd=_GATE_DIR.parent.parent,
        env=rocke_subprocess_env(),
    )
    try:
        for _ in range(100):
            if child_pid.is_file():
                break
            time.sleep(0.02)
        assert child_pid.is_file()
        process.terminate()
        assert process.wait(timeout=5) == 143
        pid = int(child_pid.read_text())
        for _ in range(100):
            if _process_is_dead(pid):
                break
            time.sleep(0.02)
        assert _process_is_dead(pid)
    finally:
        if process.poll() is None:
            process.kill()


def test_gate_only_output_does_not_trigger_file_worker_restrictions(monkeypatch):
    monkeypatch.setattr(
        run_all.subprocess,
        "run",
        lambda *args, **kwargs: _completed(0, kwargs),
    )
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        [
            "run_all.py",
            "--gate",
            "counted-structural",
            "--",
            "--junitxml=gate.xml",
        ],
    )
    assert run_all.main() == 0


def test_source_runner_rejects_nested_xdist(monkeypatch):
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--pytest-only", "--jobs", "3", "--", "-n", "2"],
    )
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "pytest_args",
    [
        ["--junitxml=report.xml"],
        ["--log-file=combined.log"],
        ["--basetemp", "shared-temp"],
        ["--maxfail", "1"],
    ],
)
def test_file_workers_reject_suite_wide_semantics(pytest_args, monkeypatch):
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--pytest-only", "--jobs", "3", "--", *pytest_args],
    )
    with pytest.raises(SystemExit) as error:
        run_all.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    (
        ["--junitxml=report.xml"],
        ["--junit-xml", "report.xml"],
        ["--log-file=combined.log"],
        ["--json-report-file=report.json"],
        ["--html=report.html"],
        ["--cov-report=xml:coverage.xml"],
    ),
)
def test_suite_outputs_are_stage_owned(arguments):
    assert _has_stage_output(arguments)


@pytest.mark.parametrize(
    "arguments",
    (
        ["--junitxml=report.xml"],
        ["--log-file=combined.log"],
        ["--json-report-file=report.json"],
        ["--html=report.html"],
        ["--cov-report=xml:coverage.xml"],
    ),
)
def test_source_runner_rejects_shared_stage_output(arguments, monkeypatch):
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--jobs", "0", "--", *arguments],
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
    monkeypatch.setattr(
        run_all.sys, "argv", ["run_all.py", "--only", "dense", "--jobs", "0"]
    )
    assert run_all.main() == 0
    assert "RESULT: PARTIAL" in capsys.readouterr().out


def test_source_runner_stops_before_resource_gates_after_unit_failure(monkeypatch):
    calls = []

    def fail_units(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(run_all.subprocess, "run", fail_units)
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--jobs", "0"])
    assert run_all.main() == 1
    assert len(calls) == 1


def test_source_runner_stops_before_compiler_gate_after_structural_failure(monkeypatch):
    calls = []
    results = iter((0, 1))

    def fail_structural(command, **kwargs):
        calls.append(command)
        return _completed(next(results), kwargs)

    monkeypatch.setattr(run_all.subprocess, "run", fail_structural)
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--jobs", "0"])
    assert run_all.main() == 1
    assert len(calls) == 2
    assert calls[-1][-1] == str(EXPLICIT_GATES["counted-structural"])


def test_source_runner_keep_going_runs_named_stages_in_order(monkeypatch):
    calls = []
    results = iter((1, 0, 0))

    def run_all_stages(command, **kwargs):
        calls.append((command, kwargs.get("env")))
        return _completed(next(results), kwargs)

    monkeypatch.setattr(run_all.subprocess, "run", run_all_stages)
    monkeypatch.setattr(
        run_all.sys, "argv", ["run_all.py", "--keep-going", "--jobs", "0"]
    )
    assert run_all.main() == 1

    unit, structural, compiler = calls
    pytest_at = unit[0].index("pytest")
    assert unit[0][pytest_at - 1 : pytest_at + 2] == [
        "-m",
        "pytest",
        str(Path(run_all.TESTS)),
    ]
    assert "PYTEST_ADDOPTS" not in unit[1]
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
        lambda *args, **kwargs: _completed(next(results), kwargs),
    )
    monkeypatch.setattr(run_all.sys, "argv", ["run_all.py", "--jobs", "0"])
    assert run_all.main() == 0
    assert "RESULT: INCOMPLETE" in capsys.readouterr().out
