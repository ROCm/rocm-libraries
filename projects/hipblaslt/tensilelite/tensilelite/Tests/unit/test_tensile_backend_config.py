# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Extended tests for tensilelite.py backend configuration parsing (patch coverage).

Targets uncovered lines reported by codecov:
- Backend dict without 'Name' key → printExit
- Backend.Name is empty string → printExit  
- Backend.Name is not a string → printExit
- Backend.Config is None → coerced to {}
- Backend.Config is not a dict → printExit
- No Backend key → defaults to 'tensile'
- executeStepsInConfig config path (BenchmarkProblems section)
- BenchmarkProblems backend_cfg path (separate code block at line ~110)
"""

import types

import pytest
import yaml

from tensilelite import tensilelite as tensilelite_module

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_backend_selection.py)
# ---------------------------------------------------------------------------

def _base_config(backend=None):
    config = {
        "GlobalParameters": {
            "MinimumRequiredVersion": "5.0.0",
            "ISA": [[9, 5, 0]],
        },
        "BenchmarkProblems": [],
    }
    if backend is not None:
        config["Backend"] = backend
    return config


def _write_config(tmp_path, config):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(config_path)


def _stub_pipeline(monkeypatch):
    """Stub out all expensive TensileLite pipeline steps. Returns captured dict."""
    captured = {}
    monkeypatch.setattr(tensilelite_module, "validateToolchain", lambda *a: ("cxx", "cc", "bundler"))
    monkeypatch.setattr(
        tensilelite_module, "makeAssemblyToolchain",
        lambda *a, **kw: types.SimpleNamespace(assembler="assembler"),
    )
    monkeypatch.setattr(
        tensilelite_module, "makeSourceToolchain",
        lambda *a, **kw: types.SimpleNamespace(compiler="compiler"),
    )
    monkeypatch.setattr(
        tensilelite_module, "makeIsaInfoMap",
        lambda isa_list, _compiler: {tuple(isa_list[0]): types.SimpleNamespace()},
    )
    monkeypatch.setattr(tensilelite_module, "assignGlobalParameters", lambda *a, **kw: None)
    monkeypatch.setattr(tensilelite_module, "argUpdatedGlobalParameters", lambda _args: {})
    monkeypatch.setattr(
        tensilelite_module, "makeDebugConfig",
        lambda *_a, **_kw: types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=False,
            printIndexAssignmentInfo=False,
        ),
    )
    monkeypatch.setattr(
        tensilelite_module, "executeStepsInConfig",
        lambda config, *a, **kw: captured.setdefault("config", config),
    )
    return captured


def _make_exit(monkeypatch):
    # Monkeypatch printExit in both Tensile module and backends.config module
    exit_func = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    monkeypatch.setattr(tensilelite_module, "printExit", exit_func)
    # Also patch in the backends.config module since parse_backend_config imports it there
    from tensilelite.backends import config as backend_config_module
    monkeypatch.setattr(backend_config_module, "printExit", exit_func)


# ---------------------------------------------------------------------------
# Backend config parsing — uncovered paths
# ---------------------------------------------------------------------------

def test_no_backend_key_defaults_to_tensile(monkeypatch, tmp_path):
    """When no Backend key in YAML, backend_name defaults to 'tensile'."""
    captured = _stub_pipeline(monkeypatch)
    config_path = _write_config(tmp_path, _base_config(backend=None))
    tensilelite_module.tensilelite([config_path, str(tmp_path / "output")])
    assert captured["config"]["Backend"]["Name"] == "tensile"
    assert captured["config"]["Backend"]["Config"] == {}


def test_backend_name_missing_from_dict_exits(monkeypatch, tmp_path):
    """Backend dict without 'Name' key → printExit."""
    _stub_pipeline(monkeypatch)
    _make_exit(monkeypatch)
    config_path = _write_config(tmp_path, _base_config(backend={"Config": {}}))
    with pytest.raises(RuntimeError, match="'Backend' must contain key 'Name'"):
        tensilelite_module.tensilelite([config_path, str(tmp_path / "output")])


def test_backend_name_empty_string_exits(monkeypatch, tmp_path):
    """Backend.Name is empty/whitespace-only string → printExit."""
    _stub_pipeline(monkeypatch)
    _make_exit(monkeypatch)
    config_path = _write_config(tmp_path, _base_config(backend={"Name": "  "}))
    with pytest.raises(RuntimeError, match="'Backend.Name' must be a non-empty string"):
        tensilelite_module.tensilelite([config_path, str(tmp_path / "output")])


def test_backend_config_none_coerced_to_empty_dict(monkeypatch, tmp_path):
    """Backend.Config is None → coerced to {} without error."""
    captured = _stub_pipeline(monkeypatch)
    # YAML: Config: null
    config_path = _write_config(tmp_path, _base_config(backend={"Name": "tensile", "Config": None}))
    tensilelite_module.tensilelite([config_path, str(tmp_path / "output")])
    assert captured["config"]["Backend"]["Config"] == {}


def test_backend_config_not_dict_exits(monkeypatch, tmp_path):
    """Backend.Config is not a dict → printExit."""
    _stub_pipeline(monkeypatch)
    _make_exit(monkeypatch)
    # Write YAML manually since yaml.safe_dump won't encode non-dict Config right
    config = _base_config(backend={"Name": "tensile", "Config": "invalid_string"})
    config_path = _write_config(tmp_path, config)
    with pytest.raises(RuntimeError, match="'Backend.Config' must be a dictionary"):
        tensilelite_module.tensilelite([config_path, str(tmp_path / "output")])


def test_backend_name_is_lowercased(monkeypatch, tmp_path):
    """Backend.Name is stripped and lowercased."""
    captured = _stub_pipeline(monkeypatch)
    config_path = _write_config(tmp_path, _base_config(backend={"Name": " TENSILE "}))
    tensilelite_module.tensilelite([config_path, str(tmp_path / "output")])
    assert captured["config"]["Backend"]["Name"] == "tensile"


def test_backend_config_preserved_as_dict(monkeypatch, tmp_path):
    """Backend.Config dict values are preserved intact."""
    captured = _stub_pipeline(monkeypatch)
    config_path = _write_config(
        tmp_path,
        _base_config(backend={"Name": "ductile", "Config": {"n_gen": 5, "pop_size": 16}}),
    )
    tensilelite_module.tensilelite([config_path, str(tmp_path / "output")])
    assert captured["config"]["Backend"]["Config"]["n_gen"] == 5
    assert captured["config"]["Backend"]["Config"]["pop_size"] == 16


# ---------------------------------------------------------------------------
# executeStepsInConfig backend_cfg path (lines ~110-122)
# ---------------------------------------------------------------------------

def test_benchmark_problems_backend_cfg_missing_name_exits(monkeypatch, tmp_path):
    """In executeStepsInConfig, backend_cfg without 'Name' → printExit."""
    monkeypatch.setattr(tensilelite_module, "validateToolchain", lambda *a: ("cxx", "cc", "bundler"))
    monkeypatch.setattr(
        tensilelite_module, "makeAssemblyToolchain",
        lambda *a, **kw: types.SimpleNamespace(assembler="assembler"),
    )
    monkeypatch.setattr(
        tensilelite_module, "makeSourceToolchain",
        lambda *a, **kw: types.SimpleNamespace(compiler="compiler"),
    )
    monkeypatch.setattr(
        tensilelite_module, "makeIsaInfoMap",
        lambda isa_list, _compiler: {tuple(isa_list[0]): types.SimpleNamespace()},
    )
    monkeypatch.setattr(tensilelite_module, "assignGlobalParameters", lambda *a, **kw: None)
    monkeypatch.setattr(tensilelite_module, "argUpdatedGlobalParameters", lambda _: {})
    monkeypatch.setattr(
        tensilelite_module, "makeDebugConfig",
        lambda *_a, **_kw: types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=False,
            printIndexAssignmentInfo=False,
        ),
    )
    exited = []
    # Monkeypatch printExit to collect messages without raising
    def collect_exit(m):
        exited.append(m)
    monkeypatch.setattr(tensilelite_module, "printExit", collect_exit)
    # Also patch in backends.config module
    from tensilelite.backends import config as backend_config_module
    monkeypatch.setattr(backend_config_module, "printExit", collect_exit)

    # Patch BenchmarkProblems.main to capture the backend_cfg passed in
    import tensilelite.BenchmarkProblems as BP
    monkeypatch.setattr(BP, "main", lambda *a, **kw: None)

    config = {
        "GlobalParameters": {"MinimumRequiredVersion": "5.0.0", "ISA": [[9, 5, 0]]},
        "BenchmarkProblems": [[]],
        "Backend": {"Config": {}},  # 'Name' missing inside executeStepsInConfig's block
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # printExit is monkeypatched to collect messages but not raise, so execution
    # continues and may hit downstream AttributeError. Validate both effects.
    with pytest.raises(AttributeError):
        tensilelite_module.tensilelite([str(config_path), str(tmp_path / "output")])
    assert any("Name" in e or "backend" in e.lower() for e in exited)


def test_benchmark_problems_backend_cfg_not_dict_exits(monkeypatch, tmp_path):
    """In executeStepsInConfig, backend_cfg is not a dict → printExit."""
    monkeypatch.setattr(tensilelite_module, "validateToolchain", lambda *a: ("cxx", "cc", "bundler"))
    monkeypatch.setattr(
        tensilelite_module, "makeAssemblyToolchain",
        lambda *a, **kw: types.SimpleNamespace(assembler="assembler"),
    )
    monkeypatch.setattr(
        tensilelite_module, "makeSourceToolchain",
        lambda *a, **kw: types.SimpleNamespace(compiler="compiler"),
    )
    monkeypatch.setattr(
        tensilelite_module, "makeIsaInfoMap",
        lambda isa_list, _compiler: {tuple(isa_list[0]): types.SimpleNamespace()},
    )
    monkeypatch.setattr(tensilelite_module, "assignGlobalParameters", lambda *a, **kw: None)
    monkeypatch.setattr(tensilelite_module, "argUpdatedGlobalParameters", lambda _: {})
    monkeypatch.setattr(
        tensilelite_module, "makeDebugConfig",
        lambda *_a, **_kw: types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=False,
            printIndexAssignmentInfo=False,
        ),
    )
    exited = []
    # Monkeypatch printExit to collect messages without raising
    def collect_exit(m):
        exited.append(m)
    monkeypatch.setattr(tensilelite_module, "printExit", collect_exit)
    # Also patch in backends.config module
    from tensilelite.backends import config as backend_config_module
    monkeypatch.setattr(backend_config_module, "printExit", collect_exit)

    import tensilelite.BenchmarkProblems as BP
    monkeypatch.setattr(BP, "main", lambda *a, **kw: None)

    config = {
        "GlobalParameters": {"MinimumRequiredVersion": "5.0.0", "ISA": [[9, 5, 0]]},
        "BenchmarkProblems": [[]],
        "Backend": "not-a-dict",  # top-level non-dict; inner block also invalid
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(AttributeError):
        tensilelite_module.tensilelite([str(config_path), str(tmp_path / "output")])
    assert len(exited) > 0


def test_execute_steps_normalizes_default_backend_cfg(monkeypatch, tmp_path):
    captured = {}

    import tensilelite.BenchmarkProblems as BP

    monkeypatch.setattr(
        BP,
        "main",
        lambda backend_cfg, *_a, **_kw: captured.setdefault("backend", backend_cfg),
    )
    monkeypatch.setattr(tensilelite_module.LibraryLogic, "main", lambda *_a, **_kw: None)
    monkeypatch.setattr(tensilelite_module.ClientWriter, "main", lambda *_a, **_kw: None)

    config = {
        "BenchmarkProblems": [],
        "UseCache": False,
    }

    tensilelite_module.executeStepsInConfig(
        config=config,
        outputPath=tmp_path,
        asmToolchain=types.SimpleNamespace(assembler=object()),
        srcToolchain=types.SimpleNamespace(compiler="cc"),
        isaInfoMap={(9, 4, 2): types.SimpleNamespace()},
        cCompiler="cc",
        debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
        deviceId=0,
        probSolDict={},
        buildOnly=True,
        solutionPoolFiles=None,
    )

    assert captured["backend"] == {}


def test_execute_steps_normalizes_missing_name_and_none_config(monkeypatch, tmp_path):
    captured = {}

    import tensilelite.BenchmarkProblems as BP

    monkeypatch.setattr(
        BP,
        "main",
        lambda backend_cfg, *_a, **_kw: captured.setdefault("backend", backend_cfg),
    )
    monkeypatch.setattr(tensilelite_module.LibraryLogic, "main", lambda *_a, **_kw: None)
    monkeypatch.setattr(tensilelite_module.ClientWriter, "main", lambda *_a, **_kw: None)

    config = {
        "BenchmarkProblems": [],
        "UseCache": False,
        "Backend": {"Config": None},
    }

    tensilelite_module.executeStepsInConfig(
        config=config,
        outputPath=tmp_path,
        asmToolchain=types.SimpleNamespace(assembler=object()),
        srcToolchain=types.SimpleNamespace(compiler="cc"),
        isaInfoMap={(9, 4, 2): types.SimpleNamespace()},
        cCompiler="cc",
        debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
        deviceId=0,
        probSolDict={},
        buildOnly=True,
        solutionPoolFiles=None,
    )

    assert captured["backend"] == {"Config": None}


def test_execute_steps_invalid_backend_type_exits(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        tensilelite_module.executeStepsInConfig(
            config={"BenchmarkProblems": [], "UseCache": False, "Backend": "bad"},
            outputPath=tmp_path,
            asmToolchain=types.SimpleNamespace(assembler=object()),
            srcToolchain=types.SimpleNamespace(compiler="cc"),
            isaInfoMap={(9, 4, 2): types.SimpleNamespace()},
            cCompiler="cc",
            debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
            deviceId=0,
            probSolDict={},
            buildOnly=True,
            solutionPoolFiles=None,
        )


def test_execute_steps_invalid_backend_config_type_exits(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        tensilelite_module.executeStepsInConfig(
            config={"BenchmarkProblems": [], "UseCache": False, "Backend": {"Name": "tensile", "Config": "bad"}},
            outputPath=tmp_path,
            asmToolchain=types.SimpleNamespace(assembler=object()),
            srcToolchain=types.SimpleNamespace(compiler="cc"),
            isaInfoMap={(9, 4, 2): types.SimpleNamespace()},
            cCompiler="cc",
            debugConfig=types.SimpleNamespace(splitGSU=False, printSolutionRejectionReason=False, printIndexAssignmentInfo=False),
            deviceId=0,
            probSolDict={},
            buildOnly=True,
            solutionPoolFiles=None,
        )


def test_alternate_format_builds_benchmark_problems(monkeypatch, tmp_path):
    captured = _stub_pipeline(monkeypatch)

    base_cfg = {
        "GlobalParameters": {"MinimumRequiredVersion": "5.0.0", "ISA": [[9, 5, 0]]},
        "ProblemType": {"OperationType": "GEMM", "DataType": "s", "DestDataType": "s"},
        "BenchmarkCommonParameters": [{"KernelLanguage": "Assembly"}],
        "ForkParameters": [{"DepthU": [16]}],
        "GroupForkParameters": [],
    }
    sizes_cfg = [[128, 128, 1, 128]]

    base_path = tmp_path / "base.yaml"
    sizes_path = tmp_path / "sizes.yaml"
    base_path.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    sizes_path.write_text(yaml.safe_dump(sizes_cfg), encoding="utf-8")

    tensilelite_module.tensilelite(
        [
            str(base_path),
            str(sizes_path),
            str(tmp_path / "output"),
            "--alternate-format",
        ]
    )

    benchmark_problems = captured["config"]["BenchmarkProblems"]
    assert benchmark_problems[0][0] == base_cfg["ProblemType"]
    assert benchmark_problems[0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"] == sizes_cfg
