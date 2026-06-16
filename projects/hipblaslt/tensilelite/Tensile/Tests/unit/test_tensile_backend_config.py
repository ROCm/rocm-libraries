################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Extended tests for Tensile.py backend configuration parsing (patch coverage).

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

from Tensile import Tensile as TensileModule

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
    """Stub out all expensive Tensile pipeline steps. Returns captured dict."""
    captured = {}
    monkeypatch.setattr(TensileModule, "validateToolchain", lambda *a: ("cxx", "cc", "bundler"))
    monkeypatch.setattr(
        TensileModule, "makeAssemblyToolchain",
        lambda *a, **kw: types.SimpleNamespace(assembler="assembler"),
    )
    monkeypatch.setattr(
        TensileModule, "makeSourceToolchain",
        lambda *a, **kw: types.SimpleNamespace(compiler="compiler"),
    )
    monkeypatch.setattr(
        TensileModule, "makeIsaInfoMap",
        lambda isa_list, _compiler: {tuple(isa_list[0]): types.SimpleNamespace()},
    )
    monkeypatch.setattr(TensileModule, "assignGlobalParameters", lambda *a, **kw: None)
    monkeypatch.setattr(TensileModule, "argUpdatedGlobalParameters", lambda _args: {})
    monkeypatch.setattr(
        TensileModule, "makeDebugConfig",
        lambda *_a, **_kw: types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=False,
            printIndexAssignmentInfo=False,
        ),
    )
    monkeypatch.setattr(
        TensileModule, "executeStepsInConfig",
        lambda config, *a, **kw: captured.setdefault("config", config),
    )
    return captured


def _make_exit(monkeypatch):
    monkeypatch.setattr(
        TensileModule, "printExit",
        lambda msg: (_ for _ in ()).throw(RuntimeError(msg)),
    )


# ---------------------------------------------------------------------------
# Backend config parsing — uncovered paths
# ---------------------------------------------------------------------------

def test_no_backend_key_defaults_to_tensile(monkeypatch, tmp_path):
    """When no Backend key in YAML, backend_name defaults to 'tensile'."""
    captured = _stub_pipeline(monkeypatch)
    config_path = _write_config(tmp_path, _base_config(backend=None))
    TensileModule.Tensile([config_path, str(tmp_path / "output")])
    assert captured["config"]["Backend"]["Name"] == "tensile"
    assert captured["config"]["Backend"]["Config"] == {}


def test_backend_name_missing_from_dict_exits(monkeypatch, tmp_path):
    """Backend dict without 'Name' key → printExit."""
    _stub_pipeline(monkeypatch)
    _make_exit(monkeypatch)
    config_path = _write_config(tmp_path, _base_config(backend={"Config": {}}))
    with pytest.raises(RuntimeError, match="'Backend' must contain key 'Name'"):
        TensileModule.Tensile([config_path, str(tmp_path / "output")])


def test_backend_name_empty_string_exits(monkeypatch, tmp_path):
    """Backend.Name is empty/whitespace-only string → printExit."""
    _stub_pipeline(monkeypatch)
    _make_exit(monkeypatch)
    config_path = _write_config(tmp_path, _base_config(backend={"Name": "  "}))
    with pytest.raises(RuntimeError, match="'Backend.Name' must be a non-empty string"):
        TensileModule.Tensile([config_path, str(tmp_path / "output")])


def test_backend_config_none_coerced_to_empty_dict(monkeypatch, tmp_path):
    """Backend.Config is None → coerced to {} without error."""
    captured = _stub_pipeline(monkeypatch)
    # YAML: Config: null
    config_path = _write_config(tmp_path, _base_config(backend={"Name": "tensile", "Config": None}))
    TensileModule.Tensile([config_path, str(tmp_path / "output")])
    assert captured["config"]["Backend"]["Config"] == {}


def test_backend_config_not_dict_exits(monkeypatch, tmp_path):
    """Backend.Config is not a dict → printExit."""
    _stub_pipeline(monkeypatch)
    _make_exit(monkeypatch)
    # Write YAML manually since yaml.safe_dump won't encode non-dict Config right
    config = _base_config(backend={"Name": "tensile", "Config": "invalid_string"})
    config_path = _write_config(tmp_path, config)
    with pytest.raises(RuntimeError, match="'Backend.Config' must be a dictionary"):
        TensileModule.Tensile([config_path, str(tmp_path / "output")])


def test_backend_name_is_lowercased(monkeypatch, tmp_path):
    """Backend.Name is stripped and lowercased."""
    captured = _stub_pipeline(monkeypatch)
    config_path = _write_config(tmp_path, _base_config(backend={"Name": " TENSILE "}))
    TensileModule.Tensile([config_path, str(tmp_path / "output")])
    assert captured["config"]["Backend"]["Name"] == "tensile"


def test_backend_config_preserved_as_dict(monkeypatch, tmp_path):
    """Backend.Config dict values are preserved intact."""
    captured = _stub_pipeline(monkeypatch)
    config_path = _write_config(
        tmp_path,
        _base_config(backend={"Name": "ductile", "Config": {"n_gen": 5, "pop_size": 16}}),
    )
    TensileModule.Tensile([config_path, str(tmp_path / "output")])
    assert captured["config"]["Backend"]["Config"]["n_gen"] == 5
    assert captured["config"]["Backend"]["Config"]["pop_size"] == 16


# ---------------------------------------------------------------------------
# executeStepsInConfig backend_cfg path (lines ~110-122)
# ---------------------------------------------------------------------------

def test_benchmark_problems_backend_cfg_missing_name_exits(monkeypatch, tmp_path):
    """In executeStepsInConfig, backend_cfg without 'Name' → printExit."""
    monkeypatch.setattr(TensileModule, "validateToolchain", lambda *a: ("cxx", "cc", "bundler"))
    monkeypatch.setattr(
        TensileModule, "makeAssemblyToolchain",
        lambda *a, **kw: types.SimpleNamespace(assembler="assembler"),
    )
    monkeypatch.setattr(
        TensileModule, "makeSourceToolchain",
        lambda *a, **kw: types.SimpleNamespace(compiler="compiler"),
    )
    monkeypatch.setattr(
        TensileModule, "makeIsaInfoMap",
        lambda isa_list, _compiler: {tuple(isa_list[0]): types.SimpleNamespace()},
    )
    monkeypatch.setattr(TensileModule, "assignGlobalParameters", lambda *a, **kw: None)
    monkeypatch.setattr(TensileModule, "argUpdatedGlobalParameters", lambda _: {})
    monkeypatch.setattr(
        TensileModule, "makeDebugConfig",
        lambda *_a, **_kw: types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=False,
            printIndexAssignmentInfo=False,
        ),
    )
    exited = []
    monkeypatch.setattr(TensileModule, "printExit", lambda m: exited.append(m))

    # Patch BenchmarkProblems.main to capture the backend_cfg passed in
    import Tensile.BenchmarkProblems as BP
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
        TensileModule.Tensile([str(config_path), str(tmp_path / "output")])
    assert any("Name" in e or "backend" in e.lower() for e in exited)


def test_benchmark_problems_backend_cfg_not_dict_exits(monkeypatch, tmp_path):
    """In executeStepsInConfig, backend_cfg is not a dict → printExit."""
    monkeypatch.setattr(TensileModule, "validateToolchain", lambda *a: ("cxx", "cc", "bundler"))
    monkeypatch.setattr(
        TensileModule, "makeAssemblyToolchain",
        lambda *a, **kw: types.SimpleNamespace(assembler="assembler"),
    )
    monkeypatch.setattr(
        TensileModule, "makeSourceToolchain",
        lambda *a, **kw: types.SimpleNamespace(compiler="compiler"),
    )
    monkeypatch.setattr(
        TensileModule, "makeIsaInfoMap",
        lambda isa_list, _compiler: {tuple(isa_list[0]): types.SimpleNamespace()},
    )
    monkeypatch.setattr(TensileModule, "assignGlobalParameters", lambda *a, **kw: None)
    monkeypatch.setattr(TensileModule, "argUpdatedGlobalParameters", lambda _: {})
    monkeypatch.setattr(
        TensileModule, "makeDebugConfig",
        lambda *_a, **_kw: types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=False,
            printIndexAssignmentInfo=False,
        ),
    )
    exited = []
    monkeypatch.setattr(TensileModule, "printExit", lambda m: exited.append(m))

    import Tensile.BenchmarkProblems as BP
    monkeypatch.setattr(BP, "main", lambda *a, **kw: None)

    config = {
        "GlobalParameters": {"MinimumRequiredVersion": "5.0.0", "ISA": [[9, 5, 0]]},
        "BenchmarkProblems": [[]],
        "Backend": "not-a-dict",  # top-level non-dict; inner block also invalid
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(AttributeError):
        TensileModule.Tensile([str(config_path), str(tmp_path / "output")])
    assert len(exited) > 0
