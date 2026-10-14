# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import types

import pytest
import yaml

from Tensile import Tensile as TensileModule

pytestmark = pytest.mark.unit


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


def _stub_tensile_pipeline(monkeypatch):
    captured = {}

    monkeypatch.setattr(TensileModule, "validateToolchain", lambda *args: ("cxx", "cc", "bundler"))
    monkeypatch.setattr(
        TensileModule,
        "makeAssemblyToolchain",
        lambda *args, **kwargs: types.SimpleNamespace(assembler="assembler"),
    )
    monkeypatch.setattr(
        TensileModule,
        "makeSourceToolchain",
        lambda *args, **kwargs: types.SimpleNamespace(compiler="compiler"),
    )
    monkeypatch.setattr(
        TensileModule,
        "makeIsaInfoMap",
        lambda isa_list, _compiler: {isa_list[0]: types.SimpleNamespace()},
    )
    monkeypatch.setattr(TensileModule, "assignGlobalParameters", lambda *args, **kwargs: None)
    monkeypatch.setattr(TensileModule, "argUpdatedGlobalParameters", lambda _args: {})
    monkeypatch.setattr(
        TensileModule,
        "makeDebugConfig",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=False,
            printIndexAssignmentInfo=False,
        ),
    )
    monkeypatch.setattr(
        TensileModule,
        "executeStepsInConfig",
        lambda config, *args, **kwargs: captured.setdefault("config", config),
    )

    return captured


def test_yaml_backend_is_normalized_and_preserved(monkeypatch, tmp_path):
    captured = _stub_tensile_pipeline(monkeypatch)
    config_path = _write_config(
        tmp_path,
        _base_config(backend={"Name": "Ductile", "Config": {"seed": 11, "n_gen": 2}}),
    )

    TensileModule.Tensile([config_path, str(tmp_path / "output")])

    assert captured["config"]["Backend"] == {"Name": "ductile", "Config": {"seed": 11, "n_gen": 2}}


def test_invalid_backend_type_exits(monkeypatch, tmp_path):
    _stub_tensile_pipeline(monkeypatch)
    # Monkeypatch printExit in both TensileModule and backends.config modules
    exit_func = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    monkeypatch.setattr(TensileModule, "printExit", exit_func)
    from Tensile.backends import config as backend_config_module
    monkeypatch.setattr(backend_config_module, "printExit", exit_func)
    config_path = _write_config(tmp_path, _base_config(backend="ductile"))

    with pytest.raises(RuntimeError, match="Invalid backend configuration"):
        TensileModule.Tensile([config_path, str(tmp_path / "output")])
