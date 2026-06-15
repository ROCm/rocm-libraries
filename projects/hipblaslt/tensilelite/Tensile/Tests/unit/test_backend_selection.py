################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

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
    monkeypatch.setattr(TensileModule, "printExit", lambda msg: (_ for _ in ()).throw(RuntimeError(msg)))
    config_path = _write_config(tmp_path, _base_config(backend="ductile"))

    with pytest.raises(RuntimeError, match="Invalid backend configuration"):
        TensileModule.Tensile([config_path, str(tmp_path / "output")])
