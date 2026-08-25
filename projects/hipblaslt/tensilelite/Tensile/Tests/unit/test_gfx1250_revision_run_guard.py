# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute-time gfx1250 revision guard. detect_gpu_revision_target is mocked."""

from unittest import mock
import types

import pytest

from Tensile import Gfx1250RunGuard as guard

pytestmark = pytest.mark.unit

GFX1250_IIM = {(12, 5, 0): object()}
GFX950_IIM = {(9, 5, 0): object()}
GFX942_IIM = {(9, 4, 2): object()}


def _config(*, asic_revision=None, marks=None, architecture="gfx1250"):
    test_params = {}
    if asic_revision is not None:
        test_params["RevisionID"] = asic_revision
    if marks is not None:
        test_params["marks"] = marks
    return {
        "TestParameters": test_params,
        "GlobalParameters": {"Architecture": architecture},
        "BenchmarkProblems": [],
        "UseCache": False,
    }


def _rev1_config():
    return _config(asic_revision=1)


def _rev0_config():
    return _config()


def _tensile_module():
    try:
        from Tensile import Tensile as TensileModule
        return TensileModule
    except ImportError as exc:
        pytest.skip(f"Tensile pipeline not importable without rocisa: {exc}")


def _guard(*, build_only=False, cpu_only=False, arch_names=None, isa_info_map=None,
           device_id=0, config=None):
    guard.guard_gfx1250_v1_run_on_v0(
        build_only=build_only,
        cpu_only=cpu_only,
        arch_names=arch_names,
        isa_info_map=isa_info_map,
        device_id=device_id,
        config=config,
    )


def _patch_probe(monkeypatch, hw):
    probe = mock.Mock(return_value=hw)
    monkeypatch.setattr(guard, "detect_gpu_revision_target", probe)
    return probe


class TestRevisionIdMetadata:
    def test_missing_field_defaults_to_rev0(self):
        assert guard.config_required_asic_revision(_rev0_config()) == 0
        assert guard.config_required_asic_revision({}) == 0
        assert guard.config_required_asic_revision(None) == 0
        assert guard.config_required_asic_revision(_rev1_config()) == 1
        assert guard.config_required_asic_revision(_config(marks=["skip-gfx1250v0"])) == 1
        assert guard.requires_gfx1250_rev1(_rev1_config())
        assert guard.requires_gfx1250_rev1(_config(marks=["skip-gfx1250v0"]))
        assert not guard.requires_gfx1250_rev1(_rev0_config())
        assert not guard.requires_gfx1250_rev1(
            _config(asic_revision=1, architecture="gfx950"))

    def test_codegen_knobs_are_ignored(self):
        cfg = {
            "GlobalParameters": {"Architecture": "gfx1250"},
            "BenchmarkProblems": [[
                {"OperationType": "GEMM", "DataType": "F8"},
                {"ForkParameters": [
                    {"ClusterDim": [[2, 2]]},
                    {"MatrixInstruction": [[16, 16, 128, 1, 1, 1, 1, 2, 2]]},
                    {"TDMInst": [3]},
                ]},
            ]],
        }
        assert guard.config_required_asic_revision(cfg) == 0
        assert not guard.requires_gfx1250_rev1(cfg)


class TestGuardMatrix:
    @pytest.mark.parametrize("hw,cfg,expect_abort", [
        ("gfx1250v0", _rev0_config(), False),
        ("gfx1250v0", _rev1_config(), True),
        ("gfx1250v0", _config(marks=["skip-gfx1250v0"]), True),
        ("gfx1250", _rev1_config(), False),
    ])
    def test_execute_guard(self, monkeypatch, capsys, hw, cfg, expect_abort):
        probe = _patch_probe(monkeypatch, hw)
        if expect_abort:
            with pytest.raises(SystemExit) as exc:
                _guard(arch_names=["gfx1250"], isa_info_map=GFX1250_IIM, config=cfg)
            assert exc.value.code == -1
            out = capsys.readouterr().out
            assert "skip-gfx1250v0" in out
            assert "RevisionID: 1" in out
            assert "revision 0" in out
            assert "--build-only" in out
            probe.assert_called_once_with(device_id=0)
        else:
            _guard(arch_names=["gfx1250"], isa_info_map=GFX1250_IIM, config=cfg)

    @pytest.mark.parametrize("kw", [{"build_only": True}, {"cpu_only": True}])
    def test_build_or_cpu_only_bypass(self, monkeypatch, kw):
        probe = _patch_probe(monkeypatch, "gfx1250v0")
        _guard(
            arch_names=["gfx1250"],
            isa_info_map=GFX1250_IIM,
            config=_rev1_config(),
            **kw,
        )
        probe.assert_not_called()

    def test_non_gfx1250_compile_ignores_rev1_yaml(self, monkeypatch):
        probe = _patch_probe(monkeypatch, "gfx1250v0")
        _guard(arch_names=["gfx950"], isa_info_map=GFX950_IIM, config=_rev1_config())
        _guard(arch_names=["gfx942"], isa_info_map=GFX942_IIM, config=_rev1_config())
        probe.assert_not_called()


class TestPytestSkipOnRev0:
    def test_should_skip_matrix(self):
        assert guard.should_skip_gfx1250_rev1_on_rev0(
            _rev1_config(),
            tensile_argv=["--gpu-targets", "gfx1250v0"],
            hardware_target="gfx1250",
        )
        assert guard.should_skip_gfx1250_rev1_on_rev0(
            _rev1_config(), tensile_argv=[], hardware_target="gfx1250v0")
        assert not guard.should_skip_gfx1250_rev1_on_rev0(
            _rev1_config(), tensile_argv=[], hardware_target="gfx1250")
        assert not guard.should_skip_gfx1250_rev1_on_rev0(
            _rev0_config(),
            tensile_argv=["--gpu-targets", "gfx1250v0"],
            hardware_target="gfx1250v0",
        )


def test_execute_steps_rev1_yaml_on_rev0_raises(monkeypatch, tmp_path, capsys):
    TensileModule = _tensile_module()
    monkeypatch.setattr(guard, "detect_gpu_revision_target", lambda **_kw: "gfx1250v0")
    called = {"bp": False}
    monkeypatch.setattr(
        TensileModule.BenchmarkProblems,
        "main",
        lambda *_a, **_kw: called.__setitem__("bp", True),
    )
    monkeypatch.setattr(TensileModule, "isaToGfx", lambda *_a, **_kw: "gfx1250")
    from contextlib import nullcontext
    monkeypatch.setattr(TensileModule, "timing_context", lambda *_a, **_kw: nullcontext())
    monkeypatch.setattr(TensileModule, "flush_timing_buffer", lambda: None)
    monkeypatch.setattr(TensileModule, "print1", lambda *_a, **_kw: None)
    with pytest.raises(SystemExit) as exc:
        TensileModule.executeStepsInConfig(
            config=_rev1_config(),
            outputPath=tmp_path,
            asmToolchain=types.SimpleNamespace(assembler=object()),
            srcToolchain=types.SimpleNamespace(compiler="cc"),
            isaInfoMap=GFX1250_IIM,
            cCompiler="cc",
            debugConfig=types.SimpleNamespace(
                splitGSU=False,
                printSolutionRejectionReason=False,
                printIndexAssignmentInfo=False,
            ),
            deviceId=0,
            probSolDict={},
            buildOnly=False,
            solutionPoolFiles=None,
            archNames=["gfx1250"],
        )
    assert exc.value.code == -1
    assert called["bp"] is False
    assert "RevisionID: 1" in capsys.readouterr().out
