# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute-time gfx1250 revision guard.

No GPU: detect_gpu_revision_target is mocked. Approved matrix:

  * RevisionID defaults to 0; those YAMLs run on gfx1250 rev0
  * skip-gfx1250v0 or RevisionID 1 aborts on gfx1250 rev0
  * the check is a no-op unless this compile/run is gfx1250
  * --build-only and --cpu-only still bypass
  * codegen knobs (ClusterDim, TDM, MX vs FP8) are not inspected

The helper lives in Tensile.Gfx1250RunGuard so these tests never import rocisa.
"""

from unittest import mock
import os

import pytest
import yaml

from Tensile import Gfx1250RunGuard as guard

pytestmark = pytest.mark.unit

GFX1250_IIM = {(12, 5, 0): object()}
GFX950_IIM = {(9, 5, 0): object()}
GFX942_IIM = {(9, 4, 2): object()}

_STREAMK_GFX1250 = os.path.join(
    os.path.dirname(__file__), "..", "common", "streamk", "gfx1250"
)


def _config(*, asic_revision=None, marks=None, architecture="gfx1250"):
    test_params = {}
    if asic_revision is not None:
        test_params["RevisionID"] = asic_revision
    if marks is not None:
        test_params["marks"] = marks
    cfg = {
        "TestParameters": test_params,
        "GlobalParameters": {"Architecture": architecture},
        "BenchmarkProblems": [],
        "UseCache": False,
    }
    return cfg


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


def _patch_probe(monkeypatch, hw, probe=None):
    if probe is None:
        probe = mock.Mock(return_value=hw)
    monkeypatch.setattr(guard, "detect_gpu_revision_target", probe)
    return probe


class TestCompileTargetHelpers:
    def test_plain_gfx1250_is_family(self):
        assert guard.is_gfx1250_family_compile(["gfx1250"], GFX1250_IIM)
        assert guard.is_gfx1250_family_compile(["gfx1250v0"], GFX1250_IIM)
        assert guard.is_gfx1250_family_compile(["gfx1250[cu=64]"], GFX1250_IIM)

    def test_implicit_isa_only_is_family(self):
        assert guard.is_gfx1250_family_compile([], GFX1250_IIM)
        assert guard.is_gfx1250_family_compile(None, GFX1250_IIM)

    def test_non_gfx1250_isa_is_not_family(self):
        assert not guard.is_gfx1250_family_compile(["gfx950"], GFX950_IIM)
        assert not guard.is_gfx1250_family_compile(["gfx942"], GFX942_IIM)
        assert not guard.is_gfx1250_family_compile([], GFX950_IIM)


class TestRevisionIdMetadata:
    def test_missing_field_defaults_to_rev0(self):
        assert guard.config_required_asic_revision(_rev0_config()) == 0
        assert guard.config_required_asic_revision({}) == 0
        assert guard.config_required_asic_revision(None) == 0

    def test_explicit_rev1(self):
        assert guard.config_required_asic_revision(_rev1_config()) == 1

    def test_skip_mark_means_rev1(self):
        cfg = _config(marks=["skip-gfx1250v0"])
        assert guard.config_required_asic_revision(cfg) == 1

    def test_global_parameters_fallback(self):
        cfg = {"GlobalParameters": {"RevisionID": 1, "Architecture": "gfx1250"}}
        assert guard.config_required_asic_revision(cfg) == 1

    def test_requires_rev1_only_when_yaml_is_gfx1250(self):
        gfx1250_rev1 = _rev1_config()
        gfx950_rev1 = _config(asic_revision=1, architecture="gfx950")
        assert guard.requires_gfx1250_rev1(gfx1250_rev1) is True
        assert guard.requires_gfx1250_rev1(gfx950_rev1) is False
        assert guard.requires_gfx1250_rev1(_rev0_config()) is False

    def test_path_under_gfx1250_counts_as_gfx1250_yaml(self):
        cfg = {"TestParameters": {"RevisionID": 1}, "GlobalParameters": {}}
        path = os.path.join(_STREAMK_GFX1250, "core", "sk_mxf8gemm_tdm.yaml")
        assert guard.config_targets_gfx1250(cfg, path) is True
        assert guard.requires_gfx1250_rev1(cfg, path) is True

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
        assert guard.requires_gfx1250_rev1(cfg) is False

    def test_sk_yaml_with_revision_id_1_is_rev1(self):
        cfg = _rev1_config()
        assert cfg["TestParameters"]["RevisionID"] == 1
        assert guard.requires_gfx1250_rev1(cfg) is True

    def test_sk_yaml_with_skip_mark_is_rev1(self):
        cfg = _config(marks=["skip-gfx1250v0"])
        assert guard.requires_gfx1250_rev1(cfg) is True

    def test_streamk_mxf8gemm_tdm_core_is_rev0(self):
        path = os.path.join(_STREAMK_GFX1250, "core", "sk_mxf8gemm_tdm.yaml")
        with open(path) as handle:
            cfg = yaml.safe_load(handle)
        assert guard.config_required_asic_revision(cfg) == 0
        assert guard.requires_gfx1250_rev1(cfg, path) is False


class TestPytestSkipOnRev0:
    """pytest/tox skip (not Tensile abort). hardware_target is injected; no GPU."""

    def test_tensile_argv_gfx1250v0_skips_rev1(self):
        assert guard.should_skip_gfx1250_rev1_on_rev0(
            _rev1_config(),
            tensile_argv=["--gpu-targets", "gfx1250v0"],
            hardware_target="gfx1250",
        )

    def test_tensile_options_comma_argv_skips_rev1(self):
        assert guard.should_skip_gfx1250_rev1_on_rev0(
            _rev1_config(),
            tensile_argv=["--gpu-targets", "gfx1250v0"],
            hardware_target="gfx1250",
        )
        from Tensile.GpuRevisionTarget import argv_selects_gfx1250v0
        assert argv_selects_gfx1250v0("--gpu-targets,gfx1250v0".split(","))
        assert not argv_selects_gfx1250v0(["--gpu-targets", "gfx1250"])

    def test_hardware_rev0_skips_rev1(self):
        assert guard.should_skip_gfx1250_rev1_on_rev0(
            _rev1_config(), tensile_argv=[], hardware_target="gfx1250v0")

    def test_hardware_rev1_does_not_skip(self):
        assert not guard.should_skip_gfx1250_rev1_on_rev0(
            _rev1_config(), tensile_argv=[], hardware_target="gfx1250")

    def test_rev0_yaml_still_runs_on_rev0(self):
        assert not guard.should_skip_gfx1250_rev1_on_rev0(
            _rev0_config(),
            tensile_argv=["--gpu-targets", "gfx1250v0"],
            hardware_target="gfx1250v0",
        )

    def test_argv_gfx1250v0_does_not_probe(self, monkeypatch):
        probe = _patch_probe(monkeypatch, "gfx1250v0")
        assert guard.should_skip_gfx1250_rev1_on_rev0(
            _rev1_config(),
            tensile_argv=["--gpu-targets", "gfx1250v0"],
        )
        probe.assert_not_called()


class TestGuardMatrix:
    def test_rev0_yaml_on_rev0_hw_allowed(self, monkeypatch):
        probe = _patch_probe(monkeypatch, "gfx1250v0")
        _guard(arch_names=["gfx1250"], isa_info_map=GFX1250_IIM, config=_rev0_config())
        probe.assert_not_called()

    def test_rev1_yaml_on_rev0_hw_aborts(self, monkeypatch, capsys):
        probe = _patch_probe(monkeypatch, "gfx1250v0")
        with pytest.raises(SystemExit) as exc:
            _guard(arch_names=["gfx1250"], isa_info_map=GFX1250_IIM, config=_rev1_config())
        assert exc.value.code == -1
        probe.assert_called_once_with(device_id=0)
        out = capsys.readouterr().out
        assert "skip-gfx1250v0" in out
        assert "RevisionID: 1" in out
        assert "revision 0" in out
        assert "--build-only" in out

    def test_skip_mark_without_field_on_rev0_aborts(self, monkeypatch):
        _patch_probe(monkeypatch, "gfx1250v0")
        with pytest.raises(SystemExit):
            _guard(
                arch_names=["gfx1250"],
                isa_info_map=GFX1250_IIM,
                config=_config(marks=["skip-gfx1250v0"]),
            )

    def test_rev1_yaml_on_rev1_hw_allowed(self, monkeypatch):
        probe = _patch_probe(monkeypatch, "gfx1250")
        _guard(arch_names=["gfx1250"], isa_info_map=GFX1250_IIM, config=_rev1_config())
        probe.assert_called_once()

    def test_build_only_rev1_on_rev0_allowed(self, monkeypatch):
        probe = _patch_probe(monkeypatch, "gfx1250v0")
        _guard(
            build_only=True,
            arch_names=["gfx1250"],
            isa_info_map=GFX1250_IIM,
            config=_rev1_config(),
        )
        probe.assert_not_called()

    def test_cpu_only_rev1_on_rev0_allowed(self, monkeypatch):
        probe = _patch_probe(monkeypatch, "gfx1250v0")
        _guard(
            cpu_only=True,
            arch_names=["gfx1250"],
            isa_info_map=GFX1250_IIM,
            config=_rev1_config(),
        )
        probe.assert_not_called()

    def test_non_gfx1250_compile_ignores_rev1_yaml(self, monkeypatch):
        probe = _patch_probe(monkeypatch, "gfx1250v0")
        _guard(arch_names=["gfx950"], isa_info_map=GFX950_IIM, config=_rev1_config())
        _guard(arch_names=["gfx942"], isa_info_map=GFX942_IIM, config=_rev1_config())
        probe.assert_not_called()

    def test_gfx950_device_is_not_rev0(self, monkeypatch):
        probe = _patch_probe(monkeypatch, "gfx950")
        _guard(arch_names=["gfx1250"], isa_info_map=GFX1250_IIM, config=_rev1_config())
        probe.assert_called_once()

    def test_device_id_is_forwarded(self, monkeypatch):
        probe = _patch_probe(monkeypatch, "gfx1250")
        _guard(
            arch_names=["gfx1250"],
            isa_info_map=GFX1250_IIM,
            device_id=3,
            config=_rev1_config(),
        )
        probe.assert_called_once_with(device_id=3)


def _stub_execute_deps(monkeypatch, TensileModule):
    from contextlib import nullcontext

    called = {"bp": False}
    monkeypatch.setattr(
        TensileModule.BenchmarkProblems,
        "main",
        lambda *_a, **_kw: called.__setitem__("bp", True),
    )
    monkeypatch.setattr(TensileModule, "isaToGfx", lambda *_a, **_kw: "gfx1250")
    monkeypatch.setattr(TensileModule, "timing_context", lambda *_a, **_kw: nullcontext())
    monkeypatch.setattr(TensileModule, "flush_timing_buffer", lambda: None)
    monkeypatch.setattr(TensileModule, "print1", lambda *_a, **_kw: None)
    return called


def _run_execute_steps(TensileModule, tmp_path, *, build_only, arch_names, config=None):
    import types

    if config is None:
        config = {"BenchmarkProblems": [], "UseCache": False}
    return TensileModule.executeStepsInConfig(
        config=config,
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
        buildOnly=build_only,
        solutionPoolFiles=None,
        archNames=arch_names,
    )


def test_execute_steps_rev0_yaml_on_rev0_allowed(monkeypatch, tmp_path):
    TensileModule = _tensile_module()
    monkeypatch.setattr(guard, "detect_gpu_revision_target", lambda **_kw: "gfx1250v0")
    called = _stub_execute_deps(monkeypatch, TensileModule)
    _run_execute_steps(
        TensileModule,
        tmp_path,
        build_only=False,
        arch_names=["gfx1250"],
        config=_rev0_config(),
    )
    assert called["bp"] is True


def test_execute_steps_rev1_yaml_on_rev0_raises(monkeypatch, tmp_path, capsys):
    TensileModule = _tensile_module()
    monkeypatch.setattr(guard, "detect_gpu_revision_target", lambda **_kw: "gfx1250v0")
    called = _stub_execute_deps(monkeypatch, TensileModule)
    with pytest.raises(SystemExit) as exc:
        _run_execute_steps(
            TensileModule,
            tmp_path,
            build_only=False,
            arch_names=["gfx1250"],
            config=_rev1_config(),
        )
    assert exc.value.code == -1
    assert called["bp"] is False
    assert "RevisionID: 1" in capsys.readouterr().out


def test_execute_steps_build_only_rev1_on_rev0_allowed(monkeypatch, tmp_path):
    TensileModule = _tensile_module()
    monkeypatch.setattr(guard, "detect_gpu_revision_target", lambda **_kw: "gfx1250v0")
    called = _stub_execute_deps(monkeypatch, TensileModule)
    _run_execute_steps(
        TensileModule,
        tmp_path,
        build_only=True,
        arch_names=["gfx1250"],
        config=_rev1_config(),
    )
    assert called["bp"] is True
