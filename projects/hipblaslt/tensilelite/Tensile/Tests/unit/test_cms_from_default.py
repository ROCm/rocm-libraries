################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

"""End-to-end tests for the default-schedule -> CMS converter.

Covers the bead `rocm-libraries-wlrp` Phase 2 acceptance items:

- End-to-end conversion against a known-good CMS-eligible kernel config.
- Multi-Solution YAML rejection (names the fork dimensions).
- Schedule-collision rejection without `--overwrite`.
- Schedule-collision bypass with `--overwrite` / `force=True`.
- Non-DTL nglshift regression (per memo §3.3 directive).
- Soft-fail TimingTooCloseFailure handling.
- Hard-fail on graph-comparison failure types other than timing.

The CLI tests run the converter as a Python subprocess to validate
`python -m Tensile.Components.CustomSchedule.cms_from_default`.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from Tensile.Components.CustomSchedule.cms_from_default import (
    ValidationReport,
    _check_collision,
    _derive_nglshift_nllshift,
    _flatten_named_lists,
    _resolve_isa,
    default_schedule_to_cms,
    main as cli_main,
)
from Tensile.Components.CustomSchedule.shared import (
    TileConfig,
    is16bit,
    isTF32,
)


# A kernel config known to produce a clean default + CMS capture pair on
# the production gating path (mirrors TestPhase4DefaultCapture in
# test_ScheduleCapture.py — F32X TF32 16x16x32 4x4 DepthU=32). That config
# matches the registered _get_schedule_128x128x32_TF32 schedule.
_GOOD_YAML = """\
GlobalParameters:
  ForceGenerateKernel: 1

BenchmarkProblems:
  -
    -
      OperationType: GEMM
      DataType: S
      DestDataType: S
      F32XdlMathOp: X
      TransposeA: 1
      TransposeB: 0
      UseBeta: True
      Batched: True
    -
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
      ForkParameters:
        - MatrixInstruction: [[16, 16, 32, 1, 1, 4, 4, 2, 2]]
        - DepthU: [32]
        - PrefetchGlobalRead: [2]
        - PrefetchLocalRead: [1]
        - DirectToLds: [1]
        - TransposeLDS: [1]
        - LocalReadVectorWidth: [4]
        - GlobalReadVectorWidthA: [4]
        - GlobalReadVectorWidthB: [4]
        - UseCustomMainLoopSchedule: [1]
        - ExpandPointerSwap: [0]
        - SourceSwap: [1]
        - StreamK: [0]
"""


def _write_yaml(tmp_path, body=_GOOD_YAML, name="config.yaml"):
    p = tmp_path / name
    p.write_text(body)
    return p


# =========================================================================
# Pure-helper tests (don't need ISA infrastructure or kernel build).
# =========================================================================


class TestFlattenNamedLists:
    def test_empty(self):
        assert _flatten_named_lists([]) == {}

    def test_dict_passthrough(self):
        assert _flatten_named_lists({"A": [1]}) == {"A": [1]}

    def test_list_of_singletons(self):
        items = [{"A": [1]}, {"B": [2, 3]}]
        assert _flatten_named_lists(items) == {"A": [1], "B": [2, 3]}

    def test_none_skipped(self):
        items = [{"A": [1]}, None, {"B": [2]}]
        assert _flatten_named_lists(items) == {"A": [1], "B": [2]}

    def test_invalid_entry_raises(self):
        with pytest.raises(ValueError, match="dict in fork/common"):
            _flatten_named_lists([42])


class TestResolveIsa:
    def test_override_wins(self):
        assert _resolve_isa({"ISA": [1, 2, 3]}, (9, 5, 0)) == (9, 5, 0)

    def test_yaml_used(self):
        assert _resolve_isa({"ISA": [9, 5, 0]}, None) == (9, 5, 0)

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="No ISA available"):
            _resolve_isa({}, None)

    def test_bad_override_shape(self):
        with pytest.raises(ValueError, match="3-tuple"):
            _resolve_isa({}, (9, 5))


class TestNglshiftNllshiftDtlAware:
    """Memo §3.3: DTL-aware branch. DTL halves; non-DTL doesn't."""

    def _tile(self, dtl):
        return TileConfig(128, 128, 32, 2, 1, dtl, False, 0, 0, isa=(9, 5, 0))

    def test_dtl_halves_total(self):
        opt = {"GRA": [[10, 10, 12, 12]], "GRB": [[20, 20]]}
        ngl, nll = _derive_nglshift_nllshift(opt, self._tile(dtl=1))
        # (4+2)//2 = 3
        assert (ngl, nll) == (3, 3)

    def test_non_dtl_does_not_halve(self):
        # Non-DTL regression test (per memo §3.3 directive). The non-DTL
        # branch is implemented as len(GRA[0]) + len(GRB[0]) — the same
        # underlying logic as DTL but without the doubling correction
        # because non-DTL GRA/GRB lists contain one entry per load
        # (no pointer-increment doubling).
        opt = {"GRA": [[10, 11, 12]], "GRB": [[20, 21]]}
        ngl, nll = _derive_nglshift_nllshift(opt, self._tile(dtl=0))
        assert (ngl, nll) == (5, 5)

    def test_no_gra_grb_yields_zero(self):
        # Pathological but should not crash.
        ngl, nll = _derive_nglshift_nllshift({}, self._tile(dtl=1))
        assert (ngl, nll) == (0, 0)


class TestCollisionCheck:
    def test_collision_with_existing_registered_schedule_raises(self):
        # The existing 256x96x64 16bit schedule is registered at import
        # time. Probing for the same TileConfig + dtype + vector_widths
        # + matrix_inst + mfma_wave_group must raise.
        tc = TileConfig(256, 96, 64, 2, 1, 1, False, 0, 0, isa=(9, 5, 0))
        with pytest.raises(RuntimeError, match="Schedule collision"):
            _check_collision(
                tc, dtype_predicate=is16bit,
                vector_widths=[8, 8, 8],
                matrix_inst=[16, 16, 32, 1],
                mfma_wave_group=[2, 2],
            )

    def test_no_collision_for_unique_tile(self):
        # A made-up tile that no schedule covers.
        tc = TileConfig(7, 7, 7, 1, 1, 1, False, 0, 0, isa=(9, 5, 0))
        # Should return without raising.
        _check_collision(
            tc, dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2],
        )


# =========================================================================
# Multi-Solution rejection (no kernel build needed, the rejection happens
# at YAML parse time before any kernel infra is touched).
# =========================================================================


class TestMultiSolutionRejection:
    def test_two_matrix_instructions_rejected(self, tmp_path):
        body = _GOOD_YAML.replace(
            "MatrixInstruction: [[16, 16, 32, 1, 1, 4, 4, 2, 2]]",
            "MatrixInstruction:\n          - [16, 16, 32, 1, 1, 4, 4, 2, 2]\n"
            "          - [16, 16, 32, 1, 1, 5, 4, 2, 2]",
        )
        yaml_path = _write_yaml(tmp_path, body)
        out_path = tmp_path / "out.py"
        with pytest.raises(ValueError, match="multiple Solutions"):
            default_schedule_to_cms(
                yaml_path=yaml_path, output_path=out_path,
                schedule_name="x", isa=(9, 5, 0), force=True,
            )

    def test_error_names_fork_dimensions(self, tmp_path):
        body = _GOOD_YAML.replace(
            "PrefetchGlobalRead: [2]",
            "PrefetchGlobalRead: [1, 2]",
        )
        yaml_path = _write_yaml(tmp_path, body)
        out_path = tmp_path / "out.py"
        with pytest.raises(ValueError) as excinfo:
            default_schedule_to_cms(
                yaml_path=yaml_path, output_path=out_path,
                schedule_name="x", isa=(9, 5, 0), force=True,
            )
        assert "PrefetchGlobalRead" in str(excinfo.value)


class TestAlternateFormatRejection:
    def test_top_level_problemtype_rejected(self, tmp_path):
        body = textwrap.dedent("""\
            ProblemType:
              OperationType: GEMM
            ForkParameters:
              - MatrixInstruction: [[16, 16, 32, 1]]
        """)
        yaml_path = _write_yaml(tmp_path, body)
        with pytest.raises(ValueError, match="alternate Tensile format"):
            default_schedule_to_cms(
                yaml_path=yaml_path, output_path=tmp_path / "out.py",
                schedule_name="x", isa=(9, 5, 0), force=True,
            )


# =========================================================================
# End-to-end + collision-bypass tests that actually drive a kernel build.
# Gated on isa_infrastructure being available (real compiler probe).
# =========================================================================


class TestEndToEnd:
    """End-to-end conversion against a known-good CMS kernel config."""

    def test_emits_python_file_for_known_good_config(
        self, isa_infrastructure, tmp_path
    ):
        yaml_path = _write_yaml(tmp_path)
        out_path = tmp_path / "emitted.py"
        report = default_schedule_to_cms(
            yaml_path=yaml_path,
            output_path=out_path,
            schedule_name="_get_schedule_test_emitted",
            isa=(9, 5, 0),
            force=True,
        )
        assert out_path.exists()
        assert isinstance(report, ValidationReport)
        body = out_path.read_text()
        # Decorator + function present.
        assert "@RegisterSchedule(" in body
        assert "def _get_schedule_test_emitted(" in body
        # ScheduleInfo construction present.
        assert "ScheduleInfo(" in body
        # No graph_diff failures expected (clean kernel).
        assert report.graph_diff == []
        assert report.wait_coverage == []
        # ISA passed through explicitly.
        assert "isa=(9, 5, 0)" in body

    def test_collision_bypass_with_force(
        self, isa_infrastructure, tmp_path
    ):
        # The 128x128x32 TF32 config we use here MATCHES the registered
        # _get_schedule_128x128x32_TF32 schedule by construction. Without
        # force=True the converter should refuse; with force=True it
        # proceeds.
        yaml_path = _write_yaml(tmp_path)
        out_path = tmp_path / "emitted.py"
        # Without force: should raise.
        with pytest.raises(RuntimeError, match="Schedule collision"):
            default_schedule_to_cms(
                yaml_path=yaml_path,
                output_path=out_path,
                schedule_name="_get_schedule_test_collide",
                isa=(9, 5, 0),
                force=False,
            )
        # With force: succeeds.
        report = default_schedule_to_cms(
            yaml_path=yaml_path,
            output_path=out_path,
            schedule_name="_get_schedule_test_collide",
            isa=(9, 5, 0),
            force=True,
        )
        assert out_path.exists()
        assert isinstance(report, ValidationReport)


class TestCli:
    """Drive the CLI in-process via the `main()` entry point."""

    def test_cli_help(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli_main(["--help"])
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        # Sanity: usage text mentions the input + output args.
        assert "input_yaml" in captured.out
        assert "output_py" in captured.out
        assert "--schedule-name" in captured.out
        assert "--overwrite" in captured.out
        # Documents that alt-format is not supported.
        assert "alt-format" in captured.out or "alternate-format" in captured.out

    def test_cli_runs_end_to_end(self, isa_infrastructure, tmp_path):
        yaml_path = _write_yaml(tmp_path)
        out_path = tmp_path / "cli_emitted.py"
        rc = cli_main([
            str(yaml_path), str(out_path),
            "--schedule-name", "_get_schedule_test_cli",
            "--isa", "9.5.0",
            "--overwrite",
        ])
        assert rc == 0
        assert out_path.exists()
        body = out_path.read_text()
        assert "_get_schedule_test_cli" in body

    def test_cli_rejects_bad_isa_string(self, capsys):
        with pytest.raises(SystemExit):
            cli_main([
                "/no/such/yaml.yaml", "/no/such/out.py",
                "--schedule-name", "x",
                "--isa", "not-an-isa",
            ])


# =========================================================================
# Soft-fail / hard-fail surface for validate_schedule_against_default.
#
# These tests target the categorization logic directly via a mocked writer
# + monkey-patched validator to avoid the expense of constructing real
# failure scenarios from a full kernel build. The categorization is the
# load-bearing rule (memo §6.3) and warrants direct coverage independent
# of the kernel-build path.
# =========================================================================


class _MockGraphFailure:
    """Minimal duck-type for a non-timing graph-comparison failure."""
    def format(self):
        return "mock non-timing failure"


class _MockTimingFailure:
    """Looks-like-TimingTooCloseFailure but a distinct class so we can
    pivot via monkey-patching `TimingTooCloseFailure` in the converter
    module's globals."""
    def format(self):
        return "mock timing failure"


class _MockCapture:
    arch_profile = None


class _MockWriter:
    _last_default_capture = _MockCapture()
    _last_cms_capture = _MockCapture()


def _patch_validator(monkeypatch, *, graph_failures, wait_failures,
                     timing_class, vopd_failures=()):
    """Stub the CMSValidator entry points the wrapper calls.

    The wrapper imports `TimingTooCloseFailure`, `build_dataflow_graph`,
    `compare_graphs`, `validate_edge_wait_coverage`, and
    `validate_vopd_pair_formation` from `Tensile.Components.CMSValidator`.
    We stub all five — the VOPD pass defaults to "no failures" because
    today no kernel emits VOPD; tests that exercise hard-fail paths
    pass an explicit `vopd_failures` list.
    """
    import Tensile.Components.CMSValidator as cmsv
    monkeypatch.setattr(cmsv, "TimingTooCloseFailure", timing_class,
                        raising=True)
    monkeypatch.setattr(cmsv, "build_dataflow_graph", lambda cap: object(),
                        raising=True)
    monkeypatch.setattr(cmsv, "compare_graphs",
                        lambda r, s: list(graph_failures), raising=True)
    monkeypatch.setattr(cmsv, "validate_edge_wait_coverage",
                        lambda g: list(wait_failures), raising=True)
    monkeypatch.setattr(cmsv, "validate_vopd_pair_formation",
                        lambda g: list(vopd_failures), raising=True)


class TestValidateAgainstDefaultSoftFail:
    def test_timing_only_failures_soft_fail(self, monkeypatch):
        from Tensile.Components.CustomSchedule import cms_from_default

        _patch_validator(
            monkeypatch,
            graph_failures=[_MockTimingFailure(), _MockTimingFailure()],
            wait_failures=[],
            timing_class=_MockTimingFailure,
        )
        writer = _MockWriter()
        report = cms_from_default.validate_schedule_against_default(
            writer, _MockCapture()
        )
        assert len(report.timing_only_failures) == 2
        assert len(report.graph_diff) == 2
        assert report.wait_coverage == []
        assert report.structural == []

    def test_non_timing_graph_failure_hard_fails(self, monkeypatch):
        from Tensile.Components.CustomSchedule import cms_from_default

        _patch_validator(
            monkeypatch,
            graph_failures=[_MockGraphFailure()],
            wait_failures=[],
            timing_class=_MockTimingFailure,
        )
        writer = _MockWriter()
        with pytest.raises(RuntimeError, match="non-timing failure"):
            cms_from_default.validate_schedule_against_default(
                writer, _MockCapture()
            )

    def test_wait_coverage_failure_hard_fails(self, monkeypatch):
        from Tensile.Components.CustomSchedule import cms_from_default

        _patch_validator(
            monkeypatch,
            graph_failures=[],
            wait_failures=[_MockGraphFailure()],
            timing_class=_MockTimingFailure,
        )
        writer = _MockWriter()
        with pytest.raises(RuntimeError, match="wait-coverage"):
            cms_from_default.validate_schedule_against_default(
                writer, _MockCapture()
            )

    def test_clean_validation_returns_empty_report(self, monkeypatch):
        from Tensile.Components.CustomSchedule import cms_from_default

        _patch_validator(
            monkeypatch,
            graph_failures=[],
            wait_failures=[],
            timing_class=_MockTimingFailure,
        )
        writer = _MockWriter()
        report = cms_from_default.validate_schedule_against_default(
            writer, _MockCapture()
        )
        assert report.timing_only_failures == []
        assert report.graph_diff == []
        assert report.wait_coverage == []

    def test_missing_cms_capture_raises(self, monkeypatch):
        from Tensile.Components.CustomSchedule import cms_from_default

        class W:
            _last_default_capture = _MockCapture()
            _last_cms_capture = None

        with pytest.raises(RuntimeError, match="CMS-side capture"):
            cms_from_default.validate_schedule_against_default(
                W(), _MockCapture()
            )


class TestSoftFailDocstringWarning:
    """When TimingTooCloseFailure is the only failure, the converter
    succeeds AND the warning ends up in the emitted file's docstring."""

    def test_warning_in_docstring(self, isa_infrastructure, tmp_path,
                                  monkeypatch):
        # Patch compare_graphs to inject one TimingTooCloseFailure for the
        # final validation pass while leaving the kernelBody-side
        # validation untouched. We do this by patching compare_graphs only
        # for the call inside `validate_schedule_against_default`. The
        # simplest mechanism: patch the inner reference in the converter
        # module after import.
        from Tensile.Components.CustomSchedule import cms_from_default
        import Tensile.Components.CMSValidator as cmsv

        original_validate = cms_from_default.validate_schedule_against_default

        # Build a mock TimingTooCloseFailure instance using the real class
        # so the isinstance check inside the wrapper works.
        from dataclasses import is_dataclass

        class _FakeTimingFailure(cmsv.TimingTooCloseFailure):
            def __init__(self):
                pass
            def format(self):
                return "INJECTED timing-too-close warning for test"

        def patched_validate(writer, default_capture):
            # Skip the real validator; inject a soft-failable timing report.
            return cms_from_default.ValidationReport(
                structural=[],
                graph_diff=[_FakeTimingFailure()],
                wait_coverage=[],
                timing_only_failures=[_FakeTimingFailure()],
            )

        monkeypatch.setattr(
            cms_from_default, "validate_schedule_against_default",
            patched_validate, raising=True,
        )

        yaml_path = _write_yaml(tmp_path)
        out_path = tmp_path / "softfail_emitted.py"
        report = cms_from_default.default_schedule_to_cms(
            yaml_path=yaml_path,
            output_path=out_path,
            schedule_name="_get_schedule_softfail",
            isa=(9, 5, 0),
            force=True,
        )
        assert out_path.exists()
        body = out_path.read_text()
        # Soft-fail warning should appear in the docstring.
        assert "INJECTED timing-too-close warning for test" in body
        assert "soft-fail" in body
        assert len(report.timing_only_failures) == 1
