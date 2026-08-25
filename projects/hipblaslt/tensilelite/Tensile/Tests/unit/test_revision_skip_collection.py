# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pytest collection skip identity for gfx1250 v0/v1.

No GPU and no rocisa: these cover hardware skip-set expansion, filename
regex, and StreamK tests that set skip-gfx1250v0 or RevisionID 1.
"""

import pathlib
from unittest import mock

import pytest
import yaml

from Tensile.Tests.gpu_detection import (
    filename_arch_token,
    resolve_skip_archs,
)

pytestmark = pytest.mark.unit

_TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent
_COMMON_DIR = _TESTS_DIR / "common"


def _skip_set(compile_archs, enumerated, revision_target):
    return set(resolve_skip_archs(
        compile_archs,
        enumerated_archs=enumerated,
        revision_target=revision_target,
    ))


class TestResolveSkipArchs:
    """Hardware skip set: expand aliases, probe only on real gfx1250."""

    def test_gpu_targets_gfx1250v0_does_not_drop_gfx1250(self):
        # The failure this guards: skip set {gfx1250v0} would miss skip-gfx1250
        # and collect those family-wide skips on v0 hardware.
        skip = _skip_set(["gfx1250v0"], enumerated=[], revision_target=None)
        assert skip == {"gfx1250", "gfx1250v0"}

    def test_bare_gpu_targets_gfx1250_without_hw_does_not_invent_v0(self):
        skip = _skip_set(["gfx1250"], enumerated=[], revision_target=None)
        assert skip == {"gfx1250"}
        assert "gfx1250v0" not in skip

    def test_real_gfx1250_rev0_is_probed_even_if_gpu_targets_says_gfx1250(self):
        skip = _skip_set(
            ["gfx1250"], enumerated=["gfx1250"], revision_target="gfx1250v0")
        assert skip == {"gfx1250", "gfx1250v0"}

    def test_real_gfx1250_rev1_skip_set_is_family_only(self):
        skip = _skip_set(
            ["gfx1250"], enumerated=["gfx1250"], revision_target="gfx1250")
        assert skip == {"gfx1250"}
        assert "gfx1250v0" not in skip

    def test_probe_failure_on_real_gfx1250_is_fail_open(self):
        # detect_gpu_revision_target() returns "gfx1250" on probe failure.
        with mock.patch(
            "Tensile.GpuRevisionTarget._probe_asic_revision",
            return_value=None,
        ), mock.patch(
            "Tensile.GpuRevisionTarget.detect_gpu_revision_target",
            return_value="gfx1250",
        ):
            skip = set(resolve_skip_archs(
                ["gfx1250"], enumerated_archs=["gfx1250"], revision_target=None))
        assert skip == {"gfx1250"}
        assert "gfx1250v0" not in skip

    def test_gfx950_does_not_apply_revision_target_default(self):
        # detect_gpu_revision_target() returns "gfx1250" for probe-fail / non-v0;
        # that must not be read as "this machine is gfx1250" on a gfx950.
        skip = _skip_set(
            ["gfx950"], enumerated=["gfx950"], revision_target="gfx1250")
        assert skip == {"gfx950"}
        assert "gfx1250" not in skip

    def test_gpu_targets_gfx1250_on_gfx950_does_not_probe(self):
        skip = _skip_set(
            ["gfx1250"], enumerated=["gfx950"], revision_target="gfx1250v0")
        assert skip == {"gfx1250"}
        assert "gfx1250v0" not in skip

    def test_non_gfx1250_skip_marks_are_unchanged(self):
        skip = _skip_set(["gfx942"], enumerated=["gfx942"], revision_target=None)
        assert skip == {"gfx942"}

    def test_detect_gpu_revision_target_is_not_called_on_gfx950(self):
        with mock.patch(
            "Tensile.GpuRevisionTarget.detect_gpu_revision_target",
            side_effect=AssertionError("must not probe on gfx950"),
        ), mock.patch(
            "Tensile.GpuRevisionTarget._probe_asic_revision",
            side_effect=AssertionError("must not probe on gfx950"),
        ):
            skip = resolve_skip_archs(
                ["gfx950"], enumerated_archs=["gfx950"], revision_target=None)
        assert set(skip) == {"gfx950"}

    def test_hip_probe_adds_gfx1250v0_on_real_rev0(self):
        with mock.patch(
            "Tensile.GpuRevisionTarget._probe_asic_revision",
            return_value=("gfx1250", 0),
        ) as probe:
            skip = resolve_skip_archs(
                ["gfx1250"], enumerated_archs=["gfx1250"], revision_target=None)
        probe.assert_called()
        assert set(skip) == {"gfx1250", "gfx1250v0"}

    def test_tensile_options_gfx1250v0_expands_skip_without_probe(self):
        from Tensile.Tests.gpu_detection import (
            gpu_targets_from_tensile_options,
            merge_pytest_compile_archs,
        )
        extra = gpu_targets_from_tensile_options("--gpu-targets,gfx1250v0")
        assert extra == ["gfx1250v0"]
        archs = merge_pytest_compile_archs("gfx1250", "--gpu-targets,gfx1250v0")
        assert archs == ["gfx1250", "gfx1250v0"]
        skip = _skip_set(archs, enumerated=["gfx1250"], revision_target="gfx1250")
        assert skip == {"gfx1250", "gfx1250v0"}


class TestFilenameArchToken:
    @pytest.mark.parametrize("filename,expected", [
        ("bf16_gfx1250.yaml", "gfx1250"),
        ("bf16_gfx1250v0.yaml", "gfx1250v0"),
        ("sk_sgemm_quick.yaml", None),
        ("gfx942.yaml", "gfx942"),
        ("", None),
    ])
    def test_revision_aliases_are_not_captured_as_gfx1250(self, filename, expected):
        assert filename_arch_token(filename) == expected

    def test_gfx1250v0_wins_over_gfx_digits(self):
        assert filename_arch_token("foo_gfx1250v0.yaml") == "gfx1250v0"


class TestStreamKGfx1250RevisionId:
    """SK tests with skip-gfx1250v0 or RevisionID 1 are rev1; others default to 0."""

    def test_sk_tests_with_skip_or_revision_id_are_rev1(self):
        from Tensile.Gfx1250RunGuard import requires_gfx1250_rev1

        root = _COMMON_DIR / "streamk" / "gfx1250"
        rev1 = []
        for path in sorted(root.rglob("*.yaml")):
            doc = yaml.safe_load(path.read_text())
            if not isinstance(doc, dict):
                continue
            tp = doc.get("TestParameters") or {}
            marked = "skip-gfx1250v0" in (tp.get("marks") or [])
            rev = int(tp.get("RevisionID", 0) or 0)
            if marked or rev >= 1:
                assert requires_gfx1250_rev1(doc, str(path))
                rev1.append(path.name)
        assert rev1

    def test_sk_test_without_skip_or_revision_id_is_rev0(self):
        from Tensile.Gfx1250RunGuard import requires_gfx1250_rev1

        path = _COMMON_DIR / "streamk/gfx1250/core/sk_mxf8gemm_tdm.yaml"
        doc = yaml.safe_load(path.read_text())
        tp = doc.get("TestParameters") or {}
        assert "skip-gfx1250v0" not in (tp.get("marks") or [])
        assert int(tp.get("RevisionID", 0) or 0) == 0
        assert not requires_gfx1250_rev1(doc, str(path))
