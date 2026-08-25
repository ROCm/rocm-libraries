# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pytest collection skip identity for gfx1250. No GPU and no rocisa."""

import pathlib
from unittest import mock

import pytest
import yaml

from Tensile.Tests.gpu_detection import resolve_skip_archs

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
    def test_gpu_targets_gfx1250v0_does_not_drop_gfx1250(self):
        skip = _skip_set(["gfx1250v0"], enumerated=[], revision_target=None)
        assert skip == {"gfx1250", "gfx1250v0"}
        skip = _skip_set(["gfx1250v0:xnack-"], enumerated=[], revision_target=None)
        assert skip == {"gfx1250", "gfx1250v0"}

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

    def test_real_gfx1250_rev1_skip_set_is_family_only(self):
        skip = _skip_set(
            ["gfx1250"], enumerated=["gfx1250"], revision_target="gfx1250")
        assert skip == {"gfx1250"}
        assert "gfx1250v0" not in skip

    def test_hip_probe_adds_gfx1250v0_on_real_rev0(self):
        with mock.patch(
            "Tensile.GpuRevisionTarget._probe_asic_revision",
            return_value=("gfx1250", 0),
        ) as probe:
            skip = resolve_skip_archs(
                ["gfx1250"],
                enumerated_archs=["gfx1250:sramecc+:xnack-"],
                revision_target=None,
            )
        probe.assert_called()
        assert set(skip) == {"gfx1250", "gfx1250v0"}

    def test_probe_failure_on_real_gfx1250_is_fail_open(self):
        with mock.patch(
            "Tensile.GpuRevisionTarget._probe_asic_revision",
            return_value=None,
        ):
            skip = set(resolve_skip_archs(
                ["gfx1250"], enumerated_archs=["gfx1250"], revision_target=None))
        assert skip == {"gfx1250"}
        assert "gfx1250v0" not in skip

    def test_gfx950_does_not_add_gfx1250v0(self):
        with mock.patch(
            "Tensile.GpuRevisionTarget._probe_asic_revision",
            side_effect=AssertionError("must not probe on gfx950"),
        ), mock.patch(
            "Tensile.GpuRevisionTarget.detect_gpu_revision_target",
            side_effect=AssertionError("must not probe on gfx950"),
        ):
            skip_injected = set(resolve_skip_archs(
                ["gfx1250"], enumerated_archs=["gfx950"], revision_target="gfx1250v0"))
            skip_live = set(resolve_skip_archs(
                ["gfx950"], enumerated_archs=["gfx950"], revision_target=None))
        assert skip_injected == {"gfx1250"}
        assert "gfx1250v0" not in skip_injected
        assert skip_live == {"gfx950"}

    def test_unknown_v1_token_does_not_enter_the_skip_set(self):
        skip = _skip_set(
            ["gfx1250"], enumerated=["gfx1250"], revision_target="gfx1250v1")
        assert skip == {"gfx1250"}
        assert "gfx1250v1" not in skip
        assert "gfx1250v0" not in skip


class TestStreamKGfx1250RevisionId:
    def test_rev1_fixture(self):
        from Tensile.Gfx1250RunGuard import requires_gfx1250_rev1

        path = _COMMON_DIR / "streamk" / "gfx1250" / "sk_mxf4gemm_tdm_ext.yaml"
        doc = yaml.safe_load(path.read_text())
        tp = doc.get("TestParameters") or {}
        assert "skip-gfx1250v0" in (tp.get("marks") or [])
        assert int(tp.get("RevisionID", 0) or 0) == 1
        assert requires_gfx1250_rev1(doc)

    def test_rev0_fixture(self):
        from Tensile.Gfx1250RunGuard import requires_gfx1250_rev1

        path = _COMMON_DIR / "streamk" / "gfx1250" / "core" / "sk_mxf8gemm_tdm.yaml"
        doc = yaml.safe_load(path.read_text())
        tp = doc.get("TestParameters") or {}
        assert "skip-gfx1250v0" not in (tp.get("marks") or [])
        assert int(tp.get("RevisionID", 0) or 0) == 0
        assert not requires_gfx1250_rev1(doc)
