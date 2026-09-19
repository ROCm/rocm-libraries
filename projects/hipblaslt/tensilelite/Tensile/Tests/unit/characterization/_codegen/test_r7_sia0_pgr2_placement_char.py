################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R7 — SIA0 (ScheduleIterAlg=0) PGR2 non-TDM global-read / tail-reset placement.

CPU-only characterization. Closes the gap that let a SIA0-only codegen change
reach develop without any CPU-PR-CI signal: every other designed config pins
``ScheduleIterAlg: [3]`` and the subtile configs use the LogicalScheduler, so
the legacy SIA0 emission path was never exercised by a content-sensitive test.

Drives the designed config
``data/_designed/gfx1250/sia0_pgr2_xf32_tn.yaml`` (F32X TN, ScheduleIterAlg=0,
PrefetchGlobalRead=2, TDMInst=0/non-TDM, StreamK=0) through the config-driven
emit harness. That kernel routes through:

  Components/SIA.py:noSchedGlobalRead
    PGR2 global-read placement (the ``_ScheduleIterAlg == 0`` arm).

  KernelWriter.py tail local-read reset
    "Tail: local read reset offsets a/b" — emitted for SIA0 non-TDM via the
    ``not (enableTDMA and enableTDMB)`` clause. With ScheduleIterAlg=0,
    StreamK=0 and non-TDM, this reset appears ONLY when that clause is present,
    so its presence is a precise, toolchain-independent (Tensile-emitted
    comment) characterization of the SIA0 non-TDM tail-reset behavior.

The projection snapshot below pins that behavior. It is derived from
``Tests/common/gemm/gfx12/xfp32_gfx1250.yaml`` (the F32X TN problem), reduced to
a single MI shape / size for a cheap emit.

CPU-only. No GPU, no compile, no hardware access.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_LIMIT = 8

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "sia0_pgr2_xf32_tn.yaml",
)


def _tail_lr_reset(src):
    """Return the SIA0 tail local-read reset markers emitted for this kernel.

    These ``addComment1`` strings come from KernelWriter.localReadResetOffsets
    in the tail-loop preamble; they are emitted by Tensile (not the assembler),
    so their presence is independent of the amdclang/hipcc version.
    """
    return {
        "tail_lr_reset_a": "Tail: local read reset offsets a" in src,
        "tail_lr_reset_b": "Tail: local read reset offsets b" in src,
    }


def test_r7_sia0_pgr2_emits_assembly():
    """SIA0 PGR2 non-TDM F32X config emits real gfx1250 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got 0 (config: {_CONFIG})"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"kernel {base!r}: assembly unexpectedly short"
        )
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx1250" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"


def test_r7_sia0_pgr2_placement_golden(snapshot):
    """Golden: SIA0 tail local-read reset markers per kernel.

    Pins the SIA0 non-TDM tail-reset behavior. A change to the SIA0 PGR2
    placement / tail-reset logic (the class of change made in #8417) flips these
    markers and surfaces a snapshot diff in CPU PR CI — the signal that was
    previously absent for the SIA0 path.
    """
    results = emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e, **_tail_lr_reset(s)} for (b, s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
