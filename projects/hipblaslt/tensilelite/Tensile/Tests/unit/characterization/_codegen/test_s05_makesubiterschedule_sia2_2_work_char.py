################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S05 - KernelWriter _makeSubIterSchedule SIA2 pack-path characterization.

Drives the designed SIA2 (2-workgroup interleave) config
(``data/test_data/_designed/gfx942/s05_makesubiterschedule_sia2_2_work.yaml``)
through the config-driven emit harness. Targets the ``scheduleIterAlg==2``
block in ``Tensile/KernelWriter.py`` -- specifically the packItems-non-empty
sub-path (pack module split / coalesced-read distribution and pack scheduling
inside the mfma loop).

``TransposeLDS=0`` forces ``UnrollMajorLDSA/B=0`` so that ``instPerPackA/B`` is
non-zero (bf16, HasEccHalf on gfx942), making the pack items non-empty and
firing the pack-distribution lines during emission (assignDerivedParameters +
emission both run in the emit call).

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "s05_makesubiterschedule_sia2_2_work.yaml",
)


def test_s05_makesubiterschedule_sia2_2_work_emits():
    """SIA2 pack-path config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s05_makesubiterschedule_sia2_2_work_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the SIA2 emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
