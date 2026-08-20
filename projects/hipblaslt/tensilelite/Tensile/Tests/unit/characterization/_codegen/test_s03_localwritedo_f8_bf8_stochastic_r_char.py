################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S03 - KernelWriterAssembly localWriteDo FP8 -> F16 conversion characterization.

Drives the designed FP8-A / half-MacType config
(``data/test_data/_designed/gfx950/s03_localwritedo_f8_bf8_stochastic_r.yaml``)
through the config-driven emit harness. Targets the ``localWriteDo`` FP8 ->
F16 LDS-write conversion in ``Tensile/KernelWriterAssembly.py``, the
``elif DataType.isAnyFloat8() and MacDataType.isHalf()`` path:

  - newBlockWidth==0.25 (GRVWA=1) VCvtScaleFP8toF16 arm,
  - newBlockWidth==0.5  (GRVWA=2) sel-select arm, and
  - newBlockWidth large (GRVWA=8) for-vi VCvtScaleFP8toF16 arm.

On gfx950 ``asmCaps["Hascvtf16_fp8_sf32"]`` is True so the VCvtScaleFP8toF16
arms fire; conversion happens on LOCAL WRITE (not ConvertAfterDS) because
DataTypeA is FP8 and MacDataTypeA is half. Forking GlobalReadVectorWidthA over
[1, 2, 8] sweeps all three newBlockWidth arms. The emit call runs
assignDerivedParameters + emission, so the target lines fire during emit.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "s03_localwritedo_f8_bf8_stochastic_r.yaml",
)


def test_s03_localwritedo_f8_bf8_stochastic_r_emits():
    """FP8->F16 localWriteDo config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s03_localwritedo_f8_bf8_stochastic_r_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the FP8 emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
