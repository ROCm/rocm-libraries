################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S07 - KernelWriter UseF32XEmulation register-layout characterization.

Drives the designed XF32-emulation config
(``data/test_data/_designed/gfx950/s07_usef32xemulation_register_layout.yaml``)
through the config-driven emit harness. Targets the TF32/F32X emulation
register-layout cluster in ``Tensile/KernelWriter.py``:

  - 9006 : doPackPreSchedulingThisLoop assignment (DirectToLds==1 + numItersPLR),
  - 9028 : the ``UseDirect32XEmulationInterleaveTreg`` full-pack arm,
  - 9035 : the ``numV = numVForIndexTranspose`` (Interleave==False) else, and
  - 9040 : the ``UseMFMAF32XEmulation`` useTransposeCodeThis assignment.

gfx950 has BOTH HasF32XEmulation and HasMFMA, so ``UseMFMAF32XEmulation`` is
True and ``ForceUnrollSubIter`` survives, keeping ``UsePLRPack`` alive so that
``doFullPackCodePrefetch`` becomes True and the full-pack arm fires during
emission. (On gfx1250 these lines are dead: no MFMA -> UsePLRPack zeroed.)

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
    "s07_usef32xemulation_register_layout.yaml",
)


def test_s07_usef32xemulation_register_layout_emits():
    """XF32-emulation config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s07_usef32xemulation_register_layout_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
