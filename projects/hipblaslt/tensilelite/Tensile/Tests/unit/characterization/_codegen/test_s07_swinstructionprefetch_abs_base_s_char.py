################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S07 - KernelWriter SwInstructionPrefetch absolute-base characterization.

Drives the designed gfx1250 absolute-base config
(``data/test_data/_designed/gfx1250/s07_swinstructionprefetch_abs_base_s.yaml``)
through the config-driven emit harness. Targets the SwInstructionPrefetch
absolute-base SGPR reservation block in ``Tensile/KernelWriter.py``:

  - the abs-base guard gated on ``swpAbsRequested`` (Absolute prefetch resolved
    for gfx1250 non-StreamK non-f64) and version == (12,5,0), and
  - the ``PreloadKernArgs`` preload-guard while loop that advances the base
    past the kernarg preload region.

With ``SwInstructionPrefetch=Absolute(2)`` and ``PreloadKernArgs=True`` on a
gfx1250 non-StreamK F4 problem, both arms fire during emission.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "s07_swinstructionprefetch_abs_base_s.yaml",
)


def test_s07_swinstructionprefetch_abs_base_s_emits():
    """Absolute-base (SwInstructionPrefetch=2) config emits kernels, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx1250" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s07_swinstructionprefetch_abs_base_s_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the abs-base emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
