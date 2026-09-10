################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S07 - KernelWriter flat (non-buffer) global-read address VGPR characterization.

Drives the designed flat-addressing config
(``data/test_data/_designed/gfx942/s07_non_bufferload_flat_global_read.yaml``)
through the config-driven emit harness. Targets the flat (non-BufferLoad)
global-read address VGPR arms in ``Tensile/KernelWriter.py``:

  - the else-of-BufferLoad ``numVgprGlobalReadAddressesB`` assignment,
  - the ``globalReadIncsUseVgpr`` (flat) ``numVgprGlobalReadIncsB`` assignment,
  - the ``startVgprGlobalReadAddressesA/B`` flat else-branch.

``KernelWriter.py`` derives the flat global-read address layout when
``BufferLoad=False``, so a simple fp32 NN GEMM with ``BufferLoad=False`` forces
those arms during ``assignDerivedParameters`` + emission. The MXSA/MXSB flat
variants are unreachable here (MXBlock forces BufferLoad on gfx942).

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
    "s07_non_bufferload_flat_global_read.yaml",
)


def test_s07_non_bufferload_flat_global_read_emits():
    """Flat (BufferLoad=0) config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s07_non_bufferload_flat_global_read_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the flat emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
