################################################################################
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""S00 - KernelWriterAssembly graTileOffsets swizzled-A LSU characterization.

Drives the designed swizzled-A + LocalSplitU config
(``data/test_data/_designed/gfx942/s00_gratileoffsets_packed_index_extr.yaml``)
through the config-driven emit harness. Targets the ``graTileOffsets``
``isSwizzled=True`` branch in ``Tensile/KernelWriterAssembly.py``, specifically
the LocalSplitU (LSU) sub-block:

  - 3649 ``tmpVgprRes = None``
  - 3656 ``module.add(vectorStaticDivide(wave_id, "Serial", ...))``

These require ``SwizzleTensorA`` (isSwizzled) together with ``LocalSplitU>1``
(``WorkGroup[2]>1``). The packed-index targets in the same branch need
``len(PackedIndices)>1``, which a plain GEMM ProblemType (single free index per
tensor) cannot produce, so those remain Category B for emit.

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
    "s00_gratileoffsets_packed_index_extr.yaml",
)


def test_s00_gratileoffsets_packed_index_extr_emits():
    """Swizzled-A + LSU config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s00_gratileoffsets_packed_index_extr_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
