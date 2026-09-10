################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S04a - KernelWriterAssembly globalWriteElements MBSK/SingleB characterization.

Drives the designed AdaptiveGemmGSUA=1 config
(``data/test_data/_designed/gfx942/s04a_globalwriteelements_mbsk_singleb.yaml``)
through the config-driven emit harness. Targets the AdaptiveGemmGSUA==1
else-branch label-wiring arms in ``Tensile/KernelWriterAssembly.py``
(``globalWriteElements``):

  - the per-algorithm mode selection (MultipleBuffer / MultipleBufferSingleKernel
    / SingleBuffer), including the ``DataType.isDouble()`` guard, and
  - the MBSK / MB label emission loop plus its AdaptiveGemmGSUA==1 tail restore.

The config forks all three GSU algorithms with AdaptiveGemmGSUA=1 and
GlobalSplitU>0 (=> noGSUBranch False) using a half datatype (MBSK rejects
double). ``emit`` runs assignDerivedParameters + emission, so these lines fire
during the emit call.

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
    "s04a_globalwriteelements_mbsk_singleb.yaml",
)


def test_s04a_globalwriteelements_mbsk_singleb_emits():
    """AdaptiveGemmGSUA=1 GSU-algo fork config emits kernels with err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s04a_globalwriteelements_mbsk_singleb_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
