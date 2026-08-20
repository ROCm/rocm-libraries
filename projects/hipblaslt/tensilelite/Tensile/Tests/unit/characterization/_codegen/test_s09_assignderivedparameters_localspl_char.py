################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S09 - assignDerivedParameters LocalSplitU / NumWaveSplitK characterization.

Drives the designed non-MI LocalSplitU/NumWaveSplitK config
(``data/test_data/_designed/gfx942/s09_assignderivedparameters_localspl.yaml``)
through the config-driven emit harness. The coverage target is
``Tensile/SolutionStructs/Solution.py`` (probe-confirmed lines 4259, 4260, 4320,
5286, 5287) which fire during ``assignDerivedParameters``.

On the non-MI (dot2/source) derivation path,
``LocalSplitU = 1 if WaveSplitK else WorkGroup[2]`` and
``NumWaveSplitK = WorkGroup[2] if WaveSplitK else 1``, so a non-MI base with
``WorkGroup[2] > 1`` drives ``LocalSplitU > 1`` while ``WaveSplitK=True`` drives
``NumWaveSplitK > 1``. Forking BufferStore/StoreRemapVectorWidth spreads the
reject/derivation sites. emit runs assignDerivedParameters + emission, so the
target lines fire during the emit call.

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
    "s09_assignderivedparameters_localspl.yaml",
)


def test_s09_assignderivedparameters_localspl_emits():
    """LocalSplitU/NumWaveSplitK config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s09_assignderivedparameters_localspl_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
