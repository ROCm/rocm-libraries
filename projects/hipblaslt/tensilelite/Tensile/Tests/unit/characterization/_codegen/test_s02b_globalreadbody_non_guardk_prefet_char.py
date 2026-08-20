################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S02b - KernelWriterAssembly globalReadBody non-guardK prefetch characterization.

Drives the designed non-BufferLoad flat global-read config
(``data/test_data/_designed/gfx942/s02b_globalreadbody_non_guardk_prefet.yaml``)
through the config-driven emit harness. Targets the ``globalReadDo`` (mode=1)
prefetch load arms in ``Tensile/KernelWriterAssembly.py``, specifically the
non-BufferLoad flat global-read address path (line 11810).

The ``config_harness`` derives only ``BenchmarkProblems[0]``; the leading entry
of the config isolates the flat (BufferLoad=False) path so the target lines fire
during ``assignDerivedParameters`` + emission.

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
    "s02b_globalreadbody_non_guardk_prefet.yaml",
)


def test_s02b_globalreadbody_non_guardk_prefet_emits():
    """Flat (BufferLoad=0) prefetch config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s02b_globalreadbody_non_guardk_prefet_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the flat emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
