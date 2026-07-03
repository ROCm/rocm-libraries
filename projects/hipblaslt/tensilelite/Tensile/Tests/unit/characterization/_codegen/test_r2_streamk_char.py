################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R2 — gfx942 StreamK family codegen characterization.

Exercises the uncovered branches of ``Tensile/Components/StreamK.py``
(methodology-A missing ranges: 202, 316, 326-409, 464+ and others)
by emitting SGEMM kernels that sweep the StreamK knob family:

  StreamK=[1, 2, 3] x StreamKAtomic=[0, 1] x StreamKXCCMapping=[0]

StreamK splits the K-loop across workgroups (modes 1/2/3 vary the
scheduling/fixup strategy). StreamKAtomic=1 triggers the atomic
reduction path (non-zero kernel["StreamKAtomic"]). Both paths
exercise heavily-missed code in StreamK.py.

CPU-only: no GPU required. The emit harness instantiates rocisa and
runs the Python+rocisa codegen stack without compiling or launching
any GPU kernels.
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
    "streamk.yaml",
)


def test_r2_streamk_gfx942_emits_assembly():
    """StreamK family sweep emits real gfx942 assembly with err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got 0"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 target"
        assert base.startswith("Cijk_"), f"Kernel {base!r} has unexpected prefix"


def test_r2_streamk_gfx942_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the StreamK sweep emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
