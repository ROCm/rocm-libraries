################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S11a - GlobalWriteBatch _prolog alpha-before-loadC characterization.

Drives the designed alpha-before-loadC config
(``data/test_data/_designed/gfx942/s11a_prolog_alpha_before_loadc_packe.yaml``)
through the config-driven emit harness. Targets the ``_prolog`` alpha handling in
``Tensile/Components/GlobalWriteBatch.py``:

  - the ``codeMulAlpha`` not-None branch (int8 MI-out -> fp32 replaceHolder):
    line 836 (``srcRegName = rh.getParams()[2].getCompleteRegName()``) and
    line 837 (``module.add(VCvtI32toF32(...))``).

The alphaBeforeLoadC path requires MIArchVgpr=True, applyAlpha, beta, and
StorePriorityOpt (KernelWriterAssembly.py:16506-16520); 836/837 additionally
require I8 in / single compute / GlobalSplitU==1. ``emit_kernels_from_config``
runs assignDerivedParameters + emission, so the target lines fire during emit.

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
    "s11a_prolog_alpha_before_loadc_packe.yaml",
)


def test_s11a_prolog_alpha_before_loadc_packe_emits():
    """alphaBeforeLoadC int8 config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s11a_prolog_alpha_before_loadc_packe_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
