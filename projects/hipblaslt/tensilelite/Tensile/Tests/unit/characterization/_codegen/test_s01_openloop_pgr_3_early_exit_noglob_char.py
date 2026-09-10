################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S01 - KernelWriterAssembly openLoop PGR>=3 early-exit characterization.

Drives the designed PrefetchGlobalRead>=3 config
(``data/test_data/_designed/gfx950/s01_openloop_pgr_3_early_exit_noglob.yaml``)
through the config-driven emit harness. Targets the ``openLoop`` PGR>=3 arm in
``Tensile/KernelWriterAssembly.py``:

  - line 7835: ``endCounter = PGR-1`` first early-exit inside ``if PGR>=3``, and
  - line 7846: the ``SCmpLeU32`` second early-exit (loopCounter<=PGR).

No existing designed config sets ``PrefetchGlobalRead >= 3`` (all are 0/1/2), so
the whole PGR>=3 openLoop arm was previously unexecuted. Solution derivation
gates PGR>=3 behind ``DirectToLds{A,B}``, ``PrefetchLocalRead>=1``, and
``ScheduleIterAlg==3``; this config copies the working gfx950 F8 TN DirectToLds
shape and sets ``PrefetchGlobalRead:[3]`` + ``PrefetchLocalRead:[1]``. The
multi-summation LRO reset arms (loopIdx != unrollIdx) need >1 summation loop and
are not reachable from a single-summation GEMM (Category B, reported honestly by
the probe).

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
    "s01_openloop_pgr_3_early_exit_noglob.yaml",
)


def test_s01_openloop_pgr_3_early_exit_noglob_emits():
    """PGR>=3 openLoop config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s01_openloop_pgr_3_early_exit_noglob_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the PGR>=3 emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
