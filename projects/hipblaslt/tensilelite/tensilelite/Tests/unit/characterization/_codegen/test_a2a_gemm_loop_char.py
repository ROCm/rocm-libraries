# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 A2A+GEMM (FusedA2A: A2AGemm) codegen characterization.

Checks that a FusedA2A: A2AGemm solution emits the shard loop, the enqueuer
election and the SDMA ring instructions, and that the result assembles for
gfx950.
"""

import os

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_COMM = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,  # characterization/
        os.pardir,  # unit/
        os.pardir,  # tensilelite/Tests/
        "common", "comm", "gfx950",
    )
)
_CONFIGS = [os.path.join(_COMM, c) for c in ("a2a_gemm_loop.yaml", "a2a_gemm_loop_s2.yaml")]

_A2A_GEMM_MARKERS = (
    "label_A2AShardLoopBegin:",
    "label_A2AWaitFlag:",
    "label_A2ATransitionEnd:",
    "label_A2ASkipEnqueue:",
    "s_atomic_inc",
    "s_atomic_umax_x2",
)


@pytest.mark.parametrize("config", _CONFIGS, ids=os.path.basename)
def test_a2a_gemm_gfx950_emits_assembly(config):
    results = emit_kernels_from_config(config, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), [
        (b, e) for (b, _s, e) in results if e != 0
    ]
    for base, src, _err in results:
        assert_assembles(src, base)
        missing = [m for m in _A2A_GEMM_MARKERS if m not in src]
        assert not missing, f"Kernel {base!r}: missing A2AGemm markers {missing}"


def test_a2a_gemm_gfx950_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for every emitted A2AGemm kernel."""
    digest = sorted(
        (
            {"basename": b, "err": e}
            for config in _CONFIGS
            for (b, _s, e) in emit_kernels_from_config(config, limit=8, arch=_ARCH)
        ),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
