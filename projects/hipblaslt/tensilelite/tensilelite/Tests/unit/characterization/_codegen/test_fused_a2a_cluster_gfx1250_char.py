# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fused GEMM+A2A (FusedA2A: GemmA2A) with a workgroup cluster.

Emits fused_a2a_cluster.yaml with ClusterDim [4, 2] and [2, 4]: both kernels
assemble and carry the work-group remap, no cluster barrier follows the
PUSH/local gate, and exactly one of them splits the remap between the AIDs.
"""

import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_CLUSTER = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "common", "comm", "gfx1250",
    "fused_a2a_cluster.yaml"))

_REMAP_WAIT = "wait FusedAM for the WG remap"
_PUSH_GATE = "AM_tiles <= WorkGroup0?"


@pytest.fixture(scope="module", params=[[4, 2], [2, 4]], ids=["cluster_dim0", "cluster_dim1"])
def clustered(request):
    """(ClusterDim, the two kernels of fused_a2a_cluster.yaml emitted with it)."""
    results = emit_kernels_from_config(_CLUSTER, limit=8, arch=_ARCH, cluster_dim=request.param)
    assert len(results) == 2, [b for (b, _s, _e) in results]
    for base, src, err in results:
        assert err == 0
        assert_assembles(src, base)
    return request.param, results


def test_fused_a2a_gfx1250_cluster_wg_remap(clustered, snapshot):
    _cluster_dim, results = clustered
    aids = []
    for base, src, _err in results:
        assert _REMAP_WAIT in src, f"Kernel {base!r}: no work-group remap"
        epilogue = src[src.index(_PUSH_GATE):]
        assert not re.search(r"^s_barrier_\w+ -3", epilogue, re.M), (
            f"Kernel {base!r}: cluster barrier past the PUSH/local gate")
        aids.append(len(re.findall(r"^s_sendmsg_rtn_b32", src, re.M)))
    assert sorted(aids) == [0, 1]
    assert sorted(b for (b, _s, _e) in results) == snapshot
