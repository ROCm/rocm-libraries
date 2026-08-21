################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 StreamK + ClusterDim codegen characterization.

Checks that a StreamK kernel with ClusterDim != [1, 1] emits the cluster
WG-id decode (RemapWorkGroupDone) and skips the raw ttmp9/ttmp7 reread in the
StreamK preLoop. Under clustering, defineAndResources leaves the cluster-decoded
rank in WorkGroup0/1/2; the reread would overwrite it, so the enableCluster
guard skips it.

Uses normal StreamK=3 with ClusterDim=[2, 1]; the guard applies equally to
StreamK modes 3, 4, and 5.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk_cluster.yaml",
)


def test_streamk_cluster_gfx1250_emits_assembly():
    """StreamK=3 + ClusterDim=[2,1]: cluster decode present, preLoop reread skipped."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 arch marker"
        # Cluster WG-id decode arm.
        assert "RemapWorkGroupDone" in src, (
            f"Kernel {base!r}: missing cluster WG-id decode ('RemapWorkGroupDone')"
        )
        # preLoop ttmp reread must be skipped under clustering ("workaround" is
        # unique to that reread block).
        assert "workaround" not in src, (
            f"Kernel {base!r}: ttmp reread emitted under ClusterDim != [1, 1]"
        )
