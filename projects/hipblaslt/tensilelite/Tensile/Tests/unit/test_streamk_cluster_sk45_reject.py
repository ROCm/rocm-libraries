#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit test: StreamK dynamic (SK4) / hybrid (SK5) modes reject ClusterDim.
#
# WG-cluster support is StreamK==3-only (the [C,1] barrier reduction and the DP
# cooperative B-multicast are both SK3 features). There is no cluster-load /
# reduction implementation for the dynamic (SK4) or hybrid (SK5) work-queue
# modes, so Solution.assignDerivedParameters rejects StreamK in {4,5} with
# ClusterDim != [1,1] outright (rather than emitting an unusable cluster kernel
# whose decoded cluster WG-id no feature consumes).
#
# These tests drive the real config -> Solution derivation path and assert:
#   * SK4/SK5 + ClusterDim != [1,1] -> 0 derived solutions (the reject fires); and
#   * the SAME config with ClusterDim = [1,1] still derives solutions, proving
#     the differentiator is the cluster (not some unrelated SK4/SK5 reject).
#
# Usage:
#   pytest test_streamk_cluster_sk45_reject.py -v
################################################################################

import copy
import os
import sys

import pytest

pytestmark = pytest.mark.unit

_DESIGNED = os.path.join(
    os.path.dirname(__file__), "characterization",
    "_codegen", "data", "test_data", "_designed", "gfx1250")
# A known-good gfx1250 StreamK=3 + ClusterDim config; we only re-fork StreamK
# (and, for the control, ClusterDim) on top of it.
_BASE = os.path.join(_DESIGNED, "streamk_cluster_coop_load.yaml")

_ARCH = "gfx1250"


def _write_variant(tmp_path, name, overrides):
    """Copy _BASE, replacing/appending the given fork parameter values."""
    from Tensile import LibraryIO
    import yaml

    cfg = copy.deepcopy(LibraryIO.read(_BASE))
    fork = cfg["BenchmarkProblems"][0][1]["ForkParameters"]
    for key, val in overrides.items():
        replaced = False
        for entry in fork:
            if key in entry:
                entry[key] = val
                replaced = True
                break
        if not replaced:
            fork.append({key: val})
    out = tmp_path / name
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None)
    return str(out)


def _derive_states(cfg_path):
    from config_harness import derive_states
    return derive_states(cfg_path, arch=_ARCH, limit_solutions=8)


# The _BASE config is StreamK=3 + ClusterDim + StreamKForceDPOnly=1. Both are
# SK3-only concepts, so when we re-fork StreamK to 4/5 we also clear
# StreamKForceDPOnly (=0) to isolate the ClusterDim as the sole reject
# differentiator (otherwise SK4/5 + FDPO=1 would itself be rejected, confounding
# the control below).
@pytest.mark.parametrize("sk", [4, 5])
def test_sk45_cluster_rejected(tmp_path, sk):
    """StreamK dynamic/hybrid + ClusterDim != [1,1] derives no solutions."""
    cfg = _write_variant(tmp_path, f"sk{sk}_cluster.yaml",
                         {"StreamK": [sk], "StreamKForceDPOnly": [0]})
    assert _derive_states(cfg) == [], (
        f"StreamK={sk} with ClusterDim != [1,1] must be rejected "
        "(cluster support is SK3-only)")


@pytest.mark.parametrize("sk", [4, 5])
def test_sk45_without_cluster_still_valid(tmp_path, sk):
    """Control: the same config with ClusterDim=[1,1] still derives solutions,
    so the reject above is caused by the cluster, not an unrelated SK4/SK5
    constraint."""
    cfg = _write_variant(tmp_path, f"sk{sk}_nocluster.yaml",
                         {"StreamK": [sk], "ClusterDim": [[1, 1]],
                          "StreamKForceDPOnly": [0]})
    assert _derive_states(cfg), (
        f"StreamK={sk} without ClusterDim should still derive solutions")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
