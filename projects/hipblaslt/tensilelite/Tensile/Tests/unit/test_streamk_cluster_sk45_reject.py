#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit test: StreamK dynamic-queue and hybrid assignments reject ClusterDim.
#
# WG-cluster support requires StaticGrid assignment. There is no cluster-load /
# reduction implementation for DynamicWorkQueue or Hybrid assignment,
# so Solution.assignDerivedParameters rejects either assignment with
# ClusterDim != [1,1] outright (rather than emitting an unusable cluster kernel
# whose decoded cluster WG-id no feature consumes).
#
# These tests drive the real config -> Solution derivation path and assert:
#   * queue/hybrid + ClusterDim != [1,1] -> 0 derived solutions; and
#   * the SAME config with ClusterDim = [1,1] still derives solutions, proving
#     the differentiator is the cluster.
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
# A known-good gfx1250 persistent DataParallel/StaticGrid cluster config.
# Override both policy dimensions and, for the control, ClusterDim.
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


# Select StreamK explicitly so the inherited DataParallel policy cannot cause
# an unrelated rejection before the cluster guard is reached.
@pytest.mark.parametrize("assignment", ["DynamicWorkQueue", "Hybrid"])
def test_sk45_cluster_rejected(tmp_path, assignment):
    """StreamK dynamic/hybrid + ClusterDim != [1,1] derives no solutions."""
    cfg = _write_variant(tmp_path, f"{assignment}_cluster.yaml",
                         {"TileProcessingStrategy": ["StreamK"],
                          "WorkAssignment": [assignment]})
    assert _derive_states(cfg) == [], (
        f"StreamK/{assignment} with ClusterDim != [1,1] must be rejected "
        "(cluster support requires StaticGrid)")


@pytest.mark.parametrize("assignment", ["DynamicWorkQueue", "Hybrid"])
def test_sk45_without_cluster_still_valid(tmp_path, assignment):
    """Control: the same config with ClusterDim=[1,1] still derives solutions,
    so the reject above is caused by the cluster, not an unrelated policy
    constraint."""
    cfg = _write_variant(tmp_path, f"{assignment}_nocluster.yaml",
                         {"TileProcessingStrategy": ["StreamK"],
                          "WorkAssignment": [assignment], "ClusterDim": [[1, 1]]})
    assert _derive_states(cfg), (
        f"StreamK/{assignment} without ClusterDim should still derive solutions")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
