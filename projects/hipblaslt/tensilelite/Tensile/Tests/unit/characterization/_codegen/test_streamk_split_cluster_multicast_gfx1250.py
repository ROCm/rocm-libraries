# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Assemble real split-K spatial-cluster kernels on gfx1250 without a device."""

from pathlib import Path

import pytest
import yaml

from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIG = Path(__file__).parent / "data/test_data/_designed/gfx1250/streamk_split_cluster_multicast.yaml"


def _check_kernel(base, src):
    assert_assembles(src, base)
    assert "logical StreamK worker = hardware cluster" in src
    assert "physical partial slot = cluster * peers + peer" in src
    assert "flag offset based on physical partial slot" in src
    assert "partials tile offset (high word) for 64-bit SRD" in src
    assert "clamp M tile to the edge" in src
    assert "clamp N tile to the edge" in src
    assert "s[sgprMulticastMaskB]" in src
    assert "retire cooperative tensor_load_to_lds" not in src
    assert "PersistentMC_SkipPrefetchSignal" not in src
    assert "DP fold: rank" not in src
    assert "global_wb" in src and "global_inv" in src
    # The phantom completion gate dominates both the reduction dispatch and
    # every possible partial publication, not just the final D store.
    phantom = src.index("phantom tiles skip the store")
    reduction = src.index("flag offset based on physical partial slot")
    assert phantom < reduction
    # Empty-worker validation precedes the sole prologue arrival.
    initial_check = src.index("Make sure there's work to do")
    initial_arrive = src.index("cluster B-multicast: elect wave 0")
    assert initial_check < initial_arrive
    # Alpha-zero nonowners leave before the first-load wait. Their dedicated
    # wait lies immediately on that path before its persistent-loop exit.
    alpha_wait = src.index("alpha-zero nonowner consumes cluster arrive")
    assert "label_PersistentLoopClose" in src[alpha_wait:alpha_wait + 700]
    close = src.index("label_PersistentLoopClose:")
    close_text = src[close:close + 2000]
    assert close_text.index("s_wait_dscnt 0") < close_text.index("s_barrier_signal -3")


def test_static_streamk_cluster_shapes_and_pgr_assemble():
    results = emit_kernels_from_config(str(_CONFIG), limit=8, arch="gfx1250", canonical=False)
    assert len(results) == 8
    assert_real_gfx1250_kernels(results)
    for base, src, _ in results:
        _check_kernel(base, src)


def test_static_streamk_cluster_shadow_init_assembles(tmp_path):
    config = yaml.safe_load(_CONFIG.read_text())
    for fork in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "ForceDisableShadowInit" in fork:
            fork["ForceDisableShadowInit"] = [False]
    path = tmp_path / "shadow.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert_real_gfx1250_kernels(results)
    for base, src, _ in results:
        _check_kernel(base, src)
        skip = src.index("skip to ShadowInitStart iter b/c numIter==0")
        assert "cluster_barrier wait" in src[skip - 500:skip]


def test_ordinary_streamk_tree_reduction_still_assembles(tmp_path):
    config = yaml.safe_load(_CONFIG.read_text())
    for fork in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "StreamKClusterMulticast" in fork:
            fork["StreamKClusterMulticast"] = [False]
        if "ClusterDim" in fork:
            fork["ClusterDim"] = [[1, 1]]
    path = tmp_path / "ordinary_streamk.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(str(path), limit=2, arch="gfx1250", canonical=False)
    assert len(results) == 2
    assert_real_gfx1250_kernels(results)
    for base, src, _ in results:
        assert_assembles(src, base)
        assert "producer cluster * peers" not in src
        assert "logical StreamK worker = hardware cluster" not in src
        assert "Tree reduction" in src
        assert "reset flag" in src
        assert "global_wb" in src and "global_inv" in src
