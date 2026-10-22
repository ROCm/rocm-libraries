# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Assemble real split-K spatial-cluster kernels on gfx1250 without a device."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIG = Path(__file__).parent / "data/test_data/_designed/gfx1250/streamk_split_cluster_multicast.yaml"


def _assembly_code(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(code for line in src.splitlines()
                     if (code := line.split("//", 1)[0].strip()))


def _assert_scc0_jump_to_close(code):
    # Check the emitted long branch's polarity, PC operands and actual target.
    # No instruction execution: SCC==1 skips the jump; SCC==0 reaches loop close.
    branch = re.match(
        r"s_cbranch_scc1 (?P<skip>label_\w+)\n"
        r"s_getpc_b64 s\[(?P<lo>\d+):(?P<hi>\d+)\]\n"
        r"s_add_i32 (?P<offset>s\d+), label_PersistentLoopClose, 4\n"
        r"s_add_u32 s(?P=lo), s(?P=lo), (?P=offset)\n"
        r"s_addc_u32 s(?P=hi), s(?P=hi), 0\n"
        r"s_setpc_b64 s\[(?P=lo):(?P=hi)\]\n"
        r"(?P=skip):\n", code)
    assert branch, "SCC==0 must branch to PersistentLoopClose"
    return branch.end()


def _assert_phantom_completion(src):
    code = _assembly_code(src)
    gate = re.search(r"^s_cmp_eq_u32 s\[sgprPersistentPhantomTile\], 0\n", code, re.M)
    assert gate
    after_gate = gate.end() + _assert_scc0_jump_to_close(code[gate.end():])
    close = code.index("\nlabel_PersistentLoopClose:")
    # Subtile epilogues can put store bodies after loop close. Every entry
    # reference must remain in the guarded completion interval, and loop close
    # must jump over those bodies rather than falling through into them.
    deferred = list(re.finditer(r"^(label_GW_\w+_Deferred):\n", code, re.M))
    kernel_end = code.index("\nlabel_KernelEnd:")
    if deferred:
        assert close < deferred[0].start() < kernel_end
        for label in deferred:
            entries = list(re.finditer(rf"^.*\b{label[1]}(?:,|$).*$", code, re.M))
            assert entries and all(after_gate <= entry.start() < close for entry in entries)
        assert re.search(
            r"s_getpc_b64 s\[(?P<lo>\d+):(?P<hi>\d+)\]\n"
            r"s_add_i32 (?P<offset>s\d+), label_KernelEnd, 4\n"
            r"s_add_u32 s(?P=lo), s(?P=lo), (?P=offset)\n"
            r"s_addc_u32 s(?P=hi), s(?P=hi), 0\n"
            r"s_setpc_b64 s\[(?P=lo):(?P=hi)\]\n$", code[close:deferred[0].start()])
    # Check concrete stores and workspace reads, including partial publication
    # and flag updates. Deferred bodies are covered by their guarded entries.
    completion = list(re.finditer(
        r"^(?:(?:buffer|global)_store_\w+ .*|buffer_load_\w+ .*sgprSrdWS.*)$", code, re.M))
    assert completion
    assert all(after_gate <= op.start() < close or
               (deferred and deferred[0].start() < op.start() < kernel_end)
               for op in completion)


def _assert_multicast_completion(src):
    code = _assembly_code(src)
    owner = re.search(
        r"^s_cmp_eq_u32 s\[sgprStreamKLocalStart\], 0\n"
        r"s_cbranch_scc1 (?P<owner>label_SKClusterAlphaOwner\w*)\n"
        r"s_barrier_wait -3\n", code, re.M)
    assert owner, "alpha-zero nonowners must consume their cluster arrival"
    end = owner.end() + _assert_scc0_jump_to_close(code[owner.end():])
    assert code[end:].startswith(owner["owner"] + ":\n")
    assert end < code.index("s_barrier_wait -3", end) < code.index("label_PersistentLoopClose:")
    close = re.search(
        r"^label_PersistentLoopClose:\n"
        r"s_cmp_ge_u32 s\[sgprPersistentIteration\], s\[sgprPersistentIterationEnd\]\n"
        r"s_cbranch_scc1 (?P<done>label_\w+)\n", code, re.M)
    assert close
    # Delimit the next-tile arrival by its actual no-next-tile branch target.
    arrival = code[close.end():code.index(close["done"] + ":", close.end())]
    assert arrival.startswith("s_wait_dscnt 0\ns_barrier_signal -1\ns_barrier_wait -1\n")
    signal = re.search(r"s_cbranch_scc0 (?P<skip>label_\w+)\n"
                       r"s_barrier_signal -3\n(?P=skip):\n", arrival)
    assert signal, "only wave zero signals after the LDS drain and workgroup barrier"


def _check_kernel(base, src):
    assert_assembles(src, base)
    metadata = yaml.safe_load(src.split(".amdgpu_metadata", 1)[1].split(".end_amdgpu_metadata", 1)[0])
    symbol = metadata["amdhsa.kernels"][0][".name"]
    assert "_PLAV2_" in symbol
    assert "_SKCM" not in symbol
    config = metadata["custom.config"]
    assert config["TileProcessingStrategy"] == "StreamK"
    assert config["WorkAssignment"] == "StaticGrid"
    assert config["ClusterDim"][0] > 1
    assert config["InternalSupportParams"]["PersistentLoopArgsVersion"] == 2
    assert "logical StreamK worker = hardware cluster" in src
    assert "physical partial slot = cluster * peers + peer" in src
    assert "flag offset based on physical partial slot" in src
    assert "partials tile offset (high word) for 64-bit SRD" in src
    assert "clamp M tile to the edge" in src
    assert "clamp N tile to the edge" in src
    multicast = config["ClusterDim"][1] > 1
    assert ("s[sgprMulticastMaskB]" in src) is multicast
    assert "retire cooperative tensor_load_to_lds" not in src
    assert "PersistentMC_SkipPrefetchSignal" not in src
    assert "DP fold: rank" not in src
    assert "global_wb" in src and "global_inv" in src
    _assert_phantom_completion(src)
    if multicast:
        # Empty-worker validation precedes the sole prologue arrival.
        initial_check = src.index("Make sure there's work to do")
        initial_arrive = src.index("cluster B-multicast: elect wave 0")
        assert initial_check < initial_arrive
        _assert_multicast_completion(src)
    else:
        assert "MulticastMask" not in src
        assert "s_barrier_signal -3" not in src
        assert "s_barrier_wait -3" not in src
        assert "alpha-zero nonowner consumes cluster arrive" not in src


def test_static_streamk_cluster_shapes_and_pgr_assemble():
    results = emit_kernels_from_config(str(_CONFIG), limit=18, arch="gfx1250", canonical=False)
    assert len(results) == 18
    assert_real_gfx1250_kernels(results)
    for base, src, _ in results:
        _check_kernel(base, src)


@pytest.mark.parametrize("cluster", [[2, 1], [2, 2]])
def test_static_streamk_cluster_shadow_init_assembles(tmp_path, cluster):
    config = yaml.safe_load(_CONFIG.read_text())
    for fork in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "ClusterDim" in fork:
            fork["ClusterDim"] = [cluster]
        if "ForceDisableShadowInit" in fork:
            fork["ForceDisableShadowInit"] = [False]
    path = tmp_path / "shadow.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert_real_gfx1250_kernels(results)
    for base, src, _ in results:
        _check_kernel(base, src)
        skip = src.index("skip to ShadowInitStart iter b/c numIter==0")
        assert ("cluster_barrier wait" in src[skip - 500:skip]) is (cluster[1] > 1)
        from Tensile.CustomKernels import getCustomKernelConfig
        (tmp_path / (base + ".s")).write_text(src)
        descriptor = getCustomKernelConfig(base, {}, str(tmp_path))
        assert descriptor["InternalSupportParams"]["PersistentLoopArgsVersion"] == 2
        assert descriptor["CustomKernel"]["grid"] == ["PersistentGrid", "One", "One"]
        assert any(arg["semantic"] == "AddressFlags" for arg in descriptor["CustomKernel"]["args"])


def test_ordinary_streamk_tree_reduction_still_assembles(tmp_path):
    config = yaml.safe_load(_CONFIG.read_text())
    for fork in config["BenchmarkProblems"][0][1]["ForkParameters"]:
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
