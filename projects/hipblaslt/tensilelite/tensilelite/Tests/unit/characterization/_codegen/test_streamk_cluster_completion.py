# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check real StreamK completion branches and assembly without executing instructions."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config

pytestmark = pytest.mark.unit
_COMMON = Path(__file__).resolve().parents[3] / "common"


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
    deferred = list(re.finditer(r"^(label_(?:GW|Fixup)_\w+_Deferred):\n", code, re.M))
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
    signal = re.search(r"v_readfirstlane_b32 (?P<elect>s\d+), v\[vgprSerial(?:[-+]\d+)?\]\n"
                       r"(?:s_(?:delay|wait)_alu [^\n]*\n)*"
                       r"s_cmp_eq_u32 (?P=elect), 0\n"
                       r"s_cbranch_scc0 (?P<skip>label_\w+)\n"
                       r"s_barrier_signal -3\n(?P=skip):\n", arrival)
    assert signal, "only wave zero signals after the LDS drain and workgroup barrier"


@pytest.mark.parametrize("feature,cluster,pgr,tree", [
    ("dense", [2, 1], 1, 0),
    ("dense", [1, 2], 1, 0),
    ("dense", [1, 8], 2, 1),
    ("dense", [2, 2], 1, 1),
    ("dense", [4, 1], 2, 0),
    ("single_wave", [2, 2], 2, 1),
    ("subtile", [2, 2], 1, 1),
    ("sparse_a", [2, 2], 1, 0),
    ("sparse_b", [2, 2], 1, 1),
    ("store_remap", [2, 1], 1, 1),
])
def test_streamk_cluster_completion_assembles(tmp_path, feature, cluster, pgr, tree):
    fixture = {
        "subtile": "streamk/gfx1250/core/sk_bf16gemm_tdm_subtile.yaml",
        "sparse_a": "sparse/gfx1250/spmm_tdm_multicast.yaml",
        "sparse_b": "sparse/gfx1250/spmm_tdm_multicast.yaml",
    }.get(feature, "streamk/gfx1250/static_streamk_cluster_multicast_boundaries.yaml")
    config = yaml.safe_load((_COMMON / fixture).read_text())
    problem = config["BenchmarkProblems"][int(feature == "sparse_b")]
    config["BenchmarkProblems"] = [problem[:2]]
    group = problem[1]
    values = {key: value[:1] for entry in group.get("ForkParameters", [])
              for key, value in entry.items()}
    values.update(TileProcessingStrategy=["StreamK"], WorkAssignment=["StaticGrid"],
                  ClusterDim=[cluster], PrefetchAcrossPersistent=[0], GlobalSplitU=[0],
                  PrefetchGlobalRead=[pgr], StreamKFixupTreeReduction=[tree])
    if feature == "single_wave":
        values["MatrixInstruction"] = [[16, 16, 32, 1, 1, 1, 1, 1, 1]]
    if feature == "store_remap":
        values["StoreRemapVectorWidth"] = [4]
    group["ForkParameters"] = [{key: value} for key, value in values.items()]
    path = tmp_path / "completion.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    assert_real_gfx1250_kernels(results)
    for base, source, _ in results:
        assert_assembles(source, base)
        _assert_phantom_completion(source)
        assert ("Tree reduction" in source) == bool(tree)
        _assert_multicast_completion(source)
        assert "s[sgprtdmBGroup1], s[sgprtdmBGroup1], s[sgprMulticastMaskB]" in source
        a_mask = "s[sgprtdmAGroup1], s[sgprtdmAGroup1], s[sgprMulticastMaskA]"
        assert (a_mask in source) == (cluster[1] > 1)
        if not tree:
            code = _assembly_code(source)
            fixup = code[code.index("\nlabel_SK_Fixup:\n"):]
            flag = re.search(r"^buffer_load_b32 [^,\n]+, [^,\n]+, s\[(?P<srd>\d+):\d+\], "
                             r"(?P<offset>s\d+) ", fixup, re.M)
            assert flag, "linear fixup must load a producer flag"
            before = fixup[:flag.start()]
            shift = (cluster[0] * cluster[1]).bit_length() - 1
            slot = re.search(rf"^s_lshl_b32 {flag['offset']}, (?P<producer>s\d+), {shift}\n"
                             rf"s_add_u32 {flag['offset']}, {flag['offset']}, s\[sgprStreamKClusterPeer\]\n"
                             rf"s_lshl_b32 {flag['offset']}, {flag['offset']}, 2\n", before, re.M)
            assert slot, "flag load must use 4 * (logical producer * peers + peer)"
            assert re.search(rf"^s_add_u32 {slot['producer']}, {slot['producer']}, 1$", fixup, re.M)
            assert re.search(rf"^s_mov_b64 s\[{flag['srd']}:\d+\], "
                             r"s\[sgprAddressFlags:sgprAddressFlags\+1\]$", before, re.M)
            # Flag descriptor setup must not overwrite the scalar byte offset.
            writes = re.findall(r"^\w+ s(?:(\d+)|\[(\d+):(\d+)\]),", before[slot.end():], re.M)
            assert all(int(flag["offset"][1:]) not in range(int(one or lo), int(one or hi) + 1)
                       for one, lo, hi in writes)
