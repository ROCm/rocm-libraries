# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from collections import defaultdict
import itertools
from pathlib import Path
import re
import shutil
import subprocess
import types

import pytest
import yaml

from config_harness import emit_kernels_from_config
from tensilelite.Common import IsaVersion
from tensilelite.Component import Component


@pytest.mark.parametrize("transpose_a,transpose_b", [(False, False), (False, True),
                                                      (True, False), (True, True)])
def test_bf16_fallback_codegen(tmp_path, transpose_a, transpose_b):
    source = Path(__file__).parent / "data/test_data/_designed/gfx90a/mac.yaml"
    config = yaml.safe_load(source.read_text())
    config["BenchmarkProblems"][0][0].update(
        DataType="b", DestDataType="s", ComputeDataType="s",
        HighPrecisionAccumulate=True, TransposeA=transpose_a, TransposeB=transpose_b)
    for entry in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "LocalReadVectorWidth" in entry:
            entry["LocalReadVectorWidth"] = [1]
    path = tmp_path / "bf16-valu.yaml"
    path.write_text(yaml.safe_dump(config))
    kernels = emit_kernels_from_config(path, limit=8, arch="gfx1030", canonical=False)
    assert kernels
    assembler = shutil.which("amdclang++")
    if not assembler:
        pytest.skip("amdclang++ is unavailable")
    for index, (_, assembly, err) in enumerate(kernels):
        assert err == 0 and "v_fma_f32" in assembly and "v_lshlrev_b32" in assembly
        assert "ds_read_u16_d16_hi" in assembly
        asm = tmp_path / f"kernel-{index}.s"
        asm.write_text(assembly)
        subprocess.run(
            [assembler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
             "-mcpu=gfx1030", "-c", str(asm), "-o", str(asm.with_suffix(".o"))],
            check=True, capture_output=True)


@pytest.mark.unit
@pytest.mark.parametrize("isa,selected", [pytest.param((10, 3, 0), True, id="gfx1030"),
                                          pytest.param((10, 1, 0), False, id="gfx1010"),
                                          pytest.param((10, 3, 1), False, id="gfx1031"),
                                          pytest.param((9, 0, 10), False, id="gfx90a"),
                                          pytest.param((12, 0, 1), False, id="gfx1201")])
def test_bf16_fallback_is_selected_for_gfx1030_only(isa, selected):
    """The packed-LDS high-half read the fallback needs is only emitted for gfx1030.

    Any other ISA has to fail codegen rather than emit kernels that read the wrong data.
    """
    fallback = Component.MAC.FMA_BF16_HPA
    states = types.SimpleNamespace(asmCaps=fallback.asmCaps, archCaps={},
                                   kernel={**fallback.kernel, "ISA": IsaVersion(*isa)})
    assert fallback.matches(types.SimpleNamespace(states=states)) is selected


# Valu register map read back from the emitted assembly. A block is one
# (tensor, prefetch buffer, inner unroll) slot, ValuA_X<buffer>_I<iui>. rocIsaPass
# inlines the MAC macros, so the unpack is read from the kernel body itself.
_VALU = r"vgprValu(?P<tensor>[AB])_X(?P<buffer>\d+)_I(?P<iui>\d+)"
_LOCAL_READ = re.compile(rf"^\s*ds_read\w*\s+v\[(?P<reg>{_VALU}(?:\+[^\]]+)?)\]", re.M)
_UNPACK = re.compile(
    rf"^\s*v_\w+\s+v\[(?P<dst>vgprValu[AB]_[^\]]+)\],[^\n]*?"
    rf"v\[(?P<src>{_VALU}(?:\+[^\]]+)?)\]", re.M)


def _index(expression, symbols):
    return sum(int(term) if term.isdigit() else _index(symbols[term], symbols)
               for term in expression.split("+"))


def _valu_traffic(assembly):
    """Registers the local reads fill and the MAC unpack writes, per Valu block."""
    symbols = {}
    for name, value in re.findall(r"^\.set\s+(\w+),\s*(\S+)", assembly, re.M):
        symbols.setdefault(name, value)  # first .set = main loop; the tail loop only rebases it

    def registers(operand):
        low, _, high = operand.partition(":")
        return set(range(_index(low, symbols), _index(high or low, symbols) + 1))

    def block(match):
        return match["tensor"], int(match["buffer"]), int(match["iui"])

    loaded = defaultdict(set)
    for read in _LOCAL_READ.finditer(assembly):
        loaded[block(read)] |= registers(read["reg"])
    unpacked = defaultdict(set)
    for unpack in _UNPACK.finditer(assembly):
        unpacked[block(unpack)] |= registers(unpack["dst"])
    return loaded, unpacked


def _label(block):
    return "Valu%s_X%u_I%u" % block


@pytest.mark.parametrize("prefetch,inner_unroll", [pytest.param(0, 1, id="plr0"),
                                                   pytest.param(1, 1, id="plr1"),
                                                   pytest.param(1, 2, id="plr1-iu2")])
def test_bf16_unpack_keeps_other_blocks_intact(tmp_path, prefetch, inner_unroll):
    """Unpacking a block for the MACs must not overwrite data local reads left in another.

    The other blocks are the buffers PrefetchLocalRead fills ahead of time, the other
    InnerUnroll slots and the other operand: nothing else reloads them before their MACs.
    The blocks one MAC unpacks together also need distinct scratch for their high halves.
    """
    source = Path(__file__).parent / "data/test_data/_designed/gfx90a/mac.yaml"
    config = yaml.safe_load(source.read_text())
    config["BenchmarkProblems"][0][0].update(
        DataType="b", DestDataType="s", ComputeDataType="s", HighPrecisionAccumulate=True)
    fork = {"PrefetchLocalRead": [prefetch], "InnerUnroll": [inner_unroll],
            "LocalReadVectorWidth": [1]}
    forks = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for entry in forks:
        for key in fork.keys() & entry.keys():
            entry[key] = fork[key]
    # The tail loop keeps InnerUnroll > 1 only if it divides the summation size; otherwise it
    # asks for MAC_<T0>x<T1>_X0_OneIUI, a macro only dot2 kernels define.
    forks.append({"AssertSummationElementMultiple": [inner_unroll]})
    path = tmp_path / "bf16-valu-blocks.yaml"
    path.write_text(yaml.safe_dump(config))
    kernels = emit_kernels_from_config(path, limit=8, arch="gfx1030", canonical=False)
    assert kernels
    for name, assembly, err in kernels:
        assert err == 0, f"{name}: assembly generation failed"
        loaded, unpacked = _valu_traffic(assembly)
        # Guard against a vacuous pass: every prefetch buffer and inner unroll is unpacked.
        assert loaded and unpacked, f"{name}: no Valu local read or MAC unpack in the assembly"
        assert {buffer for _, buffer, _ in unpacked} == set(range(prefetch + 1))
        assert {iui for _, _, iui in unpacked} == set(range(inner_unroll))
        assert unpacked.keys() <= loaded.keys()
        for unpacker, written in unpacked.items():
            for victim, held in loaded.items():
                assert unpacker == victim or not written & held, (
                    f"{name}: unpacking {_label(unpacker)} overwrites "
                    f"v{sorted(written & held)} loaded for {_label(victim)}")
        for first, second in itertools.combinations(unpacked, 2):
            shared = unpacked[first] & unpacked[second]
            assert first[1] != second[1] or not shared, (
                f"{name}: unpacking {_label(first)} and {_label(second)} both write v{sorted(shared)}")
