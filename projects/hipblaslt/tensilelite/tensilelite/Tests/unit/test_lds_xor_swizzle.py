# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LdsXorSwizzle: XOR LDS layout for gfx11 WMMA kernels without row padding.

CPU-only: derives solutions and emits assembly through config_harness.
"""

import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config, solutions_from_config

from tensilelite.Common.DataType import DataType
from tensilelite.KernelWriter import KernelWriter

ARCH = "gfx1151"

_PROBLEM_TYPE = {
    "OperationType": "GEMM",
    "DataType": "b",
    "DestDataType": "b",
    "ComputeDataType": "s",
    "HighPrecisionAccumulate": True,
    "TransposeA": True,
    "TransposeB": False,
    "UseBeta": True,
    "Batched": True,
}

# bf16 TN MT128x128x64, 8 waves: the shape the swizzle was built for.
_FORKS = {
    "MatrixInstruction": [[16, 16, 16, 1, 1, 4, 2, 2, 4]],
    "DepthU": [64],
    "PrefetchGlobalRead": [2],
    "PrefetchLocalRead": [0],
    "1LDSBuffer": [1],
    "ScheduleIterAlg": [3],
    "TransposeLDS": [1],
    "LocalReadVectorWidth": [16],
    "GlobalReadVectorWidthA": [8],
    "GlobalReadVectorWidthB": [8],
    "LdsPadA": [0],
    "LdsPadB": [0],
    "LdsBlockSizePerPadA": [0],
    "LdsBlockSizePerPadB": [0],
    "LdsXorSwizzle": [True],
    # Match the shipped gfx1151 MT128x128x64 solution so the kernel fits 256 VGPRs.
    "ClusterLocalRead": [0],
    "SourceSwap": [True],
    "MaxOccupancy": [40],
    "WorkGroup": [[64, 4, 1]],
}


def _config(tmp_path, **overrides):
    forks = dict(_FORKS)
    forks.update({k: [v] for k, v in overrides.items()})
    config = {
        "GlobalParameters": {"MinimumRequiredVersion": "5.0.0", "CpuThreads": 1},
        "BenchmarkProblems": [[
            dict(_PROBLEM_TYPE),
            {
                "InitialSolutionParameters": None,
                "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
                "ForkParameters": [{k: v} for k, v in forks.items()],
                "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [4096, 4096, 1, 4608]}]}],
            },
        ]],
    }
    path = tmp_path / "lds_xor_swizzle.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _emit(tmp_path, **overrides):
    results = emit_kernels_from_config(_config(tmp_path, **overrides), limit=1, arch=ARCH, canonical=False)
    assert len(results) == 1, "expected exactly one kernel"
    base, src, err = results[0]
    assert err == 0, f"emit failed for {base}"
    assert_assembles(src, base)
    return src


def _lds_bytes(src):
    return int(re.search(r"amdhsa_group_segment_fixed_size (\d+)", src).group(1))


def _geometry(depthU):
    kernel = {"DepthU": depthU, "ProblemType": {"DataType": DataType("b")}}
    return KernelWriter.ldsXorSwzGeometry(None, kernel)


@pytest.mark.parametrize("depthU, expected", [(16, (2, 7)), (32, (4, 7)), (64, (8, 7)), (128, (16, 8))])
def test_geometry(depthU, expected):
    assert _geometry(depthU) == expected


@pytest.mark.parametrize("depthU", [16, 32, 64, 128])
@pytest.mark.parametrize("chunk", range(2))
def test_eight_consecutive_rows_hit_distinct_bank_groups(depthU, chunk):
    # gfx11 serves ds_load_b128 as 8 lanes x 16 B (128 B, 32 banks) per cycle; a WMMA
    # operand load has those 8 lanes on 8 consecutive LDS rows, all reading one chunk.
    nChunk, keyShift = _geometry(depthU)
    rowBytes = nChunk * 16
    for firstRow in range(0, 64, 8):
        groups = set()
        for row in range(firstRow, firstRow + 8):
            addr = row * rowBytes
            addr += 16 * (chunk ^ ((addr >> keyShift) & (nChunk - 1)))
            groups.add((addr % 128) // 16)
        assert len(groups) == 8, f"DepthU={depthU} rows {firstRow}..{firstRow + 7}: {groups}"


def test_single_buffer_kernel(tmp_path):
    src = _emit(tmp_path)
    assert _lds_bytes(src) == 128 * 64 * 2 * 2  # one A + one B tile, no padding

    # Write side: LocalWriteAddr is swizzled once (shift by keyShift-4, mask (nChunk-1)<<4);
    # the stores themselves carry no extra VALU.
    assert re.search(r"v_lshrrev_b32 \S+, 3, v\[vgprLocalWriteAddrA\].*LdsXorSwizzle", src)
    assert re.search(r"v_and_b32 \S+, 0x70, .*LdsXorSwizzle: key << 4", src)
    for store in re.findall(r"^\s*ds_store_b128 .*$", src, re.M):
        assert "LdsXorSwizzle" not in store
        offset = int(re.search(r"offset:(\d+)", store).group(1)) if "offset:" in store else 0
        assert offset % 1024 == 0, store

    # Read side: chunk c reads through LocalReadAddr+c, with key-preserving immediates.
    reads = re.findall(r"^\s*ds_load_b128 \S+, v\[vgprLocalReadAddr([AB])\+(\d+)\](?: offset:(\d+))?", src, re.M)
    assert reads
    assert {int(slot) for _, slot, _ in reads} == set(range(8))
    for _, _, offset in reads:
        assert int(offset or 0) % 1024 == 0
    # One base per chunk and tensor on each kernel-argument entry path; 1LDSBuffer never swaps.
    bases = len(re.findall(r"LdsXorSwizzle: swizzled read base c=", src))
    assert bases and bases % 16 == 0
    assert "LdsXorSwizzle: swap base" not in src

    # Tail loop: each K step of 16 reads chunks through slots 0..1, so the bases shift down by
    # two chunks instead of being incremented (base c + 32 B is not the base of chunk c + 2).
    tail = src.split("label_TailLoopBeginL:", 1)[1].split("label_TailLoopEndL", 1)[0]
    assert len(re.findall(r"LdsXorSwizzle: tail K step, base \d+ <- base \d+", tail)) == 2 * 6
    assert not re.search(r"v_add\w* v\[vgprLocalReadAddr[AB]\+\d+\]", tail)


def test_double_buffer_kernel_resets_every_base(tmp_path):
    src = _emit(tmp_path, **{"1LDSBuffer": 0})
    assert _lds_bytes(src) == 2 * 128 * 64 * 2 * 2
    # Bases 1..7 follow every buffer swap; slot 0 swaps with the regular LocalReadAddr code.
    assert len(re.findall(r"LdsXorSwizzle: swap base", src)) % 7 == 0
    assert "LdsXorSwizzle: swap base" in src
    # Before the tail loop all 8 slots per tensor return to the first buffer.
    resets = re.findall(r"v\[vgprLocalReadAddr([AB])\+(\d+)\].*reset Red,Blk -> Red", src)
    for tc in "AB":
        assert {int(i) for t, i in resets if t == tc} == set(range(8)), (tc, resets)


@pytest.mark.parametrize("overrides", [
    {"LdsPadA": 8, "LdsPadB": 8},               # swizzle replaces padding
    {"DepthU": 256},                            # 512 B LDS row
    {"VectorWidthA": 4},                        # interleaves the lane's LDS rows
])
def test_rejected_combinations(tmp_path, overrides):
    assert len(solutions_from_config(_config(tmp_path), arch=ARCH)) == 1  # control
    assert solutions_from_config(_config(tmp_path, **overrides), arch=ARCH) == []


def test_rejected_without_wmma_v1(tmp_path):
    path = _config(tmp_path, MatrixInstruction=[16, 16, 16, 1, 1, 4, 2, 2, 2])
    assert solutions_from_config(path, arch="gfx942") == []


def test_default_off_emits_no_swizzle(tmp_path):
    src = _emit(tmp_path, LdsXorSwizzle=False, LdsPadA=8, LdsPadB=8,
                LdsBlockSizePerPadA=128, LdsBlockSizePerPadB=128, VectorWidthA=1, VectorWidthB=1)
    assert "LdsXorSwizzle" not in src
