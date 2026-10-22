# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Swizzled A/B staged through LDS (SwizzleTensor without DirectToVgpr).

Derives and emits kernels from the gfx11 GPU config ``swizzle_lds_gfx11.yaml``
on the CPU, so the LDS-staged codegen and its validator run in the unit lane.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml
from rocisa.enum import DataTypeEnum

from tensilelite.Common import swizzleBlockLanes, swizzleGeometry
from tensilelite.Common.DataType import DataType

from config_harness import assert_assembles, emit_kernels_from_config, solutions_from_config

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).parents[1] / "common/gemm/swizzle_lds_gfx11.yaml"
_ARCH = "gfx1151"
# Fixture groups: 0 = bf16 swizzled A, 1 = bf16 swizzled B, 2 = fp16 swizzled A.
_SWIZZLED = {0: "A", 1: "B", 2: "A"}


def _config(tmp_path, problem, overrides=None, problem_type=None):
    """One fixture group, pinned to a single fork with ``overrides`` applied."""
    overrides = overrides or {}
    config = yaml.safe_load(_FIXTURE.read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][problem:problem + 1]
    config["GlobalParameters"]["CpuThreads"] = 1
    config["GlobalParameters"]["PrintSolutionRejectionReason"] = True
    config["BenchmarkProblems"][0][0].update(problem_type or {})
    group = config["BenchmarkProblems"][0][1]
    group["ForkParameters"] = [
        {key: [values[0]] for key, values in p.items()}
        for p in group["ForkParameters"] if not overrides.keys() & p.keys()
    ] + [{key: [value]} for key, value in overrides.items()]
    path = tmp_path / "swizzle_lds.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _derive(tmp_path, monkeypatch, capsys, problem, overrides=None, problem_type=None,
            path=None, arch=_ARCH):
    """Return ``(solutions, rejection lines)``, deriving serially so rejections reach stdout."""
    import tensilelite.BenchmarkProblems as benchmark_problems

    monkeypatch.setattr(benchmark_problems, "ParallelMap2",
                        lambda function, objects, *_a, **_k: [function(*args) for args in objects])
    capsys.readouterr()
    path = path or _config(tmp_path, problem, overrides, problem_type)
    solutions = solutions_from_config(path, arch=arch)
    rejections = [line.strip() for line in capsys.readouterr().out.splitlines()
                  if line.startswith("reject:")]
    return solutions, rejections


@pytest.mark.parametrize("problem,overrides", [
    (0, {}),
    (1, {}),
    (2, {}),
    # Wave64: four k groups per wave and a single block along M.
    (0, {"WavefrontSize": 64, "WorkGroup": [64, 4, 1]}),
])
def test_emits_swizzled_lds_kernel(tmp_path, problem, overrides):
    tc = _SWIZZLED[problem]
    kernels = emit_kernels_from_config(_config(tmp_path, problem, overrides), arch=_ARCH, limit=1)
    assert len(kernels) == 1
    name, source, error = kernels[0]
    assert error == 0
    # Swizzle-order lane assignment, then the block/row tile address and the k*MI unroll offset.
    assert f"SWZ-{tc}: lane -> (row = L%16" in source
    assert f"SWZ-{tc}: gro{tc}" in source and "= [(m/16)*numKr*16 + m%16] * laneSize(8)" in source
    assert "= k * swzMorN(16)" in source
    assert "SWZ: tile index m" in source and "SWZ: unroll index k / laneSize" in source
    assert_assembles(source, name)


def test_lds_staging_loads_each_element_once(tmp_path, monkeypatch, capsys):
    # LDS staging gives every lane distinct data, so NumLoads must not carry the
    # DirectToVgpr dupFactor: one workgroup's loads cover the A tile exactly once.
    (lds,), _ = _derive(tmp_path, monkeypatch, capsys, 0)
    assert not lds["DirectToVgprA"]
    loaded = lds["NumLoadsA"] * lds["NumThreads"] * lds["GlobalReadVectorWidthA"]
    assert loaded == lds["MacroTile0"] * lds["DepthU"]

    # The DirectToVgpr path is unchanged and replicates lanes.
    (dtv,), _ = _derive(tmp_path, monkeypatch, capsys, 0, {"DirectToVgprA": True})
    loaded = dtv["NumLoadsA"] * dtv["NumThreads"] * dtv["GlobalReadVectorWidthA"]
    assert loaded > dtv["MacroTile0"] * dtv["DepthU"]


@pytest.mark.parametrize("overrides,problem_type,expected", [
    ({"WaveSeparateGlobalReadA": 1}, None,
     "LDS-staged SwizzleTensorA does not support WaveSeparateGlobalReadA"),
    ({}, {"TransposeA": False},
     "LDS-staged SwizzleTensorA needs TLUA=False"),
    ({"DepthU": 128, "AssertSummationElementMultiple": 128}, None,
     "LDS-staged SwizzleTensorA needs LSPA (8) to be a multiple of MI_M (16)"),
    # Two wave64 waves, but a 16-deep unroll only fills half a wave's four k groups.
    ({"WavefrontSize": 64, "WorkGroup": [64, 2, 1], "MatrixInstruction": [16, 16, 16, 1, 1, 2, 4, 2, 1],
      "DepthU": 16}, None,
     "LDS-staged SwizzleTensorA needs mBlocks(4) * LSCA(16)/kChunk(32) == numWaves(2)"),
    # K = 48 would pass ASEM 16, but the host pads K to 32 and the kernel to 8.
    ({"DepthU": 16, "AssertSummationElementMultiple": 16}, None,
     "LDS-staged SwizzleTensorA needs AssertSummationElementMultiple (16) to be a multiple of the swizzle layout K (32)"),
])
def test_rejects_unsupported_lds_staging(tmp_path, monkeypatch, capsys, overrides, problem_type, expected):
    solutions, rejections = _derive(tmp_path, monkeypatch, capsys, 0, overrides, problem_type)
    assert not solutions
    assert any(expected in line for line in rejections), rejections


def test_lds_guards_skip_direct_to_vgpr(tmp_path, monkeypatch, capsys):
    # WaveSeparateGlobalRead is only unsupported on the LDS path.
    _, rejections = _derive(tmp_path, monkeypatch, capsys, 0,
                            {"DirectToVgprA": True, "WaveSeparateGlobalReadA": 1})
    assert not any("LDS-staged" in line for line in rejections), rejections


def test_rejects_lds_staging_off_gfx11(tmp_path, monkeypatch, capsys):
    # gfx942 fp16 MFMA: swizzleK (32) > laneSize (8). The address formula covers it
    # (test_swizzle_address_matches_host_layout), but the path is only validated on
    # gfx11 hardware, so it must not be selectable here.
    config = {
        "GlobalParameters": {"CpuThreads": 1, "PrintSolutionRejectionReason": True},
        "BenchmarkProblems": [[
            {"OperationType": "GEMM", "DataType": "h", "DestDataType": "h", "ComputeDataType": "s",
             "HighPrecisionAccumulate": True, "TransposeA": True, "TransposeB": False,
             "SwizzleTensorA": True, "UseBeta": True, "Batched": True},
            {"BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
             "ForkParameters": [
                 {"MatrixInstruction": [[16, 16, 16, 1, 1, 1, 1, 1, 1]]}, {"DepthU": [64]},
                 {"PrefetchGlobalRead": [1]}, {"PrefetchLocalRead": [1]}, {"ScheduleIterAlg": [3]},
                 {"ExpandPointerSwap": [True]}, {"TransposeLDS": [1]},
                 {"GlobalReadVectorWidthA": [-1]}, {"GlobalReadVectorWidthB": [-1]},
                 {"DirectToVgprA": [False]}, {"UseSgprForGRO": [1]}, {"GlobalSplitU": [1]},
                 {"SourceSwap": [True]}],
             "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [256, 256, 1, 256]}]}]},
        ]],
    }
    path = tmp_path / "swizzle_lds_gfx942.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    solutions, rejections = _derive(tmp_path, monkeypatch, capsys, 0, path=path, arch="gfx942")
    assert not solutions
    assert "reject: LDS-staged SwizzleTensorA is only validated on gfx11" in rejections, rejections


def _host_offsets(dataType, m, k):
    """Element offsets of the client's pre-tiled tensor (DataInitialization.cpp), as [m, k].

    calculateKforSwizzling gives (MiK, MiKv) and PackK = 16 / MiKv / bpe; [m, k] is padded
    to (16, MiK*PackK), reshaped to [m/16, 16, k/(MiK*PackK), MiK/MiKv, MiKv*PackK] and
    permuted by (0, 2, 3, 1, 4).
    """
    miK, miKv = {DataTypeEnum.Half: (16, 4), DataTypeEnum.BFloat16: (16, 4),
                 DataTypeEnum.Float: (4, 1)}[dataType]
    packK = 16 // miKv // int(DataType(dataType).numBytes())
    blockK = miK * packK
    pm, pk = -(-m // 16) * 16, -(-k // blockK) * blockK
    tiled = np.arange(pm * pk).reshape(pm // 16, 16, pk // blockK, miK // miKv, miKv * packK)
    offsets = np.empty(pm * pk, dtype=np.int64)
    offsets[tiled.transpose(0, 2, 3, 1, 4).reshape(-1)] = np.arange(pm * pk)
    return offsets.reshape(pm, pk)[:m, :k]


def _kernel_offsets(geometry, miMorN, m, k):
    """Element offsets the LDS-staged kernel loads: graTileOffsets + graUnrollOffsets."""
    laneSize = geometry["laneSize"]
    numKr = -(-k // geometry["swizzleK"])
    rows, cols = np.arange(m)[:, None], np.arange(k)[None, :]
    tile = ((rows // miMorN) * numKr * swizzleBlockLanes(geometry, miMorN) + rows % miMorN) * laneSize
    return tile + (cols - cols % laneSize) * miMorN + cols % laneSize


# (data type, MIInputPerThread, MatrixInstK, WavefrontSize). gfx11 WMMA replicates
# operands across the wave, so swizzleK == laneSize; wave64 MFMA gives swizzleK > laneSize.
_GFX11_BF16 = (DataTypeEnum.BFloat16, 16, 16, 32)
_GFX942_FP16 = (DataTypeEnum.Half, 4, 16, 64)
_GFX942_FP32 = (DataTypeEnum.Float, 1, 4, 64)


def _geometry(dataType, miInput, miK, wavefront):
    return swizzleGeometry({"ProblemType": {"DataTypeA": DataType(dataType)}, "MIInputPerThreadA": miInput,
                            "MatrixInstM": 16, "MatrixInstK": miK, "WavefrontSize": wavefront}, "A")


@pytest.mark.parametrize("arch", [_GFX11_BF16, _GFX942_FP16, _GFX942_FP32])
@pytest.mark.parametrize("m,k", [(16, 64), (48, 64), (40, 256), (64, 96)])
def test_swizzle_address_matches_host_layout(arch, m, k):
    geometry = _geometry(*arch)
    assert k % geometry["layoutK"] == 0
    np.testing.assert_array_equal(_kernel_offsets(geometry, 16, m, k), _host_offsets(arch[0], m, k))


def test_swizzle_k_padding():
    # The reviewed MFMA case: a row block spans swizzleK = 4 * laneSize, and the kernel
    # pads K like the host, so any K matches.
    mfma = _geometry(*_GFX942_FP16)
    assert (mfma["laneSize"], mfma["swizzleK"], mfma["layoutK"]) == (8, 32, 32)
    assert swizzleBlockLanes(mfma, 16) == 64
    np.testing.assert_array_equal(_kernel_offsets(mfma, 16, 48, 48), _host_offsets(DataTypeEnum.Half, 48, 48))

    # gfx11 pads K to swizzleK = 8 but the host pads to 32, so rows past the first
    # block diverge for K = 48. Hence ASEM must be a multiple of layoutK.
    wmma = _geometry(*_GFX11_BF16)
    assert (wmma["laneSize"], wmma["swizzleK"], wmma["layoutK"]) == (8, 8, 32)
    kernel, host = _kernel_offsets(wmma, 16, 48, 48), _host_offsets(DataTypeEnum.BFloat16, 48, 48)
    np.testing.assert_array_equal(kernel[:16], host[:16])
    assert (kernel[16:] != host[16:]).all()
