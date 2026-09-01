# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Swizzled A/B staged through LDS (SwizzleTensor without DirectToVgpr).

Derives and emits kernels from the gfx11 GPU config ``swizzle_lds_gfx11.yaml``
on the CPU, so the LDS-staged codegen and its validator run in the unit lane.
"""

from pathlib import Path

import pytest
import yaml

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


def _derive(tmp_path, monkeypatch, capsys, problem, overrides=None, problem_type=None):
    """Return ``(solutions, rejection lines)``, deriving serially so rejections reach stdout."""
    import tensilelite.BenchmarkProblems as benchmark_problems

    monkeypatch.setattr(benchmark_problems, "ParallelMap2",
                        lambda function, objects, *_a, **_k: [function(*args) for args in objects])
    capsys.readouterr()
    solutions = solutions_from_config(_config(tmp_path, problem, overrides, problem_type), arch=_ARCH)
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
      "DepthU": 16, "AssertSummationElementMultiple": 16}, None,
     "LDS-staged SwizzleTensorA needs mBlocks(4) * LSCA(16)/kChunk(32) == numWaves(2)"),
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
