# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Skinny GEMM custom-kernel argument contract."""

import types

import pytest


def test_skinny_gemm_logic_args_use_spk_semantic():
    from tensilelite.KernelGenerators.SkinnyGEMM import args

    semantics = [entry["semantic"] for entry in args()]
    assert semantics.count("SizeSum") == 1
    assert "Spk" in semantics


ARG_SIZES = {"address": 8, "uint64": 8, "int32": 4, "float32": 4}


def test_skinny_gemm_args_have_only_the_x_workgroup_count():
    from tensilelite.KernelGenerators.SkinnyGEMM import args

    semantics = [entry["semantic"] for entry in args()]
    assert semantics.count("NumWorkGroupsX") == 1
    assert "NumWorkGroupsY" not in semantics
    assert "NumWorkGroupsZ" not in semantics


def test_skinny_gemm_args_kernarg_offsets():
    from tensilelite.KernelGenerators.SkinnyGEMM import args

    offsets = []
    offset = 0
    for entry in args():
        offsets.append(offset)
        offset += ARG_SIZES[entry["type"]]
    semantics = [entry["semantic"] for entry in args()]
    assert offsets[semantics.index("NumWorkGroupsX")] == 96
    assert offsets[-2:] == [100, 104]
    assert semantics[-2:] == ["ConstantZero", "ConstantZero"]
    assert offset == 108
    assert (offset + 7) // 8 * 8 == 112


@pytest.mark.parametrize(
    "grid3d,grid",
    [(True, ["TilesLongShort", "One", "One"]), (False, ["ComputeUnits", "One", "One"])],
)
def test_skinny_gemm_grid(monkeypatch, grid3d, grid):
    from tensilelite.KernelGenerators import SkinnyGEMM

    stub = types.SimpleNamespace(
        SkinnyGEMMConfig=lambda **kwargs: kwargs,
        skinnyGEMMKernelName=lambda config: "gemm",
    )
    monkeypatch.setattr(SkinnyGEMM, "_codegen", lambda: stub)
    params = dict(
        bm=1, tn=1, lanesK=64, waves=4, prefetch=2, upIters=2, group=1, numXcds=1,
        grid3d=grid3d, useBeta=False,
    )
    assert SkinnyGEMM.kernelConfig(params)["CustomKernel"]["grid"] == grid
