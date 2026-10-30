# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Skinny GEMM custom-kernel argument contract."""


def test_skinny_gemm_logic_args_use_spk_semantic():
    from tensilelite.KernelGenerators.SkinnyGEMM import args

    semantics = [entry["semantic"] for entry in args()]
    assert semantics.count("SizeSum") == 1
    assert "Spk" in semantics
