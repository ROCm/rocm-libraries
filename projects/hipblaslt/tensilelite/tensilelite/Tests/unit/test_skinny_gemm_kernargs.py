# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Skinny GEMM custom-kernel argument contract."""

import os

import tensilelite


def test_skinny_mxfp8_kernargs_use_spk_semantic():
    kernel_dir = os.path.join(
        os.path.dirname(tensilelite.__file__), "CustomKernels", "skinny_mxfp8"
    )
    kernel_names = sorted(f[:-2] for f in os.listdir(kernel_dir) if f.endswith(".s"))
    assert len(kernel_names) == 6

    for name in kernel_names:
        from tensilelite.CustomKernels import readCustomKernelConfig

        config = readCustomKernelConfig(name)
        semantics = [entry["semantic"] for entry in config["CustomKernel"]["args"]]
        assert semantics.count("SizeSum") == 1, name
        assert "Spk" in semantics, name
