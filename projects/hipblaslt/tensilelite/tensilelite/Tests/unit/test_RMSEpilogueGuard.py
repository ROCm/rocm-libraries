# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from Tensile.KernelWriterAssembly import KernelWriterAssembly


def test_non_rms_cms_kernel_with_vgpr_accumulators():
    # An ordinary CMS kernel can accumulate in VGPRs. It must not construct
    # the AGPR-only accumulator view used by the fused RMS epilogue.
    kernel = {
        "RMSEpilogue": False,
        "UseCustomMainLoopSchedule": 1,
        "UseSubtileImpl": False,
        "MIArchVgpr": True,
        "SourceSwap": False,
    }
    writer = KernelWriterAssembly.__new__(KernelWriterAssembly)

    assert str(writer.emitSubtileFusedEpilogue(kernel)) == ""
