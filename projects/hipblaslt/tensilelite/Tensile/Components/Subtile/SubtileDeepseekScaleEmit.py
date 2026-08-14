# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Helpers for DeepseekScale kernel-arg offset discovery.

The flat_load bolt-on (setupDeepseekMainloopScale, emitDeepseekScaleGR, etc.)
has been removed. DeepseekScale now uses the MX SA/SB scheduler path with
buffer_load DTL via SrdMXSA/B. This module is kept as a thin helper so that
SubtileScaleEmit.initDeepseekScaleSrd can call _scaleBufKernArgOffsets without
importing from KernelWriter directly.
"""


def _scaleBufKernArgOffsets(writer, kernel):
    """Return (offA_or_None, offB_or_None) byte offsets of ScaleABuf/ScaleBBuf.

    Byte offsets are relative to the per-GEMM kernel arg base (KernArgAddress after
    the common-args shift), computed by walking numStoreSgprNames from the argLoader
    current position.

    For non-GroupedGemm kernels, Signature.py inserts batchOffset{D,C,A,B} (4*8=32
    bytes) before the DeepseekScale args. Those entries are absent from
    numStoreSgprNames, so the walk must start 32 bytes later to land at the correct
    ScaleABuf/ScaleBBuf positions.
    """
    base = writer.argLoader.getOffset()
    names = writer.states.numStoreSgprNames
    sizes = writer.states.numStoreSgprNameSizes
    offA = offB = None
    # Account for the batchOffset block that Signature.py inserts before the
    # DeepseekScale args in non-GroupedGemm kernels.
    batchOffsetBytes = 0 if kernel["ProblemType"]["GroupedGemm"] else 32
    cur = base + batchOffsetBytes
    for name, size in zip(names, sizes):
        if name == "ScaleABuf":
            offA = cur
        elif name == "ScaleBBuf":
            offB = cur
        cur += size * 4
    return offA, offB
