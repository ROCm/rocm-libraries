# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from rocisa import rocIsa
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool
from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("output_bytes, accumulation_bytes", [(2, 4), (4, 4), (4, 2)])
def test_remap_readback_buffers_do_not_overlap(output_bytes, accumulation_bytes):
    rocIsa.getInstance().setKernel((12, 5, 0), 32)
    kernel = dict(
        WavefrontSize=32, MIWaveGroup=[2, 2], MatrixInstM=16, MatrixInstN=16,
        MatrixInstBM=1, MatrixInstBN=1, MIOutputVectorWidth=4,
        MacroTile0=32, StoreRemapVectorWidth=4,
    )
    writer = KernelWriterAssembly.__new__(KernelWriterAssembly)
    writer.states = SimpleNamespace(
        kernel=kernel, bpeCexternal=output_bytes, bpeCexternalGSU1=output_bytes,
        bpeCinternal=accumulation_bytes, bpr=4, regCaps={"MaxSgpr": 102},
    )
    writer.vgprs = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    writer.sgprBpeList = []
    writer.do = {"Sync": False}

    writer.storeRemapComputeStoreVgprs(kernel)
    buffers = writer.vgprs.storeRemapAS
    assert len(buffers) == 2
    next_temp = writer.vgprPool.checkOut(1, "next temporary")

    # Every readback must fit without overwriting another buffer or a later allocation.
    for element_bytes in (output_bytes, accumulation_bytes):
        registers_per_read = kernel["StoreRemapVectorWidth"] * element_bytes // 4
        occupied = [reg for base in buffers for reg in range(base, base + registers_per_read)]
        assert len(set(occupied)) == len(occupied)
        assert next_temp not in occupied
