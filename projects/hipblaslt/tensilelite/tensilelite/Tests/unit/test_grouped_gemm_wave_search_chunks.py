# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for KernelWriterAssembly.groupedGemmWaveSearchChunks.

The grouped GEMM wave search checks out one aligned block of 4 VGPRs per chunk
of 64 gemm headers plus 5 scratch VGPRs. groupedGemmWaveSearchChunks picks the
chunk count, and it is the only thing that keeps the search from lowering the
kernel's occupancy: it must return the largest count whose block fits below the
VGPR budget of the kernel's current occupancy (getMaxRegsForOccupancy, which
accounts for LDS and AGPRs), or 0 so the kernel keeps the linear search.

Each case builds a VGPR pool whose registers are all taken except the listed
free ranges, and runs the real occupancy code against the arch's caps.
"""

import copy
import os
import shutil
from types import SimpleNamespace

import pytest

from rocisa import rocIsa
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool
from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit

ISAS = {"gfx942": (9, 4, 2), "gfx950": (9, 5, 0)}
NUM_THREADS = 256
SGPRS = 72
# Small enough that LDS does not limit occupancy on either arch.
SMALL_LDS = 4096


def _writer(isa, vgprs, free=(), agprs=0):
    """A writer whose VGPR pool has vgprs registers, all taken except the
    (start, count) ranges in free, as after defineAndResources has allocated
    everything the kernel needs."""
    ri = rocIsa.getInstance()
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join([
        os.path.join(rocm_path, "bin"),
        os.path.join(rocm_path, "lib", "llvm", "bin"),
    ])
    ri.init(isa, shutil.which("amdclang++", path=search_path) or "/usr/bin/amdclang++")
    ri.setKernel(isa, 64)

    kw = object.__new__(KernelWriterAssembly)
    kw.states = SimpleNamespace(
        archCaps=ri.getArchCaps(), regCaps=ri.getRegCaps(), version=isa, doubleVgpr=True)
    kw.vgprPool = RegisterPool(vgprs, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    for start, count in free:
        kw.vgprPool.add(start, count, "free")
    kw.sgprPool = RegisterPool(SGPRS, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    kw.agprPool = RegisterPool(agprs, RegisterType.Accvgpr, defaultPreventOverflow=False, printRP=False)
    return kw


def _occupancy(kw, kernel, vgprs):
    return kw.getOccupancy(
        NUM_THREADS, vgprs, kw.sgprPool.size(), kernel["LdsNumBytes"], kw.agprPool.size(), True)


def _kernel(lds=SMALL_LDS):
    return {"NumThreads": NUM_THREADS, "LdsNumBytes": lds}


def _ldsForThreeWorkgroups(kw):
    """LDS bytes that limit a CU to 3 workgroups on this arch."""
    granularity = kw.states.archCaps["LdsGranularity"]
    return kw.states.archCaps["DeviceLDS"] // 3 // granularity * granularity


# With 256 threads, 72 SGPRs and little LDS, up to 64 VGPRs (8-register granules
# of the 512 unified registers) run 8 waves, the cap; 65 run 7.
CASES = {
    # 24 taken: 24..63 is free under the budget, room for 6 chunks (29 VGPRs).
    "room for 6": dict(vgprs=24, expected=6),
    # 36 free registers, but in runs of 20 and 16: the largest block that fits
    # is 3 chunks (17 VGPRs) in the first run.
    "fragmented": dict(vgprs=64, free=[(8, 20), (40, 16)], expected=3),
    # 60 taken: 60..63 is free, short of one chunk (9 VGPRs).
    "no room for one chunk": dict(vgprs=60, expected=0),
    # 64 taken sits exactly at the 8-wave boundary: any VGPR more is 7 waves.
    "at the occupancy boundary": dict(vgprs=64, expected=0),
    # 35 taken: blocks start 4-aligned at 36, so 5 chunks (36..60) fit and 6
    # (36..64) would cross the budget by one register.
    "one register past the budget": dict(vgprs=35, expected=5),
    # 40 AGPRs count against the same 512 registers, so 24 VGPRs are already
    # at the 8-wave boundary.
    "agprs at the boundary": dict(vgprs=24, agprs=40, expected=0),
}


@pytest.mark.parametrize("arch", sorted(ISAS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_chunk_count(arch, case):
    c = CASES[case]
    kw = _writer(ISAS[arch], c["vgprs"], c.get("free", ()), c.get("agprs", 0))
    assert kw.groupedGemmWaveSearchChunks(_kernel()) == c["expected"]


@pytest.mark.parametrize("arch", sorted(ISAS))
def test_lds_limited_occupancy_raises_the_budget(arch):
    """120 VGPRs run 4 waves by VGPRs alone, which leaves 120..127 (one 8-register
    granule, short of a chunk). With LDS for 3 workgroups the kernel already runs
    3, so VGPRs up to 168 keep it there and 6 chunks fit."""
    kw = _writer(ISAS[arch], 120)
    assert kw.groupedGemmWaveSearchChunks(_kernel()) == 0
    assert kw.groupedGemmWaveSearchChunks(_kernel(_ldsForThreeWorkgroups(kw))) == 6


def test_max_chunks_per_load_caps_the_count():
    kw = _writer(ISAS["gfx942"], 24)
    assert kw.groupedGemmWaveSearchChunks(_kernel(), maxChunksPerLoad=2) == 2


@pytest.mark.parametrize("arch", sorted(ISAS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_checkout_keeps_occupancy_and_one_more_chunk_would_not(arch, case):
    """What groupedGemmWaveSearch does with the count: check out 4 * U + 5 VGPRs
    aligned to 4. That keeps the kernel's occupancy, and a block for one more
    chunk would lower it."""
    c = CASES[case]
    kw = _writer(ISAS[arch], c["vgprs"], c.get("free", ()), c.get("agprs", 0))
    kernel = _kernel()
    before = _occupancy(kw, kernel, kw.vgprPool.size())
    chunks = kw.groupedGemmWaveSearchChunks(kernel)

    if chunks:
        pool = copy.deepcopy(kw.vgprPool)
        pool.checkOutAligned(4 * chunks + 5, 4, "ggWaveSearch")
        assert _occupancy(kw, kernel, pool.size()) == before

    if chunks < 6:
        pool = copy.deepcopy(kw.vgprPool)
        pool.checkOutAligned(4 * (chunks + 1) + 5, 4, "ggWaveSearch")
        assert _occupancy(kw, kernel, pool.size()) < before


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
