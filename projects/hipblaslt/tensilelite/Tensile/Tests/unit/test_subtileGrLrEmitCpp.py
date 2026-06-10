#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Byte/string parity test for the C++ GR/LR data-movement emit leaves.

`hipblaslt_incremental_refactor-grc.109` moves the rocisa construction of the
subtile GR/LR data-movement leaves (buffer-load, ds-read, GR/LR LDS swap, GR
pointer update) from Python into the C++ ``ModuleBuilder``. The Python
``SubtileGREmit`` / ``SubtileLREmit`` functions are now thin boundary calls that
resolve writer register state and delegate the rocisa construction to C++.

This test asserts that what the C++ builder produces renders to *exactly* the
same assembly string as the equivalent rocisa Python construction the emit path
performed before the port — for representative BF16 and FP8/FP4 GR/LR shapes
(single vs. multi-load buffer_load, single vs. multi-read ds_read, the SGPR and
VGPR soffset branches, and the swap / pointer-update leaves).

Pure-string test (rocisa pinned to gfx950); no GPU runtime / hip dependency. The
end-to-end GPU roundtrip coverage lives in ``test_gr_lr_roundtrip.py`` /
``test_gr_lr_roundtrip_fp8.py`` (gfx950 label paths).
"""

import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

pytest.importorskip("rocisa")
pytest.importorskip("tensile_writer.subtile.module_builder")

from tensile_writer.subtile.module_builder import ModuleBuilder


def _init_rocisa_gfx950():
    """Pin rocisa to gfx950 (wave64) for deterministic string emission."""
    import shutil
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx950")
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, 64)


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    _init_rocisa_gfx950()


@pytest.fixture(scope="module")
def mb():
    return ModuleBuilder()


# ---------------------------------------------------------------------------
# Reference rocisa construction — a verbatim copy of the pre-port Python emit
# bodies. The C++ builder must render byte-identically to these.
# ---------------------------------------------------------------------------

def _ref_single_buffer_load(tc, isGlc, isSlc, isNT, offsetK, grBaseId,
                            m0Offsets, soffset, voffs):
    from rocisa.code import Module
    from rocisa.container import MUBUFModifiers, vgpr, sgpr, mgpr
    from rocisa.instruction import SAddU32, BufferLoadB128
    module = Module()
    WriteBaseAddr = "LocalWriteBaseAddr%s" % tc
    for i, m0Offset in enumerate(m0Offsets):
        module.add(SAddU32(dst=mgpr(0), src0=sgpr(WriteBaseAddr),
                           src1=(m0Offset - offsetK)))
        mubuf = MUBUFModifiers(offen=True, offset12=offsetK, glc=isGlc,
                               slc=isSlc, nt=isNT, lds=True)
        module.add(BufferLoadB128(dst=None, vaddr=vgpr(voffs[i]),
                                  saddr=sgpr("Srd%s" % tc, 4), soffset=soffset,
                                  mubuf=mubuf,
                                  comment="grBaseId = %u, i= %u" % (grBaseId, i)))
    return module


def _ref_single_ds_read(tc, sId0, sId1, subIterK, dstVgpr, regsPerDsRead,
                        offset, dstRegOffsets, addrVgprs):
    from rocisa.code import Module
    from rocisa.container import DSModifiers, vgpr
    from rocisa.instruction import DSLoadB128
    module = Module()
    for readIdx in range(len(dstRegOffsets)):
        module.add(DSLoadB128(
            dst=vgpr(dstVgpr + dstRegOffsets[readIdx], regsPerDsRead),
            src=vgpr(addrVgprs[readIdx]),
            ds=DSModifiers(offset=offset),
            comment="Subtile%s[%u, %u] subIterK=%u read=%u" % (
                tc, sId0, sId1, subIterK, readIdx)))
    return module


def _ref_gr_ptr_update(tc, inc):
    from rocisa.code import Module
    from rocisa.container import sgpr
    from rocisa.instruction import SAddU32, SAddCU32
    module = Module(f"GR Ptr Update ({tc})")
    module.add(SAddU32(dst=sgpr(f"Srd{tc}"), src0=sgpr(f"Srd{tc}"), src1=inc,
                       comment=f"{tc}: advance SRD by {inc} bytes"))
    module.add(SAddCU32(dst=sgpr(f"Srd{tc}+1"), src0=sgpr(f"Srd{tc}+1"), src1=0,
                        comment=f"{tc}: carry"))
    return module


def _ref_gr_lds_buffer_swap(tc):
    from rocisa.code import Module
    from rocisa.container import sgpr
    from rocisa.instruction import SXorB32
    module = Module()
    module.addComment0("Emit code to swap %s GR m0 offsets" % tc)
    module.add(SXorB32(dst=sgpr(f"LocalWriteBaseAddr{tc}"),
                       src0=sgpr(f"LocalWriteBaseAddr{tc}"),
                       src1=sgpr(f"Swap{tc}"), comment=""))
    return module


def _ref_lr_lds_buffer_swap(tc, voffs, vswaps):
    from rocisa.code import Module
    from rocisa.container import vgpr
    from rocisa.instruction import VXorB32
    module = Module()
    module.addComment0("Emit code to swap %s LR vgpr offsets" % tc)
    for i in range(len(voffs)):
        module.add(VXorB32(dst=vgpr(voffs[i]), src0=vgpr(voffs[i]),
                           src1=vgpr(vswaps[i]), comment=""))
    return module


# ---------------------------------------------------------------------------
# single_buffer_load
# ---------------------------------------------------------------------------

# (label, tc, isGlc, isSlc, isNT, offsetK, grBaseId, m0Offsets, soffset_kind, voffs)
# soffset_kind: ("sgpr", idx) -> sgpr(idx) | ("imm", 0) -> int 0
_BUFFER_LOAD_CASES = [
    # BF16 1x1: one load per subtile, shared SGPR soffset, offsetK=0.
    ("bf16_single_sgpr", "A", False, False, False, 0, 0, [0], ("sgpr", 40), [12]),
    # BF16 with K offset and non-temporal flags.
    ("bf16_offsetK_nt", "B", True, True, True, 128, 3, [256], ("sgpr", 41), [13]),
    # FP4/FP8 loadRatioGR<1: two loads per subtile (numGRPerSubtile=2).
    ("fp4_two_loads", "A", False, False, False, 0, 0, [0, 512], ("sgpr", 42), [14, 15]),
    # VGPR-offset fallback branch: soffset is the immediate 0.
    ("vgpr_fallback_imm", "B", False, False, False, 64, 1, [128, 640], ("imm", 0), [16, 17]),
]


@pytest.mark.parametrize("case", _BUFFER_LOAD_CASES, ids=lambda c: c[0])
def test_single_buffer_load_matches_python(mb, case):
    from rocisa.container import sgpr
    (_label, tc, isGlc, isSlc, isNT, offsetK, grBaseId, m0Offsets,
     soffset_kind, voffs) = case
    soffset = sgpr(soffset_kind[1]) if soffset_kind[0] == "sgpr" else soffset_kind[1]

    cpp = str(mb.single_buffer_load(tc, isGlc, isSlc, isNT, offsetK, grBaseId,
                                    m0Offsets, soffset, voffs))
    py = str(_ref_single_buffer_load(tc, isGlc, isSlc, isNT, offsetK, grBaseId,
                                     m0Offsets, soffset, voffs))
    assert cpp == py, f"buffer_load mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"
    assert "buffer_load" in cpp.lower()


# ---------------------------------------------------------------------------
# single_ds_read
# ---------------------------------------------------------------------------

# (label, tc, sId0, sId1, subIterK, dstVgpr, regsPerDsRead, offset,
#  dstRegOffsets, addrVgprs)
_DS_READ_CASES = [
    # BF16/FP4: one ds_read_b128 per tile (4 regs).
    ("bf16_single_read", "A", 0, 0, 0, 8, 4, 0, [0], [3]),
    # Non-zero subtile/offset.
    ("bf16_offset", "B", 1, 0, 1, 20, 4, 256, [0], [5]),
    # FP8 8-VGPR tile: two ds_read_b128, advancing dst by regsPerDsRead.
    ("fp8_two_reads", "A", 0, 1, 0, 16, 4, 128, [0, 4], [6, 7]),
]


@pytest.mark.parametrize("case", _DS_READ_CASES, ids=lambda c: c[0])
def test_single_ds_read_matches_python(mb, case):
    (_label, tc, sId0, sId1, subIterK, dstVgpr, regsPerDsRead, offset,
     dstRegOffsets, addrVgprs) = case
    cpp = str(mb.single_ds_read(tc, sId0, sId1, subIterK, dstVgpr,
                                regsPerDsRead, offset, dstRegOffsets, addrVgprs))
    py = str(_ref_single_ds_read(tc, sId0, sId1, subIterK, dstVgpr,
                                 regsPerDsRead, offset, dstRegOffsets, addrVgprs))
    assert cpp == py, f"ds_read mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"
    assert "ds_read" in cpp.lower() or "ds_load" in cpp.lower()


# ---------------------------------------------------------------------------
# gr_ptr_update / gr_lds_buffer_swap / lr_lds_buffer_swap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tc,inc", [("A", 128), ("B", 256), ("A", 64)])
def test_gr_ptr_update_matches_python(mb, tc, inc):
    cpp = str(mb.gr_ptr_update(tc, inc))
    py = str(_ref_gr_ptr_update(tc, inc))
    assert cpp == py, f"gr_ptr_update mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"


@pytest.mark.parametrize("tc", ["A", "B"])
def test_gr_lds_buffer_swap_matches_python(mb, tc):
    cpp = str(mb.gr_lds_buffer_swap(tc))
    py = str(_ref_gr_lds_buffer_swap(tc))
    assert cpp == py, f"gr_lds_buffer_swap mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"


@pytest.mark.parametrize("tc,voffs,vswaps", [
    ("A", [10], [20]),            # single LR offset (BF16/FP4)
    ("B", [10, 11], [20, 21]),    # two LR offsets (FP8 numLRPerSubtile=2)
])
def test_lr_lds_buffer_swap_matches_python(mb, tc, voffs, vswaps):
    cpp = str(mb.lr_lds_buffer_swap(tc, voffs, vswaps))
    py = str(_ref_lr_lds_buffer_swap(tc, voffs, vswaps))
    assert cpp == py, f"lr_lds_buffer_swap mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"
