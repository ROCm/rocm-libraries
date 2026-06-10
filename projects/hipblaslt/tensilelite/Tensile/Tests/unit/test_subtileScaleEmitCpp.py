#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Byte/string parity test for the C++ MX scale GR/LR data-movement emit leaves.

`hipblaslt_incremental_refactor-grc.111` moves the rocisa construction of the
MX scale-factor (MXSA/MXSB) GR/LR data-movement leaves (DTL scale buffer_load,
scale ds_read_b32, scale SRD pointer update) from Python into the C++
``ModuleBuilder``. The Python ``SubtileScaleEmit`` functions are now thin
boundary calls that resolve writer register state and delegate the rocisa
construction to C++. The scale GR/LR LDS-swap leaves are byte-identical to the
AB leaves and reuse ``gr_lds_buffer_swap`` / ``lr_lds_buffer_swap`` with the
MXSA/MXSB component tag (covered here for the scale tags).

This test asserts that what the C++ builder produces renders to *exactly* the
same assembly string as the equivalent rocisa Python construction the scale
emit path performed before the port — for MXFP8 / MXFP4 representative shapes.

Pure-string test (rocisa pinned to gfx950); no GPU runtime / hip dependency.
The end-to-end gfx950 GPU coverage lives in the subtile_mxfp8 / subtile_mxfp4
integration yaml label paths.
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
# Reference rocisa construction — a verbatim copy of the pre-port Python scale
# emit bodies. The C++ builder must render byte-identically to these.
# ---------------------------------------------------------------------------

def _ref_scale_gr_load(tc, isGlc, isSlc, isNT, voff):
    from rocisa.code import Module
    from rocisa.container import MUBUFModifiers, vgpr, sgpr, mgpr
    from rocisa.instruction import SMovB32, BufferLoadB128
    module = Module()
    module.addComment0("Scale GR: %s (DTL: BufferLoadB128 -> LDS)" % tc)
    module.add(SMovB32(dst=mgpr(0), src=sgpr("LocalWriteBaseAddr%s" % tc),
                       comment="scale%s: M0 = scaleLdsBase" % tc))
    mubuf = MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc,
                           nt=isNT, lds=True)
    module.add(BufferLoadB128(dst=None, vaddr=vgpr(voff),
                              saddr=sgpr("Srd%s" % tc, 4), soffset=0,
                              mubuf=mubuf,
                              comment="scale%s: DTL b128 load" % tc))
    return module


def _ref_scale_ds_read(tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k):
    from rocisa.code import Module
    from rocisa.container import DSModifiers, vgpr
    from rocisa.instruction import DSLoadB32
    module = Module()
    if k >= 0:
        comment = "scale%s[group%u,K=%u]: load 4B from LDS" % (
            tc, scaleGroupIdx, k)
    else:
        comment = "scale%s[group%u]: load 4B from LDS" % (tc, scaleGroupIdx)
    module.add(DSLoadB32(dst=vgpr(vdst), src=vgpr(addrVgpr),
                         ds=DSModifiers(offset=dsOffset), comment=comment))
    return module


def _ref_scale_gr_ptr_update(tc, inc):
    from rocisa.code import Module
    from rocisa.container import sgpr
    from rocisa.instruction import SAddU32, SAddCU32
    module = Module()
    module.addComment0("Scale SRD update: %s += %u" % (tc, inc))
    module.add(SAddU32(dst=sgpr("Srd%s" % tc), src0=sgpr("Srd%s" % tc),
                       src1=inc))
    module.add(SAddCU32(dst=sgpr("Srd%s+1" % tc), src0=sgpr("Srd%s+1" % tc),
                        src1=0))
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
# scale_gr_load — MXFP8 / MXFP4 DTL scale buffer_load
# ---------------------------------------------------------------------------

# (label, tc, isGlc, isSlc, isNT, voff)
_SCALE_GR_LOAD_CASES = [
    ("mxsa_plain", "MXSA", False, False, False, 12),
    ("mxsb_plain", "MXSB", False, False, False, 13),
    ("mxsa_glc", "MXSA", True, False, False, 14),
    ("mxsb_nt", "MXSB", False, False, True, 15),
    ("mxsa_glc_slc_nt", "MXSA", True, True, True, 16),
]


@pytest.mark.parametrize("case", _SCALE_GR_LOAD_CASES, ids=lambda c: c[0])
def test_scale_gr_load_matches_python(mb, case):
    _label, tc, isGlc, isSlc, isNT, voff = case
    cpp = str(mb.scale_gr_load(tc, isGlc, isSlc, isNT, voff))
    py = str(_ref_scale_gr_load(tc, isGlc, isSlc, isNT, voff))
    assert cpp == py, f"scale_gr_load mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"
    assert "buffer_load" in cpp.lower()


# ---------------------------------------------------------------------------
# scale_ds_read — scale ds_read_b32 per group (PGR=0 and scheduler comments)
# ---------------------------------------------------------------------------

# (label, tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k)
_SCALE_DS_READ_CASES = [
    # PGR=0 path: no K index in the comment (k < 0).
    ("mxsa_group0_noK", "MXSA", 8, 3, 0, 0, -1),
    ("mxsb_group1_noK", "MXSB", 12, 5, 64, 1, -1),
    # Scheduler emit path: K index present.
    ("mxsa_group0_k0", "MXSA", 8, 3, 0, 0, 0),
    ("mxsb_group2_k1", "MXSB", 20, 5, 256, 2, 1),
]


@pytest.mark.parametrize("case", _SCALE_DS_READ_CASES, ids=lambda c: c[0])
def test_scale_ds_read_matches_python(mb, case):
    _label, tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k = case
    cpp = str(mb.scale_ds_read(tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k))
    py = str(_ref_scale_ds_read(tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k))
    assert cpp == py, f"scale_ds_read mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"
    assert "ds_read" in cpp.lower() or "ds_load" in cpp.lower()


# ---------------------------------------------------------------------------
# scale_gr_ptr_update / scale swap leaves (MXSA/MXSB tags)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tc,inc", [
    ("MXSA", 64), ("MXSB", 128), ("MXSA", 512),
])
def test_scale_gr_ptr_update_matches_python(mb, tc, inc):
    cpp = str(mb.scale_gr_ptr_update(tc, inc))
    py = str(_ref_scale_gr_ptr_update(tc, inc))
    assert cpp == py, f"scale_gr_ptr_update mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"


@pytest.mark.parametrize("tc", ["MXSA", "MXSB"])
def test_scale_gr_lds_buffer_swap_matches_python(mb, tc):
    cpp = str(mb.gr_lds_buffer_swap(tc))
    py = str(_ref_gr_lds_buffer_swap(tc))
    assert cpp == py, f"scale gr_lds_buffer_swap mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"


@pytest.mark.parametrize("tc,voffs,vswaps", [
    ("MXSA", [10], [20]),
    ("MXSB", [10], [21]),
])
def test_scale_lr_lds_buffer_swap_matches_python(mb, tc, voffs, vswaps):
    cpp = str(mb.lr_lds_buffer_swap(tc, voffs, vswaps))
    py = str(_ref_lr_lds_buffer_swap(tc, voffs, vswaps))
    assert cpp == py, f"scale lr_lds_buffer_swap mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"
