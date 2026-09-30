################################################################################
# SPDX-License-Identifier: MIT
################################################################################

"""Regressions for gfx1250 async LDS store-back and s_wait_asynccnt.

Covers:
  * _SWaitAsynccnt emits `s_wait_asynccnt <n>`.
  * global_store_async_from_lds_* emits the correct b8/b32/b64/b128 mnemonic.
  * the VGPR-MSB window (s_set_vgpr_msb + byte-form operands) is materialized
    for high VGPR indices (>= 256) -- the missing-setMsb fix in mem.hpp.
"""

import re

import pytest
import rocisa
from rocisa.container import sgpr, vgpr
from rocisa.instruction import (
    GlobalStoreAsyncFromLdsB8,
    GlobalStoreAsyncFromLdsB32,
    GlobalStoreAsyncFromLdsB64,
    GlobalStoreAsyncFromLdsB128,
    _SWaitAsynccnt,
)

_ISA = (12, 5, 0)


@pytest.fixture(scope="module", autouse=True)
def _isa_context():
    import os
    import shutil

    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [
            os.path.join(rocm_path, "bin"),
            os.path.join(rocm_path, "lib", "llvm", "bin"),
        ]
    )
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
    rocisa.rocIsa.getInstance().init(_ISA, assembler, False)
    rocisa.rocIsa.getInstance().setKernel(_ISA, 32)


def test_swaitasynccnt_emits():
    assert str(_SWaitAsynccnt(0)).strip() == "s_wait_asynccnt 0"
    assert str(_SWaitAsynccnt(2)).strip() == "s_wait_asynccnt 2"


def test_swaitasynccnt_comment():
    assert (
        str(_SWaitAsynccnt(0, "drain before reuse")).strip()
        == "s_wait_asynccnt 0                                  // drain before reuse"
    )


@pytest.mark.parametrize(
    "cls, suffix",
    [
        (GlobalStoreAsyncFromLdsB8, "b8"),
        (GlobalStoreAsyncFromLdsB32, "b32"),
        (GlobalStoreAsyncFromLdsB64, "b64"),
        (GlobalStoreAsyncFromLdsB128, "b128"),
    ],
)
def test_async_store_mnemonic_low_vgpr(cls, suffix):
    # Low VGPRs (msb == 0): plain form, no s_set_vgpr_msb. Seed the tracker at 0
    # (already-low state) so no MSB transition is emitted; -1 would be the
    # indeterminate state and force an s_set_vgpr_msb 0.
    rocisa.rocIsa.getInstance().setVgprMsb(0)
    out = str(cls(vgpr(0, 2), vgpr(4), sgpr(30, 2))).strip()
    assert out.startswith(f"global_store_async_from_lds_{suffix} ")
    assert "s_set_vgpr_msb" not in out
    # SGPR saddr never carries an MSB offset.
    assert "s[30:31]" in out


def test_async_store_high_vgpr_emits_msb_window():
    # High VGPRs (idx >= 256 => msb == 1) must emit s_set_vgpr_msb and print the
    # operands in byte-form (idx - 256). saddr is an SGPR, so only the two VGPR
    # operands (vaddr, dsaddr) participate; msbComment => src0:1, src1:1.
    rocisa.rocIsa.getInstance().setVgprMsb(-1)
    out = str(GlobalStoreAsyncFromLdsB128(vgpr(300, 4), vgpr(320), sgpr(10, 2))).strip()

    assert "s_set_vgpr_msb" in out
    assert re.search(r"src0:\s*1,\s*src1:\s*1", out)
    assert "global_store_async_from_lds_b128" in out
    # byte-form operands, SGPR unaffected
    assert "v[300-256:303-256]" in out
    assert "v[320-256]" in out
    assert "s[10:11]" in out


def test_async_store_low_vgpr_no_offset():
    rocisa.rocIsa.getInstance().setVgprMsb(0)
    out = str(GlobalStoreAsyncFromLdsB32(vgpr(0, 2), vgpr(4), sgpr(30, 2))).strip()
    # No byte-form offset artifacts for low VGPRs.
    assert "-256" not in out
    assert out == "global_store_async_from_lds_b32 v[0:1], v4, s[30:31]"
