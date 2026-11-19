################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from rocisa.code import Module
from rocisa.container import DSModifiers, vgpr
from rocisa.instruction import \
    DSStoreB8, DSStoreB16, DSStoreB32, DSStoreB64, DSStoreB128, \
    DSLoadU16, DSLoadB32, DSLoadB64, DSLoadB128

_BPS_TO_DS_STORE = {1: DSStoreB8, 2: DSStoreB16, 4: DSStoreB32, 8: DSStoreB64, 16: DSStoreB128}
_BPL_TO_DS_LOAD  = {2: DSLoadU16, 4: DSLoadB32, 8: DSLoadB64, 16: DSLoadB128}

def dsStore(bps, dstAddr, src, ds, comment="", memToken=None):
    """Select and return a DSStore instruction by byte width (1/2/4/8/16/32).

    bps=32 (BF16 storeRemap with gwvw=16, fp32 with gwvw=8) has no native
    single-instruction store on this ISA. rocisa exposes a DSStoreB256
    binding but its toString() half-split currently segfaults when the
    source vgpr came in through `vgpr(intIdx, 8)` (the storeRemap path).
    Handle bps=32 by Python-side splitting into two consecutive DSStoreB128
    at offset / offset+16, using the lower / upper 4-dword halves of src.
    Callers do `module.add(dsStore(...))`, which accepts either a single
    Instruction or a multi-instruction Module.
    """
    bps_int = int(bps)
    if bps_int == 32:
        srcLo = _halfWidthVgpr(src, shift=0)
        srcHi = _halfWidthVgpr(src, shift=4)
        dsLoOffset = ds.offset if ds is not None else 0
        dsHiOffset = dsLoOffset + 16
        dsLo = DSModifiers(offset=dsLoOffset)
        dsHi = DSModifiers(offset=dsHiOffset)
        instLo = DSStoreB128(dstAddr=dstAddr, src=srcLo, ds=dsLo,
                             comment=(comment + " (B256 lo)").strip())
        instHi = DSStoreB128(dstAddr=dstAddr, src=srcHi, ds=dsHi,
                             comment=(comment + " (B256 hi)").strip())
        if memToken is not None:
            instLo.setMemToken(memToken)
            instHi.setMemToken(memToken)
        mod = Module("dsStore_b256_split")
        mod.add(instLo)
        mod.add(instHi)
        return mod
    cls = _BPS_TO_DS_STORE.get(bps_int)
    assert cls, f"dsStore: unsupported bps={bps}"
    inst = cls(dstAddr=dstAddr, src=src, ds=ds, comment=comment)
    if memToken is not None:
        inst.setMemToken(memToken)
    return inst


def _halfWidthVgpr(src, shift):
    """Return a 4-dword (B128) vgpr whose start is shifted by `shift` dwords from
    `src`'s start. Used by dsStore() to split a B256 source into B128 halves.

    Works directly off the rocisa RegisterContainer fields (regIdx / regName)
    rather than parsing str(src). This avoids display-only artifacts of
    RegisterContainer.toString() that string parsing would trip on:
      - HasVgprMSB compensation suffix, e.g. "v[300-256:...]"
      - the "vgpr" prefix toString prepends to named registers, e.g.
        "v[vgprFoo+4:...]" (the stored regName.name is the bare "Foo"), which
        would double-prefix to "vgprvgprFoo" if fed back through vgpr().
    """
    regName = getattr(src, "regName", None)
    assert regName is not None or hasattr(src, "regIdx"), \
        f"_halfWidthVgpr: expected a rocisa RegisterContainer, got {type(src)!r}"
    if regName is not None:
        # Named register: bare base name + offset list (no display prefix).
        base = regName.name
        total = sum(regName.getOffsets()) + shift
        nameStr = base if total == 0 else f"{base}+{total}"
        return vgpr(nameStr, 4)
    # Numeric register: regIdx is the true (logical) index, no MSB artifact.
    return vgpr(int(src.regIdx) + shift, 4)

def dsLoad(bpl, dst, src, ds, comment="", memToken=None):
    """Select and return a DSLoad instruction by byte width (2/4/8/16)."""
    cls = _BPL_TO_DS_LOAD.get(int(bpl))
    assert cls, f"dsLoad: unsupported bpl={bpl}"
    inst = cls(dst=dst, src=src, ds=ds, comment=comment)
    if memToken is not None:
        inst.setMemToken(memToken)
    return inst

def _vgprOffset(base, n):
    """Compute base+n for vgpr offset, handling both str and int base."""
    if isinstance(base, str):
        return f"{base}+{int(n)}"
    return int(base + int(n))
