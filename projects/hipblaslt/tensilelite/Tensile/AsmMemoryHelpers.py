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

from rocisa.instruction import \
    DSStoreB8, DSStoreB16, DSStoreB32, DSStoreB64, DSStoreB128, \
    DSLoadU16, DSLoadB32, DSLoadB64, DSLoadB128

_BPS_TO_DS_STORE = {1: DSStoreB8, 2: DSStoreB16, 4: DSStoreB32, 8: DSStoreB64, 16: DSStoreB128}
_BPL_TO_DS_LOAD  = {2: DSLoadU16, 4: DSLoadB32, 8: DSLoadB64, 16: DSLoadB128}

def dsStore(bps, dstAddr, src, ds, comment="", memToken=None):
    """Select and return a DSStore instruction by byte width (1/2/4/8/16)."""
    cls = _BPS_TO_DS_STORE.get(int(bps))
    assert cls, f"dsStore: unsupported bps={bps}"
    inst = cls(dstAddr=dstAddr, src=src, ds=ds, comment=comment)
    if memToken is not None:
        inst.setMemToken(memToken)
    return inst

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
