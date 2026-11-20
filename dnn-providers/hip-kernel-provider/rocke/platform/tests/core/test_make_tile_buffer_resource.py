# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``make_tile_buffer_resource`` -- the large-tensor (``addr64``) forward
addressing primitive: a 64-bit tile origin folded into the pointer, with an
i32 ``num_records`` clamped to ``2**31-1`` so the existing OOB-zero buffer
trick keeps working.

CPU only: no kernel in this file reads a GPU-measured value. The op-sequence
assertions pin the lowering (readfirstlane.i64, the un-``inbounds`` GEP,
cmp+select rather than smax/smin -- see the module docstring on why),
and compiling through COMGR all the way to a real HSACO is the strongest
CPU-only proof that the emitted LLVM IR is well-typed (a mismatched-width
intrinsic call, as scalar ``smax``/``smin`` would emit for an i64 operand,
fails COMGR's verifier).
"""

from __future__ import annotations

import re

from rocke.core.ir import F16, I32, I64, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.helpers.compile import compile_kernel
from rocke.helpers.tensor_view import make_tile_buffer_resource


def _build(*, soffset=None):
    b = IRBuilder("k")
    ptr = b.param("A", PtrType(F16, "global"))
    origin = b.param("origin", I64)
    total = b.param("total", I64)
    rsrc = make_tile_buffer_resource(
        b, ptr, origin_i64=origin, total_bytes_i64=total, soffset=soffset
    )
    voffset = b.const_i32(0)
    b.buffer_load_vN(rsrc.rsrc, voffset, rsrc.soffset, F16, 1)
    b.ret()
    return b.kernel, rsrc


def test_default_soffset_is_const_i32_zero():
    _, rsrc = _build()
    assert rsrc.soffset.type.name == "i32"


def test_lowering_scalarizes_the_origin_and_clamps_with_select_not_smax():
    kernel, _ = _build()
    ll = lower_kernel_to_llvm(kernel, arch="gfx950")
    # Origin is scalarized (readfirstlane + the asm sgpr-pin idiom) before it
    # reaches the pointer or the buffer descriptor.
    assert re.search(r"call i64 @llvm\.amdgcn\.readfirstlane\.i64\(i64 %origin\)", ll)
    assert re.search(r'asm "", "=s,0"\(i64 %ufm\d+\)', ll)
    # global_ptr_offset: a plain (non-inbounds) i8 GEP, not global_ptr_add's
    # inbounds form -- an empty/fully-masked tile's origin may legally land
    # outside the allocation.
    assert re.search(r"getelementptr i8, ptr addrspace\(1\) %A, i64 %sgpr\d+", ll)
    assert "inbounds" not in ll
    # The clamp is cmp+select, not smax/smin (which would mismatch widths;
    # see the helper's docstring): one icmp slt/select for the floor at 0,
    # one icmp sgt/select for the ceiling at 2**31-1.
    assert len(re.findall(r"icmp slt i64 .*, 0", ll)) == 1
    assert len(re.findall(r"icmp sgt i64 .*, 2147483647", ll)) == 1
    assert len(re.findall(r"select i1 %\w+, i64 ", ll)) == 2
    assert "@llvm.smax" not in ll and "@llvm.smin" not in ll
    assert re.search(r"trunc i64 %sel\d+ to i32", ll)
    assert re.search(
        r"@llvm\.amdgcn\.make\.buffer\.rsrc\.p1\(ptr addrspace\(1\) %gptr\d+, "
        r"i16 0, i32 %tr\d+,",
        ll,
    )


def test_compiles_to_a_real_hsaco():
    """The strongest CPU-only check: COMGR assembles and verifies the IR.

    A width-mismatched intrinsic call (what the scalar smax/smin ops would
    emit for an i64 operand) fails to assemble; this proves the i64 clamp
    path produces IR COMGR accepts end to end.
    """
    kernel, _ = _build()
    artifact = compile_kernel(kernel, arch="gfx950")
    assert artifact.hsaco_bytes > 0
