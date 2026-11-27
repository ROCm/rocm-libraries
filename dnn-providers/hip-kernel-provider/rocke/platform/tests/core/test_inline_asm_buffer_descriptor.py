# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""An inline-asm operand that is a buffer descriptor takes its LLVM type.

``tile.buffer_rsrc`` results are ``<4 x i32>`` in the IR, but the LLVM value
that ``make.buffer.rsrc`` defines is a ``ptr addrspace(8)``. An inline-asm call
that printed such an operand with its IR type referenced a value under the
wrong type, which the LLVM parser rejects. Both engines print the descriptor as
``ptr addrspace(8)``, keep every other ``<4 x i32>`` operand as it is, and the
module compiles and links (an LDS DMA through ``buffer_load ... lds`` as the
gfx950 conv kernels issue it).
"""

from __future__ import annotations

import re

from rocke.core.ir import F16, I32, IRBuilder, PtrType
from rocke.core.llvm_profile import LLVM_FLAVORS
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.ir_validity import assert_kernel_target_accepts

_DMA = "s_mov_b32 m0, $3\n\tbuffer_load_dwordx4 $0, $1, $2 offen lds"


def _dma_kernel(name: str):
    """One 16-byte ``buffer_load ... lds`` per lane, its descriptor an asm operand."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(F16, "global"))
    rsrc = b.buffer_rsrc(x, b.const_i32(64 * 16))
    lds = b.smem_alloc(F16, [1, 64 * 8], name_hint="dst")
    dst = b.readfirstlane(b.trunc(b.smem_addr_of(lds), I32))
    voff = b.mul(b.thread_id_x(), b.const_i32(16))
    b.inline_asm(_DMA, "v,s,s,s,~{m0},~{memory}", [voff, rsrc, b.const_i32(0), dst])
    b.ret()
    return b.kernel


def _vector_kernel(name: str):
    """A ``<4 x i32>`` asm operand that is not a buffer descriptor."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(I32, "global"))
    v = b.global_load_vN(x, b.const_i32(0), I32, 4, align=16)
    b.inline_asm("; $0", "v", [v])
    b.ret()
    return b.kernel


def _asm_call(text: str) -> str:
    (line,) = [ln for ln in text.splitlines() if " asm sideeffect " in ln]
    return line


def test_both_engines_type_a_descriptor_operand_as_its_llvm_definition():
    for flavor in LLVM_FLAVORS:
        text = lower_with_both_engines(
            _dma_kernel("asm_rsrc"), "gfx950", llvm_flavor=flavor
        )
        (rsrc,) = re.findall(
            r"(%\S+) = call ptr addrspace\(8\) @llvm\.amdgcn\.make\.buffer\.rsrc", text
        )
        call = _asm_call(text)
        assert f"ptr addrspace(8) {rsrc}" in call, flavor
        assert "<4 x i32>" not in call, flavor


def test_both_engines_keep_the_ir_type_of_other_vector_operands():
    for flavor in LLVM_FLAVORS:
        text = lower_with_both_engines(
            _vector_kernel("asm_v4"), "gfx950", llvm_flavor=flavor
        )
        assert re.search(r"\(<4 x i32> %\S+\)$", _asm_call(text)), flavor
        assert "ptr addrspace(8)" not in _asm_call(text), flavor


def test_a_descriptor_operand_compiles_and_links():
    assert_kernel_target_accepts(_dma_kernel("asm_rsrc_link"), "gfx950")
