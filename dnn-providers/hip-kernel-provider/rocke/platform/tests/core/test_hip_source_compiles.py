# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The HIP source both engines emit is accepted by hipcc.

Each kernel here holds one construct that the HIP lowering once spelled in a
way hipcc rejects or miscompiles:

* a loop-carried value whose type the loop declared as ``auto`` with no
  initializer;
* a global load or store that moves a value of another type than the
  pointer's element, which the LLVM route moves at the op's alignment;
* a global atomic add of a value of another type than the pointer's element
  (the LLVM route steps the pointer by the value's type);
* a pointer moved by ``global_ptr_add``, which came back as ``const char*``;
* a packed two-element atomic add, whose address clang took to be aligned to
  one element only;
* one value stored twice in a block through the typed buffer stores, which
  redeclared the store's temporary.

For every target the kernel compiles for, both engines must emit the same
text and ``hipcc --genco`` must accept it; where the fix is about what clang
makes of the text, the device LLVM IR hipcc emits is checked too. No GPU is
needed. A target the installed hipcc does not know is skipped; the CI image's
newer hipcc knows them all.
"""

from __future__ import annotations

import functools
import re
import shutil

import pytest
from rocke.core.arch.target import base_arch_from_target_id
from rocke.core.ir import (
    BF16,
    F16,
    F32,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    VectorType,
)
from rocke.helpers.compile import (
    compile_kernel_via_hipcc,
    emit_device_llvm_ir_via_hipcc,
    hipcc_accepts_target,
)
from rocke.tests.engines import lower_hip_with_both_engines

ALL_TARGETS = (
    "gfx90a",
    "gfx942",
    "gfx950",
    "gfx1151",
    "gfx1201",
    "gfx1250",
    "gfx11-generic",
)
# Targets with both packed global atomic adds (f16 and bf16).
PACKED_ATOMIC_TARGETS = ("gfx942", "gfx950", "gfx1250")


# One probe per target and run: whether the hipcc on PATH knows it.
_hipcc_accepts = functools.cache(hipcc_accepts_target)


def _accesses_alias_anything(llvm_ir: str, access_pattern: str) -> bool:
    """True when every load or store matching ``access_pattern`` in
    ``llvm_ir`` carries the TBAA tag of ``char``, which may alias any type."""
    tags = re.findall(access_pattern + r".*!tbaa (!\d+)", llvm_ir)
    if not tags:
        return False
    for tag in tags:
        tag_definition = re.search(rf"^{tag} = !{{(!\d+), ", llvm_ir, re.MULTILINE)
        if tag_definition is None:
            return False
        if not re.search(
            rf'^{tag_definition.group(1)} = !{{!"omnipotent char"',
            llvm_ir,
            re.MULTILINE,
        ):
            return False
    return True


def _hip_source_compiled(kernel: KernelDef, target: str) -> str:
    """The HIP source both engines emit for ``kernel`` on ``target``, after
    hipcc has compiled it; skips when hipcc is missing or does not know
    ``target``."""
    source = lower_hip_with_both_engines(kernel, base_arch_from_target_id(target))
    if shutil.which("hipcc") is None:
        pytest.skip("hipcc not found on PATH")
    if not _hipcc_accepts(target):
        pytest.skip(f"the hipcc on PATH does not know {target}")
    assert compile_kernel_via_hipcc(kernel, arch=target).hsaco
    return source


def _loop_carry_kernel() -> KernelDef:
    """A loop carrying an i32x4 and a bf16x8 (types the loop once declared
    ``auto``), stored after the loop."""
    b = IRBuilder("hip_loop_carry")
    ints = b.param("I", PtrType(I32, "global"))
    bf16_values = b.param("H", PtrType(BF16, "global"))
    trips = b.param("trips", I32)
    lane = b.thread_id_x()
    int_init = b.global_load(ints, lane, VectorType(I32, 4), align=16)
    bf16_init = b.global_load(bf16_values, lane, VectorType(BF16, 8), align=16)
    loop = b.scf_for_iter(
        b.const_i32(0),
        trips,
        b.const_i32(1),
        [("acc_i", int_init), ("acc_h", bf16_init)],
    )
    with loop as (_, (acc_i, acc_h)):
        b.scf_yield(b.add(acc_i, acc_i), acc_h)
    int_out, bf16_out = loop.results
    b.global_store(ints, lane, int_out, align=16)
    b.global_store(bf16_values, lane, bf16_out, align=16)
    b.ret()
    return b.kernel


def _retyped_access_kernel() -> KernelDef:
    """An f32x8 loaded and stored through an f32 pointer at 4-byte alignment,
    an f16 stored through a bf16 pointer, and an i32x4 loaded through an f32
    pointer by an op with no ``align`` attr."""
    b = IRBuilder("hip_retyped_access")
    src = b.param("S", PtrType(F32, "global"))
    dst = b.param("D", PtrType(F32, "global"))
    out = b.param("O", PtrType(BF16, "global"))
    ints = b.param("I", PtrType(I32, "global"))
    lane = b.thread_id_x()
    wide = b.global_load(src, lane, VectorType(F32, 8), align=4)
    b.global_store(dst, lane, wide, align=4)
    b.global_store(out, lane, b.global_load(src, lane, F16, align=2), align=2)
    # Written past the builder, which always records an alignment.
    unaligned_type = VectorType(I32, 4)
    unaligned = b._op(
        "memref.global_load_typed",
        [src, lane],
        [unaligned_type],
        attrs={"elem_type": unaligned_type.name},
    ).result
    b.global_store(ints, lane, unaligned, align=16)
    b.ret()
    return b.kernel


def _retyped_atomic_kernel() -> KernelDef:
    """f32 atomic adds, with and without a returned value, through an i32
    pointer."""
    b = IRBuilder("hip_retyped_atomic")
    counters = b.param("P", PtrType(I32, "global"))
    lane = b.thread_id_x()
    one = b.const_f32(1.0)
    before = b.global_atomic_add(counters, lane, one)
    b.global_atomic_add_f32(counters, b.add(lane, b.const_i32(64)), before)
    b.ret()
    return b.kernel


def _moved_pointer_kernel() -> KernelDef:
    """An i32 counter reached through ``global_ptr_add``: an atomic add, a load
    and a store through the moved pointer."""
    b = IRBuilder("hip_moved_pointer")
    counters = b.param("C", PtrType(I32, "global"))
    byte_off = b.param("off", I64)
    moved = b.global_ptr_add(counters, byte_off)
    lane = b.thread_id_x()
    before = b.global_atomic_add(moved, lane, b.const_i32(1))
    neighbour = b.global_load(moved, b.add(lane, b.const_i32(1)), I32, align=4)
    b.global_store(
        moved, b.add(lane, b.const_i32(64)), b.add(before, neighbour), align=4
    )
    b.ret()
    return b.kernel


def _packed_atomic_kernel() -> KernelDef:
    """Packed f16 and bf16 atomic adds at even element indices."""
    b = IRBuilder("hip_packed_atomic")
    f16_values = b.param("H", PtrType(F16, "global"))
    bf16_values = b.param("B", PtrType(BF16, "global"))
    even = b.mul(b.thread_id_x(), b.const_i32(2))
    b.global_atomic_add_pk_f16(
        f16_values,
        even,
        b.global_load(f16_values, even, VectorType(F16, 2), align=4),
    )
    b.global_atomic_add_pk_bf16(
        bf16_values,
        even,
        b.global_load(bf16_values, even, VectorType(BF16, 2), align=4),
    )
    b.ret()
    return b.kernel


def _repeated_buffer_store_kernel() -> KernelDef:
    """One f16, one bf16 and one f32x4 value, each stored twice in one block
    through the typed buffer stores."""
    b = IRBuilder("hip_repeated_buffer_store")
    f16_values = b.param("H", PtrType(F16, "global"))
    bf16_values = b.param("B", PtrType(BF16, "global"))
    f32_values = b.param("F", PtrType(F32, "global"))
    num_bytes = b.param("num_bytes", I32)
    lane = b.thread_id_x()
    soffset = b.const_i32(0)
    voffset = b.mul(lane, b.const_i32(16))
    f16_rsrc = b.buffer_rsrc(f16_values, num_bytes)
    bf16_rsrc = b.buffer_rsrc(bf16_values, num_bytes)
    f32_rsrc = b.buffer_rsrc(f32_values, num_bytes)
    f16_value = b.global_load(f16_values, lane, F16, align=2)
    bf16_value = b.global_load(bf16_values, lane, BF16, align=2)
    f32x4_value = b.global_load(f32_values, lane, VectorType(F32, 4), align=16)
    for repeat_offset in (voffset, b.add(voffset, b.const_i32(4))):
        b.buffer_store_f16(f16_rsrc, repeat_offset, soffset, f16_value)
        b.buffer_store_bf16(bf16_rsrc, repeat_offset, soffset, bf16_value)
        b.buffer_store_vN_f32(f32_rsrc, repeat_offset, soffset, f32x4_value, 4)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_loop_carried_values_are_declared_with_their_types(target):
    source = _hip_source_compiled(_loop_carry_kernel(), target)
    assert re.search(r"^\s*i32x4 for\d+;$", source, re.MULTILINE), source
    assert re.search(r"^\s*bf16x8 for\d+;$", source, re.MULTILINE), source
    assert not re.search(r"^\s*auto \w+;$", source, re.MULTILINE), source


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_a_value_of_another_type_moves_its_bits_at_the_ops_alignment(target):
    kernel = _retyped_access_kernel()
    source = _hip_source_compiled(kernel, target)
    for alias_declaration, access in (
        ("typedef f32x8 f32x8_align4 __attribute__((aligned(4), may_alias));",
         r"f32x8 gl\d+ = reinterpret_cast<f32x8_align4\*>\(S\)\[\w+\];"),
        ("typedef f32x8 f32x8_align4 __attribute__((aligned(4), may_alias));",
         r"reinterpret_cast<f32x8_align4\*>\(D\)\[\w+\] = gl\d+;"),
        ("typedef fp16 fp16_align2 __attribute__((aligned(2), may_alias));",
         r"reinterpret_cast<fp16_align2\*>\(O\)\[\w+\] = gl\d+;"),
        ("typedef i32x4 i32x4_align1 __attribute__((aligned(1), may_alias));",
         r"i32x4 \w+ = reinterpret_cast<i32x4_align1\*>\(S\)\[\w+\];"),
    ):  # fmt: skip
        assert re.search(re.escape(alias_declaration) + " " + access, source), (
            alias_declaration,
            access,
            source,
        )
    llvm_ir = emit_device_llvm_ir_via_hipcc(kernel, arch=target)
    for access in (
        r"load <8 x float>, ptr addrspace\(1\) %\w+, align 4\b",
        r"store <8 x float> %\w+, ptr addrspace\(1\) %\w+, align 4\b",
        r"store half %\w+, ptr addrspace\(1\) %\w+, align 2\b",
        r"load <4 x i32>, ptr addrspace\(1\) %\w+, align 1\b",
    ):
        assert re.search(access, llvm_ir), (access, llvm_ir)
        assert _accesses_alias_anything(llvm_ir, access), (access, llvm_ir)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_an_atomic_add_of_another_type_adds_through_the_values_type(target):
    kernel = _retyped_atomic_kernel()
    source = _hip_source_compiled(kernel, target)
    assert re.search(
        r"float atom_add\d+ = atomicAdd\(&reinterpret_cast<float\*>\(P\)\[\w+\], ",
        source,
    ), source
    assert re.search(
        r"^\s*atomicAdd\(reinterpret_cast<float\*>\(P\) \+ \w+, atom_add\d+\);$",
        source,
        re.MULTILINE,
    ), source
    llvm_ir = emit_device_llvm_ir_via_hipcc(kernel, arch=target)
    float_adds = re.findall(
        r"atomicrmw fadd ptr addrspace\(1\) %\w+, float .*, align 4\b", llvm_ir
    )
    assert len(float_adds) == 2, llvm_ir


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_a_moved_pointer_keeps_its_element_type(target):
    source = _hip_source_compiled(_moved_pointer_kernel(), target)
    assert re.search(
        r"\bint\* gptr\d+ = \(int\*\)\(\(char\*\)C \+ \(int64_t\)off\);", source
    ), source
    for access in (
        r"atomicAdd\(&gptr\d+\[\w+\], ",
        r"\bint gl\d+ = gptr\d+\[\w+\];",
        r"^\s*gptr\d+\[\w+\] = \w+;$",
    ):
        assert re.search(access, source, re.MULTILINE), (access, source)


@pytest.mark.parametrize("target", PACKED_ATOMIC_TARGETS)
def test_a_packed_atomic_add_addresses_the_packed_value(target):
    kernel = _packed_atomic_kernel()
    source = _hip_source_compiled(kernel, target)
    assert re.search(
        r"__builtin_amdgcn_global_atomic_fadd_v2f16\(reinterpret_cast<f16x2\*>\(H \+ \w+\), ",
        source,
    ), source
    assert re.search(
        r"__builtin_amdgcn_global_atomic_fadd_v2bf16\(reinterpret_cast<bf16x2\*>\(B \+ \w+\), ",
        source,
    ), source
    # The packed value's 4-byte alignment reaches the atomic; through an
    # element pointer clang aligned it to one 2-byte element.
    llvm_ir = emit_device_llvm_ir_via_hipcc(kernel, arch=target)
    for packed_type in ("half", "bfloat"):
        assert re.search(
            rf"atomicrmw fadd ptr addrspace\(1\) %\w+, <2 x {packed_type}> %\w+ "
            r".*, align 4\b",
            llvm_ir,
        ), (packed_type, llvm_ir)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_a_value_stored_twice_through_a_typed_buffer_store_compiles(target):
    source = _hip_source_compiled(_repeated_buffer_store_kernel(), target)
    stores = re.findall(
        r"\{ (\w+(?: \w+)?) _bs; __builtin_memcpy\(&_bs, &\w+, (\d+)\); ", source
    )
    assert sorted(stores) == sorted(
        [("unsigned short", "2")] * 4 + [("i32x4", "16")] * 2
    ), stores
