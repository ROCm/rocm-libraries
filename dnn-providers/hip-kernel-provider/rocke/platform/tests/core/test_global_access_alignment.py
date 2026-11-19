# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The alignment a global vector load or store claims, in every lowerer.

``memref.global_load_vN`` and ``memref.global_store_vN`` claim their ``align``
attribute, or, when they carry none, the natural alignment of the access: its
byte width when that is a power of two, else the element width
(``core.ir.claimed_align``). The builders always set ``align``, to the
caller's value or that same natural alignment, so only serialized or
hand-written IR reaches the lowerers' default; these tests feed such IR to the
LLVM and HIP lowerers. Each case runs per engine: the Python engine
everywhere, the C++ engine where its binding is built, emitting the same text
or refusing with the same message and status. The builder cases check the
``align`` the builders record. The scalar global loads and stores check a
given ``align`` with the same rule when they are built, and read it with that
rule (``core.ir.align_attr``) when they are lowered; without one they keep
their own claims. ``test_global_access_alignment.cpp`` runs the same cases
through the C API.
"""

from __future__ import annotations

import re

import pytest
from rocke.core.ir import (
    BF8E5M2,
    BF16,
    F16,
    F32,
    F64,
    FP8E4M3,
    I1,
    I8,
    I16,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    Op,
    PtrType,
    Type,
    VectorType,
)
from rocke.core.ir_serialize import parse
from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM20
from rocke.tests.engines import (
    ENGINES,
    assert_engine_rejects,
    assert_engine_rejects_hip,
    lower_hip_with_engine,
    lower_with_engine,
)

_ARCH = "gfx942"


def _copy(elem: str, lanes: int) -> KernelDef:
    """Serialized IR that loads ``lanes`` ``elem`` values from global memory and
    stores them back, neither op carrying an ``align`` attribute."""
    return parse(
        f"""ckdsl.ir v1
kernel @global_copy_without_align {{
  params {{
    %p : ptr<{elem},global>
  }}
  region @entry {{
    %c0 : i32 = arith.constant (  ) {{ ity = s:"i32", value = i:0 }}
    %v : vec<{elem}x{lanes}> = memref.global_load_vN ( %p, %c0 ) {{ elem_type = s:"{elem}", vec = i:{lanes} }}
    memref.global_store_vN ( %p, %c0, %v ) {{ elem_type = s:"{elem}", vec = i:{lanes} }}
    cf.return (  )
  }}
}}
"""
    )


def _store_of_a_parameter(elem: str, lanes: int) -> KernelDef:
    """Serialized IR storing a vector parameter, with no ``align`` attribute."""
    return parse(
        f"""ckdsl.ir v1
kernel @global_store_without_align {{
  params {{
    %p : ptr<{elem},global>
    %v : vec<{elem}x{lanes}>
  }}
  region @entry {{
    %c0 : i32 = arith.constant (  ) {{ ity = s:"i32", value = i:0 }}
    memref.global_store_vN ( %p, %c0, %v ) {{ elem_type = s:"{elem}", vec = i:{lanes} }}
    cf.return (  )
  }}
}}
"""
    )


def _scalar_copy() -> KernelDef:
    """Serialized IR that loads an ``f16`` and an ``f32`` from global memory and
    stores them back through the four scalar global ops, none of them carrying
    an ``align`` attribute."""
    return parse(
        """ckdsl.ir v1
kernel @global_scalar_copy_without_align {
  params {
    %h : ptr<f16,global>
    %f : ptr<f32,global>
  }
  region @entry {
    %c0 : i32 = arith.constant (  ) { ity = s:"i32", value = i:0 }
    %x : f16 = memref.global_load ( %h, %c0 )
    %y : f32 = memref.global_load_typed ( %f, %c0 ) { elem_type = s:"f32" }
    memref.global_store ( %h, %c0, %x ) { elem_type = s:"f16" }
    memref.global_store_typed ( %f, %c0, %y )
    cf.return (  )
  }
}
"""
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("lanes", "llvm_vector", "claim"), [(4, "<4 x float>", 16), (3, "<3 x float>", 4)]
)
def test_an_llvm_vector_access_without_align_claims_its_natural_alignment(
    engine, lanes, llvm_vector, claim
):
    """Sixteen bytes claim 16; a 12-byte <3 x float> claims its 4-byte element,
    where the load used to print an invalid ``align 6`` and the store was
    refused."""
    ll = lower_with_engine(
        _copy("f32", lanes), _ARCH, engine=engine, llvm_flavor=LLVM_FLAVOR_LLVM20
    )
    assert re.search(
        rf"= load {re.escape(llvm_vector)}, ptr addrspace\(1\) \S+, align {claim}$",
        ll,
        re.MULTILINE,
    )
    assert re.search(
        rf"store {re.escape(llvm_vector)} \S+, ptr addrspace\(1\) \S+, align {claim}$",
        ll,
        re.MULTILINE,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_a_hip_vector_access_without_align_claims_its_natural_alignment(engine):
    """A <4 x float> load and store claim 16 bytes and use the direct vector
    access; a <3 x float> pair claims 4 and copies only its 12-byte payload."""
    four = lower_hip_with_engine(_copy("f32", 4), _ARCH, engine=engine)
    assert "__builtin_memcpy" not in four
    assert "f32x4 v = *reinterpret_cast<const f32x4*>(p + c0);" in four
    assert "*reinterpret_cast<f32x4*>(p + c0) = v;" in four
    three = lower_hip_with_engine(_copy("f32", 3), _ARCH, engine=engine)
    assert "__builtin_memcpy(&v, __builtin_assume_aligned(p + c0, 4), 12);" in three
    assert "__builtin_memcpy(__builtin_assume_aligned(p + c0, 4), &v, 12);" in three


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("lanes,align", [(4, 16), (4, 1), (3, 4), (1, 1)])
@pytest.mark.parametrize("rebased", [False, True])
def test_hip_global_access_indexes_in_storage_units(engine, lanes, align, rebased):
    b = IRBuilder("word_copy")
    # The rebased case retains ptr<i32> in IR but the HIP emitter spells
    # global_ptr_add's result as const char*. Both cases need word addressing.
    ptr = PtrType(I32 if rebased else I8, "global")
    src = b.param("src", ptr)
    dst = b.param("dst", ptr)
    si = b.param("si", I64)
    di = b.param("di", I32)
    if rebased:
        src = b.global_ptr_add(src, b.const_i64(16))
        dst = b.global_ptr_add(dst, b.const_i64(16))
    if lanes == 1:
        value = b.global_load(src, si, I32, align=align)
        b.global_store(dst, di, value, align=align)
    else:
        value = b.global_load_vN(src, si, I32, lanes, align=align)
        if lanes != 3:  # The builder admits 96-bit loads, not stores.
            b.global_store_vN(dst, di, value, lanes, align=align)

    hip = lower_hip_with_engine(b.kernel, _ARCH, engine=engine)
    src_addr = f"reinterpret_cast<const unsigned char*>({src.name[1:]}) + (int64_t)si * (int64_t)sizeof(int)"
    dst_addr = (
        f"const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>({dst.name[1:]}))"
        " + (int64_t)di * (int64_t)sizeof(int)"
    )
    val = value.name[1:]
    if lanes == 4 and align == 16:
        assert f"*reinterpret_cast<const i32x4*>({src_addr})" in hip
        assert f"*reinterpret_cast<i32x4*>({dst_addr}) = {val};" in hip
    else:
        payload = 4 * lanes
        assert (
            f"__builtin_memcpy(&{val}, __builtin_assume_aligned({src_addr}, {align}), {payload});"
            in hip
        )
        if lanes != 3:
            assert (
                f"__builtin_memcpy(__builtin_assume_aligned({dst_addr}, {align}), &{val}, {payload});"
                in hip
            )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "elem,cpp_type,vec_type", [(F16, "fp16", "f16x4"), (BF16, "bf16", "bf16x4")]
)
@pytest.mark.parametrize("align", [8, 2])
def test_hip_rebased_moe_weights_use_halfword_indices(
    engine, elem, cpp_type, vec_type, align
):
    b = IRBuilder("grouped_moe_weights")
    weights = b.param("weights", PtrType(elem, "global"))
    tile_offset = b.param("tile_offset", I32)
    # A nonzero expert slab byte rebase, followed by an element (not byte) index.
    expert_offset = b.const_i64(256)
    base = b.global_ptr_add(weights, expert_offset)
    value = b.global_load_vN(base, tile_offset, elem, 4, align=align)

    hip = lower_hip_with_engine(b.kernel, _ARCH, engine=engine)
    ptr = base.name[1:]
    assert (
        f"const char* {ptr} = (const char*)weights + (int64_t){expert_offset.name[1:]};"
        in hip
    )
    addr = f"reinterpret_cast<const unsigned char*>({ptr}) + (int64_t)tile_offset * (int64_t)sizeof({cpp_type})"
    if align == 8:
        assert f"*reinterpret_cast<const {vec_type}*>({addr})" in hip
    else:
        assert (
            f"__builtin_memcpy(&{value.name[1:]}, __builtin_assume_aligned({addr}, 2), 8);"
            in hip
        )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("elem", "cpp_t", "size"), [(F16, "fp16", 2), (F32, "float", 4)]
)
@pytest.mark.parametrize("weak", [False, True])
def test_hip_scalar_load_store_match_the_claimed_alignment(
    engine, elem, cpp_t, size, weak
):
    b = IRBuilder("scalar_copy")
    ptr = b.param("p", PtrType(elem, "global"))
    idx = b.param("i", I32)
    align = 1 if weak else size
    value = b.global_load(ptr, idx, elem, align=align)
    b.global_store(ptr, idx, value, align=align)
    hip = lower_hip_with_engine(b.kernel, _ARCH, engine=engine)
    val = value.name[1:]
    if weak:
        base = "reinterpret_cast<const unsigned char*>(p)"
        offset = f" + (int64_t)i * (int64_t)sizeof({cpp_t})"
        assert f"__builtin_assume_aligned({base}{offset}, 1), {size});" in hip
        assert (
            f"__builtin_assume_aligned(const_cast<unsigned char*>({base}){offset}, 1), &{val}, {size});"
            in hip
        )
        assert "p[i]" not in hip
    else:
        assert f"{cpp_t} {val} = p[i];" in hip
        assert f"p[i] = {val};" in hip


@pytest.mark.parametrize("engine", ENGINES)
def test_hip_legacy_fp16_load_on_a_rebased_pointer(engine):
    b = IRBuilder("legacy_half_load")
    ptr = b.param("p", PtrType(F16, "global"))
    idx = b.param("i", I32)
    base = b.global_ptr_add(ptr, b.const_i64(16))
    value = b.global_load_f16(base, idx)
    hip = lower_hip_with_engine(b.kernel, _ARCH, engine=engine)
    val = value.name[1:]
    addr = f"reinterpret_cast<const unsigned char*>({base.name[1:]}) + (int64_t)i * (int64_t)sizeof(fp16)"
    assert f"__builtin_memcpy(&{val}, __builtin_assume_aligned({addr}, 2), 2);" in hip


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("align", [None, 1, 2])
def test_hip_legacy_fp16_store_on_a_rebased_pointer(engine, align):
    b = IRBuilder("legacy_half_store")
    ptr = b.param("p", PtrType(F16, "global"))
    idx = b.param("i", I32)
    value = b.param("value", F16)
    base = b.global_ptr_add(ptr, b.const_i64(16))
    b.store_f16(base, idx, value)
    store = b.kernel.body.ops[-1]
    if align is None:
        del store.attrs["align"]
    elif align == 1:
        store.attrs["align"] = align
    hip = lower_hip_with_engine(b.kernel, _ARCH, engine=engine)
    addr = (
        f"const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>({base.name[1:]}))"
        " + (int64_t)i * (int64_t)sizeof(fp16)"
    )
    assert (
        f"__builtin_memcpy(__builtin_assume_aligned({addr}, {align or 2}), &value, 2);"
        in hip
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_hip_legacy_fp16_store_rejects_non_fp16_value(engine):
    b = IRBuilder("legacy_half_store_wrong_type")
    ptr = b.param("p", PtrType(F16, "global"))
    idx = b.param("i", I32)
    value = b.param("value", F32)
    base = b.global_ptr_add(ptr, b.const_i64(16))
    b.store_f16(base, idx, value)
    assert_engine_rejects_hip(
        b.kernel,
        _ARCH,
        engine=engine,
        error_type=ValueError,
        message="memref.global_store: expected an f16 value",
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("rebased", [False, True])
@pytest.mark.parametrize(
    "method,elem,cpp_t,builtin,subscript",
    [
        ("global_atomic_add", I32, "int", "atomicAdd", True),
        ("global_atomic_add", F32, "float", "atomicAdd", True),
        ("global_atomic_add_f32", F32, "float", "atomicAdd", False),
        ("global_flag_store", I32, "int", "__hip_atomic_store", True),
        ("global_flag_wait_eq", I32, "int", "__hip_atomic_load", True),
        (
            "global_atomic_add_pk_f16",
            F16,
            "fp16",
            "__builtin_amdgcn_global_atomic_fadd_v2f16",
            False,
        ),
        (
            "global_atomic_add_pk_bf16",
            BF16,
            "bf16",
            "__builtin_amdgcn_global_atomic_fadd_v2bf16",
            False,
        ),
    ],
)
def test_hip_global_atomic_uses_scalar_access_units(
    engine, rebased, method, elem, cpp_t, builtin, subscript
):
    b = IRBuilder("atomic_access")
    base = b.param("p", PtrType(elem, "global"))
    packed = method.startswith("global_atomic_add_pk_")
    idx = b.param("idx", I64 if packed else I32)
    val = b.param("val", VectorType(elem, 2) if packed else elem)
    ptr = b.global_ptr_add(base, b.const_i64(16)) if rebased else base
    getattr(b, method)(ptr, idx, val)
    b.ret()
    hip = lower_hip_with_engine(b.kernel, "gfx950", engine=engine)
    if rebased:
        match = re.search(r"const char\* (\w+) = \(const char\*\)p \+", hip)
        assert match is not None
        byte_base = f"reinterpret_cast<const unsigned char*>({match[1]})"
        read = method == "global_flag_wait_eq"
        if not read:
            byte_base = f"const_cast<unsigned char*>({byte_base})"
        qualifier = "const " if read else ""
        address = (
            f"reinterpret_cast<{qualifier}{cpp_t}*>({byte_base}"
            f" + (int64_t)idx * (int64_t)sizeof({cpp_t}))"
        )
    else:
        address = "&p[idx]" if subscript else "p + idx"
    if packed:
        aligned = re.search(
            rf"typedef {cpp_t} (\w+) __attribute__\(\(aligned\(4\)\)\);", hip
        )
        assert aligned is not None
        address = f"reinterpret_cast<{aligned[1]}*>({address})"
    assert f"{builtin}({address}, " in hip
    assert "__builtin_memcpy" not in hip


@pytest.mark.parametrize("engine", ENGINES)
def test_hip_rebased_scalar_i1_uses_a_byte(engine):
    b = IRBuilder("bool_copy")
    ptr = b.param("p", PtrType(I1, "global"))
    idx = b.param("i", I32)
    base = b.global_ptr_add(ptr, b.const_i64(3))
    value = b.global_load(base, idx, I1)
    b.global_store(base, idx, value)
    hip = lower_hip_with_engine(b.kernel, _ARCH, engine=engine)
    addr = f"reinterpret_cast<const unsigned char*>({base.name[1:]})"
    offset = " + (int64_t)i * (int64_t)sizeof(unsigned char)"
    val = value.name[1:]
    assert f"bool {val} = (*({addr}{offset}) & 1u) != 0;" in hip
    assert (
        f"*(const_cast<unsigned char*>({addr}){offset}) = (unsigned char){val};" in hip
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_a_sub_byte_vector_store_without_align_is_refused(engine):
    """Four ``i1`` lanes have no byte width to claim, so the LLVM lowering
    refuses the store rather than give it two bytes per lane. (The HIP
    lowering refuses the ``i1`` element type before it reaches the
    alignment.)"""
    assert_engine_rejects(
        _store_of_a_parameter("i1", 4),
        _ARCH,
        engine=engine,
        error_type=ValueError,
        message="type_bytes failed - type=i1 (1 bits), expected a whole number of bytes",
        llvm_flavor=LLVM_FLAVOR_LLVM20,
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("align", "error_type", "refusal"),
    [
        (6, ValueError, "align=6, expected a positive power of two"),
        (True, TypeError, "align=bool, expected an integer"),
    ],
)
def test_an_llvm_vector_load_refuses_an_invalid_align(
    engine, align, error_type, refusal
):
    """The LLVM vector load checks its ``align`` as the store does, instead of
    printing an invalid claim such as ``align 6``."""
    kernel = _copy("f32", 3)
    (load,) = [op for op in kernel.body.ops if op.name == "memref.global_load_vN"]
    load.attrs["align"] = align
    assert_engine_rejects(
        kernel,
        _ARCH,
        engine=engine,
        error_type=error_type,
        message=f"memref.global_load_vN failed - {refusal}",
        llvm_flavor=LLVM_FLAVOR_LLVM20,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_an_llvm_scalar_global_access_without_align_keeps_its_claim(engine):
    """Without ``align`` the ``f16`` load and store claim 2 bytes and the typed
    load and store claim 1, as each did before it read the shared rule."""
    ll = lower_with_engine(
        _scalar_copy(), _ARCH, engine=engine, llvm_flavor=LLVM_FLAVOR_LLVM20
    )
    assert re.search(r"= load half, ptr addrspace\(1\) \S+, align 2$", ll, re.MULTILINE)
    assert re.search(
        r"= load float, ptr addrspace\(1\) \S+, align 1$", ll, re.MULTILINE
    )
    assert re.search(
        r"store half \S+, ptr addrspace\(1\) \S+, align 2$", ll, re.MULTILINE
    )
    assert re.search(
        r"store float \S+, ptr addrspace\(1\) \S+, align 1$", ll, re.MULTILINE
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "op_name",
    [
        "memref.global_load",
        "memref.global_load_typed",
        "memref.global_store",
        "memref.global_store_typed",
    ],
)
@pytest.mark.parametrize(
    ("align", "error_type", "refusal"),
    [
        (True, TypeError, "align=bool, expected an integer"),
        (8.0, TypeError, "align=float, expected an integer"),
        (24, ValueError, "align=24, expected a positive power of two"),
    ],
)
def test_an_llvm_scalar_global_access_refuses_an_invalid_align(
    engine, op_name, align, error_type, refusal
):
    """A scalar global load or store refuses an ``align`` that is not a
    positive power of two with the vector ops' text, instead of printing
    ``align 24`` or reading ``True`` or ``8.0`` as another alignment."""
    kernel = _scalar_copy()
    (access,) = [op for op in kernel.body.ops if op.name == op_name]
    access.attrs["align"] = align
    assert_engine_rejects(
        kernel,
        _ARCH,
        engine=engine,
        error_type=error_type,
        message=f"{op_name} failed - {refusal}",
        llvm_flavor=LLVM_FLAVOR_LLVM20,
    )


@pytest.mark.parametrize(
    ("elem", "lanes", "claim"),
    [(F32, 4, 16), (F32, 3, 4), (F16, 6, 2), (F16, 16, 32), (I8, 12, 1)],
)
def test_the_global_vector_load_builder_defaults_to_the_natural_alignment(
    elem, lanes, claim
):
    """Without ``align`` the builder records ``vector_access_align``: the byte
    width when it is a power of two, else the element width. A given power of
    two is recorded as given."""
    b = IRBuilder("global_load_default_align")
    p = b.param("p", PtrType(elem, "global"))
    c0 = b.const_i32(0)
    b.global_load_vN(p, c0, elem, lanes)
    b.global_load_vN(p, c0, elem, lanes, align=8)
    loads = [op for op in b.kernel.body.ops if op.name == "memref.global_load_vN"]
    assert [op.attrs["align"] for op in loads] == [claim, 8]


@pytest.mark.parametrize(
    ("elem", "lanes", "claim"), [(F32, 4, 16), (F32, 1, 4), (F64, 2, 16), (I8, 16, 16)]
)
def test_the_global_vector_store_builder_defaults_to_the_natural_alignment(
    elem, lanes, claim
):
    """Without ``align`` the store builder records ``vector_access_align``,
    for a scalar value (``n=1``) as for a vector."""
    b = IRBuilder("global_store_default_align")
    p = b.param("p", PtrType(elem, "global"))
    v = b.param("v", elem if lanes == 1 else VectorType(elem, lanes))
    b.global_store_vN(p, b.const_i32(0), v, lanes)
    (store,) = [op for op in b.kernel.body.ops if op.name == "memref.global_store_vN"]
    assert store.attrs["align"] == claim


@pytest.mark.parametrize(
    ("align", "error_type", "refusal"),
    [
        (6, ValueError, "align=6, expected a positive power of two"),
        (0, ValueError, "align=0, expected a positive power of two"),
        (-4, ValueError, "align=-4, expected a positive power of two"),
        (True, TypeError, "align=bool, expected an integer"),
        (8.0, TypeError, "align=float, expected an integer"),
    ],
)
def test_the_global_vector_builders_refuse_an_invalid_align(align, error_type, refusal):
    """A given ``align`` that is not a positive power of two is refused when
    the op is built, with the lowering's text, rather than recorded (``True``
    as 1, ``8.0`` as 8, ``0`` as the default)."""
    b = IRBuilder("global_vector_bad_align")
    p = b.param("p", PtrType(F32, "global"))
    v = b.param("v", VectorType(F32, 4))
    c0 = b.const_i32(0)
    with pytest.raises(error_type) as load_refusal:
        b.global_load_vN(p, c0, F32, 4, align=align)
    assert str(load_refusal.value) == f"memref.global_load_vN failed - {refusal}"
    with pytest.raises(error_type) as store_refusal:
        b.global_store_vN(p, c0, v, 4, align=align)
    assert str(store_refusal.value) == f"memref.global_store_vN failed - {refusal}"


# Every scalar global builder: its name, the op it records, the element type it
# moves, and the ``align`` it records when the caller gives none (``None``:
# the op records no ``align``).
_SCALAR_BUILDERS = [
    ("global_load", "memref.global_load_typed", F32, 1),
    ("global_load_f16", "memref.global_load", F16, 2),
    ("global_load_f32", "memref.global_load_typed", F32, 4),
    ("global_load_i32", "memref.global_load_typed", I32, 4),
    ("global_load_i64", "memref.global_load_typed", I64, 8),
    ("global_load_bf16", "memref.global_load_typed", BF16, 2),
    ("global_load_f64", "memref.global_load_typed", F64, 8),
    ("global_load_i8", "memref.global_load_typed", I8, 1),
    ("global_load_i16", "memref.global_load_typed", I16, 2),
    ("global_load_bf8e5m2", "memref.global_load_typed", BF8E5M2, 1),
    ("global_load_fp8e4m3", "memref.global_load_typed", FP8E4M3, 1),
    ("masked_global_load", "memref.global_load_typed", F32, 1),
    ("global_store", "memref.global_store_typed", F32, 1),
    ("store", "memref.global_store_typed", F32, None),
]


def _build_scalar_access(builder_name: str, elem: Type, **align: object) -> Op:
    """Build one access through scalar global builder ``builder_name`` on a
    ``ptr<elem>`` parameter, passing ``align`` only when given; return the
    memory op it recorded."""
    b = IRBuilder(f"{builder_name}_align")
    ptr = b.param("p", PtrType(elem, "global"))
    idx = b.const_i32(0)
    if builder_name == "global_load":
        b.global_load(ptr, idx, elem, **align)
    elif builder_name == "masked_global_load":
        mask = b.param("mask", I1)
        other = b.param("other", elem)
        b.masked_global_load(ptr, idx, mask, other, elem, **align)
    elif builder_name == "global_store":
        b.global_store(ptr, idx, b.param("v", elem), **align)
    elif builder_name == "store":
        b.store(b.param("v", elem), ptr, idx, **align)
    else:
        getattr(b, builder_name)(ptr, idx, **align)
    (access,) = [op for op in b.kernel.body.ops if op.name.startswith("memref.global_")]
    return access


@pytest.mark.parametrize(
    ("builder_name", "op_name", "elem", "default"),
    _SCALAR_BUILDERS,
    ids=[row[0] for row in _SCALAR_BUILDERS],
)
def test_a_scalar_global_builder_records_its_default_or_the_given_align(
    builder_name, op_name, elem, default
):
    """Without ``align`` each scalar builder records its own default (``store``
    records none); a given power of two is recorded as given."""
    omitted = _build_scalar_access(builder_name, elem)
    assert omitted.name == op_name
    assert omitted.attrs.get("align") == default
    assert _build_scalar_access(builder_name, elem, align=8).attrs["align"] == 8


@pytest.mark.parametrize(
    ("builder_name", "op_name", "elem"),
    [row[:3] for row in _SCALAR_BUILDERS],
    ids=[row[0] for row in _SCALAR_BUILDERS],
)
@pytest.mark.parametrize(
    ("align", "error_type", "refusal"),
    [
        (0, ValueError, "align=0, expected a positive power of two"),
        (-4, ValueError, "align=-4, expected a positive power of two"),
        (3, ValueError, "align=3, expected a positive power of two"),
        (24, ValueError, "align=24, expected a positive power of two"),
        (True, TypeError, "align=bool, expected an integer"),
        (8.0, TypeError, "align=float, expected an integer"),
    ],
)
def test_a_scalar_global_builder_refuses_an_invalid_align(
    builder_name, op_name, elem, align, error_type, refusal
):
    """A given ``align`` that is not a positive power of two is refused when
    the op is built, with the lowering's text, rather than recorded (``True``
    as 1, ``8.0`` as 8)."""
    with pytest.raises(error_type) as refused:
        _build_scalar_access(builder_name, elem, align=align)
    assert str(refused.value) == f"{op_name} failed - {refusal}"
