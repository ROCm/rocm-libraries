# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The alignment a global vector load or store claims, in every lowerer.

``memref.global_load_vN`` and ``memref.global_store_vN`` claim their ``align``
attribute, or, when they carry none, the natural alignment of the access: its
byte width when that is a power of two, else the element width
(``core.ir.claimed_align``). The builders always set ``align``, so only
serialized or hand-written IR reaches the default; these tests feed such IR to
the LLVM and HIP lowerers. Each case runs per engine: the Python engine
everywhere, the C++ engine where its binding is built, emitting the same text
or refusing with the same message and status.
``test_global_access_alignment.cpp`` runs the same cases through the C API.
"""

from __future__ import annotations

import re

import pytest
from rocke.core.ir import KernelDef
from rocke.core.ir_serialize import parse
from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM20
from rocke.tests.engines import (
    ENGINES,
    assert_engine_rejects,
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
