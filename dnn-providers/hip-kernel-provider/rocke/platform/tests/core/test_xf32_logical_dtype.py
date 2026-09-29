# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``xf32`` is a matrix-selection tag, not a storage format.

It names "fp32 operands, reduced-precision matrix path" -- either a native XF32
MMA or the BF16x3 emulation. Operands stay fp32 in memory and in registers, so
``xf32`` must be recognised by dtype normalization and the arch catalog while
staying out of the IR scalar type system and every memory-op path. These tests
pin that boundary: registering the tag must not quietly make it loadable.
"""

import pytest
from rocke.core.dtypes import normalize_dtype


@pytest.mark.parametrize(
    "spelling",
    ["xf32", "XF32", " xf32 ", " XF32\t", "tf32", "TF32", " Tf32 "],
)
def test_spellings_canonicalise_to_xf32(spelling):
    """Both the AMD (``xf32``) and NVIDIA/CK (``tf32``) spellings land on one key."""
    assert normalize_dtype(spelling) == "xf32"


def test_canonical_key_is_idempotent():
    assert normalize_dtype(normalize_dtype("tf32")) == "xf32"


def test_is_not_an_ir_scalar_type():
    """No IR scalar type carries the name -- xf32 has no storage representation.

    ``fp6e2m3`` is the precedent: a logical dtype the catalog knows about that
    the IR type system deliberately does not.
    """
    from rocke.core import ir

    # The scalar singletons are exactly the module-level ``Type`` instances;
    # composites (vector/ptr/smem) are subclasses built on demand, never globals.
    scalar_names = {t.name for t in vars(ir).values() if type(t) is ir.Type}
    assert "f32" in scalar_names, "sanity: real scalars are discovered"
    assert "xf32" not in scalar_names
    assert "tf32" not in scalar_names


def test_memory_ops_reject_it():
    """The I/O dtype gate refuses xf32, so no load/store can be built with it."""
    from rocke.helpers.io import io_ir_type

    for spelling in ("xf32", "tf32"):
        with pytest.raises(ValueError, match="unsupported I/O dtype"):
            io_ir_type(spelling)


def test_xf32_appears_only_as_an_operand_dtype():
    """`xf32` selects a matrix path; it never names a result.

    An atom may take xf32 on A and B -- that is the whole point of the tag --
    but C and D describe values that really exist in registers, and no such
    value is ever "xf32". This is what keeps the tag out of any code that would
    need a representation for it.
    """
    from rocke.core.arch import ArchTarget
    from rocke.core.arch.target import known_arches

    for gfx in known_arches():
        for op in ArchTarget.from_gfx(gfx).mma.ops:
            for field in (op.c_dtype, op.result_dtype):
                assert field != "xf32", f"{gfx}: {op.op_id} accumulates into xf32"


def test_xf32_operands_are_confined_to_gfx950_emulated_rows():
    """Pins the blast radius of enabling the tag.

    Only gfx950 ships XF32 today, and only as emulation. Every other arch's
    selection is untouched by construction, and an accidental native-looking
    xf32 row (one with no expansion behind it) is caught here rather than in
    the lowering.
    """
    from rocke.core.arch import ArchTarget
    from rocke.core.arch.target import known_arches

    for gfx in known_arches():
        for op in ArchTarget.from_gfx(gfx).mma.ops:
            if "xf32" not in (op.a_dtype, op.b_dtype):
                continue
            assert gfx == "gfx950", f"{gfx}: unexpected xf32 atom {op.op_id}"
            assert op.a_dtype == op.b_dtype == "xf32", f"{op.op_id}: mixed operands"
            assert not op.is_native, f"{op.op_id}: xf32 atom claims a native encoding"
