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


def test_no_shipped_catalog_atom_claims_it_yet():
    """Registering the name alone must not enable selection on any arch.

    Catalog rows land in a later step; until then every atom's operand and
    accumulator dtypes stay fp32/bf16/etc. so ordinary fp32 GEMM selection is
    provably unchanged by this commit.
    """
    from rocke.core.arch import ArchTarget
    from rocke.core.arch.target import known_arches

    for gfx in known_arches():
        for op in ArchTarget.from_gfx(gfx).mma.ops:
            for field in (op.a_dtype, op.b_dtype, op.c_dtype, op.result_dtype):
                assert field != "xf32", f"{gfx}: {op.op_id} already claims xf32"
