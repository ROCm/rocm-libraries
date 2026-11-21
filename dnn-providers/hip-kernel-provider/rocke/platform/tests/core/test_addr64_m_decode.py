# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``tile_origin_decode_i64`` / ``carry_decode_i32`` -- the addr64 M-decode
pair: an exact i64 division decoding the workgroup-uniform tile origin
``m0 -> (n0, ho0, wo0)`` once per workgroup, and a small i32 carry-propagation
decoding each lane's ``m = m0 + d -> (n, ho, wo)`` without any magic numbers.

CPU only: these are pure index-arithmetic helpers with no GPU-measured value.
The op-sequence assertions pin the lowering (sdiv/srem chains peeling the
trailing dim first, matching ``UnmergeMagicDynamic``'s decode order), and
compiling through COMGR is the strongest CPU-only proof the emitted i64/i32
mix is well-typed.
"""

from __future__ import annotations

import re

from rocke.core.ir import I32, I64, IRBuilder
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.helpers.compile import compile_kernel
from rocke.helpers.tensor_view import carry_decode_i32, tile_origin_decode_i64


def _build_origin_decode():
    b = IRBuilder("k")
    m0 = b.param("m0", I64)
    ho = b.param("Ho", I64)
    wo = b.param("Wo", I64)
    tile_origin_decode_i64(b, m0, [ho, wo])
    b.ret()
    return b.kernel


def test_origin_decode_peels_trailing_dim_first():
    kernel = _build_origin_decode()
    ll = lower_kernel_to_llvm(kernel, arch="gfx950")
    # Exact division via div + sub(quot*dim), matching UnmergeMagicDynamic's
    # remainder computation: wo0 peeled first (divides m0 itself by Wo), then
    # n0 is the final quotient (divides the Wo-quotient by Ho).
    assert len(re.findall(r"= sdiv i64 ", ll)) == 2
    assert re.search(r"sdiv i64 %m0, %Wo", ll)
    assert re.search(r"sdiv i64 %div\d+, %Ho", ll)


def test_origin_decode_rank_two_has_one_trailing_coord():
    b = IRBuilder("k")
    m0 = b.param("m0", I64)
    wo = b.param("Wo", I64)
    coords = tile_origin_decode_i64(b, m0, [wo])
    assert len(coords) == 2


def test_origin_decode_compiles_to_a_real_hsaco():
    kernel = _build_origin_decode()
    artifact = compile_kernel(kernel, arch="gfx950")
    assert artifact.hsaco_bytes > 0


def _build_carry_decode():
    b = IRBuilder("k")
    d = b.param("d", I32)
    n0 = b.param("n0", I32)
    ho0 = b.param("ho0", I32)
    wo0 = b.param("wo0", I32)
    ho_dim = b.param("Ho", I32)
    wo_dim = b.param("Wo", I32)
    carry_decode_i32(b, d, [n0, ho0, wo0], [ho_dim, wo_dim])
    b.ret()
    return b.kernel


def test_carry_decode_adds_delta_to_innermost_coord_first():
    kernel = _build_carry_decode()
    ll = lower_kernel_to_llvm(kernel, arch="gfx950")
    # wo = (wo0 + d) % Wo; carry = (wo0 + d) / Wo; ho = (ho0 + carry) % Ho; ...
    assert re.search(r"= add (?:nsw )?i32 %wo0, %d", ll)
    assert len(re.findall(r"= sdiv i32 ", ll)) == 2
    assert len(re.findall(r"= srem i32 ", ll)) == 2


def test_carry_decode_rejects_mismatched_dims():
    b = IRBuilder("k")
    d = b.param("d", I32)
    n0 = b.param("n0", I32)
    ho0 = b.param("ho0", I32)
    wo0 = b.param("wo0", I32)
    wo_dim = b.param("Wo", I32)
    try:
        carry_decode_i32(b, d, [n0, ho0, wo0], [wo_dim])
    except ValueError as exc:
        assert "!=" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched dims")


def test_carry_decode_compiles_to_a_real_hsaco():
    kernel = _build_carry_decode()
    artifact = compile_kernel(kernel, arch="gfx950")
    assert artifact.hsaco_bytes > 0
