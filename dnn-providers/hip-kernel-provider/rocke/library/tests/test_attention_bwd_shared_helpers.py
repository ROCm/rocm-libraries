# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shared helpers of the attention backward family glue (CPU only).

* ``as_i32`` / ``as_i64``: an int becomes one constant, a value is passed
  through (``as_i64`` widens an i32 by ``zext`` or ``sext``);
* ``deal_chunks`` / ``chunk_coords``: every chunk of a tile is dealt to
  exactly one (thread, round), the partial rounds are flagged, and the
  coordinates are the row-major ``(row, column)`` of the chunk, evaluated on
  the typed integer builder;
* ``HostInt``: the band formulas and the DMA lane map evaluate on it
  (``land`` / ``lor`` are the logical ops of compare results and the bitwise
  ops of ints);
* ``I32_FAR`` is the one "past every row / key / bound" constant;
* ``attn_bwd_stage_vec`` is the one staging-width rule of the general and
  the unified request;
* ``ArchAttentionDenseBwdSpec``: the per-arch dense specs are the class bound
  to their target (``ARCH``, ``BODY_VARIANTS``, descriptor key), and the
  per-arch builders refuse another class.
"""

from __future__ import annotations

import pytest
from kernels.common._attention_bwd_addr import (
    as_i32,
    as_i64,
    chunk_coords,
    deal_chunks,
)
from kernels.common._attention_bwd_band import HostInt, static_kv_tile_plan
from kernels.common._attention_bwd_lds import DmaSlabLayout
from kernels.common._attention_bwd_ring import dma_lane_source
from kernels.common.attention_bwd_plan import AttnBwdRequest, attn_bwd_stage_vec
from kernels.common.attention_bwd_unified_plan import AttnBwdUnifiedRequest
from kernels.common.attention_dense_bwd_spec import (
    DENSE_BWD_BODY_VARIANTS,
    ArchAttentionDenseBwdSpec,
)
from rocke.core.ir import I32, I64, IRBuilder
from rocke.helpers.attention_band import I32_FAR

from ._attention_bwd_int_eval import IntEvalBuilder, i32


def _n_ops(b: IRBuilder) -> int:
    return len(b.kernel.body.ops)


def test_as_i32_and_as_i64():
    b = IRBuilder("k")
    v = b.param("v", I32)
    n0 = _n_ops(b)
    assert as_i32(b, v) is v and _n_ops(b) == n0
    c = as_i32(b, 7)
    assert c.type == I32 and _n_ops(b) == n0 + 1
    w = as_i64(b, v)
    assert w.type == I64 and b.kernel.body.ops[-1].name.endswith("zext")
    s = as_i64(b, v, signed=True)
    assert s.type == I64 and b.kernel.body.ops[-1].name.endswith("sext")
    assert as_i64(b, w) is w
    assert as_i64(b, 5).type == I64


@pytest.mark.parametrize(
    "rows,per_row,vec", [(16, 4, 8), (64, 16, 8), (5, 1, 8), (7, 3, 2)]
)
@pytest.mark.parametrize("threads", [1, 7, 64, 256])
def test_deal_chunks_covers_every_chunk_once(rows, per_row, vec, threads):
    total = rows * per_row
    b = IntEvalBuilder()
    seen = {}
    for t in range(threads):
        rounds = list(deal_chunks(b, i32(b, t), total, threads))
        assert len(rounds) == -(-total // threads)
        for k, (chunk, partial) in enumerate(rounds):
            assert chunk.v == t + k * threads
            assert partial == ((k + 1) * threads > total)
            if chunk.v >= total:
                assert partial  # only a partial round reaches past the tile
                continue
            r, c = chunk_coords(b, chunk, per_row, vec)
            assert (r.v, c.v) == (chunk.v // per_row, (chunk.v % per_row) * vec)
            seen[chunk.v] = seen.get(chunk.v, 0) + 1
    assert seen == {x: 1 for x in range(total)}


def test_chunk_coords_one_chunk_per_row_has_no_division():
    b = IRBuilder("k")
    chunk = b.param("c", I32)
    n0 = _n_ops(b)
    r, c = chunk_coords(b, chunk, 1, 8)
    assert r is chunk and _n_ops(b) == n0 + 1  # the column constant only


def test_host_int_logic_and_bits():
    h = HostInt()
    assert h.land(True, False) is False and h.lor(False, True) is True
    assert h.land(6, 3) == 2 and h.lor(4, 1) == 5 and h.xor(5, 1) == 4
    assert h.lshr(17, 3) == 2 and h.div(-7, 2) == -3  # sdiv truncates
    plan = static_kv_tile_plan(s_q=100, s_kv=80, k0=64, k_m0=16, k_n0=16, right=0)
    assert plan.n_qt > 0
    for cols in (128, 32):  # a full 128-byte line per row, and folded rows
        layout = DmaSlabLayout(rows=64, cols=cols, elem_bytes=2)
        for piece in range(layout.rows * cols * 2 // (64 * 16)):
            for lane in range(64):
                got = dma_lane_source(h, layout, piece, lane)
                assert got == layout.dma_source(64 * piece + lane), (cols, piece, lane)


def test_far_is_one_constant():
    import kernels.common._attention_bwd_addr as A
    import kernels.common._attention_bwd_band as B
    import kernels.common.attention_bwd_plan as P
    import kernels.common.attention_bwd_unified_plan as U

    assert I32_FAR == 1 << 30
    for mod in (A, B, P, U):
        assert mod.I32_FAR is I32_FAR
        for old in ("_FAR", "_ROWS_CAP", "_BAND_MAX"):
            assert not hasattr(mod, old), (mod.__name__, old)


def test_one_stage_vec_rule_for_both_requests():
    general = AttnBwdRequest(b=2, h_q=8, h_k=8, h_v=8, s_q=64, s_kv=64, d_qk=64, d_v=64)
    unified = AttnBwdUnifiedRequest(
        num_seqs=2, h_q=8, h_kv=2, d_qk=64, d_v=64, total_q=50, max_seqlen_q=30,
        max_seqlen_kv=100, page=16, num_blocks=20, max_blocks_per_seq=7,
    )  # fmt: skip
    for req in (general, unified):
        assert attn_bwd_stage_vec(req) == 8
        assert all(s % 8 == 0 for s in req.used_strides())
    import dataclasses

    assert attn_bwd_stage_vec(dataclasses.replace(general, tensor_alignment=8)) == 1
    assert attn_bwd_stage_vec(dataclasses.replace(unified, tensor_alignment=8)) == 1


def test_arch_dense_spec_binding():
    from kernels.gfx942.attention_dense_bwd import Gfx942AttentionDenseBwdSpec as S942
    from kernels.gfx942.attention_dense_bwd import build_attention_dense_bwd as b942
    from kernels.gfx950.attention_dense_bwd import Gfx950AttentionDenseBwdSpec as S950

    for cls, arch in ((S942, "gfx942"), (S950, "gfx950")):
        assert issubclass(cls, ArchAttentionDenseBwdSpec)
        assert cls.ARCH == arch and cls.BODY_VARIANTS == DENSE_BWD_BODY_VARIANTS[arch]
        assert cls.__module__ == f"kernels.{arch}.attention_dense_bwd"
    shape = dict(seqlen_q=128, seqlen_kv=128, num_query_heads=8, num_kv_heads=8)
    with pytest.raises(ValueError, match="body_variant"):
        S942(head_size=64, body_variant="trload", **shape)
    with pytest.raises(TypeError, match="Gfx942AttentionDenseBwdSpec"):
        b942(S950(head_size=64, **shape), arch="gfx942")
    with pytest.raises(ValueError, match="arch"):

        class _Bad(ArchAttentionDenseBwdSpec, arch="gfx1151"):
            pass
