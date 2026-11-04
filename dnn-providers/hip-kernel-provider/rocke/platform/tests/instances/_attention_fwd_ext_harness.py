# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test-only shell kernel built from the attention inner-body ``ext`` options.

It is a thin SDPA forward "shell": parameter decode, batch strides by 64-bit
pointer rebase, a runtime two-sided diagonal band, an optional additive bias,
and a log-sum-exp epilogue, all expressed through ``AttnFwdExt`` hooks over the
shared MFMA / WMMA inner body.  It exists to exercise the extension points; it
is not a production kernel and is not registered anywhere.
"""

from __future__ import annotations

from rocke.core.arch import ArchTarget
from rocke.core.ir import BF16, F16, F32, I32, I64, IRBuilder, PtrType
from rocke.helpers.attention_band import AttnRuntimeBounds, BandEmitter
from rocke.helpers.attention_fwd_ext import (
    LOG2E,
    AttnFwdExt,
    natural_lse_from_log2_stats,
)
from rocke.helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_K,
    mfma_attention_fwd_inner_body,
)

NEG_INF = float("-inf")
BLOCK_M = 16
_FILLS = {"neg_inf": NEG_INF, "big": -1e30, "mild": -1e20}

_SCALARS = (
    "seqlen_q seqlen_k q_b q_h q_t k_b k_h k_t v_b v_h v_t o_b o_h o_t "
    "lse_b lse_h stats_h bias_b bias_h bias_q left right diag"
).split()


def make_probe_kernel(
    name,
    *,
    arch,
    head_size,
    group=1,
    dtype="f16",
    has_bias=False,
    v_lds_stage=False,
    fold_batch=False,
    return_valid=True,
    fill="neg_inf",
    use_band_range=False,
    ext_mode="hooks",
    raw_stats=False,
    guard_k=True,
    lse_hook_factory=None,
):
    """Build the probe kernel; returns ``(kernel, params)``.

    ``params`` lists ``(name, struct_format)`` in kernel-argument order
    (``Q`` = 64-bit pointer, ``f`` = f32, ``i`` = i32) for host-side packing.

    * ``fold_batch``: the batch is folded into ``q_tile_base`` (physical row
      ``batch * q_fold + tile * 16``) while ``q_pos_base`` carries the in-sequence
      row, the way a packed-varlen caller does.  Q/O are then not rebased.
    * ``return_valid``: the epilogue hook returns the band row validity, so dead
      rows are decided from the bounds; otherwise only the body's max-sentinel
      validity applies (requires a true -inf or <= -1e30 fill).
    * ``use_band_range``: derive the per-CTA K tile range from the band.
    * ``lse_hook_factory``: ``f(b, lse_ptr, head=, lse_h=, band_valid=)`` returns
      an epilogue hook that replaces the inline LSE store (the raw statistics
      store, when enabled, still runs).  ``band_valid`` is the band row validity
      callable, or None when ``return_valid`` is off.
    * ``ext_mode``: ``"hooks"`` (default) passes the full extension; ``"omit"``
      leaves the ``ext`` keyword out, ``"none"`` passes ``None`` and ``"empty"``
      passes ``AttnFwdExt()``; the last three build the identical default body.  An
      ``AttnFwdExt`` instance is passed through unchanged.
    """
    elem = BF16 if dtype == "bf16" else F16
    esize = 2
    wave = ArchTarget.from_gfx(arch).wave_size
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = wave
    params = []

    def ptr(n, t, **kw):
        params.append((n, "Q"))
        return b.param(n, PtrType(t, "global"), align=16, **kw)

    Q = ptr("Q", elem, noalias=True, readonly=True)
    K = ptr("K", elem, noalias=True, readonly=True)
    V = ptr("V", elem, noalias=True, readonly=True)
    O = ptr("O", elem, noalias=True, writeonly=True)  # noqa: E741
    LSE = ptr("LSE", F32, noalias=True, writeonly=True)
    BIAS = ptr("BIAS", F32, noalias=True, readonly=True)
    STATS = ptr("STATS", F32, noalias=True, writeonly=True)

    def scalar(n, t, fmt):
        params.append((n, fmt))
        return b.param(n, t)

    scale_log2 = scalar("scale_log2", F32, "f")
    P = {n: scalar(n, I32, "i") for n in _SCALARS}
    q_fold = scalar("q_fold", I32, "i") if fold_batch else None

    tile = b.block_id_x()
    head = b.block_id_y()
    batch = b.block_id_z()
    kv_head = head if group == 1 else b.div(head, b.const_i32(group))

    batch64 = b.zext(batch, I64)

    def rebase(p, stride_elems, nbytes):
        off = b.mul(b.mul(batch64, b.zext(stride_elems, I64)), b.const_i64(nbytes))
        return b.global_ptr_add(p, off)

    Kb = rebase(K, P["k_b"], esize)
    Vb = rebase(V, P["v_b"], esize)
    Qb = Q if fold_batch else rebase(Q, P["q_b"], esize)
    Ob = O if fold_batch else rebase(O, P["o_b"], esize)
    LSEb = rebase(LSE, P["lse_b"], 4)
    STATSb = rebase(STATS, P["lse_b"], 8) if raw_stats else None
    BIASb = rebase(BIAS, P["bias_b"], 4) if has_bias else None

    sq, sk = P["seqlen_q"], P["seqlen_k"]
    band = BandEmitter(b, AttnRuntimeBounds(sq, sk, P["diag"], P["left"], P["right"]))
    fill_c = b.const_f32(_FILLS[fill])
    log2e_c = b.const_f32(LOG2E)

    def score_hook(bb, s, c):
        if has_bias:
            ok = bb.land(c.q_valid, c.k_valid)
            idx = bb.add(
                bb.add(bb.mul(head, P["bias_h"]), bb.mul(c.q_row, P["bias_q"])),
                c.k_col,
            )
            bias = bb.masked_global_load(
                BIASb, idx, ok, bb.const_f32(0.0), F32, align=4
            )
            s = bb.fadd(s, bb.fmul(bias, log2e_c))
        rb = band.row_band(c.q_row)
        keep = band.cell_keep(rb, c.k_col, band.col_in(c.k_col))
        return bb.select(keep, s, fill_c)

    def stats_store(bb, e):
        idx = bb.add(bb.mul(head, P["stats_h"]), e.q_row)
        s2 = bb.mul(idx, bb.const_i32(2))
        with bb.scf_if(bb.land(e.is_row_leader, e.in_range)):
            bb.global_store(STATSb, s2, e.m_log2, align=4)
            bb.global_store(STATSb, bb.add(s2, bb.const_i32(1)), e.l, align=4)

    def band_valid(bb, e):
        return band.row_band(e.q_row).valid

    lse_hook = None
    if lse_hook_factory is not None:
        lse_hook = lse_hook_factory(
            b,
            LSEb,
            head=head,
            lse_h=P["lse_h"],
            band_valid=band_valid if return_valid else None,
        )

    def epilogue_hook(bb, e):
        if lse_hook is not None:
            ret = lse_hook(bb, e)
            if raw_stats:
                stats_store(bb, e)
            return ret
        rb = band.row_band(e.q_row)
        valid = bb.land(e.row_valid, rb.valid) if return_valid else e.row_valid
        nat = natural_lse_from_log2_stats(bb, e.m_log2, e.l)
        val = bb.select(valid, nat, bb.const_f32(NEG_INF))
        idx = bb.add(bb.mul(head, P["lse_h"]), e.q_row)
        with bb.scf_if(bb.land(e.is_row_leader, e.in_range)):
            bb.global_store(LSEb, idx, val, align=4)
        if raw_stats:
            stats_store(bb, e)
        return rb.valid if return_valid else None

    ext = None
    if ext_mode == "empty":
        ext = AttnFwdExt()
    elif isinstance(ext_mode, AttnFwdExt):
        ext = ext_mode
    elif ext_mode == "hooks":
        ext = AttnFwdExt(
            score_hook=score_hook,
            seqlen_q=sq,
            guard_seqlen_k=guard_k,
            epilogue_hook=epilogue_hook,
        )

    tile_row = b.mul(tile, b.const_i32(BLOCK_M))
    kw = {}
    if fold_batch:
        q_tile_base = b.add(b.mul(batch, q_fold), tile_row)
        kw["q_pos_base"] = tile_row
    else:
        q_tile_base = tile_row
    if use_band_range:
        start, stop = band.k_tile_range(tile_row, BLOCK_M, MFMA_ATTN_BLOCK_K)
        kw["k_tile_start"], kw["k_tile_stop"] = start, stop

    mfma_attention_fwd_inner_body(
        b,
        Q=Qb,
        K=Kb,
        V=Vb,
        O=Ob,
        head_size=head_size,
        seqlen_k=sk,
        q_tile_base=q_tile_base,
        head_idx=head,
        kv_head_idx=kv_head,
        stride_q_token=P["q_t"],
        stride_q_head=P["q_h"],
        stride_k_token=P["k_t"],
        stride_k_head=P["k_h"],
        stride_v_token=P["v_t"],
        stride_v_head=P["v_h"],
        stride_o_token=P["o_t"],
        stride_o_head=P["o_h"],
        scale_log2=scale_log2,
        dtype=dtype,
        mask_mode="none",
        wmma_v_lds_stage=v_lds_stage,
        arch=arch,
        **({} if ext_mode == "omit" else {"ext": ext}),
        **kw,
    )
    b.ret()
    return b.kernel, params
