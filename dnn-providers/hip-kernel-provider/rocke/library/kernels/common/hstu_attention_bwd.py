# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HSTU attention backward baseline kernels.

This mirrors the FlyDSL split-kernel contract:

* ``which="dv"`` owns one KV row and produces dV.
* ``which="dk"`` owns one KV row and produces dK.
* ``which="dq"`` owns one query row and produces dQ.

The implementation is deliberately scalar per output row/dimension so it is easy
to validate and serves as the Rocke ABI/math baseline before the gfx950 MFMA
tiling is lifted in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple

from rocke.core.ir import I32, KernelDef, PtrType, Value
from rocke.helpers.atoms import MfmaAtom
from rocke.helpers.hstu_attention import hstu_mask_keep, hstu_silu_and_grad
from rocke.helpers.io import io_ir_type, load_scalar_as_f32, store_scalar_from_f32
from rocke.helpers.spec import SignatureBuilder, kernel_name_join


__all__ = [
    "HstuBwdSpec",
    "HstuBwdWhich",
    "build_hstu_attention_bwd",
    "hstu_attention_bwd_block_size",
    "hstu_attention_bwd_grid",
    "hstu_attention_bwd_signature",
    "is_valid_spec",
]


HstuBwdWhich = Literal["dv", "dk", "dq"]


@dataclass(frozen=True)
class HstuBwdSpec:
    """One HSTU backward kernel configuration."""

    num_heads: int
    head_dim: int
    hidden_dim: int
    batch: int
    max_seq_len: int
    dtype: str = "f16"
    causal: bool = True
    max_attn_len: int = 0
    contextual_seq_len: int = 0
    has_targets: bool = False
    has_perm: bool = False
    alpha: float = 1.0
    which: HstuBwdWhich = "dv"
    use_mfma_body: bool = False
    block_size: int = 64
    # Tiled multi-wave config (P3): when ``block_m > 0`` the tiled builder is
    # used -- a CTA owns ``block_m`` rows of the output-row dim and the full
    # ``out_dim``, split across ``num_waves`` wave64 warps, streaming the
    # reduction dim in ``block_n`` tiles through swizzled LDS. ``block_m == 0``
    # keeps the simple one-16x16-tile-per-CTA MFMA path.
    block_m: int = 0
    block_n: int = 16
    num_waves: int = 4
    waves_per_eu: int = 0
    name: str = "rocke_hstu_attention_bwd"

    @property
    def tiled(self) -> bool:
        return self.use_mfma_body and self.block_m > 0

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.which,
            self.dtype,
            f"H{self.num_heads}",
            f"HD{self.head_dim}",
            f"VD{self.hidden_dim}",
            f"B{self.batch}",
            f"N{self.max_seq_len}",
            *(
                [f"BM{self.block_m}", f"BN{self.block_n}", f"W{self.num_waves}"]
                if self.tiled
                else []
            ),
            flags={
                "window": self.max_attn_len > 0,
                "contextual": self.contextual_seq_len > 0,
                "targets": self.has_targets,
                "perm": self.has_perm,
                "mfma": self.use_mfma_body,
            },
        )


def is_valid_spec(spec: HstuBwdSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    if not (arch.startswith("gfx942") or arch.startswith("gfx950")):
        return False, f"unsupported arch {arch!r}; expected gfx942/gfx950"
    if spec.dtype not in ("f16", "fp16", "bf16"):
        return False, f"dtype {spec.dtype!r} must be f16/fp16/bf16"
    if spec.which not in ("dv", "dk", "dq"):
        return False, f"which {spec.which!r} must be one of dv/dk/dq"
    if not spec.causal:
        return False, "HSTU backward baseline supports causal attention only"
    if spec.num_heads <= 0 or spec.batch <= 0 or spec.max_seq_len <= 0:
        return False, "num_heads, batch, and max_seq_len must be positive"
    if spec.head_dim <= 0 or spec.head_dim % 16:
        return (
            False,
            f"head_dim must be positive and divisible by 16, got {spec.head_dim}",
        )
    if spec.hidden_dim <= 0 or spec.hidden_dim % 16:
        return (
            False,
            f"hidden_dim must be positive and divisible by 16, got {spec.hidden_dim}",
        )
    if spec.max_attn_len < 0 or spec.contextual_seq_len < 0:
        return False, "max_attn_len and contextual_seq_len must be non-negative"
    if not math.isfinite(spec.alpha):
        return False, f"alpha must be finite, got {spec.alpha}"
    out_dim = spec.hidden_dim if spec.which == "dv" else spec.head_dim
    if spec.tiled:
        return _is_valid_tiled(spec)
    if spec.block_size not in (64, 128, 256):
        return False, f"block_size {spec.block_size} must be one of 64/128/256"
    if spec.use_mfma_body and spec.block_size != 64:
        return False, "HSTU MFMA body currently requires one wave64 CTA"
    if out_dim > spec.block_size * 8:
        return False, (
            f"{spec.which} output dim {out_dim} would need more than 8 elems/thread "
            f"with block_size={spec.block_size}"
        )
    return True, "ok"


def _is_valid_tiled(spec: HstuBwdSpec) -> Tuple[bool, str]:
    bm, bn, nw = spec.block_m, spec.block_n, spec.num_waves
    if nw <= 0 or bm <= 0 or bn <= 0:
        return False, "tiled config requires block_m, block_n, num_waves > 0"
    if bm % (nw * 16):
        return False, f"block_m {bm} must be a multiple of num_waves*16 ({nw * 16})"
    if bn % 16:
        return False, f"block_n {bn} must be a multiple of 16 (got {bn})"
    block_threads = nw * 64
    head_dim_k = ((spec.head_dim + 63) // 64) * 64
    for name, dim in (("head", head_dim_k), ("hidden", spec.hidden_dim)):
        tile = bn * dim
        # need a power-of-two vector width <= 8 that evenly distributes the
        # staging DMA across the block's threads.
        if not any(tile % (block_threads * vec) == 0 for vec in (8, 4, 2, 1)):
            return False, f"streamed {name} tile {tile} does not divide the DMA pass"
    lds_bytes = bn * head_dim_k * 2 + bn * spec.hidden_dim * 2
    if lds_bytes > 65536:
        return False, f"tiled LDS {lds_bytes} B exceeds 64 KB budget"
    return True, "ok"


def _declare_params(b, spec: HstuBwdSpec) -> dict[str, Value]:
    ty = io_ir_type(spec.dtype)
    return {
        "q": b.param("q", PtrType(ty, "global"), noalias=True, readonly=True, align=16),
        "k": b.param("k", PtrType(ty, "global"), noalias=True, readonly=True, align=16),
        "v": b.param("v", PtrType(ty, "global"), noalias=True, readonly=True, align=16),
        "do": b.param(
            "do", PtrType(ty, "global"), noalias=True, readonly=True, align=16
        ),
        "seq_offsets": b.param(
            "seq_offsets",
            PtrType(I32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        ),
        "num_targets": b.param(
            "num_targets",
            PtrType(I32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        ),
        "perm": b.param(
            "perm", PtrType(I32, "global"), noalias=True, readonly=True, align=4
        ),
        "out": b.param(
            "out", PtrType(ty, "global"), noalias=True, writeonly=True, align=16
        ),
    }


def _token_base(b, token: Value, head: Value, dim: int, num_heads: int) -> Value:
    stride_token = b.const_i32(num_heads * dim)
    return b.add(b.mul(token, stride_token), b.mul(head, b.const_i32(dim)))


def _batch_idx(b, params: dict[str, Value], spec: HstuBwdSpec) -> Value:
    batch_slot = b.block_id_z()
    if spec.has_perm:
        return b.global_load_i32(params["perm"], batch_slot)
    return batch_slot


def _seq_bounds(b, params: dict[str, Value], spec: HstuBwdSpec) -> tuple[Value, Value]:
    batch_idx = _batch_idx(b, params, spec)
    seq_start = b.global_load_i32(params["seq_offsets"], batch_idx)
    seq_end = b.global_load_i32(params["seq_offsets"], b.add(batch_idx, b.const_i32(1)))
    return seq_start, seq_end


def _mask_keep(b, q_local: Value, k_local: Value, max_id: Value, spec: HstuBwdSpec):
    return hstu_mask_keep(
        b,
        q_local=q_local,
        k_local=k_local,
        max_id=max_id,
        max_attn_len=spec.max_attn_len,
        contextual_seq_len=spec.contextual_seq_len,
        has_targets=spec.has_targets,
    )


def _qk_score(
    b, params: dict[str, Value], q_tok: Value, k_tok: Value, head: Value, spec
):
    q_base = _token_base(b, q_tok, head, spec.head_dim, spec.num_heads)
    k_base = _token_base(b, k_tok, head, spec.head_dim, spec.num_heads)
    score = b.const_f32(0.0)
    for d in range(spec.head_dim):
        qv = load_scalar_as_f32(
            b, params["q"], b.add(q_base, b.const_i32(d)), dtype=spec.dtype
        )
        kv = load_scalar_as_f32(
            b, params["k"], b.add(k_base, b.const_i32(d)), dtype=spec.dtype
        )
        score = b.fma(qv, kv, score)
    return score


def _silu_and_grad(b, score: Value, spec: HstuBwdSpec):
    return hstu_silu_and_grad(b, score, spec.alpha)


def _do_v_dot(
    b, params: dict[str, Value], q_tok: Value, k_tok: Value, head: Value, spec
):
    do_base = _token_base(b, q_tok, head, spec.hidden_dim, spec.num_heads)
    v_base = _token_base(b, k_tok, head, spec.hidden_dim, spec.num_heads)
    acc = b.const_f32(0.0)
    for d in range(spec.hidden_dim):
        dov = load_scalar_as_f32(
            b, params["do"], b.add(do_base, b.const_i32(d)), dtype=spec.dtype
        )
        vv = load_scalar_as_f32(
            b, params["v"], b.add(v_base, b.const_i32(d)), dtype=spec.dtype
        )
        acc = b.fma(dov, vv, acc)
    return acc


def _emit_output_slice(b, params: dict[str, Value], token: Value, head: Value, spec):
    out_dim = spec.hidden_dim if spec.which == "dv" else spec.head_dim
    ept = (out_dim + spec.block_size - 1) // spec.block_size
    tid = b.thread_id_x()
    out_base = _token_base(b, token, head, out_dim, spec.num_heads)
    for e in range(ept):
        d = b.add(tid, b.const_i32(e * spec.block_size))
        with b.scf_if(b.cmp_lt(d, b.const_i32(out_dim))):
            init = b.const_f32(0.0)
            if spec.which in ("dv", "dk"):
                lower, upper = _seq_bounds(b, params, spec)
                loop = b.scf_for_iter(
                    lower,
                    upper,
                    b.const_i32(1),
                    [("acc", init)],
                    iv_name="q_tok",
                )
                with loop as (q_tok, (acc,)):
                    q_local = b.sub(q_tok, lower)
                    k_local = b.sub(token, lower)
                    keep = _mask_keep(
                        b, q_local, k_local, _max_id(b, params, spec), spec
                    )
                    score = _qk_score(b, params, q_tok, token, head, spec)
                    silu, grad = _silu_and_grad(b, score, spec)
                    if spec.which == "dv":
                        do_base = _token_base(
                            b, q_tok, head, spec.hidden_dim, spec.num_heads
                        )
                        do_val = load_scalar_as_f32(
                            b, params["do"], b.add(do_base, d), dtype=spec.dtype
                        )
                        contrib = b.fmul(
                            b.fmul(silu, b.const_f32(1.0 / spec.max_seq_len)),
                            do_val,
                        )
                    else:
                        da = _do_v_dot(b, params, q_tok, token, head, spec)
                        q_base = _token_base(
                            b, q_tok, head, spec.head_dim, spec.num_heads
                        )
                        q_val = load_scalar_as_f32(
                            b, params["q"], b.add(q_base, d), dtype=spec.dtype
                        )
                        ds = b.fmul(
                            b.fmul(b.const_f32(1.0 / spec.max_seq_len), grad),
                            da,
                        )
                        contrib = b.fmul(b.fmul(ds, b.const_f32(spec.alpha)), q_val)
                    contrib = b.select(keep, contrib, b.const_f32(0.0))
                    b.scf_yield(b.fadd(acc, contrib))
                val = loop.results[0]
            else:
                lower, upper = _seq_bounds(b, params, spec)
                loop = b.scf_for_iter(
                    lower,
                    upper,
                    b.const_i32(1),
                    [("acc", init)],
                    iv_name="k_tok",
                )
                with loop as (k_tok, (acc,)):
                    q_local = b.sub(token, lower)
                    k_local = b.sub(k_tok, lower)
                    keep = _mask_keep(
                        b, q_local, k_local, _max_id(b, params, spec), spec
                    )
                    score = _qk_score(b, params, token, k_tok, head, spec)
                    _, grad = _silu_and_grad(b, score, spec)
                    da = _do_v_dot(b, params, token, k_tok, head, spec)
                    k_base = _token_base(b, k_tok, head, spec.head_dim, spec.num_heads)
                    k_val = load_scalar_as_f32(
                        b, params["k"], b.add(k_base, d), dtype=spec.dtype
                    )
                    ds = b.fmul(
                        b.fmul(b.const_f32(1.0 / spec.max_seq_len), grad),
                        da,
                    )
                    contrib = b.select(
                        keep,
                        b.fmul(b.fmul(ds, b.const_f32(spec.alpha)), k_val),
                        b.const_f32(0.0),
                    )
                    b.scf_yield(b.fadd(acc, contrib))
                val = loop.results[0]
            store_scalar_from_f32(
                b, params["out"], b.add(out_base, d), val, dtype=spec.dtype
            )


def _max_id(b, params: dict[str, Value], spec: HstuBwdSpec) -> Value:
    batch_idx = _batch_idx(b, params, spec)
    seq_start, seq_end = _seq_bounds(b, params, spec)
    seq_len = b.sub(seq_end, seq_start)
    max_id = seq_len
    if spec.contextual_seq_len > 0:
        max_id = b.add(
            b.sub(seq_len, b.const_i32(spec.contextual_seq_len)), b.const_i32(1)
        )
    if spec.has_targets:
        num_target = b.global_load_i32(params["num_targets"], batch_idx)
        shifted = b.sub(max_id, num_target)
        max_id = b.select(b.cmp_gt(num_target, b.const_i32(0)), shifted, max_id)
    return max_id


def _mfma_atom_for_spec(spec: HstuBwdSpec) -> MfmaAtom:
    if spec.dtype in ("f16", "fp16"):
        return MfmaAtom.f16_16x16x16()
    if spec.dtype == "bf16":
        return MfmaAtom.bf16_16x16x16()
    raise ValueError(f"unsupported HSTU MFMA dtype {spec.dtype!r}")


def _native_zero(b, spec: HstuBwdSpec) -> Value:
    return b.cast_f32_to(b.const_f32(0.0), io_ir_type(spec.dtype))


def _build_hstu_attention_bwd_mfma(
    spec: HstuBwdSpec, arch: str = "gfx950"
) -> KernelDef:
    """MFMA-tiled HSTU backward (dv/dk/dq), zero scalar dot products.

    Unified design mirroring the FlyDSL split kernels and rocke's own MFMA
    forward body (:func:`rocke.helpers.mfma_attention.mfma_attention_fwd_inner_body`):

    * Each CTA owns one ``16``-row tile of the output row dim (``kv`` for
      dv/dk, ``q`` for dq) and one ``16``-col output tile; it streams the
      reduction dim (``q`` for dv/dk, ``kv`` for dq) in ``16``-token tiles.
    * **GEMM1 (feature-dim contraction, ``A·Bᵀ`` form)** computes the score
      ``S[own, stream] = Σ_d A[own,d]·B[stream,d]`` as an MFMA chain over the
      head-dim (both operands loaded ``[row, d-slice]`` -- identical to the
      forward QKᵀ). dK/dQ additionally run a second such GEMM for
      ``dA = dO·Vᵀ`` over the hidden dim.
    * The score fragment is gated (HSTU mask + SiLU / SiLU') in the C-layout,
      cast to the native dtype, and **re-laid-out through a 16×16 LDS tile**
      into the A-operand layout (the same transpose trick the forward body
      uses for P).
    * **GEMM2 (token-dim contraction, ``A·B`` form)** accumulates the output
      ``out[own, d] += Σ_stream gate[own,stream]·B[stream,d]`` (B = dO for dv,
      Q for dk, K for dq) into a per-lane f32 accumulator.
    * Epilogue scales (``1/N`` for dv, ``alpha`` for dk/dq) and stores.

    No per-lane scalar QK / dO·V recompute remains: the only per-element loads
    are the streamed GEMM2 B-operand gather (inherent to the matmul, matching
    the forward PV V-load).
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid hstu_attention_bwd spec: {why}")

    from rocke.core.ir import IRBuilder

    atom = _mfma_atom_for_spec(spec)
    dtype_ir = io_ir_type(spec.dtype)
    apl = atom.a_per_lane  # 4 for the 16x16x16 atom
    own_is_kv = spec.which in ("dv", "dk")
    out_dim = spec.hidden_dim if spec.which == "dv" else spec.head_dim
    need_da = spec.which in ("dk", "dq")
    inv_n = 1.0 / spec.max_seq_len
    n_feat_s = spec.head_dim // 16
    n_feat_da = spec.hidden_dim // 16

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    params = _declare_params(b, spec)

    seq_start, seq_end = _seq_bounds(b, params, spec)
    seq_len = b.sub(seq_end, seq_start)
    max_id = _max_id(b, params, spec)
    head = b.block_id_y()

    d_tiles = out_dim // atom.n
    x = b.block_id_x()
    d_tile_base = b.mul(b.mod(x, b.const_i32(d_tiles)), b.const_i32(atom.n))
    own_base = b.mul(b.div(x, b.const_i32(d_tiles)), b.const_i32(atom.m))

    lane = b.thread_id_x()
    c16 = b.const_i32(16)
    m_in = b.mod(lane, c16)  # row within the 16-tile (A/B row for GEMM1)
    m_blk = b.div(lane, c16)  # 0..3 -- which 4-slot along k / c
    k_lane_start = b.mul(m_blk, b.const_i32(apl))

    def feat_vec(ptr, tok, dim, fa):
        """Style-1 operand: this lane's contiguous ``apl`` d-values of one atom."""
        base = _token_base(b, tok, head, dim, spec.num_heads)
        d_start = b.add(b.const_i32(fa * 16), k_lane_start)
        return b.global_load_vN(ptr, b.add(base, d_start), dtype_ir, apl, align=apl * 2)

    # Owned A-row token (loop-invariant): clamp OOB to token 0 (gated later).
    own_row_local = b.add(own_base, m_in)
    own_row_in = b.cmp_lt(own_row_local, seq_len)
    own_tok = b.add(seq_start, b.select(own_row_in, own_row_local, b.const_i32(0)))

    # Pre-load the owned (loop-invariant) style-1 operands.
    # S GEMM owned operand: dv/dk -> K, dq -> Q  (contract head_dim).
    own_s_ptr = params["q"] if spec.which == "dq" else params["k"]
    own_s_vecs = [
        feat_vec(own_s_ptr, own_tok, spec.head_dim, fa) for fa in range(n_feat_s)
    ]
    own_da_vecs = None
    if need_da:
        # dA GEMM owned operand: dk -> V, dq -> dO  (contract hidden_dim).
        own_da_ptr = params["v"] if spec.which == "dk" else params["do"]
        own_da_vecs = [
            feat_vec(own_da_ptr, own_tok, spec.hidden_dim, fa)
            for fa in range(n_feat_da)
        ]

    # 16x16 LDS tile re-lays out the gate C-fragment into the GEMM2 A-operand.
    gate_lds = b.smem_alloc(dtype_ir, [16, 16], name_hint="hstu_gate")

    # Streamed style-1 B-operand pointer/dim (S GEMM): dv/dk -> Q, dq -> K.
    stream_s_ptr = params["k"] if spec.which == "dq" else params["q"]
    # dA GEMM streamed B: dk -> dO, dq -> V.
    stream_da_ptr = params["do"] if spec.which == "dk" else params["v"]
    # GEMM2 B-operand (output-dim): dv -> dO, dk -> Q, dq -> K.
    out_b_ptr = {"dv": params["do"], "dk": params["q"], "dq": params["k"]}[spec.which]

    # Causal streamed-range limiting (skips fully-masked tiles). Only safe for
    # pure causal (no window / contextual opener / targets, which all widen the
    # attended region); the variants keep the full [0, seq_len) sweep and rely
    # on the per-cell mask.
    pure_causal = (
        spec.max_attn_len == 0 and spec.contextual_seq_len == 0 and not spec.has_targets
    )
    if pure_causal and own_is_kv:
        # dv/dk own a kv tile and reduce over q; causal keeps q >= kv, so every
        # streamed q-tile below the owned kv tile is fully masked.
        loop_lo, loop_hi = own_base, seq_len
    elif pure_causal:
        # dq owns a q tile and reduces over kv; causal keeps kv <= q, so streamed
        # kv-tiles above the owned q tile are fully masked.
        loop_lo = b.const_i32(0)
        own_end = b.add(own_base, b.const_i32(16))
        loop_hi = b.select(b.cmp_lt(own_end, seq_len), own_end, seq_len)
    else:
        loop_lo, loop_hi = b.const_i32(0), seq_len

    acc0 = atom.zero_acc(b)
    with b.scf_if(b.cmp_lt(own_base, seq_len)):
        loop = b.scf_for_iter(
            loop_lo,
            loop_hi,
            b.const_i32(16),
            [("acc", acc0)],
            iv_name="stream_base",
        )
        with loop as (stream_base, (acc,)):
            stream_row_local = b.add(stream_base, m_in)
            stream_in = b.cmp_lt(stream_row_local, seq_len)
            stream_tok = b.add(
                seq_start, b.select(stream_in, stream_row_local, b.const_i32(0))
            )

            # GEMM1: S[own, stream] = sum_d own[own,d] * stream[stream,d].
            score = atom.zero_acc(b)
            for fa in range(n_feat_s):
                bvec = feat_vec(stream_s_ptr, stream_tok, spec.head_dim, fa)
                score = atom.emit(b, own_s_vecs[fa], bvec, score)

            # dA[own, stream] = sum_hd own_da[own,hd] * stream_da[stream,hd].
            da = None
            if need_da:
                da = atom.zero_acc(b)
                for fa in range(n_feat_da):
                    bvec = feat_vec(stream_da_ptr, stream_tok, spec.hidden_dim, fa)
                    da = atom.emit(b, own_da_vecs[fa], bvec, da)

            # Gate the C-fragment (lane holds own = m_blk*4+r, stream = m_in),
            # then publish into the LDS re-layout tile.
            for r in range(atom.c_per_lane):
                own_pos = b.add(
                    own_base, b.add(b.mul(m_blk, b.const_i32(4)), b.const_i32(r))
                )
                own_pos_in = b.cmp_lt(own_pos, seq_len)
                if own_is_kv:
                    q_pos, k_pos = stream_row_local, own_pos
                else:
                    q_pos, k_pos = own_pos, stream_row_local
                keep = _mask_keep(b, q_pos, k_pos, max_id, spec)
                active = b.land(b.land(keep, own_pos_in), stream_in)
                s = b.vec_extract(score, r)
                if spec.which == "dv":
                    silu, _ = _silu_and_grad(b, s, spec)
                    val = b.select(active, silu, b.const_f32(0.0))
                else:
                    _, grad = _silu_and_grad(b, s, spec)
                    da_r = b.vec_extract(da, r)
                    gated = b.fmul(b.fmul(b.const_f32(inv_n), grad), da_r)
                    val = b.select(active, gated, b.const_f32(0.0))
                p_row = b.add(b.mul(m_blk, b.const_i32(4)), b.const_i32(r))
                b.smem_store_vN(
                    gate_lds, [p_row, m_in], b.cast_f32_to(val, dtype_ir), 1
                )
            b.sync()

            # GEMM2 A-operand: read the re-laid-out gate as A[own=m_in, k=stream].
            a_vec = b.zero_vec(dtype_ir, apl)
            for j in range(apl):
                col = b.add(k_lane_start, b.const_i32(j))
                pv = b.vec_extract(
                    b.smem_load_vN(gate_lds, m_in, col, dtype=dtype_ir, n=1), 0
                )
                a_vec = b.vec_insert(a_vec, pv, j)

            # GEMM2 B-operand: out_b[stream, out_col] (4 strided stream rows).
            out_col = b.add(d_tile_base, m_in)
            b_vec = b.zero_vec(dtype_ir, apl)
            for j in range(apl):
                srow = b.add(stream_base, b.add(k_lane_start, b.const_i32(j)))
                sin = b.cmp_lt(srow, seq_len)
                stok = b.add(seq_start, b.select(sin, srow, b.const_i32(0)))
                addr = b.add(
                    _token_base(b, stok, head, out_dim, spec.num_heads), out_col
                )
                raw = b.global_load(out_b_ptr, addr, dtype_ir, align=2)
                b_vec = b.vec_insert(
                    b_vec, b.select(sin, raw, _native_zero(b, spec)), j
                )

            acc = atom.emit(b, a_vec, b_vec, acc)
            b.sync()  # close the WAR hazard on gate_lds before the next store
            b.scf_yield(acc)

        # Epilogue: out[own, out_col] = acc * scale (own = m_blk*4+r, out = m_in).
        result = loop.results[0]
        c_scale = b.const_f32(inv_n if spec.which == "dv" else spec.alpha)
        out_col = b.add(d_tile_base, m_in)
        for r in range(atom.c_per_lane):
            own_pos = b.add(
                own_base, b.add(b.mul(m_blk, b.const_i32(4)), b.const_i32(r))
            )
            with b.scf_if(b.cmp_lt(own_pos, seq_len)):
                out_tok = b.add(seq_start, own_pos)
                addr = b.add(
                    _token_base(b, out_tok, head, out_dim, spec.num_heads), out_col
                )
                val = b.fmul(b.vec_extract(result, r), c_scale)
                store_scalar_from_f32(b, params["out"], addr, val, dtype=spec.dtype)

    b.ret()
    return b.kernel


def _build_hstu_attention_bwd_tiled(
    spec: HstuBwdSpec, arch: str = "gfx950"
) -> KernelDef:
    """Tiled, multi-wave HSTU backward mirroring the FlyDSL kernel shape.

    Each CTA owns ``block_m`` rows of the output-row dim (kv for dv/dk, q for
    dq) split across ``num_waves`` wave64 warps, and the full ``out_dim``. It
    streams the reduction dim in ``block_n`` tiles, staging the two streamed
    operands once per tile through LDS:

    * ``lds_head`` -- the streamed head-dim tensor (Q for dv/dk, K for dq),
      XOR-swizzled ``col ^ ((row & 7) << 3)`` (gfx950), stride padded to a
      multiple of 64. Feeds GEMM1's B-operand (and GEMM2's B-operand for dk/dq).
    * ``lds_hidden`` -- the streamed hidden-dim tensor (dO for dv/dk, V for dq),
      row-major. Feeds GEMM2's B-operand (dv) or the dA GEMM's B-operand (dk/dq).

    GEMM1 (score S over head-dim, per (own, stream) sub-tile pair) is computed
    once and its gated fragment is re-laid-out through ``lds_gate`` into the
    GEMM2 A-operand, then **reused across all ``out_dim/16`` output sub-tiles**
    (the key amortization). dk/dq additionally run a dA = dO·Vᵀ GEMM before the
    gate. Epilogue folds ``1/N`` (dv) / ``alpha`` (dk/dq) and stores.
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid hstu_attention_bwd tiled spec: {why}")

    from rocke.core.ir import IRBuilder

    atom = _mfma_atom_for_spec(spec)
    dtype_ir = io_ir_type(spec.dtype)
    apl = atom.a_per_lane  # 4
    cpl = atom.c_per_lane  # 4
    BM, BN, NW = spec.block_m, spec.block_n, spec.num_waves
    BT = NW * 64
    RPW = BM // NW
    OWN_SUB = RPW // 16
    STREAM_SUB = BN // 16
    own_is_kv = spec.which in ("dv", "dk")
    need_da = spec.which in ("dk", "dq")
    out_dim = spec.hidden_dim if spec.which == "dv" else spec.head_dim
    D_CHUNKS = out_dim // 16
    K_STEPS = spec.head_dim // 16
    DA_STEPS = spec.hidden_dim // 16
    HEAD_DIM_K = ((spec.head_dim + 63) // 64) * 64
    inv_n = 1.0 / spec.max_seq_len

    # Tensor roles (see docstring).
    own_s_name = "k" if own_is_kv else "q"  # GEMM1 resident A operand
    own_da_name = "v" if spec.which == "dk" else "do"  # dA resident A operand
    head_stream_name = "q" if own_is_kv else "k"  # staged into lds_head
    hidden_stream_name = "do" if own_is_kv else "v"  # staged into lds_hidden
    gemm2_from_head = spec.which in ("dk", "dq")  # GEMM2 B source

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BT
    if spec.waves_per_eu > 0:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu
    params = _declare_params(b, spec)

    seq_start, seq_end = _seq_bounds(b, params, spec)
    seq_len = b.sub(seq_end, seq_start)
    max_id = _max_id(b, params, spec)
    head = b.block_id_y()
    own_tile_base = b.mul(b.block_id_x(), b.const_i32(BM))

    tid = b.thread_id_x()
    c16 = b.const_i32(16)
    lane = b.mod(tid, b.const_i32(64))
    wave_id = b.div(tid, b.const_i32(64))
    m_in = b.mod(lane, c16)
    m_blk = b.div(lane, c16)
    k_lane_start = b.mul(m_blk, b.const_i32(apl))
    wave_own_off = b.mul(wave_id, b.const_i32(RPW))

    lds_head = b.smem_alloc(dtype_ir, [BN, HEAD_DIM_K], name_hint="hstu_head")
    lds_hidden = b.smem_alloc(dtype_ir, [BN, spec.hidden_dim], name_hint="hstu_hidden")

    def swz(row, col):
        return b.xor(col, b.shl(b.land(row, b.const_i32(7)), b.const_i32(3)))

    def tbase(tok, dim):
        return _token_base(b, tok, head, dim, spec.num_heads)

    def own_tok(og):
        row = b.add(
            own_tile_base, b.add(wave_own_off, b.add(b.const_i32(og * 16), m_in))
        )
        return b.add(seq_start, b.select(b.cmp_lt(row, seq_len), row, b.const_i32(0)))

    # Resident owned operands, hoisted out of the streamed loop.
    own_s_vecs = []
    for og in range(OWN_SUB):
        base = tbase(own_tok(og), spec.head_dim)
        own_s_vecs.append(
            [
                b.global_load_vN(
                    params[own_s_name],
                    b.add(base, b.add(b.const_i32(ks * 16), k_lane_start)),
                    dtype_ir,
                    apl,
                    align=apl * 2,
                )
                for ks in range(K_STEPS)
            ]
        )
    own_da_vecs = None
    if need_da:
        own_da_vecs = []
        for og in range(OWN_SUB):
            base = tbase(own_tok(og), spec.hidden_dim)
            own_da_vecs.append(
                [
                    b.global_load_vN(
                        params[own_da_name],
                        b.add(base, b.add(b.const_i32(ks * 16), k_lane_start)),
                        dtype_ir,
                        apl,
                        align=apl * 2,
                    )
                    for ks in range(DA_STEPS)
                ]
            )

    def _pick_vec(dim):
        for vec in (8, 4, 2, 1):
            if (BN * dim) % (BT * vec) == 0:
                return vec
        return 1

    def stage(lds, ptr_name, dim, stream_base, swizzled):
        vec = _pick_vec(dim)
        for p in range((BN * dim) // (BT * vec)):
            flat = b.mul(b.add(b.const_i32(p * BT), tid), b.const_i32(vec))
            row = b.div(flat, b.const_i32(dim))
            col = b.mod(flat, b.const_i32(dim))
            tok_local = b.add(stream_base, row)
            in_seq = b.cmp_lt(tok_local, seq_len)
            tok = b.add(seq_start, b.select(in_seq, tok_local, b.const_i32(0)))
            gv = b.global_load_vN(
                params[ptr_name],
                b.add(tbase(tok, dim), col),
                dtype_ir,
                vec,
                align=vec * 2,
            )
            scol = swz(row, col) if swizzled else col
            b.smem_store_vN(lds, [row, scol], gv, vec)

    def read_head_pack(ng, ks):
        row = b.add(b.const_i32(ng * 16), m_in)
        col = b.add(b.const_i32(ks * 16), k_lane_start)
        return b.smem_load_vN(lds_head, row, swz(row, col), dtype=dtype_ir, n=apl)

    def read_hidden_pack(ng, ks):
        row = b.add(b.const_i32(ng * 16), m_in)
        col = b.add(b.const_i32(ks * 16), k_lane_start)
        return b.smem_load_vN(lds_hidden, row, col, dtype=dtype_ir, n=apl)

    # Causal streamed-range limiting at block_n granularity (pure causal only).
    pure_causal = (
        spec.max_attn_len == 0 and spec.contextual_seq_len == 0 and not spec.has_targets
    )
    if pure_causal and own_is_kv:
        loop_lo = b.mul(b.div(own_tile_base, b.const_i32(BN)), b.const_i32(BN))
        loop_hi = seq_len
    elif pure_causal:
        loop_lo = b.const_i32(0)
        own_end = b.add(own_tile_base, b.const_i32(BM))
        loop_hi = b.select(b.cmp_lt(own_end, seq_len), own_end, seq_len)
    else:
        loop_lo, loop_hi = b.const_i32(0), seq_len

    n_acc = OWN_SUB * D_CHUNKS
    iter_args = [(f"acc{i}", atom.zero_acc(b)) for i in range(n_acc)]

    with b.scf_if(b.cmp_lt(own_tile_base, seq_len)):
        loop = b.scf_for_iter(
            loop_lo,
            loop_hi,
            b.const_i32(BN),
            iter_args=iter_args,
            iv_name="stream_base",
        )
        with loop as (stream_base, accs):
            accs = list(accs)
            # ---- stage streamed operands into LDS ----
            stage(lds_head, head_stream_name, spec.head_dim, stream_base, True)
            stage(lds_hidden, hidden_stream_name, spec.hidden_dim, stream_base, False)
            b.sync()

            # ---- GEMM1 (+ dA) + gate -> register fragment ----
            # GEMM1/dA are computed as ``C[m=stream, n=own]`` (A=streamed,
            # B=resident-own). Feeding the gated C-fragment DIRECTLY as the
            # GEMM2 A-operand yields the transpose ``A[own, stream]`` the output
            # GEMM needs -- no LDS round-trip for the score (one fewer barrier),
            # matching FlyDSL's fragment reuse.
            gate_frags = [[None] * STREAM_SUB for _ in range(OWN_SUB)]
            for og in range(OWN_SUB):
                own_pos = b.add(
                    own_tile_base,
                    b.add(wave_own_off, b.add(b.const_i32(og * 16), m_in)),
                )
                own_pos_in = b.cmp_lt(own_pos, seq_len)
                for ng in range(STREAM_SUB):
                    score = atom.zero_acc(b)
                    for ks in range(K_STEPS):
                        score = atom.emit(
                            b, read_head_pack(ng, ks), own_s_vecs[og][ks], score
                        )
                    da = None
                    if need_da:
                        da = atom.zero_acc(b)
                        for ks in range(DA_STEPS):
                            da = atom.emit(
                                b, read_hidden_pack(ng, ks), own_da_vecs[og][ks], da
                            )
                    frag = b.zero_vec(dtype_ir, cpl)
                    for r in range(cpl):
                        stream_pos = b.add(
                            stream_base,
                            b.add(
                                b.const_i32(ng * 16),
                                b.add(b.mul(m_blk, b.const_i32(4)), b.const_i32(r)),
                            ),
                        )
                        stream_pos_in = b.cmp_lt(stream_pos, seq_len)
                        if own_is_kv:
                            q_pos, k_pos = stream_pos, own_pos
                        else:
                            q_pos, k_pos = own_pos, stream_pos
                        keep = _mask_keep(b, q_pos, k_pos, max_id, spec)
                        active = b.land(b.land(keep, own_pos_in), stream_pos_in)
                        s = b.vec_extract(score, r)
                        if spec.which == "dv":
                            silu, _ = _silu_and_grad(b, s, spec)
                            val = b.select(active, silu, b.const_f32(0.0))
                        else:
                            _, grad = _silu_and_grad(b, s, spec)
                            da_r = b.vec_extract(da, r)
                            gated = b.fmul(b.fmul(b.const_f32(inv_n), grad), da_r)
                            val = b.select(active, gated, b.const_f32(0.0))
                        frag = b.vec_insert(frag, b.cast_f32_to(val, dtype_ir), r)
                    gate_frags[og][ng] = frag

            # ---- GEMM2: reuse the gate fragment across all d-chunks ----
            for og in range(OWN_SUB):
                for c in range(D_CHUNKS):
                    ai = og * D_CHUNKS + c
                    cur = accs[ai]
                    col = b.add(b.const_i32(c * 16), m_in)
                    for ng in range(STREAM_SUB):
                        b_vec = b.zero_vec(dtype_ir, apl)
                        for j in range(apl):
                            row = b.add(
                                b.const_i32(ng * 16),
                                b.add(k_lane_start, b.const_i32(j)),
                            )
                            if gemm2_from_head:
                                elem = b.vec_extract(
                                    b.smem_load_vN(
                                        lds_head,
                                        row,
                                        swz(row, col),
                                        dtype=dtype_ir,
                                        n=1,
                                    ),
                                    0,
                                )
                            else:
                                elem = b.vec_extract(
                                    b.smem_load_vN(
                                        lds_hidden, row, col, dtype=dtype_ir, n=1
                                    ),
                                    0,
                                )
                            b_vec = b.vec_insert(b_vec, elem, j)
                        cur = atom.emit(b, gate_frags[og][ng], b_vec, cur)
                    accs[ai] = cur
            b.sync()
            b.scf_yield(*accs)

        # ---- epilogue ----
        results = loop.results
        c_scale = b.const_f32(inv_n if spec.which == "dv" else spec.alpha)
        for og in range(OWN_SUB):
            for c in range(D_CHUNKS):
                acc_v = results[og * D_CHUNKS + c]
                out_col = b.add(b.const_i32(c * 16), m_in)
                for r in range(cpl):
                    own_pos = b.add(
                        own_tile_base,
                        b.add(
                            wave_own_off,
                            b.add(
                                b.const_i32(og * 16),
                                b.add(b.mul(m_blk, b.const_i32(4)), b.const_i32(r)),
                            ),
                        ),
                    )
                    with b.scf_if(b.cmp_lt(own_pos, seq_len)):
                        out_tok = b.add(seq_start, own_pos)
                        addr = b.add(tbase(out_tok, out_dim), out_col)
                        val = b.fmul(b.vec_extract(acc_v, r), c_scale)
                        store_scalar_from_f32(
                            b, params["out"], addr, val, dtype=spec.dtype
                        )

    b.ret()
    return b.kernel


def hstu_attention_bwd_block_size(spec: HstuBwdSpec) -> int:
    """Threads-per-CTA for the launch grid."""
    if spec.tiled:
        return spec.num_waves * 64
    return spec.block_size


def build_hstu_attention_bwd(spec: HstuBwdSpec, arch: str = "gfx950") -> KernelDef:
    if spec.tiled:
        return _build_hstu_attention_bwd_tiled(spec, arch)
    if spec.use_mfma_body:
        return _build_hstu_attention_bwd_mfma(spec, arch)

    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid hstu_attention_bwd spec: {why}")

    from rocke.core.ir import IRBuilder

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    params = _declare_params(b, spec)

    seq_start, seq_end = _seq_bounds(b, params, spec)
    seq_len = b.sub(seq_end, seq_start)
    local_token = b.block_id_x()
    token_in_seq = b.cmp_lt(local_token, seq_len)
    token = b.add(seq_start, local_token)
    head = b.block_id_y()

    with b.scf_if(token_in_seq):
        _emit_output_slice(b, params, token, head, spec)

    b.ret()
    return b.kernel


def hstu_attention_bwd_grid(spec: HstuBwdSpec) -> Tuple[int, int, int]:
    if spec.tiled:
        own_tiles = (spec.max_seq_len + spec.block_m - 1) // spec.block_m
        return (own_tiles, spec.num_heads, spec.batch)
    if spec.use_mfma_body:
        atom = _mfma_atom_for_spec(spec)
        row_tiles = (spec.max_seq_len + atom.m - 1) // atom.m
        out_dim = spec.hidden_dim if spec.which == "dv" else spec.head_dim
        d_tiles = out_dim // atom.n
        return (row_tiles * d_tiles, spec.num_heads, spec.batch)
    return (spec.max_seq_len, spec.num_heads, spec.batch)


def hstu_attention_bwd_signature(spec: HstuBwdSpec):
    return (
        SignatureBuilder()
        .ptr("q", spec.dtype)
        .ptr("k", spec.dtype)
        .ptr("v", spec.dtype)
        .ptr("do", spec.dtype)
        .ptr("seq_offsets", "i32")
        .ptr("num_targets", "i32")
        .ptr("perm", "i32")
        .ptr("out", spec.dtype)
        .build()
    )
