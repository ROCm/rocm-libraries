# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split-KV combine for MLA decode: fp32 partials -> normalized latent + LSE.

The second stage of a split-KV decode (:mod:`kernels.mla.mla_decode`
with ``num_seg > 1``). Each ``(token, segment, head)`` partial is an
unnormalized latent accumulator ``acc`` with its running max ``m`` (log2
units) and denominator ``l``, all relative to that segment's ``m``. With
``M = max_s m_s`` and ``w_s = 2**(m_s - M)``::

    latent = sum_s w_s acc_s / sum_s w_s l_s
    lse    = (M + log2(sum_s w_s l_s)) * ln(2)

The rule needs the partials only to be *mutually* consistent, so a segment
that kept a stale max under the lazy rescale merges exactly. An empty segment
reports ``(acc, m, l) = (0, NEG_INIT, 0)``: it weighs ``2**(NEG_INIT - M) = 0``
against any live one, and a row with every segment empty (``S_k = 0``) merges
to a zero latent and ``lse = -inf``.

One wave per ``(token, head)``; each of the 64 lanes owns ``r_kv / 64``
contiguous latent columns, so every partial row is read with fully coalesced
vector loads. Arch-neutral: plain VALU and global memory, no MFMA, no LDS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rocke.core.ir import BF16, F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.attention import safe_inv_l as _safe_inv_l
from rocke.helpers.spec import SignatureBuilder

__all__ = [  # noqa: RUF022  -- ordered spec -> gate -> launch geometry -> build
    "NEG_INIT",
    "MlaDecodeCombineSpec",
    "SUPPORTED_ARCHES",
    "supports_mla_decode_combine",
    "mla_decode_combine_block",
    "mla_decode_combine_grid",
    "mla_decode_combine_signature",
    "build_mla_decode_combine",
]

_WAVE_SIZE = 64

#: Wave64 targets; nothing else in the kernel is arch-specific.
SUPPORTED_ARCHES = ("gfx942", "gfx950")

#: The running max every decode latent kernel starts a segment from, and so
#: the ``m`` an empty segment reports. Finite on purpose: with every segment
#: empty the merge's ``M`` equals it too, so ``2**(m_s - M)`` is 1 rather than
#: ``-inf - -inf = NaN``, and ``l = 0`` then zeroes the row.
NEG_INIT = -1e30


@dataclass(frozen=True)
class MlaDecodeCombineSpec:
    """One combine configuration. ``num_heads`` is the only problem shape;
    the segment count is a kernel argument."""

    num_heads: int
    r_kv: int = 512
    # (token, head) rows per workgroup, one wave each.
    num_warps: int = 4
    dtype: str = "bf16"

    def __post_init__(self) -> None:
        if self.dtype != "bf16":
            raise ValueError(
                f"MLA decode combine writes bf16 only, got dtype={self.dtype!r}"
            )
        for name in ("num_heads", "r_kv"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError(f"num_warps must be one of 1/2/4/8, got {self.num_warps}")

    @property
    def threads(self) -> int:
        return _WAVE_SIZE * self.num_warps

    @property
    def cols_per_lane(self) -> int:
        return self.r_kv // _WAVE_SIZE

    def kernel_name(self) -> str:
        return (
            f"mla_decode_combine_h{self.num_heads}_rkv{self.r_kv}"
            f"_w{self.num_warps}_{self.dtype}"
        )


def supports_mla_decode_combine(
    spec: MlaDecodeCombineSpec, *, arch: str
) -> tuple[bool, str]:
    """``(ok, reason)``; ``reason`` is empty when ``ok``."""
    if arch.split(":")[0] not in SUPPORTED_ARCHES:
        return False, (
            f"MLA decode combine supports {SUPPORTED_ARCHES} (wave64), got arch={arch!r}"
        )
    if spec.num_heads % spec.num_warps != 0:
        return False, (
            f"num_heads={spec.num_heads} must be a multiple of num_warps="
            f"{spec.num_warps}: one wave per head"
        )
    # Each lane loads its columns as whole 4-wide f32 vectors and stores them
    # as one bf16 vector of the same width.
    if spec.r_kv % (_WAVE_SIZE * 4) != 0 or spec.cols_per_lane not in (4, 8, 16):
        return False, (
            f"r_kv={spec.r_kv} must give each of the {_WAVE_SIZE} lanes 4, 8 or 16 "
            "columns"
        )
    return True, ""


def mla_decode_combine_block(spec: MlaDecodeCombineSpec) -> tuple[int, int, int]:
    return (spec.threads, 1, 1)


def mla_decode_combine_grid(
    spec: MlaDecodeCombineSpec, *, total_q: int
) -> tuple[int, int, int]:
    """``(head groups, tokens, 1)``."""
    if total_q <= 0:
        raise ValueError(f"total_q must be positive, got {total_q}")
    return (spec.num_heads // spec.num_warps, total_q, 1)


def mla_decode_combine_signature(spec: MlaDecodeCombineSpec):
    """``out`` / ``lse`` as the single-pass decode kernel writes them, then the
    workspace the split kernel filled, then its segment count."""
    return (
        SignatureBuilder()
        .ptr("out_ptr", spec.dtype)
        .ptr("lse_ptr", "f32")
        .ptr("part_acc_ptr", "f32")
        .ptr("part_ml_ptr", "f32")
        .scalar("num_seg", "i32")
        .build()
    )


def build_mla_decode_combine(
    spec: MlaDecodeCombineSpec, *, arch: str = "gfx942"
) -> KernelDef:
    ok, why = supports_mla_decode_combine(spec, arch=arch)
    if not ok:
        raise NotImplementedError(why)
    H = spec.num_heads
    R_KV = spec.r_kv
    CPL = spec.cols_per_lane
    LN2 = math.log(2.0)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    out = b.param(
        "out_ptr", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16
    )
    lse = b.param(
        "lse_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=16
    )
    part_acc = b.param(
        "part_acc_ptr", PtrType(F32, "global"), noalias=True, readonly=True, align=16
    )
    part_ml = b.param(
        "part_ml_ptr", PtrType(F32, "global"), noalias=True, readonly=True, align=8
    )
    num_seg = b.param("num_seg", I32)

    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(_WAVE_SIZE))
    wave = b.div(tid, b.const_i32(_WAVE_SIZE))
    head = b.readfirstlane(
        b.add(b.mul(b.block_id_x(), b.const_i32(spec.num_warps)), wave)
    )
    token = b.block_id_y()

    # Row (token, segment s, head) of the workspace is token * num_seg * H +
    # s * H + head; consecutive segments of one row are H rows apart.
    row0 = b.add(b.mul(b.mul(token, num_seg), b.const_i32(H)), head)
    col = b.mul(lane, b.const_i32(CPL))

    # Pass 1: the global max. Every lane reads the same wave-uniform (m, l).
    m_loop = b.scf_for_iter(
        b.const_i32(0),
        num_seg,
        b.const_i32(1),
        iter_args=[("m_max", b.const_f32(NEG_INIT))],
        iv_name="seg_max",
    )
    with m_loop as (s, state):
        row = b.add(row0, b.mul(s, b.const_i32(H)))
        m_s = b.global_load_f32(part_ml, b.mul(row, b.const_i32(2)))
        b.scf_yield(b.fmax(state[0], m_s))
    big_m = m_loop.results[0]

    # Pass 2: rescale and sum. 2**(m_s - M) <= 1, and v_exp_f32 flushes the
    # NEG_INIT of an empty segment to an exact 0.
    n_vec = CPL // 4
    acc_loop = b.scf_for_iter(
        b.const_i32(0),
        num_seg,
        b.const_i32(1),
        iter_args=[("l_sum", b.const_f32(0.0))]
        + [(f"acc{i}", b.zero_vec_f32(4)) for i in range(n_vec)],
        iv_name="seg_sum",
    )
    with acc_loop as (s, state):
        row = b.add(row0, b.mul(s, b.const_i32(H)))
        ml = b.global_load_vN(part_ml, b.mul(row, b.const_i32(2)), F32, 2, align=8)
        w = b.exp2_fast(b.fsub(b.vec_extract(ml, 0), big_m))
        l_sum = b.fadd(state[0], b.fmul(w, b.vec_extract(ml, 1)))
        base = b.add(b.mul(row, b.const_i32(R_KV)), col)
        accs = []
        for i in range(n_vec):
            v = b.global_load_vN(part_acc, b.add(base, b.const_i32(4 * i)), F32, 4)
            acc = state[1 + i]
            for r in range(4):
                acc = b.vec_insert(
                    acc,
                    b.fadd(b.vec_extract(acc, r), b.fmul(w, b.vec_extract(v, r))),
                    r,
                )
            accs.append(acc)
        b.scf_yield(l_sum, *accs)
    l_sum = acc_loop.results[0]
    accs = acc_loop.results[1:]

    inv = _safe_inv_l(b, l_sum)
    vals = b.vec_pack(
        [b.fmul(b.vec_extract(a, r), inv) for a in accs for r in range(4)], F32
    )
    out_row = b.add(
        b.mul(b.add(b.mul(token, b.const_i32(H)), head), b.const_i32(R_KV)), col
    )
    b.global_store_vN(out, out_row, b.vec_trunc_f32_to_bf16(vals), CPL, align=2 * CPL)
    with b.scf_if(b.cmp_eq(lane, b.const_i32(0))):
        value = b.fmul(b.fadd(big_m, b.log2(l_sum)), b.const_f32(LN2))
        b.global_store(lse, b.add(b.mul(token, b.const_i32(H)), head), value, align=4)
    return b.kernel
