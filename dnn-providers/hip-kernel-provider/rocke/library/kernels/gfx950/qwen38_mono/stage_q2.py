# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Q2 of the mono QSA layer as a rocKE stage function (hybrid path).

Bit-identical re-implementation of ATOM ``kernels/stages/qsa.py`` ``q2``
(rows <= ``layout.max_rows``, plain and seq builds; not the grouped and wide
builds). Task ``t`` (26): q head ``t``, or k head ``t - 24``. Wave 0 (wave
``r`` of request row ``r`` in batched builds) holds the head, lane ``l`` dims
``4 l .. 4 l + 3``: the direct-scale RMS norm (square sum as the pair tree
then the lane butterfly; ``inv`` through f64), then RoPE at row ``L`` on
lanes < 16. The head goes out as bf16 pairs (``qn``). The k tasks also append
row ``L`` of the key and value caches, task 24 the raw index key's.

Every floating-point operation is the mono's, in the mono's order, without
fast-math flags: the function is linked into the mono's module and compiled
by the same LLVM, so equal IR operations give equal bits. Lane exchanges may
use other instructions (``StageKnobs.xshfl``): an exchange moves bits, and
each butterfly step is ``v + partner``, which is commutative.
"""

from __future__ import annotations

from rocke.core.ir import F32, I32, I64, PtrType
from rocke.helpers import mailbox as M

from .stage_fn import register_stage

SIGN = -(2**31)
R256 = 0x3B800000  # 1/256

# Test-only mutants (``stage_fn --mutate``): the A/B must see each one differ.
#   rope_partner: RoPE partner lane ^ 4 instead of ^ 8 (every variant)
#   rope_row: every row's RoPE table row from row 0's L (rows > 1, distinct L)
#   cache_row: the appended cache row's in-block offset (L % 4) ^ 1 (every variant)
MUTANTS = ("rope_partner", "rope_row", "cache_row")
MUTATE = None  # one of MUTANTS, or None (the real stage)


def _supported(spec, layout) -> bool:
    return spec.group == 0 and 1 <= spec.rows <= layout.max_rows and (
        not spec.seq or spec.rows > 1
    )


def _f32c(b, bits):
    return b.bitcast(b.const_i32(bits), F32)


def _bf16r(b, x):
    return b.cast_to_f32(b.trunc_f32_to_bf16(x))


def _unpack(b, words):
    out = []
    for w in words:
        out += list(M.bf2_f32(b, w))
    return out


def _xshfl(b, v, off, impl, lane=None):
    """``v`` of lane ``lane ^ off`` (f32 or i32)."""
    is_f = v.type == F32
    x = b.bitcast(v, I32) if is_f else v
    if impl in ("dpp", "dpp2") and off in (1, 2, 8):
        r = b.dpp_xor(x, off)
    elif impl == "dpp2" and off == 4:
        # lanes with bit 2 clear read lane + 4 (row_shl), the others lane - 4
        # (row_shr); the lanes a shift leaves unfilled are never selected
        up = b.mov_dpp(x, row_shl=4)
        dn = b.mov_dpp(x, row_shr=4)
        r = b.select(b.cmp_eq(b.land(lane, b.const_i32(4)), b.const_i32(0)), up, dn)
    else:
        r = b.warp_shuffle_xor(x, off)
    return b.bitcast(r, F32) if is_f else r


def _pair_sum(b, v, swap):
    """``v + v[lane ^ 16]`` (``permlane16_swap``) or ``^ 32``
    (``permlane32_swap``): one of the swapped pair is the lane's own value,
    the other its partner's (as ATOM ``xred``: op(lhs, rhs); commutative)."""
    lo, hi = swap(b.bitcast(v, I32), b.bitcast(v, I32))
    return b.fadd(b.bitcast(lo, F32), b.bitcast(hi, F32))


def _natural_sum(b, v, impl, lane=None):
    """Wave sum, ATOM ``hc.natural_sum``: ``v + v[lane ^ off]`` for off = 1, 2,
    ..., 32 (every lane ends with the same bits: each step is commutative)."""
    for off in (1, 2, 4, 8):
        v = b.fadd(v, _xshfl(b, v, off, impl, lane))
    if impl == "dpp2":
        v = _pair_sum(b, v, b.permlane16_swap)
        return _pair_sum(b, v, b.permlane32_swap)
    v = b.fadd(v, _xshfl(b, v, 16, impl, lane))
    if impl == "dpp":
        # one of the swapped pair is the lane's own value, the other its
        # lane ^ 32 partner (as ATOM ``xred``: op(lhs, rhs))
        lo, hi = b.permlane32_swap(b.bitcast(v, I32), b.bitcast(v, I32))
        return b.fadd(b.bitcast(lo, F32), b.bitcast(hi, F32))
    return b.fadd(v, _xshfl(b, v, 32, impl))


def _q2(b, a, lay, spec):
    rows, seq = spec.rows, spec.seq
    impl = spec.knobs.xshfl
    mbk = M.MailboxKnobs(impl=spec.knobs.mailbox_impl)
    c = b.const_i32
    lane, wave = a.lane, a.wave
    tag, t = a.tag, a.task
    # request row of this wave (batched builds: wave r runs row r)
    rw = c(0) if rows == 1 else b.smin(wave, c(rows - 1))
    L = a.ls[0]
    for q in range(1, rows):
        L = b.select(b.cmp_eq(rw, c(q)), a.ls[q], L)

    nstate = len(lay.qsa_state)
    # tbl="smem": scalar loads (constant space) of the table and page index
    space = "constant" if spec.knobs.tbl == "smem" else "global"
    tbl = b.int_to_ptr(a.qsa, PtrType(I64, space))

    def entry(i):
        return b.readfirstlane(b.global_load_i64(tbl, i))

    # seq builds: one table row, the request's; batched: row rw's caches and
    # RoPE tables, row 0's norm weights (ATOM ``qsa_row``).
    rbase = b.mul(rw, c(nstate)) if (rows > 1 and not seq) else c(0)

    def row_entry(name):
        return entry(b.add(rbase, c(lay.state_index(name))))

    a_pages = row_entry("pages")
    a_key = row_entry("key")
    a_value = row_entry("value")
    a_index_key = row_entry("index_key")
    a_cos = row_entry("cos")
    a_sin = row_entry("sin")
    a_qn = entry(c(lay.state_index("q_norm")))
    a_kn = entry(c(lay.state_index("k_norm")))

    def mb(region):
        addr = b.add(a.scratch, b.const_i64(lay.sc[region]))
        return M.mailbox_from_i64(b, addr, c(-1), mbk)

    qp, qn = mb("qp"), mb("qn")

    def mb_ix(i):
        return b.add(i, b.mul(rw, c(lay.row_pairs))) if rows > 1 else i

    hw = lay.head_words
    nq = lay.q_heads
    with b.scf_if(b.cmp_lt(t, c(nq + lay.kv_heads))):
        mine = b.cmp_eq(wave, c(0)) if rows == 1 else b.cmp_lt(wave, c(rows))
        with b.scf_if(mine):
            # the mono hardcodes L // 4 and L % 4 (index_compress 4) and takes
            # b // PAGE_BLOCKS, b % PAGE_BLOCKS: shifts / masks need powers of two
            assert lay.index_compress == 4, lay.index_compress
            pb = lay.page // lay.index_compress  # index blocks per page
            pb_sh = pb.bit_length() - 1
            assert pb >= 1 and pb == 1 << pb_sh, f"PAGE_BLOCKS {pb} is not a power of two"
            ldiv = b.lshr(L, c(2))  # L // 4 (L >= 0)
            pages = b.int_to_ptr(a_pages, PtrType(I32, space))
            page = b.readfirstlane(b.global_load_i32(pages, b.lshr(ldiv, c(pb_sh))))
            blk = b.add(b.mul(page, c(pb)), b.land(ldiv, c(pb - 1)))
            blk4 = b.mul(c(4), blk)
            l3 = b.land(L, c(3))
            if MUTATE == "cache_row":
                l3 = b.xor(l3, c(1))
            row_l = b.add(blk4, l3)
            is_q = b.cmp_lt(t, c(nq))
            kv = b.smax(b.sub(t, c(nq)), c(0))
            first = b.select(
                is_q,
                b.add(c(lay.p_q), b.mul(c(2 * hw), t)),
                b.add(c(lay.p_k), b.mul(c(hw), kv)),
            )
            wsrc = b.select(is_q, a_qn, a_kn)
            two_l = b.mul(c(2), lane)
            wv = b.global_load_vN(b.int_to_ptr(wsrc, PtrType(I32, "global")), two_l, I32, 2)
            tab = b.add(
                b.mul(a.ls[0] if MUTATE == "rope_row" else L, c(lay.rope_words)),
                b.mul(c(2), b.smin(lane, c(lay.rotary_dim // 4 - 1))),
            )
            cs = b.global_load_vN(b.int_to_ptr(a_cos, PtrType(I32, "global")), tab, I32, 2)
            sn = b.global_load_vN(b.int_to_ptr(a_sin, PtrType(I32, "global")), tab, I32, 2)
            x = M.poll(b, [(qp, mb_ix(b.add(first, two_l)), 2)], tag, mbk)[0]

            # norm (ATOM qsa.norm, direct scale)
            xs = _unpack(b, x)
            ws = _unpack(b, [b.vec_extract(wv, 0), b.vec_extract(wv, 1)])
            sq = [b.fmul(v, v) for v in xs]
            while len(sq) > 1:
                sq = [b.fadd(sq[2 * i], sq[2 * i + 1]) for i in range(len(sq) // 2)]
            stab = b.fadd(
                b.fmul(_natural_sum(b, sq[0], impl, lane), _f32c(b, R256)),
                _f32c(b, lay.eps_bits),
            )
            root = b.sqrt(b.cvt_f32_to_f64(stab))
            inv = b.cvt_f64_to_f32(b.fdiv(b.const_f64(1.0), root))
            y = [_bf16r(b, b.fmul(b.fmul(v, inv), s)) for v, s in zip(xs, ws)]

            # RoPE (ATOM qsa.rope, partner in lane ^ 8)
            off = 4 if MUTATE == "rope_partner" else 8
            neg = b.select(
                b.cmp_eq(b.land(lane, c(off)), c(0)), c(SIGN), c(0)
            )
            active = b.cmp_lt(lane, c(lay.rotary_dim // 4))
            css = _unpack(b, [b.vec_extract(cs, 0), b.vec_extract(cs, 1)])
            sns = _unpack(b, [b.vec_extract(sn, 0), b.vec_extract(sn, 1)])
            out_v = []
            for v, co, si in zip(y, css, sns):
                p = _xshfl(b, v, off, impl, lane)
                partner = b.bitcast(b.xor(b.bitcast(p, I32), neg), F32)
                r = _bf16r(
                    b,
                    b.fadd(_bf16r(b, b.fmul(v, co)), _bf16r(b, b.fmul(partner, si))),
                )
                out_v.append(b.select(active, r, v))
            out = [M.bf16_pair(b, out_v[0], out_v[1]), M.bf16_pair(b, out_v[2], out_v[3])]
            M.put2(b, qn, mb_ix(b.add(b.mul(t, c(hw)), two_l)), out[0], out[1], tag, mbk)

            with b.scf_if(b.cmp_ge(t, c(nq))):
                at = b.add(
                    b.mul(b.add(b.mul(row_l, c(lay.kv_heads)), kv), c(hw)), two_l
                )
                kptr = b.int_to_ptr(a_key, PtrType(I32, "global"))
                b.global_store_vN(kptr, at, b.vec_pack(out, I32), 2)
                v = M.poll(
                    b, [(qp, mb_ix(b.add(c(lay.p_v), b.add(b.mul(c(hw), kv), two_l))), 2)],
                    tag, mbk,
                )[0]  # fmt: skip
                vptr = b.int_to_ptr(a_value, PtrType(I32, "global"))
                b.global_store_vN(vptr, at, b.vec_pack(v, I32), 2)
            with b.scf_if(b.cmp_eq(t, c(nq))):
                ik = M.poll(b, [(qp, mb_ix(b.add(c(lay.p_ik), lane)), 1)], tag, mbk)[0][0]
                iptr = b.int_to_ptr(a_index_key, PtrType(I32, "global"))
                b.global_store(
                    iptr, b.add(b.mul(row_l, c(lay.index_words)), lane), ik, align=4
                )


register_stage("q2", _q2, _supported)
