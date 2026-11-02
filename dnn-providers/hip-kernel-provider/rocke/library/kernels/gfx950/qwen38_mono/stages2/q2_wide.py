# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP2d: Q2 (q / k RMS norm + RoPE, the K / V / index-key cache appends) of the grouped builds as a rocKE stage.

The rocKE Q2 (:mod:`..stage_q2`) serves the plain and seq builds of up to 4 rows. A grouped build runs FlyDSL's
``qsa.q2`` (rows <= 4) or ``qsa_wide.q2`` (passes of 8 request rows past ``WAVES`` rows); every row is its own
table row (its request's pages, caches and RoPE tables) at its own ``L`` (row ``g S + j``: its group's first row's
``pos`` + ``j``, ATOM ``qsa_group.qsa_len``). This module emits that stage with :mod:`..stage_q2`'s operations
(the mono's norm with the direct scale and the f64 inverse root, RoPE with the lane-8 partner, bf16 rounding at the
mono's points), bitwise by construction, for ``r4g2`` .. ``r16g4``:

* pass ``p``: wave ``w`` runs request row ``w + 8 p`` (FlyDSL wide);
* ``L`` per row: ``args`` (the ABI's ``l0 .. l3``, rows <= 4: ATOM passes ``p.Ls``) | ``table`` (each group's
  ``pos`` loaded by the function, as FlyDSL wide does per stage; the only form past 4 rows);
* knobs as :class:`..stage_fn.StageKnobs`: ``xshfl`` (lane exchanges), ``tbl`` (scalar / vector table loads),
  ``mailbox_impl``; ``mutate`` (test only, :data:`MUTANTS`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType
from rocke.helpers import mailbox as M

from .. import stage_q2 as Q
from ..device_function import DeviceFunctionKnobs

WAVES = 8
MUTANTS = ("rope_partner", "group_l", "pass_row", "cache_row")
# rope_partner: RoPE partner lane ^ 4 (every variant); group_l: every row at its group's first row's L (j dropped);
# pass_row: pass p > 0 reads row w + 8 (p - 1)'s inputs and publishes them as row w + 8 p's; cache_row: the cache
# row's in-block offset (L % 4) ^ 1


@dataclass(frozen=True)
class Q2WideKnobs:
    mailbox_impl: str = "asm"
    xshfl: str = "dpp2"
    tbl: str = "smem"
    lsrc: str = "auto"  # auto (args at rows <= 4, else table) | args | table
    mutate: str = ""

    def __post_init__(self) -> None:
        if self.xshfl not in ("dpp", "lds", "dpp2"):
            raise ValueError("xshfl must be dpp|lds|dpp2")
        if self.tbl not in ("vmem", "smem"):
            raise ValueError("tbl must be vmem|smem")
        if self.lsrc not in ("auto", "args", "table"):
            raise ValueError("lsrc must be auto|args|table")
        if self.mutate not in ("",) + MUTANTS:
            raise ValueError(f"mutate must be '' or one of {MUTANTS}")


def knobs_spec(k: Q2WideKnobs) -> str:
    d = Q2WideKnobs()
    return ",".join(f"{f}={getattr(k, f)}" for f in k.__dataclass_fields__ if getattr(k, f) != getattr(d, f))


def knobs_from_spec(spec: str) -> Q2WideKnobs:
    kw = {}
    for kv in filter(None, (x.strip() for x in (spec or "").split(","))):
        k, _, v = kv.partition("=")
        kw[k] = v
    return Q2WideKnobs(**kw)


def emit_q2_grouped(b, a, lay, rows: int, group: int, k: Q2WideKnobs) -> None:
    impl = k.xshfl
    if k.mailbox_impl == "ir":
        from .. import mailbox_ir  # noqa: F401
    mbk = M.MailboxKnobs(impl=k.mailbox_impl)
    c = b.const_i32
    lane, wave = a.lane, a.wave
    tag, t = a.tag, a.task
    nstate = len(lay.qsa_state)
    space = "constant" if k.tbl == "smem" else "global"
    tbl = b.int_to_ptr(a.qsa, PtrType(I64, space))
    lsrc = k.lsrc if k.lsrc != "auto" else ("args" if rows <= 4 else "table")
    if lsrc == "args" and rows > 4:
        raise ValueError("lsrc=args: the ABI passes 4 rows' L")

    def entry(i):
        return b.readfirstlane(b.global_load_i64(tbl, i))

    # every group's L (the group's first row's pos), loaded once (uniform)
    if lsrc == "table":
        lg = []
        for g in range(rows // group):
            pos = entry(c(g * group * nstate + lay.state_index("pos")))
            lg.append(b.readfirstlane(b.global_load_i32(b.int_to_ptr(pos, PtrType(I32, space)), c(0))))
        ls = []
        for g in range(rows // group):
            ls += [lg[g]] + [lg[g] if k.mutate == "group_l" else b.add(lg[g], c(j)) for j in range(1, group)]
    else:
        ls = list(a.ls[:rows])
        if k.mutate == "group_l":
            ls = [ls[r - r % group] for r in range(rows)]
    a_qn = entry(c(lay.state_index("q_norm")))
    a_kn = entry(c(lay.state_index("k_norm")))

    def mb(region):
        addr = b.add(a.scratch, b.const_i64(lay.sc[region]))
        return M.mailbox_from_i64(b, addr, c(-1), mbk)

    qp, qn = mb("qp"), mb("qn")
    hw = lay.head_words
    nq = lay.q_heads
    assert lay.index_compress == 4, lay.index_compress
    pb = lay.page // lay.index_compress
    pb_sh = pb.bit_length() - 1
    with b.scf_if(b.cmp_lt(t, c(nq + lay.kv_heads))):
        for p in range(-(-rows // WAVES)):
            q = b.add(wave, c(WAVES * p))
            rin = b.smin(q, c(rows - 1))  # the row whose inputs this pass reads
            rout = rin
            if k.mutate == "pass_row" and p > 0:  # inputs of the previous pass's row, published as this row's
                rin = b.smin(b.add(wave, c(WAVES * (p - 1))), c(rows - 1))
            with b.scf_if(b.cmp_lt(q, c(rows))):
                L = ls[0]
                for r in range(1, rows):
                    L = b.select(b.cmp_eq(rin, c(r)), ls[r], L)
                rbase = b.mul(rin, c(nstate))

                def row_entry(name, rbase=rbase):
                    return entry(b.add(rbase, c(lay.state_index(name))))

                a_pages = row_entry("pages")
                a_key = row_entry("key")
                a_value = row_entry("value")
                a_index_key = row_entry("index_key")
                a_cos = row_entry("cos")
                a_sin = row_entry("sin")

                def ix_in(i, rin=rin):
                    return b.add(i, b.mul(rin, c(lay.row_pairs)))

                def ix_out(i, rout=rout):
                    return b.add(i, b.mul(rout, c(lay.row_pairs)))

                ldiv = b.lshr(L, c(2))
                pages = b.int_to_ptr(a_pages, PtrType(I32, space))
                page = b.readfirstlane(b.global_load_i32(pages, b.lshr(ldiv, c(pb_sh))))
                blk = b.add(b.mul(page, c(pb)), b.land(ldiv, c(pb - 1)))
                l3 = b.land(L, c(3))
                if k.mutate == "cache_row":
                    l3 = b.xor(l3, c(1))
                row_l = b.add(b.mul(c(4), blk), l3)
                is_q = b.cmp_lt(t, c(nq))
                kv = b.smax(b.sub(t, c(nq)), c(0))
                first = b.select(is_q, b.add(c(lay.p_q), b.mul(c(2 * hw), t)), b.add(c(lay.p_k), b.mul(c(hw), kv)))
                wsrc = b.select(is_q, a_qn, a_kn)
                two_l = b.mul(c(2), lane)
                wv = b.global_load_vN(b.int_to_ptr(wsrc, PtrType(I32, "global")), two_l, I32, 2)
                tab = b.add(b.mul(L, c(lay.rope_words)), b.mul(c(2), b.smin(lane, c(lay.rotary_dim // 4 - 1))))
                cs = b.global_load_vN(b.int_to_ptr(a_cos, PtrType(I32, "global")), tab, I32, 2)
                sn = b.global_load_vN(b.int_to_ptr(a_sin, PtrType(I32, "global")), tab, I32, 2)
                x = M.poll(b, [(qp, ix_in(b.add(first, two_l)), 2)], tag, mbk)[0]
                xs = Q._unpack(b, x)
                ws = Q._unpack(b, [b.vec_extract(wv, 0), b.vec_extract(wv, 1)])
                sq = [b.fmul(v, v) for v in xs]
                while len(sq) > 1:
                    sq = [b.fadd(sq[2 * i], sq[2 * i + 1]) for i in range(len(sq) // 2)]
                stab = b.fadd(b.fmul(Q._natural_sum(b, sq[0], impl, lane), Q._f32c(b, Q.R256)),
                              Q._f32c(b, lay.eps_bits))  # fmt: skip
                root = b.sqrt(b.cvt_f32_to_f64(stab))
                inv = b.cvt_f64_to_f32(b.fdiv(b.const_f64(1.0), root))
                y = [Q._bf16r(b, b.fmul(b.fmul(v, inv), s)) for v, s in zip(xs, ws)]
                off = 4 if k.mutate == "rope_partner" else 8
                neg = b.select(b.cmp_eq(b.land(lane, c(off)), c(0)), c(Q.SIGN), c(0))
                active = b.cmp_lt(lane, c(lay.rotary_dim // 4))
                css = Q._unpack(b, [b.vec_extract(cs, 0), b.vec_extract(cs, 1)])
                sns = Q._unpack(b, [b.vec_extract(sn, 0), b.vec_extract(sn, 1)])
                out_v = []
                for v, co, si in zip(y, css, sns):
                    pv = Q._xshfl(b, v, off, impl, lane)
                    partner = b.bitcast(b.xor(b.bitcast(pv, I32), neg), F32)
                    r = Q._bf16r(b, b.fadd(Q._bf16r(b, b.fmul(v, co)), Q._bf16r(b, b.fmul(partner, si))))
                    out_v.append(b.select(active, r, v))
                out = [M.bf16_pair(b, out_v[0], out_v[1]), M.bf16_pair(b, out_v[2], out_v[3])]
                M.put2(b, qn, ix_out(b.add(b.mul(t, c(hw)), two_l)), out[0], out[1], tag, mbk)
                with b.scf_if(b.cmp_ge(t, c(nq))):
                    at = b.add(b.mul(b.add(b.mul(row_l, c(lay.kv_heads)), kv), c(hw)), two_l)
                    b.global_store_vN(b.int_to_ptr(a_key, PtrType(I32, "global")), at, b.vec_pack(out, I32), 2)
                    v = M.poll(b, [(qp, ix_in(b.add(c(lay.p_v), b.add(b.mul(c(hw), kv), two_l))), 2)], tag, mbk)[0]
                    b.global_store_vN(b.int_to_ptr(a_value, PtrType(I32, "global")), at, b.vec_pack(v, I32), 2)
                with b.scf_if(b.cmp_eq(t, c(nq))):
                    ik = M.poll(b, [(qp, ix_in(b.add(c(lay.p_ik), lane)), 1)], tag, mbk)[0][0]
                    b.global_store(b.int_to_ptr(a_index_key, PtrType(I32, "global")),
                                   b.add(b.mul(row_l, c(lay.index_words)), lane), ik, align=4)  # fmt: skip


def build_function_kernel(rows: int, group: int, k: Q2WideKnobs, lay):
    from ..stage_fn import ABI_PARAMS

    if group < 2 or rows % group or rows < 2 * group or rows > 16:
        raise ValueError(f"bad grouped variant r{rows}g{group}")
    b = IRBuilder(f"rocke_q38_q2_r{rows}g{group}")
    b.kernel.attrs["max_workgroup_size"] = lay.threads
    b.fence_barriers = "compiler"
    vals = {}
    for name, t in ABI_PARAMS:
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    for name in ("tag", "task", "layer", "l0", "l1", "l2", "l3"):
        vals[name] = b.readfirstlane(vals[name])
    a = SimpleNamespace(**vals, tid=tid, lane=b.land(tid, b.const_i32(63)),
                        wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))),
                        ls=[vals["l0"], vals["l1"], vals["l2"], vals["l3"]])  # fmt: skip
    emit_q2_grouped(b, a, lay, rows, group, k)
    b.ret()
    return b.kernel


def write_functions(out_dir, variants, k: Q2WideKnobs, lay, flavor=None) -> list:
    from ..device_function import lower_device_function, write_device_function

    out = []
    for v in variants:
        m = re.fullmatch(r"r(\d+)g(\d+)", v)
        if not m:
            raise ValueError(f"{v}: grouped variants only")
        kern = build_function_kernel(int(m.group(1)), int(m.group(2)), k, lay)
        fn = lower_device_function(kern, arch="gfx950", symbol=f"rocke_q38_q2_{v}", llvm_flavor=flavor,
                                   knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip
        paths = write_device_function(fn, out_dir, stem=f"q2_{v}", bitcode=False)
        meta = json.load(open(paths["json"]))
        meta["stage_knobs"] = knobs_spec(k)
        with open(paths["json"], "w", newline="\n") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")
        out.append({"variant": v, **paths})
    return out


def main(argv=None) -> int:
    from ..mono_layout import from_atom

    ap = argparse.ArgumentParser(description="emit the WP2d grouped Q2 stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="r4g2,r6g3,r8g4,r9g3,r12g3,r12g4,r16g4")
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--atom-root", default=None)
    a = ap.parse_args(argv)
    print(json.dumps(write_functions(a.out, a.variants.split(","), knobs_from_spec(a.knobs), from_atom(a.atom_root),
                                     flavor=a.flavor), indent=1))  # fmt: skip
    return 0


__all__ = ["MUTANTS", "Q2WideKnobs", "build_function_kernel", "emit_q2_grouped", "knobs_from_spec", "knobs_spec",
           "main", "write_functions"]  # fmt: skip

if __name__ == "__main__":
    sys.exit(main())
