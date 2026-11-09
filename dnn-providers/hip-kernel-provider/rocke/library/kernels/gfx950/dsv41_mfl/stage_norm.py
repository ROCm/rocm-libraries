# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""monofly's ``NormQuant`` stage (RMSNorm -> MXFP8, an optional BF16 copy) of its DeepSeek-V4.1 layer kernels as ONE
rocKE stage function (hybrid path: monofly links it into its FlyDSL layer kernel in place of the stage's body,
``NormQuant(rk=...)``, off by default; the rocKE layer kernel of ``monofly-study dsv41/ports/layer`` links it with
the rest of the layer).

Bit-identical re-implementation of monofly ``blocks/gfx950/norm.py`` ``_emit_stage`` / ``_stage_body`` /
``norm_quant`` / ``put_row`` / ``chan.task_done`` for the forms V4.1's layers run: the CTA sum (``sum="cta"``), the
plain weight rounded once (``plain=True, plain_once=True``: y = bf16(x rstd w)), no pitch, a BF16 channel source
(tagged pairs, this rank's scratch, dual copies) or a TP-gathered one (``src_gathered``: the owners' column slices in
this rank's peer copy, ``peer.tile_index``), the MXFP8 output tagged or bulk + flag (``Proto.FLAG``), the BF16 copy
tagged. A task: part ``task % split`` of row ``task // split``.

Every floating-point op is monofly's, in its order: the sum of squares by ``v_dot2c_f32_bf16`` (chunk ``v`` into
accumulator ``v % 4``, then pairwise), the wave butterfly (32, 16, 8, 4, 2, 1), the waves' partials in LDS added in
wave order, ``mean = sum * f32(1 / width)``, ``v_rsq_f32(mean + eps)``, ``(x * rstd) * w``, the group amax by
``llvm.maximum`` then the lanes' butterfly, E8M0 of its bf16 rounding, ``v_cvt_scalef32_pk_fp8_bf16`` of the bf16
words (``mfabi`` docstring: FlyDSL's float semantics).

ABI (``rocke_v41mfl_norm_<variant>``, ``lds="arg"``: the stage's LDS base, ``nrows x 8`` f32 words of partials)::

    void (i32 tag, i32 task, i64 gamma,
          i64 src, i32 src_loc, i32 src_fl,
          i64 q, i32 q_loc, i32 q_fl, i64 s, i32 s_loc, i32 s_fl,
          i64 f, i32 f_loc, i32 f_fl, i64 bf, i32 bf_loc, i32 bf_fl, i32 lds_base)

``src``: the source region (a gathered one: rank 0's slot of this launch in this rank's peer copy, ``peer.from_rank``;
polled at system scope); ``q`` / ``s``: the MXFP8 channel's words and codes; ``f``: its flags (bulk + flag; else
unused), ``bf``: the BF16 copy (unused without one). Regions as ``mfabi.Region`` (copy-0 address, this CTA's copy
offset, flags).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, fields

from rocke.core.ir import F32, I32, I64, IRBuilder

from ..dsv41_mf.mfabi import CM_DEV, Emit, Region, f32c

THREADS, WAVES, WAVE = 512, 8, 64
GROUP_WORDS = 16  # bf16 words of a 32-group
ACCS = 4
CM_SYS = 17
SYMBOL_PREFIX = "rocke_v41mfl_"
STAGE = "norm"
ABI = (
    ("tag", "i32"), ("task", "i32"), ("gamma", "i64"),
    ("src", "i64"), ("src_loc", "i32"), ("src_fl", "i32"),
    ("q", "i64"), ("q_loc", "i32"), ("q_fl", "i32"), ("s", "i64"), ("s_loc", "i32"), ("s_fl", "i32"),
    ("f", "i64"), ("f_loc", "i32"), ("f_fl", "i32"), ("bf", "i64"), ("bf_loc", "i32"), ("bf_fl", "i32"),
)  # fmt: skip
# Test-only mutants (each must make the bitwise A/B differ or fail the gate):
#   out_bit: every fp8 word's low bit flipped   code: every scale code + 1   acc: a slot's last word out of the sum
#   bf_bit:  every BF16 copy word's low bit flipped   noflag: no flag put (bulk + flag: the consumers hang)
MUTANTS = ("out_bit", "code", "acc", "bf_bit", "noflag")


@dataclass(frozen=True)
class WordMap:
    """monofly ``norm.WordMap``: slots of ``wps`` words over ``words`` words interleaved over ``threads``."""

    words: int
    threads: int
    wps: int

    @property
    def slots(self) -> int:
        return self.words // self.wps

    @property
    def chunks(self) -> int:
        return -(-self.slots // self.threads)

    @property
    def lanes(self) -> int:
        return GROUP_WORDS // self.wps

    def limit(self, v: int) -> int:
        return min(self.threads, self.slots - v * self.threads)

    @property
    def waves(self) -> int:
        return -(-self.limit(0) // WAVE)


@dataclass(frozen=True)
class NormKnobs:
    """The variant (monofly ``NormSpec`` + the stage's channels) and the function's knobs."""

    width: int = 5120
    eps: float = 1e-20
    threads: int = 320
    wps: int = 8
    split: int = 1
    sum_threads: int = 0
    sum_wps: int = 0
    out: str = "tagged"  # the MXFP8 output's protocol: "tagged" | "flag"
    bf: int = 0  # a BF16 copy (tagged)
    tp: int = 0  # a gathered source: TP ranks (0: a channel of this rank's scratch)
    rows: int = 1  # (gathered) the step's rows
    tile: int = 16  # (gathered) pairs a tile
    pause: int = 0
    poll_max: int = 12
    mutate: str = ""

    def __post_init__(self) -> None:
        if self.width % (32 * self.split) or self.wps not in (2, 4, 8, 16):
            raise ValueError(f"{self}: width must split into whole groups; wps 2, 4, 8 or 16")
        if self.out not in ("tagged", "flag"):
            raise ValueError("out: tagged | flag")
        if self.split > 1 and not (self.sum_threads and self.sum_wps):
            raise ValueError("split > 1: give the sum map (sum_threads, sum_wps)")
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate: one of {MUTANTS}")

    @property
    def qmap(self) -> WordMap:
        return WordMap(self.width // 2 // self.split, self.threads, self.wps)

    @property
    def smap(self) -> WordMap:
        return self.qmap if self.split == 1 else WordMap(self.width // 2, self.sum_threads, self.sum_wps)

    def variant(self) -> str:
        v = f"w{self.width}t{self.threads}x{self.wps}s{self.split}"
        if self.split > 1:
            v += f"m{self.sum_threads}x{self.sum_wps}"
        v += "f" if self.out == "flag" else "t"
        v += "b" if self.bf else ""
        v += f"g{self.tp}r{self.rows}l{self.tile}" if self.tp else ""
        d = NormKnobs()
        extra = [f"{n}{getattr(self, n)}" for n in ("pause", "poll_max") if getattr(self, n) != getattr(d, n)]
        extra += [f"mut_{self.mutate}"] if self.mutate else []
        return "_".join([v, *extra]) + ("" if self.eps == 1e-20 else f"_e{self.eps:g}".replace("-", "m"))


def knobs_from_spec(spec: str) -> NormKnobs:
    kw, types = {}, {f.name: f.type for f in fields(NormKnobs)}
    for part in (spec or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        if k not in types:
            raise ValueError(f"unknown knob {k!r}")
        t = types[k]
        kw[k] = int(v) if t in ("int", int) else float(v) if t in ("float", float) else v
    return NormKnobs(**kw)


def symbol(knobs: NormKnobs) -> str:
    return f"{SYMBOL_PREFIX}{STAGE}_{knobs.variant()}"


def lds_words(knobs: NormKnobs) -> int:
    return WAVES  # one row a task: the waves' partials


def build_norm(knobs: NormKnobs = NormKnobs()):
    k = knobs
    qm, sm = k.qmap, k.smap
    shared = sm == qm
    row_words = k.width // 2  # (no pitch) the row's bf16 words
    b = IRBuilder(symbol(k))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = Emit(b, lds_words(k), bar="fence", pause=k.pause, poll_max=k.poll_max)
    c = e.c
    rfl = b.readfirstlane
    tag, task = p["tag"], p["task"]
    src = Region(e, p["src"], p["src_loc"], p["src_fl"])
    rq = Region(e, p["q"], p["q_loc"], p["q_fl"])
    rs = Region(e, p["s"], p["s_loc"], p["s_fl"])
    rf = Region(e, p["f"], p["f_loc"], p["f_fl"]) if k.out == "flag" else None
    rbf = Region(e, p["bf"], p["bf_loc"], p["bf_fl"]) if k.bf else None
    r_gamma = e.rsrc(p["gamma"])
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = rfl(b.lshr(tid, c(6)))
    if k.split == 1:
        row, part = task, c(0)
    else:
        row = b.div(task, c(k.split))
        part = b.sub(task, b.mul(row, c(k.split)))

    def slot_word(m: WordMap, v: int):
        """monofly ``_slot_word``: the first word (in the map's range) of chunk ``v``'s slot, clamped into it."""
        s_ = b.add(tid, c(v * m.threads))
        top = THREADS - 1 + v * m.threads
        return b.mul(s_ if top < m.slots else b.smin(s_, c(m.slots - 1)), c(m.wps))

    # ---- gamma (independent of the rows: first), plain loads of the part's words
    gwords = []
    for v in range(qm.chunks):
        w0 = b.add(b.mul(part, c(qm.words)), slot_word(qm, v))
        ws = []
        for j in range(0, qm.wps, 4):
            n = min(4, qm.wps - j)
            got = b.buffer_load_cp(r_gamma, b.mul(b.add(w0, c(j)), c(4)), c(0), dwords=n, aux=0)
            ws += [b.vec_extract(got, i) for i in range(n)] if n > 1 else [got]
        gwords.append(ws)

    # ---- the row's words: tagged pairs (2 a spec), the part's (qmap) then, split > 1, the whole row's (smap)
    def remap(w):
        if not k.tp:
            return w
        wp = row_words
        q = wp // k.tp
        cc = b.mod(w, c(wp))
        rr = b.div(w, c(wp))
        cl = b.mod(cc, c(q))
        local = b.add(b.mul(b.add(b.mul(b.div(cl, c(k.tile)), c(k.rows)), rr), c(k.tile)), b.mod(cl, c(k.tile)))
        return b.add(b.mul(b.div(cc, c(q)), c(k.rows * q)), local)

    rbase = b.mul(row, c(row_words))

    def map_specs(m: WordMap, base):
        out = []
        for v in range(m.chunks):
            w = b.add(base, slot_word(m, v))
            out += [remap(b.add(w, c(j))) for j in range(0, m.wps, 2)]
        return out

    pairs = map_specs(qm, b.add(rbase, b.mul(part, c(qm.words))))
    if not shared:
        pairs += map_specs(sm, rbase)
    aux = CM_SYS if k.tp else CM_DEV
    got = []
    for b0 in range(0, len(pairs), k.poll_max):
        batch = pairs[b0 : b0 + k.poll_max]
        got += b.tagged_poll(tag, [(src.rsrc, src.load_off(b.mul(pr, c(8))), 4) for pr in batch], aux=aux,
                             pause=e.pause_text(), reload="launder")  # fmt: skip
    words = [w for g in got for w in (b.vec_extract(g, 0), b.vec_extract(g, 2))]

    def chunk_words(m: WordMap, flat):
        per = m.wps
        return [flat[v * per : (v + 1) * per] for v in range(m.chunks)], flat[m.chunks * per :]

    qwords, rest = chunk_words(qm, words)
    swords = qwords if shared else chunk_words(sm, rest)[0]

    # ---- sum of squares (monofly ``_sum_partial``: v_dot2c, chunk v into accumulator v % 4, then pairwise)
    def dot2(a, acc):
        return b.inline_asm("v_dot2c_f32_bf16 $0, $1, $2", "=v,v,v,0", [a, a, acc], F32, sideeffect=False,
                            result_name_hint="dot2")  # fmt: skip

    zero = c(0)
    accs = []
    for v in range(sm.chunks):
        ok = None if sm.limit(v) >= THREADS else b.cmp_lt(tid, c(sm.limit(v)))
        ss = accs[v % ACCS] if v >= ACCS else b.const_f32(0.0)
        for w in swords[v][: -1 if k.mutate == "acc" and v == sm.chunks - 1 else None]:
            ws = w if ok is None else b.select(ok, w, zero)
            ss = dot2(ws, ss)
        if v < ACCS:
            accs.append(ss)
        else:
            accs[v % ACCS] = ss
    while len(accs) > 1:
        accs = [b.fadd(accs[i], accs[i + 1]) if i + 1 < len(accs) else accs[i] for i in range(0, len(accs), 2)]
    psum = accs[0]

    # ---- the CTA sum (monofly ``row_rstd``: two barriers, lane 0 of each wave, the waves in order)
    wsum = e.bfly([psum], (32, 16, 8, 4, 2, 1), b.fadd)[0]
    e.barrier()
    with b.scf_if(b.cmp_eq(lane, c(0))):
        e.stf(wave, wsum)
    e.barrier()
    tot = e.ldf(c(0))
    for w in range(1, sm.waves):
        tot = b.fadd(tot, e.ldf(c(w)))
    rstd = b.rsqrt(b.fadd(b.fmul(tot, e.f(1.0 / k.width)), e.f(k.eps)))

    # ---- quantize (monofly ``quantize_slot``, plain once: y = bf16((x rstd) w)) and put (``put_row``)
    wbase = b.add(rbase, b.mul(part, c(qm.words)))
    rgroup = b.lshr(wbase, c(4))

    def cvt(word, scale, old, hi: bool):
        tmpl = "v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2" + (" op_sel:[0,0,1]" if hi else "")
        return b.inline_asm(tmpl, "=v,v,v,0", [word, scale, old], I32, sideeffect=False, result_name_hint="fp8")

    for v in range(qm.chunks):
        vals = []
        for jw, w in enumerate(qwords[v]):
            lo = b.bitcast(b.shl(w, c(16)), F32)
            hi = b.bitcast(b.land(w, c(-65536)), F32)
            g = gwords[v][jw]
            glo = b.bitcast(b.shl(g, c(16)), F32)
            ghi = b.bitcast(b.land(g, c(-65536)), F32)
            vals += [b.fmul(b.fmul(lo, rstd), glo), b.fmul(b.fmul(hi, rstd), ghi)]
        y = [e.bf_pair(vals[2 * j], vals[2 * j + 1]) for j in range(qm.wps)]
        am = b.fabs(vals[0])
        for x in vals[1:]:
            am = e.fmaximum(am, b.fabs(x))
        offs = tuple(1 << i for i in range(qm.lanes.bit_length() - 1))
        if offs:
            am = e.bfly([am], offs, e.fmaximum)[0]
        code = e.e8m0(e.bfr(am))
        scale = b.bitcast(b.shl(code, c(23)), F32)
        qw = [cvt(y[j + 1], scale, cvt(y[j], scale, c(0), False), True) for j in range(0, qm.wps, 2)]
        if k.mutate == "out_bit":
            qw = [b.xor(x, c(1)) for x in qw]
        if k.mutate == "code":
            code = b.add(code, c(1))
        w = b.mul(b.add(tid, c(v * qm.threads)), c(qm.wps))  # (the slot's word, unclamped: valid where tid < limit)
        with b.scf_if(b.cmp_lt(tid, c(qm.limit(v)))):
            qi = b.lshr(b.add(wbase, w), c(1))  # the fp8 word index
            if k.out == "flag" and len(qw) % 4 == 0:
                for j in range(0, len(qw), 4):
                    rq.store(b.vec_pack(qw[j : j + 4], I32), b.mul(b.add(qi, c(j)), c(4)))
            else:
                for j in range(0, len(qw), 2):
                    ws2 = qw[j : j + 2]
                    if k.out == "flag":  # dense words
                        rq.store(b.vec_pack(ws2, I32) if len(ws2) > 1 else ws2[0], b.mul(b.add(qi, c(j)), c(4)))
                    else:
                        rq.put(b.add(qi, c(j)), [x for wd in ws2 for x in (wd, tag)])
            with b.scf_if(b.cmp_eq(b.land(tid, c(qm.lanes - 1)), c(0))):
                gi = b.add(rgroup, b.lshr(w, c(4)))
                if k.out == "flag":
                    rs.store(code, b.mul(gi, c(4)))
                else:
                    rs.put(gi, [code, tag])
            if rbf is not None:
                bw = y if k.mutate != "bf_bit" else [b.xor(x, c(1)) for x in y]
                for j in range(0, len(bw), 2):
                    rbf.put(b.add(b.add(wbase, w), c(j)), [bw[j], tag, bw[j + 1], tag])

    # ---- the task's end (monofly ``chan.task_done``: a FLAG output's words drained, the CTA met, one flag)
    if rf is not None and k.mutate != "noflag":
        b.s_waitcnt(vmcnt=0)
        e.barrier()
        with b.scf_if(b.cmp_eq(tid, c(0))):
            rf.put(task, [c(1), tag])
    b.ret()
    return b.kernel


def lower_norm_function(knobs: NormKnobs = NormKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(build_norm(knobs), arch=arch, symbol=symbol(knobs), llvm_flavor=flavor,
                                 knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip


def write_norm_function(out: str, knobs: NormKnobs = NormKnobs(), *, arch="gfx950", flavor=None,
                        bitcode="auto") -> dict:  # fmt: skip
    from ..qwen38_mono.device_function import write_device_function

    fn = lower_norm_function(knobs, arch=arch, flavor=flavor)
    paths = write_device_function(fn, out, stem=f"{STAGE}_{knobs.variant()}", bitcode=bitcode)
    meta = {"symbol": symbol(knobs), "knobs": asdict(knobs), "lds_words": lds_words(knobs), "abi": list(ABI),
            **{kk: v for kk, v in paths.items()}}  # fmt: skip
    import os

    path = os.path.join(out, f"{STAGE}_{knobs.variant()}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
    meta["json"] = path
    return meta


def write_set(out: str, specs: list[str], extra: str = "", *, arch="gfx950", flavor=None, name="base") -> str:
    """One function a variant (``specs``: knob specs of the layer's norm stages) + ``extra`` knobs on each; the set's
    index ``norm_set_<name>.json``: {variant spec (without the extra knobs) -> function json}."""
    import os

    os.makedirs(out, exist_ok=True)
    idx = {"variants": {}}
    for sp in specs:
        kn = knobs_from_spec(",".join(x for x in (sp, extra) if x))
        meta = write_norm_function(out, kn, arch=arch, flavor=flavor)
        base = knobs_from_spec(sp)
        idx["variants"][base.variant()] = meta["json"]
    path = os.path.join(out, f"norm_set_{name}.json")
    with open(path, "w") as f:
        json.dump(idx, f, indent=1)
    return path


# The V4.1 layers' norm stages at TP2 (monofly recipe ``_nq``; rows: the gathered tile index's)
# The V4.1 layers' norm forms at TP2 (monofly recipe ``_nq``: attn_norm 5120 320 x 8 (a source's split 2: 320 x 4 with
# the 320 x 8 sum), q_norm 1280 160 x 4, kv_norm 512 64 x 4, ffn_norm 5120 split 4 160 x 4 with the 320 x 8 sum), each
# with the MXFP8 output tagged or bulk + flag, with or without the BF16 copy, from this rank's channel or gathered
FORMS = (
    "width=5120,threads=320,wps=8",
    "width=5120,threads=320,wps=4,split=2,sum_threads=320,sum_wps=8",
    "width=1280,threads=160,wps=4",
    "width=512,threads=64,wps=4",
    "width=5120,threads=160,wps=4,split=4,sum_threads=320,sum_wps=8",
)
# the forms L3 / L21 (TP2, 6 rows) run: the mutant sets' (the gate's must-fail arms)
V41_L3 = (
    "width=5120,threads=320,wps=8,out=flag,tp=2,rows=6",
    "width=1280,threads=160,wps=4",
    "width=512,threads=64,wps=4,bf=1",
    "width=5120,threads=160,wps=4,split=4,sum_threads=320,sum_wps=8,bf=1,tp=2,rows=6",
)


def all_specs(rows) -> list[str]:
    out = []
    for f in FORMS:
        for o in ("", ",out=flag"):
            for bf in ("", ",bf=1"):
                for g in ["", *[f",tp=2,rows={r}" for r in rows]]:
                    out.append(f + o + bf + g)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="1,6,16", help="the step's rows (the gathered variants)")
    ap.add_argument("--knobs", action="append", default=None,
                    help="extra knobs on every variant (one set each; a mutate= set: the L3 / L21 6-row forms only)")
    ap.add_argument("--flavor", default=None)
    a = ap.parse_args(argv)
    every = all_specs([int(r) for r in a.rows.split(",")])
    for extra in a.knobs or [""]:
        name = "base" if not extra else extra.replace(",", "_").replace("=", "")
        specs = list(V41_L3) if "mutate=" in extra else every
        print(write_set(a.out, specs, extra, flavor=a.flavor, name=name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
