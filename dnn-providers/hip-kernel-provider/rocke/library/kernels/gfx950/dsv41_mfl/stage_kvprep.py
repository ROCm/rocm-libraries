# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""monofly's ``KVPrep`` stage (the step's key rows: RoPE of the last 64 dims, the MXFP8 QAT round trip, bf16) of its
DeepSeek-V4.1 layer kernels as ONE rocKE stage function (hybrid path: ``KVPrep(rk=...)``, off by default).

Bit-identical re-implementation of monofly ``blocks/gfx950/attention/mla_window.py`` ``_kvprep`` (engine helpers
``row_pos``, ``rope_given`` / ``rope_pair``, ``group_code``, ``qat_word``): a task is row ``task``; threads 0..15 its
32-dim groups: the position (``pos[r // spec] + r % spec``), the RoPE pairs' (cos, sin) loaded before the poll, the
row's group polled (tagged pairs), the rotary groups (the last two) rotated as PyTorch's GPU complex multiply
(``fma(re, c, -(im s))``, ``fma(im, c, re s)``, each rounded to bf16), the group's E8M0 code from
``max(x, -x)`` by ``llvm.maximum``, the E4M3 words (``mfabi.quant_word``) and their values back (exact conversion
times the power-of-two scale), put as bf16 pairs.

ABI (``rocke_v41mfl_kvprep``)::

    void (i32 tag, i32 task, i32 spec, i64 pos, i64 rope, i64 kv, i32 kv_loc, i32 kv_fl, i64 kvn, i32 kvn_loc,
          i32 kvn_fl)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

from rocke.core.ir import F32, FP8E4M3, I32, I64, IRBuilder, VectorType

from ..dsv41_mf.mfabi import Emit, Region

THREADS = 512
HD, ROPE = 512, 64
SYMBOL_PREFIX = "rocke_v41mfl_"
STAGE = "kvprep"
ABI = (
    ("tag", "i32"), ("task", "i32"), ("spec", "i32"), ("pos", "i64"), ("rope", "i64"),
    ("kv", "i64"), ("kv_loc", "i32"), ("kv_fl", "i32"), ("kvn", "i64"), ("kvn_loc", "i32"), ("kvn_fl", "i32"),
)  # fmt: skip
#   out_bit: every output word's low bit flipped   norope: no RoPE   code: every group's code + 1
MUTANTS = ("out_bit", "norope", "code")


@dataclass(frozen=True)
class KVPrepKnobs:
    pause: int = 0
    mutate: str = ""

    def __post_init__(self) -> None:
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate: one of {MUTANTS}")

    def variant(self) -> str:
        v = [f"pause{self.pause}"] if self.pause else []
        v += [f"mut_{self.mutate}"] if self.mutate else []
        return "_".join(v) or "base"


def symbol(knobs: KVPrepKnobs) -> str:
    return f"{SYMBOL_PREFIX}{STAGE}" + ("" if knobs.variant() == "base" else f"_{knobs.variant()}")


def build_kvprep(knobs: KVPrepKnobs = KVPrepKnobs()):
    k = knobs
    b = IRBuilder(symbol(k))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = Emit(b, 4, bar="fence", pause=k.pause)
    c = e.c
    rfl = b.readfirstlane
    tag, r, spec = p["tag"], p["task"], p["spec"]
    kv = Region(e, p["kv"], p["kv_loc"], p["kv_fl"])
    kvn = Region(e, p["kvn"], p["kvn_loc"], p["kvn_fl"])
    tid = b.thread_id_x()
    with b.scf_if(b.cmp_lt(tid, c(HD // 32))):
        g = tid
        bq = b.div(r, spec)
        base = rfl(b.buffer_load_cp(e.rsrc(p["pos"]), b.mul(bq, c(4)), c(0), dwords=1, aux=0))
        pos = b.add(base, b.sub(r, b.mul(bq, spec)))
        e0 = b.add(b.mul(r, c(HD)), b.mul(g, c(32)))
        first = b.div(b.smax(b.sub(b.mul(g, c(32)), c(HD - ROPE)), c(0)), c(2))
        r_rope = e.rsrc(p["rope"])
        cs = []
        for i in range(16):  # (issued before the poll, every thread: clamped, unused below the rotary groups)
            w = b.add(b.mul(pos, c(ROPE)), b.mul(b.add(first, c(i)), c(2)))
            v = b.buffer_load_cp(r_rope, b.mul(w, c(4)), c(0), dwords=2, aux=0)
            cs.append((b.bitcast(b.vec_extract(v, 0), F32), b.bitcast(b.vec_extract(v, 1), F32)))
        got = e.poll(tag, [(kv, b.add(b.lshr(e0, c(1)), c(2 * j)), 2) for j in range(8)])
        vals = []
        for gw in got:
            for q in (0, 2):
                w = b.vec_extract(gw, q)
                vals += [b.bitcast(b.shl(w, c(16)), F32), b.bitcast(b.land(w, c(-65536)), F32)]
        if k.mutate == "norope":
            rot = vals
        else:
            is_rope = b.cmp_ge(g, c((HD - ROPE) // 32))
            rot = []
            for i in range(16):
                re, im = vals[2 * i], vals[2 * i + 1]
                cc, ss = cs[i]
                ra = e.bfr(e.ffma(re, cc, b.fneg(b.fmul(im, ss))))
                rb = e.bfr(e.ffma(im, cc, b.fmul(re, ss)))
                rot += [b.select(is_rope, ra, re), b.select(is_rope, rb, im)]

        def ab(x):  # monofly abs_f32: max(x, -x)
            return e.fmaximum(x, b.fneg(x))

        am = ab(rot[0])
        for v in rot[1:]:
            am = e.fmaximum(am, ab(v))
        code = e.e8m0(am)
        if k.mutate == "code":
            code = b.add(code, c(1))
        scale = b.bitcast(b.shl(code, c(23)), F32)
        back = []
        for w in range(8):
            word = e.quant_word(rot[4 * w : 4 * w + 4], code)
            q4 = b.cvt_pk_f32_fp8x4(b.vec_bitcast(word, VectorType(FP8E4M3, 4)))  # (exact)
            back += [b.fmul(b.vec_extract(q4, i), scale) for i in range(4)]
        for j in range(8):
            w0 = e.bf_pair(back[4 * j], back[4 * j + 1])
            w1 = e.bf_pair(back[4 * j + 2], back[4 * j + 3])
            if k.mutate == "out_bit":
                w0, w1 = b.xor(w0, c(1)), b.xor(w1, c(1))
            kvn.put(b.lshr(b.add(e0, c(4 * j)), c(1)), [w0, tag, w1, tag])
    b.ret()
    return b.kernel


def lower_kvprep_function(knobs: KVPrepKnobs = KVPrepKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(build_kvprep(knobs), arch=arch, symbol=symbol(knobs), llvm_flavor=flavor,
                                 knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip


def write_kvprep_function(out: str, knobs: KVPrepKnobs = KVPrepKnobs(), *, arch="gfx950", flavor=None) -> dict:
    from ..qwen38_mono.device_function import write_device_function

    fn = lower_kvprep_function(knobs, arch=arch, flavor=flavor)
    stem = f"{STAGE}_{knobs.variant()}"
    paths = write_device_function(fn, out, stem=stem, bitcode="auto")
    meta = {"symbol": symbol(knobs), "knobs": asdict(knobs), "abi": list(ABI), **dict(paths)}
    path = os.path.join(out, f"{stem}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
    meta["json"] = path
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--knobs", action="append", default=None, help="mutate=<m> / pause=<n> (one function each)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for spec in a.knobs or [""]:
        kw = dict(x.split("=", 1) for x in spec.split(",") if x)
        kn = KVPrepKnobs(**{kk: (int(v) if kk == "pause" else v) for kk, v in kw.items()})
        print(write_kvprep_function(a.out, kn)["json"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
