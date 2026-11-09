# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""monofly's ``RingWrite`` stage (the step's key row into the window ring once every attention task of its request
has read the ring) of its DeepSeek-V4.1 layer kernels as ONE rocKE stage function (hybrid path: ``RingWrite(rk=...)``,
off by default).

Bit-identical re-implementation of monofly ``blocks/gfx950/attention/mla_window.py`` ``_ring_write``: row ``task``'s
request ``b = r // spec`` and position ``pos[b] + r % spec``; threads ``tid < spec x tiles`` poll the attention's
ring-read flags (``<attn>.done``, one pair each), threads ``tid < spec x mper`` the merge's (``<attn>.mdone``), a CTA
barrier, then threads ``tid < 256`` poll the row's key word (``kvn``, a pair each) and store it, plain, into the ring
at slot ``pos % ring_slots``.

ABI (``rocke_v41mfl_ringw``)::

    void (i32 tag, i32 task, i32 spec, i32 tiles, i32 mper, i32 ring_slots, i64 pos, i64 ring,
          i64 done, i32 done_loc, i32 done_fl, i64 mdone, i32 mdone_loc, i32 mdone_fl,
          i64 kvn, i32 kvn_loc, i32 kvn_fl, i32 lds_base)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

from rocke.core.ir import I32, I64, IRBuilder

from ..dsv41_mf.mfabi import Emit, Region

THREADS = 512
HD = 512
SYMBOL_PREFIX = "rocke_v41mfl_"
STAGE = "ringw"
ABI = (
    ("tag", "i32"), ("task", "i32"), ("spec", "i32"), ("tiles", "i32"), ("mper", "i32"), ("ring_slots", "i32"),
    ("pos", "i64"), ("ring", "i64"),
    ("done", "i64"), ("done_loc", "i32"), ("done_fl", "i32"), ("mdone", "i64"), ("mdone_loc", "i32"),
    ("mdone_fl", "i32"), ("kvn", "i64"), ("kvn_loc", "i32"), ("kvn_fl", "i32"),
)  # fmt: skip
#   out_bit: every stored word's low bit flipped   nowait: no flag poll (the ring may be overwritten while read)
#   slot: the key into the next slot (the window reads a stale key)
MUTANTS = ("out_bit", "nowait", "slot")


@dataclass(frozen=True)
class RingwKnobs:
    pause: int = 0
    mutate: str = ""

    def __post_init__(self) -> None:
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate: one of {MUTANTS}")

    def variant(self) -> str:
        v = [f"pause{self.pause}"] if self.pause else []
        v += [f"mut_{self.mutate}"] if self.mutate else []
        return "_".join(v) or "base"


def symbol(knobs: RingwKnobs) -> str:
    return f"{SYMBOL_PREFIX}{STAGE}" + ("" if knobs.variant() == "base" else f"_{knobs.variant()}")


def build_ringw(knobs: RingwKnobs = RingwKnobs()):
    k = knobs
    b = IRBuilder(symbol(k))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = Emit(b, 4, bar="fence", pause=k.pause)
    c = e.c
    rfl = b.readfirstlane
    tag, r, spec = p["tag"], p["task"], p["spec"]
    tid = b.thread_id_x()
    bq = b.div(r, spec)
    base = rfl(b.buffer_load_cp(e.rsrc(p["pos"]), b.mul(bq, c(4)), c(0), dwords=1, aux=0))
    pos = b.add(base, b.sub(r, b.mul(bq, spec)))
    if k.mutate != "nowait":
        done = Region(e, p["done"], p["done_loc"], p["done_fl"])
        nt = b.mul(spec, p["tiles"])
        with b.scf_if(b.cmp_lt(tid, nt)):
            e.poll(tag, [(done, b.add(b.mul(bq, nt), tid), 1)])
        mdone = Region(e, p["mdone"], p["mdone_loc"], p["mdone_fl"])
        nm = b.mul(spec, p["mper"])
        with b.scf_if(b.cmp_lt(tid, nm)):  # (mper 0: no thread)
            e.poll(tag, [(mdone, b.add(b.mul(bq, nm), tid), 1)])
    e.barrier()
    kvn = Region(e, p["kvn"], p["kvn_loc"], p["kvn_fl"])
    with b.scf_if(b.cmp_lt(tid, c(HD // 2))):
        w = b.vec_extract(e.poll(tag, [(kvn, b.add(b.mul(r, c(HD // 2)), tid), 1)])[0], 0)
        if k.mutate == "out_bit":
            w = b.xor(w, c(1))
        ps = b.add(pos, c(1)) if k.mutate == "slot" else pos
        slot = b.add(b.mul(bq, p["ring_slots"]), b.mod(ps, p["ring_slots"]))
        at = b.add(b.mul(slot, c(HD // 2)), tid)
        b.buffer_store_cp(w, e.rsrc(p["ring"]), b.mul(at, c(4)), c(0), aux=0)
    b.ret()
    return b.kernel


def lower_ringw_function(knobs: RingwKnobs = RingwKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(build_ringw(knobs), arch=arch, symbol=symbol(knobs), llvm_flavor=flavor,
                                 knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip


def write_ringw_function(out: str, knobs: RingwKnobs = RingwKnobs(), *, arch="gfx950", flavor=None) -> dict:
    from ..qwen38_mono.device_function import write_device_function

    fn = lower_ringw_function(knobs, arch=arch, flavor=flavor)
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
        kn = RingwKnobs(**{kk: (int(v) if kk == "pause" else v) for kk, v in kw.items()})
        print(write_ringw_function(a.out, kn)["json"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
