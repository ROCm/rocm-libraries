# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""monofly V4.1's ``Store`` stage (a BF16 / F32 channel's row -> plain memory: what the next launch reads) as ONE rocKE
stage function (hybrid path: ``Store(rk=...)``, off by default).

Bit-identical re-implementation of monofly ``models/deepseek_v41/recipe.py`` ``_store``: row ``task``; thread ``tid``
its pairs ``tid, tid + 512, ..`` of the row (``width`` pairs a row): each polled (one tagged pair) and its word stored
plain at word ``task x width + i`` of ``dst``.

ABI (``rocke_v41mfl_store_w<width>``)::

    void (i32 tag, i32 task, i64 src, i32 src_loc, i32 src_fl, i64 dst, i32 lds_base)
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
SYMBOL_PREFIX = "rocke_v41mfl_"
STAGE = "store"
ABI = (("tag", "i32"), ("task", "i32"), ("src", "i64"), ("src_loc", "i32"), ("src_fl", "i32"), ("dst", "i64"))
#   out_bit: every stored word's low bit flipped
MUTANTS = ("out_bit",)


@dataclass(frozen=True)
class StoreKnobs:
    width: int = 2560  # pairs a row
    pause: int = 0
    mutate: str = ""

    def __post_init__(self) -> None:
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate: one of {MUTANTS}")

    def variant(self) -> str:
        v = [f"w{self.width}"] + ([f"pause{self.pause}"] if self.pause else [])
        return "_".join(v + ([f"mut_{self.mutate}"] if self.mutate else []))


def symbol(knobs: StoreKnobs) -> str:
    return f"{SYMBOL_PREFIX}{STAGE}_{knobs.variant()}"


def build_store(knobs: StoreKnobs = StoreKnobs()):
    k = knobs
    b = IRBuilder(symbol(k))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = Emit(b, 4, bar="fence", pause=k.pause)
    c = e.c
    tag, t = p["tag"], p["task"]
    src = Region(e, p["src"], p["src_loc"], p["src_fl"])
    r_dst = e.rsrc(p["dst"])
    tid = b.thread_id_x()
    for j in range(-(-k.width // THREADS)):
        i = b.add(tid, c(j * THREADS))
        with b.scf_if(b.cmp_lt(i, c(k.width))):
            at = b.add(b.mul(t, c(k.width)), i)
            w = b.vec_extract(e.poll(tag, [(src, at, 1)])[0], 0)
            if k.mutate == "out_bit":
                w = b.xor(w, c(1))
            b.buffer_store_cp(w, r_dst, b.mul(at, c(4)), c(0), aux=0)
    b.ret()
    return b.kernel


def lower_store_function(knobs: StoreKnobs = StoreKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(build_store(knobs), arch=arch, symbol=symbol(knobs), llvm_flavor=flavor,
                                 knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip


def write_set(out: str, widths, extra: str = "", *, name="base") -> str:
    """One function a row width (pairs) + ``extra`` knobs; the set's index ``store_set_<name>.json``:
    {"w<width>": function json}."""
    from ..qwen38_mono.device_function import write_device_function

    os.makedirs(out, exist_ok=True)
    idx = {"variants": {}}
    for w in widths:
        kw = dict(x.split("=", 1) for x in extra.split(",") if x)
        kn = StoreKnobs(width=int(w), **{kk: (int(v) if kk == "pause" else v) for kk, v in kw.items()})
        fn = lower_store_function(kn)
        stem = f"{STAGE}_{kn.variant()}"
        paths = write_device_function(fn, out, stem=stem, bitcode="auto")
        meta = {"symbol": symbol(kn), "knobs": asdict(kn), "abi": list(ABI), **dict(paths)}
        path = os.path.join(out, f"{stem}.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=1)
        idx["variants"][f"w{w}"] = path
    path = os.path.join(out, f"store_set_{name}.json")
    with open(path, "w") as f:
        json.dump(idx, f, indent=1)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="2560,5120,256,512", help="pairs a row (the V4.1 stores')")
    ap.add_argument("--knobs", action="append", default=None, help="extra knobs (one set each)")
    a = ap.parse_args(argv)
    for extra in a.knobs or [""]:
        name = "base" if not extra else extra.replace(",", "_").replace("=", "")
        print(write_set(a.out, a.widths.split(","), extra, name=name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
