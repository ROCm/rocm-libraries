#!/usr/bin/env python3
###############################################################################
#
# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
###############################################################################
"""Problem corpus of the rocKE selection training sweep.

Space-filling design (scrambled Halton over the log2 extents of the GEMM view) per data-type row,
with random mode splits and random storage orders; a refinement round adds jittered neighbours of
problems where the selector lost (refine).

Spec lines (one problem each, read by sweep_driver.cpp):
  C <id> <dtype> <compute> <bilinear> <nA> <nB> <nD> modesA.. extA.. strA.. modesB.. extB.. strB..
    modesD.. extD.. strD..                       (C shares D's layout when bilinear)
  R <id> <dtype> <compute> <op> <beta> <nA> <nD> modesA.. extA.. strA.. modesD.. extD.. strD..

usage:
  gen_corpus.py contraction --seed S --per-row N --out FILE [--rows h16f,h16b,s,c,d,z]
  gen_corpus.py reduction   --seed S --per-row N --out FILE
  gen_corpus.py refine      --from SPECS --worst IDS --seed S --per 3 --out FILE
"""

import argparse
import math
import random

CROWS = {
    # row: dtype, compute, element bytes, flop cap (log2 of M*N*K), log2 extent range
    "h16f": ("f16", "32F", 2, 35, (3, 13.5)),
    "h16b": ("bf16", "32F", 2, 35, (3, 13.5)),
    "s": ("f32", "32F", 4, 34, (3, 13)),
    "c": ("c32", "C32F", 8, 32, (3, 12.5)),
    "d": ("f64", "64F", 8, 33, (3, 13)),
    "z": ("c64", "C64F", 16, 31, (3, 12)),
    "hx": ("f32", "16BF", 4, 34, (3, 13)),
    "hy": ("f32", "16F", 4, 34, (3, 13)),
}
RROWS = {
    "rh": ("f16", "32F", 2),
    "rb": ("bf16", "32F", 2),
    "rs": ("f32", "32F", 4),
    "rd": ("f64", "64F", 8),
}
MAX_TENSOR_BYTES = 1 << 28
PRIMES = (2, 3, 5, 7, 11)


def halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def split_extent(x, k, rng):
    """k extents (>= 2 unless x is small) whose product is close to x."""
    if k == 1 or x < 4:
        return [max(1, x)]
    out = []
    rest = x
    for j in range(k - 1):
        lg = math.log2(rest)
        share = lg / (k - j)
        e = int(round(2 ** (share * rng.uniform(0.5, 1.5))))
        e = max(2, min(e, rest // 2))
        out.append(e)
        rest = max(1, (rest + e - 1) // e)
    out.append(max(1, rest))
    rng.shuffle(out)
    return out


def round_extent(v, rng):
    """Mostly odd sizes; sometimes a multiple of 64 or a power of two."""
    u = rng.random()
    if u < 0.2 and v >= 64:
        return int(round(v / 64)) * 64
    if u < 0.3:
        return 1 << int(round(math.log2(max(v, 1))))
    return max(1, int(round(v)))


def packed_strides(modes, ext, rng):
    order = list(range(len(modes)))
    rng.shuffle(order)
    strides = [0] * len(modes)
    s = 1
    for j in order:
        strides[j] = s
        s *= ext[modes[j]]
    return strides


def tensor_fields(modes, ext, strides):
    return (
        [str(m) for m in modes]
        + [str(ext[m]) for m in modes]
        + [str(x) for x in strides]
    )


def contraction_spec(pid, row, lgm, lgn, lgk, rng):
    dtype, compute, es, cap, (lo, hi) = CROWS[row]
    M = round_extent(2**lgm, rng)
    N = round_extent(2**lgn, rng)
    K = round_extent(2**lgk, rng)
    # flop cap and tensor size cap: shrink the largest extent
    while (
        math.log2(M * N * K) > cap or max(M * K, N * K, M * N) * es > MAX_TENSOR_BYTES
    ):
        if M >= N and M >= K:
            M = max(1, M * 3 // 4)
        elif N >= K:
            N = max(1, N * 3 // 4)
        else:
            K = max(1, K * 3 // 4)
    nm = rng.choice((1, 1, 2, 2, 3))
    nn = rng.choice((1, 1, 2))
    nk = rng.choice((1, 1, 2, 3))
    em, en, ek = (
        split_extent(M, nm, rng),
        split_extent(N, nn, rng),
        split_extent(K, nk, rng),
    )
    ext = {}
    mm, mn, mk = [], [], []
    code = 1
    for group, es_ in ((mm, em), (mn, en), (mk, ek)):
        for e in es_:
            ext[code] = e
            group.append(code)
            code += 1
    ma = mm + mk
    mb = mn + mk
    md = mm + mn
    for x in (ma, mb, md):
        rng.shuffle(x)
    bil = 1 if rng.random() < 0.5 else 0
    sa = packed_strides(ma, ext, rng)
    sb = packed_strides(mb, ext, rng)
    sd = packed_strides(md, ext, rng)
    f = [
        "C",
        str(pid),
        dtype,
        compute,
        str(bil),
        str(len(ma)),
        str(len(mb)),
        str(len(md)),
    ]
    f += (
        tensor_fields(ma, ext, sa)
        + tensor_fields(mb, ext, sb)
        + tensor_fields(md, ext, sd)
    )
    return " ".join(f)


def reduction_spec(pid, row, lgtotal, frac_kept, rng):
    dtype, compute, es = RROWS[row]
    total = int(2**lgtotal)
    rank = rng.choice((2, 2, 3, 3, 4))
    exts = split_extent(total, rank, rng)
    rank = len(exts)
    if rank < 2:
        exts = [max(2, exts[0] // 2), 2]
        rank = 2
    modes = list(range(1, rank + 1))
    ext = dict(zip(modes, exts))
    nkept = max(1, min(rank - 1, int(round(frac_kept * (rank - 1))) or 1))
    ma = modes[:]
    rng.shuffle(ma)
    kept = ma[:]
    rng.shuffle(kept)
    md = kept[:nkept]
    sa = packed_strides(ma, ext, rng)
    sd = packed_strides(md, ext, rng)
    op = "ADD" if rng.random() < 0.8 else "MAX"
    beta = "1" if rng.random() < 0.5 else "0"
    f = ["R", str(pid), dtype, compute, op, beta, str(len(ma)), str(len(md))]
    f += tensor_fields(ma, ext, sa) + tensor_fields(md, ext, sd)
    return " ".join(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("contraction", "reduction", "refine"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-row", type=int, default=100)
    ap.add_argument("--rows", default="")
    ap.add_argument("--first-id", type=int, default=0)
    ap.add_argument("--from", dest="src", default="")
    ap.add_argument("--worst", default="")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lines = []
    pid = a.first_id
    if a.kind == "contraction":
        rows = a.rows.split(",") if a.rows else list(CROWS)
        for row in rows:
            lo, hi = CROWS[row][4]
            shift = [rng.random() for _ in range(3)]
            for i in range(a.per_row):
                u = [(halton(i + 1, PRIMES[d]) + shift[d]) % 1.0 for d in range(3)]
                lg = [lo + (hi - lo) * x for x in u]
                lines.append(contraction_spec(pid, row, *lg, rng))
                pid += 1
    elif a.kind == "reduction":
        rows = a.rows.split(",") if a.rows else list(RROWS)
        for row in rows:
            shift = [rng.random() for _ in range(2)]
            for i in range(a.per_row):
                u = [(halton(i + 1, PRIMES[d]) + shift[d]) % 1.0 for d in range(2)]
                lines.append(reduction_spec(pid, row, 8 + 18 * u[0], u[1], rng))
                pid += 1
    else:
        # jittered neighbours of the worst problems: every extent scaled by 2^U(-0.5, 0.5),
        # storage orders redrawn
        worst = set(int(x) for x in open(a.worst).read().split())
        for line in open(a.src):
            f = line.split()
            if not f or int(f[1]) not in worst:
                continue
            for _ in range(a.per):
                g = list(f)
                g[1] = str(pid)
                pid += 1
                if f[0] == "C":
                    na, nb, nd = int(f[5]), int(f[6]), int(f[7])
                    base = 8
                    groups = [na, nb, nd]
                else:
                    na, nd = int(f[6]), int(f[7])
                    base = 8
                    groups = [na, nd]
                # new extents per mode code
                ext = {}
                pos = base
                for n in groups:
                    modes = g[pos : pos + n]
                    exts = g[pos + n : pos + 2 * n]
                    for m, e in zip(modes, exts):
                        if m not in ext:
                            ext[m] = max(
                                1, int(round(int(e) * 2 ** rng.uniform(-0.5, 0.5)))
                            )
                    pos += 3 * n
                pos = base
                for n in groups:
                    modes = g[pos : pos + n]
                    e2 = {m: ext[m] for m in modes}
                    st = packed_strides(modes, e2, rng)
                    for j, m in enumerate(modes):
                        g[pos + n + j] = str(ext[m])
                        g[pos + 2 * n + j] = str(st[j])
                    pos += 3 * n
                lines.append(" ".join(g))
    with open(a.out, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{a.out}: {len(lines)} problems, ids {a.first_id}..{pid - 1}")


if __name__ == "__main__":
    main()
