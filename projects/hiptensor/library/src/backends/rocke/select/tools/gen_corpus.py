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
with random mode splits and random storage orders; --aligned makes a share of the problems' extents
multiples of 8 (the GEMM bodies' K tiles divide them, so rocKE's tensor GETT planner serves more of
them), --padded pads a share of the tensors' strides. A refinement round adds jittered neighbours of
problems where the selector lost (refine); `pick` writes the lines of chosen problem ids (the
problems train_select.py propose chose from a pool). --big draws large problems instead: GEMM-view
extents from 2^9 up to 2^2 past the row's range, the flop cap raised by 2^7 and tensors up to
1 GiB (the sizes of the requirements' large rows, past the default design's caps).

--outer F makes a share F of the problems outer products (no contracted mode: D = A B with every
mode of A and of B in D), which hipTensor admits. Batch modes (in A, B and D), modes of one tensor
only and contractions without M or N modes are not drawn: hipTensor refuses them
(hiptensorCreateContraction returns HIPTENSOR_STATUS_NOT_SUPPORTED). --style api draws the problems
of the API validation harness instead (1-3 modes per group, 75 % of the problems with extents that
are multiples of 8, packed tensors in random mode orders, tensors of 2^12 .. 2^25 elements, f16 /
bf16; problem i of a corpus is the harness's problem of seed --seed + i).

Spec lines (one problem each, read by sweep_driver.cpp):
  C <id> <dtype> <compute> <bilinear> <nA> <nB> <nD> modesA.. extA.. strA.. modesB.. extB.. strB..
    modesD.. extD.. strD..                       (C shares D's layout when bilinear)
  R <id> <dtype> <compute> <op> <beta> <nA> <nD> modesA.. extA.. strA.. modesD.. extD.. strD..

usage:
  gen_corpus.py contraction --seed S --per-row N --out FILE [--rows h16f,h16b,s,c,d,z]
                            [--aligned 0.6] [--padded 0.15] [--first-id I] [--big]
                            [--outer 0.1] [--style halton|api]
  gen_corpus.py reduction   --seed S --per-row N --out FILE
  gen_corpus.py refine      --from SPECS --worst IDS --seed S --per 3 --out FILE
  gen_corpus.py pick        --from SPECS --worst IDS --out FILE
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


def packed_strides(modes, ext, rng, pad=0.0):
    """Strides of a random storage order; with probability `pad` the second and later
    dimensions start a few elements further than packed (padded strides)."""
    order = list(range(len(modes)))
    rng.shuffle(order)
    padded = len(modes) > 1 and rng.random() < pad
    strides = [0] * len(modes)
    s = 1
    for j in order:
        strides[j] = s
        s *= ext[modes[j]]
        if padded:
            s += rng.choice((1, 3, 8, 16))
    return strides


def align8(v, rng):
    """v rounded to a multiple of 8 (small extents kept)."""
    if v < 8:
        return v
    return max(8, int(round(v / 8)) * 8)


def tensor_fields(modes, ext, strides):
    return (
        [str(m) for m in modes]
        + [str(ext[m]) for m in modes]
        + [str(x) for x in strides]
    )


def contraction_spec(
    pid, row, lgm, lgn, lgk, rng, aligned=0.0, padded=0.0, big=False, outer=False
):
    """outer: no contracted mode (K = 1)."""
    dtype, compute, es, cap, (lo, hi) = CROWS[row]
    max_bytes = MAX_TENSOR_BYTES
    if big:
        cap += 7
        max_bytes = 1 << 30
    M = round_extent(2**lgm, rng)
    N = round_extent(2**lgn, rng)
    K = 1 if outer else round_extent(2**lgk, rng)
    # flop cap and tensor size cap: shrink the largest extent
    while math.log2(M * N * K) > cap or max(M * K, N * K, M * N) * es > max_bytes:
        if M >= N and M >= K:
            M = max(1, M * 3 // 4)
        elif N >= K:
            N = max(1, N * 3 // 4)
        else:
            K = max(1, K * 3 // 4)
    nm = rng.choice((1, 1, 2, 2, 3))
    nn = rng.choice((1, 1, 2))
    nk = rng.choice((1, 1, 2, 3))
    if outer:
        nk = 0
    em, en, ek = (
        split_extent(M, nm, rng),
        split_extent(N, nn, rng),
        split_extent(K, nk, rng) if nk else [],
    )
    if rng.random() < aligned:
        em, en, ek = ([align8(e, rng) for e in g] for g in (em, en, ek))
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
    sa = packed_strides(ma, ext, rng, padded)
    sb = packed_strides(mb, ext, rng, padded)
    sd = packed_strides(md, ext, rng, padded)
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


API_DTYPES = {"f16": ("f16", "32F"), "bf16": ("bf16", "32F")}


def _api_split(rng, lg, n, friendly):
    cuts = sorted(rng.uniform(0, lg) for _ in range(n - 1))
    out, prev = [], 0.0
    for c in cuts + [lg]:
        e = max(1, round(2 ** (c - prev)))
        prev = c
        if friendly:
            e = max(8, (e + 4) // 8 * 8)
        elif rng.random() < 0.4:
            e |= 1
        out.append(min(e, 8192))
    return out


def api_spec(pid, seed):
    """The problem the API validation harness draws for `seed` (the same random stream), as a
    spec line: tensors packed in their mode order (hipTensor's column-major descriptors, the
    first mode fastest)."""
    rng = random.Random(seed)
    while True:
        nm, nn, nk = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 2)
        if nm + nk <= 5 and nn + nk <= 5 and nm + nn <= 5:
            break
    friendly = rng.random() < 0.75
    while True:
        lm, ln, lk = rng.uniform(5, 14), rng.uniform(5, 14), rng.uniform(4, 13)
        em = _api_split(rng, lm, nm, friendly)
        en = _api_split(rng, ln, nn, friendly)
        ek = _api_split(rng, lk, nk, friendly)
        M, N, K = math.prod(em), math.prod(en), math.prod(ek)
        if max(M * K, N * K, M * N) <= 2**25 and min(M * K, N * K, M * N) >= 2**12:
            break
    letters = list("abcdefghijklmnopqrstuvwxyz")
    rng.shuffle(letters)
    ms, ns, ks = letters[:nm], letters[nm : nm + nn], letters[nm + nn : nm + nn + nk]
    ext = dict(zip(ms + ns + ks, em + en + ek))

    def order(v):
        v = list(v)
        if rng.random() < 0.8:
            rng.shuffle(v)
        return v

    A, B, D = order(ms + ks), order(ns + ks), order(ms + ns)
    dtype = rng.choice(["f16", "bf16"])
    bil = 1 if rng.choice(["scale", "bilinear"]) == "bilinear" else 0
    code = {m: i + 1 for i, m in enumerate(sorted(ext))}

    def fields(t):
        st, s = [], 1
        for m in t:
            st.append(s)
            s *= ext[m]
        return (
            [str(code[m]) for m in t] + [str(ext[m]) for m in t] + [str(x) for x in st]
        )

    dt, compute = API_DTYPES[dtype]
    f = ["C", str(pid), dt, compute, str(bil), str(len(A)), str(len(B)), str(len(D))]
    return " ".join(f + fields(A) + fields(B) + fields(D))


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
    ap.add_argument("kind", choices=("contraction", "reduction", "refine", "pick"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-row", type=int, default=100)
    ap.add_argument("--rows", default="")
    ap.add_argument("--first-id", type=int, default=0)
    ap.add_argument("--from", dest="src", default="")
    ap.add_argument("--worst", default="")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--aligned", type=float, default=0.0)
    ap.add_argument("--padded", type=float, default=0.0)
    ap.add_argument("--big", action="store_true")
    ap.add_argument(
        "--outer",
        type=float,
        default=0.0,
        help="share of outer products (no contracted mode)",
    )
    ap.add_argument("--style", choices=("halton", "api"), default="halton")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lines = []
    pid = a.first_id
    if a.kind == "contraction" and a.style == "api":
        # --per-row problems of the API harness's generator, seeds seed .. seed + n - 1
        for i in range(a.per_row):
            lines.append(api_spec(pid, a.seed + i))
            pid += 1
    elif a.kind == "contraction":
        rows = a.rows.split(",") if a.rows else list(CROWS)
        for row in rows:
            lo, hi = CROWS[row][4]
            if a.big:
                lo, hi = 9, hi + 2
            shift = [rng.random() for _ in range(3)]
            for i in range(a.per_row):
                u = [(halton(i + 1, PRIMES[d]) + shift[d]) % 1.0 for d in range(3)]
                lg = [lo + (hi - lo) * x for x in u]
                # outer products draw from a stream of their own (a corpus without them is the
                # one of the earlier generator for the same seed)
                outer = (
                    a.outer > 0.0
                    and random.Random(a.seed * 1000003 + pid).random() < a.outer
                )
                lines.append(
                    contraction_spec(
                        pid, row, *lg, rng, a.aligned, a.padded, a.big, outer
                    )
                )
                pid += 1
    elif a.kind == "reduction":
        rows = a.rows.split(",") if a.rows else list(RROWS)
        for row in rows:
            shift = [rng.random() for _ in range(2)]
            for i in range(a.per_row):
                u = [(halton(i + 1, PRIMES[d]) + shift[d]) % 1.0 for d in range(2)]
                lines.append(reduction_spec(pid, row, 8 + 18 * u[0], u[1], rng))
                pid += 1
    elif a.kind == "pick":
        chosen = set(int(x) for x in open(a.worst).read().split())
        for line in open(a.src):
            f = line.split()
            if f and int(f[1]) in chosen:
                lines.append(line.strip())
        pid = a.first_id + len(lines)
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
    if a.kind == "pick":
        print(f"{a.out}: {len(lines)} problems (their ids kept)")
    else:
        print(f"{a.out}: {len(lines)} problems, ids {a.first_id}..{pid - 1}")


if __name__ == "__main__":
    main()
