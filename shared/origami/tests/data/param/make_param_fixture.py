#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Writes the tiny param_v1 test model used by test_nn.cpp ([nn][param]); numpy + msgpack only.

  python3 make_param_fixture.py        -> param_fixture.msgpack, param_fixture_rows.txt, param_fixture_expected.txt

The model is random (seeded): 8 features, one hidden layer of 8, the 12 heads with 2-3 values each, 12
catalog rows (row 5 has a WGM value outside the head's vocabulary, so it scores unk_logp on that head).
Scoring follows the param_v1 contract: score(row) = sum_h head_weight[h] * log_softmax_h[value index],
order = score descending, ties by ascending row.
"""
import math
from pathlib import Path

import msgpack
import numpy as np

FEATURES = ["log2m", "log2n", "log2k", "log2b", "log2_tiles_64x64", "waveeff_64x64_cu40", "kmod64_is0", "b_gt1"]
HEADS = {"MT0": [32, 64, 128], "MT1": [32, 64, 128], "DU": [16, 32, 64], "GSU": [0, 1, 2], "WGM": [1, 8],
         "SU": [0, 32], "SUM": [0, 1], "SUS": [0, 2], "PGR": [1, 2], "TLDS": [0, 1], "LDSB": [0, 1], "DTVB": [0, 1]}
UNKNOWN_WGM = 4
HERE = Path(__file__).resolve().parent


def cdiv(a, c):
    return (a + c - 1) // c


def feature(name, m, n, k, b):
    t = cdiv(m, 64) * cdiv(n, 64) * b
    return {"log2m": math.log2(m), "log2n": math.log2(n), "log2k": math.log2(k), "log2b": math.log2(b),
            "log2_tiles_64x64": math.log2(t), "waveeff_64x64_cu40": t / (cdiv(t, 40) * 40),
            "kmod64_is0": float(k % 64 == 0), "b_gt1": float(b > 1)}[name]


def main():
    rng = np.random.default_rng(20261116)
    heads = list(HEADS)
    sizes = [len(HEADS[h]) for h in heads]
    off = np.cumsum([0] + sizes)
    F, H, O = len(FEATURES), 8, int(sum(sizes))
    mu = rng.normal(8.0, 2.0, F).astype(np.float32)
    sd = rng.uniform(0.5, 3.0, F).astype(np.float32)
    sd[6] = 0.0  # sd == 0 is read as 1
    W0, b0 = rng.normal(0, 0.7, (H, F)).astype(np.float32), rng.normal(0, 0.3, H).astype(np.float32)
    W1, b1 = rng.normal(0, 0.9, (O, H)).astype(np.float32), rng.normal(0, 0.3, O).astype(np.float32)
    rows = rng.integers(0, 3, (12, len(heads)))
    for j, s in enumerate(sizes):
        rows[:, j] %= s
    rows[5, heads.index("WGM")] = -1
    f32 = lambda x: [float(v) for v in np.asarray(x, dtype=np.float32).ravel()]
    d = {"version": 1, "problem_key": "gfx1151_Cijk_Alik_Bljk_TEST",
         "features": {"names": FEATURES, "mu": f32(mu), "sd": f32(sd)},
         "layers": [{"in": F, "out": H, "W": f32(W0), "b": f32(b0), "act": "relu"},
                    {"in": H, "out": O, "W": f32(W1), "b": f32(b1), "act": "none"}],
         "heads": [{"name": h, "offset": int(off[j]), "size": int(sizes[j])} for j, h in enumerate(heads)],
         "head_weight": [1.0] * len(heads), "unk_logp": -20.0, "solutions": rows.tolist(),
         "solution_names": [f"row{i}" for i in range(len(rows))], "topk_default": 1,
         "head_values": HEADS, "tie_break": "ascending SolutionIndex (stable sort by -score)", "meta": {}}

    sd64 = np.where(sd == 0, 1.0, sd.astype(np.float64))
    lay = [(W0.astype(np.float64), b0.astype(np.float64), True), (W1.astype(np.float64), b1.astype(np.float64), False)]

    def order(m, n, k, b):
        x = (np.array([feature(nm, m, n, k, b) for nm in FEATURES]) - mu.astype(np.float64)) / sd64
        h = x
        for W, bb, relu in lay:
            h = W @ h + bb
            h = np.maximum(h, 0.0) if relu else h
        s = np.zeros(len(rows))
        for j in range(len(heads)):
            z = h[off[j]:off[j + 1]]
            z = z - z.max()
            lp = z - math.log(np.exp(z).sum())
            s += np.where(rows[:, j] >= 0, lp[np.maximum(rows[:, j], 0)], d["unk_logp"])
        return [int(i) for i in np.lexsort((np.arange(len(s)), -s))], x, s

    shapes = [(1, 1, 1, 1), (4096, 4096, 4096, 1), (17, 33, 65, 3), (3000, 128, 7000, 1), (64, 64, 64, 64),
              (8192, 1024, 256, 1), (100, 20000, 300, 2), (5, 5000, 5, 1)]
    tvs, expected = [], []
    for m, n, k, b in shapes:
        o, x, s = order(m, n, k, b)
        tvs.append({"m": m, "n": n, "k": k, "b": b, "x": [float(v) for v in x], "order": o})
        expected.append(" ".join(map(str, [m, n, k, b] + o)))
        assert len(set(np.round(s, 9))) == len(s), "fixture scores must not tie"
    d["test_vectors"] = tvs
    (HERE / "param_fixture.msgpack").write_bytes(msgpack.packb(d, use_bin_type=True))
    vals = [" ".join(str(HEADS[h][r[j]] if r[j] >= 0 else UNKNOWN_WGM) for j, h in enumerate(heads)) for r in rows]
    (HERE / "param_fixture_rows.txt").write_text("# " + " ".join(heads) + "\n" + "\n".join(vals) + "\n")
    (HERE / "param_fixture_expected.txt").write_text("# m n k b, then the rows best first\n" + "\n".join(expected) + "\n")


if __name__ == "__main__":
    main()
