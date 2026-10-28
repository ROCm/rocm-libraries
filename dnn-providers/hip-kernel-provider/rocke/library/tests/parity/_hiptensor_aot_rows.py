# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/_hiptensor_aot_rows.py -- the hiptensor_aot kernel rows of
# hipTensor's kernel pack (the descriptor tables of hipTensor's pack builder),
# shared by the hiptensor_aot_* parity emitters. The C emitters enumerate the
# same rows in the same order.
_REAL = ("f16", "bf16", "f32", "f64")

EW_ROWS_1 = [
    ("f16", "f16"),
    ("f16", "f32"),
    ("bf16", "bf16"),
    ("bf16", "f32"),
    ("f32", "f32"),
    ("f32", "f16"),
    ("f32", "bf16"),
    ("f64", "f64"),
    ("f16", "f64"),
    ("bf16", "f64"),
    ("f32", "f64"),
]
EW_ROWS_2 = [
    ("f16", "f16", "f16"),
    ("f16", "f16", "f32"),
    ("bf16", "bf16", "bf16"),
    ("bf16", "bf16", "f32"),
    ("f32", "f32", "f32"),
    ("f32", "f16", "f16"),
    ("f32", "bf16", "bf16"),
    ("f64", "f64", "f64"),
    ("f64", "f16", "f16"),
    ("f64", "bf16", "bf16"),
    ("f64", "f32", "f32"),
]
EW_ROWS_3 = [
    ("f16", "f16", "f16", "f16"),
    ("f16", "f16", "f16", "f32"),
    ("bf16", "bf16", "bf16", "bf16"),
    ("bf16", "bf16", "bf16", "f32"),
    ("f32", "f32", "f32", "f32"),
    ("f64", "f64", "f64", "f64"),
]
EW_ROWS_C1 = [("f32", "f32"), ("f64", "f64"), ("f32", "f64")]
EW_ROWS_C2 = [("f32", "f32", "f32"), ("f64", "f64", "f64"), ("f64", "f32", "f32")]
EW_ROWS_1_MIXED = [
    (a, d) for d in ("f16", "bf16") for a in _REAL if (a, d) not in EW_ROWS_1
]
EW_ROWS_2_MIXED = [
    (t, c, d)
    for t in ("f32", "f64")
    for c in _REAL
    for d in _REAL
    if (t, c, d) not in EW_ROWS_2
]
EW_ROWS_C2_MIXED = [
    (t, c, d)
    for t in ("f32", "f64")
    for c in ("f32", "f64")
    for d in ("f32", "f64")
    if (t, c, d) not in EW_ROWS_C2
]
REDUCE_ROWS = [("f16", "f16"), ("bf16", "bf16"), ("f32", "f32"), ("f64", "f64")]


def ew_params(idx64, mixed=True):
    """The EwSpec arguments of the generic elementwise rows (the pack's order)."""
    rows1 = EW_ROWS_1 + (EW_ROWS_1_MIXED if mixed else [])
    rows2 = EW_ROWS_2 + (EW_ROWS_2_MIXED if mixed else [])
    rowsc2 = EW_ROWS_C2 + (EW_ROWS_C2_MIXED if mixed else [])
    out = []
    for a, d in rows1:
        out.append(dict(arity=1, a=a, d=d, idx64=idx64))
    for a, c, d in rows2:
        out.append(dict(arity=2, a=a, c=c, d=d, idx64=idx64))
    for a, bb, c, d in EW_ROWS_3:
        out.append(dict(arity=3, a=a, b=bb, c=c, d=d, idx64=idx64))
    for a, d in EW_ROWS_C1:
        out.append(dict(arity=1, a=a, d=d, cplx=True, idx64=idx64))
    for a, c, d in rowsc2:
        out.append(dict(arity=2, a=a, c=c, d=d, cplx=True, idx64=idx64))
    out += [dict(p, math=True) for p in out if not p.get("cplx")]
    return out


def ew_rows():
    return [p for idx64 in (False, True) for p in ew_params(idx64)]


def reduce_rows():
    out = []
    for idx64 in (False, True):
        for math in (False, True):
            for a, d in REDUCE_ROWS:
                for mode in ("row", "col", "final"):
                    out.append(dict(mode=mode, a=a, d=d, idx64=idx64, math=math))
    return out


def plain_rows():
    """The real rows of the API's elementwise operations (32-bit offsets, no math)."""
    return [
        p
        for p in ew_params(False, mixed=False)
        if not p.get("cplx") and not p.get("math")
    ]


def ewt_rows():
    """Every plain row, then its folded class (several small planes per tile)."""
    keys = ("arity", "a", "b", "c", "d")
    out = []
    for p in plain_rows():
        for fold in (False, True):
            q = {k: v for k, v in p.items() if k in keys}
            if fold:
                q["fold"] = True
            out.append(q)
    return out


def ewv_rows():
    """(base, vec, unit) of the pack's vector class kernels."""
    from kernels.common.hiptensor_aot.ew import EwSpec
    from kernels.common.hiptensor_aot.jit_ew import jit_classes

    out = []
    for p in plain_rows():
        for c in jit_classes([EwSpec(**p)]):
            if c.vec > 1:
                out.append((p, c.vec, tuple(c.unit)))
    return out


def jit_rows():
    """(base, vec, unit) of every JIT class of the generic rows (32- and 64-bit offsets, math
    rows included)."""
    from kernels.common.hiptensor_aot.ew import EwSpec
    from kernels.common.hiptensor_aot.jit_ew import jit_classes

    out = []
    for idx64 in (False, True):
        for p in ew_params(idx64, mixed=False):
            for c in jit_classes([EwSpec(**p)]):
                out.append((p, c.vec, tuple(c.unit)))
    return out


def geometry(names, seed):
    """Distinct geometry values for ``names`` (the C emitters draw the same values from the
    same 64-bit LCG)."""
    state = (seed * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    out = {}
    for n in names:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out[n] = 1000 + (state >> 33) % ((1 << 30) - 1000)
    return out
