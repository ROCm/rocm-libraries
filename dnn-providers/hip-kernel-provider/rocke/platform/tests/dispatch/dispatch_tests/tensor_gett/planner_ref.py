# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reference lines of the Python tensor GETT planner / GEMM dispatcher for the C++ mirror
(rocke/tensor_gett_planner.h, rocke/gemm_dispatch.h), the random problem generators, and the
build of the C++ driver (tensor_gett_planner_cli.cpp) against the engine archive.

The line formats are the ones tensor_gett_planner_cli prints; test_tensor_gett_planner_cpp.py
compares the two engines line by line. ``python planner_ref.py ref <problems.jsonl> <out>`` writes
the Python lines of a problem file (used to shard the comparison over processes)."""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLATFORM = HERE.parents[3]
CLI_SRC = HERE / "tensor_gett_planner_cli.cpp"
MODEL_DIR = PLATFORM / "python" / "rocke" / "dispatch" / "gemm" / "selector_models"

#: environment overrides of the Python dispatcher / planner that the C++ mirror does not read
#: (it is the Python code at their defaults); the comparison runs without them
PY_ONLY_ENV = (
    "ROCKE_TENSOR_GETT_ROUTE",
    "ROCKE_GEMM_SELECTOR",
    "ROCKE_GEMM_SELECTOR_NATIVE",
    "ROCKE_GEMM_SELECTOR_MODEL_DIR",
    "ROCKE_GEMM_SKINNY_SELECT",
    "ROCKE_GEMM_SKINNY_MODEL_DIR",
    "ROCKE_GEMM_SPLIT_K",
    "ROCKE_GEMM_DTL_ADMIT",
    "ROCKE_GEMM_RILV_BODY",
    "ROCKE_GEMM_RILV_HB_LOOP",
)

_NAMES = "abcdefghijklmnopqrstuvwxyz"
#: CU counts of the comparison (full devices and partitions)
CUS = (256, 256, 256, 128, 64, 32, 240, 304, 80)


def g17(x: float) -> str:
    return format(x, ".17g")


# --------------------------------------------------------------------------- generators


def _split(total_log2, n, rng):
    if n == 1:
        cuts = [total_log2]
    else:
        cuts = sorted(rng.uniform(0, total_log2) for _ in range(n - 1))
        cuts = [b - a for a, b in zip([0] + cuts, cuts + [total_log2])]
    out = []
    for c in cuts:
        e = max(1, int(round(2**c)))
        if e == 1 and rng.random() < 0.7:
            e = rng.choice((2, 4, 8, 3))
        r = rng.random()
        if r < 0.45:
            q = rng.choice((8, 16, 32, 64))
            e = max(q, (e + q // 2) // q * q) if e >= q // 2 else e
        elif r < 0.6:
            e = e | 1
        out.append(e)
    return out


def _strides(modes, ext, rng):
    """None (packed row-major) or the element strides of a random memory order with padding."""
    if rng.random() < 0.55:
        return None
    order = list(modes)
    if rng.random() < 0.5:
        rng.shuffle(order)
    st, acc = {}, 1
    for m in reversed(order):
        st[m] = acc
        acc *= ext[m]
        if rng.random() < 0.3:
            acc += rng.choice((1, 2, 8, 16, 64, 3))
    if rng.random() < 0.05:
        k = rng.choice(order)
        st[k] = st[k] * 2
    return st


def contraction(rng, arch: str = "gfx950") -> dict:
    """A random contraction: 1-3 M / N / K modes (tensor ranks 2-6), random mode orders and
    padded strides, sizes from tiny to large, f16 / bf16, scale / bilinear, a CU count; a few
    with a batch mode (no GETT route)."""
    while True:
        nm, nn, nk = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)
        if nm + nk <= 6 and nn + nk <= 6 and nm + nn <= 6 and nm + nk >= 2:
            break
    names = list(_NAMES)
    rng.shuffle(names)
    M = names[:nm]
    N = names[nm : nm + nn]
    K = names[nm + nn : nm + nn + nk]
    batch = names[nm + nn + nk : nm + nn + nk + 1] if rng.random() < 0.03 else []
    big = rng.random() < 0.08
    lo = 0 if rng.random() < 0.15 else 4
    lm = rng.uniform(lo, 15.5 if big else 13.5)
    ln = rng.uniform(lo, 15.5 if big else 13.5)
    lk = rng.uniform(lo, 15 if big else 13)
    ext = {}
    for ms, lg in ((M, lm), (N, ln), (K, lk)):
        for m, e in zip(ms, _split(lg, len(ms), rng)):
            ext[m] = e
    for b in batch:
        ext[b] = rng.randint(1, 8)
    ma, mb, md = M + K + batch, N + K + batch, M + N + batch
    for lst in (ma, mb, md):
        if rng.random() < 0.8:
            rng.shuffle(lst)
    return dict(
        a=ma,
        b=mb,
        d=md,
        ext=ext,
        sa=_strides(ma, ext, rng),
        sb=_strides(mb, ext, rng),
        sd=_strides(md, ext, rng),
        dtype=rng.choice(("f16", "bf16")),
        bilinear=rng.random() < 0.5,
        cus=rng.choice(CUS),
        arch=arch,
    )


def contraction_kc_own(rng, arch: str = "gfx950") -> dict:
    """A random contraction whose operands are (or are one stride away from) the K-contiguous
    copies the K-contiguous GETT reads: packed in kc_operand_modes order, then, half the time,
    one mode's stride doubled (often an extent-1 / extent-2 mode)."""
    import rocke.dispatch.families.tensor_gett as TG
    from rocke.instances.common.tensor import (
        GettKnobs,
        TensorGettSpec,
        kc_operand_modes,
        launch_operands,
    )

    for _ in range(50):
        p = contraction(rng, arch)
        if rng.random() < 0.5:
            m = rng.choice(sorted(p["ext"]))
            p["ext"][m] = rng.choice((1, 2))
        spec = TensorGettSpec.make(
            p["a"],
            p["b"],
            p["d"],
            p["ext"],
            dtype=p["dtype"],
            knobs=GettKnobs(bilinear=p["bilinear"]),
        )
        try:
            sp = TG._spec_kc(TG.TensorGettRequest(problem=spec, arch="gfx950"))
        except Exception:  # noqa: BLE001 - no K-contiguous spec
            sp = None
        if sp is None:
            continue
        for nm, order in zip(launch_operands(sp), kc_operand_modes(sp)):
            st, acc = {}, 1
            for m in reversed(order):
                st[m] = acc
                acc *= p["ext"][m]
            if rng.random() < 0.5:
                small = [m for m in order if p["ext"][m] <= 2]
                m = rng.choice(small if small and rng.random() < 0.7 else list(order))
                st[m] *= 2
            p["sa" if nm == "A" else "sb"] = st
        return p
    return contraction(rng, arch)


def contraction_edge(rng, arch: str = "gfx950") -> dict:
    """A contraction on a planner threshold: the 256 x 256 tile count of its GEMM view one
    below, at or one above the persistent MN body's (TENSOR_GETT_MN_PERSISTENT_MIN_WAVES x
    CUs) or the CU count, K at the K-contiguous short-run bound (TENSOR_GETT_KC_SHORT_RUN_MIN_K
    +- 0 / 64) or random, and D's fastest run 64 / 128 / 256 (TENSOR_GETT_KC_MIN_D_RUN).
    """
    import rocke.dispatch.families.tensor_gett as TG

    cus = rng.choice(CUS)
    waves = int(TG.TENSOR_GETT_MN_PERSISTENT_MIN_WAVES)
    t = max(1, rng.choice((waves * cus, waves * cus, cus)) + rng.choice((-1, 0, 0, 1)))
    a = max(1, int(round(t ** rng.uniform(0.3, 0.7))))
    b = -(-t // a)
    names = list(_NAMES)
    rng.shuffle(names)
    ext = {}

    def split(total, ms):
        # total = prod(extents): the fastest mode takes a power of two dividing it
        inner = [
            f for f in (8, 16, 32, 64, 128, 256) if total % f == 0 and total // f >= 1
        ]
        f = rng.choice(inner) if inner and len(ms) > 1 else total
        ext[ms[-1]] = f
        if len(ms) > 1:
            ext[ms[0]] = total // f

    mt = a * 256 - rng.choice((0, 0, 128))
    nt = b * 256 - rng.choice((0, 0, 128))
    kmin = int(TG.TENSOR_GETT_KC_SHORT_RUN_MIN_K)
    kt = rng.choice(
        (kmin, kmin - 64, kmin + 64, rng.choice((256, 512, 1024, 2048, 8192)))
    )
    M = names[0 : rng.choice((1, 2))]
    N = names[2 : 2 + rng.choice((1, 2))]
    K = names[4 : 4 + rng.choice((1, 2))]
    split(mt, M)
    split(nt, N)
    split(kt, K)
    ma, mb, md = M + K, N + K, M + N
    for lst in (ma, mb, md):
        if rng.random() < 0.5:
            rng.shuffle(lst)
    return dict(
        a=ma,
        b=mb,
        d=md,
        ext=ext,
        sa=None,
        sb=None,
        sd=None,
        dtype=rng.choice(("f16", "bf16")),
        bilinear=rng.random() < 0.5,
        cus=cus,
        arch=arch,
    )


def gemm(rng, arch: str = "gfx950") -> dict:
    """A random GEMM request of the layouts the planner issues (CRR, RCR; RRR / CCR rarely)."""

    def dim(lo, hi):
        e = max(1, int(round(2 ** rng.uniform(lo, hi))))
        r = rng.random()
        if r < 0.5:
            q = rng.choice((8, 16, 64, 128, 256))
            e = max(q, (e + q // 2) // q * q)
        elif r < 0.6:
            e |= 1
        return e

    return dict(
        M=dim(0, 15),
        N=dim(0, 15),
        K=dim(2, 15),
        dtype=rng.choice(("fp16", "bf16")),
        layout=rng.choice(("CRR", "CRR", "RCR", "RCR", "RRR", "CCR")),
        cus=rng.choice(CUS),
        arch=arch,
    )


# rule thresholds of dispatch/gemm (common.py / skinny.py) the edge generator lands on: output
# tile counts (256 x 256 / 128 x 128 tiles) and K values
_EDGE_TILES = (
    "RILV_MIN_TILES",
    "RILV_HB_BIG_TILES",
    "RILV_PAD_FAST_TILE_MIN_TILES",
    "DTL_RING_SMALL_RAGGED_MAX_TILES",
    "DTL_BIG_GRID_TILES",
    "DTL_BIG_GRID_WIDE_TILES",
    "DTL_BIG_GRID_RAGGED_WIDE_TILES",
    "G1_DTL_K_TAIL_MIN_TILES",
    "G1_RING_ANY_SPLIT_RULE[0]",
    "RING_ANY_SPLIT_FEW_TILES",
    "R3_RING_SPLIT_MIN_TILES",
    "R3_RING_SPLIT_MAX_TILES",
)
_EDGE_K = (
    "RILV_MIN_K",
    "RILV_HB_MIN_K",
    "RILV_HB_BIG_MIN_K",
    "RILV_PERSIST_MAX_K",
    "RILV_SHORTK_MIN_K",
    "SMALL_TILE_SHORTK_MIN_K",
    "SMALL_TILE_SHORTK_MAX_K",
    "DTL_RING_SMALL_K32_MIN_K",
    "DTL_RING_SMALL_K32_AGPR_MIN_K",
    "DTL_RING_SMALL_RAGGED_MAX_K",
    "DTL_RING_SMALL_S3_MIN_K",
    "DTL_BIG_GRID_SMALL_K",
    "DTL_BIG_GRID_WIDE_MIN_K",
    "R3_KPROXY_MIN_K",
    "R3_QGEO_MIN_K",
    "G1_RING_ANY_SPLIT_RULE[1]",
    "RING_ANY_SPLIT_MIN_K",
    "RING_ANY_SPLIT_FEW_MIN_K",
    "R3_RING_SPLIT_MIN_K",
)


def _edge_values():
    from rocke.dispatch.gemm import common, skinny

    env = {**vars(skinny), **vars(common)}
    tiles = sorted(
        {int(eval(e, {}, env)) for e in _EDGE_TILES}
    )  # noqa: S307 - fixed names
    ks = sorted({int(eval(e, {}, env)) for e in _EDGE_K})  # noqa: S307
    return tiles, ks


def gemm_edge(rng, arch: str = "gfx950", edges=None) -> dict:
    """A GEMM request on a rule threshold: a tile count (of 256 x 256 or 128 x 128 tiles, also
    the CU count and twice it) one below, at or one above a threshold, ragged or whole tiles, and
    K at a K threshold (+- 0 / 8 / 64)."""
    tiles, ks = edges or _edge_values()
    cus = rng.choice(CUS)
    t = rng.choice(tiles + [cus, 2 * cus]) + rng.choice((-1, 0, 0, 1))
    t = max(1, t)
    a = max(1, int(round(t ** rng.uniform(0.2, 0.8))))
    b = max(1, -(-t // a) if rng.random() < 0.5 else t // a)
    T = rng.choice((256, 256, 128))
    M = a * T + rng.choice((0, 0, 0, 0, 1, -1, 8, -8, -64, -T // 2))
    N = b * T + rng.choice((0, 0, 0, 0, 1, -1, 8, -8, -64, -T // 2))
    if rng.random() < 0.5:
        M, N = N, M
    K = max(8, rng.choice(ks) + rng.choice((0, 0, -8, 8, -64, 64, -1)))
    return dict(
        M=max(1, M),
        N=max(1, N),
        K=K,
        dtype=rng.choice(("fp16", "bf16")),
        layout=rng.choice(("CRR", "RCR", "RCR", "RRR", "CCR")),
        cus=cus,
        arch=arch,
    )


def _spec_types():
    import dataclasses
    import typing

    from rocke.instances.common.gemm_universal import TraitSpec

    hints = typing.get_type_hints(TraitSpec)
    out = {}
    for f in dataclasses.fields(TraitSpec):
        t = hints[f.name]
        args = typing.get_args(t)
        if type(None) in args:
            out[f.name] = "opt"
        elif t is bool:
            out[f.name] = "bool"
        elif t is int:
            out[f.name] = "int"
        else:
            out[f.name] = "str"
    return out


def spec_pool(n: int = 80, seed: int = 4242) -> list:
    """Config strings of the Python dispatcher's universal picks on random requests and on
    requests at its rule thresholds (the base specs of :func:`spec_problem`)."""
    from rocke.dispatch.gemm import GemmRequest, dispatch_gemm_bf16, dispatch_gemm_fp16
    from rocke.dispatch.gemm.selector import config_string
    from rocke.instances.common.gemm_skinny import SkinnyGemmSpec

    rng = random.Random(seed)
    edges = _edge_values()
    out = []
    for i in range(2 * n):
        p = gemm(rng) if i % 2 == 0 else gemm_edge(rng, "gfx950", edges)
        lay = p["layout"]
        req = GemmRequest(
            M=p["M"],
            N=p["N"],
            K=p["K"],
            arch="gfx950",
            dtype=p["dtype"],
            layout=lay,
            trans_a=lay[0] == "C",
            trans_b=lay[1] == "C",
        )
        disp = dispatch_gemm_bf16 if p["dtype"] == "bf16" else dispatch_gemm_fp16
        try:
            res = disp(req)
        except Exception:  # noqa: BLE001 - no candidate
            continue
        if not isinstance(res.spec, SkinnyGemmSpec):
            cfg = config_string(res.spec)
            if cfg not in out:
                out.append(cfg)
    return out


_TILE_VALUES = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 512)
_INT_VALUES = (0, 1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 256, 1024)


def spec_problem(rng, pool: list, types=None) -> dict:
    """A gemm_universal spec near the dispatcher's picks: a pool config with 0-4 tile / trait
    fields changed (half of them levers the pool config sets; bools flipped, ints to small /
    power-of-two / neighbour values, strings to a value another pool config uses), a dtype,
    layout and launch shape."""
    from rocke.dispatch.gemm.selector import config_string, parse_config
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    types = types or _spec_types()
    strs = {}
    for cfg in pool:
        for k, v in parse_config(cfg)[1].items():
            if types.get(k) == "str":
                strs.setdefault(k, set()).add(v)
    while True:
        tile, trait = parse_config(rng.choice(pool))
        for _ in range(rng.choice((0, 1, 1, 2, 2, 3, 4))):
            if rng.random() < 0.25:
                k = rng.choice(sorted(tile))
                v = tile[k]
                tile[k] = rng.choice(_TILE_VALUES + (v * 2, max(1, v // 2)))
                continue
            # half the changes move a lever the base spec sets (to a neighbour value)
            near = bool(trait) and rng.random() < 0.5
            k = rng.choice(sorted(trait)) if near else rng.choice(sorted(types))
            kind = types[k]
            v = trait.get(k, getattr(TraitSpec(), k))
            if kind == "bool":
                trait[k] = not v
            elif kind == "int" and near:
                trait[k] = rng.choice((v + 1, max(0, v - 1), v * 2, v // 2, 0, 1))
            elif kind == "int":
                trait[k] = rng.choice(_INT_VALUES + (v + 1, max(0, v - 1), v * 2))
            elif kind == "opt":
                trait[k] = rng.choice((None, 0, 1, 2, 4, 8))
            elif strs.get(k):
                trait[k] = rng.choice(sorted(strs[k]) + [getattr(TraitSpec(), k)])
        dt = rng.choice(("fp16", "bf16"))
        lay = rng.choice(("RCR", "RCR", "CRR", "RRR", "CCR"))
        try:
            spec = UniversalGemmSpec(
                name="selector_library",
                tile=TileSpec(**tile),
                trait=TraitSpec(**trait),
                data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout=lay),
                wave_size=64,
            )
        except Exception:  # noqa: BLE001 - the dataclasses refuse it: not a spec
            continue
        g = gemm(rng)
        return dict(
            cfg=config_string(spec),
            dtype=dt,
            layout=lay,
            M=g["M"],
            N=g["N"],
            K=g["K"],
            cus=256,
            arch="gfx950",
        )


def _models(arch: str) -> dict:
    """{kind: [model, bag...]} of the shipped models of ``arch`` (Python objects)."""
    from rocke.dispatch.gemm import selector as S
    from rocke.dispatch.gemm import skinny_select as SS

    out = {}
    for kind in ("rcr", "native"):
        m = S.load_model(arch, kind)
        if m is not None:
            out[kind] = [m.ensemble] + list(m.bags)
    sk = SS.load(arch)
    if sk is not None:
        out["skinny"] = [sk[1]] + list(sk[2])
    return out


def _tree_values(ens) -> dict:
    """{feature index: sorted split thresholds} of a TreeEnsemble or a skinny _Packed model."""
    vals: dict = {}
    trees = getattr(ens, "trees", None)
    if trees is not None:
        for t in trees:
            if t.num_leaves > 1:
                for f, th in zip(t.split_feature, t.threshold):
                    vals.setdefault(int(f), set()).add(float(th))
    else:
        for i in range(ens.sf.shape[0]):
            if ens.single[i]:
                continue
            for f, th, lc in zip(ens.sf[i], ens.th[i], ens.lc[i]):
                if lc != -1 or th != 0.0:
                    vals.setdefault(int(f), set()).add(float(th))
    return {f: sorted(v) for f, v in vals.items()}


def synthetic_models(work: Path, seed: int = 9507) -> list:
    """Copies of two shipped models (a selector and a skinny model) with every split's
    decision_type drawn from 0 / 2 / 4 / 6 / 8 / 10 (missing type none, zero or NaN; default
    left or right): the shipped models only use one, so these walk the other branches.
    """
    rng = random.Random(seed)
    out = []
    for name in ("gemm_rcr_gfx950.model.txt", "skinny_cfg.main.model.txt"):
        lines = []
        for ln in (MODEL_DIR / name).read_text(encoding="utf-8").splitlines():
            if ln.startswith("decision_type="):
                n = len(ln.split("=", 1)[1].split())
                ln = "decision_type=" + " ".join(
                    str(rng.choice((0, 2, 4, 6, 8, 10))) for _ in range(n)
                )
            lines.append(ln)
        path = Path(work) / ("synthetic_" + name)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out.append(str(path))
    return out


_FILE_MODELS: dict = {}


def _file_model(path: str):
    from rocke.dispatch.gemm.selector import TreeEnsemble
    from rocke.dispatch.gemm.skinny_select import _Packed

    if path not in _FILE_MODELS:
        ens = TreeEnsemble.from_text(Path(path).read_text(encoding="utf-8"))
        _FILE_MODELS[path] = (ens, _Packed(ens))
    return _FILE_MODELS[path]


def tree_problems(rng, n: int, arch: str = "gfx950", files=()) -> list:
    """Feature rows for every shipped model of ``arch`` (model and bags) and for the model
    files ``files`` (scored both ways): each feature on a split threshold of that feature,
    just above it, between two, outside the range, or zero / a sub-1e-35 value (LightGBM's
    missing-as-zero branch)."""
    import math

    models = _models(arch)
    out = []
    keys = [(k, b) for k, ms in sorted(models.items()) for b in range(-1, len(ms) - 1)]
    keys += [("file", f) for f in files]
    for i in range(n):
        kind, bag = keys[i % len(keys)]
        ens = _file_model(bag)[0] if kind == "file" else models[kind][bag + 1]
        vals = _tree_values(ens)
        width = max(vals) + 1 if vals else 1
        x = []
        for f in range(width):
            ths = vals.get(f) or [0.0]
            r = rng.random()
            th = rng.choice(ths)
            if r < 0.3:
                v = th
            elif r < 0.5:
                v = math.nextafter(th, math.inf)
            elif r < 0.75:
                j = rng.randrange(len(ths))
                v = (ths[j] + ths[min(j + 1, len(ths) - 1)]) / 2
            elif r < 0.85:
                v = rng.choice(
                    (ths[0] - 1.0 - abs(ths[0]), ths[-1] + 1.0 + abs(ths[-1]))
                )
            else:
                v = rng.choice((0.0, 1e-36, -1e-36))
            x.append(v)
        if kind == "file":
            out.append(
                dict(
                    arch=arch,
                    model="file",
                    file=bag,
                    packed=rng.random() < 0.5,
                    x=x,
                    cus=256,
                )
            )
        else:
            out.append(dict(arch=arch, model=kind, bag=bag, x=x, cus=256))
    return out


def tree_line(p: dict) -> str:
    """The model's score of a feature row (tensor_gett_planner_cli --trees)."""
    if p.get("file"):
        ens, packed = _file_model(p["file"])
        if p.get("packed"):
            return g17(float(packed.predict([p["x"]])[0]))
        return g17(float(ens.predict_one(p["x"])))
    models = _models(p["arch"])
    ms = models.get(p["model"])
    if ms is None or p["bag"] + 1 >= len(ms):
        return "-"
    ens = ms[p["bag"] + 1]
    if p["model"] == "skinny":
        return g17(float(ens.predict([p["x"]])[0]))
    return g17(float(ens.predict_one(p["x"])))


def spec_neighbours(pool: list, rng, types=None) -> list:
    """Every one-lever step of every pool config: each lever it sets moved to its neighbour
    values (bools flipped, ints +-1 / x2 / /2 / 0 / 1, strings to the other pool values) and
    each tile field doubled / halved; with a random dtype, layout and launch shape."""
    from rocke.dispatch.gemm.selector import config_string, parse_config
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    types = types or _spec_types()
    strs: dict = {}
    for cfg in pool:
        for k, v in parse_config(cfg)[1].items():
            if types.get(k) == "str":
                strs.setdefault(k, set()).add(v)
    out = []
    for cfg in pool:
        tile, trait = parse_config(cfg)
        steps = []
        for k, v in sorted(tile.items()):
            steps += [
                ({**tile, k: x}, trait)
                for x in sorted({v * 2, max(1, v // 2)})
                if x != v
            ]
        for k, v in sorted(trait.items()):
            kind = types.get(k)
            if kind == "bool":
                vals = [not v]
            elif kind == "int":
                vals = sorted({v + 1, max(0, v - 1), v * 2, v // 2, 0, 1} - {v})
            elif kind == "opt":
                vals = [x for x in (None, 0, 1, 2) if x != v]
            else:
                vals = sorted(strs.get(k, set()) - {v})
            steps += [(tile, {**trait, k: x}) for x in vals]
        for t, tr in steps:
            dt = rng.choice(("fp16", "bf16"))
            lay = rng.choice(("RCR", "RCR", "CRR", "RRR", "CCR"))
            try:
                spec = UniversalGemmSpec(
                    name="selector_library",
                    tile=TileSpec(**t),
                    trait=TraitSpec(**tr),
                    data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout=lay),
                    wave_size=64,
                )
            except Exception:  # noqa: BLE001 - the dataclasses refuse it: not a spec
                continue
            g = gemm(rng)
            out.append(
                dict(
                    cfg=config_string(spec),
                    dtype=dt,
                    layout=lay,
                    M=g["M"],
                    N=g["N"],
                    K=g["K"],
                    cus=256,
                    arch="gfx950",
                )
            )
    return out


def valid_line(p: dict) -> str:
    """is_valid_spec and, for a valid spec, launch_shape_error (tensor_gett_planner_cli
    --valid)."""
    from rocke.dispatch.gemm.selector import parse_config
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        is_valid_spec,
        launch_shape_error,
    )

    tile, trait = parse_config(p["cfg"])
    dt = p["dtype"]
    spec = UniversalGemmSpec(
        name="selector_library",
        tile=TileSpec(**tile),
        trait=TraitSpec(**trait),
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout=p["layout"]),
        wave_size=64,
    )
    ok = is_valid_spec(spec, arch=p["arch"])[0]
    if not ok:
        # launch_shape_error is defined on valid specs only
        return "valid=0 launch=-"
    launch = launch_shape_error(spec, p["M"], p["N"], p["K"]) is None
    return f"valid=1 launch={int(launch)}"


def table_problems() -> list:
    """The problems of the shipped AOT and tuned GETT route tables (gfx950)."""
    out = []
    for fname in ("tensor_gett_aot.json", "tensor_gett_tuned.json"):
        path = PLATFORM / "python" / "rocke" / "dispatch" / "families" / fname
        data = json.loads(path.read_text(encoding="utf-8")).get("entries", {})
        for arch, table in sorted(data.items()):
            for _key, e in sorted(table.items()):
                p = e["problem"]
                out.append(
                    dict(
                        a=list(p["modes_a"]),
                        b=list(p["modes_b"]),
                        d=list(p["modes_d"]),
                        ext={m: int(v) for m, v in p["extents"].items()},
                        sa=None,
                        sb=None,
                        sd=None,
                        dtype=p["dtype"],
                        bilinear=bool(p["bilinear"]),
                        cus=256,
                        arch=arch,
                    )
                )
    return out


# --------------------------------------------------------------------------- Python lines


def _set_cus(arch: str, cus: int):
    from rocke.dispatch.gemm import selector as S
    from rocke.helpers import split_k as SK

    SK._ARCH_NUM_CUS[arch] = int(cus)
    # the decision cache is keyed without the CU count
    S._DECISION_CACHE.clear()


def contraction_line(p: dict) -> str:
    """The Python planner's line for one problem (the format of tensor_gett_planner_cli)."""
    import rocke.dispatch.families.tensor_gett as TG
    from rocke.instances.common.tensor import GettKnobs, TensorGettSpec

    _set_cus(p["arch"], p["cus"])
    spec = TensorGettSpec.make(
        p["a"],
        p["b"],
        p["d"],
        p["ext"],
        strides_a=p["sa"],
        strides_b=p["sb"],
        strides_d=p["sd"],
        dtype=p["dtype"],
        knobs=GettKnobs(bilinear=p["bilinear"]),
    )
    req = TG.TensorGettRequest(problem=spec, arch=p["arch"])
    try:
        plan = TG.plan_tensor_gett(req)
    except Exception:  # noqa: BLE001 - the planner raises: no route
        return "none"
    tm = tk = "-"
    if "tuned route table" not in plan.explanation:
        try:
            mn = TG._plan_mn(req)
        except ValueError:
            mn = None
        kc = TG._plan_kc(req)
        if mn is not None and kc is not None:
            t_mn, t_kc = TG._plan_cost(req, mn), TG._plan_cost(req, kc)
            tm = "-" if t_mn is None else g17(t_mn)
            tk = "-" if t_kc is None else g17(t_kc)
    g = plan.gett
    packs = ";".join(
        f"{s.operand}:{s.spec.kernel_name()}:{','.join(map(str, s.grid))}:{s.block[0]}:"
        f"{s.workspace_bytes}"
        for s in plan.packs
    )
    return (
        f"{plan.kind}|{g.candidate.name}|{g.spec.kernel_name()}|{','.join(map(str, g.grid))}|"
        f"{g.block[0]}|{plan.workspace_bytes}|{packs}|{tm}|{tk}"
    )


def gemm_line(p: dict) -> str:
    """The Python GEMM dispatcher's line for one request (tensor_gett_planner_cli --gemm)."""
    from rocke.dispatch.gemm import GemmRequest, dispatch_gemm_bf16, dispatch_gemm_fp16
    from rocke.dispatch.gemm.plan import predicted_us
    from rocke.dispatch.gemm.selector import config_string
    from rocke.instances.common.gemm_skinny import SkinnyGemmSpec

    _set_cus(p["arch"], p["cus"])
    lay = p["layout"]
    req = GemmRequest(
        M=p["M"],
        N=p["N"],
        K=p["K"],
        arch=p["arch"],
        dtype=p["dtype"],
        layout=lay,
        trans_a=lay[0] == "C",
        trans_b=lay[1] == "C",
    )
    disp = dispatch_gemm_bf16 if p["dtype"] == "bf16" else dispatch_gemm_fp16
    try:
        res = disp(req)
    except Exception:  # noqa: BLE001 - no candidate
        return "none"
    s = res.spec
    out = f"{res.candidate.name} {config_string(s)}"
    if isinstance(s, SkinnyGemmSpec):
        return out + f" L{s.layout} d{s.dtype}"
    t = predicted_us(req, s)
    return out + f" L{s.data.layout} bs{s.block_size} t={'-' if t is None else g17(t)}"


def ir_line(p: dict) -> str:
    """The SHA-256 of the LLVM IR (ROCKE_LLVM_FLAVOR, the Python engine's lowering) of every kernel
    of the Python planner's route (tensor_gett_planner_cli --ir): "<kind>|<pack sha>;...|<GETT
    sha>", or "none"."""
    import hashlib

    import rocke.dispatch.families.tensor_gett as TG
    from rocke.instances.common.tensor import (
        GettKnobs,
        TensorGettSpec,
        build_tensor_gett,
    )

    try:
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
    except ImportError:  # pragma: no cover - older tree
        from rocke import lower_kernel_to_llvm as lower

    _set_cus(p["arch"], p["cus"])
    spec = TensorGettSpec.make(
        p["a"],
        p["b"],
        p["d"],
        p["ext"],
        strides_a=p["sa"],
        strides_b=p["sb"],
        strides_d=p["sd"],
        dtype=p["dtype"],
        knobs=GettKnobs(bilinear=p["bilinear"]),
    )
    req = TG.TensorGettRequest(problem=spec, arch=p["arch"])
    try:
        plan = TG.plan_tensor_gett(req)
    except Exception:  # noqa: BLE001 - the planner raises: no route
        return "none"

    def sha(kernel) -> str:
        return hashlib.sha256(lower(kernel, arch=p["arch"]).encode("utf-8")).hexdigest()

    packs = ";".join(sha(TG.build_pack_step(s, p["arch"])) for s in plan.packs)
    return (
        f"{plan.kind}|{packs}|{sha(build_tensor_gett(plan.gett.spec, arch=p['arch']))}"
    )


def cpp_comparable(line: str) -> str:
    """The C++ line without its diagnostics (builder / source tags)."""
    return line.split(" builder=")[0]


def _worker(args):
    kind, chunk = args
    for k in PY_ONLY_ENV:
        os.environ.pop(k, None)
    fn = {
        "contraction": contraction_line,
        "gemm": gemm_line,
        "valid": valid_line,
        "trees": tree_line,
        "ir": ir_line,
    }[kind]
    return [fn(p) for p in chunk]


def python_lines(kind: str, problems: list, jobs: int = 0) -> list:
    """The Python lines of ``problems`` (``kind`` "contraction" / "gemm"), computed in ``jobs``
    processes (0: the CPU count)."""
    jobs = jobs or os.cpu_count() or 1
    jobs = max(1, min(jobs, len(problems) // 8 or 1))
    if jobs == 1:
        return _worker((kind, problems))
    size = -(-len(problems) // (jobs * 4))
    chunks = [problems[i : i + size] for i in range(0, len(problems), size)]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(jobs) as pool:
        parts = pool.map(_worker, [(kind, c) for c in chunks])
    return [line for part in parts for line in part]


# --------------------------------------------------------------------------- the C++ driver


def _cxx() -> str:
    for c in ("c++", "clang++", "g++"):
        p = shutil.which(c)
        if p:
            return p
    raise RuntimeError("no C++ compiler (c++ / clang++ / g++) on PATH")


def ensure_archive(build_root: Path | None = None) -> Path:
    """The engine archive: ``ROCKE_CORE_ARCHIVE`` if set, else ``<build_root>/librocke_core.a``
    (built with CMake when missing)."""
    env = os.environ.get("ROCKE_CORE_ARCHIVE", "")
    if env:
        return Path(env)
    root = Path(build_root or (Path(tempfile.gettempdir()) / "rocke_verify"))
    archive = root / "librocke_core.a"
    if not archive.exists():
        subprocess.run(
            [
                "cmake",
                "-S",
                str(PLATFORM),
                "-B",
                str(root),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                str(root),
                "--target",
                "rocke_core",
                "-j",
                str(os.cpu_count() or 1),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    return archive


def build_cli(out_dir: Path, archive: Path | None = None) -> Path:
    """Compile tensor_gett_planner_cli against the engine archive."""
    archive = archive or ensure_archive()
    exe = out_dir / (
        "tensor_gett_planner_cli.exe" if os.name == "nt" else "tensor_gett_planner_cli"
    )
    cmd = [
        _cxx(),
        "-std=c++20",
        "-O2",
        "-I",
        str(PLATFORM / "cpp" / "include"),
        "-I",
        str(PLATFORM / "cpp" / "instances" / "common"),
        str(CLI_SRC),
        str(archive),
        "-lm",
        *(["-ldl", "-pthread"] if sys.platform == "linux" else []),
        "-o",
        str(exe),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("tensor_gett_planner_cli build failed:\n" + r.stderr[-4000:])
    return exe


def cpp_lines(exe: Path, kind: str, problems: list, work: Path, jobs: int = 1) -> list:
    """The C++ lines of ``problems``, ``jobs`` driver processes over contiguous shards."""
    jobs = max(1, min(jobs, len(problems) // 16 or 1))
    size = -(-len(problems) // jobs) if problems else 1
    flag = {
        "contraction": [],
        "gemm": ["--gemm"],
        "valid": ["--valid"],
        "trees": ["--trees"],
        "ir": ["--ir"],
    }[kind]
    procs = []
    for j, i in enumerate(range(0, len(problems), size)):
        path = work / f"{kind}.{j}.jsonl"
        path.write_text(
            "".join(json.dumps(p) + "\n" for p in problems[i : i + size]),
            encoding="utf-8",
        )
        cmd = [str(exe), str(MODEL_DIR), str(path)] + flag
        procs.append(
            subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        )
    out = []
    for pr in procs:
        so, se = pr.communicate()
        if pr.returncode != 0:
            raise RuntimeError(f"{exe.name} rc={pr.returncode}:\n{(so + se)[-4000:]}")
        out += [ln.split(" ", 1)[1] for ln in so.splitlines()]
    return out


def main(argv) -> int:
    if len(argv) >= 4 and argv[1] in ("contraction", "gemm", "valid"):
        problems = [
            json.loads(ln) for ln in Path(argv[2]).read_text().splitlines() if ln
        ]
        lines = python_lines(argv[1], problems, int(argv[4]) if len(argv) > 4 else 0)
        Path(argv[3]).write_text("".join(f"{i} {ln}\n" for i, ln in enumerate(lines)))
        return 0
    if len(argv) >= 5 and argv[1] == "gen":
        kind, n, seed = argv[2], int(argv[3]), int(argv[4])
        arch = argv[5] if len(argv) > 5 else "gfx950"
        rng = random.Random(seed)
        if kind == "valid":
            pool = spec_pool()
            types = _spec_types()
            probs = [spec_problem(rng, pool, types) for _ in range(n)]
        elif kind == "edge":
            edges = _edge_values()
            probs = [gemm_edge(rng, arch, edges) for _ in range(n)]
        else:
            fn = contraction if kind == "contraction" else gemm
            probs = [fn(rng, arch) for _ in range(n)]
        if kind == "contraction" and arch == "gfx950" and "--tables" in argv:
            probs = table_problems() + probs
        sys.stdout.write("".join(json.dumps(p) + "\n" for p in probs))
        return 0
    sys.stderr.write(
        "usage: planner_ref.py gen <contraction|gemm|edge|valid> <n> <seed> [arch] [--tables]\n"
        "       planner_ref.py <contraction|gemm|valid> <problems.jsonl> <out> [jobs]\n"
    )
    return 2


if __name__ == "__main__":
    sys.path.insert(0, str(PLATFORM / "python"))
    raise SystemExit(main(sys.argv))
