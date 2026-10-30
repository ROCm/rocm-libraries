# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side rules of the tuned tiled elementwise classes (:mod:`.ewp`): the classes of the
kernel pack (the palette of each target), the tuned problems (canonical problem -> class) and
the class choice of a problem.

hipTensor's rocKE backend implements the same rules in C++ (``rocke_ewp_rule.hpp``, used by
``rocke_plan.cpp`` and by the C++ pack builder, whose ``--ewp-picks`` mode rocKE's pack
cross-check test compares with :func:`choose`); the C++ engine carries the table
(``hiptensor_aot_ewp_table.cpp``, written from this module by
``library/builders/hiptensor/gen_ewp_table.py``).

Problems are hipTensor's canonical elementwise problems: ``dims`` = ``(extent, [stride per
tensor in A [B] [C] D order])``, extent-1 modes dropped, ordered by D's strides, modes
contiguous in every tensor merged (``host.canonicalize(dims, key=D)``).

The choice (:func:`choose`): the tuned class of a tuned problem when it serves the layout;
otherwise the first class of the palette (in table order) of the problem's data types whose
kinds match the layout, which serves it and launches at least :data:`MIN_CTAS` CTAs; when none
launches that many, the serving class with the most CTAs (the first of equals).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

from .ewp import EwpSpec, ewp_kinds, ewp_plan

#: CTAs the untuned choice wants a launch to have (four per CU of the largest target).
MIN_CTAS = 1024
#: Targets with ewp classes in the pack (streaming accesses and the measurements behind the
#: palette: gfx942 / gfx950).
ARCHES = ("gfx942", "gfx950")
#: LDS a class of each target may use.
LDS_LIMIT = {"gfx942": 65536, "gfx950": 163840}

#: The data-type rows with ewp classes: the API's rows (hipTensor's planElementwise) whose
#: tensors share one element size: permutations f16 / f32 / f64, binary f16 / bf16 / f32 / f64,
#: trinary f16 / f32 / f64.
DTYPE_ROWS = {
    1: [
        ("f16", None, None, "f16"),
        ("f32", None, None, "f32"),
        ("f64", None, None, "f64"),
    ],
    2: [
        ("f16", None, "f16", "f16"),
        ("bf16", None, "bf16", "bf16"),
        ("f32", None, "f32", "f32"),
        ("f64", None, "f64", "f64"),
    ],
    3: [
        ("f16", "f16", "f16", "f16"),
        ("f32", "f32", "f32", "f32"),
        ("f64", "f64", "f64", "f64"),
    ],
}


def _kinds(arity: int) -> List[str]:
    out = []
    for m in range(1, 1 << arity):
        out.append(
            "".join("T" if m >> (arity - 1 - i) & 1 else "V" for i in range(arity))
        )
    return sorted(out, key=lambda k: (-k.count("T"), k))


#: Default class templates (knobs without data types / kinds), in preference order, per element
#: size: mode t first, then mode a. ``to`` / ``tt`` in elements (mode t) or run positions (mode a).
def _t(to: int, tt: int, block: int, ilp: int, swz=True, **kw) -> dict:
    d = dict(mode="t", to=to, tt=tt, block=block, nt=3, ilp=ilp, **kw)
    d["swz"] = (
        1 if swz else 0
    )  # 1: the default swizzle of the class (resolved in class_spec)
    return d


def _a(to: int, tt: int, tw: int, block: int, ilp: int = 0, **kw) -> dict:
    d = dict(mode="a", to=to, tt=tt, tw=tw, block=block, nt=3, ilp=ilp, **kw)
    d["swz"] = 1
    return d


TEMPLATES: Dict[int, List[dict]] = {
    2: [
        _t(128, 128, 256, 2),
        _t(64, 64, 128, 2),
        _a(4, 16, 8, 256),
        _a(8, 16, 4, 256),
        _a(16, 16, 2, 256),
        _a(16, 32, 1, 256),
    ],
    4: [
        _t(128, 64, 256, 2),
        _t(64, 32, 128, 2),
        _t(32, 32, 64, 0),
        _a(4, 16, 8, 256),
        _a(8, 16, 4, 256),
        _a(16, 16, 2, 256),
        _a(16, 32, 1, 256),
    ],
    8: [
        _t(64, 64, 256, 2),
        _t(32, 32, 128, 2),
        _t(16, 16, 64, 0),
        _a(4, 16, 8, 256),
        _a(8, 16, 4, 256),
        _a(16, 16, 2, 256),
        _a(16, 32, 1, 256),
    ],
}

#: Tuned problems per target: (canonical problem key, class). gfx950: the API's permutation and
#: elementwise doc rows with a transposing layout (f16 / f32), each class the fastest of an
#: interleaved confirmation of the search engine's best candidates (cold timing, gfx950).
TUNED: Dict[str, List[Tuple[str, dict]]] = {
    "gfx942": [],
    "gfx950": [
        (
            "2|f16,,f16,f16|8192:8192,1,1;8192:1,8192,8192",
            dict(arity=2, a="f16", d="f16", c="f16", kinds="TV", to=256, nt=3, ilp=4),
        ),  # ewbin_r2_doc_rev f16
        (
            "2|f32,,f32,f32|8192:8192,1,1;8192:1,8192,8192",
            dict(
                arity=2,
                a="f32",
                d="f32",
                c="f32",
                kinds="TV",
                to=256,
                tt=32,
                block=512,
                nt=3,
                swz=16,
                swsh=8,
                tpc=2,
                ilp=2,
            ),
        ),  # ewbin_r2_doc_rev f32
        (
            "2|f16,,f16,f16|1024:524288,1,1;1024:512,1024,1024;512:1,1048576,1048576",
            dict(
                arity=2,
                a="f16",
                d="f16",
                c="f16",
                kinds="TV",
                to=128,
                tt=128,
                nt=3,
                swz=16,
                swsh=7,
                ilp=4,
            ),
        ),  # ewbin_r3_doc_rev f16
        (
            "2|f32,,f32,f32|1024:524288,1,1;1024:512,1024,1024;512:1,1048576,1048576",
            dict(
                arity=2,
                a="f32",
                d="f32",
                c="f32",
                kinds="TV",
                to=128,
                nt=3,
                swz=16,
                swsh=7,
                tpc=2,
                ilp=1,
            ),
        ),  # ewbin_r3_doc_rev f32
        (
            "2|f16,,f16,f16|112:1835008,1,1;112:16384,112,112;256:64,12544,12544;64:1,3211264,3211264",
            dict(
                arity=2,
                a="f16",
                d="f16",
                c="f16",
                kinds="TV",
                to=128,
                block=128,
                nt=3,
                swz=16,
                swsh=7,
                tpc=2,
                ilp=4,
            ),
        ),  # ewbin_r4_doc_rev f16
        (
            "2|f32,,f32,f32|112:1835008,1,1;112:16384,112,112;256:64,12544,12544;64:1,3211264,3211264",
            dict(
                arity=2,
                a="f32",
                d="f32",
                c="f32",
                kinds="TV",
                tt=32,
                block=128,
                nt=3,
                swz=16,
                swsh=6,
                tpc=2,
                ilp=2,
            ),
        ),  # ewbin_r4_doc_rev f32
        (
            "2|f16,,f16,f16|32:8388608,1,1;64:131072,32,32;64:2048,2048,2048;64:32,131072,131072;32:1,8388608,8388608",
            dict(
                arity=2,
                a="f16",
                d="f16",
                c="f16",
                kinds="TV",
                to=256,
                tt=128,
                block=512,
                nt=3,
                swz=16,
                swsh=8,
                walk="t",
                tpc=2,
                ilp=2,
            ),
        ),  # ewbin_r5_doc_rev f16
        (
            "2|f32,,f32,f32|32:8388608,1,1;64:131072,32,32;64:2048,2048,2048;64:32,131072,131072;32:1,8388608,8388608",
            dict(
                arity=2,
                a="f32",
                d="f32",
                c="f32",
                kinds="TV",
                to=256,
                tt=32,
                nt=3,
                ilp=2,
            ),
        ),  # ewbin_r5_doc_rev f32
        (
            "2|f16,,f16,f16|16:8388608,1,1;16:524288,16,16;32:16384,256,256;32:512,8192,8192;32:16,262144,262144;16:1,8388608,8388608",
            dict(
                arity=2,
                a="f16",
                d="f16",
                c="f16",
                kinds="TV",
                to=128,
                tt=128,
                nt=3,
                swz=16,
                swsh=7,
                walk="t",
                tpc=4,
                ilp=1,
            ),
        ),  # ewbin_r6_doc_rev f16
        (
            "2|f32,,f32,f32|16:8388608,1,1;16:524288,16,16;32:16384,256,256;32:512,8192,8192;32:16,262144,262144;16:1,8388608,8388608",
            dict(
                arity=2,
                a="f32",
                d="f32",
                c="f32",
                kinds="TV",
                to=128,
                nt=3,
                swz=16,
                swsh=7,
                tpc=2,
                ilp=1,
            ),
        ),  # ewbin_r6_doc_rev f32
        (
            "3|f32,f32,f32,f32|8192:8192,8192,1,1;8192:1,1,8192,8192",
            dict(
                arity=3,
                a="f32",
                d="f32",
                b="f32",
                c="f32",
                kinds="TTV",
                to=32,
                tt=128,
                nt=3,
                swz=16,
                swsh=5,
                tpc=2,
                ilp=1,
            ),
        ),  # ewtri_r2_doc_rev f32
        (
            "3|f32,f32,f32,f32|1024:524288,524288,1,1;1024:512,512,1024,1024;512:1,1,1048576,1048576",
            dict(
                arity=3,
                a="f32",
                d="f32",
                b="f32",
                c="f32",
                kinds="TTV",
                nt=3,
                swz=16,
                swsh=6,
                walk="t",
                tpc=2,
                ilp=2,
            ),
        ),  # ewtri_r3_doc_rev f32
        (
            "3|f32,f32,f32,f32|112:1835008,1835008,1,1;112:16384,16384,112,112;256:64,64,12544,12544;64:1,1,3211264,3211264",
            dict(
                arity=3,
                a="f32",
                d="f32",
                b="f32",
                c="f32",
                kinds="TTV",
                tt=128,
                nt=3,
                swz=16,
                swsh=6,
                tpc=2,
                ilp=2,
            ),
        ),  # ewtri_r4_doc_rev f32
        (
            "3|f32,f32,f32,f32|32:8388608,8388608,1,1;64:131072,131072,32,32;64:2048,2048,2048,2048;64:32,32,131072,131072;32:1,1,8388608,8388608",
            dict(
                arity=3,
                a="f32",
                d="f32",
                b="f32",
                c="f32",
                kinds="TTV",
                to=256,
                tt=32,
                nt=3,
                swz=16,
                swsh=8,
                ilp=4,
            ),
        ),  # ewtri_r5_doc_rev f32
        (
            "3|f32,f32,f32,f32|16:8388608,8388608,1,1;16:524288,524288,16,16;32:16384,16384,256,256;32:512,512,8192,8192;32:16,16,262144,262144;16:1,1,8388608,8388608",
            dict(
                arity=3,
                a="f32",
                d="f32",
                b="f32",
                c="f32",
                kinds="TTV",
                to=128,
                nt=3,
                walk="t",
                ilp=4,
            ),
        ),  # ewtri_r6_doc_rev f32
        (
            "1|f16,,,f16|8192:8192,1;8192:1,8192",
            dict(arity=1, a="f16", d="f16", to=256, nt=3, swz=16, swsh=8),
        ),  # perm_r2_doc f16
        (
            "1|f32,,,f32|8192:8192,1;8192:1,8192",
            dict(arity=1, a="f32", d="f32", to=32, tt=128, nt=3, tpc=2, ilp=2),
        ),  # perm_r2_doc f32
        (
            "1|f16,,,f16|1024:524288,1;524288:1,1024",
            dict(
                arity=1,
                a="f16",
                d="f16",
                to=128,
                tt=128,
                block=512,
                nt=3,
                swz=16,
                swsh=7,
            ),
        ),  # perm_r3_doc f16
        (
            "1|f32,,,f32|1024:524288,1;524288:1,1024",
            dict(
                arity=1, a="f32", d="f32", to=128, tt=128, nt=3, swz=16, swsh=7, ilp=2
            ),
        ),  # perm_r3_doc f32
        (
            "1|f16,,,f16|1024:512,1;512:1,1024;1024:524288,524288",
            dict(
                arity=1,
                a="f16",
                d="f16",
                to=128,
                tt=256,
                block=512,
                nt=3,
                swz=16,
                swsh=7,
            ),
        ),  # perm_r3_innerid f16
        (
            "1|f32,,,f32|1024:512,1;512:1,1024;1024:524288,524288",
            dict(
                arity=1,
                a="f32",
                d="f32",
                to=128,
                block=512,
                nt=3,
                swz=16,
                swsh=7,
                ilp=1,
            ),
        ),  # perm_r3_innerid f32
        (
            "1|f16,,,f16|64:1,1;12544:16384,64;256:64,802816",
            dict(
                arity=1,
                a="f16",
                d="f16",
                mode="a",
                to=4,
                tt=16,
                tw=8,
                nt=3,
                swz=2,
                swsh=5,
            ),
        ),  # perm_r4_doc f16
        (
            "1|f32,,,f32|64:1,1;12544:16384,64;256:64,802816",
            dict(
                arity=1,
                a="f32",
                d="f32",
                mode="a",
                to=16,
                tt=16,
                tw=8,
                block=512,
                nt=3,
                swz=2,
                swsh=7,
                walk="t",
                ilp=1,
            ),
        ),  # perm_r4_doc f32
        (
            "1|f16,,,f16|112:16384,1;256:64,112;64:1,28672;112:1835008,1835008",
            dict(arity=1, a="f16", d="f16", to=256, nt=3, ilp=2),
        ),  # perm_r4_innerid f16
        (
            "1|f32,,,f32|112:16384,1;256:64,112;64:1,28672;112:1835008,1835008",
            dict(arity=1, a="f32", d="f32", to=256, nt=3, ilp=4),
        ),  # perm_r4_innerid f32
        (
            "1|f16,,,f16|32:1,1;32:8388608,32;64:32,1024;64:131072,65536;64:2048,4194304",
            dict(arity=1, a="f16", d="f16", mode="a", to=8, tt=16, tw=4, nt=3),
        ),  # perm_r5_doc f16
        (
            "1|f32,,,f32|32:1,1;32:8388608,32;64:32,1024;64:131072,65536;64:2048,4194304",
            dict(
                arity=1,
                a="f32",
                d="f32",
                mode="a",
                to=4,
                tt=16,
                tw=8,
                nt=3,
                swz=2,
                swsh=5,
            ),
        ),  # perm_r5_doc f32
        (
            "1|f16,,,f16|64:131072,1;64:2048,64;64:32,4096;32:1,262144;32:8388608,8388608",
            dict(
                arity=1,
                a="f16",
                d="f16",
                to=256,
                tt=128,
                block=512,
                nt=3,
                swz=16,
                swsh=8,
            ),
        ),  # perm_r5_innerid f16
        (
            "1|f32,,,f32|64:131072,1;64:2048,64;64:32,4096;32:1,262144;32:8388608,8388608",
            dict(
                arity=1,
                a="f32",
                d="f32",
                to=128,
                tt=256,
                block=512,
                nt=3,
                swz=16,
                swsh=7,
                walk="t",
                tpc=2,
                ilp=1,
            ),
        ),  # perm_r5_innerid f32
        (
            "1|f16,,,f16|16:1,1;32:16384,16;32:16,512;16:8388608,16384;32:512,262144;16:524288,8388608",
            dict(arity=1, a="f16", d="f16", mode="a", to=32, tt=16, tw=2, nt=3),
        ),  # perm_r6_doc f16
        (
            "1|f32,,,f32|16:1,1;32:16384,16;32:16,512;16:8388608,16384;32:512,262144;16:524288,8388608",
            dict(
                arity=1,
                a="f32",
                d="f32",
                mode="a",
                to=8,
                tt=16,
                tw=4,
                nt=3,
                swz=4,
                swsh=5,
            ),
        ),  # perm_r6_doc f32
        (
            "1|f16,,,f16|16:524288,1;32:16384,16;32:512,512;32:16,16384;16:1,524288;16:8388608,8388608",
            dict(arity=1, a="f16", d="f16", to=128, tt=256, block=512, nt=3),
        ),  # perm_r6_innerid f16
        (
            "1|f32,,,f32|16:524288,1;32:16384,16;32:512,512;32:16,16384;16:1,524288;16:8388608,8388608",
            dict(
                arity=1,
                a="f32",
                d="f32",
                to=128,
                tt=128,
                block=512,
                nt=3,
                swz=16,
                swsh=7,
                walk="t",
                ilp=2,
            ),
        ),  # perm_r6_innerid f32
    ],
}


def class_spec(row, arity: int, kinds: str, knobs: dict) -> EwpSpec:
    """The class of a template: ``swz`` 1 stands for the class's default swizzle
    (:func:`.ewp.default_swizzle`)."""
    import dataclasses

    from .ewp import default_swizzle

    a, b, c, d = row
    k = dict(knobs)
    sw = k.pop("swz", 0)
    s = EwpSpec(arity=arity, a=a, b=b, c=c, d=d, kinds=kinds, **k)
    if sw == 1:
        z, sh = default_swizzle(s)
        s = dataclasses.replace(s, swz=z, swsh=sh)
    return s


def _fits(s: EwpSpec, arch: str) -> bool:
    try:
        s.check()
    except ValueError:
        return False
    return s.kinds.count("T") * s.tile_bytes() <= LDS_LIMIT[arch]


def pack_classes(arch: str) -> List[EwpSpec]:
    """The classes of the target's pack, in choice order: per arity, data-type row, kinds and
    template; then the classes of the tuned problems not already there."""
    if arch not in ARCHES:
        return []
    out: List[EwpSpec] = []
    seen = set()
    for arity in (1, 2, 3):
        for row in DTYPE_ROWS[arity]:
            es = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}[row[3]]
            for kinds in _kinds(arity):
                for t in TEMPLATES.get(es, []):
                    if arity == 3 and t["mode"] != "t":
                        continue  # trinary: element transpositions only
                    s = class_spec(row, arity, kinds, t)
                    if s not in seen and _fits(s, arch):
                        seen.add(s)
                        out.append(s)
    for _key, c in TUNED.get(arch, []):
        s = EwpSpec(**c)
        if s not in seen and _fits(s, arch):
            seen.add(s)
            out.append(s)
    return out


def problem_key(arity: int, dts: Sequence[Optional[str]], dims) -> str:
    """The canonical problem: arity, the data types of A, B, C, D and the canonical modes."""
    a, b, c, d = dts
    return f"{arity}|{a or ''},{b or ''},{c or ''},{d or ''}|" + ";".join(
        f"{e}:" + ",".join(str(x) for x in st) for e, st in dims
    )


def fnv1a64(text: str) -> int:
    h = 0xCBF29CE484222325
    for ch in text.encode():
        h ^= ch
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def routes(arch: str) -> Dict[str, str]:
    """The pack routes of the tuned problems: ``ewp.tuned.<fnv>`` = the class's kernel name,
    ``ewp.tuned.<fnv>.key`` = the canonical problem."""
    out = {}
    for key, c in TUNED.get(arch, []):
        h = f"{fnv1a64(key):016x}"
        out[f"ewp.tuned.{h}"] = EwpSpec(**c).kernel_name()
        out[f"ewp.tuned.{h}.key"] = key
    return out


def _ctas(grid) -> int:
    return grid[0] * grid[1] * grid[2]


def choose(arch: str, arity: int, dts, dims, classes: Optional[List[EwpSpec]] = None):
    """``(spec, (args, grid), source)`` of a problem (source "tuned" or "rule"), or None when
    no class of the target serves it."""
    classes = pack_classes(arch) if classes is None else classes
    a, b, c, d = dts
    key = problem_key(arity, dts, dims)
    for k, cdict in TUNED.get(arch, []):
        if k == key:
            s = EwpSpec(**cdict)
            if s in classes:
                pl = ewp_plan(s, dims)
                if pl is not None:
                    return s, pl, "tuned"
    names = {1: ["A", "D"], 2: ["A", "C", "D"], 3: ["A", "B", "C", "D"]}[arity]
    kinds = {m: ewp_kinds(names, dims, m) for m in ("t", "a")}
    best = None
    for s in classes:
        if (s.arity, s.a, s.b, s.c, s.d) != (arity, a, b, c, d) or s.kinds != kinds[
            s.mode
        ]:
            continue
        pl = ewp_plan(s, dims)
        if pl is None:
            continue
        n = _ctas(pl[1])
        if n >= MIN_CTAS:
            return s, pl, "rule"
        if best is None or n > _ctas(best[1][1]):
            best = (s, pl)
    if best is not None:
        return best[0], best[1], "rule"
    return None


def class_dict(s: EwpSpec) -> dict:
    return asdict(s)


__all__ = [
    "ARCHES",
    "DTYPE_ROWS",
    "LDS_LIMIT",
    "MIN_CTAS",
    "TEMPLATES",
    "TUNED",
    "choose",
    "class_dict",
    "fnv1a64",
    "pack_classes",
    "problem_key",
    "routes",
]
