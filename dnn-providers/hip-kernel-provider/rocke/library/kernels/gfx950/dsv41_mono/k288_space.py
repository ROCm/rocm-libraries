# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""W17 K1: the knob surface of ``k288_gemm`` (+ the code-object backend) as a ``rocke.dispatch.tuning.search``
``SearchSpace`` (PLAN decision 15). Level 0 of every param is the shipped default; ``backend`` is a categorical param
(COMGR | Rust direct), so the search picks the compiler per served shape. Every config is the same exact algorithm
(f32 accumulation in the MFMA's K order, ATOM's quantization rule): the space has no tolerance-class knob, so every
candidate is gated by the same float64 bound. ``xin="fp8"`` means ATOM's ``quantize_fp8`` launch + the GEMM on its codes
(timed together by the backend).

``python -m kernels.gfx950.dsv41_mono.k288_space`` prints the space.
"""

from __future__ import annotations

import json
from dataclasses import fields

from kernels.gfx950.dsv41_mono.k288_gemm import K288Knobs
from rocke.dispatch.tuning.search import SearchSpace
from rocke.dispatch.tuning.search.params import Param

BACKENDS = ("comgr", "direct")

# every built value of every performance knob (losers included); the default first
LEVELS = {
    "tiles": (1, 2, 3, 4, 6, 8, 12, 16, 24),
    "waves": (4, 1, 2, 8),
    "mt": (1, 2, 4),
    "depth": (3, 1, 2),
    "wfirst": (True, False),
    "xb": (2, 1, 4, 8),
    "bra": (1, 0),
    "sb": (True, False),
    "ep": ("direct", "lds"),
    "order": ("nfast", "mfast"),
    "wcp": ("default", "nt"),
    "xin": ("bf16", "fp8"),
}
ORDINAL = ("tiles", "depth", "xb")


def params(backends=BACKENDS):
    out = []
    for name, vals in LEVELS.items():
        if name in ORDINAL:
            p = Param.ordinal(name, list(vals))
        elif all(isinstance(v, bool) for v in vals):
            p = Param.categorical(name, list(vals))
        else:
            p = Param.categorical(name, list(vals))
        out.append(p)
    out.append(Param.categorical("backend", list(backends)))
    return out


def knobs_of(config: dict, k: int = 288) -> K288Knobs:
    kw = {f.name: config[f.name] for f in fields(K288Knobs) if f.name in config}
    kw["k"] = k
    return K288Knobs(**kw)


def is_valid(config: dict, k: int = 288) -> bool:
    try:
        knobs_of(config, k).check()
    except ValueError:
        return False
    return True


def k288_space(k: int = 288, backends=BACKENDS, fixed=None, name="k288") -> SearchSpace:
    return SearchSpace(params(backends), is_valid=lambda c: is_valid(c, k), fixed=fixed or {}, name=f"{name}_k{k}")


def default_config() -> dict:
    return {n: v[0] for n, v in LEVELS.items()} | {"backend": "comgr"}


if __name__ == "__main__":
    sp = k288_space()
    print(json.dumps(sp.describe(), indent=1, default=str)[:4000])
    print("size", sp.size)
