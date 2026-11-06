# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""W15 PORT K2-moe: the K2b port's knob surface as a search space for the rocKE search engine (PLAN decision 15).

Every hard-coded choice of the K2b stage functions (``stage_moe.MoeKnobs``) is one :class:`Param`. Every param is
categorical with level 0 = ATOM's choice (the parity port), so the genome of all zeros is the bitwise baseline and a
signed lookback / hill climb starts there (an ordinal param's level 0 would be its smallest value instead). The
per-stage code-object backend (COMGR or the Rust direct compiler, with its lever set) is a categorical param too.

A config maps to:

* function sets: :func:`function_sets` -> ``{set name: (MoeKnobs spec, stages)}``, the ``W15M_FN`` entries the rig's
  ``gen`` phase builds (``stage_moe`` CLI ``--knobs`` / ``--stages``);
* an extern spec: :func:`extern_spec` -> ``ATOM_DSV41_RK_EXTERN`` / ``rk_bench`` ``ext:`` items, one function set per
  extern stage.

Numerics classes: ``chains != 2`` reassociates the down K sum (TOLERANCE class: refused unless
``allow_tolerance=True``); ``nan=num`` differs from ATOM only for NaN inputs (bitwise on finite data, kept in the
space). Everything else is bitwise and must pass the in-place gate (``ATOM_DSV41_RK_LEV_CHECK``).

    python -m kernels.gfx950.dsv41_mono.k2moe_space        # prints the space and the parity config's sets
"""

from __future__ import annotations

import json
import sys

from .stage_moe import MoeKnobs

# (param, values with ATOM's / the parity port's first, the stages its function set covers)
DOWN = (
    ("r_max", (2, 1)),
    ("pslot", (0, 1)),
    ("smap", (0, 1)),
    ("sop", (0, 1)),
    ("sh_first", (1, 0)),
    ("wfirst", (1, 0)),
    ("chains", (2, 1, 3)),
)
UG = (
    ("ug_sop", (0, 1, 2)),
    ("ug_ra", (0, 1)),
)
COMMON = (
    ("pause", (1, 0)),
    ("poll_max", (12, 4, 8, 16)),
    ("w_cpol", (2, 0, 16)),
    ("nan", ("prop", "num")),
)
# code-object backend per extern stage group (decision 18: direct only where it wins on both nodes)
BACKENDS = ("comgr", "direct", "direct+Dm", "direct+Db")
BACKEND_STAGES = ("x8q", "route", "down", "reduce", "ug", "shared")


def params():
    from rocke.dispatch.tuning.search.params import Param

    out = []
    for name, vals in DOWN:
        out.append(Param.categorical(f"down.{name}", list(vals)))
    for name, vals in UG:
        out.append(Param.categorical(f"ug.{name}", list(vals)))
    for name, vals in COMMON:
        out.append(Param.categorical(f"all.{name}", list(vals)))
    for st in BACKEND_STAGES:
        out.append(Param.categorical(f"cc.{st}", list(BACKENDS)))
    return out


def is_valid(config: dict, *, allow_tolerance: bool = False) -> bool:
    if not allow_tolerance and config.get("down.chains", 2) != 2:
        return False
    # the down stage's route-table read is at most r_max (task, slot) blocks a wave (MoeKnobs check)
    try:
        _stage_knobs(config, "down")
        _stage_knobs(config, "ug")
    except ValueError:
        return False
    return True


def search_space(*, allow_tolerance: bool = False):
    from rocke.dispatch.tuning.search.params import SearchSpace

    return SearchSpace(params(), is_valid=lambda c: is_valid(c, allow_tolerance=allow_tolerance),
                       name="dsv41_k2moe")  # fmt: skip


def _stage_knobs(config: dict, group: str) -> MoeKnobs:
    kw = {}
    for name, vals in COMMON:
        kw[name] = config.get(f"all.{name}", vals[0])
    if group == "down":
        for name, vals in DOWN:
            kw[name] = config.get(f"down.{name}", vals[0])
    if group == "ug":
        for name, vals in UG:
            kw[name] = config.get(f"ug.{name}", vals[0])
    return MoeKnobs(**kw)


def function_sets(config: dict) -> dict:
    """``{set name: (knob spec, stages)}``: one set for the stages that only take the common knobs, one for down, one
    for ug (each named by its spec, so equal specs share a set)."""
    sets = {}

    def put(group, stages):
        spec = _stage_knobs(config, group).spec()
        name = "p_" + (spec.replace(",", "_").replace("=", "") or "base") + ("" if group == "base" else f"_{group}")
        sets.setdefault(name, (spec, set()))[1].update(stages)
        return name

    names = {"base": put("base", ("xq", "route", "reduce", "shared", "shm")), "down": put("down", ("down",)),
             "ug": put("ug", ("ug",))}  # fmt: skip
    return {"sets": {k: (v[0], sorted(v[1])) for k, v in sets.items()}, "by_group": names}


def extern_spec(config: dict, root: str = "@") -> dict:
    """extern stage -> (function set dir, backend). ``root`` prefixes the set name (``@`` = the rig's fn root)."""
    fs = function_sets(config)["by_group"]
    stage_group = {"x8q": "base", "route": "base", "reduce": "base", "shx": "base", "shm": "base", "shared": "base",
                   "down": "down", "ug": "ug", "ugx": "ug"}  # fmt: skip
    cc_of = {"x8q": "x8q", "route": "route", "reduce": "reduce", "down": "down", "ug": "ug", "ugx": "ug",
             "shared": "shared", "shx": "shared", "shm": "shared"}  # fmt: skip
    return {st: (root + fs[g], config.get(f"cc.{cc_of[st]}", "comgr")) for st, g in stage_group.items()}


def main(argv=None) -> int:
    sp = search_space()
    base = {p.name: p.levels[0][0][1] for p in sp.params}
    print(json.dumps({"params": [p.describe() for p in sp.params], "parity": base,
                      "parity_sets": function_sets(base), "parity_extern": extern_spec(base)}, indent=1))
    assert all(v == p.levels[0][0][1] for p, v in zip(sp.params, base.values())), "level 0 is the parity value"
    assert MoeKnobs() == _stage_knobs(base, "down") == _stage_knobs(base, "ug"), "parity config = MoeKnobs()"
    return 0


if __name__ == "__main__":
    sys.exit(main())
