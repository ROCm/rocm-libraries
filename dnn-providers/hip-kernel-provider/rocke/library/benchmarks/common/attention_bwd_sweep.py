# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sized raw-lever sweep driver for the attention backward (general family).

* **Legal space.** Candidates are full ``AttnBwdSpec`` knob sets. Legality is
  the family validator (``validate_attn_bwd_spec``: catalog, couplings, the
  LDS phase plan and the two-budget register estimate with the prune
  allowance) plus the emission check (values that are not built are recorded
  ``not_built`` and never compiled). There is no second model.
* **Lever families and release.** Every knob belongs to one of the eight
  lever families of the release order (:data:`LEVER_FAMILIES`). A family that
  is not released (``--release``) contributes only the arch's portable
  default; a released family contributes its whole catalog. Knobs the family
  module marks not yet effective (``NOT_YET_EFFECTIVE_KNOBS``) are pinned to
  one value until they become effective. Records list the released families.
* **Registry.** The knobs, their values, families and owning stages come from
  the backward tuning registry (``dispatch/attention_bwd/tuning_common.py``):
  :func:`catalog` and :data:`KNOB_FAMILY` are read from it. The knobs of the
  first sweep (:data:`SWEPT_KNOBS`) are crossed in their stage as before; each
  stage adds its other registry knobs (``*_EXTRA``) one at a time around every
  product point (``--level production``) or crossed with it (``--level
  full``). A registry knob is a candidate key only where it differs from its
  default, so earlier record keys stay valid.
* **Funnel** (stage words ``geometry``, ``structure``, ``schedule``,
  ``codegen``, ``swizzle``, ``alternate``, ``runtime``, ``confirm``; fixed
  keep widths :data:`KEEP`). Each stage builds its candidates from the merged
  records of the stage before it (every shard file), drops illegal and
  duplicate candidates (two candidates whose resolved specs agree apart from
  not-yet-effective knobs compile one kernel), compiles the rest in this job
  (thread or process pool, on-disk cache keyed by spec, arch, flavor, comgr and
  source hash), applies the resource gate (scratch, spills, budgets, hot-loop
  AGPR copies), and times the survivors with the timing harness on the
  slice's representative cells: up to ``--arms-per-attempt`` rocKE
  configurations and the competitor bar per attempt, five or more kept
  fresh-process attempts, every configuration correctness-gated (pruning set,
  timed shape) in the timing process. ``confirm`` runs the final top 2 on the
  family cohort (every census cell of the slice's head size).
* **Counts** (``--count``) and the job plan (``--count --plan``) per primary
  slice, from the validator, with the wall-budget inputs read by name from a
  private constants file (values never printed).

stdout carries only counts and verdict tokens. Records (JSONL) go to a
results directory outside the source tree: ``<out>/<slice>/<stage>/
shard-<i>of<n>.jsonl``, appended and resumed (a re-run skips recorded keys).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dispatch.attention_bwd.tuning_common import (
    BWD_KNOB_AXES,
    KIND_DQ,
    LEVER_FAMILIES,
    SWEEP_LEVELS,
    backward_axis,
    backward_spec_legality,
    backward_tuning_axes,
    knob_family,
)
from dispatch.attention_bwd.tuning_common import (
    CODEGEN_STRATEGIES as CODEGEN_STRATEGIES,  # re-exported (stage K values)
)
from kernels.common._attention_bwd_tiles import (
    attn_bwd_occupancy,
    g02_grids,
    natural_grids,
    replicated_g02,
    small_tile_grids,
)
from kernels.common.attention_bwd import (
    AGPR_ALLOC_BACKEND,
    BLOCK_K4_VALUES,
    DECLARATION_ONLY,
    INERT_KNOBS,
    NOT_YET_EFFECTIVE_KNOBS,
    STAGE_TABLE_SCHEDS,
    AttnBwdSpec,
    attn_bwd_arch_facts,
    attn_bwd_default_geometry,
    lds_footprint_bytes,
    lds_plan,
    register_estimates,
)
from kernels.common.attention_bwd_dq import validate_attn_bwd_dq_spec
from kernels.common.attention_bwd_plan import AUX_KNOBS, DQ_KNOBS
from kernels.common.attention_bwd_plan import RUNTIME_KNOBS as PLAN_RUNTIME_KNOBS

FAMILIES = ("general", "dense", "unified")
# XCDs per device the stage R XCD-remap values assume (the request's
# num_xcds hint; rocke.helpers.grid NUM_XCDS_*).
SWEEP_NUM_XCDS = MappingProxyType({"gfx942": 8, "gfx950": 8})
# The num_cus hint of a mirror-pairing candidate: the arm replaces it with the
# CU count of the device it runs on (attention_bwd_rocke_arm.DEVICE_NUM_CUS).
DEVICE_NUM_CUS = "device"
# Families whose spec validator and builder exist in this tree.
SWEEPABLE_FAMILIES = ("general",)
FAMILY_ARCHS = {
    "general": ("gfx942", "gfx950", "gfx1151", "gfx1201"),
    "dense": ("gfx942", "gfx950"),
    "unified": ("gfx942", "gfx950", "gfx1151", "gfx1201"),
}
FAMILY_D = {"general": (32, 64, 128), "dense": (64, 128), "unified": (32, 64, 128)}

STAGE_WORDS = (
    "geometry",
    "structure",
    "schedule",
    "codegen",
    "swizzle",
    "alternate",
    "runtime",
    "confirm",
)
# Keep widths per stage (fixed; recorded with every sweep record).
KEEP = {
    "geometry": 8,
    "structure": 8,
    "schedule": 4,
    "codegen": 4,
    "swizzle": 4,
    "alternate": 4,
    "runtime": 2,
}
STRUCTURE_FROM_GEOMETRIES = 4  # the structure stage explores the top 4 geometries
ALTERNATE_STRUCTURES = 16  # the alternate stage re-runs its top 16 structures
SECONDARY_FROM_PRIMARY = 16  # secondary slices recompile the primary's top 16
MAX_TIED_BUFFERS = 3
# Row-major LDS buffers whose row pad stage S enumerates, in priority order
# (the full level crosses the first MAX_TIED_BUFFERS present ones; every
# present one is varied on its own at both levels).
PAD_BUFFERS = (
    "ds",
    "q",
    "do",
    "dkv_out",
    "pt",
    "k_res",
    "v_res",
    "k_stage",
    "v_stage",
)
SCHED_PAIRS = (
    ("none", False),
    ("iglp0", False),
    ("iglp1", False),
    ("fences_only", False),
    ("stage_table", False),
    ("stage_table", True),
    ("stage_table_atomics", False),
    ("stage_table_atomics", True),
)

# The eight lever families (release order) and every knob's family come from
# the backward tuning registry (dispatch/attention_bwd/tuning_common.py).
KNOB_FAMILY = dict(knob_family())
ALL_RELEASED = frozenset(LEVER_FAMILIES)

GEOMETRY_KNOBS = ("waves", "block_m", "block_n", "block_k4", "warp_grid_g4")
STRUCTURE_KNOBS = (
    "atom_g02",
    "atom_g13",
    "atom_g4",
    "pt_route",
    "warp_grid_g02",
    "warp_grid_g13",
    "kv_residency",
    "kt_source",
    "transpose_source",
    "global_path",
    "ring_depth",
    "s_in_agpr",
)
SCHEDULE_KNOBS = ("sched", "setprio", "waves_per_eu", "agpr_alloc")
MINOR_KNOBS = ("scheduler_strategy", "edge_tiles", "ws_layout", "acc_in_lds")
# Knobs whose non-default values can crash the compiler itself (the gfx942
# comgr segfaults on the d64 main kernel under the "iterative-ilp" machine
# scheduler): a candidate with one compiles in a child process whatever the
# compile mode, so a crash is that candidate's ``compile_crash`` record and
# not the end of the sweep.
CRASH_ISOLATED_KNOBS = ("scheduler_strategy",)


def crash_isolated(spec) -> bool:
    """Whether ``spec`` is compiled in a child process (see above)."""
    return any(getattr(spec, k, None) is not None for k in CRASH_ISOLATED_KNOBS)


SWEPT_KNOBS = (
    GEOMETRY_KNOBS + STRUCTURE_KNOBS + SCHEDULE_KNOBS + MINOR_KNOBS + ("head_pack",)
)
# Runtime plan values: never in a compile key (the plan's own knob names).
RUNTIME_KNOBS = tuple(PLAN_RUNTIME_KNOBS)
assert set(RUNTIME_KNOBS) == {
    "g_split",
    "lb_order",
    "xcd_chunk",
    "scale_placement",
    "use_worklist",
}
# Registered knobs every candidate carries since the first sweep records
# (SWEPT_KNOBS, the swizzle stage's lds_swizzle, the runtime values). Every
# other registered knob is a key of a candidate only where its value differs
# from the default, so the record keys of earlier candidates do not change.
_ALWAYS_KEYED = frozenset(SWEPT_KNOBS) | {"lds_swizzle"} | set(RUNTIME_KNOBS)
KEYED_IF_SET = MappingProxyType(
    {a.name: a.default for a in BWD_KNOB_AXES if a.name not in _ALWAYS_KEYED}
)


def _stage_extras(stage: str, product: Sequence[str] = ()) -> tuple[str, ...]:
    """Sweepable registry axes of ``stage`` that the stage enumerates besides
    its product knobs (``product``) and the legacy tuples."""
    legacy = set(SWEPT_KNOBS) | set(RUNTIME_KNOBS) | {"lds_swizzle"} | set(product)
    return tuple(
        a.name
        for a in BWD_KNOB_AXES
        if a.stage == stage
        and a.name not in legacy
        and not a.plan_owned
        and a.name not in NOT_YET_EFFECTIVE_KNOBS
        and a.kind != KIND_DQ  # the dQ kernel's values: dq_values / stage R
    )


# Registry knobs each stage adds to its legacy product (in registry order).
# Stage H crosses agpr_route with agpr_alloc (it needs a reservation) and
# enumerates g4_slicing with block_k4 (the slice width; see g4_slicing_values).
STRUCTURE_EXTRA = _stage_extras("structure")
SCHEDULE_EXTRA = _stage_extras("schedule", ("agpr_route", "g4_slicing"))
CODEGEN_EXTRA = _stage_extras("codegen")
SWIZZLE_EXTRA = _stage_extras("swizzle")
RUNTIME_EXTRA = tuple(a for a in _stage_extras("runtime") if a in AUX_KNOBS)
assert set(RUNTIME_EXTRA) == set(AUX_KNOBS)
# The split dQ kernel's fields (stage T, on split points: see dq_values) and
# its XCD chunk (stage R of split candidates).
DQ_EXTRA = tuple(
    a.name for a in BWD_KNOB_AXES if a.kind == KIND_DQ and a.stage == "structure"
)


# --------------------------------------------------------------------------- slices
@dataclass(frozen=True)
class SliceKey:
    """Problem-class fields of a sweep slice (the primary slice by default)."""

    arch: str
    family: str
    head_size: int
    dtype: str = "bf16"
    mask_class: str = "band"
    seq_mode: str = "batched"
    dkv_mode: str = "direct"
    stage_vec: int = 8
    kind: str = "primary"  # "primary" or a secondary slice name

    def tag(self) -> str:
        base = f"{self.arch}-{self.family}-d{self.head_size}"
        return base if self.kind == "primary" else f"{base}-{self.kind}"

    def spec_class(self) -> dict:
        return {
            "head_size": self.head_size,
            "dtype": self.dtype,
            "mask_class": self.mask_class,
            "seq_mode": self.seq_mode,
            "dkv_mode": self.dkv_mode,
            "stage_vec": self.stage_vec,
        }


SECONDARY_SLICES = {
    "fp16": {"dtype": "fp16"},
    "mask_none": {"mask_class": "none"},
    "thd": {"seq_mode": "thd"},
    "atomic": {"dkv_mode": "atomic"},
    "sv1": {"stage_vec": 1},
    "head_pack": {},
}


def primary_slice(arch: str, family: str, d: int) -> SliceKey:
    return SliceKey(
        arch, family, d, seq_mode="thd" if family == "unified" else "batched"
    )


def secondary_slice(primary: SliceKey, kind: str) -> SliceKey:
    if kind not in SECONDARY_SLICES:
        raise ValueError(f"secondary slice {kind!r} not in {tuple(SECONDARY_SLICES)}")
    return replace(primary, kind=kind, **SECONDARY_SLICES[kind])


def all_primary_slices() -> list[SliceKey]:
    return [
        primary_slice(arch, fam, d)
        for fam in FAMILIES
        for arch in FAMILY_ARCHS[fam]
        for d in FAMILY_D[fam]
    ]


# --------------------------------------------------------------------------- catalog
def _atom_k(atom: str) -> int:
    return int(atom.rsplit("x", 1)[1])


def default_knobs(sl: SliceKey) -> dict:
    """The arch's portable start point for every swept knob (the shipped
    correctness configuration), with the natural warp grids."""
    d = attn_bwd_default_geometry(
        sl.arch, sl.head_size, sl.dtype, sl.stage_vec, sl.seq_mode
    )
    out = {k: d[k] for k in d if k in SWEPT_KNOBS}
    out["warp_grid_g02"] = (1, d["waves"])
    out["warp_grid_g13"] = (d["waves"], 1)
    out["agpr_alloc"] = None
    out["ws_layout"] = "head_major"
    out["head_pack"] = sl.kind == "head_pack"
    missing = set(SWEPT_KNOBS) - set(out)
    if missing:
        raise AssertionError(f"no default for swept knobs {sorted(missing)}")
    return out


@cache
def catalog(arch: str, level: str = "full") -> dict:
    """Legal values of every registered knob on ``arch`` at sweep ``level``
    (before couplings), from the backward tuning registry: one entry per
    registry axis (its :meth:`KnobAxis.values`) plus the two composite axes the
    stages enumerate, ``residency`` (``kv_residency`` x ``kt_source`` pairs
    with ``kt_source = "reg"`` only on register-resident K / V) and
    ``sched_pairs`` (``sched`` x ``setprio``, ``setprio`` only inside a stage
    table)."""
    axes = {a.name: a for a in backward_tuning_axes(arch, level)}
    cat = {name: tuple(a.values(arch, level)) for name, a in axes.items()}
    cat["residency"] = tuple(
        (kv, kt)
        for kv in cat["kv_residency"]
        for kt in cat["kt_source"]
        if kt != "reg" or kv == "reg"
    )
    cat["sched_pairs"] = tuple(
        (s, p)
        for s in cat["sched"]
        for p in cat["setprio"]
        if not p or s in STAGE_TABLE_SCHEDS
    )
    return cat


@dataclass(frozen=True)
class Release:
    """Released lever families and the sweep level (value masks follow).

    ``level = "production"`` enumerates every legacy knob as a product and
    varies each other registered knob on its own around every product point
    (values marked ``full_only`` excluded); ``"full"`` crosses every
    registered knob with every value (the per-shape dispatcher sweep).
    """

    families: frozenset = frozenset({1})
    level: str = "production"

    def __post_init__(self) -> None:
        if self.level not in SWEEP_LEVELS:
            raise ValueError(f"sweep level {self.level!r} not in {SWEEP_LEVELS}")

    def has(self, knob: str) -> bool:
        if knob in NOT_YET_EFFECTIVE_KNOBS:
            return False  # pinned until the emission depends on it
        return KNOB_FAMILY[knob] in self.families

    def values(self, sl: SliceKey, knob: str, default):
        """The value mask of ``knob``: the catalog when released, else the default."""
        if not self.has(knob):
            return (default,)
        return tuple(catalog(sl.arch, self.level).get(knob, (default,)))

    def label(self) -> str:
        return ",".join(str(f) for f in sorted(self.families))

    @classmethod
    def parse(cls, text: str | None, level: str = "production") -> "Release":
        if not text:
            return cls(level=level)
        if text == "all":
            return cls(ALL_RELEASED, level)
        fams = frozenset(int(x) for x in text.split(",") if x.strip())
        bad = fams - set(LEVER_FAMILIES)
        if bad:
            raise ValueError(f"unknown lever families {sorted(bad)}")
        return cls(fams, level)


# --------------------------------------------------------------------------- candidates
def _norm(v):
    if isinstance(v, list):
        return tuple(_norm(x) for x in v)
    return v


@dataclass(frozen=True)
class Candidate:
    """A full knob set of one slice (``knobs``) plus runtime plan values."""

    slice: SliceKey
    knobs: tuple[tuple[str, Any], ...]
    runtime: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def make(
        cls, sl: SliceKey, knobs: Mapping, runtime: Mapping | None = None
    ) -> "Candidate":
        """A candidate; a :data:`KEYED_IF_SET` knob at its default is left out
        (it is the default either way), so earlier keys stay valid."""

        def items(m):
            return tuple(
                sorted(
                    (k, _norm(v))
                    for k, v in m.items()
                    if not (k in KEYED_IF_SET and _norm(v) == KEYED_IF_SET[k])
                )
            )

        return cls(sl, items(knobs), items(runtime or {}))

    @property
    def k(self) -> dict:
        return dict(self.knobs)

    @property
    def r(self) -> dict:
        return dict(self.runtime)

    def with_knobs(self, **kw) -> "Candidate":
        return Candidate.make(self.slice, {**self.k, **kw}, self.r)

    def with_runtime(self, **kw) -> "Candidate":
        return Candidate.make(self.slice, self.k, {**self.r, **kw})

    def retarget(self, sl: SliceKey) -> "Candidate":
        return Candidate.make(sl, self.k, self.r)

    def spec(self) -> AttnBwdSpec:
        """The unresolved spec (the dK / dV mode follows the runtime head split)."""
        cls = self.slice.spec_class()
        if int(self.r.get("g_split", 1)) > 1:
            cls["dkv_mode"] = "atomic"
        if "scale_placement" in self.r and "static_scale" not in self.k:
            # the plan compiles main for the runtime scale placement
            cls["static_scale"] = self.r["scale_placement"]
        # the decode switches the plan sets for its runtime values
        if self.r.get("lb_order") == "interleave":
            cls["lb_interleave"] = True
        if self.r.get("lb_order") == "mirror":
            cls["lb_mirror"] = True
        if self.r.get("xcd_chunk") and self.r.get("num_xcds"):
            cls["xcd_remap"] = True
        if self.r.get("use_worklist"):
            cls["worklist"] = True
        # the dQ kernel's knobs (dq_mode = "split") are not main spec fields
        main = {k: v for k, v in self.k.items() if k not in DQ_KNOBS}
        return AttnBwdSpec(**cls, **main)

    def dq_knobs(self) -> dict:
        """The split dQ kernel's ``AttnBwdDqSpec`` fields of this candidate
        (its XCD chunk is a runtime value)."""
        return {
            k[3:]: v for k, v in self.k.items() if k in DQ_KNOBS and k != "dq_xcd_chunk"
        }

    def spec_fields(self) -> dict:
        """Compile-time knobs (``AttnBwdSpec`` names; JSON form)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.knobs}

    def key(self) -> str:
        blob = json.dumps(
            {
                "arch": self.slice.arch,
                "family": self.slice.family,
                "class": self.slice.spec_class(),
                "knobs": self.spec_fields(),
                "runtime": dict(self.runtime),
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:20]

    def tuning_knobs(self) -> dict:
        """The knobs a configuration hands to the plan as tuning values.

        ``head_pack`` is plan-owned (the plan packs heads when ``s_q * G`` fits
        ``block_m``), so it is passed only by the ``head_pack`` slice, which
        forces it on; a ``head_pack = False`` copied from any other slice would
        switch packing off for every short-query request of the class.
        """
        k = self.k
        if self.slice.kind != "head_pack":
            k.pop("head_pack", None)
        return k

    def config(self):
        from benchmarks.common.attention_bwd_rocke_arm import RockeConfig

        return RockeConfig(self.key(), self.tuning_knobs(), self.r)

    def record(self) -> dict:
        return {
            "key": self.key(),
            "spec": self.spec_fields(),
            "runtime": dict(self.runtime),
            "slice": _slice_dict(self.slice),
        }

    @classmethod
    def from_record(cls, rec: Mapping) -> "Candidate":
        sl = SliceKey(**rec["slice"])
        return cls.make(
            sl,
            {k: _norm(v) for k, v in rec["spec"].items()},
            rec.get("runtime") or {},
        )


def _slice_dict(sl: SliceKey) -> dict:
    return {f: getattr(sl, f) for f in SliceKey.__dataclass_fields__}


# --------------------------------------------------------------------------- legality
@dataclass(frozen=True)
class Legality:
    ok: bool
    rule: str | None = None
    detail: str = ""
    resolved: AttnBwdSpec | None = None
    arch_vgpr: int = 0
    agpr: int = 0
    lds: int = 0
    dq: Any = None  # the resolved split dQ spec (dq_mode = "split")


_RULES = (
    # the exact refusals of resolved specs that fail the resource gate first:
    # their messages quote the failing row, which can name another rule
    ("RESOURCE_REFUSED", "resource_refused"),
    ("LAYOUT_SPILLING_COMBOS", "resource_refused"),
    ("workgroup share", "workgroup_share"),
    ("LDS plan", "lds"),
    ("waves_per_eu", "waves_per_eu"),
    ("agpr_alloc=(0, 0) needs", "vgpr_form"),
    ("registers per wave", "agpr_alloc_share"),
    ("below the AGPR estimate", "agpr_alloc"),
    ("AGPR estimate", "agpr"),
    ("arch-VGPR estimate", "arch_vgpr"),
    ("register estimate", "registers"),
    ("not in (", "catalog"),
    ("on gfx", "catalog"),
)


def rule_token(message: str) -> str:
    """A token for a validator message (``coupling`` when no rule matches)."""
    for needle, tok in _RULES:
        if needle in message:
            return tok
    return "coupling"


@cache
def _legality(sl: SliceKey, knobs: tuple, runtime: tuple) -> Legality:
    """The registry's legality (the family validator, then the emission
    check), plus the register estimates and the LDS footprint."""
    c = Candidate(sl, knobs, runtime)
    try:
        spec = c.spec()
    except ValueError as ex:
        return Legality(False, rule_token(str(ex)), str(ex)[:300])
    v = backward_spec_legality(spec, sl.arch)
    if v.resolved is None:
        return Legality(False, rule_token(v.error), v.error[:300])
    s = v.resolved
    if v.not_built:
        return Legality(False, "not_built", "; ".join(v.not_built)[:300], s)
    dq = None
    dq_knobs = c.dq_knobs()
    if s.dq_mode == "split":
        # the dQ kernel the plan builds for this main spec, validated alike
        try:
            dq = validate_attn_bwd_dq_spec(s.dq_spec(**dq_knobs), sl.arch)
        except ValueError as ex:
            return Legality(False, "dq_" + rule_token(str(ex)), str(ex)[:300], s)
    elif dq_knobs:
        return Legality(False, "coupling", "dQ kernel knobs need dq_mode='split'", s)
    est = register_estimates(s, sl.arch)
    return Legality(
        True,
        None,
        "",
        s,
        est["arch_vgpr"],
        est["agpr"],
        lds_footprint_bytes(s, sl.arch),
        dq,
    )


def legality(c: Candidate) -> Legality:
    return _legality(c.slice, c.knobs, c.runtime)


def effective_key(c: Candidate) -> str:
    """Key of the kernel a legal candidate emits: its resolved spec without the
    not-yet-effective knobs, without the knobs that are only a declaration for
    this spec (``DECLARATION_ONLY``) or change nothing under it
    (``INERT_KNOBS``), and without the fields at their declared default (so a
    new defaulted field does not move the keys), plus the runtime values."""
    lg = legality(c)
    if lg.resolved is None:
        return "illegal:" + c.key()
    r = lg.resolved
    declared = AttnBwdSpec.__dataclass_fields__
    fields = {
        f: getattr(r, f)
        for f in declared
        if f not in NOT_YET_EFFECTIVE_KNOBS and f != "name"
    }
    # "backend" emits what a policy result of None emits
    if fields["agpr_alloc"] == AGPR_ALLOC_BACKEND:
        fields["agpr_alloc"] = None
    for knob, inert in DECLARATION_ONLY.items():
        if inert(r):
            fields.pop(knob)
    for knob, inert in INERT_KNOBS.items():
        if knob in fields and inert(r, c.slice.arch):
            fields.pop(knob)
    fields = {f: v for f, v in fields.items() if v != declared[f].default}
    blob = repr(sorted(fields.items())) + json.dumps(dict(c.runtime), sort_keys=True)
    if lg.dq is not None:
        # split dQ: the dQ kernel is part of the configuration (its name salts
        # every resolved field)
        blob += "|dq:" + lg.dq.kernel_name()
    return hashlib.sha256(f"{c.slice.arch}|{blob}".encode()).hexdigest()[:20]


def _declares(c: Candidate) -> bool:
    """True when ``c`` sets a declaration-only knob that is inert for it."""
    r = legality(c).resolved
    return any(
        bool(getattr(r, knob)) for knob, inert in DECLARATION_ONLY.items() if inert(r)
    )


def dedupe(cands: Iterable[Candidate]) -> tuple[list[Candidate], dict[str, str]]:
    """``(distinct legal candidates, {duplicate key: representative key})``.

    The representative of an effective configuration is the member that makes
    the declaration (``s_in_agpr = True`` where it is declaration-only: its
    hot-loop AGPR-copy budget matches where the backend places S / dP), then
    the smallest key, so the result does not depend on order or shard.
    """
    groups: dict[str, list[Candidate]] = {}
    for c in cands:
        if legality(c).ok:
            groups.setdefault(effective_key(c), []).append(c)
    reps, dup = [], {}
    for members in groups.values():
        members = sorted(members, key=lambda x: (not _declares(x), x.key()))
        reps.append(members[0])
        for m in members[1:]:
            dup[m.key()] = members[0].key()
    reps.sort(key=lambda x: x.key())
    return reps, dup


# --------------------------------------------------------------------------- spaces
def _knob_context(c: Candidate, point: Mapping) -> dict:
    """Spec field values seen by a registry coupling test: the slice's problem
    fields, the candidate's knobs and the stage point on top."""
    return {**c.slice.spec_class(), **c.k, **point}


def _extra_values(
    c: Candidate, rel: Release, name: str, ctx: Mapping, *, coupled: bool = True
) -> list:
    """The values of registry knob ``name`` a stage enumerates on top of
    ``ctx``: the released catalog (the current value alone when the family is
    not released), minus the values the registry marks not built (they are
    never emitted) and, with ``coupled``, the values its coupling test rules
    out on ``ctx``."""
    axis = backward_axis(name)
    cur = ctx.get(name, axis.default)
    vals = rel.values(c.slice, name, cur)
    return [
        v
        for v in vals
        if v == cur
        or (v not in axis.not_built and (not coupled or axis.applies(v, ctx)))
    ]


def with_extras(
    c: Candidate, rel: Release, points: Sequence[Mapping], names: Sequence[str]
) -> list[dict]:
    """``points`` widened by the registry knobs ``names`` of the stage.

    ``production``: every point, then every point with one knob moved to
    each of its other values (the knobs are varied one at a time around every
    point). ``full``: every point crossed with every combination of the knobs
    (combinations whose couplings fail are dropped). A point that already
    sets one of the knobs keeps it as its current value.
    """
    out: list[dict] = []
    full = rel.level == "full"
    for p in points:
        ctx = _knob_context(c, p)
        # full: the couplings are tested per combination (a knob coupled to
        # another knob of the same stage, such as counted_issue to loop_sync,
        # is reachable only through such a combination)
        axes = [
            (n, _extra_values(c, rel, n, ctx, coupled=not full))
            for n in names
            if rel.has(n)
        ]
        if full:
            for combo in itertools.product(*(vals for _, vals in axes)):
                q = {**p, **dict(zip((n for n, _ in axes), combo))}
                qctx = {**ctx, **q}
                if all(backward_axis(n).applies(q[n], qctx) for n, _ in axes):
                    out.append(q)
            continue
        out.append(dict(p))
        for n, vals in axes:
            cur = ctx.get(n, backward_axis(n).default)
            out += [{**p, n: v} for v in vals if v != cur]
    return out


def _pinned_k4(n: int, atom_g4: str, preferred: int) -> int | None:
    a = _atom_k(atom_g4)
    if n % preferred == 0 and preferred % a == 0:
        return preferred
    for k4 in BLOCK_K4_VALUES:
        if n % k4 == 0 and k4 % a == 0:
            return k4
    return None


def _default_atoms(sl: SliceKey, k4: int) -> dict:
    f = attn_bwd_arch_facts(sl.arch)
    if f.matrix_path == "wmma":
        return {"atom_g02": "wmma16x16x16", "atom_g13": "wmma16x16x16"} | {
            "atom_g4": "wmma16x16x16"
        }
    wide = f.wide_k_atom and k4 % 32 == 0 and sl.head_size % 32 == 0
    a = "16x16x32" if wide else "16x16x16"
    return {"atom_g02": a, "atom_g13": "16x16x16", "atom_g4": a}


def geometries(sl: SliceKey, rel: Release) -> list[dict]:
    """Geometry knob sets satisfying the divisibility rules (budgets: later).

    ``W = 1`` only for the 16 x 16 debug / fallback tile; the G4 grid is
    ``(1, W)`` or ``(2, W / 2)``. ``block_k4`` is pinned per geometry (the
    arch default when it divides ``kN0``): under the stage's default
    ``g4_slicing = "step"`` it is inert, and stage H enumerates it together
    with the slicing values (:func:`g4_slicing_values`).
    """
    base = default_knobs(sl)
    cat = catalog(sl.arch)
    ws = cat["waves"] if rel.has("waves") else (base["waves"],)
    ns = cat["block_n"] if rel.has("block_n") else (base["block_n"],)
    ms = cat["block_m"] if rel.has("block_m") else (base["block_m"],)
    out = []
    for w, n, m in itertools.product(ws, ns, ms):
        if w == 1 and (n, m) != (16, 16):
            continue
        # the tile must split over the waves by some grid triple (the natural
        # grids need kN0 % (16 W) == 0; the small-tile grids cover kN0 = 16 /
        # 32 on two and four waves)
        triples = small_tile_grids(sl.head_size, m, n, w)
        if not triples:
            continue
        if rel.has("warp_grid_g4"):
            grids = list(dict.fromkeys(t[2] for t in triples))
        else:
            grids = [tuple(base["warp_grid_g4"])]
        for gr, gc in grids:
            if gr * gc != w or sl.head_size % (16 * gc) or m % (16 * gr):
                continue
            k4 = _pinned_k4(n, "16x16x16", base["block_k4"])
            for k4 in [k4] if k4 else []:
                out.append(
                    {
                        "waves": w,
                        "block_n": n,
                        "block_m": m,
                        "block_k4": k4,
                        "warp_grid_g4": (gr, gc),
                    }
                )
    return out


def _lds_route_grids(d: int, base: Mapping, w: int) -> list[tuple]:
    """The ``(G0 / G2, G1 / G3)`` grid pairs of the LDS route on a geometry:
    every pair of :func:`small_tile_grids` whose G4 grid is the geometry's
    (the natural pair excluded: it is the default). A replicated G0 / G2 grid
    (fewer waves than the CTA) only where no full G0 / G2 grid splits the tile:
    elsewhere replication is dominated (its extra waves compute discarded S /
    dP, and a full grid of the same tile spills less)."""
    g4 = tuple(base["warp_grid_g4"])
    triples = [
        t
        for t in small_tile_grids(d, base["block_m"], base["block_n"], w)
        if t[2] == g4 and t[:2] != natural_grids(w)
    ]
    km, kn = base["block_m"], base["block_n"]
    full = any(
        not replicated_g02(g, w) and km % (16 * g[0]) == 0 and kn % (16 * g[1]) == 0
        for g in g02_grids(w)
    )
    return list(
        dict.fromkeys(t[:2] for t in triples if not (full and replicated_g02(t[0], w)))
    )


def structures(c: Candidate, rel: Release, *, fallback: bool = False) -> list[dict]:
    """Structure knob sets of a candidate's geometry: released families take
    their catalog, the others keep the candidate's values; couplings applied
    while enumerating; budgets and emission by :func:`legality`.

    ``fallback`` (the geometry stage's search for the cheapest legal
    structure, :func:`geometry_stage`) keeps the candidate's ``global_path``
    and ``ring_depth`` and leaves out ``transpose_source = "xt_lds"`` (unless
    the candidate has it). None of them can win that search: the register
    estimate does not decrease with a ring, the DMA path or the XT images
    (a register ring adds the prefetch sets; the DMA path and the images add
    no register term), they only add LDS buffers and couplings (so they make
    no illegal structure legal), and each adds a changed structure knob
    against the same structure without it (ties go to fewer changed knobs,
    then to the smaller LDS plan, where the plain reader beats the images).
    """
    sl, base = c.slice, c.k
    cat = catalog(sl.arch)
    w = base["waves"]
    a02s = cat["atom_g02"] if rel.has("atom_g02") else (base["atom_g02"],)
    a13s = cat["atom_g13"] if rel.has("atom_g13") else (base["atom_g13"],)
    a4s = cat["atom_g4"] if rel.has("atom_g4") else (base["atom_g4"],)
    pts = cat["pt_route"] if rel.has("pt_route") else (base["pt_route"],)
    if rel.has("kv_residency"):
        res = cat["residency"]
    else:
        res = ((base["kv_residency"], base["kt_source"]),)
    tss = rel.values(sl, "transpose_source", base["transpose_source"])
    gps = rel.values(sl, "global_path", base["global_path"])
    rings = rel.values(sl, "ring_depth", base["ring_depth"])
    if fallback:
        tss = [ts for ts in tss if ts != "xt_lds" or ts == base["transpose_source"]]
        gps, rings = (base["global_path"],), (base["ring_depth"],)
    sias = rel.values(sl, "s_in_agpr", base["s_in_agpr"])
    out = []
    for a02, a13, a4, pt in itertools.product(a02s, a13s, a4s, pts):
        if a4.endswith("x32") and base["block_k4"] % 32:
            continue
        if a13.endswith("x32") and base["block_m"] < 32:
            continue
        if rel.has("warp_grid_g02"):
            grids = [((1, w), (w, 1))]
            if pt == "lds" and w >= 2:
                grids += _lds_route_grids(sl.head_size, base, w)
        else:
            grids = [(tuple(base["warp_grid_g02"]), tuple(base["warp_grid_g13"]))]
        for (g02, g13), (kv, kt), ts, gp, ring, sia in itertools.product(
            grids, res, tss, gps, rings, sias
        ):
            if gp == "dma" and ts == "xt_lds":
                continue
            out.append(
                {
                    "atom_g02": a02,
                    "atom_g13": a13,
                    "atom_g4": a4,
                    "pt_route": pt,
                    "warp_grid_g02": g02,
                    "warp_grid_g13": g13,
                    "kv_residency": kv,
                    "kt_source": kt,
                    "transpose_source": ts,
                    "global_path": gp,
                    "ring_depth": ring,
                    "s_in_agpr": sia,
                }
            )
    pts = with_extras(c, rel, out, STRUCTURE_EXTRA)
    dq = dq_values(c, rel)
    if dq == [{}]:
        return pts
    # a point that leaves split dQ drops the candidate's dQ-kernel knobs
    # (back to their defaults, which a candidate key leaves out)
    no_dq = {k: backward_axis(k).default for k in c.k if k in DQ_KNOBS}
    wide = []
    for p in pts:
        if {**c.k, **p}.get("dq_mode") == "split":
            wide += [{**p, **v} for v in dq]
        else:
            wide.append({**p, **no_dq})
    return wide


def dq_values(c: Candidate, rel: Release) -> list[dict]:
    """The split dQ kernel's knob sets of a stage T split point: the dQ
    kernel's defaults, then each dQ field (:data:`DQ_EXTRA`) moved to each of
    its other built values on its own, at both levels (a crossed dQ space
    multiplies every split point by thousands; it is the job of a nested dQ
    stage on the best split candidates, which is not built). A wave count
    also comes paired with ``dq_block_m = 16 * waves`` (the geometry needs
    ``16 * waves | block_m``, so a wave count alone is illegal where the start
    tile is too short, e.g. eight waves on 64 rows). The validators
    (:func:`legality`: main and dQ kernel) prune. ``[{}]`` when family 8 is
    not released."""
    if not rel.has("dq_mode"):
        return [{}]
    out: list[dict] = [{}]
    for n in DQ_EXTRA:
        axis = backward_axis(n)
        out += [
            {n: v}
            for v in rel.values(c.slice, n, axis.default)
            if v != axis.default and v not in axis.not_built
        ]
    m_axis = backward_axis("dq_block_m")
    rows = set(rel.values(c.slice, "dq_block_m", m_axis.default)) - set(
        m_axis.not_built
    )
    out += [
        {"dq_waves": v["dq_waves"], "dq_block_m": 16 * v["dq_waves"]}
        for v in list(out)
        if set(v) == {"dq_waves"} and 16 * v["dq_waves"] in rows
    ]
    return out


def _unreachable_hints(c: Candidate, wpes) -> set:
    """``waves_per_eu`` hints (> 1) the candidate's LDS plan keeps below
    (:func:`attn_bwd_occupancy`'s ``hint_reachable``): the occupancy the hint
    asks for is out of reach, so the point is not enumerated (the validator's
    register budget rules the reachable ones)."""
    if attn_bwd_arch_facts(c.slice.arch).matrix_path != "mfma":
        return set()
    out = set()
    for wpe in wpes:
        if wpe is None or wpe <= 1:
            continue
        lg = legality(c.with_knobs(waves_per_eu=wpe))
        if lg.ok and not attn_bwd_occupancy(lg.resolved, c.slice.arch).hint_reachable:
            out.add(wpe)
    return out


def schedule_values(c: Candidate, rel: Release, *, extras: bool = True) -> list[dict]:
    """Stage H knob sets: sched x setprio x waves_per_eu x agpr_alloc x
    agpr_route, widened by the stage's other registry knobs
    (:data:`SCHEDULE_EXTRA`, see :func:`with_extras`; ``extras=False`` gives
    the product alone).

    ``agpr_alloc`` keeps four values at most: ``None`` (the policy),
    ``"backend"`` (no reservation even where the policy picks the VGPR form),
    ``(0, 0)`` and the smallest legal ``(n, n)`` reservation (a larger one only
    lowers the occupancy the backend may choose); duplicates of the resolved
    kernel are removed later by :func:`dedupe`. ``agpr_route`` (a split AGPR
    plan) needs a reservation, so its values cross the ``(n, n)`` allocation
    only; every other point keeps the default (attr) route.
    """
    sl, base = c.slice, c.k
    cat = catalog(sl.arch)
    pairs = cat["sched_pairs"] if rel.has("sched") else ((base["sched"], False),)
    wpes = rel.values(sl, "waves_per_eu", base["waves_per_eu"])
    if rel.has("agpr_alloc"):
        allocs: list = [None, AGPR_ALLOC_BACKEND, (0, 0)]
        sized = [a for a in cat["agpr_alloc"] if isinstance(a, tuple) and a != (0, 0)]
        for a in sized:
            if legality(c.with_knobs(agpr_alloc=a)).rule not in (
                "agpr_alloc",
                "agpr",
                "agpr_alloc_share",
            ):
                allocs.append(a)
                break
    else:
        allocs = [base["agpr_alloc"]]
    route_default = backward_axis("agpr_route").default
    routes = rel.values(sl, "agpr_route", base.get("agpr_route", route_default))
    out = []
    unreachable = _unreachable_hints(c, wpes)
    for (sched, prio), wpe, alloc in itertools.product(pairs, wpes, allocs):
        if wpe in unreachable:
            continue
        point = {"sched": sched, "setprio": prio, "waves_per_eu": wpe}
        point["agpr_alloc"] = alloc
        sized = isinstance(alloc, tuple) and alloc[0] > 0
        for route in routes if sized else (route_default,):
            out.append({**point, "agpr_route": route})
    g4 = g4_slicing_values(c, rel)
    if len(g4) > 1:
        # the (g4_slicing, block_k4) pair is one knob of the stage: crossed
        # with every point (full) or moved alone around every point
        if rel.level == "full":
            out = [{**p, **q} for p in out for q in g4]
        else:
            out = out + [{**p, **q} for p in out for q in g4[1:]]
    return with_extras(c, rel, out, SCHEDULE_EXTRA) if extras else out


def g4_slicing_values(c: Candidate, rel: Release) -> list[dict]:
    """``(g4_slicing, block_k4)`` points of stage H, the candidate's own first.

    ``block_k4`` only sizes the G4 dS key slices, so it is enumerated with the
    slicing values: every non-default ``g4_slicing`` the registry admits on top
    of the candidate (MFMA register accumulators, ``dq_issue = "block"``, G3
    right before G4 for ``"slice_db_g3"``) at every kK4 that is a multiple of
    the G4 atom K and divides ``kN0``. Points that emit the same code (a
    one-step ``"slice"``, a one-slice ``"slice_db"``) resolve to one spec and
    are merged by :func:`dedupe`.
    """
    base = c.k
    cur = {
        "g4_slicing": base.get("g4_slicing", backward_axis("g4_slicing").default),
        "block_k4": base["block_k4"],
    }
    if not rel.has("g4_slicing"):
        return [cur]
    ctx = _knob_context(c, {})
    axis = backward_axis("g4_slicing")
    k4a, n = _atom_k(base["atom_g4"]), base["block_n"]
    k4s = [k for k in catalog(c.slice.arch)["block_k4"] if n % k == 0 and k % k4a == 0]
    out = [cur]
    for v in rel.values(c.slice, "g4_slicing", cur["g4_slicing"]):
        if v == axis.default or v in axis.not_built or not axis.applies(v, ctx):
            continue
        out += [
            {"g4_slicing": v, "block_k4": k}
            for k in k4s
            if (v, k) != (cur["g4_slicing"], cur["block_k4"])
        ]
    return out


def minor_values(c: Candidate, rel: Release, *, extras: bool = True) -> list[dict]:
    """Stage K knob sets: codegen strategy x edge_tiles x ws_layout x
    acc_in_lds, widened by the stage's other registry knobs
    (:data:`CODEGEN_EXTRA`: step order, dK / dV epilogue)."""
    sl, base = c.slice, c.k
    out = [
        {"scheduler_strategy": cg, "edge_tiles": et, "ws_layout": ws, "acc_in_lds": acc}
        for cg, et, ws, acc in itertools.product(
            rel.values(sl, "scheduler_strategy", base["scheduler_strategy"]),
            rel.values(sl, "edge_tiles", base["edge_tiles"]),
            rel.values(sl, "ws_layout", base["ws_layout"]),
            rel.values(sl, "acc_in_lds", base["acc_in_lds"]),
        )
    ]
    return with_extras(c, rel, out, CODEGEN_EXTRA) if extras else out


def swizzle_ties(c: Candidate) -> list[tuple[str, tuple[str, ...]]]:
    """Buffers whose LDS swizzle the conflict predictor cannot decide.

    The predictor exists for the XT images (``choose_xt_swizzle`` with the
    image's reading atom, as the planner calls it): a buffer enters with every
    candidate tied with the best on (read, write) multiplicity. Row-major and
    DMA buffers have no predictor in this tree, so they never tie (their
    swizzle stays the planner's choice; stage S moves the row-major ones).
    """
    lg = legality(c)
    if not lg.ok:
        return []
    from kernels.common._attention_bwd_lds import (
        XtLayout,
        choose_xt_swizzle,
        default_xt_writer,
    )

    plan = lds_plan(lg.resolved, c.slice.arch)
    threads = lg.resolved.waves * attn_bwd_arch_facts(c.slice.arch).wave_size
    out = []
    for name, obj in plan.layouts:
        if not isinstance(obj, XtLayout):
            continue
        writer = default_xt_writer(obj.cols, obj.rows, threads)
        atom_k = 4 * obj.chunk_bytes // obj.elem_bytes  # the planner's chunk rule
        best, table = choose_xt_swizzle(
            obj.rows, obj.cols, writer, c.slice.arch, atom_k=atom_k
        )
        pb = table[best].get("prediction")
        if pb is None:
            continue
        tied = tuple(
            sw
            for sw, t in table.items()
            if "prediction" in t
            and t["prediction"] is not None
            and (t["prediction"]["read"], t["prediction"]["write"])
            == (pb["read"], pb["write"])
        )
        if len(tied) > 1:
            out.append((name, tied))
    return sorted(out)


def _swizzle_map(text: str | None) -> dict[str, str]:
    out = {}
    for item in (text or "").split(","):
        name, sep, val = item.partition("=")
        if sep:
            out[name.strip()] = val.strip()
    return out


def _swizzle_text(m: Mapping[str, str]) -> str | None:
    return ",".join(f"{k}={m[k]}" for k in sorted(m)) or None


def pad_buffers(c: Candidate) -> list[str]:
    """Row-major LDS buffers of a legal candidate whose row pad stage S
    enumerates, in :data:`PAD_BUFFERS` order: every present buffer with a
    row-major layout, except ``dkv_out`` where the epilogue never reads it
    (atomic dK / dV)."""
    lg = legality(c)
    if not lg.ok:
        return []
    plan = lds_plan(lg.resolved, c.slice.arch)
    present = {b.name for b in plan.buffers if b.layout.startswith("rowmajor:")}
    if lg.resolved.dkv_mode == "atomic":
        present.discard("dkv_out")
    return [n for n in PAD_BUFFERS if n in present]


def swizzle_values(c: Candidate, rel: Release) -> list[dict]:
    """Stage S knob sets of one candidate (on top of its predictor ties).

    Row layouts (``lds_swizzle`` per row-major buffer, values from the
    registry: the row pads and the line ``xor``): ``production`` moves one
    buffer's value at a time, over every present buffer; ``full`` crosses the first
    :data:`MAX_TIED_BUFFERS` present buffers and moves each further one on its
    own. ``lds_order`` and the stage's other registry knobs
    (:data:`SWIZZLE_EXTRA`) are added by :func:`with_extras`.
    """
    base = _swizzle_map(c.k.get("lds_swizzle"))
    points: list[dict] = [{}]
    if rel.has("lds_swizzle"):
        axis = backward_axis("lds_swizzle")
        pads = [v for v in rel.values(c.slice, "lds_swizzle", None) if v is not None]
        pads = [v for v in pads if v not in axis.not_built]
        bufs = pad_buffers(c)
        crossed = bufs[:MAX_TIED_BUFFERS] if rel.level == "full" else []
        singles = [b for b in bufs if b not in crossed]

        def padded(updates) -> dict:
            # "none" is a row-major buffer's default: spelled by omission
            m = {**base, **updates}
            return {"lds_swizzle": _swizzle_text(
                {k: v for k, v in m.items() if not (k in updates and v == "none")}
            )}  # fmt: skip

        for combo in itertools.product(*([pads] * len(crossed))):
            if any(base.get(b, "none") != v for b, v in zip(crossed, combo)):
                points.append(padded(dict(zip(crossed, combo))))
        for bname in singles:
            cur = base.get(bname, "none")
            points += [padded({bname: v}) for v in pads if v != cur]
    return with_extras(c, rel, points, SWIZZLE_EXTRA)


def _structure_cands(
    c: Candidate, rel: Release, *, fallback: bool = False
) -> list[Candidate]:
    return [c.with_knobs(**s) for s in structures(c, rel, fallback=fallback)]


@dataclass
class GeometryStage:
    candidates: list
    fallback: int = 0  # geometries whose default structure is illegal
    illegal: int = 0  # geometries with no legal structure at all


def geometry_stage(sl: SliceKey, rel: Release) -> GeometryStage:
    """Every legal geometry with the default structure (or, where that is
    illegal, the cheapest legal structure over every built structure value,
    released or not; see :func:`fallback_order`)."""
    out = GeometryStage([])
    base = default_knobs(sl)
    for g in geometries(sl, rel):
        knobs = {**base, **g, **_default_atoms(sl, g["block_k4"])}
        knobs["warp_grid_g02"] = (1, g["waves"])
        knobs["warp_grid_g13"] = (g["waves"], 1)
        c = Candidate.make(sl, knobs)
        if legality(c).ok:
            out.candidates.append(c)
            continue
        legal = [
            x
            for x in _structure_cands(c, Release(ALL_RELEASED), fallback=True)
            if legality(x).ok
        ]
        if not legal:
            out.illegal += 1
            continue
        out.fallback += 1
        out.candidates.append(min(legal, key=lambda x: fallback_order(x, c)))
    return out


def fallback_order(x: Candidate, default: Candidate) -> tuple:
    """Sort key of a fallback structure ``x`` for the geometry of ``default``.

    Fewest arch VGPRs first, then fewest AGPRs, then the structure nearest the
    arch's portable default (fewest structure knobs changed), then the smaller
    LDS plan; the candidate key only breaks exact ties. The arch-VGPR estimate
    alone leaves tens of structures tied on some geometries.
    """
    lg = legality(x)
    changed = sum(x.k.get(k) != default.k.get(k) for k in STRUCTURE_KNOBS)
    return (lg.arch_vgpr, lg.agpr, changed, lg.lds, x.key())


# --------------------------------------------------------------------------- stages
def _runtime_built(sl: SliceKey, rel: Release, name: str) -> list:
    """Registry values of runtime knob ``name`` the plan builds."""
    axis = backward_axis(name)
    return [v for v in rel.values(sl, name, axis.default) if v not in axis.not_built]


def _runtime_values(sl: SliceKey, cells: Sequence, rel: Release) -> list[dict]:
    """Stage R values: G_split divisors x lb_order x scale placement (built
    values only; the plan refuses the others), widened by the prep / convert
    tuning values (:data:`RUNTIME_EXTRA`, as :func:`with_extras` does: one at
    a time at ``production``, crossed at ``full``)."""
    if not rel.has("g_split"):
        return [{}]
    groups = sorted({c.hq // c.hkv for c in cells}) or [1]
    splits = sorted({g for grp in groups for g in range(1, grp + 1) if grp % g == 0})
    orders = _runtime_built(sl, rel, "lb_order")
    if sl.mask_class == "none":
        # mirror pairing needs a band (the plan refuses it without a mask)
        orders = [o for o in orders if o != "mirror"]
    out = []
    for gs, lb, sp in itertools.product(
        splits, orders, _runtime_built(sl, rel, "scale_placement")
    ):
        p = {"g_split": gs, "lb_order": lb, "scale_placement": sp}
        if lb == "mirror":
            # the plan pairs only a grid that fills the device: the arm reads
            # the CU count of the device it runs on (a request hint)
            p["num_cus"] = DEVICE_NUM_CUS
        out.append(p)
    # THD slices: the work list on each natural-order point (it fixes the CTA
    # order, so the plan refuses it with another lb_order)
    wl_axis = backward_axis("use_worklist")
    if rel.has("use_worklist"):
        out += [
            {**p, "use_worklist": v}
            for p in list(out)
            for v in _runtime_built(sl, rel, "use_worklist")
            if v != wl_axis.default and wl_axis.applies(v, {**p, **sl.spec_class()})
        ]
    # XCD remap (the request's XCD count rides along as a device hint, see
    # attention_bwd_rocke_arm.REQUEST_HINTS).
    extras = [(n, _runtime_built(sl, rel, n)) for n in RUNTIME_EXTRA if rel.has(n)]
    if rel.has("xcd_chunk") and sl.arch in SWEEP_NUM_XCDS:
        hint = {"num_xcds": SWEEP_NUM_XCDS[sl.arch]}
        chunks = [
            {"xcd_chunk": v, **hint} if v else {}
            for v in _runtime_built(sl, rel, "xcd_chunk")
        ]
        extras.append(("xcd", chunks))

    def point(n, v) -> dict:
        return v if n == "xcd" else {n: v}

    if rel.level == "full":
        return [
            {**p, **{k: v for n, x in zip((n for n, _ in extras), combo)
                     for k, v in point(n, x).items()}}
            for p in out
            for combo in itertools.product(*(v for _, v in extras))
        ]  # fmt: skip
    wide = []
    for p in out:
        wide.append(p)
        for n, vals in extras:
            default = {} if n == "xcd" else backward_axis(n).default
            wide += [{**p, **point(n, v)} for v in vals if v != default]
    return wide


def _dq_xcd_values(sl: SliceKey, rel: Release) -> list[dict]:
    """Stage R values of the split dQ kernel's XCD chunk (non-default ones,
    each with the request's XCD count as a device hint)."""
    if not rel.has("dq_xcd_chunk") or sl.arch not in SWEEP_NUM_XCDS:
        return []
    hint = {"num_xcds": SWEEP_NUM_XCDS[sl.arch]}
    return [
        {"dq_xcd_chunk": v, **hint}
        for v in _runtime_built(sl, rel, "dq_xcd_chunk")
        if v
    ]


def stage_candidates(
    stage: str,
    sl: SliceKey,
    *,
    rel: Release,
    prior: Mapping[str, Sequence[Candidate]] | None = None,
    cells: Sequence = (),
) -> list[Candidate]:
    """Candidates of a funnel stage from the ranked keep lists of earlier stages.

    ``prior`` maps a stage word to its ranked candidates (best first).
    """
    prior = prior or {}
    if stage == "geometry":
        return geometry_stage(sl, rel).candidates
    if stage == "structure":
        top = prior.get("geometry", ())[:STRUCTURE_FROM_GEOMETRIES]
        return [x for c in top for x in _structure_cands(c, rel)]
    if stage == "schedule":
        top = prior.get("structure", ())[: KEEP["structure"]]
        if sl.kind != "primary":
            top = [c.retarget(sl) for c in prior.get("primary", ())]
        return [c.with_knobs(**h) for c in top for h in schedule_values(c, rel)]
    if stage == "codegen":
        top = prior.get("schedule", ())[: KEEP["schedule"]]
        return [c.with_knobs(**m) for c in top for m in minor_values(c, rel)]
    if stage == "swizzle":
        top = list(prior.get("codegen", ())[: KEEP["codegen"]])
        tied = []
        for c in top:
            ties = swizzle_ties(c)[:MAX_TIED_BUFFERS]
            if not ties:
                tied.append(c)
                continue
            names = [n for n, _ in ties]
            for combo in itertools.product(*(vals for _, vals in ties)):
                tied.append(
                    c.with_knobs(
                        lds_swizzle=",".join(f"{n}={v}" for n, v in zip(names, combo))
                    )
                )
        out = []
        for c in tied:
            out += [c.with_knobs(**p) for p in swizzle_values(c, rel)]
        return list(dict.fromkeys(out))
    if stage == "alternate":
        # stage T re-run on its top 16 structures with the H / K / S winners
        winner = (prior.get("swizzle") or prior.get("codegen") or ())[:1]
        structs = prior.get("structure", ())[:ALTERNATE_STRUCTURES]
        if not winner:
            return []
        w = winner[0].k
        keys = SCHEDULE_KNOBS + MINOR_KNOBS + ("lds_swizzle",)
        apply = {k: w[k] for k in keys if k in w}
        # registry knobs of H / K / S: the winner's value or the default
        for k in ("agpr_route",) + SCHEDULE_EXTRA + CODEGEN_EXTRA + SWIZZLE_EXTRA:
            apply[k] = w.get(k, KEYED_IF_SET[k])
        return [c.with_knobs(**apply) for c in structs]
    if stage == "alternate_schedule":
        top = prior.get("alternate", ())[: KEEP["structure"]]
        return [c.with_knobs(**h) for c in top for h in schedule_values(c, rel)]
    if stage == "runtime":
        top = (
            prior.get("alternate_schedule")
            or prior.get("alternate")
            or prior.get("swizzle")
            or prior.get("codegen")
            or prior.get("schedule")
            or ()
        )[: KEEP["alternate"]]
        vals = _runtime_values(sl, cells, rel)
        out = [c.with_runtime(**v) for c in top for v in vals]
        # split-dQ candidates: the dQ kernel's XCD chunk, one value at a time
        dq_xcd = _dq_xcd_values(sl, rel)
        out += [
            c.with_runtime(**v)
            for c in top
            if c.k.get("dq_mode") == "split"
            for v in dq_xcd
        ]
        return out
    if stage == "confirm":
        return list(prior.get("runtime", ())[: KEEP["runtime"]])
    raise ValueError(f"unknown stage {stage!r}")


# --------------------------------------------------------------------------- counts
@dataclass
class SliceCount:
    slice: SliceKey
    release: Release
    geometries: int = 0
    fallback: int = 0
    structures_max: int = 0
    product: int = 0  # distinct legal, built compile specs over the released space
    not_built: int = 0
    stages: dict = field(default_factory=dict)


def count_slice(sl: SliceKey, rel: Release) -> SliceCount:
    """Counts from the validator over the released space (see module doc).

    ``product`` is the number of distinct legal and built kernels of the
    released legacy-knob space (geometry x structure x schedule x minor
    products, deduplicated; ``product_scope=legacy`` in the count line): the
    registry knobs the stages add (:func:`with_extras`) multiply it beyond
    what an enumeration can count. The stage bounds follow the keep widths
    and include those knobs: the schedule bound from every structure's stage
    H set, the codegen bound from the stage K set of each structure's first
    stage H point, the swizzle bound from the stage S set of each structure.
    """
    out = SliceCount(sl, rel)
    gs = geometry_stage(sl, rel)
    out.geometries, out.fallback = len(gs.candidates), gs.fallback
    per_geom, h_max, k_max, s_max = [], 0, 0, 1
    seen: set = set()
    for g in gs.candidates:
        structs = [x for x in _structure_cands(g, rel)]
        legal = []
        for s in structs:
            lg = legality(s)
            if lg.rule == "not_built":
                out.not_built += 1
            if lg.ok:
                legal.append(s)
        legal, _ = dedupe(legal)
        per_geom.append(len(legal))
        for s in legal:
            hs_all, _ = dedupe(s.with_knobs(**h) for h in schedule_values(s, rel))
            h_max = max(h_max, len(hs_all))
            if hs_all:
                k_all, _ = dedupe(
                    hs_all[0].with_knobs(**m) for m in minor_values(hs_all[0], rel)
                )
                k_max = max(k_max, len(k_all))
            s_max = max(s_max, len(swizzle_values(s, rel)))
            hs, _ = dedupe(
                s.with_knobs(**h) for h in schedule_values(s, rel, extras=False)
            )
            for h in hs:
                ms, _ = dedupe(
                    h.with_knobs(**m) for m in minor_values(h, rel, extras=False)
                )
                k_max = max(k_max, len(ms))
                for m in ms:
                    seen.add(effective_key(m))
    out.product = len(seen)
    out.structures_max = max(per_geom, default=0)
    t_bound = sum(sorted(per_geom, reverse=True)[:STRUCTURE_FROM_GEOMETRIES])
    h_bound = KEEP["structure"] * h_max
    s_bound = KEEP["codegen"] * max(
        s_max, 2 ** MAX_TIED_BUFFERS if rel.has("lds_swizzle") else 1
    )
    out.stages = {
        "geometry": out.geometries,
        "structure": t_bound,
        "schedule": h_bound,
        "codegen": KEEP["schedule"] * k_max,
        "swizzle": s_bound if rel.has("lds_swizzle") else 0,
        "alternate": ALTERNATE_STRUCTURES + h_bound,
        "runtime": 0,
        "confirm": 0,
    }
    return out


def total_compiles(sc: SliceCount) -> int:
    return sum(v for k, v in sc.stages.items() if k not in ("runtime", "confirm"))


# --------------------------------------------------------------------------- plan
@dataclass(frozen=True)
class WallInputs:
    """Per-arch wall-budget inputs (seconds), read by name from a private file."""

    t_compile: float  # one main-kernel compile, single thread
    compile_speedup: float  # measured speed-up of the in-job pool
    t_run: float  # one arm in one attempt: gate + timing windows
    t_attempt: float  # fresh-process start-up of one attempt
    t_reference: float  # reference process per (cell, job)
    t_gate: float  # pruning set per rocKE arm per attempt
    survival: float = 1.0  # measured fraction of compiled specs that are timed

    @classmethod
    def from_mapping(cls, m: Mapping) -> "WallInputs":
        return cls(
            float(m["t_compile_serial"]),
            float(m.get("compile_speedup", 1.0)),
            float(m["t_run"]),
            float(m.get("t_attempt", 0.0)),
            float(m["t_reference"]),
            float(m.get("t_gate", 0.0)),
            float(m.get("survival", 1.0)),
        )


def plan_jobs(
    n_compiled: int,
    n_timed: int,
    w: WallInputs,
    *,
    n_cells: int,
    attempts: int,
    bars: int = 1,
    arms_per_attempt: int = 12,
    job_cap_s: float = 1200.0,
    margin_s: float = 180.0,
) -> int:
    """Jobs from the wall-budget formula (compile wall + timing wall).

    Timing: survivors go ``arms_per_attempt`` at a time, each group paired
    with ``bars`` competitor arms in the same attempts; every group pays the
    attempt start-ups, every rocKE arm pays the pruning-set gate, and every
    (cell, job) pays one reference process.
    """
    compile_wall = n_compiled * w.t_compile / max(1.0, w.compile_speedup)
    groups = math.ceil(n_timed / max(1, arms_per_attempt)) if n_timed else 0
    per_attempt = lambda arms: w.t_attempt + arms * w.t_run  # noqa: E731
    time_wall = 0.0
    for gi in range(groups):
        arms = min(arms_per_attempt, n_timed - gi * arms_per_attempt)
        time_wall += n_cells * attempts * (per_attempt(arms + bars) + arms * w.t_gate)
    time_wall += n_cells * w.t_reference if groups else 0.0
    total = compile_wall + time_wall
    return max(1, math.ceil(total / (job_cap_s - margin_s)))


def _load_plan_consts(path: str | None) -> dict | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("per_arch", data)


# --------------------------------------------------------------------------- cells
def _census_cells():
    from benchmarks.common.attention_bwd_bench import census_cells, load_census

    return census_cells(load_census())


def cell_in_slice(cell, sl: SliceKey) -> bool:
    band = cell.mask != "none"
    thd = cell.is_thd
    atomic = cell.h_v != cell.hkv
    return (
        cell.d == sl.head_size
        and cell.dtype == sl.dtype
        and band == (sl.mask_class == "band")
        and thd == (sl.seq_mode == "thd")
        and atomic == (sl.dkv_mode == "atomic")
        and cell.stage_vec == sl.stage_vec
    )


def skv_bucket(cell) -> str:
    s = max(cell.kv_lengths) if cell.is_thd else cell.skv
    return "short" if s <= 1024 else ("medium" if s <= 4096 else "long")


def representative_cells(sl: SliceKey, cells: Sequence | None = None) -> list:
    """Up to three core cells of the slice, one per ``S_kv`` bucket (highest
    weight, then census order); census cells of the slice when it has no core
    cell; class representatives when it has no census cell."""
    from benchmarks.common.attention_bwd_bench import class_representatives

    cells = list(cells) if cells is not None else _census_cells()
    pool = [c for c in cells if c.core and cell_in_slice(c, sl)]
    if not pool:
        pool = [c for c in cells if cell_in_slice(c, sl)]
    if not pool:
        pool = [c for c in class_representatives(cells) if cell_in_slice(c, sl)]
    out = {}
    for c in sorted(pool, key=lambda c: -c.weight):  # stable: census order on ties
        out.setdefault(skv_bucket(c), c)
    order = ("short", "medium", "long")
    return [out[b] for b in order if b in out]


def cohort_cells(sl: SliceKey, cells: Sequence | None = None) -> list:
    """The family cohort of a slice: every census cell of its head size."""
    cells = list(cells) if cells is not None else _census_cells()
    return [c for c in cells if c.d == sl.head_size]


# --------------------------------------------------------------------------- records
def stage_dir(out: Path, sl: SliceKey, stage: str) -> Path:
    return out / sl.tag() / stage


def shard_path(out: Path, sl: SliceKey, stage: str, shard: tuple[int, int]) -> Path:
    i, n = shard
    return stage_dir(out, sl, stage) / f"shard-{i}of{n}.jsonl"


def read_records(out: Path, sl: SliceKey, stage: str) -> dict[str, dict]:
    """Merged records of every shard file of a stage (last record per key)."""
    recs: dict[str, dict] = {}
    d = stage_dir(out, sl, stage)
    if not d.is_dir():
        return recs
    for p in sorted(d.glob("shard-*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "key" in r:
                recs[r["key"]] = r
    return recs


def ranked(out: Path, sl: SliceKey, stage: str, keep: int | None = None) -> list:
    """Timed candidates of a stage, best score first (ties by key)."""
    recs = [
        r
        for r in read_records(out, sl, stage).values()
        if r.get("status") == "timed" and r.get("score") is not None
    ]
    recs.sort(key=lambda r: (r["score"], r["key"]))
    if keep is not None:
        recs = recs[:keep]
    return [Candidate.from_record(r) for r in recs]


def confirm_verdicts(out: Path, sl: SliceKey) -> dict[str, dict]:
    """Per finalist key: the cohort cell verdict counts of the confirm stage."""
    res: dict[str, dict] = {}
    for key, r in read_records(out, sl, "confirm").items():
        cand, _, cell = key.partition("|")
        if not cell:
            continue
        res.setdefault(cand, {})[cell] = r.get("verdict") or r.get("status")
    out_counts = {}
    for cand, cells in res.items():
        cnt: dict[str, int] = {}
        for v in cells.values():
            cnt[str(v)] = cnt.get(str(v), 0) + 1
        out_counts[cand] = {"cells": len(cells), "verdicts": cnt}
    return out_counts


def alternate_changed(out: Path, sl: SliceKey) -> bool:
    """Whether stage A moved the top configuration to another geometry or
    structure than the winner it started from (then stage H runs once more,
    as ``alternate_schedule``, on A's top 8)."""
    a = ranked(out, sl, "alternate", 1)
    w = ranked(out, sl, "swizzle", 1) or ranked(out, sl, "codegen", 1)
    if not a or not w:
        return False
    keys = GEOMETRY_KNOBS + STRUCTURE_KNOBS
    return {k: a[0].k.get(k) for k in keys} != {k: w[0].k.get(k) for k in keys}


def shard_of(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % n


def append_jsonl(path: Path, rec: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# --------------------------------------------------------------------------- scoring
def score_from_summaries(key: str, summaries: Sequence[Mapping]) -> dict:
    """Score of one configuration over its cells: the geometric mean of
    ``M_rocke / M_bar`` (lower is better), the per-cell verdicts and the
    worst status; ``score`` is ``None`` unless every cell was timed."""
    from benchmarks.common.attention_bwd_bench import rocke_arm_name

    arm = rocke_arm_name(key)
    ratios, cells, worst = [], {}, None
    for s in summaries:
        a = s.get("arms", {}).get(arm)
        bar = s.get("arms", {}).get(s.get("bar_arm") or "", {})
        verdict = (s.get("verdicts") or {}).get(arm)
        entry = {"verdict": verdict, "bar_arm": s.get("bar_arm")}
        if a is None:
            entry["status"] = "missing"
        else:
            entry["status"] = a.get("status")
            if a.get("status") == "ok":
                entry.update(
                    median_ms=a["median_ms"],
                    spread=a["spread"],
                    attempt_medians_ms=a["attempt_medians_ms"],
                )
                if bar.get("status") == "ok":
                    entry.update(
                        bar_median_ms=bar["median_ms"], bar_spread=bar["spread"]
                    )
                    ratios.append(a["median_ms"] / bar["median_ms"])
        if entry["status"] != "ok" and worst is None:
            worst = entry["status"]
        cells[s["id"]] = entry
    score = None
    if ratios and len(ratios) == len(summaries) and worst is None:
        score = math.exp(statistics.fmean(math.log(r) for r in ratios))
    return {"score": score, "cells": cells, "worst": worst}


# --------------------------------------------------------------------------- driver
@dataclass
class Options:
    out: Path
    arch: str
    release: Release
    shard: tuple[int, int] = (0, 1)
    cpus: int = 16
    compile_mode: str = "thread"
    compile_cache: Path | None = None
    gate: str | None = None
    gate_constants: str | None = None
    attempts: int = 6
    arms_per_attempt: int = 12
    deadline_s: float = 1e9
    expect_flavor: str | None = None
    expect_comgr: str | None = None
    dry_run: bool = False
    seed: int = 42


def _stdout(line: str) -> None:
    print(line, flush=True)


class Driver:
    """Runs one funnel stage of one slice in this job (see module doc)."""

    def __init__(self, opts: Options) -> None:
        self.o = opts
        self.t0 = time.time()
        self._cache = None
        self._gate = None
        self._toolchain = None

    # -- helpers ------------------------------------------------------------
    def remaining(self) -> float:
        return self.o.deadline_s - (time.time() - self.t0)

    def toolchain(self):
        if self._toolchain is None:
            from benchmarks.common.attention_bwd_compile import toolchain_identity

            self._toolchain = toolchain_identity()
        return self._toolchain

    def cache(self):
        if self._cache is None:
            from benchmarks.common.attention_bwd_compile import (
                CompileCache,
                backward_source_hash,
            )

            root = self.o.compile_cache or (self.o.out / "cache")
            self._cache = CompileCache(
                root, toolchain=self.toolchain(), source_hash=backward_source_hash()
            )
        return self._cache

    def gate(self):
        if self._gate is None and self.o.gate:
            from benchmarks.common.attention_bwd_bench import load_gate

            self._gate = load_gate(self.o.gate)
        return self._gate

    def base_record(self, sl: SliceKey, stage: str) -> dict:
        tc = self.toolchain() if not self.o.dry_run else None
        return {
            "stage": stage,
            "released": sorted(self.o.release.families),
            "level": self.o.release.level,
            "keep": KEEP.get(stage),
            "not_yet_effective": sorted(NOT_YET_EFFECTIVE_KNOBS),
            "toolchain": tc.record() if tc else None,
        }

    # -- prior stages ---------------------------------------------------------
    def prior(self, sl: SliceKey) -> dict:
        out = {}
        for st in STAGE_WORDS + ("alternate_schedule",):
            if st in KEEP:
                out[st] = ranked(self.o.out, sl, st, None)
        if sl.kind != "primary":
            prim = replace(
                sl,
                kind="primary",
                dtype="bf16",
                mask_class="band",
                seq_mode="batched",
                dkv_mode="direct",
                stage_vec=8,
            )
            src = (
                ranked(self.o.out, prim, "alternate_schedule", None)
                or ranked(self.o.out, prim, "alternate", None)
                or ranked(self.o.out, prim, "structure", None)
            )
            out["primary"] = src[:SECONDARY_FROM_PRIMARY]
        return out

    # -- stage ------------------------------------------------------------------
    def run_stage(self, sl: SliceKey, stage: str) -> dict:
        cells = cohort_cells(sl) if stage == "confirm" else representative_cells(sl)
        prior = self.prior(sl)
        cands = stage_candidates(
            stage, sl, rel=self.o.release, prior=prior, cells=cells
        )
        path = shard_path(self.o.out, sl, stage, self.o.shard)
        done = read_records(self.o.out, sl, stage)
        counts = {
            "candidates": len(cands),
            "cells": len(cells),
            "skipped_done": 0,
            "illegal": 0,
            "not_built": 0,
            "duplicate": 0,
            "compile_fail": 0,
            "pruned": 0,
            "correctness_fail": 0,
            "unsupported": 0,
            "timed": 0,
            "deadline": 0,
        }
        base = self.base_record(sl, stage)
        i, n = self.o.shard
        if stage == "confirm":
            # the cohort is split over shards; every shard times every finalist
            cells = cells[i::n]
            counts["cells"] = len(cells)
            mine, reps, dup = list(cands), list(cands), {}
            have = set(done)
            done = {
                c.key(): {}
                for c in cands
                if all(f"{c.key()}|{x.id}" in have for x in cells)
            }
        else:
            mine = [c for c in cands if shard_of(c.key(), n) == i]
            # duplicates are resolved over the whole stage, so every shard
            # agrees on the representative it compiles
            all_reps, dup = dedupe(cands)
            reps = [c for c in all_reps if shard_of(c.key(), n) == i]
        # illegal / not built / duplicates are recorded without compiling
        for c in mine:
            if c.key() in done:
                counts["skipped_done"] += 1
                continue
            lg = legality(c)
            if not lg.ok:
                tok = "not_built" if lg.rule == "not_built" else f"pruned:{lg.rule}"
                counts["not_built" if lg.rule == "not_built" else "illegal"] += 1
                append_jsonl(
                    path, {**base, **c.record(), "status": tok, "detail": lg.detail}
                )
            elif c.key() in dup:
                counts["duplicate"] += 1
                append_jsonl(
                    path,
                    {**base, **c.record(), "status": f"duplicate:{dup[c.key()]}"},
                )
        todo = [c for c in reps if c.key() not in done]
        if self.o.dry_run:
            for c in todo:
                append_jsonl(path, {**base, **c.record(), "status": "enumerated"})
            counts["enumerated"] = len(todo)
            return counts
        survivors = self._compile_and_prune(sl, stage, todo, path, base, counts)
        if survivors:
            self._time(sl, stage, survivors, cells, path, base, counts)
        return counts

    def _compile_and_prune(self, sl, stage, todo, path, base, counts) -> list:
        from benchmarks.common.attention_bwd_compile import compile_batch

        items = [("main", legality(c).resolved) for c in todo]
        # split candidates: their dQ kernel compiles and is gated too
        dq_at = {}
        for j, c in enumerate(todo):
            if legality(c).dq is not None:
                dq_at[j] = len(items)
                items.append(("dq", legality(c).dq))
        deadline = self.t0 + self.o.deadline_s
        # candidates that can crash the compiler compile in child processes
        iso = [j for j, (_st, s) in enumerate(items) if crash_isolated(s)]
        if self.o.compile_mode == "process" or not iso:
            iso = []
        rest = [j for j in range(len(items)) if j not in set(iso)]
        res: list = [None] * len(items)
        for idx, mode in ((rest, self.o.compile_mode), (iso, "process")):
            if not idx:
                continue
            got = compile_batch(
                [items[j] for j in idx],
                arch=sl.arch,
                cache=self.cache(),
                workers=self.o.cpus,
                mode=mode,
                deadline=deadline,
            )
            for j, r in zip(idx, got):
                res[j] = r
        survivors = []
        gate = self.gate()
        for j, (c, r) in enumerate(zip(todo, res)):
            lg = legality(c)
            rd = res[dq_at[j]] if j in dq_at else None
            rec = {
                **base,
                **c.record(),
                "resolved": repr(lg.resolved),
                "estimate": {"arch_vgpr": lg.arch_vgpr, "agpr": lg.agpr, "lds": lg.lds},
            }
            if r.error == "deadline" or (rd is not None and rd.error == "deadline"):
                counts["deadline"] += 1
                continue  # not recorded: a later shard run resumes it
            if not r.ok:
                rec.update(status=r.error, detail=r.detail)
                counts["not_built" if r.error == "not_built" else "compile_fail"] += 1
                append_jsonl(path, rec)
                continue
            rec.update(
                kernel=r.entry.kernel_name,
                hsaco_sha=r.entry.hsaco_sha,
                cache_key=r.entry.key,
                cached=r.cached,
            )
            if rd is not None:
                if not rd.ok:
                    rec.update(status=f"dq_{rd.error}", detail=rd.detail)
                    counts[
                        "not_built" if rd.error == "not_built" else "compile_fail"
                    ] += 1
                    append_jsonl(path, rec)
                    continue
                rec.update(
                    dq_kernel=rd.entry.kernel_name, dq_hsaco_sha=rd.entry.hsaco_sha
                )
            rule, res_rec = None, None
            checks = [("static_prune", "", lg.resolved, r)]
            if rd is not None:
                checks.append(("static_prune_dq", "dq_", lg.dq, rd))
            failed = False
            for fn, prefix, spec, got in checks:
                if gate is None or not hasattr(gate, fn):
                    continue
                try:
                    rule, one = getattr(gate, fn)(sl.arch, spec, got.entry.hsaco)
                except Exception as ex:  # noqa: BLE001 - never time unchecked code
                    rec.update(
                        status=f"{prefix}resource_gate_error:{type(ex).__name__}",
                        detail=str(ex)[:300],
                    )
                    counts["compile_fail"] += 1
                    append_jsonl(path, rec)
                    failed = True
                    break
                rec[f"{prefix}resources"] = one
                if rule:
                    rec["status"] = f"pruned:{prefix}{rule}"
                    counts["pruned"] += 1
                    append_jsonl(path, rec)
                    failed = True
                    break
            if failed:
                continue
            if "resources" not in rec:
                rec["resources"] = res_rec
            survivors.append((c, rec))
        return survivors

    def _time(self, sl, stage, survivors, cells, path, base, counts) -> None:
        if not cells:
            for c, rec in survivors:
                rec["status"] = "no_cells"
                append_jsonl(path, rec)
            return
        try:
            self._time_groups(sl, stage, survivors, cells, path, counts)
        finally:
            # the reference cache is per (cell, job): tensors, not records
            import shutil

            shutil.rmtree(self._ref_dir(sl, stage), ignore_errors=True)

    def _ref_dir(self, sl, stage) -> Path:
        i, n = self.o.shard
        return stage_dir(self.o.out, sl, stage) / f"timing-{i}of{n}" / "ref"

    def _time_groups(self, sl, stage, survivors, cells, path, counts) -> None:
        from benchmarks.common import attention_bwd_bench as bb
        from benchmarks.common.attention_bwd_rocke_arm import write_configs

        ids = ",".join(c.id for c in cells)
        ref_dir = self._ref_dir(sl, stage)
        tdir = ref_dir.parent
        per = max(1, self.o.arms_per_attempt)
        for gi in range(0, len(survivors), per):
            group = survivors[gi : gi + per]
            left = self.remaining()
            if left <= 60:
                counts["deadline"] += len(group)
                continue
            gdir = tdir / f"group-{gi // per}"
            gdir.mkdir(parents=True, exist_ok=True)
            cfg_path = gdir / "configs.json"
            write_configs(cfg_path, [c.config() for c, _ in group])
            argv = [
                "--ids",
                ids,
                "--arch",
                sl.arch,
                "--out-dir",
                str(gdir),
                "--rocke-configs",
                str(cfg_path),
                "--competitor-arms",
                "default",
                "--compile-cache",
                str(self.cache().root),
                "--attempts",
                str(self.o.attempts),
                "--deadline-s",
                str(max(0.0, left - 60)),
                "--seed",
                str(self.o.seed),
                "--ref-dir",
                str(ref_dir),
            ]
            for flag, val in (
                ("--gate", self.o.gate),
                ("--gate-constants", self.o.gate_constants),
                ("--expect-flavor", self.o.expect_flavor),
                ("--expect-comgr", self.o.expect_comgr),
            ):
                if val:
                    argv += [flag, val]
            bb.main(argv)
            summaries = bb.latest_summaries(gdir / "cells.jsonl")
            env = _last_env(gdir / "env.jsonl")
            for c, rec in group:
                got = [summaries[x.id] for x in cells if x.id in summaries]
                if stage == "confirm":
                    # one record per (finalist, cohort cell): the cell verdict
                    for s_ in got:
                        sc = score_from_summaries(c.key(), [s_])
                        cell_rec = {**rec, **sc, "env": env, "cell_ids": [s_["id"]]}
                        cell_rec["key"] = f"{c.key()}|{s_['id']}"
                        cell_rec["status"] = (
                            "timed" if sc["score"] is not None else sc["worst"]
                        )
                        cell_rec["verdict"] = sc["cells"][s_["id"]]["verdict"]
                        append_jsonl(path, cell_rec)
                    if len(got) < len(cells):
                        counts["deadline"] += 1
                    else:
                        counts["timed"] += 1
                    continue
                if len(got) < len(cells):
                    counts["deadline"] += 1
                    continue  # resumed by a later run of this shard
                sc = score_from_summaries(c.key(), got)
                rec.update(sc, env=env, cell_ids=[x.id for x in cells])
                if sc["score"] is not None:
                    rec["status"] = "timed"
                    counts["timed"] += 1
                else:
                    rec["status"] = sc["worst"] or "untimed"
                    key = (
                        "correctness_fail"
                        if rec["status"] == "correctness_fail"
                        else "unsupported"
                    )
                    counts[key] += 1
                append_jsonl(path, rec)


def _last_env(path: Path) -> dict | None:
    if not path.exists():
        return None
    env = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("end"):
            env = r
    return env


# --------------------------------------------------------------------------- cli
def _count_line(sc: SliceCount) -> str:
    st = " ".join(f"{k}={v}" for k, v in sc.stages.items())
    return (
        f"count arch={sc.slice.arch} family={sc.slice.family} d={sc.slice.head_size} "
        f"released={sc.release.label()} level={sc.release.level} "
        f"geometries={sc.geometries} "
        f"geometry_fallback={sc.fallback} structures_max={sc.structures_max} "
        f"product={sc.product} product_scope=legacy not_built={sc.not_built} "
        f"compiles={total_compiles(sc)} {st} "
        f"not_yet_effective={','.join(sorted(NOT_YET_EFFECTIVE_KNOBS))}"
    )


def _print_counts(
    slices: Iterable[SliceKey],
    rel: Release,
    plan_consts: Mapping | None,
    *,
    attempts: int,
    arms_per_attempt: int,
) -> None:
    for sl in slices:
        if sl.family not in SWEEPABLE_FAMILIES:
            _stdout(
                f"count arch={sl.arch} family={sl.family} d={sl.head_size} "
                "status=no_validator"
            )
            continue
        sc = count_slice(sl, rel)
        line = _count_line(sc)
        if plan_consts is not None:
            t = plan_consts.get(sl.arch) or {}
            try:
                w = WallInputs.from_mapping(t)
            except (KeyError, TypeError, ValueError):
                _stdout(line + " jobs=unknown")
                continue
            n_cells = len(representative_cells(sl))
            rt_vals = _runtime_values(sl, representative_cells(sl), rel)
            n_rt = KEEP["alternate"] * len(rt_vals)
            # a head split makes dK / dV atomic: one more kernel per split value
            n_rt_compile = KEEP["alternate"] * len(
                {v["g_split"] for v in rt_vals if v.get("g_split", 1) > 1}
            )

            def stage_jobs(survival: float) -> dict:
                jobs = {}
                for stage in (
                    "geometry", "structure", "schedule", "codegen", "swizzle",
                    "alternate",
                ):  # fmt: skip
                    n = sc.stages[stage]
                    jobs[stage] = plan_jobs(
                        n, math.ceil(n * survival), w, n_cells=n_cells,
                        attempts=attempts, arms_per_attempt=arms_per_attempt,
                    )  # fmt: skip
                jobs["runtime"] = plan_jobs(
                    n_rt_compile, n_rt, w, n_cells=n_cells, attempts=attempts,
                    arms_per_attempt=arms_per_attempt,
                )  # fmt: skip
                jobs["confirm"] = plan_jobs(
                    0, KEEP["runtime"], w, n_cells=len(cohort_cells(sl)),
                    attempts=attempts, arms_per_attempt=arms_per_attempt,
                )  # fmt: skip
                return jobs

            upper = stage_jobs(1.0)
            line += (
                f" cells={n_cells} "
                + " ".join(f"jobs_{k}={v}" for k, v in upper.items())
                + f" jobs_upper_bound={sum(upper.values())}"
            )
            if w.survival < 1.0:
                line += f" jobs_expected={sum(stage_jobs(w.survival).values())}"
        _stdout(line)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default=None)
    ap.add_argument("--family", default=None, choices=FAMILIES)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument(
        "--slice",
        default="primary",
        choices=("primary",) + tuple(SECONDARY_SLICES),
        help="primary slice or a secondary slice (recompiles the primary top 16)",
    )
    ap.add_argument(
        "--release",
        default=None,
        help="released lever families, e.g. '1' or '1,2' or 'all' (default 1)",
    )
    ap.add_argument(
        "--level",
        default="production",
        choices=SWEEP_LEVELS,
        help="production: legacy knobs crossed, registry knobs one at a time; "
        "full: every registered knob crossed (the per-shape dispatcher sweep)",
    )
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--constants", default=None, help="private wall-budget inputs")
    ap.add_argument(
        "--stage", default=None, choices=STAGE_WORDS + ("alternate_schedule",)
    )
    ap.add_argument("--dry-run", action="store_true", help="enumerate and record only")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--cpus", type=int, default=16, help="in-job compile workers")
    ap.add_argument(
        "--compile-mode",
        default="thread",
        choices=("thread", "process", "serial"),
        help="in-job compile pool (thread: comgr runs outside the GIL; measured "
        "on both CDNA node classes to give the same code objects as serial)",
    )
    ap.add_argument("--compile-cache", default=None)
    ap.add_argument("--gate", default=os.environ.get("ROCKE_BWD_PRUNE_GATE"))
    ap.add_argument("--gate-constants", default=None)
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--arms-per-attempt", type=int, default=12)
    ap.add_argument("--deadline-s", type=float, default=1e9)
    ap.add_argument("--expect-flavor", default=None)
    ap.add_argument("--expect-comgr", default=None)
    return ap


def _guard_out(path: str) -> Path:
    out = Path(path).expanduser().resolve()
    root = Path(__file__).resolve().parents[3]
    if out == root or root in out.parents:
        raise SystemExit("results directory must be outside the source tree")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    rel = Release.parse(args.release, args.level)
    slices = [
        s
        for s in all_primary_slices()
        if (args.arch is None or s.arch == args.arch)
        and (args.family is None or s.family == args.family)
        and (args.d is None or s.head_size == args.d)
    ]
    if args.slice != "primary":
        slices = [secondary_slice(s, args.slice) for s in slices]
    if args.count:
        _print_counts(
            slices,
            rel,
            _load_plan_consts(args.constants) if args.plan else None,
            attempts=args.attempts,
            arms_per_attempt=args.arms_per_attempt,
        )
        return 0
    if not args.stage:
        make_parser().print_help()
        return 0
    if not args.out_dir:
        raise SystemExit("--out-dir required (outside the source tree)")
    out = _guard_out(args.out_dir)
    si, sn = (int(x) for x in args.shard.split("/"))
    if not (sn >= 1 and 0 <= si < sn):
        raise SystemExit("--shard must be i/n with 0 <= i < n")
    if not args.dry_run:
        if args.arch is None or args.d is None:
            raise SystemExit("a timed stage needs --arch and --d")
        if not args.gate:
            raise SystemExit("a timed stage needs the pruning-set gate (--gate)")
        from benchmarks.common.attention_bwd_bench import toolchain_problems

        bad = toolchain_problems(args)
        if bad:
            _stdout("abort: " + ",".join(bad))
            return 3
    slices = [s for s in slices if s.family in SWEEPABLE_FAMILIES]
    for sl in slices:
        if args.stage == "alternate_schedule" and not alternate_changed(out, sl):
            _stdout(f"stage={args.stage} slice={sl.tag()} skipped:unchanged")
            continue
        opts = Options(
            out=out,
            arch=sl.arch,
            release=rel,
            shard=(si, sn),
            cpus=args.cpus,
            compile_mode=args.compile_mode,
            compile_cache=Path(args.compile_cache) if args.compile_cache else None,
            gate=args.gate,
            gate_constants=args.gate_constants,
            attempts=args.attempts,
            arms_per_attempt=args.arms_per_attempt,
            deadline_s=args.deadline_s,
            expect_flavor=args.expect_flavor,
            expect_comgr=args.expect_comgr,
            dry_run=args.dry_run,
        )
        counts = Driver(opts).run_stage(sl, args.stage)
        _stdout(
            f"stage={args.stage} slice={sl.tag()} released={rel.label()} "
            + " ".join(f"{k}={v}" for k, v in counts.items())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
