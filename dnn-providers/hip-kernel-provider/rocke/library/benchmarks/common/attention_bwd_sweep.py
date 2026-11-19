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
* **Dense family.** A dense slice runs only ``runtime`` and ``confirm``: its
  candidates are the general slice's survivors on the dense route, crossed
  with the dense-only knobs; ``confirm`` judges the best one against the
  general winner per cell, and ``--emit-dense-table`` writes the dense keys
  it wins (see the "dense family" section).
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
# Families whose spec validator and builder exist in this tree. The dense
# family does not run the body funnel: its candidates take the body knobs of
# the general funnel's survivors (see "dense family" below).
SWEEPABLE_FAMILIES = ("general", "dense")
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
        """The unresolved spec (the dK / dV mode follows the runtime head split).

        Dense candidates: the shared body's spec (the dense-only knobs left
        out, no head packing, the grid remap a dense field)."""
        dense = self.slice.family == "dense"
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
        if self.r.get("xcd_chunk") and self.r.get("num_xcds") and not dense:
            cls["xcd_remap"] = True
        if self.r.get("use_worklist"):
            cls["worklist"] = True
        # the dQ kernel's knobs (dq_mode = "split") are not main spec fields
        main = {k: v for k, v in self.k.items() if k not in DQ_KNOBS}
        if dense:
            main = {k: v for k, v in main.items() if k not in DENSE_ONLY_KNOBS}
            main["head_pack"] = False
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
        if self.slice.kind != "head_pack" or self.slice.family == "dense":
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
    if sl.family == "dense":
        bad = _dense_problem(c, s)
        if bad is not None:
            return Legality(False, bad[0], bad[1][:300], s)
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
    if c.slice.family == "dense":
        # the dense-only knobs are not body fields
        blob += "|dense:" + repr(
            sorted((k, v) for k, v in c.knobs if k in DENSE_ONLY_KNOBS)
        )
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
    if lg.resolved.dkv_mode == "atomic" and lg.resolved.dkv_sink != "slices":
        present.discard("dkv_out")  # (the slice sink stages through it)
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
            # the plan pairs only a grid that overfills the device: the arm
            # reads the CU count of the device it runs on (a request hint;
            # the arm also passes it by default, this pins it)
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


def _sink_values(sl: SliceKey, rel: Release) -> list:
    """Stage R values of the atomic-mode dK / dV sink (built, non-default),
    crossed with the points whose dK / dV are atomic."""
    if not rel.has("dkv_sink"):
        return []
    axis = backward_axis("dkv_sink")
    return [
        v
        for v in rel.values(sl, "dkv_sink", axis.default)
        if v != axis.default and v not in axis.not_built
    ]


# Codegen levers that act on the slice sink's epilogue only (its staging, store
# width and bounds); stage R moves each on the slice-sink points of a
# candidate.
SLICE_LEVERS = ("dkv_epilogue", "dkv_store_vec", "ws_bounds", "dkv_cell_vec")


def sink_points(c: Candidate, rel: Release) -> list[dict]:
    """Knob sets of the non-default dK / dV sinks on top of candidate ``c``:
    each sink alone, then each :data:`SLICE_LEVERS` value that applies there,
    one at a time (``dkv_cell_vec`` with the per-slot epilogue it needs)."""
    sl = c.slice
    out = []
    for s in _sink_values(sl, rel):
        base = {**sl.spec_class(), **c.k, "dkv_mode": "atomic", "dkv_sink": s}
        out.append({"dkv_sink": s})
        for name in SLICE_LEVERS:
            if not rel.has(name):
                continue
            axis = backward_axis(name)
            cur = base.get(name, axis.default)
            for v in rel.values(sl, name, axis.default):
                if v == cur:
                    continue
                extra = {name: v}
                if name == "dkv_cell_vec" and base.get("dkv_epilogue") != "direct":
                    extra["dkv_epilogue"] = "direct"
                if axis.applies(v, {**base, **extra}):
                    out.append({"dkv_sink": s, **extra})
    return out


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
    if sl.family == "dense":
        return dense_stage_candidates(stage, sl, rel=rel, prior=prior, cells=cells)
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
        # the atomic-mode dK / dV sink (a codegen knob) and the levers that
        # act on the slice sink only, also on every point a head split makes
        # atomic: on a direct slice the codegen stage cannot enumerate them
        # (inert there)
        out += [
            c.with_knobs(**p).with_runtime(**v)
            for c in top
            if c.k.get("dkv_sink", "atomic") == "atomic"
            for p in sink_points(c, rel)
            for v in vals
            if sl.dkv_mode == "atomic" or int(v.get("g_split", 1)) > 1
        ]
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


# --------------------------------------------------------------------------- dense family
# The dense route (``runtime["route"] = "dense"``) runs the shared body with
# compile-time shapes. Its body knobs are not swept again: the dense slice of
# (arch, d) seeds its candidates from the general slice's survivors (the top
# of the general runtime stage, with their runtime values), crosses them with
# the dense-only knobs in the ``runtime`` stage (they do not change the body),
# and ``confirm`` times its best candidate against the general winner on the
# dense cohort (dense-eligible census cells of the head size, short-query GQA
# cells excluded: the general plan packs their heads, which the dense route
# does not build). A dense key enters ``DENSE_BWD_TUNED`` only from
# :func:`dense_table_entries` (a paired win over the general winner on every
# cell of the key, a clean A/A control, the pruning-set gate passed).
DENSE_STAGES = ("runtime", "confirm")
# Dense spec fields that are not fields of the general family's spec.
DENSE_ONLY_KNOBS = ("oob", "epilogue", "persistent", "persist_decode")
# Dense-only knobs the runtime stage enumerates (the registry's dense axes
# that are compile-time, built and not plan-owned; edge_tiles, ws_layout and
# dq_mode are dense spec fields there). dq_mode moves a seed between atomic
# dQ and split dQ (the dense dQ kernel): a split point drops to atomic without
# its dQ-kernel values, an atomic point goes split with the dQ defaults, and a
# split point also tries the dQ kernel's edge / interior key loops (with
# compile-time lengths the dense dQ kernel drops the runs proved empty).
DENSE_RUNTIME_AXES = (
    "oob",
    "edge_tiles",
    "epilogue",
    "ws_layout",
    "persistent",
    "persist_decode",
    "dq_mode",
)
DENSE_SEEDS = KEEP["alternate"]  # general survivors seeding the dense stage
# The dense grid remap is the plan's (xcd_chunk with the request's XCDs).
_DENSE_ROUTE = "dense"
_PROBE_LEN = 4096
_CPU_NUM_CUS = 64


@cache
def _dense_axes() -> dict:
    from dispatch.attention_bwd.dense_tuning import DENSE_BWD_KNOB_AXES

    return {a.name: a for a in DENSE_BWD_KNOB_AXES if a.name in DENSE_RUNTIME_AXES}


def _dense_problem(c: "Candidate", body) -> tuple[str, str] | None:
    """``(rule, detail)`` when a dense candidate cannot be built for any
    shape of its slice (beyond the body's own legality), else ``None``."""
    from kernels.common.attention_dense_bwd_plan import dense_bwd_spec_class
    from kernels.common.attention_dense_bwd_spec import _shell_not_built

    missing = _shell_not_built(body)
    if missing:
        return "not_built", "; ".join(missing)
    if c.r.get("use_worklist"):
        return "coupling", "the work list is a general-family role"
    if c.r.get("route") != _DENSE_ROUTE:
        return "coupling", "a dense candidate plans the dense route"
    sl = c.slice
    fields = {
        k: c.k[k]
        for k in DENSE_RUNTIME_AXES
        if k in c.k and k not in ("edge_tiles", "ws_layout")
    }
    try:
        dense_bwd_spec_class(sl.arch)(
            seqlen_q=_PROBE_LEN, seqlen_kv=_PROBE_LEN, num_query_heads=8,
            num_kv_heads=2, head_size=sl.head_size, dtype=sl.dtype,
            causal=sl.mask_class == "band", ragged=True,
            edge_tiles=c.k.get("edge_tiles", True),
            ws_layout=c.k.get("ws_layout", "head_major"), **fields,
        )  # fmt: skip
    except ValueError as ex:
        return rule_token(str(ex)), str(ex)
    return None


def dense_points(c: "Candidate", rel: Release) -> list[dict]:
    """Dense-only knob points around candidate ``c`` (its current values
    first): one knob moved at a time at ``production``, crossed at ``full``
    (``persist_decode`` only with ``persistent``). Values the registry marks
    not built are never enumerated."""
    axes = _dense_axes()
    cur = {n: c.k.get(n, ax.default) for n, ax in axes.items()}

    def vals(n):
        ax = axes[n]
        if ax.family not in rel.families:
            return [cur[n]]
        return [v for v in ax.values(c.slice.arch, rel.level) if v not in ax.not_built]

    def ok(p):
        # the couplings read the seed's body knobs too (dq_mode: the split
        # main kernel's)
        k = {**c.k, **p}
        return all(axes[n].applies(p[n], k) for n in axes)

    if rel.level == "full":
        names = list(axes)
        out = []
        for combo in itertools.product(*(vals(n) for n in names)):
            p = dict(zip(names, combo))
            if ok(p):
                out.append(p)
        return out
    out = [dict(cur)]
    for n in axes:
        for v in vals(n):
            if v == cur[n]:
                continue
            p = {**cur, n: v}
            if n == "persistent" and v:
                p["persist_decode"] = axes["persist_decode"].default
            if ok(p) and p not in out:
                out.append(p)
    if cur.get("persistent") is not True:
        # persist_decode moves together with persistent = True
        for v in vals("persist_decode"):
            p = {**cur, "persistent": True, "persist_decode": v}
            if v != axes["persist_decode"].default and ok(p) and p not in out:
                out.append(p)
    return out


def _dense_dq_points(c: "Candidate", rel: Release) -> list["Candidate"]:
    """The candidates of :func:`dense_points` around ``c`` (its own point
    first), with the dQ-kernel values following ``dq_mode``: an atomic point
    carries none (a split seed's are dropped), a split point keeps the seed's
    (or the defaults), and the split point that differs from the seed in
    ``dq_mode`` alone (or not at all) also tries ``dq_edge_tiles = True``."""
    pts = dense_points(c, rel)
    seed = pts[0] if pts else {}
    out = []
    for p in pts:
        x = c.with_knobs(**p)
        if x.k.get("dq_mode", "atomic") != "split":
            if any(k in DQ_KNOBS for k in x.k):
                keep = {k: v for k, v in x.k.items() if k not in DQ_KNOBS}
                x = Candidate.make(x.slice, keep, x.r)
            out.append(x)
            continue
        out.append(x)
        alone = all(p[n] == seed.get(n) for n in p if n != "dq_mode")
        if alone and not x.k.get("dq_edge_tiles", False):
            out.append(x.with_knobs(dq_edge_tiles=True))
    return list(dict.fromkeys(out))


def _dense_xcd_values(sl: SliceKey, rel: Release) -> list[dict]:
    """Non-zero XCD chunks of the registry (built values), each with the
    request's XCD count as a device hint (the dense ``xcd_remap`` axis)."""
    if not rel.has("xcd_chunk") or sl.arch not in SWEEP_NUM_XCDS:
        return []
    hint = {"num_xcds": SWEEP_NUM_XCDS[sl.arch]}
    return [{"xcd_chunk": v, **hint} for v in _runtime_built(sl, rel, "xcd_chunk") if v]


def general_slice_of(sl: SliceKey) -> SliceKey:
    """The general slice whose survivors seed dense slice ``sl``."""
    return replace(sl, family="general")


def _dense_seed(sl: SliceKey, g: "Candidate") -> "Candidate | None":
    """General survivor ``g`` as a dense candidate of ``sl`` (``None`` when
    its runtime values have no dense counterpart: mirror pairing, the work
    list)."""
    r = dict(g.r)
    if r.get("lb_order") == "mirror" or r.get("use_worklist"):
        return None
    r.pop("num_cus", None)  # only mirror pairing reads the CU hint
    r["route"] = _DENSE_ROUTE
    knobs = {k: v for k, v in g.k.items() if k != "head_pack"}
    return Candidate.make(sl, knobs, r)


def dense_stage_candidates(
    stage: str,
    sl: SliceKey,
    *,
    rel: Release,
    prior: Mapping[str, Sequence["Candidate"]],
    cells: Sequence = (),
) -> list["Candidate"]:
    """Candidates of a dense slice's stage (see the section comment).

    ``prior`` holds the dense slice's own ranked stages and, under
    ``general:<stage>``, the general slice's. ``runtime``: the top
    :data:`DENSE_SEEDS` general runtime-stage survivors (or, before that
    stage ran, the general pre-runtime top crossed with the runtime values),
    each crossed with :func:`dense_points`. ``confirm``: the general winner
    first (the A/A twin), then the best dense candidate.
    """
    if stage == "runtime":
        seeds = list(prior.get("general:runtime", ())[:DENSE_SEEDS])
        if not seeds:
            top = (
                prior.get("general:alternate_schedule")
                or prior.get("general:alternate")
                or prior.get("general:swizzle")
                or prior.get("general:codegen")
                or prior.get("general:schedule")
                or ()
            )[:DENSE_SEEDS]
            vals = _runtime_values(general_slice_of(sl), cells, rel)
            seeds = [g.with_runtime(**v) for g in top for v in vals]
        xcd = _dense_xcd_values(sl, rel)
        out = []
        for g in seeds:
            d = _dense_seed(sl, g)
            if d is None:
                continue
            for i, x in enumerate(_dense_dq_points(d, rel)):
                p = {n: x.k.get(n, ax.default) for n, ax in _dense_axes().items()}
                if p.get("persistent"):
                    # a persistent grid is sized from the device's CU count
                    out.append(x.with_runtime(num_cus=DEVICE_NUM_CUS))
                    continue
                out.append(x)
                # the dense grid remap (xcd_remap, derived by the plan from
                # xcd_chunk and the request's XCD count) on the non-persistent
                # grid: around the seed's point (production) or every point
                if not d.r.get("xcd_chunk") and (i == 0 or rel.level == "full"):
                    out += [x.with_runtime(**v) for v in xcd]
        return list(dict.fromkeys(out))
    if stage == "confirm":
        gen = list(prior.get("general:runtime", ())[:1])
        top = list(prior.get("runtime", ())[:1])
        return gen + top if gen and top else []
    return []


class _StandIn:
    """Shape / stride / pointer stand-in for a device tensor (CPU planning
    of a census cell; logical ``[B, H, S, D]`` with the cell's layout)."""

    def __init__(self, shape, strides) -> None:
        self.shape, self._st = shape, strides
        self.dtype = "torch.bfloat16"

    def stride(self, i=None):
        return self._st if i is None else self._st[i]

    def data_ptr(self) -> int:
        return 1 << 20


def cell_request(cell):
    """The request of a dense-layout census cell, planned without a device
    (contiguous tensors in the cell's layout, 256-byte aligned)."""
    from benchmarks.common.attention_bwd_rocke_arm import request_for_cell, scale_of

    if cell.is_thd:
        raise ValueError("a THD cell has no dense-layout request")
    t = {}
    for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
        q_side = n in ("q", "o", "do", "dq")
        h = cell.hq if q_side else (cell.hkv if n in ("k", "dk") else cell.h_v)
        s, d = (cell.sq if q_side else cell.skv), cell.d
        st = (
            (h * s * d, s * d, d, 1)
            if cell.layout == "bhsd"
            else (s * h * d, d, h * d, 1)
        )
        t[n] = _StandIn((cell.b, h, s, d), st)
    return request_for_cell(cell, t, scale=scale_of(cell))


def head_packed(cell, block_m: int) -> bool:
    """Whether the general plan packs the query heads of ``cell`` (a GQA
    group whose ``G * s_q`` rows fit one q tile): such cells stay general."""
    group = cell.hq // cell.hkv if cell.hkv else 1
    return group > 1 and cell.sq * group <= block_m


def dense_cells(sl: SliceKey, cells: Sequence, block_m: int) -> list:
    """The cells of ``cells`` the dense route may take for slice ``sl``:
    dense-eligible on its arch and not head-packed by the general plan."""
    from kernels.common.attention_bwd_plan import dense_bwd_eligibility

    out = []
    for c in cells:
        if c.is_thd or head_packed(c, block_m):
            continue
        try:
            req = cell_request(c)
        except ValueError:
            continue
        if dense_bwd_eligibility(req, sl.arch)[0]:
            out.append(c)
    return out


def _max_block_m(cands: Sequence["Candidate"]) -> int:
    """The largest resolved ``block_m`` of the legal candidates (0: none)."""
    out = 0
    for c in cands:
        r = legality(c).resolved
        if r is not None:
            out = max(out, int(r.block_m))
    return out


def dense_plan(c: "Candidate", cell):
    """The dense plan of candidate ``c`` on ``cell`` (CPU; ``ValueError``
    when the dense route refuses it)."""
    from benchmarks.common.attention_bwd_rocke_arm import REQUEST_HINTS, config_policy
    from kernels.common.attention_dense_bwd_plan import attn_dense_bwd_plan

    cfg = c.config()
    req = cell_request(cell)
    hints = {k: cfg.runtime[k] for k in REQUEST_HINTS if k in cfg.runtime}
    if hints.get("num_cus") == DEVICE_NUM_CUS:
        # the device's CU count is a kernarg (num_persistent): any positive
        # count plans the same kernel
        hints["num_cus"] = _CPU_NUM_CUS
    if hints:
        req = replace(req, **hints)
    runtime = {k: v for k, v in cfg.runtime.items() if k != "route"}
    pol = config_policy(type(cfg)(cfg.label, cfg.knobs, runtime))
    return attn_dense_bwd_plan(req, c.slice.arch, policy=pol)


def dense_vs_general(
    summary: Mapping,
    dense_key: str,
    general_key: str,
    constants: Mapping | None,
    *,
    gated: bool,
) -> dict:
    """Paired verdict of a dense candidate against the general winner on one
    confirm cell (both timed in the same attempts). ``win``: the dense arm
    passes against the general one and the general arm fails against the
    dense one (faster beyond the noise band); ``aa``: the A/A controls of both
    arms (the confirm stage times a twin of every arm, ``--aa-control all``:
    the two arms differ in kernel and runtime values, so one arm's twin does
    not bound the other's noise); ``ok`` only when both twins are clean;
    ``gated``: the timing ran with the pruning-set gate."""
    from benchmarks.common import attention_bwd_bench as bb

    arms = summary.get("arms", {})
    d = arms.get(bb.rocke_arm_name(dense_key), {})
    g = arms.get(bb.rocke_arm_name(general_key), {})
    out = {
        "general_key": general_key,
        "aa": _pair_aa(summary, dense_key, general_key),
        "gated": bool(gated),
        "win": False,
        "verdict": None,
    }
    if constants is None:
        out["verdict"] = "no_constants"
        return out
    if d.get("status") != "ok" or g.get("status") != "ok":
        out["verdict"] = "untimed"
        return out
    a, b = d["attempt_medians_ms"], g["attempt_medians_ms"]
    if min(len(a), len(b)) < bb.MIN_KEPT_ATTEMPTS:
        out["verdict"] = "incomplete"
        return out
    done = int(summary.get("reruns_done", 0))
    fwd = bb.paired_verdict(a, b, constants, reruns_done=done)
    rev = bb.paired_verdict(b, a, constants, reruns_done=done)
    out.update(
        verdict=fwd,
        reverse=rev,
        ratio=statistics.median(a) / statistics.median(b),
        win=fwd == "pass" and rev == "fail",
    )
    return out


def _pair_aa(summary: Mapping, *keys: str) -> str | None:
    """The A/A verdict of the arms ``keys`` (each needs its own twin):
    ``fail`` / ``incomplete`` / ``None`` (a twin missing) / ``ok``."""
    from benchmarks.common import attention_bwd_bench as bb

    per = summary.get("aa_controls")
    if per is None:
        return None  # no per-arm twins (an older record): not judged
    got = [per.get(bb.rocke_arm_name(k)) for k in keys]
    if "fail" in got:
        return "fail"
    if "incomplete" in got:
        return "incomplete"
    if None in got:
        return None
    return "ok"


def dense_tuned_value(c: "Candidate") -> dict:
    """The ``DENSE_BWD_TUNED`` value of dense candidate ``c``: its knobs and
    runtime values (request hints and the route switch left out)."""
    from benchmarks.common.attention_bwd_rocke_arm import REQUEST_HINTS

    out = dict(c.tuning_knobs())
    out.update(
        {k: v for k, v in c.r.items() if k != "route" and k not in REQUEST_HINTS}
    )
    return {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}


def dense_table_entries(out: Path, sl: SliceKey) -> list[dict]:
    """``DENSE_BWD_TUNED`` entries of dense slice ``sl`` from its confirm
    records: ``[{"arch", "key", "knobs", "cells"}]`` for every dense key
    whose every confirm cell (one dense candidate) has a paired win of the
    dense candidate over the general winner (:func:`dense_vs_general`), a
    clean A/A control (``aa = "ok"``), a timing run under the pruning-set
    gate (``gated``) and a timed dense arm (status ``timed``: it passed the
    gate and the timed-shape check in its timing process)."""
    from kernels.common.attention_bwd_plan import _dense_key

    recs = read_records(out, sl, "confirm")
    by_key: dict[tuple, list] = {}
    cells = {c.id: c for c in _census_cells()}
    for key, r in recs.items():
        if r.get("slice", {}).get("family") != "dense" or "|" not in key:
            continue
        cell = cells.get(key.partition("|")[2])
        if cell is None:
            continue
        dk = (sl.arch, _dense_key(cell_request(cell)))
        by_key.setdefault(dk, []).append(r)
    entries = []
    for dk, rs in sorted(by_key.items(), key=lambda kv: repr(kv[0])):
        cands = {r["key"].partition("|")[0] for r in rs}
        ok = len(cands) == 1 and all(
            r.get("status") == "timed"
            and (r.get("vs_general") or {}).get("gated") is True
            and (r.get("vs_general") or {}).get("win") is True
            and (r.get("vs_general") or {}).get("aa") == "ok"
            for r in rs
        )
        if not ok:
            continue
        c = Candidate.from_record(rs[0])
        entries.append(
            {
                "arch": dk[0],
                "key": json.loads(json.dumps(dk[1])),
                "knobs": dense_tuned_value(c),
                "cells": sorted(r["key"].partition("|")[2] for r in rs),
            }
        )
    return entries


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
        if sl.family == "dense":
            # the general slice's survivors seed the dense candidates
            gsl = general_slice_of(sl)
            for st in STAGE_WORDS + ("alternate_schedule",):
                if st in KEEP:
                    out[f"general:{st}"] = ranked(self.o.out, gsl, st, None)
            return out
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
        if sl.family == "dense" and stage not in DENSE_STAGES:
            return {"skipped": "dense_seeds_from_general"}
        cells = cohort_cells(sl) if stage == "confirm" else representative_cells(sl)
        prior = self.prior(sl)
        cands = stage_candidates(
            stage, sl, rel=self.o.release, prior=prior, cells=cells
        )
        if sl.family == "dense":
            # the cells the dense route may take (no head-packed cell under
            # any candidate's tile)
            cells = dense_cells(sl, cells, _max_block_m(cands))
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
        dense = [c for c in todo if c.slice.family == "dense"]
        todo = [c for c in todo if c.slice.family != "dense"]
        survivors = self._compile_and_prune(sl, stage, todo, path, base, counts)
        if dense:
            survivors += self._compile_and_prune_dense(dense, cells, path, base, counts)
        if survivors:
            self._time(sl, stage, survivors, cells, path, base, counts)
        return counts

    def _compile_and_prune_dense(self, todo, cells, path, base, counts) -> list:
        """Dense candidates: one dense kernel per (candidate, cell) (the
        shape is compiled in), each through the resource gate with the
        candidate's resolved body spec. A candidate no cell can plan is
        ``unsupported``; any kernel that fails to compile or prunes fails it."""
        from benchmarks.common.attention_bwd_compile import compile_batch

        items, owner, refused = [], [], {}
        for j, c in enumerate(todo):
            refused[j] = []
            for cell in cells:
                try:
                    p = dense_plan(c, cell)
                except ValueError as ex:
                    refused[j].append(f"{cell.id}: {ex}"[:200])
                    continue
                items.append(("dense", p.specs["dense"]))
                owner.append(j)
                if "dense_dq" in p.specs:
                    # split dQ: the cell's dense dQ kernel compiles and is
                    # gated too
                    items.append(("dense_dq", p.specs["dense_dq"]))
                    owner.append(j)
        deadline = self.t0 + self.o.deadline_s
        iso = {
            i for i, j in enumerate(owner) if crash_isolated(legality(todo[j]).resolved)
        }
        if self.o.compile_mode == "process":
            iso = set()
        res: list = [None] * len(items)
        for idx, mode in (
            ([i for i in range(len(items)) if i not in iso], self.o.compile_mode),
            (sorted(iso), "process"),
        ):
            if not idx:
                continue
            got = compile_batch(
                [items[i] for i in idx], arch=todo[0].slice.arch, cache=self.cache(),
                workers=self.o.cpus, mode=mode, deadline=deadline,
            )  # fmt: skip
            for i, r in zip(idx, got):
                res[i] = r
        gate = self.gate()
        survivors = []
        for j, c in enumerate(todo):
            lg = legality(c)
            mine = [res[i] for i, o in enumerate(owner) if o == j]
            kinds = [items[i] for i, o in enumerate(owner) if o == j]
            stages = [st for st, _ in kinds]
            rec = {
                **base,
                **c.record(),
                "resolved": repr(lg.resolved),
                "estimate": {"arch_vgpr": lg.arch_vgpr, "agpr": lg.agpr, "lds": lg.lds},
                "refused_cells": refused[j],
            }
            if any(r.error == "deadline" for r in mine):
                counts["deadline"] += 1
                continue  # resumed by a later run of this shard
            if not mine:
                rec.update(status="unsupported", detail="; ".join(refused[j])[:300])
                counts["unsupported"] += 1
                append_jsonl(path, rec)
                continue
            bad = next((i for i, r in enumerate(mine) if not r.ok), None)
            if bad is not None:
                prefix = "dq_" if stages[bad] == "dense_dq" else ""
                rec.update(status=prefix + mine[bad].error, detail=mine[bad].detail)
                bad = mine[bad]
                counts["not_built" if bad.error == "not_built" else "compile_fail"] += 1
                append_jsonl(path, rec)
                continue
            rec.update(
                kernel=[r.entry.kernel_name for r in mine],
                hsaco_sha=[r.entry.hsaco_sha for r in mine],
                cached=all(r.cached for r in mine),
            )
            failed = False
            if gate is not None and hasattr(gate, "static_prune"):
                rec["resources"] = []
                for (st, spec), r in zip(kinds, mine):
                    # a dense dQ kernel: the dQ rules on its resolved dQ spec
                    dq = st == "dense_dq"
                    prefix = "dq_" if dq else ""
                    if dq and not hasattr(gate, "static_prune_dq"):
                        continue
                    try:
                        if dq:
                            rule, one = gate.static_prune_dq(
                                c.slice.arch, spec.dq, r.entry.hsaco
                            )
                        else:
                            rule, one = gate.static_prune(
                                c.slice.arch, lg.resolved, r.entry.hsaco
                            )
                    except Exception as ex:  # noqa: BLE001 - never time unchecked code
                        rec.update(
                            status=f"{prefix}resource_gate_error:{type(ex).__name__}",
                            detail=str(ex)[:300],
                        )
                        counts["compile_fail"] += 1
                        failed = True
                        break
                    rec["resources"].append(one)
                    if rule:
                        rec["status"] = f"pruned:{prefix}{rule}"
                        counts["pruned"] += 1
                        failed = True
                        break
            if failed:
                append_jsonl(path, rec)
                continue
            survivors.append((c, rec))
        return survivors

    def _compile_and_prune(self, sl, stage, todo, path, base, counts) -> list:
        from benchmarks.common.attention_bwd_compile import compile_batch

        if not todo:
            return []
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
        consts = None
        if sl.family == "dense" and stage == "confirm":
            # the dense candidate is judged against the general winner timed
            # in the same attempts (one group)
            per = max(per, len(survivors))
            if self.o.gate_constants:
                consts = bb.load_gate_constants(self.o.gate_constants)
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
            if stage == "confirm":
                # every finalist with its own A/A twin: the arms differ in
                # kernel and runtime values (dense_vs_general, the table rule)
                argv += ["--aa-control", "all"]
            bb.main(argv)
            summaries = bb.latest_summaries(gdir / "cells.jsonl")
            env = _last_env(gdir / "env.jsonl")
            for c, rec in group:
                got = [summaries[x.id] for x in cells if x.id in summaries]
                if stage == "confirm":
                    # one record per (finalist, cohort cell): the cell verdict
                    general = next(
                        (g for g, _ in group if g.slice.family == "general"), None
                    )
                    for s_ in got:
                        sc = score_from_summaries(c.key(), [s_])
                        cell_rec = {**rec, **sc, "env": env, "cell_ids": [s_["id"]]}
                        cell_rec["key"] = f"{c.key()}|{s_['id']}"
                        cell_rec["status"] = (
                            "timed" if sc["score"] is not None else sc["worst"]
                        )
                        cell_rec["verdict"] = sc["cells"][s_["id"]]["verdict"]
                        if c.slice.family == "dense" and general is not None:
                            cell_rec["vs_general"] = dense_vs_general(
                                s_, c.key(), general.key(), consts,
                                gated=bool(self.o.gate),
                            )  # fmt: skip
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
        if sl.family == "dense":
            # seeded from the general survivors: the dense-only points per seed
            seed = _dense_seed(
                sl, Candidate.make(general_slice_of(sl), default_knobs(sl))
            )
            _stdout(
                f"count arch={sl.arch} family=dense d={sl.head_size} "
                f"status=seeded_from_general seeds={DENSE_SEEDS} "
                f"dense_points={len(dense_points(seed, rel))} "
                f"stages={','.join(DENSE_STAGES)}"
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
            # a head split makes dK / dV atomic: one more kernel per split
            # value; each slice-sink point is one more kernel too (g_split is
            # a kernarg, so one per point whatever the split), timed on every
            # atomic runtime point
            atomic_pts = [
                v for v in rt_vals if sl.dkv_mode == "atomic" or v.get("g_split", 1) > 1
            ]
            rep = Candidate.make(sl, default_knobs(sl))
            n_sink = len(sink_points(rep, rel)) if atomic_pts else 0
            n_rt = KEEP["alternate"] * (len(rt_vals) + n_sink * len(atomic_pts))
            splits = {v["g_split"] for v in rt_vals if v.get("g_split", 1) > 1}
            n_rt_compile = KEEP["alternate"] * (len(splits) + n_sink)

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
    ap.add_argument(
        "--gate",
        default=None,
        help="module:callable of the pruning-set gate provider (explicit only: "
        "no environment default; a timed stage needs it, and every bench "
        "command the sweep runs carries it)",
    )
    ap.add_argument("--gate-constants", default=None)
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--arms-per-attempt", type=int, default=12)
    ap.add_argument("--deadline-s", type=float, default=1e9)
    ap.add_argument("--expect-flavor", default=None)
    ap.add_argument("--expect-comgr", default=None)
    ap.add_argument(
        "--emit-dense-table",
        default=None,
        metavar="PATH",
        help="write the DENSE_BWD_TUNED entries the dense confirm records earn "
        "(paired win over the general winner, clean A/A, gated) to PATH (JSON, "
        "outside the source tree); needs --out-dir",
    )
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
    if args.emit_dense_table:
        if not args.out_dir:
            raise SystemExit("--emit-dense-table needs --out-dir")
        out = _guard_out(args.out_dir)
        dest = _guard_out(args.emit_dense_table)
        entries = []
        for sl in slices:
            if sl.family == "dense":
                entries += dense_table_entries(out, sl)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps({"entries": entries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _stdout(f"dense_table entries={len(entries)}")
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
