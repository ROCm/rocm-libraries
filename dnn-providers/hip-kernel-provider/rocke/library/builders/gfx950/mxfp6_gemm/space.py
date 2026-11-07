# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 MXFP6 GEMM's knob space as a rocKE search space.

The space is read from the kernel module every time it is built, so knobs the
kernel adds are searched without a change here:

* ``KNOBS`` (``{field: legal values}``, first value = the spec default) is the
  authoritative table when the kernel module has one; spec fields it does not
  list are held at the base value (reported by :meth:`Mxfp6Space.describe`);
* without a table every ``Mxfp6GemmSpec`` field is derived from its type: a
  bool takes both values, a string the module constant tuple that holds its
  default (``FRAG_MODES`` for ``frag_mode``), an int a small neighbourhood
  of its default (``auto_unlisted`` applies the same rules to fields a table
  leaves out);
* the problem shape, measurement probes and test mutants (:data:`NOT_KNOBS`)
  are never searched.

Legality is a cheap pre-filter that never spends a compile: the kernel's own
``validate_spec`` (divisibility, ring depth, AGPR count, slot ranges), the LDS
budget (:data:`LDS_LIMIT`) and a register-budget estimate
(:func:`reg_estimate`; it rejects only configs that are over budget by more
than :data:`REG_SLACK`, the compile gate is the authority). The search space
is a :class:`~rocke.dispatch.tuning.search.params.SearchSpace` built from
:class:`~rocke.dispatch.tuning.axes.KnobAxis` choices, whose ``()`` level is
the *base* (the warm start), so a decoded genome is the overrides on the base.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rocke.dispatch.tuning.axes import KnobAxis, choices
from rocke.dispatch.tuning.search.params import Genome, SearchSpace, params_from_axes

__all__ = [
    "LDS_LIMIT",
    "NOT_KNOBS",
    "REG_SLACK",
    "Mxfp6Space",
    "default_config",
    "kernel_fingerprint",
    "kernel_module",
    "knob_table",
    "legal",
    "overrides_of",
    "reg_estimate",
]

KERNEL_MODULE = "kernels.gfx950.mxfp6_gemm.gemm"
#: Spec fields that are not performance knobs.
NOT_KNOBS = ("m", "n", "k", "probe", "mutant")
LDS_LIMIT = 160 * 1024
#: One wave per SIMD owns 512 VGPRs per lane (256 arch + 256 AGPR).
VGPR_FILE = 512
ARCH_VGPRS = 256
#: The register estimate is approximate: reject only configs over budget by more than this factor.
REG_SLACK = 1.25
#: Fixed per-wave VGPR overhead of the estimate (addresses, loop state) and scale registers.
REG_MISC = 24
REG_SCALES = 32


def kernel_module(name: str = KERNEL_MODULE) -> ModuleType:
    """The kernel module (imported fresh by the caller's process: a job imports the staged tree)."""
    return importlib.import_module(name)


def default_config(mod: ModuleType) -> Dict[str, object]:
    return {f.name: f.default for f in fields(mod.Mxfp6GemmSpec) if f.default is not MISSING}


def _type_of(mod: ModuleType, name: str, default: object) -> type:
    return type(default)


def _constant_levels(mod: ModuleType, name: str, default: object) -> Optional[Tuple]:
    """A module constant tuple of strings holding ``default`` whose name starts with the field's (``FRAG_MODES``)."""
    stem = name.upper()
    for attr in sorted(vars(mod)):
        val = getattr(mod, attr)
        if attr.startswith(stem) and isinstance(val, tuple) and val and all(isinstance(v, str) for v in val):
            if default in val:
                return (default,) + tuple(v for v in val if v != default)
    return None


def _auto_levels(mod: ModuleType, name: str, default: object) -> Optional[Tuple]:
    if isinstance(default, bool):
        return (default, not default)
    if isinstance(default, str):
        return _constant_levels(mod, name, default)
    if isinstance(default, int):
        cand = [default, default - 1, default + 1, default * 2, default // 2]
        lo = -1 if default < 0 else 0
        out: List[int] = []
        for v in cand:
            if v >= lo and v not in out:
                out.append(v)
        return tuple(out) if len(out) > 1 else None
    return None


def knob_table(mod: ModuleType, *, auto_unlisted: Optional[bool] = None) -> Tuple[Dict[str, Tuple], Dict[str, str]]:
    """``(knobs, fixed)``: ``{field: levels (default first)}`` and ``{field: why it is held}``.

    ``auto_unlisted`` (default: True only when the module has no ``KNOBS``)
    derives levels for fields the table does not list.
    """
    defaults = default_config(mod)
    table = getattr(mod, "KNOBS", None)
    if auto_unlisted is None:
        auto_unlisted = table is None
    knobs: Dict[str, Tuple] = {}
    fixed: Dict[str, str] = {}
    for name in defaults:
        if name in NOT_KNOBS:
            fixed[name] = "not a knob (shape / probe / mutant)"
            continue
        if table is not None and name in table:
            levels = tuple(dict.fromkeys(table[name]))
            if len(levels) > 1:
                knobs[name] = levels
            else:
                fixed[name] = "KNOBS lists one value"
            continue
        if auto_unlisted:
            levels = _auto_levels(mod, name, defaults[name])
            if levels:
                knobs[name] = levels
                continue
            fixed[name] = "no derivable levels"
        else:
            fixed[name] = "not in KNOBS (held at the base)"
    if table is not None:
        for name in table:
            if name not in defaults:
                fixed[name] = "stale KNOBS entry (no such spec field): ignored"
    return knobs, fixed


def kernel_fingerprint(mod: ModuleType) -> str:
    """sha256 over the kernel package's Python sources: a search result holds only for this body."""
    h = hashlib.sha256()
    pkg = Path(mod.__file__).resolve().parent
    for p in sorted(pkg.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


# ----------------------------------------------------------------- legality


def reg_estimate(mod: ModuleType, spec) -> Dict[str, object]:
    """Per-lane register estimate of one wave (approximate; the compile gate decides).

    Fragment VGPRs: one 128-K tile of A and B fragments (6 VGPRs per 32-value
    fragment), twice for a prefetching schedule (the next tile's set is live
    while this tile's MFMAs run); scales and a fixed overhead on top;
    accumulators in AGPRs (or VGPRs). Waves beyond one per SIMD split the
    512-entry file.
    """
    geo = mod.atom_geometry(spec.atom)
    fm = spec.tile_m // spec.waves_m // geo.rows
    fn = spec.tile_n // spec.waves_n // geo.rows
    ksteps = 128 // geo.k
    sets = 2 if getattr(spec, "frag_mode", "prefetch") == "prefetch" else 1
    frag = (fm + fn) * ksteps * 6 * sets
    acc = mod.acc_regs(spec) if hasattr(mod, "acc_regs") else fm * fn * geo.rows * geo.rows // 64
    agpr = bool(getattr(spec, "acc_agpr", True))
    arch = frag + REG_SCALES + REG_MISC + (0 if agpr else acc)
    nw = spec.waves_m * spec.waves_n
    per_simd = max(1, math.ceil(nw / 4))
    budget = VGPR_FILE // per_simd
    total = arch + (acc if agpr else 0)
    arch_limit = min(ARCH_VGPRS, budget)
    ok = arch <= arch_limit * REG_SLACK and total <= budget * REG_SLACK
    return {"arch": arch, "acc": acc, "total": total, "budget": budget, "waves_per_simd": per_simd, "ok": ok}


def legal(mod: ModuleType, spec) -> Tuple[bool, str]:
    """Cheap legality: the kernel's ``validate_spec``, LDS, and the register estimate."""
    try:
        ok, why = mod.validate_spec(spec)
    except Exception as e:  # noqa: BLE001 - a validator crash is an illegal config
        return False, f"validate_spec raised {type(e).__name__}: {e}"
    if not ok:
        return False, why
    if hasattr(mod, "lds_bytes") and mod.lds_bytes(spec) > LDS_LIMIT:
        return False, f"LDS {mod.lds_bytes(spec)} B > {LDS_LIMIT}"
    try:
        r = reg_estimate(mod, spec)
    except Exception as e:  # noqa: BLE001
        return True, f"no register estimate ({type(e).__name__})"
    if not r["ok"]:
        return False, f"register estimate {r['arch']} arch / {r['total']} total > budget {r['budget']}"
    return True, ""


def overrides_of(mod: ModuleType, spec_or_cfg) -> Dict[str, object]:
    """The fields of a spec (or full config dict) that differ from the spec defaults."""
    d = default_config(mod)
    cfg = asdict(spec_or_cfg) if not isinstance(spec_or_cfg, Mapping) else dict(spec_or_cfg)
    return {k: v for k, v in cfg.items() if k in d and v != d[k]}


def label(over: Mapping[str, object]) -> str:
    """Short config label: ``k=v,k=v`` (``base`` when empty)."""
    if not over:
        return "base"
    return ",".join(f"{k}={int(v) if isinstance(v, bool) else v}" for k, v in sorted(over.items()))


# ----------------------------------------------------------------- the space


class Mxfp6Space:
    """The search space over ``Mxfp6GemmSpec`` knobs around a base config.

    ``base``: overrides on the spec defaults (the warm start: the kernel
    agent's current best); ``exclude``: knob names held at the base;
    ``extra``: ``{field: levels}`` added to / replacing table entries;
    ``pseudo``: ``{name: levels}`` search axes that are not spec fields (the
    compiler ``backend``: ``("llvm", "rust")``); the first level is the base,
    they ride in the decoded config / overrides and never reach the spec.
    """

    def __init__(
        self,
        mod: Optional[ModuleType] = None,
        *,
        base: Optional[Mapping[str, object]] = None,
        exclude: Sequence[str] = (),
        extra: Optional[Mapping[str, Sequence[object]]] = None,
        auto_unlisted: Optional[bool] = None,
        shape: Optional[Tuple[int, int, int]] = None,
        pseudo: Optional[Mapping[str, Sequence[object]]] = None,
    ):
        self.mod = mod or kernel_module()
        self.defaults = default_config(self.mod)
        self.pseudo: Dict[str, Tuple] = {k: tuple(dict.fromkeys(v)) for k, v in (pseudo or {}).items()}
        clash = [k for k in self.pseudo if k in self.defaults]
        if clash:
            raise ValueError(f"pseudo knobs {clash} shadow spec fields")
        base = dict(base or {})
        self.base_pseudo = {k: base.pop(k, v[0]) for k, v in self.pseudo.items()}
        if shape is not None:
            base.update(m=shape[0], n=shape[1], k=shape[2])
        unknown = [k for k in base if k not in self.defaults]
        if unknown:
            raise ValueError(f"base sets unknown spec fields {unknown}")
        self.base_spec = replace(self.mod.Mxfp6GemmSpec(), **base)
        ok, why = legal(self.mod, self.base_spec)
        if not ok:
            raise ValueError(f"base config is not legal: {why}")
        knobs, fixed = knob_table(self.mod, auto_unlisted=auto_unlisted)
        for name, levels in (extra or {}).items():
            if name not in self.defaults:
                raise ValueError(f"extra knob {name!r} is not a spec field")
            knobs[name] = tuple(dict.fromkeys(levels))
            fixed.pop(name, None)
        for name in exclude:
            if knobs.pop(name, None) is not None:
                fixed[name] = "excluded by the caller"
        for name, levels in self.pseudo.items():
            if name not in exclude and len(levels) > 1:
                knobs[name] = levels
        self.knobs = knobs
        self.fixed = fixed
        base_values = {**asdict(self.base_spec), **self.base_pseudo}
        axes = [choices(name, levels) for name, levels in knobs.items()]
        self._legal_cache: Dict[str, Tuple[bool, str]] = {}
        self.space = SearchSpace(
            params_from_axes(axes, base_values),
            is_valid=self._is_valid,
            sampler=self._sample,
            name="mxfp6_gemm_gfx950",
            fingerprint_extra=json.dumps({"base": overrides_of(self.mod, self.base_spec), "pseudo": self.base_pseudo,
                                          "kernel": kernel_fingerprint(self.mod)}, sort_keys=True, default=str),
        )
        self.base_genome: Genome = self.space.encode({})
        self.rejects: Dict[str, int] = {}

    # ------------------------------------------------------------ mapping
    def _cfg(self, genome_or_cfg) -> Dict[str, object]:
        return dict(genome_or_cfg) if isinstance(genome_or_cfg, Mapping) else self.space.decode(genome_or_cfg)

    def spec_of(self, genome_or_cfg) -> object:
        cfg = self._cfg(genome_or_cfg)
        return replace(self.base_spec, **{k: v for k, v in cfg.items() if k not in self.pseudo})

    def pseudo_of(self, genome_or_cfg) -> Dict[str, object]:
        """The pseudo-knob values of a config (base values filled in)."""
        cfg = self._cfg(genome_or_cfg)
        return {k: cfg.get(k, v) for k, v in self.base_pseudo.items()}

    def overrides(self, genome_or_cfg) -> Dict[str, object]:
        """Overrides on the spec defaults (what ``bench --arms`` / the kernel agent take), plus every pseudo knob's
        value (explicit even at its base level, so a config dict always names its compiler)."""
        out = overrides_of(self.mod, self.spec_of(genome_or_cfg))
        out.update(self.pseudo_of(genome_or_cfg))
        return out

    def diff_vs_base(self, genome_or_cfg) -> Dict[str, object]:
        b = {**asdict(self.base_spec), **self.base_pseudo}
        cur = {**asdict(self.spec_of(genome_or_cfg)), **self.pseudo_of(genome_or_cfg)}
        return {k: v for k, v in cur.items() if v != b[k]}

    def base_value(self, name: str):
        return self.base_pseudo[name] if name in self.pseudo else getattr(self.base_spec, name)

    def genome_of(self, overrides: Mapping[str, object]) -> Genome:
        """The genome of a config given as overrides on the spec defaults (repaired if needed)."""
        b = {**asdict(self.base_spec), **self.base_pseudo}
        full = {**self.defaults, **{k: v[0] for k, v in self.pseudo.items()}, **b, **dict(overrides)}
        cfg = {k: v for k, v in full.items() if k in b and v != b[k]}
        return self.space.encode(cfg)

    def key(self, genome) -> Optional[str]:
        return self.space.key(genome)

    # ------------------------------------------------------------ sampling
    def _sample(self, rng) -> Optional[Dict[str, object]]:
        """A legal config near the base: 1..8 random knob changes (geometric-ish), legality-checked. Uniform
        genomes over many coupled knobs are almost never legal; this walk keeps random / space-filling draws valid."""
        names = list(self.knobs)
        b = {**asdict(self.base_spec), **self.base_pseudo}
        for _ in range(24):
            k = min(len(names), 1 + int(rng.expovariate(0.45)))
            cfg = {}
            for name in rng.sample(names, k):
                v = rng.choice(self.knobs[name])
                if v != b[name]:
                    cfg[name] = v
            if cfg and self._is_valid(cfg):
                return cfg
        return None

    # ------------------------------------------------------------ legality
    def _is_valid(self, config: dict) -> bool:
        k = json.dumps(sorted(config.items()), default=str)
        hit = self._legal_cache.get(k)
        if hit is None:
            try:
                spec = replace(self.base_spec, **{k: v for k, v in config.items() if k not in self.pseudo})
            except TypeError as e:
                hit = (False, str(e))
            else:
                hit = legal(self.mod, spec)
            self._legal_cache[k] = hit
            if not hit[0]:
                reason = hit[1].split(":")[0][:60]
                self.rejects[reason] = self.rejects.get(reason, 0) + 1
        return hit[0]

    def legal(self, spec) -> Tuple[bool, str]:
        return legal(self.mod, spec)

    # ------------------------------------------------------------ report
    def describe(self) -> dict:
        return {
            "kernel_fingerprint": kernel_fingerprint(self.mod),
            "space_fingerprint": self.space.fingerprint(),
            "base": overrides_of(self.mod, self.base_spec),
            "pseudo": {k: list(v) for k, v in self.pseudo.items()},
            "knobs": {k: list(v) for k, v in self.knobs.items()},
            "fixed": dict(self.fixed),
            "params": [p.describe() for p in self.space.params],
            "size_bound": self.space.size(),
        }
