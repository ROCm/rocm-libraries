# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Offline replay of recorded monokernel A/B data through the search engine.

Two recorded data shapes become replay problems (a search space over a
catalog adapter restricted to the recorded configs, a backend, the recorded
optimum):

- **round tables** (:func:`load_sched_rounds`): the schedule track's in-situ
  layer A/B files (``ab_<round>_<kind>_<variant>_abs.json``: per round, every
  arm's ratio vs that round's mono, arms shuffled every round, controls
  ``mono2`` and ``mono_rc``). :class:`RoundReplayBackend` serves round ``r``
  of every arm of one interleaved batch together, so the pairing of the
  recorded rounds is preserved and the controls are real.
- **lookback grids** (:func:`load_lookback_grid`): a ``climb_b3.py`` record
  (``lookback``: every cell of the signed-lookback grid timed against the
  climb's best; ``lookback_confirm``: the top cells re-timed). Each cell has
  one recorded batch, so the replay noise is the engine's
  :class:`~rocke.dispatch.tuning.search.measure.TableBackend` noise, sized
  from the recorded re-timings of the confirmed cells (data, not a guess).

Arm names are parsed into knob configs of the catalog adapters (``hb132``
-> ``hold.b1=32``; ``rk:split=2,mb_impl=ir`` -> ``split=2,
mailbox.impl=ir``). Ratios only: these files carry absolute times next to
the ratios, and only the ratios are read.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rocke.dispatch.tuning.search.measure import CompileOutcome
from rocke.dispatch.tuning.search.params import Param, SearchSpace
from rocke.dispatch.tuning.search.trial import Fidelity

__all__ = [
    "LookbackGrid",
    "RoundReplayBackend",
    "RoundTable",
    "load_lookback_grid",
    "load_sched_rounds",
    "parse_b3_arm",
    "parse_sched_arm",
    "listed_space",
]

_HOLD = re.compile(r"^h([a-z]\d)(\d+)$")


def parse_sched_arm(name: str) -> dict | None:
    """A schedule-rig arm name as ``ATOM_Q38_SCHED`` knobs, or ``None`` for
    the baseline / controls. ``mono`` -> {}; ``N`` / ``N_<holds>`` ->
    ``numa=1`` + holds; ``N_copy0`` / ``N_remote`` -> ``numa=2`` / ``3``;
    ``hb132`` -> ``hold.b1=32``; ``hd128`` -> ``head_sleep=128``; the
    round-1 names ``hb1_32`` / ``hd_128`` / ``hd_none`` too."""
    if name in ("mono",):
        return {}
    if name in ("mono2", "mono_rc"):
        return None
    knobs: dict[str, int] = {}
    toks = name.split("_")
    if toks[0] == "N":
        knobs["numa"] = 1
        toks = toks[1:]
        if toks == ["copy0"]:
            return {"numa": 2}
        if toks == ["remote"]:
            return {"numa": 3}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("hd"):
            v = t[2:] or (toks[i + 1] if i + 1 < len(toks) else "")
            if not t[2:]:
                i += 1
            knobs["head_sleep"] = 0 if v == "none" else int(v)
        elif _HOLD.match(t):
            st, v = _HOLD.match(t).groups()
            knobs[f"hold.{st}"] = int(v)
        elif re.fullmatch(r"h[a-z]\d", t) and i + 1 < len(toks):
            knobs[f"hold.{t[1:]}"] = int(toks[i + 1])
            i += 1
        else:
            return None  # an arm outside the hold / numa space (pause, place, ...)
        i += 1
    return knobs


def parse_b3_arm(name: str) -> dict:
    """``rk:split=2,wait=barrier,mb_impl=ir`` -> ``{"split": 2, "wait":
    "barrier", "mailbox.impl": "ir"}``."""
    body = name.split(":", 1)[1] if ":" in name else name
    out = {}
    for part in filter(None, body.split(",")):
        k, _, v = part.partition("=")
        if k.startswith("mb_"):
            k = "mailbox." + k[3:]
        if v in ("True", "False"):
            val: object = v == "True"
        else:
            try:
                val = int(v)
            except ValueError:
                val = v
        out[k] = val
    return out


@dataclass
class RoundTable:
    """Per config (canonical knob dict as sorted JSON): per-round ratios vs
    the baseline, plus the controls' rounds."""

    configs: dict[str, dict] = field(default_factory=dict)
    rounds: dict[str, list[float]] = field(default_factory=dict)
    controls: dict[str, list[float]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def add(self, knobs: Mapping, ratios: Sequence[float], source: str) -> None:
        k = json.dumps(dict(sorted(knobs.items())), sort_keys=True)
        if k in self.rounds:
            return  # first source wins (same arm re-timed in a later round file)
        self.configs[k] = dict(knobs)
        self.rounds[k] = [float(x) for x in ratios]
        self.sources[k] = source

    def medians(self) -> dict[str, float]:
        return {k: statistics.median(v) for k, v in self.rounds.items()}

    def oracle(self) -> tuple[str, float]:
        m = self.medians()
        k = min(m, key=lambda x: (m[x], x))
        return k, m[k]


def load_sched_rounds(files: Sequence[str | Path], variant: str) -> RoundTable:
    """Merge the round files of one variant (``r1`` / ``r4s``) into a
    :class:`RoundTable` of hold / numa configs."""
    t = RoundTable()
    for f in files:
        d = json.loads(Path(f).read_text())
        if variant not in d:
            continue
        rows = d[variant]
        arms = sorted({a for r in rows for a in r["ratio"]})
        for a in arms:
            vals = [r["ratio"][a] for r in rows if a in r["ratio"]]
            if a in ("mono2", "mono_rc"):
                t.controls.setdefault(a, []).extend(vals)
                continue
            knobs = parse_sched_arm(a)
            if knobs is None:
                continue
            t.add(knobs, vals, Path(f).name)
    return t


class RoundReplayBackend:
    """A :class:`RoundTable` as an engine backend: ``measure_batch`` serves
    one recorded round (the same index for every item of the batch, so the
    batch is paired as the recorded rounds were); ``measure`` (alone) the next
    round of that config. Round order is a seeded permutation per backend."""

    def __init__(self, table: RoundTable, key_of: Mapping[str, str], *, seed: int = 0):
        """``key_of``: engine config key -> table key (JSON knobs)."""
        self.table = table
        self.key_of = dict(key_of)
        self.seed = int(seed)
        self._batch = 0
        self._calls: dict[str, int] = {}
        n = max((len(v) for v in table.rounds.values()), default=1)
        self.order = list(range(n))
        random.Random(f"rounds:{seed}").shuffle(self.order)
        self.measures = 0

    def environment(self) -> dict:
        return {"backend": "round-replay"}

    def compile(self, config, key):
        if key not in self.key_of or self.key_of[key] not in self.table.rounds:
            return CompileOutcome(False, detail="not in the recorded table")
        return CompileOutcome(True, artifact=self.key_of[key])

    def check(self, config, key, artifact):
        return True, "recorded runs passed the bitwise gate (and the mutant failed)"

    def _round(self, tk: str, r: int) -> float:
        v = self.table.rounds[tk]
        return v[self.order[r % len(self.order)] % len(v)]

    def measure(self, config, key, artifact, fidelity: Fidelity):
        self.measures += 1
        n = self._calls.get(artifact, 0)
        self._calls[artifact] = n + fidelity.reps
        return [self._round(artifact, n + i) for i in range(fidelity.reps)]

    def measure_batch(self, items, fidelity: Fidelity):
        self.measures += len(items)
        r0 = self._batch * fidelity.reps
        self._batch += 1
        return [[self._round(art, r0 + i) for i in range(fidelity.reps)] for _c, _k, art in items]


@dataclass
class LookbackGrid:
    """A recorded signed-lookback grid: knobs of each cell, its ratio vs the
    climb's best, the confirmation re-timings and the noise they imply."""

    variant: str
    configs: dict[str, dict]
    value: dict[str, float]
    confirm: dict[str, float]
    climb_best: dict
    start: dict
    noise: float
    space_values: dict

    def oracle(self) -> tuple[str, float]:
        k = min(self.value, key=lambda x: (self.value[x], x))
        return k, self.value[k]


def load_lookback_grid(path: str | Path, *, start: Mapping | None = None) -> LookbackGrid:
    """A ``climb_b3.py`` record. ``start``: the cold-start knobs (default:
    the grid cell that differs from the base build only by the schedule's
    ``numa`` lever, i.e. the shipped B3 on the NUMA layout)."""
    d = json.loads(Path(path).read_text())
    configs, value = {}, {}
    for arm, rec in d["lookback"].items():
        k = parse_b3_arm(arm)
        key = json.dumps(dict(sorted(k.items())), sort_keys=True)
        configs[key] = k
        value[key] = float(rec["ratio_vs_climb_best"])
    confirm = {}
    diffs = []
    for arm, rec in (d.get("lookback_confirm") or {}).items():
        key = json.dumps(dict(sorted(parse_b3_arm(arm).items())), sort_keys=True)
        confirm[key] = float(rec["ratio_vs_climb_best"])
        if key in value:
            diffs.append(abs(math.log(confirm[key]) - math.log(value[key])))
    # per-call sd from re-timed pairs: median |d| = 0.6745 sqrt(2) sd
    noise = statistics.median(diffs) / (0.6745 * math.sqrt(2.0)) if len(diffs) >= 3 else 0.003
    st = dict(start) if start is not None else {"numa": 1}
    ls = d.get("lookback_space") or {}
    # newer records map lever -> values; older ones list the levers only
    space_values = {k: [_lit(v) for v in vals] for k, vals in ls.items()} if isinstance(ls, Mapping) else {}
    return LookbackGrid(
        variant=d.get("variant", ""),
        configs=configs,
        value=value,
        confirm=confirm,
        climb_best=parse_b3_arm("rk:" + d.get("climb_best", "")),
        start=st,
        noise=max(noise, 0.001),
        space_values={("mailbox." + k[3:] if k.startswith("mb_") else k): v for k, v in space_values.items()},
    )


def _lit(v: str):
    v = v.strip("'\"")
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        return v


def listed_space(
    configs: Mapping[str, Mapping],
    defaults: Mapping[str, object],
    *,
    name: str,
    order: Mapping[str, Sequence] | None = None,
) -> tuple[SearchSpace, dict]:
    """A search space whose valid points are exactly the recorded
    ``configs`` (``{table key: knobs}``), with the shipped default
    (``defaults[knob]``) at level 0 of every knob, so the all-zero genome is
    the shipped build as in the catalog adapters. Levels follow ``order``
    (the adapter's domain) where given. Repair and sampling land on recorded
    configs only (the engine's member index). Returns ``(space, {engine key:
    table key})``."""
    from rocke.dispatch.tuning.search.params import _MemberIndex, config_items_key

    rows = list(configs.items())
    knobs = sorted({k for _t, c in rows for k in c})
    params = []
    for k in knobs:
        seen = [c.get(k, defaults.get(k)) for _t, c in rows]
        vals = list(dict.fromkeys(seen))
        if order and k in order:
            rank = {repr(v): i for i, v in enumerate(order[k])}
            vals.sort(key=lambda v: rank.get(repr(v), len(rank)))
        d = defaults.get(k)
        if d in vals:
            vals.remove(d)
        params.append(Param.categorical(k, [d] + vals))
    flat = [{k: c.get(k, defaults.get(k)) for k in knobs} for _t, c in rows]
    members = {}
    for i, f in enumerate(flat):
        members.setdefault(config_items_key(f), i)
    member_rows = [flat[i] for i in members.values()]
    index = _MemberIndex(params, member_rows)
    space = SearchSpace(
        params,
        canonical=lambda c: config_items_key(c) if config_items_key(c) in members else None,
        name=name,
        fingerprint_extra=sorted(members),
        sampler=lambda rng: member_rows[rng.randrange(len(member_rows))],
        nearest=index.nearest,
        listed=index.genomes,
        nearest_first=len(member_rows) / max(1, math.prod(p.size for p in params)) < 0.25,
    )
    space._members = members
    index.bind(space)
    key_map = {config_items_key(f): tk for (tk, _c), f in zip(rows, flat)}
    return space, key_map
