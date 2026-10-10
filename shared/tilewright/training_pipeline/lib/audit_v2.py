# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Audit protocol v2: each audit GEMM's near-winner kernels and the picks of
the models scored, benched together in repeated separate processes and
scored by per-kernel medians.

The exhaustive pass (protocol v1, lib/audit.py) reads each kernel of an
audit GEMM once, so a host-side glitch of one reading moves a winner or a
pick. Protocol v2 measures again only what the scores depend on:

Sets. The set of a GEMM of a group is its candidates, the kernels within
`set_tolerance` of the exhaustive pass's winner (at most `set_max`, fastest
first, a skip row counted at its warm-up), plus the top-1 pick of every
model registered with the v2 measurement (`models/`).

Bench. Every part of a generation is benched `repetitions` times, each a
separate stage03 process (one GPU, adaptive timing, no skip screen; a
context group through the stage03c co-tenant wrapper) over the part's
GEMMs in an order shuffled per repetition, each GEMM's set in one line, the
whole set in a seeded shuffle of its own per repetition (`kernel_order`
shuffled) led by a sacrificial copy of its first kernel, whose row the log
readers set aside (`sacrificial_kernel`). A long line runs its first kernels
before the device settles at its power limit and its last ones after, so a
fixed order would time a kernel listed late (a pick appended to its set) at
a lower clock than the near-winners before it; the order each repetition
timed is recorded per GEMM (`orders`). Every process starts with stage03's
discarded warm-up line (`warmup_line`; its kernels one of each library file
the manifest's bench setup names, in turn). A measurement planned before
these settings existed keeps the order and the processes it was benched
with (`PREDATING`: each set as planned, rotated by the repetition, no copy,
no warm-up line). A canary, one fixed launch-bound
GEMM of the exclusive group timed alone (`canary_timings` problems in one
process, its reading their median), is read before and after the
repetitions; when either reading is more than `canary_tolerance` off the
canary's reference median on that GPU (the median of
`canary_reference_readings` readings taken once), the part is benched again
after `canary_backoff_s`, at most `canary_attempts` times, and then flagged.
A GEMM whose winner or a pick spreads more than `spread_limit`
((max - min) / median of its readings) is benched once more on its own
(the same scheme) and its new readings replace the old; one that still
spreads is flagged. Flags are recorded, never dropped.

Hung lines. A repetition's stage03 run benches a line that hangs once
more in its retry pass, in a process of its own, and quarantines it in the
run when that does not bench it to the end, under the measurement's own
protocol (`hung_scope`: audit_v2, or audit_sustained for the sustained
measurement), which no other bench of the run reads; the canary's runs
quarantine nothing. A GEMM whose line is quarantined so, or that a run
leaves out because it was quarantined before, is dropped from its part
(`dropped` in the GEMM's result: why, the attempt and the repetition), and
counted and left out of the scores, not waited for (lib/audit.py).

Unmeasured parts. A stage03 run that fails without a bench of its lines (no
block log, nothing hung or left out: its GPU could not bench, such as one
the run holds alone, which the runs leave out at once; `Run.unmeasured`)
measures nothing, so its part is left unwritten, its claim released for
another worker. A worker takes no part while the run holds its GPU
(`work`), and stops once no part is left.

Scores. The median of a kernel needs `min_readings` readings. The winner of
a GEMM is the smallest median of its set, and a pick's efficiency is the
winner's median over the pick's (lib/audit.py `evaluate`, record `v2`).
Kernels outside a set keep their single exhaustive reading; they are over
`set_tolerance` behind the winner, so they enter only ceilings below
1 / (1 + set_tolerance).

Increments. Scoring a model whose pick is not in a GEMM's set needs that
GEMM benched again: `plan` with the model registered starts a generation of
the GEMMs whose sets grew, each with its whole set (candidates and every
pick), so a winner and the picks are always measured together. A GEMM's
measurement is its latest complete generation's; while a newer generation
is benched the older one still scores the models whose picks it holds.

Builds. Picks, sets, bench lines and medians name kernels by solution
index, which only one hipBLASLt build defines (lib/kernel_identity.py). With
the library's identity, `plan` records the kernel ids of every model's picks
and of every generation's kernels, and refuses to plan on top of records of
another build; the bench (`check_bench_library`) refuses a generation whose
indices name other kernels in the library it benches with; and
`measurements` maps a generation's medians to the indices the library gives
their kernels now. Records written before kernel identities were recorded
are judged and mapped the same way by the audit set's kernel digests as of
their plan (`AuditKernels`); a GEMM with a median of a kernel those digests
cannot place is not scored. The bench takes the library a manifest without
`bench.library` benches with from its config, as `plan` does.

Layout of `<audit>/v2/`:
  v2.json                     manifest: settings, canary, models, bench
                              setup (the library files included), generations
  models/<name>.json          a model's picks per group and GEMM
  canary/reference_<gpu>.json the canary's reference readings on one GPU
  bench/<group>/              the context wrappers (hipblaslt-bench,
                              context.json, cotenant_logs/)
  gen_NNN/plan.json           per group, the parts: GEMMs with their sets
                              and bench lines
  gen_NNN/<group>/part_NNNN/  a<k>/ per attempt (canary_start/, rep_<r>/,
                              canary_end/: stage03 in- and out-dirs),
                              spread/ for the spread re-bench, and
                              result.json once the part is complete
  progress.json               GEMMs and parts done per group
  sustained/                  the sustained composite's measurement, in
                              this layout (lib/audit_sustained.py)
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from . import kernel_identity as ki
from . import rebench as rb
from .bench_log import iter_problems

DIRNAME = "v2"
MANIFEST = "v2.json"
SCHEMA = "audit-v2"
RESULT = "result.json"
PROGRESS = "progress.json"
DEFAULTS: Dict[str, Any] = {
    "repetitions": 5,
    "set_tolerance": 0.10,
    "set_max": 20,
    "spread_limit": 0.30,
    "min_readings": 3,
    "canary_tolerance": 0.05,
    "canary_attempts": 3,
    "canary_timings": 3,
    "canary_reference_readings": 5,
    "canary_backoff_s": 120.0,
    "seed": 0,
    "kernel_order": "shuffled",
    "sacrificial_kernel": True,
    "warmup_line": True,
}
KERNEL_ORDERS = ("shuffled", "rotated")
# How a measurement whose settings predate `kernel_order`,
# `sacrificial_kernel` and `warmup_line` was benched; its later generations
# keep it.
PREDATING: Dict[str, Any] = {
    "kernel_order": "rotated",
    "sacrificial_kernel": False,
    "warmup_line": False,
}
Shape = Tuple[int, int, int, int]
Key = Tuple[str, Shape]


class V2Error(RuntimeError):
    """The v2 measurement cannot be planned or benched."""


def v2_dir(audit_dir: Path) -> Path:
    return Path(audit_dir) / DIRNAME


def _read(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _write(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def manifest(d: Path) -> Optional[Dict[str, Any]]:
    m = _read(Path(d) / MANIFEST)
    return m if isinstance(m, dict) and m.get("schema") == SCHEMA else None


def settings_of(m: Mapping[str, Any]) -> Dict[str, Any]:
    """The settings a measurement benches with: its manifest's over DEFAULTS,
    a setting its manifest predates as it was benched then (PREDATING)."""
    return {**DEFAULTS, **PREDATING, **dict(m.get("settings") or {})}


def rep_order(
    kernels: Sequence[int], st: Mapping[str, Any], rep: int, key: Sequence[Any]
) -> List[int]:
    """The order repetition `rep` times a GEMM's set in: the whole set in a
    shuffle seeded by `st["seed"]`, `key` (what the round benches: its tag,
    attempt and GEMM) and `rep` (`kernel_order` shuffled), else the set as
    planned rotated left by `rep` places (rotated)."""
    if st["kernel_order"] == "rotated":
        return rb.rotate([int(k) for k in kernels], rep)
    from .planner import shuffled

    return shuffled([int(k) for k in kernels], int(st["seed"]), *key, rep)


def shape_of(g: Mapping[str, Any]) -> Shape:
    return (int(g["m"]), int(g["n"]), int(g["k"]), int(g["batch_count"]))


def gen_dir(d: Path, gen: int) -> Path:
    return Path(d) / f"gen_{int(gen):03d}"


def part_path(d: Path, gen: int, label: str, i: int) -> Path:
    return gen_dir(d, gen) / label / f"part_{int(i):04d}"


# ── sets ─────────────────────────────────────────────────────────────────────


def candidates(g: Mapping[str, Any], tolerance: float, max_kernels: int) -> List[int]:
    """The kernels of an exhaustive GEMM within `tolerance` of its winner,
    fastest first, at most `max_kernels`."""
    limit = float(g["winner_us"]) * (1.0 + float(tolerance))
    rows = sorted(
        (float(c["us"]), int(c["sol_idx_global"]))
        for c in g["candidates"]
        if float(c["us"]) > 0 and float(c["us"]) <= limit
    )
    return [sid for _, sid in rows[: max(1, int(max_kernels))]]


def choose_canary(gemms: Iterable[Mapping[str, Any]]) -> Tuple[Shape, int]:
    """(shape, kernel) of the canary: the exclusive GEMM with the least
    work (launch-bound) and its exhaustive winner."""
    best = None
    for g in gemms:
        tested = [c for c in g["candidates"] if not c["is_skip"] and c["us"] > 0]
        if not tested:
            continue
        s = shape_of(g)
        win = min(tested, key=lambda c: (c["us"], c["sol_idx_global"]))
        key = (s[0] * s[1] * s[2] * s[3], float(win["us"]), s)
        if best is None or key < best[0]:
            best = (key, s, int(win["sol_idx_global"]))
    if best is None:
        raise V2Error("no exclusive audit GEMM has a timed kernel for the canary")
    return best[1], best[2]


# ── models and plans ─────────────────────────────────────────────────────────


def _model_name_ok(name: str) -> bool:
    return bool(name) and all(ch.isalnum() or ch in "._-" for ch in name)


def models(d: Path) -> Dict[str, Dict[str, Any]]:
    """{name: record} of the registered models (sha256, source, picks
    {label: [[m, n, k, b, sid], ...]})."""
    out = {}
    for p in sorted((Path(d) / "models").glob("*.json")):
        rec = _read(p)
        if isinstance(rec, dict):
            out[p.stem] = rec
    return out


def picks_of(rec: Mapping[str, Any]) -> Dict[Key, int]:
    out: Dict[Key, int] = {}
    for label, rows in (rec.get("picks") or {}).items():
        for m, n, k, b, sid in rows:
            out[(label, (int(m), int(n), int(k), int(b)))] = int(sid)
    return out


def plans(d: Path) -> List[Dict[str, Any]]:
    m = manifest(d) or {}
    out = []
    for gen in m.get("generations") or []:
        p = _read(gen_dir(d, gen["index"]) / "plan.json")
        if isinstance(p, dict):
            out.append(p)
    return out


def current_sets(d: Path) -> Dict[Key, List[int]]:
    """{(label, shape): kernels} of the latest generation planning each
    GEMM."""
    out: Dict[Key, List[int]] = {}
    for p in plans(d):
        for label, parts in p["groups"].items():
            for part in parts:
                for e in part:
                    out[(label, tuple(e["shape"]))] = list(e["kernels"])
    return out


@dataclass
class Model:
    name: str
    sha256: str
    source: str
    picks: Dict[Key, int]


def _time(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class AuditKernels:
    """What the solution indices of a v2 record written before kernel
    identities were recorded name, by the audit set's kernel digests
    (`kernel_digest`). The code that wrote such records planned only while
    each solution index of the pool named the kernel the latest generation
    benching that index had recorded under it, so a record planned at a
    given time meant, under an index, the digest the latest generation
    finished by then recorded there (`at`). Its medians are read on the
    assumption, which that code did not check, that it was benched with
    the build it was planned with.

    `then` holds the digest each index named (None where the generations
    do not tell; without a plan time, where they recorded several under
    it), `now` the pool's digests, `by_digest` the index of each (None
    where the pool holds it under several), and `generations` each
    generation's finish time and digests, oldest first."""

    then: Mapping[int, Optional[str]]
    now: Mapping[int, str]
    by_digest: Mapping[str, Optional[int]]
    generations: Tuple[Tuple[Optional[datetime], Mapping[int, str]], ...] = ()

    @classmethod
    def of(
        cls, audit_manifest: Mapping[str, Any], now: Mapping[int, str]
    ) -> "AuditKernels":
        """From the audit set's manifest and the pool's {index: digest}: the
        view of a record whose plan time is not known (every generation)."""
        gens = tuple(
            (
                _time(gen.get("finished")),
                {int(s): str(x) for s, x in (gen.get("kernels") or {}).items()},
            )
            for gen in audit_manifest.get("generations") or []
        )
        then: Dict[int, Optional[str]] = {}
        for _, kernels in gens:
            for sid, digest in kernels.items():
                then[sid] = digest if then.get(sid, digest) == digest else None
        by_digest: Dict[str, Optional[int]] = {}
        for sid, digest in sorted(now.items()):
            by_digest[digest] = None if digest in by_digest else int(sid)
        return cls(then, dict(now), by_digest, gens)

    def at(self, planned: Any) -> "AuditKernels":
        """The view of a record planned at `planned` (its ISO time): the
        generations finished by then, a later one's digest under an index
        replacing an earlier one's. This view when that time or a
        generation's finish time is not known."""
        t = _time(planned)
        if t is None or any(f is None for f, _ in self.generations):
            return self
        then: Dict[int, Optional[str]] = {}
        try:
            for finished, kernels in self.generations:
                if finished <= t:
                    then.update(kernels)
        except TypeError:
            return self
        return replace(self, then=then)

    def index(self, sid: int) -> Optional[int]:
        """The index the kernel of `sid` has in the pool now; None when it
        left the pool or cannot be placed (`unplaced`). Negative ids
        stay."""
        sid = int(sid)
        if sid < 0:
            return sid
        digest = self.then.get(sid)
        if digest is None:
            return None
        if self.now.get(sid) == digest:
            return sid
        return self.by_digest.get(digest)

    def unplaced(self, sids: Iterable[int]) -> List[int]:
        """Those of `sids` that `index` cannot place: the generations do not
        tell which kernel the index named, or the pool holds that kernel
        under several other indices now. A kernel that left the pool is not
        among them; no model can pick it."""
        return [
            s
            for s in sorted({int(s) for s in sids})
            if self.index(s) is None
            and (self.then.get(s) is None or self.then[s] in self.by_digest)
        ]

    def moves(self, sids: Iterable[int]) -> Dict[int, Optional[int]]:
        """{index: `index` of it} of those of `sids` that name another
        kernel or none now, or one the generations do not tell."""
        return {
            s: self.index(s)
            for s in sorted({int(s) for s in sids})
            if self.index(s) != s
        }

    def explain(
        self, what: str, sids: Iterable[int], identity: Optional[ki.LibraryIdentity]
    ) -> str:
        """Why the indices `sids` of record `what` do not name its kernels
        now, with an example (an index that cannot be placed first), naming
        a kernel by its kernel id in `identity`'s library when the pool
        still holds it."""
        named = {int(s) for s in sids}
        moved = self.moves(named)
        s = next(iter(self.unplaced(moved) or moved))
        now, digest = moved[s], self.then.get(s)
        if digest is None:
            example = (
                f"the audit generations do not tell which kernel solution {s} named"
            )
        elif now is not None:
            kid = identity.ids.get(now) if identity is not None else None
            kernel = kid or f"the kernel of audit digest {digest}"
            example = f"solution {s} named {kernel}, at solution {now} now"
        elif digest in self.by_digest:
            example = (
                f"solution {s} named the kernel of audit digest {digest}, which "
                f"the pool holds under several solution indices now"
            )
        else:
            example = (
                f"solution {s} named the kernel of audit digest {digest}, which "
                f"the pool no longer holds"
            )
        return (
            f"{what} records no kernel identities, and by the audit set's kernel "
            f"digests {len(moved)} of the {len(named)} solution indices it names "
            f"do not name the same kernel now (e.g. {example})"
        )


def plan_indices(doc: Mapping[str, Any]) -> List[int]:
    """The solution indices a generation's plan names: its GEMMs' sets,
    candidates and picks."""
    out = set()
    for parts in (doc.get("groups") or {}).values():
        for part in parts:
            for e in part:
                out.update(int(k) for k in e.get("kernels") or [])
                out.update(int(k) for k in e.get("candidates") or [])
                out.update(int(k) for k in (e.get("picks") or {}).values())
    return sorted(out)


def planned_times(m: Optional[Mapping[str, Any]]) -> Dict[int, Any]:
    """{generation: when its plan was written} of a v2 manifest."""
    return {
        int(g["index"]): g.get("created") for g in (m or {}).get("generations") or []
    }


def require_same_build(
    d: Path,
    identity: Optional[ki.LibraryIdentity],
    kernels: Optional[AuditKernels] = None,
) -> None:
    """V2Error when a generation or a registered model of the measurement in
    `d` names kernels by solution indices that name other kernels in
    `identity`'s library: by its record, or, for one without a record, by
    the audit set's kernel digests (`kernels` as of its plan)."""
    when = planned_times(manifest(d))
    docs = [
        (f"v2 generation {p['index']} in {d}", p, plan_indices(p), when.get(p["index"]))
        for p in plans(d)
    ]
    docs += [
        (
            f"v2 model {name} in {d}",
            rec,
            list(picks_of(rec).values()),
            rec.get("registered"),
        )
        for name, rec in models(d).items()
    ]
    for what, doc, sids, planned in docs:
        why = None
        try:
            rec = ki.read_record(doc)
            if rec is not None and identity is not None:
                ki.Remap.of(rec, identity, what).require_same()
        except (ki.BuildChangedError, ValueError) as e:
            why = str(e)
        else:
            view = kernels.at(planned) if kernels is not None else None
            if rec is None and view is not None and view.moves(sids):
                why = view.explain(what, sids, identity)
        if why is not None:
            raise V2Error(
                f"{why}; the v2 measurement cannot mix builds: bench it with the "
                f"build it was planned with, or start a new one"
            )


def plan(
    d: Path,
    *,
    gemms: Mapping[Key, Mapping[str, Any]],
    lines: Mapping[Shape, str],
    parts: Mapping[str, Sequence[Sequence[Shape]]],
    new_models: Sequence[Model],
    settings: Optional[Mapping[str, Any]] = None,
    bench: Optional[Mapping[str, Any]] = None,
    part_shapes: int = 32,
    identity: Optional[ki.LibraryIdentity] = None,
    kernels: Optional[AuditKernels] = None,
) -> Optional[int]:
    """Register `new_models` and plan the generation that measures them:
    generation 0 (every GEMM of `gemms`, {(label, shape): exhaustive GEMM},
    in the parts `parts` {label: [[shape, ...], ...]}) when there is none,
    else the GEMMs whose set lacks a registered pick, in parts of
    `part_shapes`. `lines` holds each shape's exhaustive bench line, and
    `bench` the bench setup (binary, config, wrappers). With `identity`, the
    library the picks and `gemms` index, the kernel ids are recorded and
    earlier records must name the same kernels (`require_same_build`); with
    `kernels`, earlier records without kernel identities must too.
    Returns the new generation's index, or None when every set holds every
    pick."""
    d = Path(d)
    if identity is not None or kernels is not None:
        require_same_build(d, identity, kernels)
    m = manifest(d)
    if m is None:
        if not gemms:
            raise V2Error("no audit GEMM to plan")
        st = {**DEFAULTS, **dict(settings or {})}
        canary_shape, canary_kernel = choose_canary(
            g for (label, _), g in gemms.items() if label == "exclusive"
        )
        m = {
            "schema": SCHEMA,
            "created": datetime.now().isoformat(timespec="seconds"),
            "settings": st,
            "canary": {
                "label": "exclusive",
                "shape": list(canary_shape),
                "kernel": canary_kernel,
                "line": rb.rebench_line(lines[canary_shape], [canary_kernel]),
                **(
                    {ki.RECORD_KEY: ki.record(identity, [canary_kernel])}
                    if identity is not None
                    else {}
                ),
            },
            "bench": dict(bench or {}),
            "generations": [],
        }
    elif bench:
        m["bench"] = {**dict(m.get("bench") or {}), **dict(bench)}
    st = m["settings"]
    have = models(d)
    for mod in new_models:
        if not _model_name_ok(mod.name):
            raise V2Error(f"model name {mod.name!r} must be letters, digits, . _ -")
        prev = have.get(mod.name)
        if prev is not None and prev.get("sha256") != mod.sha256:
            raise V2Error(
                f"a model named {mod.name!r} with another sha256 is registered in {d}"
            )
    for mod in new_models:
        by_label: Dict[str, List[List[int]]] = {}
        for (label, s), sid in sorted(mod.picks.items()):
            by_label.setdefault(label, []).append([*s, int(sid)])
        rec: Dict[str, Any] = {
            "sha256": mod.sha256,
            "source": mod.source,
            "registered": datetime.now().isoformat(timespec="seconds"),
            "picks": by_label,
        }
        if identity is not None:
            rec[ki.RECORD_KEY] = ki.record(identity, mod.picks.values())
        _write(d / "models" / f"{mod.name}.json", rec)
    picks: Dict[Key, Dict[str, int]] = {}
    for name, rec in models(d).items():
        for key, sid in picks_of(rec).items():
            picks.setdefault(key, {})[name] = sid
    sets = current_sets(d)
    first = not m["generations"]

    def entry(key: Key) -> Dict[str, Any]:
        g = gemms[key]
        cands = candidates(g, st["set_tolerance"], st["set_max"])
        kernels = list(dict.fromkeys([*sets.get(key, []), *cands]))
        row_ids = {int(c["sol_idx_global"]) for c in g["candidates"]}
        for sid in (picks.get(key) or {}).values():
            if sid in row_ids and sid not in kernels:
                kernels.append(sid)
        return {
            "shape": list(key[1]),
            "kernels": kernels,
            "candidates": cands,
            "picks": dict(sorted((picks.get(key) or {}).items())),
            "line": rb.rebench_line(lines[key[1]], kernels),
        }

    groups: Dict[str, List[List[Dict[str, Any]]]] = {}
    for label, label_parts in parts.items():
        if first:
            chosen = [[s for s in p if (label, tuple(s)) in gemms] for p in label_parts]
        else:
            grown = []
            for p in label_parts:
                for s in p:
                    key = (label, tuple(s))
                    if key not in gemms:
                        continue
                    have_set = set(sets.get(key, []))
                    row_ids = {
                        int(c["sol_idx_global"]) for c in gemms[key]["candidates"]
                    }
                    need = {
                        sid
                        for sid in (picks.get(key) or {}).values()
                        if sid in row_ids and sid not in have_set
                    }
                    if need or key not in sets:
                        grown.append(s)
            size = max(1, int(part_shapes))
            chosen = [grown[i : i + size] for i in range(0, len(grown), size)]
        out = [[entry((label, tuple(s))) for s in p] for p in chosen if p]
        if out:
            groups[label] = out
    if not groups:
        _write(d / MANIFEST, m)
        return None
    index = len(m["generations"])
    gd = gen_dir(d, index)
    if gd.exists():
        shutil.rmtree(gd)
    doc: Dict[str, Any] = {
        "index": index,
        "models": sorted(models(d)),
        "groups": groups,
    }
    if identity is not None:
        named = {int(m["canary"]["kernel"])}
        for parts_ in groups.values():
            for part in parts_:
                for e in part:
                    named.update(int(k) for k in e["kernels"])
        doc[ki.RECORD_KEY] = ki.record(identity, named)
    _write(gd / "plan.json", doc)
    m["generations"].append(
        {
            "index": index,
            "created": datetime.now().isoformat(timespec="seconds"),
            "models": [mod.name for mod in new_models],
            "n_gemms": sum(len(p) for parts_ in groups.values() for p in parts_),
            "n_parts": sum(len(parts_) for parts_ in groups.values()),
        }
    )
    _write(d / MANIFEST, m)
    return index


# ── readings ─────────────────────────────────────────────────────────────────


def readings_of_logs(logs: Iterable[Sequence[str]]) -> Dict[Shape, Dict[int, float]]:
    """{shape: {kernel: latency}} of the complete problems of one process's
    logs (a kernel's first timed row)."""
    out: Dict[Shape, Dict[int, float]] = {}
    for lines in logs:
        for prob in iter_problems(lines):
            shape = prob.shape()
            if not prob.complete or shape is None:
                continue
            per = out.setdefault(shape, {})
            for t in prob.tested:
                try:
                    us = float(t.fields["us"])
                except (KeyError, TypeError, ValueError):
                    continue
                if t.sol_index is not None and math.isfinite(us) and us > 0:
                    per.setdefault(int(t.sol_index), us)
    return out


def canary_timings(
    logs: Iterable[Sequence[str]], shape: Shape, kernel: int
) -> List[float]:
    out = []
    for lines in logs:
        for prob in iter_problems(lines):
            if not prob.complete or prob.shape() != tuple(shape):
                continue
            for t in prob.tested:
                if t.sol_index == kernel:
                    try:
                        us = float(t.fields["us"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if math.isfinite(us) and us > 0:
                        out.append(us)
                    break
    return out


def spread(values: Sequence[float]) -> Optional[float]:
    """(max - min) / median, None without a reading."""
    v = [x for x in values if x is not None]
    if not v:
        return None
    med = statistics.median(v)
    return (max(v) - min(v)) / med if med > 0 else None


def medians(
    readings: Mapping[int, Sequence[Optional[float]]], min_readings: int
) -> Dict[int, float]:
    out = {}
    for sid, vals in readings.items():
        v = [x for x in vals if x is not None]
        if len(v) >= int(min_readings):
            out[int(sid)] = float(statistics.median(v))
    return out


# ── bench ────────────────────────────────────────────────────────────────────


@dataclass
class Run:
    """One stage03 run of a Runner: its exit status, block logs, wall time,
    attempts quarantined for another process's use (`quarantined`), the
    GEMMs it dropped: those whose line hung, again in its retry pass, and
    is quarantined now (`hung`), and those it left out because the run had
    quarantined their line before (`left_out`), and the problems its
    summary names."""

    rc: int
    logs: List[List[str]]
    wall_s: float
    quarantined: int = 0
    hung: List[Shape] = field(default_factory=list)
    left_out: List[Shape] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def unmeasured(self) -> bool:
        """The run failed without a bench of its lines: no block log, and no
        line hung or left out (its GPU could not bench: the run held it,
        its cores stayed busy, its lease or platform check was refused). It
        is no measurement of its GEMMs."""
        return self.rc != 0 and not self.logs and not self.hung and not self.left_out


class Unmeasured(RuntimeError):
    """A stage03 run of a part benched nothing (`Run.unmeasured`): the part
    is left unwritten, for a worker whose GPU can bench it."""


def _unmeasured(run_dir: Path, run: Run) -> Unmeasured:
    why = "; ".join(run.problems) or f"see {run_dir / 'stage03.log'}"
    return Unmeasured(
        f"the stage03 run in {run_dir} benched nothing (exit {run.rc}): {why}"
    )


Runner = Callable[[Path, List[str], Path], Run]


def hung_scope(regime: str = "v2") -> str:
    """The protocol a measurement's hung lines are quarantined under in the
    run (lib/gpu_health.py `quarantine_lines`): one per regime, so that a
    line that hung in a v2 repetition leaves the later repetitions and parts
    of that measurement and nothing else."""
    return f"audit_{regime}"


def _dropped_of(s03: Mapping[str, Any]) -> Tuple[List[Shape], List[Shape]]:
    """(hung, left out) of a stage03 summary (`Run`)."""
    guard = s03.get("guard") or {}

    def shapes(recs: Any, want: Callable[[Mapping[str, Any]], bool]) -> List[Shape]:
        out: List[Shape] = []
        for r in recs or []:
            try:
                if isinstance(r, Mapping) and want(r):
                    out.append(tuple(int(v) for v in r["shape"]))  # type: ignore
            except (KeyError, TypeError, ValueError):
                continue
        return out

    return (
        shapes(guard.get("hung_lines"), lambda r: bool(r.get("quarantined"))),
        shapes(guard.get("quarantined_lines_left_out"), lambda r: True),
    )


STAGE03 = (
    Path(__file__).resolve().parent.parent
    / "stages"
    / "stage03_load_balance_offline_tuning.py"
)


def stage03_runner(
    config_yaml: Optional[Path],
    *,
    device: str = "0",
    stall_timeout_s: float = 300.0,
    startup_grace_s: float = 900.0,
    children: Any = None,
    scope: str = hung_scope(),
    retry_hung: bool = True,
    warmup_line: Optional[bool] = None,
    library: Optional[Mapping[str, Any]] = None,
) -> Runner:
    """A Runner benching `lines` in one stage03 run of one block (one bench
    process unless one crashes) on `device`, in `run_dir`: a line that hangs
    is benched once more in the run's retry pass (`retry_hung`) and then
    quarantined in the run under `scope` (stage03 --hung-scope), which
    later runs of that scope leave out; `scope` none quarantines nothing,
    as the canary's runs do. A GPU the run holds alone is left out at once
    (stage03 --leave-held-gpus): the run benches nothing (`Run.unmeasured`),
    so that its part goes back to the workers on the other GPUs. With
    `warmup_line` True or False, every process runs with or without stage03's
    warm-up line (None: as the config says), its kernels from the library
    files of `library` ({"dir", "stems"}, a manifest's `bench.library`)."""

    def run(run_dir: Path, lines: List[str], binary: Path) -> Run:
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        (run_dir / "shapes.yaml").write_text(
            "".join(ln.rstrip("\n") + "\n" for ln in lines)
        )
        cmd = [sys.executable, str(STAGE03), "--in-dir", str(run_dir)]
        cmd += ["--out-dir", str(run_dir / "stage03")]
        cmd += ["--bench-binary", str(binary), "--devices", str(device)]
        cmd += ["--blocks-per-gpu", "1", "--no-canary", "--quiet"]
        cmd += ["--stall-timeout-s", str(stall_timeout_s)]
        cmd += ["--startup-grace-s", str(startup_grace_s)]
        cmd += ["--hung-scope", str(scope), "--leave-held-gpus"]
        if retry_hung:
            cmd.append("--retry-hung")
        if warmup_line is not None:
            cmd += ["--warmup-line" if warmup_line else "--no-warmup-line"]
        if warmup_line is not False and library and library.get("stems"):
            cmd += ["--identity-dir", str(library["dir"])]
            for stem in library["stems"]:
                cmd += ["--identity-stem", str(stem)]
        if config_yaml is not None:
            cmd += ["--config-yaml", str(config_yaml)]
        t0 = time.time()
        with (run_dir / "stage03.log").open("w") as log:
            if children is not None:
                rc = children.popen(cmd, stdout=log, stderr=subprocess.STDOUT).wait()
            else:
                rc = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT
                ).returncode
        logs = [
            rb.read_log_lines(p.read_text(errors="replace"))
            for p in sorted((run_dir / "stage03" / "logs").glob("block_*.log"))
        ]
        s03 = _read(run_dir / "stage03" / "stage03_summary.json") or {}
        hung, left_out = _dropped_of(s03)
        return Run(
            rc,
            logs,
            time.time() - t0,
            len(s03.get("quarantined") or []),
            hung,
            left_out,
            [str(p) for p in s03.get("problems") or []],
        )

    return run


@dataclass
class Bench:
    """What a worker benches with: the manifest's settings, canary and
    bench setup, the runner and the GPU it is on (`gpu`, the key of its
    canary reference), and what tells whether the run holds that GPU
    (`held`: why, else None; lib/gpu_health.py), which `work` reads every
    `poll_s` while it does, for at most `max_wait_s` (0: no limit)."""

    d: Path
    m: Dict[str, Any]
    runner: Runner
    gpu: str
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = lambda s: None
    # the runner of the canary's readings (default `runner`), which never
    # quarantines its line (stage03_runner's scope none)
    canary_runner: Optional[Runner] = None
    held: Callable[[], Optional[str]] = lambda: None
    poll_s: float = 30.0
    max_wait_s: float = 0.0

    @property
    def st(self) -> Dict[str, Any]:
        return settings_of(self.m)

    def binary(self, label: str) -> Path:
        b = self.m.get("bench") or {}
        wrapper = (b.get("wrappers") or {}).get(label)
        if label != "exclusive" and not wrapper:
            raise V2Error(f"no context wrapper for group {label} in {self.d}")
        return Path(wrapper or b["binary"])

    # ── canary ──

    def canary_reading(self, run_dir: Path) -> Dict[str, Any]:
        """One reading of the canary; Unmeasured when its run benched
        nothing."""
        c = self.m["canary"]
        lines = [c["line"]] * int(self.st["canary_timings"])
        runner = self.canary_runner or self.runner
        run = runner(run_dir, lines, self.binary("exclusive"))
        if run.unmeasured:
            raise _unmeasured(run_dir, run)
        t = canary_timings(run.logs, tuple(c["shape"]), int(c["kernel"]))
        return {
            "timings": t,
            "reading": float(statistics.median(t)) if t else None,
            "rc": run.rc,
            "taken": datetime.now().isoformat(timespec="seconds"),
        }

    def reference(self) -> float:
        path = self.d / "canary" / f"reference_{self.gpu}.json"
        rec = _read(path)
        if isinstance(rec, dict) and rec.get("median"):
            return float(rec["median"])
        n = int(self.st["canary_reference_readings"])
        work = self.d / "canary" / f"work_{self.gpu}"
        readings = []
        for i in range(n):
            r = self.canary_reading(work / f"reading_{i}")
            readings.append(r)
        vals = [r["reading"] for r in readings if r["reading"] is not None]
        if len(vals) < max(1, (n + 1) // 2):
            raise V2Error(
                f"the canary gave {len(vals)} of {n} reference readings on GPU "
                f"{self.gpu}; see {work}"
            )
        med = float(statistics.median(vals))
        _write(
            path,
            {
                "gpu": self.gpu,
                "canary": self.m["canary"]["shape"] + [self.m["canary"]["kernel"]],
                "readings": [r["reading"] for r in readings],
                "median": med,
                "taken": datetime.now().isoformat(timespec="seconds"),
            },
        )
        shutil.rmtree(work, ignore_errors=True)
        return med

    def canary_ok(self, r: Mapping[str, Any], ref: float) -> bool:
        v = r.get("reading")
        return v is not None and abs(v / ref - 1.0) <= float(
            self.st["canary_tolerance"]
        )

    # ── one bench round: repetitions between two canary readings ──

    def bench_round(
        self,
        root: Path,
        label: str,
        entries: Sequence[Mapping[str, Any]],
        tag: str,
    ) -> Dict[str, Any]:
        """Bench `entries` (`repetitions` processes between canary
        readings), again while the canary is off (at most
        `canary_attempts` times); the readings of the last attempt, the
        kernel order of each of its repetitions per GEMM (`orders`) and the
        GEMMs it dropped (`dropped`: {shape: why, the repetition}) because
        their line hung, again in its retry pass (`Run.hung`), or the run
        had quarantined it before (`Run.left_out`). Unmeasured when a
        repetition or canary run benched nothing (`Run.unmeasured`)."""
        st = self.st
        ref = self.reference()
        attempts = []
        reads: Dict[Shape, Dict[int, List[Optional[float]]]] = {}
        orders: Dict[Shape, List[List[int]]] = {}
        dropped: Dict[Shape, Dict[str, Any]] = {}
        sacrificial = bool(st["sacrificial_kernel"])
        for a in range(1, int(st["canary_attempts"]) + 1):
            adir = root / f"a{a}"
            shutil.rmtree(adir, ignore_errors=True)
            start = self.canary_reading(adir / "canary_start")
            reps = []
            reads = {
                tuple(e["shape"]): {int(k): [] for k in e["kernels"]} for e in entries
            }
            orders = {tuple(e["shape"]): [] for e in entries}
            dropped = {}
            for r in range(int(st["repetitions"])):
                order = list(entries)
                random.Random(f"{st['seed']}/{tag}/{a}/{r}").shuffle(order)
                lines = []
                for e in order:
                    s = tuple(e["shape"])
                    ks = rep_order(e["kernels"], st, r, (tag, a, *s))
                    orders[s].append(ks)
                    lines.append(rb.reread_line(e["line"], ks, sacrificial=sacrificial))
                run = self.runner(adir / f"rep_{r}", lines, self.binary(label))
                if run.unmeasured:
                    raise _unmeasured(adir / f"rep_{r}", run)
                got = readings_of_logs(run.logs)
                for e in entries:
                    s = tuple(e["shape"])
                    for k in e["kernels"]:
                        reads[s][int(k)].append(got.get(s, {}).get(int(k)))
                now = [("hung", tuple(x)) for x in run.hung]
                now += [("quarantined", tuple(x)) for x in run.left_out]
                for why, s in now:
                    if s in reads and s not in dropped:
                        dropped[s] = {"why": why, "attempt": a, "rep": r}
                        self.log(
                            f"{tag}: GEMM {'x'.join(map(str, s))} dropped in "
                            f"repetition {r}: "
                            + (
                                "its line hung, again in the retry pass"
                                if why == "hung"
                                else "the run quarantined its line before"
                            )
                        )
                reps.append(
                    {
                        "rep": r,
                        "rc": run.rc,
                        "wall_s": round(run.wall_s, 3),
                        "gemms_timed": sum(
                            1 for e in entries if tuple(e["shape"]) in got
                        ),
                        "quarantined": run.quarantined,
                        "gemm_order": [list(e["shape"]) for e in order],
                        **(
                            {"dropped": [list(x) for _, x in now if x in reads]}
                            if any(x in reads for _, x in now)
                            else {}
                        ),
                    }
                )
            end = self.canary_reading(adir / "canary_end")
            ok = self.canary_ok(start, ref) and self.canary_ok(end, ref)
            attempts.append(
                {
                    "attempt": a,
                    "canary_start": start,
                    "canary_end": end,
                    "canary_ok": ok,
                    "reps": reps,
                }
            )
            self.log(
                f"{tag} attempt {a}: canary {start['reading']} / {end['reading']} "
                f"(reference {ref:.4g}) {'ok' if ok else 'off'}"
            )
            if ok:
                break
            if a < int(st["canary_attempts"]):
                self.sleep(float(st["canary_backoff_s"]))
        return {
            "reference": ref,
            "attempts": attempts,
            "canary_flagged": not attempts[-1]["canary_ok"],
            "readings": reads,
            "orders": orders,
            "dropped": dropped,
        }

    def watched(self, e: Mapping[str, Any], med: Mapping[int, float]) -> List[int]:
        """The kernels whose spread is checked: the winner and the picks."""
        out = []
        if med:
            out.append(min(med, key=lambda k: (med[k], k)))
        out += [int(s) for s in (e.get("picks") or {}).values()]
        return list(dict.fromkeys(out))

    def wide(self, e: Mapping[str, Any], reads: Mapping[int, Sequence]) -> bool:
        med = medians(reads, self.st["min_readings"])
        lim = float(self.st["spread_limit"])
        for k in self.watched(e, med):
            s = spread([x for x in reads.get(k, []) if x is not None])
            if s is not None and s > lim:
                return True
        return False

    def bench_part(self, gen: int, label: str, i: int, entries: List[Dict[str, Any]]):
        """Bench one part and write its result.json."""
        t0 = time.time()
        root = part_path(self.d, gen, label, i)
        main = self.bench_round(root, label, entries, f"g{gen}/{label}/{i}")
        reads, orders = main["readings"], main["orders"]
        dropped = dict(main["dropped"])
        wide = [
            e
            for e in entries
            if tuple(e["shape"]) not in dropped
            and self.wide(e, reads[tuple(e["shape"])])
        ]
        respread = None
        if wide:
            respread = self.bench_round(
                root / "spread", label, wide, f"g{gen}/{label}/{i}/spread"
            )
            for e in wide:
                reads[tuple(e["shape"])] = respread["readings"][tuple(e["shape"])]
                orders[tuple(e["shape"])] = respread["orders"][tuple(e["shape"])]
            for s, why in respread["dropped"].items():
                dropped.setdefault(s, dict(why, spread=True))
        wide_ids = {tuple(e["shape"]) for e in wide}
        st = self.st
        gemms = []
        for e in entries:
            s = tuple(e["shape"])
            med = medians(reads[s], st["min_readings"])
            win = min(med, key=lambda k: (med[k], k)) if med else None
            watched = self.watched(e, med)
            spreads = {
                str(k): spread([x for x in reads[s].get(k, []) if x is not None])
                for k in watched
            }
            gemms.append(
                {
                    "shape": list(s),
                    "kernels": list(e["kernels"]),
                    "picks": dict(e.get("picks") or {}),
                    "readings": {str(k): v for k, v in reads[s].items()},
                    "median": {str(k): v for k, v in med.items()},
                    "winner": win,
                    "spread": spreads,
                    "rebenched": s in wide_ids,
                    "flagged": s in wide_ids and self.wide(e, reads[s]),
                    "orders": [list(o) for o in orders[s]],
                    **({"dropped": dropped[s]} if s in dropped else {}),
                }
            )

        def strip(r: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
            if r is None:
                return None
            skip = ("readings", "orders", "dropped")
            return {k: v for k, v in r.items() if k not in skip}

        res = {
            "generation": gen,
            "label": label,
            "part": i,
            "gpu": self.gpu,
            "settings": dict(st),
            "bench": strip(main),
            "spread_bench": strip(respread),
            "canary_flagged": bool(
                main["canary_flagged"] or (respread or {}).get("canary_flagged")
            ),
            "n_spread_flagged": sum(1 for g in gemms if g["flagged"]),
            "n_dropped": sum(1 for g in gemms if g.get("dropped")),
            "gemms": gemms,
            "wall_s": round(time.time() - t0, 3),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        _write(root / RESULT, res)
        return res


# ── work queue ───────────────────────────────────────────────────────────────


def jobs(d: Path) -> Iterator[Tuple[int, str, int, List[Dict[str, Any]]]]:
    """(generation, label, part, entries) of every planned part, oldest
    generation first, the exclusive group first."""
    for p in plans(d):
        labels = sorted(p["groups"], key=lambda k: (k != "exclusive", k))
        for label in labels:
            for i, entries in enumerate(p["groups"][label]):
                yield p["index"], label, i, entries


@dataclass
class Claim:
    fd: int

    def release(self) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)


def claim(root: Path) -> Optional[Claim]:
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / ".claim.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return Claim(fd)


def work(bench: Bench) -> int:
    """Bench every unclaimed incomplete part until none is left; returns the
    number benched. While the run holds the worker's GPU (`Bench.held`) it
    claims no part: it reads the hold again every `poll_s` while a part is
    left to bench, and returns once none is, or after `max_wait_s`. A part
    whose stage03 runs bench nothing (Unmeasured) is left unwritten, its
    claim released for another worker; the worker then waits so while the
    run holds its GPU, and else raises Unmeasured: its GPU cannot bench."""
    n = 0
    since: Optional[float] = None
    noted: Optional[str] = None
    while True:
        why = bench.held()
        if why is not None:
            left = pending(bench.d)
            if not left:
                return n
            now = time.monotonic()
            since = now if since is None else since
            if bench.max_wait_s > 0 and now - since >= bench.max_wait_s:
                bench.log(
                    f"the run held this worker's GPU for longer than "
                    f"bench.health.max_wait_s {bench.max_wait_s:g} s ({why}); it "
                    f"stops with {len(left)} GEMM bench(es) left to the others"
                )
                return n
            if why != noted:
                bench.log(
                    f"the run holds this worker's GPU ({why}): it takes no part "
                    f"while it does; {len(left)} GEMM bench(es) left"
                )
                noted = why
            bench.sleep(bench.poll_s)
            continue
        since = noted = None
        took = False
        failed: Optional[Unmeasured] = None
        for gen, label, i, entries in jobs(bench.d):
            root = part_path(bench.d, gen, label, i)
            if (root / RESULT).is_file():
                continue
            c = claim(root)
            if c is None:
                continue
            try:
                if (root / RESULT).is_file():
                    continue
                bench.log(f"gen {gen} {label} part {i}: {len(entries)} GEMMs")
                try:
                    bench.bench_part(gen, label, i, entries)
                except Unmeasured as e:
                    bench.log(f"gen {gen} {label} part {i} is left unwritten: {e}")
                    failed = e
                else:
                    n += 1
                    took = True
                    write_progress(bench.d)
            finally:
                c.release()
            break
        if failed is not None:
            if bench.held() is None:
                raise failed
            continue
        if not took:
            return n


# ── measurements and status ──────────────────────────────────────────────────


@dataclass
class Measurement:
    """The v2 measurement of one GEMM: its set's medians, the winner, the
    generation and whether a spread or the canary flagged it. `unplaced`
    holds the kernels of its medians that the audit set's kernel digests
    cannot place in the pool now (`AuditKernels.unplaced`); the winner is
    unknown then, so it scores no pick. `dropped` says why its latest
    generation dropped it (its line hung, again in a retry, or was
    quarantined): its medians are an earlier generation's, if any, and a
    pick they lack is left out of the scores, not waited for."""

    generation: int
    kernels: List[int]
    median: Dict[int, float]
    flagged: bool
    canary_flagged: bool
    unplaced: List[int] = field(default_factory=list)
    dropped: Optional[Dict[str, Any]] = None

    @property
    def winner_us(self) -> Optional[float]:
        return min(self.median.values()) if self.median else None


def measurements(
    d: Path,
    identity: Optional[ki.LibraryIdentity] = None,
    kernels: Optional[AuditKernels] = None,
) -> Dict[Key, Measurement]:
    """{(label, shape): Measurement} from each GEMM's latest complete
    generation; with `identity`, its kernels are named by the indices
    `identity`'s library gives them (`ki.remap_of` of the generation's
    record), kernels it lacks left out. A generation without a record is
    mapped so by the audit set's kernel digests as of its plan (`kernels`):
    the medians of kernels that left the pool are left out, and those of
    kernels it cannot place make the GEMM's `unplaced`. Else it is read as
    it is. A generation that dropped the GEMM (`dropped` in its result)
    keeps the earlier generation's measurement with `dropped` set, or gives
    one without medians."""
    when = planned_times(manifest(d))
    index_of: Dict[int, Callable[[int], Optional[int]]] = {}
    views: Dict[int, AuditKernels] = {}
    for p in plans(d):
        gen = int(p["index"])
        what = f"v2 generation {gen} in {d}"
        if kernels is None or ki.read_record(p) is not None:
            remap = ki.remap_of(p, identity, what)
            if not remap.same:
                index_of[gen] = remap.index
            continue
        view = kernels.at(when.get(gen))
        sids = plan_indices(p)
        if view.moves(sids):
            ki.report_once(
                f"{view.explain(what, sids, identity)}; its medians are read at "
                f"their kernels' current indices, those of kernels that left the "
                f"pool left out, and a GEMM with a median of a kernel it cannot "
                f"place is not scored"
            )
            views[gen] = view
        else:
            ki.report_once(
                f"{what} records no kernel identities; by the audit set's kernel "
                f"digests the solution indices it names still name the same "
                f"kernels",
                warn=False,
            )
    out: Dict[Key, Measurement] = {}
    for gen, label, i, _ in jobs(d):
        res = _read(part_path(d, gen, label, i) / RESULT)
        if not isinstance(res, dict):
            continue
        view = views.get(int(gen))
        index = view.index if view is not None else index_of.get(int(gen))
        for g in res["gemms"]:
            named = [int(k) for k in g["kernels"]]
            median = {int(k): float(v) for k, v in g["median"].items()}
            unplaced = view.unplaced(median) if view is not None else []
            if index is not None:
                named = [s for s in map(index, named) if s is not None]
                median = {
                    index(k): v for k, v in median.items() if index(k) is not None
                }
            key = (label, tuple(g["shape"]))
            if g.get("dropped"):
                why = dict(g["dropped"], generation=int(gen))
                prev = out.get(key)
                out[key] = (
                    replace(prev, dropped=why)
                    if prev is not None
                    else Measurement(
                        gen,
                        named,
                        {},
                        bool(g["flagged"]),
                        bool(res.get("canary_flagged")),
                        [],
                        why,
                    )
                )
                continue
            out[key] = Measurement(
                gen,
                named,
                median,
                bool(g["flagged"]),
                bool(res.get("canary_flagged")),
                unplaced,
            )
    return out


def check_bench_library(
    d: Path,
    m: Mapping[str, Any],
    audit_manifest: Optional[Mapping[str, Any]] = None,
    library: Optional[Mapping[str, Any]] = None,
) -> None:
    """V2Error when the bench's library (`bench.library` of the manifest,
    else `library`: its directory and files) gives a solution index of a
    generation with parts left to bench, or the canary's, another kernel
    than the plan recorded; with the audit set's manifest (`audit_manifest`)
    also than one without a record named, by the audit set's kernel digests
    as of its plan (`AuditKernels`). Without a library, V2Error while a
    generation has parts left to bench."""
    pending = {
        gen
        for gen, label, i, _ in jobs(d)
        if not (part_path(d, gen, label, i) / RESULT).is_file()
    }
    lib = (m.get("bench") or {}).get("library") or library
    if not lib:
        if pending:
            raise V2Error(
                f"{d / MANIFEST} records no bench library and none was given: "
                f"nothing can check that the bench lines of v2 generation(s) "
                f"{', '.join(str(g) for g in sorted(pending))} time the kernels "
                f"they were planned with"
            )
        return
    try:
        identity = ki.library_identity(lib["dir"], lib["stems"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise V2Error(f"cannot read the bench library's kernel identities: {e}")
    kernels = None
    if audit_manifest is not None:
        from .audit import pool_digests

        kernels = AuditKernels.of(
            audit_manifest, pool_digests(identity.kernels.values())
        )
    when = planned_times(m)
    canary = m.get("canary") or {}
    docs = [
        (
            "the v2 canary",
            canary,
            [int(canary.get("kernel", -1))],
            m.get("created"),
        )
    ]
    docs += [
        (f"v2 generation {p['index']}", p, plan_indices(p), when.get(p["index"]))
        for p in plans(d)
        if int(p["index"]) in pending
    ]
    for what, doc, sids, planned in docs:
        why = None
        try:
            rec = ki.read_record(doc)
            if rec is not None:
                ki.Remap.of(rec, identity, f"{what} in {d}").require_same()
        except (ki.BuildChangedError, ValueError) as e:
            why = str(e)
        else:
            view = kernels.at(planned) if kernels is not None else None
            if rec is None and view is not None and view.moves(sids):
                why = view.explain(f"{what} in {d}", sids, identity)
        if why is not None:
            raise V2Error(
                f"{why}; its bench lines would time other kernels in {lib['dir']}"
            )


def pending(d: Path) -> Dict[Key, int]:
    """{(label, shape): generation} of GEMMs in parts not benched yet."""
    out: Dict[Key, int] = {}
    for gen, label, i, entries in jobs(d):
        if not (part_path(d, gen, label, i) / RESULT).is_file():
            for e in entries:
                out[(label, tuple(e["shape"]))] = gen
    return out


def progress(d: Path) -> Dict[str, Any]:
    per: Dict[str, Dict[str, Any]] = {}
    walls = []
    flagged_parts = 0
    dropped = 0
    for gen, label, i, entries in jobs(d):
        u = per.setdefault(
            label,
            {"parts_done": 0, "parts_total": 0, "gemms_done": 0, "gemms_total": 0},
        )
        u["parts_total"] += 1
        u["gemms_total"] += len(entries)
        res = _read(part_path(d, gen, label, i) / RESULT)
        if isinstance(res, dict):
            u["parts_done"] += 1
            u["gemms_done"] += len(entries)
            walls.append((len(entries), float(res.get("wall_s") or 0)))
            flagged_parts += bool(res.get("canary_flagged"))
            dropped += sum(1 for g in res.get("gemms") or [] if g.get("dropped"))
    done = sum(u["gemms_done"] for u in per.values())
    total = sum(u["gemms_total"] for u in per.values())
    n = sum(a for a, _ in walls)
    return {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "gemms_done": done,
        "gemms_total": total,
        "bench_s_per_gemm": (sum(b for _, b in walls) / n) if n else None,
        "parts_canary_flagged": flagged_parts,
        "gemms_dropped": dropped,
        "groups": per,
    }


def write_progress(d: Path) -> Dict[str, Any]:
    p = progress(d)
    _write(Path(d) / PROGRESS, p)
    return p


def status_line(d: Path, regime: str = "v2") -> str:
    """One line on the measurement in `d` (`regime` names it when there is
    none)."""
    m = manifest(d)
    if m is None:
        return f"no {regime} measurement in {d}"
    p = progress(d)
    out = (
        f"{m.get('regime') or 'v2'}: {p['gemms_done']} of {p['gemms_total']} "
        f"planned GEMM benches done over {len(m['generations'])} generation(s)"
    )
    if p["parts_canary_flagged"]:
        out += f", {p['parts_canary_flagged']} part(s) flagged by the canary"
    if p["gemms_dropped"]:
        out += (
            f", {p['gemms_dropped']} GEMM bench(es) dropped (the line hung, again "
            f"in a retry)"
        )
    return out


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
