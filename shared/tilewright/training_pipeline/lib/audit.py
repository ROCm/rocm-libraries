# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The audit set: a fixed set of GEMMs per grid cell, benched exhaustively
once over every kernel the deployed model can rank (pool and bank), that no
round ever trains on. Every round measures its models on it against the
true fastest kernel, so the number tracks the goal (95% of the winner per
cell) instead of a planned list's best.

`run_orchestrator.py --mode audit --config <training config> --run-dir
<audit dir>` builds it with the training config's library, bank, bench
settings and contexts; a training config points at it with `audit.dir`.
Layout of an audit directory:

  audit_set.json        manifest: settings, source runs, shape counts per
                        context, the benched kernels per generation (their
                        kernel digests, and the library digest of the
                        build, lib/kernel_identity.py), status
  .audit_complete.json  written when every generation is enriched
  progress.json         parts done per context, throughput, expected finish
  run_layout.json       the directory names of its stages (lib/run_layout.py)
  stage01/              the shapes (stage01 --mode initial with `audit.seed`,
                        history = every shape of the source runs)
  stage02/              exhaustive plan: every pool and bank kernel per shape
                        (stage01b/ in an audit directory written before the
                        planner became stage02, lib/run_layout.py)
  bench/<label>/part_NNNN/   one resumable unit: shapes.yaml, stage03/,
                        stage03b/ and a stage marker; <label> is
                        `exclusive` or a context label (a group), whose
                        wrapper (hipblaslt-bench, context.json) sits in
                        bench/<label>
  bench/<label>/stage04/     the group's enriched rows, written once its
                        parts are all benched; generation 0 is these
                        (audits built before them have one stage04/ instead)
  increments/inc_NNN/   a later kernel-pool change: the new kernels only,
                        benched on every audit GEMM, with their own stage04;
                        increment.json lists them with their kernel ids

Contexts: every configured context (trained and held out) benches
`audit.shapes_per_context` of each cell's `audit.shapes_per_cell` shapes
through the stage03c wrapper; the exclusive context benches all of them.

Never trained on: rounds draw no audit shape (stage01 --exclude-shapes-yaml,
which also forbids re-use), and stage05 refuses any input inside an audit
directory (`audit_inputs`). An evaluation also drops audit shapes found in
the rounds the model trained on and reports how many.

A generation records the kernel digest (`kernel_digest`: name, engine config
and attributes) of every solution index it benched, so its rows follow their
kernels when a hipBLASLt rebuild renumbers the pool (`generation_maps`): a
kernel counts as benched under any index, a renumbering alone needs no
increment, and the rows of each generation are read at the index the pool
gives their kernel now.

An evaluation is `ok` when the audit is complete and has benched every kernel
of the current pool. While generation 0 is benched it is `partial`: the
groups whose parts are all benched and enriched are evaluated, every other
group is null with its own status (`group_status`), and `status_text` says
which (`partial: exclusive complete, 5 contexts benching`). Otherwise it is
null (status `missing` / `incomplete` / `error`, every number None). It
never falls back to another measurement, and a group with a missing part is
never scored. `compare` pairs two evaluations GEMM by GEMM on a group both
evaluated; `cell_gate` judges per cell whether a candidate regresses
against the bundle that ships, its cell rules family-wise (`decline_p`).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

MANIFEST = "audit_set.json"
COMPLETE_MARKER = ".audit_complete.json"
PROGRESS = "progress.json"
RESULT_FILE = "audit.json"
UNITS = "units.json"
STAGE_MARKER = ".stage_complete.json"
GROUP_SOURCE = "source.json"
EXCLUSIVE = "exclusive"
SCHEMA_VERSION = "audit-v1"
RESULT_SCHEMA = "audit-eval-v2"
DEFAULTS: Dict[str, Any] = {
    "dir": None,
    "shapes_per_cell": 16,
    "shapes_per_context": 4,
    "seed": 4242,
    "target": 0.95,
    "bootstrap": 2000,
    "part_shapes": 32,
}
# Settings an audit directory is bound to; a later invocation must agree.
BOUND_KEYS = ("shapes_per_cell", "shapes_per_context", "seed")
CLASSES = ("pool_limited", "identity_limited", "model_limited", "passing")
# The per-cell non-regression gate (`cell_gate`; the config's `gate`
# section). Its cell rules, on the groups `groups` names: a cell
# significantly worse beyond `cell_tolerance`, a cell confidently fallen
# below `cell_floor`, family-wise at `alpha` over their cells. Its tail
# rule, in every group: more new picks collapsing below `tail_floor` (to at
# most `tail_collapse` of the shipped pick) than the group lifts back above
# it, plus `tail_budget`.
CELL_GATE_DEFAULTS: Dict[str, Any] = {
    "cell_tolerance": 0.01,
    "cell_floor": 0.85,
    "alpha": 0.05,
    "tail_floor": 0.5,
    "tail_collapse": 0.5,
    "tail_budget": 0,
    "groups": "exclusive",
}
CELL_GATE_GROUPS = ("exclusive", "all")
# Findings a gate message names per kind before it counts the rest.
_SHOWN = 4
# In the cell tests a GEMM's change counts at most a factor of two: a larger
# one is the tail rule's, and left whole it would set the spread of the null
# distribution every other cell is judged against.
_CELL_CAP = math.log(2.0)
PROTOCOLS = {
    "v1": "v1 single reading",
    "v2": "v2 median of repeated joint benches",
}
ShapeKey = Tuple[int, int, int, int]


def settings(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """The config's `audit` section over `DEFAULTS`."""
    return {**DEFAULTS, **dict(cfg.get("audit") or {})}


def config_dir(cfg: Mapping[str, Any]) -> Optional[Path]:
    d = settings(cfg)["dir"]
    return Path(str(d)) if d else None


def shapes_yaml(audit_dir: Path) -> Path:
    return Path(audit_dir) / "stage01" / "shapes.yaml"


def context_fraction(st: Mapping[str, Any]) -> float:
    return float(st["shapes_per_context"]) / float(st["shapes_per_cell"])


def bench_config(cfg: Mapping[str, Any], devices: Optional[Sequence[int]] = None):
    """The config the audit's stages run with: the training config with an
    exhaustive planner (no model, no seed bootstrap), every context (trained
    and held out) benching `shapes_per_context` per cell and enriched with
    the exclusive rows, the near-winner re-reads on, nothing deployed, and no
    correctness section: the set benches every kernel of the pool, and the
    runs that share it leave out the kernels their own blocklists name when
    they evaluate on it (lib/correctness.py)."""
    from .contexts import contexts_from_config

    st = settings(cfg)
    out = json.loads(json.dumps(dict(cfg)))
    planner = dict(out.get("planner") or {})
    planner.update(enabled=True, mode="exhaustive", model_file=None)
    planner["ranker"] = "neighbors"
    out["planner"] = planner
    frac = context_fraction(st)
    ctx = dict(out.get("contexts") or {})
    entries = []
    for c in contexts_from_config(cfg):
        entries.append(
            {
                "cu_budget": c.cu_budget,
                "schedule": c.schedule,
                "cotenant": c.cotenant,
                "fraction": frac,
                "name": c.label,
            }
        )
    ctx.update(list=entries, holdout=[], replan=False)
    out["contexts"] = ctx
    bench = dict(out.get("bench") or {})
    bench["rebench"] = {**dict(bench.get("rebench") or {}), "enabled": True}
    out["bench"] = bench
    deploy = dict(out.get("deploy") or {})
    deploy.update(apply_patch=False, build_after_apply=False)
    out["deploy"] = deploy
    out["stage07"] = {"parity_sample_trained": 0, "bench_yamls": []}
    out["stage08"] = {"datasets": []}
    out.pop("correctness", None)
    if devices is not None:
        out["runtime"] = {**dict(out["runtime"]), "devices": [int(d) for d in devices]}
    return out


def context_labels(cfg: Mapping[str, Any]) -> Dict[Tuple[int, str], str]:
    """{(cu_budget, schedule): label} of the config's contexts, exclusive
    included."""
    from .contexts import contexts_from_config

    out = {(0, "default"): EXCLUSIVE}
    for c in contexts_from_config(cfg):
        out[(int(c.cu_budget), c.schedule)] = c.label
    return out


# ── manifest, status ─────────────────────────────────────────────────────────


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def manifest(audit_dir: Path) -> Optional[Dict[str, Any]]:
    m = read_json(Path(audit_dir) / MANIFEST)
    return m if isinstance(m, dict) else None


def is_complete(audit_dir: Path) -> bool:
    return (Path(audit_dir) / COMPLETE_MARKER).is_file() and manifest(
        audit_dir
    ) is not None


def kernel_digest(kernel: Mapping[str, Any]) -> str:
    """Identity of a pool kernel beyond its solution index: name, engine
    config and attributes."""
    from . import features as fs

    body = {
        "name": str(kernel.get("kernel_name") or ""),
        "config": fs.config_kwargs_from_row(kernel),
        "attributes": dict(kernel.get("attributes") or {}),
    }
    return hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def pool_digests(kernels: Iterable[Mapping[str, Any]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for k in kernels:
        sid = k.get("sol_idx_global")
        if sid is not None and int(sid) not in out:
            out[int(sid)] = kernel_digest(k)
    return out


def benched_digests(m: Mapping[str, Any]) -> Set[str]:
    """The kernel digests the audit's generations benched, under whatever
    solution index."""
    return {
        str(d)
        for gen in m.get("generations") or []
        for d in (gen.get("kernels") or {}).values()
    }


def unbenched(m: Mapping[str, Any], current: Mapping[int, str]) -> List[int]:
    """Solutions of `current` ({solution: digest}) whose kernel no
    generation benched."""
    have = benched_digests(m)
    return sorted(s for s, d in current.items() if d not in have)


def status(
    audit_dir: Optional[Path], current: Optional[Mapping[int, str]] = None
) -> Tuple[str, Optional[str]]:
    """("ok", None) when the audit at `audit_dir` is complete and covers the
    pool `current`; else ("missing" | "incomplete", why)."""
    if audit_dir is None:
        return "missing", "no audit directory is configured"
    d = Path(audit_dir)
    m = manifest(d)
    if m is None:
        return "missing", f"{d} holds no {MANIFEST}"
    if m.get("schema") != SCHEMA_VERSION:
        return "incomplete", f"{d / MANIFEST} is not an {SCHEMA_VERSION} manifest"
    if not (d / COMPLETE_MARKER).is_file():
        prog = read_json(d / PROGRESS) or {}
        done = prog.get("shapes_done")
        total = prog.get("shapes_total")
        where = f" ({done}/{total} GEMMs benched)" if done is not None else ""
        return "incomplete", f"the audit set is still being benched{where}"
    if current is not None:
        missing = unbenched(m, current)
        if missing:
            return "incomplete", (
                f"{len(missing)} kernel(s) of the current pool are not benched on "
                f"the audit set; run --mode audit on it to bench them"
            )
    return "ok", None


@dataclass
class Group:
    """One group (`exclusive` or a context label) of an audit's generation 0:
    `status` ok (benched and enriched, `stage04` its rows), benching or
    not_enriched."""

    label: str
    status: str
    reason: Optional[str]
    parts_done: int
    parts_total: int
    stage04: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "parts_done": self.parts_done,
            "parts_total": self.parts_total,
        }


@dataclass
class State:
    """What an evaluation can use of an audit directory now (`state`)."""

    status: str
    reason: Optional[str]
    groups: Dict[str, Group] = field(default_factory=dict)
    generations: List[Tuple[List[Path], "GenerationMap"]] = field(default_factory=list)

    @property
    def evaluable(self) -> List[str]:
        return [k for k, g in self.groups.items() if g.status == "ok"]

    def text(self) -> str:
        return status_text(self.status, self.reason, self.groups)


def units(audit_dir: Path) -> Optional[Dict[str, Any]]:
    u = read_json(Path(audit_dir) / "bench" / UNITS)
    return u if isinstance(u, dict) else None


def part_dir(audit_dir: Path, label: str, i: int) -> Path:
    return Path(audit_dir) / "bench" / label / f"part_{i:04d}"


def group_parts(audit_dir: Path, label: str, unit: Mapping[str, Any]) -> List[Path]:
    return [part_dir(audit_dir, label, i) for i in range(len(unit["parts"]))]


def group_stage04(audit_dir: Path, label: str) -> Path:
    return Path(audit_dir) / "bench" / label / "stage04"


def group_logs(audit_dir: Path, label: str, unit: Mapping[str, Any]) -> List[Path]:
    """Every bench log of the group's parts (stage03 and stage03b)."""
    out: List[Path] = []
    for p in group_parts(audit_dir, label, unit):
        for st in ("stage03", "stage03b"):
            out += sorted((p / st / "logs").glob("*.log"))
    return out


def logs_fingerprint(audit_dir: Path, label: str, unit: Mapping[str, Any]):
    """{log path relative to the audit: [size, mtime_ns]} of the group."""
    root = Path(audit_dir)
    out = {}
    for f in group_logs(root, label, unit):
        s = f.stat()
        out[str(f.relative_to(root))] = [s.st_size, s.st_mtime_ns]
    return out


def enriched_copy(
    enrich_root: Optional[Path], audit_dir: Path, label: str, unit: Mapping[str, Any]
) -> Optional[Path]:
    """`<enrich_root>/<label>/stage04` when it enriched exactly the group's
    current logs (its source.json), else None."""
    if enrich_root is None:
        return None
    d = Path(enrich_root) / label / "stage04"
    src = read_json(d / GROUP_SOURCE)
    if not (d / STAGE_MARKER).is_file() or not isinstance(src, dict):
        return None
    if Path(str(src.get("audit_dir"))).resolve() != Path(audit_dir).resolve():
        return None
    if src.get("logs") != logs_fingerprint(audit_dir, label, unit):
        return None
    return d


def state(
    audit_dir: Optional[Path],
    current: Optional[Mapping[int, str]] = None,
    enrich_root: Optional[Path] = None,
) -> State:
    """`status` with the groups: a complete audit is `ok` over its
    generations; while generation 0 is benched it is `partial` when a group
    is benched and enriched (in the audit, or by `enrich_from_logs` under
    `enrich_root`), `incomplete` when none is or when the plan misses a
    kernel of `current`; a pending kernel increment is `incomplete`."""
    st, why = status(audit_dir, current)
    if st == "ok":
        assert audit_dir is not None
        m = manifest(audit_dir) or {}
        u = units(audit_dir) or {}
        groups = {
            k: Group(k, "ok", None, len(v["parts"]), len(v["parts"]))
            for k, v in u.items()
        }
        try:
            gens = generation_dirs(audit_dir, m, current)
        except ValueError as e:
            return State("incomplete", str(e))
        return State("ok", None, groups, gens)
    if st == "missing" or audit_dir is None:
        return State(st, why)
    d = Path(audit_dir)
    m = manifest(d) or {}
    if m.get("schema") != SCHEMA_VERSION:
        return State(st, why)
    if m.get("generations"):
        return State("incomplete", why)
    planned = dict(m.get("planned_kernels") or {})
    u = units(d)
    if not planned or u is None:
        return State("incomplete", why)
    if current is not None:
        missing = unbenched({"generations": [{"kernels": planned}]}, current)
        if missing:
            return State(
                "incomplete",
                f"{len(missing)} kernel(s) of the current pool are not in the audit "
                f"set's plan",
            )
    gmap = generation_maps({"generations": [{"kernels": planned}]}, current)[0]
    groups: Dict[str, Group] = {}
    for label, unit in u.items():
        parts = group_parts(d, label, unit)
        done = sum(1 for p in parts if (p / STAGE_MARKER).is_file())
        g = Group(label, "benching", None, done, len(parts))
        if done < len(parts):
            g.reason = f"{done} of {len(parts)} parts benched"
        else:
            own = group_stage04(d, label)
            g.stage04 = (
                own
                if (own / STAGE_MARKER).is_file()
                else enriched_copy(enrich_root, d, label, unit)
            )
            if g.stage04 is None:
                g.status = "not_enriched"
                g.reason = (
                    "every part is benched; the group's rows are not enriched yet"
                )
            else:
                g.status, g.reason = "ok", None
        groups[label] = g
    ok = [g.stage04 for g in groups.values() if g.status == "ok"]
    if not ok:
        return State("incomplete", why, groups)
    return State("partial", why, groups, [([p for p in ok if p], gmap)])


def status_text(
    status_: str, reason: Optional[str], groups: Mapping[str, Group]
) -> str:
    """One line on an evaluation's coverage: `complete`,
    `partial: exclusive complete, 5 contexts benching`, or the null status
    with its reason."""
    if status_ == "ok":
        return "complete"
    if status_ != "partial":
        return f"{status_}: {reason}" if reason else status_
    done = [k for k, g in groups.items() if g.status == "ok"]
    rest: Dict[str, List[str]] = {}
    for k, g in groups.items():
        if g.status != "ok":
            rest.setdefault(g.status.replace("_", " "), []).append(k)
    words = [", ".join(done) + " complete"]
    for what, labels in rest.items():
        ctx = [k for k in labels if k != EXCLUSIVE]
        named = ([EXCLUSIVE] if EXCLUSIVE in labels else []) + (
            [f"{len(ctx)} context{'s' if len(ctx) != 1 else ''}"] if ctx else []
        )
        words.append(f"{' and '.join(named)} {what}")
    return "partial: " + ", ".join(words)


# ── never trained on ─────────────────────────────────────────────────────────


def audit_root(path: Path) -> Optional[Path]:
    """The audit directory `path` lies in (itself or an ancestor holding an
    audit manifest), else None."""
    p = Path(path).resolve()
    for d in (p, *p.parents):
        if (d / MANIFEST).is_file():
            return d
    return None


def audit_inputs(paths: Iterable[Any]) -> List[str]:
    """Errors for every path inside an audit directory."""
    out = []
    for p in paths:
        root = audit_root(Path(str(p)))
        if root is not None:
            out.append(f"{p} is inside the audit set {root}, which is never trained on")
    return out


def existing_paths(values: Iterable[Any]) -> List[Path]:
    """Every existing file or directory among `values` (paths, strings and
    nested lists of them, as argparse leaves them)."""
    out: List[Path] = []
    for v in values:
        if isinstance(v, (list, tuple)):
            out += existing_paths(v)
        elif isinstance(v, (str, Path)) and str(v) and Path(str(v)).exists():
            out.append(Path(str(v)))
    return out


def read_shapes(path: Path) -> Set[ShapeKey]:
    from .bench_yaml import parse_shapes_yaml

    p = Path(path)
    return set(parse_shapes_yaml(p)) if p.is_file() else set()


_ROUND = re.compile(r"round_(\d+)")


def _round_index(p: Path) -> Optional[int]:
    m = _ROUND.fullmatch(p.name)
    return int(m.group(1)) if m else None


def rounds_until(round_dir: Path, inclusive: bool) -> List[Path]:
    """The rounds of `round_dir`'s run before it (and itself with
    `inclusive`)."""
    rd = Path(round_dir).resolve()
    idx = _round_index(rd)
    if idx is None:
        return []
    out = []
    for p in sorted(rd.parent.glob("round_*")):
        j = _round_index(p)
        if j is not None and (j < idx or (inclusive and j == idx)):
            out.append(p)
    return sorted(out, key=lambda p: _round_index(p) or 0)


def trained_shapes(round_dirs: Iterable[Path]) -> Set[ShapeKey]:
    """Every shape the rounds drew (stage01/shapes.yaml holds the re-used
    and the seed shapes too, and every context benches a subset)."""
    out: Set[ShapeKey] = set()
    for rd in round_dirs:
        out |= read_shapes(Path(rd) / "stage01" / "shapes.yaml")
    return out


# ── loading ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenerationMap:
    """How the solution indices of one generation's rows map to the pool's
    now: `mapping` holds the indices it benched (None: the row is left
    out); with `keep_others` (a generation that recorded no kernels) any
    other index keeps itself, else its rows are left out."""

    mapping: Mapping[int, Optional[int]]
    keep_others: bool = False

    def index(self, sid: int) -> Optional[int]:
        sid = int(sid)
        if sid < 0:
            return sid
        if sid in self.mapping:
            return self.mapping[sid]
        return sid if self.keep_others else None


def generation_maps(
    m: Mapping[str, Any], current: Optional[Mapping[int, str]] = None
) -> List[GenerationMap]:
    """Per generation of the manifest `m`, oldest first, the index each
    solution it benched has in `current` ({solution: kernel digest}, the
    pool now): the index of its kernel's digest, None when the kernel left
    the pool, several solutions hold it, or a later generation benched it
    again. Without `current` every index keeps itself unless a later
    generation benched that index again; ValueError then when the
    generations were benched with different hipBLASLt builds (their
    `library_digest`s differ, or they name one kernel under several
    indices). A generation that recorded no kernels keeps its indices
    except those a later generation benched."""
    gens = [
        {int(s): str(d) for s, d in (g.get("kernels") or {}).items()}
        for g in m.get("generations") or []
    ]
    latest: Dict[str, int] = {}
    held: Dict[str, Set[int]] = {}
    for gi, ks in enumerate(gens):
        for sid, d in ks.items():
            latest[d] = gi
            held.setdefault(d, set()).add(sid)
    builds = {
        g.get("library_digest")
        for g in m.get("generations") or []
        if g.get("library_digest")
    }
    if current is None and (len(builds) > 1 or any(len(v) > 1 for v in held.values())):
        raise ValueError(
            "the audit's generations were benched with different hipBLASLt "
            "builds (their library digests differ, or they name one kernel "
            "under several solution indices): only the current pool can map "
            "their rows"
        )
    by_digest: Dict[str, Optional[int]] = {}
    for sid, d in sorted((current or {}).items()):
        by_digest[d] = None if d in by_digest else int(sid)
    out = []
    for gi, ks in enumerate(gens):
        later = {s for k in gens[gi + 1 :] for s in k}
        if not ks or current is None:
            mapping = {s: (None if s in later else s) for s in set(ks) | later}
            out.append(GenerationMap(mapping, keep_others=not ks))
            continue
        mapping = {}
        for sid, d in ks.items():
            if latest[d] != gi:
                mapping[sid] = None
            elif current.get(sid) == d:
                mapping[sid] = sid
            else:
                mapping[sid] = by_digest.get(d)
        out.append(GenerationMap(mapping))
    return out


def generation_dirs(
    audit_dir: Path, m: Mapping[str, Any], current: Optional[Mapping[int, str]] = None
) -> List[Tuple[List[Path], GenerationMap]]:
    """(enriched dirs, `generation_maps` entry) per generation, oldest
    first; a generation's `stage04` is one directory or one per group."""
    out = []
    for gen, gmap in zip(m.get("generations") or [], generation_maps(m, current)):
        dirs = gen["stage04"]
        out.append(
            (
                [
                    Path(audit_dir) / str(d)
                    for d in ([dirs] if isinstance(dirs, str) else dirs)
                ],
                gmap,
            )
        )
    return out


def load_gemms(
    audit_dir: Path,
    quiet: bool = True,
    generations: Optional[Sequence[Tuple[Sequence[Path], GenerationMap]]] = None,
    current: Optional[Mapping[int, str]] = None,
) -> List[Dict[str, Any]]:
    """The audit's GEMMs (stage05 GEMM dicts) over `generations` (default
    the manifest's, mapped to the pool `current`, see `generation_maps`),
    each row at the solution index its kernel has now: a kernel benched
    again by a later generation keeps only that one's rows."""
    from . import evaluate as ev
    from .stage05_split import stage05_module

    gens = (
        generation_dirs(audit_dir, manifest(audit_dir) or {}, current)
        if generations is None
        else list(generations)
    )
    load = stage05_module().load_enriched_chunks
    merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for dirs, gmap in gens:
        for d in dirs:
            if not any(Path(d).glob("chunk_*.csv")):
                continue
            cells, _ = load(str(d), quiet)
            for gs in cells.values():
                for g in gs:
                    keep = []
                    for c in g["candidates"]:
                        sid = gmap.index(c["sol_idx_global"])
                        if sid is not None:
                            keep.append(dict(c, sol_idx_global=sid))
                    key = ev.gemm_key(g)
                    if key in merged:
                        merged[key]["candidates"].extend(keep)
                    else:
                        merged[key] = dict(g, candidates=keep)
    out = []
    for g in merged.values():
        by_sid: Dict[int, Dict[str, Any]] = {}
        for c in g["candidates"]:
            prev = by_sid.get(c["sol_idx_global"])
            if prev is None or (prev["is_skip"], prev["us"]) > (c["is_skip"], c["us"]):
                by_sid[c["sol_idx_global"]] = c
        g["candidates"] = list(by_sid.values())
        tested = [c["us"] for c in g["candidates"] if not c["is_skip"]]
        if not g["candidates"]:
            continue
        g["winner_us"] = (
            min(tested) if tested else min(c["us"] for c in g["candidates"])
        )
        out.append(g)
    return out


# ── statistics ───────────────────────────────────────────────────────────────


@dataclass
class Row:
    """One audit GEMM under one model."""

    shape: ShapeKey
    cell: str
    context: str
    sel_eff: Optional[float]
    ceiling_rules: Optional[float]
    ceiling_table_rules: Optional[float]
    ceiling_table: Optional[float]
    pick: Optional[int] = None
    miss: Optional[str] = None
    ceiling_served: Optional[float] = None


def _geo(logs: np.ndarray) -> Optional[float]:
    return float(math.exp(logs.mean())) if logs.size else None


def _boot_means(logs: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, logs.size, (reps, logs.size))
    return logs[idx].mean(axis=1)


def _ci(stats: Optional[np.ndarray]) -> Optional[List[float]]:
    if stats is None or stats.size == 0:
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(math.exp(lo)), float(math.exp(hi))]


def _logs(values: Iterable[Optional[float]]) -> np.ndarray:
    return np.asarray(
        [math.log(v) for v in values if v is not None and v > 0 and math.isfinite(v)],
        dtype=np.float64,
    )


def classify(
    sel_eff: Optional[float],
    sel_ci: Optional[Sequence[float]],
    ceiling: Optional[float],
    ceiling_ci: Optional[Sequence[float]],
    target: float,
    served: Optional[float] = None,
    served_ci: Optional[Sequence[float]] = None,
) -> Tuple[Optional[str], bool]:
    """(class, confident): passing when the selection efficiency reaches
    `target`, else pool_limited when the ceiling under the serving rules is
    below it too, else identity_limited when the ceiling of the keys as the
    runtime serves them (`served`, when given) is, else model_limited;
    confident when the 95% intervals keep the decisive number on its side of
    the target."""
    if sel_eff is None or ceiling is None:
        return None, False
    if sel_eff >= target:
        return "passing", bool(sel_ci and sel_ci[0] >= target)
    if ceiling < target:
        return "pool_limited", bool(ceiling_ci and ceiling_ci[1] < target)
    if served is not None and served < target:
        return "identity_limited", bool(served_ci and served_ci[1] < target)
    lower = served_ci if served is not None else ceiling_ci
    return "model_limited", bool(
        lower and sel_ci and lower[0] >= target and sel_ci[1] < target
    )


def summarize_rows(
    rows: Sequence[Row], target: float, bootstrap: int, seed: int = 0
) -> Optional[Dict[str, Any]]:
    """Per-cell geomeans against the true best with bootstrap intervals over
    each cell's GEMMs, the equal-weight mean of the cell geomeans (in log
    space, its interval resampled within every cell), the GEMM-weighted
    geomean, the minimum cell and the classification. None without an
    evaluated GEMM."""
    rng = np.random.default_rng(seed)
    reps = max(0, int(bootstrap))
    by_cell: Dict[str, List[Row]] = {}
    for r in rows:
        by_cell.setdefault(r.cell, []).append(r)
    cells: Dict[str, Dict[str, Any]] = {}
    boot_cells: List[np.ndarray] = []
    weights: List[int] = []
    for cell in sorted(by_cell):
        rs = by_cell[cell]
        sel = _logs(r.sel_eff for r in rs)
        if not sel.size:
            continue
        rule = _logs(r.ceiling_rules for r in rs)
        served = _logs(r.ceiling_served for r in rs)
        boot = _boot_means(sel, reps, rng) if reps and sel.size > 1 else None
        rule_boot = _boot_means(rule, reps, rng) if reps and rule.size > 1 else None
        served_boot = (
            _boot_means(served, reps, rng) if reps and served.size > 1 else None
        )
        entry = {
            "n": int(sel.size),
            "n_not_evaluated": len(rs) - int(sel.size),
            "n_miss": sum(1 for r in rs if r.miss is not None),
            "sel_eff": _geo(sel),
            "sel_eff_ci": _ci(boot),
            "ceiling_rules": _geo(rule),
            "ceiling_rules_ci": _ci(rule_boot),
            "ceiling_served": _geo(served),
            "ceiling_served_ci": _ci(served_boot),
            "ceiling_table_rules": _geo(_logs(r.ceiling_table_rules for r in rs)),
            "ceiling_table": _geo(_logs(r.ceiling_table for r in rs)),
        }
        entry["class"], entry["confident"] = classify(
            entry["sel_eff"],
            entry["sel_eff_ci"],
            entry["ceiling_rules"],
            entry["ceiling_rules_ci"],
            target,
            entry["ceiling_served"],
            entry["ceiling_served_ci"],
        )
        cells[cell] = entry
        boot_cells.append(boot if boot is not None else np.full(reps, sel.mean()))
        weights.append(int(sel.size))
    if not cells:
        return None
    means = np.asarray([math.log(c["sel_eff"]) for c in cells.values()])
    eq_boot = np.vstack(boot_cells).mean(axis=0) if reps else None
    w = np.asarray(weights, dtype=np.float64)
    pooled_boot = (
        (w[:, None] * np.vstack(boot_cells)).sum(0) / w.sum() if reps else None
    )
    worst = min(cells, key=lambda c: cells[c]["sel_eff"])
    counts = {k: sum(1 for c in cells.values() if c["class"] == k) for k in CLASSES}
    confident = {
        k: sum(1 for c in cells.values() if c["class"] == k and c["confident"])
        for k in CLASSES
    }
    ceil = [c["ceiling_rules"] for c in cells.values() if c["ceiling_rules"]]
    served_ceil = [c["ceiling_served"] for c in cells.values() if c["ceiling_served"]]
    return {
        "n_gemms": int(w.sum()),
        "n_not_evaluated": sum(c["n_not_evaluated"] for c in cells.values()),
        "n_cells": len(cells),
        "cell_mean": float(math.exp(means.mean())),
        "cell_mean_ci": _ci(eq_boot),
        "gemm_geomean": float(math.exp((w * means).sum() / w.sum())),
        "gemm_geomean_ci": _ci(pooled_boot),
        "min_cell": {"cell": worst, **cells[worst]},
        "n_cells_below_target": sum(1 for c in cells.values() if c["sel_eff"] < target),
        "n_cells_confidently_below": sum(
            1
            for c in cells.values()
            if c["sel_eff_ci"] is not None and c["sel_eff_ci"][1] < target
        ),
        "ceiling_rules_cell_mean": (
            float(math.exp(np.mean([math.log(v) for v in ceil]))) if ceil else None
        ),
        "ceiling_served_cell_mean": (
            float(math.exp(np.mean([math.log(v) for v in served_ceil])))
            if served_ceil
            else None
        ),
        "classes": counts,
        "classes_confident": confident,
        "per_cell": cells,
    }


# ── evaluation ───────────────────────────────────────────────────────────────


@dataclass
class Spec:
    """How a stage evaluates on the audit set."""

    audit_dir: Optional[Path]
    library_dir: Optional[Path]
    library_stem: str
    hardware_device: int = 0
    pick_measurement: str = "warmup"
    target: float = DEFAULTS["target"]
    bootstrap: int = DEFAULTS["bootstrap"]
    trained_rounds: List[Path] = field(default_factory=list)
    enrich_root: Optional[Path] = None
    protocol: str = "both"
    v2_dir: Optional[Path] = None


def null_record(status_: str, reason: Optional[str], **extra: Any) -> Dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "protocol": "v1",
        "protocol_text": PROTOCOLS["v1"],
        "status": status_,
        "reason": reason,
        "status_text": status_text(status_, reason, {}),
        "headline": None,
        "groups": None,
        **extra,
    }


def _label_of(labels: Mapping[Tuple[int, str], str], c: Any) -> str:
    return labels.get((int(c.cu_budget), c.schedule), f"cu{c.cu_budget}_{c.schedule}")


_LOADED: Dict[Any, List[Dict[str, Any]]] = {}


def cached_gemms(
    audit_dir: Path, generations: Sequence[Tuple[Sequence[Path], GenerationMap]]
) -> List[Dict[str, Any]]:
    """`load_gemms` of `generations`, kept while their enriched directories
    keep their completion markers."""

    def stamp(d: Path) -> Any:
        try:
            return (Path(d) / STAGE_MARKER).stat().st_mtime_ns
        except OSError:
            return None

    key = tuple(
        (
            tuple((str(d), stamp(d)) for d in dirs),
            tuple(sorted(gmap.mapping.items())),
            gmap.keep_others,
        )
        for dirs, gmap in generations
    )
    if key not in _LOADED:
        _LOADED.clear()
        _LOADED[key] = load_gemms(audit_dir, generations=generations)
    return _LOADED[key]


def _ceiling(winner: float, cands: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """winner over the fastest of `cands`, each costed as a pick would be
    (its timed row, else its warm-up), so a subset never has the higher
    ceiling; capped at 1 like the selection efficiency."""
    costs = [c["us"] for c in cands if c["us"] > 0]
    if not costs or not winner > 0:
        return None
    return min(1.0, winner / min(costs))


def candidate_keys(
    pool: Sequence[Mapping[str, Any]], header: Mapping[str, Any]
) -> Tuple[Dict[int, Tuple[Any, ...]], Dict[Tuple[Any, ...], List[int]]]:
    """({solution: key}, {key: its solutions in pool order}) of `pool` as the
    engine groups it for the model whose MLREC header is `header`: an
    MLREC_v3 model by view key, and launch key when it serves launch keys;
    an MLREC_v2 model by its view key, every config field."""
    from . import evaluate as ev
    from . import features as fs
    from . import hipblaslt_launch

    v3 = int(header["version"]) == 3
    launch = None
    if v3 and (header.get("serving") or {}).get("launch_keys"):
        launch = hipblaslt_launch.HipblasltLaunch(str(header["arch"]))
    key_of: Dict[int, Tuple[Any, ...]] = {}
    members: Dict[Tuple[Any, ...], List[int]] = {}
    for k in pool:
        cfg = fs.config_kwargs_from_row(k)
        if v3:
            key = fs.v3_candidate_key(
                cfg, k.get("attributes"), header["attribute_spec"], launch
            )
        else:
            key = tuple(int(cfg[f]) for f in ev.CONFIG_FIELDS)
        sid = int(k["sol_idx_global"])
        key_of[sid] = key
        members.setdefault(key, []).append(sid)
    return key_of, members


def _served_ceiling(
    winner: float,
    cands: Sequence[Mapping[str, Any]],
    allowed: Set[int],
    key_of: Mapping[int, Tuple[Any, ...]],
    members: Mapping[Tuple[Any, ...], Sequence[int]],
) -> Optional[float]:
    """winner over the fastest key as the runtime serves it: each key at the
    cost of its first pool kernel with a row (costed as a pick is), keys
    whose kernels the serving rules reject (`allowed`: the solutions they
    admit) left out; capped at 1 like the selection efficiency."""
    by_sid = {int(c["sol_idx_global"]): c for c in cands}
    served = []
    for key in {key_of[s] for s in by_sid if s in key_of}:
        first = next(by_sid[s] for s in members[key] if s in by_sid)
        if int(first["sol_idx_global"]) in allowed:
            served.append(first)
    return _ceiling(winner, served)


def _capped(sel_eff: Optional[float]) -> Optional[float]:
    return None if sel_eff is None else min(1.0, sel_eff)


@dataclass
class _Prepared:
    """The GEMMs an evaluation scores and what it knows of the audit."""

    now: State
    base: Dict[str, Any]
    pool: Any
    library: Any
    labels: Dict[Tuple[int, str], str]
    gemms: List[Dict[str, Any]]
    n_dropped: int
    leaked: List[ShapeKey]
    identity: Any = None


def _shape_of(g: Mapping[str, Any]) -> ShapeKey:
    return (int(g["m"]), int(g["n"]), int(g["k"]), int(g["batch_count"]))


def _prepare(
    data: bytes,
    spec: Spec,
    cfg: Mapping[str, Any],
    gemms: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[Dict[str, Any]], Optional[_Prepared]]:
    """(null record, None) when the audit cannot evaluate, else (None, the
    audit GEMMs of the evaluable groups in the pool, trained shapes left
    out)."""
    from . import evaluate as ev
    from .bank_pool import config_identity, config_pool

    base: Dict[str, Any] = {
        "audit_dir": str(spec.audit_dir) if spec.audit_dir else None,
        "model_sha256": hashlib.sha256(data).hexdigest(),
    }
    if spec.library_dir is None:
        return (
            null_record("error", "the audit needs the library directory", **base),
            None,
        )
    pool = config_pool(cfg, spec.library_dir, spec.library_stem, spec.hardware_device)
    now = state(spec.audit_dir, pool_digests(pool.kernels), spec.enrich_root)
    base["group_status"] = {k: g.to_dict() for k, g in now.groups.items()}
    if now.status not in ("ok", "partial"):
        return null_record(now.status, now.reason, **base), None
    assert spec.audit_dir is not None
    labels = context_labels(cfg)
    if gemms is None:
        gemms = cached_gemms(spec.audit_dir, now.generations)
    gemms = [g for g in gemms if _label_of(labels, ev.context_of(g)) in now.evaluable]
    library = ev.pool_from_kernels(pool.kernels)
    gemms, _, n_dropped = ev.restrict_to_pool(gemms, library)
    seen = trained_shapes(spec.trained_rounds)
    leaked = sorted({_shape_of(g) for g in gemms if _shape_of(g) in seen})
    gemms = [g for g in gemms if _shape_of(g) not in seen]
    if not gemms:
        return null_record("error", "no audit GEMM left to evaluate", **base), None
    identity = config_identity(
        cfg, spec.library_dir, spec.library_stem, spec.hardware_device
    )
    base["library_digest"] = identity.digest
    return None, _Prepared(
        now, base, pool, library, labels, gemms, n_dropped, leaked, identity
    )


class _Scorer:
    """Rows of one model: the selection efficiency it is given and the
    ceilings of the GEMM's rows under the model's serving rules."""

    def __init__(self, data: bytes, prep: _Prepared, hardware: Any) -> None:
        from . import mlrec
        from .feasibility import (
            non_temporal_availability,
            non_temporal_families,
            soft_rules,
        )

        self.header = mlrec.read_header(data)
        self.rules = soft_rules(
            self.header.get("serving") or {}, int(self.header["version"])
        )
        self.nt = non_temporal_availability(prep.pool.kernels)
        self.families = (
            non_temporal_families(prep.pool.kernels)
            if self.rules.get("non_temporal_family_rule")
            else None
        )
        self.table_ids = {int(k["sol_idx_global"]) for k in prep.pool.table}
        self.key_of, self.members = candidate_keys(prep.library, self.header)
        self.hardware = hardware
        self.labels = prep.labels

    def label(self, g: Mapping[str, Any]) -> str:
        from . import evaluate as ev

        return _label_of(self.labels, ev.context_of(g))

    def row(self, g: Mapping[str, Any], r: Any, sel_eff: Optional[float]) -> Row:
        from .feasibility import passes_gates
        from .features import config_kwargs_from_row, problem_kwargs_from_row
        from .grid import cell_key

        problem = problem_kwargs_from_row(g)
        ok = [
            c
            for c in g["candidates"]
            if passes_gates(
                problem,
                config_kwargs_from_row(c),
                lds_bytes=int(self.hardware.lds_bytes),
                nt_a_available=self.nt[0],
                nt_b_available=self.nt[1],
                nt_families=self.families,
                **self.rules,
            )
        ]
        w = float(g["winner_us"])
        allowed = {int(x["sol_idx_global"]) for x in ok}
        table = self.table_ids
        return Row(
            shape=_shape_of(g),
            cell=cell_key(*_shape_of(g)),
            context=self.label(g),
            sel_eff=sel_eff,
            ceiling_served=_served_ceiling(
                w, g["candidates"], allowed, self.key_of, self.members
            ),
            ceiling_rules=_ceiling(w, ok),
            ceiling_table_rules=_ceiling(
                w, [c for c in ok if c["sol_idx_global"] in table]
            ),
            ceiling_table=_ceiling(
                w, [c for c in g["candidates"] if c["sol_idx_global"] in table]
            ),
            pick=r.pick_sol_idx,
            miss=r.miss,
        )


def _summaries(
    rows: Sequence[Row], labels: Iterable[str], spec: Spec
) -> Dict[str, Optional[Dict[str, Any]]]:
    groups: Dict[str, Optional[Dict[str, Any]]] = {k: None for k in labels}
    for label in [EXCLUSIVE] + sorted({r.context for r in rows} - {EXCLUSIVE}):
        sel = [r for r in rows if r.context == label]
        if sel:
            groups[label] = summarize_rows(sel, spec.target, spec.bootstrap)
            if groups[label] is not None:
                groups[label]["per_gemm"] = [
                    [*r.shape, r.sel_eff] for r in sorted(sel, key=lambda r: r.shape)
                ]
    return groups


def _run_model(data: bytes, prep: _Prepared, spec: Spec, hardware: Any, tw: Any):
    from . import evaluate as ev

    out = ev.evaluate_model(
        data,
        prep.gemms,
        hardware=hardware,
        pools=ev.pools_for(prep.gemms, prep.library),
        tw=tw,
        pick_measurement=spec.pick_measurement,
    )
    ev.require_consistent_routing(out, "audit")
    return out


def evaluable_gemms(
    spec: Spec, cfg: Mapping[str, Any]
) -> Dict[Tuple[str, ShapeKey], Dict[str, Any]]:
    """{(group, shape): exhaustive GEMM} of every evaluable audit GEMM in
    the pool; raises when the audit cannot evaluate."""
    import dataclasses

    from . import evaluate as ev

    null, prep = _prepare(b"", dataclasses.replace(spec, trained_rounds=[]), cfg, None)
    if prep is None:
        assert null is not None
        raise RuntimeError(f"the audit cannot evaluate: {null['status_text']}")
    return {
        (_label_of(prep.labels, ev.context_of(g)), _shape_of(g)): g for g in prep.gemms
    }


def picks(
    data: bytes,
    spec: Spec,
    cfg: Mapping[str, Any],
    *,
    hardware: Any,
    tw: Any = None,
    gemms: Optional[List[Dict[str, Any]]] = None,
) -> Dict[Tuple[str, ShapeKey], int]:
    """{(group, shape): the model's top-1 pick} on every evaluable audit
    GEMM (none left out as trained on); raises when the audit cannot
    evaluate."""
    import dataclasses

    null, prep = _prepare(
        data, dataclasses.replace(spec, trained_rounds=[]), cfg, gemms
    )
    if prep is None:
        assert null is not None
        raise RuntimeError(f"the audit cannot evaluate: {null['status_text']}")
    from . import evaluate as ev

    out = _run_model(data, prep, spec, hardware, tw)
    return {
        (_label_of(prep.labels, ev.context_of(g)), _shape_of(g)): int(r.pick_sol_idx)
        for g, r in zip(prep.gemms, out.results)
        if r.pick_sol_idx is not None
    }


def evaluate(
    data: bytes,
    spec: Spec,
    cfg: Mapping[str, Any],
    *,
    hardware: Any,
    tw: Any = None,
    gemms: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """The model file `data` on the audit set (see the module docstring),
    under protocol v1 and, with `spec.protocol` "both", protocol v2 in the
    record's `v2` (lib/audit_v2.py); raises on evaluation errors, returns a
    null record when the audit is missing or incomplete."""
    null, prep = _prepare(data, spec, cfg, gemms)
    if prep is None:
        assert null is not None
        return null
    now, base = prep.now, prep.base
    m = manifest(spec.audit_dir) or {}
    out = _run_model(data, prep, spec, hardware, tw)
    scorer = _Scorer(data, prep, hardware)
    rows = [
        scorer.row(g, r, _capped(r.sel_eff)) for g, r in zip(prep.gemms, out.results)
    ]
    groups = _summaries(rows, now.groups, spec)
    for label, g in now.groups.items():
        if g.status == "ok" and groups.get(label) is None:
            g.status, g.reason = "error", "no GEMM of the group left to evaluate"
            base["group_status"][label] = g.to_dict()
    head = groups.get(EXCLUSIVE)
    rec = {
        "schema": RESULT_SCHEMA,
        "protocol": "v1",
        "protocol_text": PROTOCOLS["v1"],
        "status": now.status,
        "reason": now.reason,
        "status_text": now.text(),
        **base,
        "weight_dtype": out.weight_dtype,
        "pick_measurement": spec.pick_measurement,
        "serving_rules": scorer.rules,
        "target": spec.target,
        "bootstrap": spec.bootstrap,
        "audit_generations": len(m.get("generations") or []),
        "n_gemms": len(rows),
        "n_dropped_outside_pool": prep.n_dropped,
        "n_excluded_trained": len(prep.leaked),
        "excluded_trained_shapes": [list(s) for s in prep.leaked[:50]],
        "headline": headline_of(head) if head else None,
        "groups": groups,
    }
    if spec.protocol == "both":
        rec["v2"] = _evaluate_v2(spec, prep, out, scorer)
    return rec


def pool_medians(mm: Any, in_pool: Set[int]) -> Any:
    """A v2 measurement (`lib.audit_v2.Measurement`) without the medians of
    kernels outside the pool `in_pool` (solution indices), such as kernels
    the config's blocklist leaves out of it or those of a library file the
    run does not read (a bank the audit set measured for a run without
    one), so that its winner is the fastest of the pool, as in protocol
    v1."""
    if mm is None or in_pool.issuperset(mm.median):
        return mm
    return replace(mm, median={k: v for k, v in mm.median.items() if k in in_pool})


def _v2_null(status_: str, reason: str, **extra: Any) -> Dict[str, Any]:
    return {
        "protocol": "v2",
        "protocol_text": PROTOCOLS["v2"],
        "status": status_,
        "reason": reason,
        "status_text": status_text(status_, reason, {}),
        "headline": None,
        "groups": None,
        **extra,
    }


def _evaluate_v2(
    spec: Spec, prep: _Prepared, out: Any, scorer: _Scorer
) -> Dict[str, Any]:
    """The protocol v2 record of the model whose v1 evaluation is `out`:
    every GEMM scored on its v2 measurement (lib/audit_v2.py), its winner the
    fastest median of a kernel of the pool (so never a kernel the config's
    blocklist leaves out of it); a group with a GEMM whose measurement lacks
    the model's pick, or holds a median of a kernel the audit set's digests
    cannot place (`unplaced`), is null with the counts."""
    from . import audit_v2 as v2

    assert spec.audit_dir is not None
    d = Path(spec.v2_dir) if spec.v2_dir else v2.v2_dir(spec.audit_dir)
    m = v2.manifest(d)
    if m is None:
        return _v2_null("missing", f"no v2 measurement in {d}", v2_dir=str(d))
    kernels = v2.AuditKernels.of(
        manifest(spec.audit_dir) or {}, pool_digests(prep.pool.kernels)
    )
    meas = v2.measurements(d, prep.identity, kernels)
    in_pool = {int(k["sol_idx_global"]) for k in prep.pool.kernels}
    waiting = v2.pending(d)
    rows: List[Row] = []
    tally: Dict[str, Dict[str, int]] = {}
    for g, r in zip(prep.gemms, out.results):
        label = scorer.label(g)
        key = (label, _shape_of(g))
        t = tally.setdefault(
            label,
            {
                "n": 0,
                "benching": 0,
                "needs_increment": 0,
                "unplaced": 0,
                "flagged": 0,
                "canary": 0,
            },
        )
        t["n"] += 1
        mm = pool_medians(meas.get(key), in_pool)
        pick = r.pick_sol_idx
        if mm is None or mm.unplaced or pick is None or int(pick) not in mm.median:
            if key in waiting:
                t["benching"] += 1
            elif mm is not None and mm.unplaced:
                t["unplaced"] += 1
            else:
                t["needs_increment"] += 1
            continue
        w = mm.winner_us
        g2 = dict(g, winner_us=w)
        g2["candidates"] = [
            (
                dict(c, us=mm.median[int(c["sol_idx_global"])], is_skip=False)
                if int(c["sol_idx_global"]) in mm.median
                else c
            )
            for c in g["candidates"]
        ]
        rows.append(scorer.row(g2, r, w / mm.median[int(pick)]))
        t["flagged"] += mm.flagged
        t["canary"] += mm.canary_flagged
    groups = _summaries(rows, prep.now.groups, spec)
    group_status: Dict[str, Dict[str, Any]] = {}
    for label, g1 in prep.now.groups.items():
        t = tally.get(label) or {}
        gs: Dict[str, Any] = {
            "status": g1.status,
            "reason": g1.reason,
            "n_gemms": t.get("n", 0),
            "n_benching": t.get("benching", 0),
            "n_needs_increment": t.get("needs_increment", 0),
            "n_unplaced": t.get("unplaced", 0),
            "n_spread_flagged": t.get("flagged", 0),
            "n_canary_flagged": t.get("canary", 0),
        }
        if g1.status == "ok":
            unplaced = gs["n_unplaced"]
            missing = gs["n_benching"] + gs["n_needs_increment"] + unplaced
            if missing:
                gs["status"] = (
                    "needs_new_measurement"
                    if unplaced
                    else "benching" if gs["n_benching"] else "needs_increment"
                )
                gs["reason"] = (
                    f"{missing} of {gs['n_gemms']} GEMMs have no v2 measurement "
                    f"of this model's pick ({gs['n_benching']} being benched, "
                    f"{gs['n_needs_increment']} need an increment"
                    + (
                        f", {unplaced} were measured with kernels the audit set's "
                        f"digests cannot place now and need a new v2 measurement"
                        if unplaced
                        else ""
                    )
                    + ")"
                )
                groups[label] = None
            elif groups.get(label) is None:
                gs["status"], gs["reason"] = "error", "no GEMM of the group left"
        group_status[label] = gs
    shown = {
        k: Group(k, v["status"], v["reason"], 0, 0) for k, v in group_status.items()
    }
    ok = [k for k, v in group_status.items() if v["status"] == "ok"]
    if ok and len(ok) == len(group_status) and prep.now.status == "ok":
        st, why = "ok", None
    elif ok:
        st, why = "partial", None
    else:
        st, why = "incomplete", "no group has a v2 measurement of every pick"
    head = groups.get(EXCLUSIVE)
    return {
        "protocol": "v2",
        "protocol_text": PROTOCOLS["v2"],
        "status": st,
        "reason": why,
        "status_text": status_text(st, why, shown),
        "group_status": group_status,
        "v2_dir": str(d),
        "settings": m.get("settings"),
        "canary": {k: (m.get("canary") or {}).get(k) for k in ("shape", "kernel")},
        "generations": len(m.get("generations") or []),
        "n_gemms": len(rows),
        "headline": headline_of(head) if head else None,
        "groups": groups,
    }


def headline_of(group: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "n_gemms",
        "n_cells",
        "cell_mean",
        "cell_mean_ci",
        "gemm_geomean",
        "gemm_geomean_ci",
        "n_cells_below_target",
        "n_cells_confidently_below",
        "classes",
        "classes_confident",
    )
    out = {k: group.get(k) for k in keys}
    mc = group.get("min_cell") or {}
    out["min_cell"] = {k: mc.get(k) for k in ("cell", "sel_eff", "sel_eff_ci", "n")}
    return out


def evaluated_group(rec: Mapping[str, Any], label: str) -> Optional[Dict[str, Any]]:
    """The record's summary of `label` when that group was evaluated."""
    if rec.get("status") not in ("ok", "partial"):
        return None
    g = (rec.get("groups") or {}).get(label)
    gs = (rec.get("group_status") or {}).get(label) or {}
    if not g or gs.get("status", "ok") != "ok":
        return None
    return g


def per_gemm_of(rec: Mapping[str, Any], label: str) -> Dict[ShapeKey, float]:
    """{shape: selection efficiency} of group `label`'s `per_gemm` in an
    evaluated record; a GEMM without a positive efficiency is left out."""
    rows = ((rec.get("groups") or {}).get(label) or {}).get("per_gemm") or []
    return {
        (int(r[0]), int(r[1]), int(r[2]), int(r[3])): float(r[4])
        for r in rows
        if r[4] is not None and r[4] > 0
    }


def compare(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    label: str = EXCLUSIVE,
    bootstrap: int = DEFAULTS["bootstrap"],
    seed: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(change, None) from `before` to `after` on group `label`, paired GEMM
    by GEMM over the GEMMs both evaluated: per cell the geomean ratio of the
    selection efficiencies with a bootstrap interval over the cell's GEMMs
    (`improved` / `regressed` when the interval excludes 1), the
    equal-weight mean over cells and the GEMM-weighted geomean of the
    ratios with intervals; else (None, why) when either record lacks the
    group."""
    from .grid import cell_key

    for what, rec in (("before", before), ("after", after)):
        g = evaluated_group(rec, label)
        if g is None:
            return None, (
                f"the {what} evaluation ({rec.get('what') or '?'}) has no {label} "
                f"group ({rec.get('status_text') or rec.get('status')})"
            )
        if "per_gemm" not in g:
            return None, f"the {what} evaluation holds no per-GEMM results"
    a, b = per_gemm_of(before, label), per_gemm_of(after, label)
    by_cell: Dict[str, List[ShapeKey]] = {}
    for s in sorted(set(a) & set(b)):
        by_cell.setdefault(cell_key(*s), []).append(s)
    if not by_cell:
        return None, f"the evaluations share no evaluated {label} GEMM"
    rng = np.random.default_rng(seed)
    reps = max(0, int(bootstrap))
    cells: Dict[str, Dict[str, Any]] = {}
    boots: List[np.ndarray] = []
    weights: List[int] = []
    for cell in sorted(by_cell):
        shapes = by_cell[cell]
        la = np.log([a[s] for s in shapes])
        lb = np.log([b[s] for s in shapes])
        d = lb - la
        boot = _boot_means(d, reps, rng) if reps and d.size > 1 else None
        ci = _ci(boot)
        moved = None
        if ci is not None and ci[0] > 1:
            moved = "improved"
        elif ci is not None and ci[1] < 1:
            moved = "regressed"
        cells[cell] = {
            "n": int(d.size),
            "ratio": float(math.exp(d.mean())),
            "ratio_ci": ci,
            "before": _geo(la),
            "after": _geo(lb),
            "moved": moved,
        }
        boots.append(boot if boot is not None else np.full(reps, d.mean()))
        weights.append(int(d.size))
    means = np.asarray([math.log(c["ratio"]) for c in cells.values()])
    w = np.asarray(weights, dtype=np.float64)
    stack = np.vstack(boots) if reps else None
    return {
        "group": label,
        "before": before.get("what"),
        "after": after.get("what"),
        "n_gemms": int(w.sum()),
        "n_cells": len(cells),
        "cell_mean_ratio": float(math.exp(means.mean())),
        "cell_mean_ratio_ci": _ci(stack.mean(axis=0)) if stack is not None else None,
        "gemm_geomean_ratio": float(math.exp((w * means).sum() / w.sum())),
        "gemm_geomean_ratio_ci": (
            _ci((w[:, None] * stack).sum(0) / w.sum()) if stack is not None else None
        ),
        "improved": [c for c, v in cells.items() if v["moved"] == "improved"],
        "regressed": [c for c, v in cells.items() if v["moved"] == "regressed"],
        "per_cell": cells,
    }, None


def protocol_record(rec: Mapping[str, Any], protocol: str) -> Dict[str, Any]:
    """The record of `protocol`: the record itself for v1, its `v2`
    sub-record (named like the record) for v2, {} when it has none."""
    if protocol == "v1":
        return dict(rec)
    sub = rec.get("v2")
    if not isinstance(sub, dict):
        return {}
    return dict(sub, what=rec.get("what"), model_sha256=rec.get("model_sha256"))


def view(rec: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    """(record, protocol) a report shows of an evaluation: its v2 record
    when v2 is complete for the model, else the record itself (v1)."""
    sub = protocol_record(rec, "v2")
    if sub.get("status") == "ok":
        return sub, "v2"
    return dict(rec), "v1"


def protocol_of(records: Sequence[Mapping[str, Any]], label: str = EXCLUSIVE) -> str:
    """v2 when it evaluated `label` for every record, else v1."""
    if records and all(
        evaluated_group(protocol_record(r, "v2"), label) for r in records
    ):
        return "v2"
    return "v1"


def paired_change(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    label: str = EXCLUSIVE,
    bootstrap: int = DEFAULTS["bootstrap"],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    """`compare` under v2 when it evaluated `label` for both records, else
    under v1, and the protocol used."""
    proto = protocol_of([before, after], label)
    c, why = compare(
        protocol_record(before, proto), protocol_record(after, proto), label, bootstrap
    )
    if c is not None:
        c["protocol"] = proto
    return c, why, proto


def not_worse(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    label: str = EXCLUSIVE,
    bootstrap: int = DEFAULTS["bootstrap"],
) -> Tuple[Optional[bool], str, Optional[Dict[str, Any]], str]:
    """(verdict, why, change, protocol): `after` is not worse than `before`
    when its equal-weight mean over cells of the paired per-cell ratios on
    group `label` (`paired_change`, under v2 when both records have it) is
    at least 1; the verdict is None, with why, when the records do not
    pair on that group."""
    change, why, proto = paired_change(before, after, label, bootstrap)
    if change is None:
        return None, str(why), None, proto
    ratio = float(change["cell_mean_ratio"])
    text = (
        f"cell mean x{ratio:.4f}{_ratio_ci(change['cell_mean_ratio_ci'])}, GEMM "
        f"geomean x{change['gemm_geomean_ratio']:.4f} over {change['n_gemms']} "
        f"paired {label} GEMMs / {change['n_cells']} cells, "
        f"{len(change['improved'])} cell(s) improved and "
        f"{len(change['regressed'])} regressed [{PROTOCOLS[proto]}]"
    )
    return ratio >= 1.0, text, change, proto


def cell_gate_settings(cfg: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The config's `gate` section over CELL_GATE_DEFAULTS (what `cell_gate`
    takes); None when `gate.per_cell` is false."""
    g = dict(cfg.get("gate") or {})
    if g.get("per_cell", True) is False:
        return None
    out = {k: (v if g.get(k) is None else g[k]) for k, v in CELL_GATE_DEFAULTS.items()}
    for k in ("cell_tolerance", "cell_floor", "alpha", "tail_floor", "tail_collapse"):
        out[k] = float(out[k])
    out["tail_budget"] = int(out["tail_budget"])
    out["groups"] = str(out["groups"])
    return out


def min_flips(alpha: float) -> int:
    """The fewest sign flips with which the cell tests can reach `alpha`:
    the smallest p-value `decline_p` gives is 2 / (flips + 1)."""
    return int(math.floor(2.0 / float(alpha) + 1e-9))


def _group_labels(rec: Mapping[str, Any]) -> List[str]:
    return [
        *(rec.get("groups") or {}),
        *(protocol_record(rec, "v2").get("groups") or {}),
    ]


def _step_down(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Westfall-Young step-down adjusted p-values for low values of `obs`
    (one per hypothesis) against `null` (sign flips x hypotheses): the most
    extreme hypothesis against the minimum over all of them, the next
    against the minimum over the rest, each p at least the one before."""
    flips = null.shape[0]
    order = np.argsort(obs, kind="stable")
    rest = np.minimum.accumulate(null[:, order][:, ::-1], axis=1)[:, ::-1]
    at = obs[order] + 1e-12 * (1.0 + np.abs(obs[order]))
    raw = (1.0 + (rest <= at[None, :]).sum(axis=0)) / (flips + 1.0)
    out = np.empty(len(obs))
    out[order] = np.minimum(np.maximum.accumulate(raw), 1.0)
    return out


def decline_p(
    cells: Mapping[Any, Tuple[Sequence[ShapeKey], Sequence[float]]],
    flips: int,
    seed: int = 0,
) -> Dict[Any, float]:
    """Family-wise p-values of a decline, one per key of `cells` ({key:
    (shapes, log ratios after / before of those GEMMs)}, the keys being the
    cells of one family): two Westfall-Young step-down tests over the keys
    with `flips` random GEMM-wise sign flips (one sign per shape, shared by
    every key that holds it), one on each key's mean log ratio (a large
    decline), one on its studentized sum, the sum over the root of the sum
    of squares (a consistent one), each change capped at a factor of two
    (`_CELL_CAP`). A key's p is twice the smaller of its two adjusted
    p-values, at most 1. When the two bundles are exchangeable (each GEMM's
    two efficiencies as likely the other way round), the chance that any
    key of the family gets a p below alpha is at most alpha. A key without
    GEMMs gets 1, and so does every key without flips."""
    keys = list(cells)
    if not keys:
        return {}
    if flips < 1:
        return {k: 1.0 for k in keys}
    index: Dict[ShapeKey, int] = {}
    parts: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    obs = np.zeros((2, len(keys)))
    for j, k in enumerate(keys):
        shapes, d = cells[k]
        dc = np.clip(np.asarray(d, dtype=np.float64), -_CELL_CAP, _CELL_CAP)
        if not dc.size:
            parts.append(None)
            continue
        root = float(np.sqrt((dc * dc).sum()))
        w = np.vstack([dc / dc.size, dc / root if root > 0 else np.zeros_like(dc)])
        obs[:, j] = w.sum(axis=1)
        cols = [index.setdefault(tuple(int(v) for v in s), len(index)) for s in shapes]
        parts.append((np.asarray(cols, dtype=np.int64), w))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(int(flips), max(1, len(index))), dtype=np.int8)
    signs = signs * 2 - 1
    null = np.zeros((2, int(flips), len(keys)))
    for j, part in enumerate(parts):
        if part is not None:
            cols, w = part
            null[:, :, j] = (signs[:, cols].astype(np.float64) @ w.T).T
    p = np.minimum(_step_down(obs[0], null[0]), _step_down(obs[1], null[1]))
    return {k: float(min(1.0, 2.0 * p[j])) for j, k in enumerate(keys)}


def _paired_cells(
    before: Mapping[str, Any], after: Mapping[str, Any], label: str, proto: str
) -> Dict[str, Tuple[List[ShapeKey], np.ndarray, np.ndarray]]:
    """{cell: (shapes, log ratios after / before, log efficiencies after)}
    of group `label`'s GEMMs both records evaluated under `proto`."""
    from .grid import cell_key

    a = per_gemm_of(protocol_record(before, proto), label)
    b = per_gemm_of(protocol_record(after, proto), label)
    by_cell: Dict[str, List[ShapeKey]] = {}
    for s in sorted(set(a) & set(b)):
        by_cell.setdefault(cell_key(*s), []).append(s)
    return {
        cell: (
            shapes,
            np.log([b[s] for s in shapes]) - np.log([a[s] for s in shapes]),
            np.log([b[s] for s in shapes]),
        )
        for cell, shapes in by_cell.items()
    }


def _crossings(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    label: str,
    proto: str,
    floor: float,
    collapse: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(new, lifted) picks of group `label` under `proto`. New: a GEMM whose
    pick in `after` falls below `floor` from at or above it in `before`, to
    at most `collapse` of `before`'s efficiency (a collapse, not a step
    across the line). Lifted: the reverse, `before`'s pick below the floor
    and at most `collapse` of `after`'s. Each is `confirmed` by a repeated
    measurement: under v2 (the median of repeated joint benches) always;
    under v1 when `after`'s v2 record evaluated the group and its
    measurement of the GEMM (`after_v2`) says the same."""
    from .grid import cell_key

    a = per_gemm_of(protocol_record(before, proto), label)
    b = per_gemm_of(protocol_record(after, proto), label)
    check: Optional[Dict[ShapeKey, float]] = None
    if proto != "v2":
        sub = protocol_record(after, "v2")
        check = per_gemm_of(sub, label) if evaluated_group(sub, label) else {}

    def falls(x: float, y: float) -> bool:
        return y < floor <= x and y <= collapse * x

    new, lifted = [], []
    for s in sorted(set(a) & set(b)):
        down = falls(a[s], b[s])
        if not (down or falls(b[s], a[s])):
            continue
        entry: Dict[str, Any] = {
            "shape": list(s),
            "cell": cell_key(*s),
            "before": a[s],
            "after": b[s],
            "confirmed": True,
        }
        if check is not None:
            seen = check.get(s)
            entry["after_v2"] = seen
            entry["confirmed"] = seen is not None and (
                falls(a[s], seen) if down else falls(seen, a[s])
            )
        (new if down else lifted).append(entry)
    return new, lifted


def cell_gate(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    settings: Optional[Mapping[str, Any]] = None,
    bootstrap: int = DEFAULTS["bootstrap"],
) -> Dict[str, Any]:
    """Per-cell non-regression of `after` against `before`, the bundle that
    ships, on every group either names: each group both evaluated is paired
    GEMM by GEMM under v2 when both records have it for the group, else v1
    (`paired_change`, its intervals resampled within each cell), with the
    thresholds of `settings` (CELL_GATE_DEFAULTS; a floor of 0 is off).

    The cell rules judge the groups `settings["groups"]` names (`exclusive`,
    or `all`) as one family: `decline_p` with `bootstrap` sign flips gives
    each of their cells a family-wise p of a decline (`p`), so that when
    the two bundles are exchangeable the chance of a finding in any of
    those cells is at most alpha. A cell is
      regressed  significantly worse beyond the tolerance: its p is below
                 alpha and its paired ratio below 1 - cell_tolerance;
      floor      confidently fallen below the floor: its p is below alpha,
                 its geomean on the paired GEMMs was at or above cell_floor
                 in `before`, and the 95% interval of `after`'s geomean
                 there lies below it.
    Every other group is tested the same way, as a family of its own, and
    its findings are reported (`deciding` false).

    The tail rule judges every group (`_crossings`): a group keeps `after`
    out (`tail_refuses`) when its new picks collapsing below tail_floor
    that a repeated measurement confirms outnumber the confirmed ones it
    lifts back above it by more than tail_budget; every other new collapse
    is reported.

    `passed` is False with a finding that decides; else None when no group
    the cell rules judge pairs, or `bootstrap` gives too few flips to reach
    alpha (`cell_rules` says so); else True. `reason` names the cells and
    GEMMs; a group that does not pair says why (`reason`)."""
    st = {**CELL_GATE_DEFAULTS, **dict(settings or {})}
    tol, floor = float(st["cell_tolerance"]), float(st["cell_floor"])
    alpha = float(st["alpha"])
    tail, collapse = float(st["tail_floor"]), float(st["tail_collapse"])
    budget = int(st["tail_budget"])
    flips = max(0, int(bootstrap))
    need = min_flips(alpha)
    cell_rules = (
        "on"
        if flips >= need
        else (
            f"off: {flips} sign flips (audit.bootstrap) cannot reach alpha "
            f"{alpha:g}, which takes at least {need}"
        )
    )
    labels = list(
        dict.fromkeys([EXCLUSIVE, *_group_labels(before), *_group_labels(after)])
    )
    groups: Dict[str, Dict[str, Any]] = {}
    paired: Dict[str, Dict[str, Tuple[List[ShapeKey], np.ndarray, np.ndarray]]] = {}
    changed: Dict[str, Dict[str, Any]] = {}
    for label in labels:
        deciding = st["groups"] == "all" or label == EXCLUSIVE
        change, why, proto = paired_change(before, after, label, bootstrap)
        if change is None:
            groups[label] = {"protocol": proto, "deciding": deciding, "reason": why}
            continue
        changed[label] = change
        paired[label] = _paired_cells(before, after, label, proto)
        new, lifted = (
            _crossings(before, after, label, proto, tail, collapse)
            if tail > 0
            else ([], [])
        )
        n_new = sum(1 for t in new if t["confirmed"])
        n_lifted = sum(1 for t in lifted if t["confirmed"])
        refuses = n_new > n_lifted + budget
        for t in new:
            t["decides"] = bool(refuses and t["confirmed"])
        groups[label] = {
            "protocol": proto,
            "deciding": deciding,
            "n_gemms": change["n_gemms"],
            "n_cells": change["n_cells"],
            "p": {},
            "regressed": [],
            "floor": [],
            "tail": new,
            "tail_lifted": lifted,
            "n_tail_new": n_new,
            "n_tail_lifted": n_lifted,
            "tail_refuses": refuses,
        }
    p_of: Dict[Tuple[str, str], float] = {}
    if cell_rules == "on":
        families = [[k for k in paired if groups[k]["deciding"]]]
        families += [[k] for k in paired if not groups[k]["deciding"]]
        for i, family in enumerate(families):
            p_of.update(
                decline_p(
                    {(k, c): v[:2] for k in family for c, v in paired[k].items()},
                    flips,
                    seed=i,
                )
            )
    rng = np.random.default_rng(0)
    for label, change in changed.items():
        g = groups[label]
        for cell, c in change["per_cell"].items():
            p = p_of.get((label, cell))
            if p is None:
                continue
            g["p"][cell] = p
            if p >= alpha:
                continue
            entry = {
                "cell": cell,
                "n": c["n"],
                "ratio": c["ratio"],
                "ratio_ci": c["ratio_ci"],
                "p": p,
                "before": c["before"],
                "after": c["after"],
            }
            if c["ratio"] < 1.0 - tol:
                g["regressed"].append(entry)
            if floor > 0 and c["before"] >= floor:
                lb = paired[label][cell][2]
                own = _ci(_boot_means(lb, flips, rng)) if lb.size > 1 else None
                if own is not None and own[1] < floor:
                    g["floor"].append(dict(entry, after_ci=own))
    compared = {k: g for k, g in groups.items() if "reason" not in g}
    deciding_groups = {k: g for k, g in compared.items() if g["deciding"]}

    def count(gs: Mapping[str, Mapping[str, Any]], kind: str) -> int:
        return sum(len(g[kind]) for g in gs.values())

    tail_groups = [k for k, g in compared.items() if g["tail_refuses"]]
    n_tail = sum(compared[k]["n_tail_new"] for k in tail_groups)
    n_cells = count(deciding_groups, "regressed") + count(deciding_groups, "floor")
    passed: Optional[bool]
    if n_cells or tail_groups:
        passed = False
    elif not deciding_groups or cell_rules != "on":
        passed = None
    else:
        passed = True
    out = {
        "settings": st,
        "before": before.get("what"),
        "after": after.get("what"),
        "passed": passed,
        "cell_rules": cell_rules,
        "n_groups": len(deciding_groups),
        "n_cells_tested": sum(len(paired[k]) for k in deciding_groups),
        "n_regressed": count(deciding_groups, "regressed"),
        "n_floor": count(deciding_groups, "floor"),
        "n_tail": n_tail,
        "tail_groups": tail_groups,
        "n_reported": count(compared, "regressed")
        + count(compared, "floor")
        - n_cells
        + count(compared, "tail")
        - n_tail,
        "not_compared": sorted(k for k in groups if k not in compared),
        "groups": groups,
    }
    out["reason"] = cell_gate_text(out)
    return out


def _shape_text(shape: Sequence[int]) -> str:
    m, n, k, b = (int(v) for v in shape)
    return f"{m}x{n}x{k}" + (f" batch {b}" if b != 1 else "")


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_SHOWN])
    more = len(items) - _SHOWN
    return shown + (f" and {more} more" if more > 0 else "")


def _rule_texts(st: Mapping[str, Any]) -> Tuple[str, str]:
    """(cell rules, tail rule) of gate settings `st` as the verdicts name
    them."""
    tol = f"{float(st['cell_tolerance']) * 100:.1f}%"
    floor = float(st["cell_floor"])
    cells = f"significantly worse beyond {tol}" + (
        f" or confidently fallen below {floor * 100:.1f}%" if floor > 0 else ""
    )
    budget = int(st["tail_budget"])
    if float(st["tail_floor"]) <= 0:
        return cells, "the tail rule off (gate.tail_floor 0)"
    tails = (
        f"more new picks collapsing below {float(st['tail_floor']) * 100:.1f}% "
        f"than it lifted"
    ) + (f" plus {budget}" if budget else "")
    return cells, tails


def cell_gate_text(result: Mapping[str, Any]) -> str:
    """One line on a `cell_gate` result: the cells and GEMMs that keep the
    bundle out (the first few of each kind, then how many more), else what
    it was judged on."""
    st = {**CELL_GATE_DEFAULTS, **dict(result.get("settings") or {})}
    groups = result.get("groups") or {}
    compared = {k: g for k, g in groups.items() if "reason" not in g}
    if not compared:
        why = "; ".join(f"{k}: {g.get('reason')}" for k, g in groups.items())
        return f"no group pairs the two bundles ({why})"
    deciding = {k: g for k, g in compared.items() if g.get("deciding")}
    cells_text, tails_text = _rule_texts(st)
    protos = ", ".join(f"{k} {g['protocol']}" for k, g in compared.items())
    notes = []
    skipped = [k for k, g in groups.items() if g.get("deciding") and "reason" in g]
    if skipped:
        notes.append(f"{len(skipped)} group(s) not compared ({', '.join(skipped)})")
    rules = result.get("cell_rules") or "on"
    if rules != "on":
        notes.append(f"the cell rules were {rules}")
    if result.get("n_reported"):
        notes.append(
            f"{result['n_reported']} finding(s) that do not decide (cells of groups "
            f"outside gate.groups {st['groups']}; collapses within a group's "
            f"budget or that no repeated measurement confirms)"
        )
    note = "".join(f"; {n}" for n in notes)
    if result.get("passed") is True:
        tails_ok = (
            tails_text
            if tails_text.startswith("the tail rule off")
            else f"no group with {tails_text}"
        )
        return (
            f"no cell {cells_text} (family-wise alpha {float(st['alpha']):g} over "
            f"{result.get('n_cells_tested')} cells of {', '.join(deciding)}), and "
            f"{tails_ok} ({protos}){note}"
        )
    parts = []
    if result.get("passed") is None:
        parts.append(
            "the cell rules could not decide"
            if deciding
            else "no group the cell rules judge pairs the two bundles"
        )
    reg = [
        f"{k} {c['cell']} x{c['ratio']:.4f}{_ratio_ci(c['ratio_ci'])} p {c['p']:.2g}"
        for k, g in deciding.items()
        for c in g["regressed"]
    ]
    if reg:
        tol = f"{float(st['cell_tolerance']) * 100:.1f}%"
        parts.append(
            f"{len(reg)} cell(s) significantly worse beyond {tol}: " + _listed(reg)
        )
    fell = [
        f"{k} {c['cell']} {c['after'] * 100:.2f}% (was {c['before'] * 100:.2f}%) "
        f"p {c['p']:.2g}"
        for k, g in deciding.items()
        for c in g["floor"]
    ]
    if fell:
        parts.append(
            f"{len(fell)} cell(s) confidently fell below "
            f"{float(st['cell_floor']) * 100:.1f}%: " + _listed(fell)
        )
    tails = [
        f"{k} ({g['n_tail_new']} new, {g['n_tail_lifted']} lifted: "
        + _listed(
            [
                f"{_shape_text(t['shape'])} ({t['cell']}) {t['after'] * 100:.2f}% "
                f"(was {t['before'] * 100:.2f}%)"
                for t in g["tail"]
                if t["confirmed"]
            ]
        )
        + ")"
        for k, g in compared.items()
        if g.get("tail_refuses")
    ]
    if tails:
        parts.append(f"{len(tails)} group(s) with {tails_text}: " + ", ".join(tails))
    return "; ".join(parts) + f" ({protos}){note}"


def changes(
    records: Sequence[Mapping[str, Any]],
    bootstrap: int = DEFAULTS["bootstrap"],
    protocol: str = "v1",
) -> List[Dict[str, Any]]:
    """`compare` of each record with the one before it on every group
    either names, under `protocol`; a group missing from either is a reason,
    never a number."""
    out = []
    recs = [protocol_record(r, protocol) for r in records]
    for before, after in zip(recs, recs[1:]):
        labels = list(
            dict.fromkeys([*(before.get("groups") or {}), *(after.get("groups") or {})])
        ) or [EXCLUSIVE]
        groups: Dict[str, Any] = {}
        for label in labels:
            c, why = compare(before, after, label, bootstrap)
            groups[label] = c if c is not None else {"reason": why}
        out.append(
            {
                "before": before.get("what"),
                "after": after.get("what"),
                "protocol": protocol,
                "protocol_text": PROTOCOLS[protocol],
                "groups": groups,
            }
        )
    return out


def change_lines(change: Mapping[str, Any]) -> List[str]:
    """A line per group of a `changes` entry."""
    out = []
    proto = PROTOCOLS.get(str(change.get("protocol") or "v1"), "")
    for label, c in (change.get("groups") or {}).items():
        head = f"{change.get('before')} -> {change.get('after')} {label}"
        if "reason" in c:
            out.append(f"{head}: not compared ({c['reason']}) [{proto}]")
            continue
        out.append(
            f"{head}: cell mean x{c['cell_mean_ratio']:.4f}"
            f"{_ratio_ci(c['cell_mean_ratio_ci'])}, GEMM geomean "
            f"x{c['gemm_geomean_ratio']:.4f}{_ratio_ci(c['gemm_geomean_ratio_ci'])} "
            f"over {c['n_gemms']} paired GEMMs / {c['n_cells']} cells; "
            f"{len(c['improved'])} cell(s) improved, {len(c['regressed'])} regressed "
            f"[{proto}]"
        )
    return out


def _ratio_ci(ci: Optional[Sequence[float]]) -> str:
    return "" if not ci else f" [{ci[0]:.4f}, {ci[1]:.4f}]"


def evaluate_to(
    path: Path,
    data: bytes,
    spec: Spec,
    cfg: Mapping[str, Any],
    *,
    hardware: Any,
    tw: Any = None,
    what: str = "",
    source: str = "",
) -> Dict[str, Any]:
    """`evaluate`, written to `path`; an error becomes a null record with
    status `error` (the audit only measures, it never fails a stage)."""
    try:
        rec = evaluate(data, spec, cfg, hardware=hardware, tw=tw)
    except Exception as e:  # noqa: BLE001
        rec = null_record(
            "error",
            f"{type(e).__name__}: {e}",
            audit_dir=str(spec.audit_dir) if spec.audit_dir else None,
            model_sha256=hashlib.sha256(data).hexdigest(),
        )
    rec.update(what=what, model=source)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_ready(rec), indent=2) + "\n")
    return rec


def json_ready(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:.2f}%"


def _ci_text(ci: Optional[Sequence[float]]) -> str:
    return "" if not ci else f" [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]"


def _group_lines(rec: Mapping[str, Any], tag: str) -> List[str]:
    out = []
    if rec.get("status") == "partial":
        out.append(f"{tag} {rec.get('status_text')}")
    for label, g in (rec.get("groups") or {}).items():
        if not g:
            gs = (rec.get("group_status") or {}).get(label) or {}
            out.append(f"{tag} {label}: null ({gs.get('status')}: {gs.get('reason')})")
            continue
        mc = g["min_cell"]
        cl = g["classes"]
        out.append(
            f"{tag} {label}: cell mean {_pct(g['cell_mean'])}{_ci_text(g['cell_mean_ci'])}"
            f", GEMM geomean {_pct(g['gemm_geomean'])}, {g['n_gemms']} GEMMs / "
            f"{g['n_cells']} cells, {g['n_cells_below_target']} below "
            f"{rec['target'] * 100:g}%, min {mc['cell']} {_pct(mc['sel_eff'])}; "
            f"pool-limited {cl['pool_limited']}, identity-limited "
            f"{cl['identity_limited']}, model-limited {cl['model_limited']}, "
            f"passing {cl['passing']}"
        )
    return out


def summary_lines(rec: Mapping[str, Any]) -> List[str]:
    """A few lines describing an evaluation record: v1 (`audit ...`), then
    v2 (`audit v2 ...`) when the record has it."""
    if rec.get("status") not in ("ok", "partial"):
        return [f"audit: null ({rec.get('status')}: {rec.get('reason')})"]
    out = _group_lines(rec, "audit")
    if rec.get("n_excluded_trained"):
        out.append(
            f"audit: {rec['n_excluded_trained']} audit shape(s) found in training "
            f"rounds were left out"
        )
    sub = rec.get("v2")
    if isinstance(sub, dict):
        if sub.get("status") in ("ok", "partial"):
            out += _group_lines(dict(sub, target=rec["target"]), "audit v2")
        else:
            out.append(f"audit v2: null ({sub.get('status')}: {sub.get('reason')})")
    return out


# ── stage and CLI glue ───────────────────────────────────────────────────────


def add_arguments(ap: Any) -> None:
    """--audit-dir / --audit-target / --audit-bootstrap of stage04b and
    stage06."""
    g = ap.add_argument_group("audit set (lib/audit.py)")
    g.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="evaluate the model on this audit set too (audit.json); the "
        "result is null when it is missing or incomplete",
    )
    g.add_argument("--audit-target", type=float, default=DEFAULTS["target"])
    g.add_argument("--audit-bootstrap", type=int, default=DEFAULTS["bootstrap"])
    g.add_argument(
        "--audit-trained-round-dir",
        type=Path,
        action="append",
        default=None,
        help="a round whose shapes the model trained on, repeatable (default: "
        "the rounds the training round dir implies); the audit leaves their "
        "shapes out",
    )


def trained_rounds(args: Any, default: Sequence[Path]) -> List[Path]:
    """The --audit-trained-round-dir rounds, else `default`."""
    given = getattr(args, "audit_trained_round_dir", None)
    return list(given) if given else list(default)


def bundle_model(bundle_dir: Path) -> Tuple[Path, str]:
    """(model file, source) of a bundle directory (a round, or a final
    retrain): its stage06's staged bundle, the bytes it deployed, else its
    stage05 models.pt."""
    d = Path(bundle_dir)
    rec = read_json(d / "stage06" / "deploy_record.json") or {}
    staged = rec.get("staged_model")
    if staged and Path(str(staged)).is_file():
        return Path(str(staged)), "stage06 staged bundle"
    if staged:
        found = sorted((d / "stage06").rglob(Path(str(staged)).name))
        if len(found) == 1:
            return found[0], "stage06 staged bundle"
    return d / "stage05" / "models.pt", "stage05 models.pt"


def bundle_trained_rounds(bundle_dir: Path) -> List[Path]:
    """The rounds whose shapes the bundle in `bundle_dir` trained on: a
    round's own and the earlier ones; a final retrain's, as its
    final_retrain.json names them."""
    from . import run_bundles

    rounds = rounds_until(bundle_dir, True)
    if rounds or run_bundles.round_index(Path(bundle_dir)) is not None:
        return rounds
    run = Path(bundle_dir).resolve().parent
    return [run / name for name in run_bundles.trained_rounds(bundle_dir)]


def model_bytes(path: Path, arch: str, weight_dtype: str) -> bytes:
    """MLREC bytes of a model file: a stage05 `models.pt` bundle is written
    at `weight_dtype`, as stage06 deploys it; anything else is read as
    MLREC."""
    path = Path(path)
    if path.suffix != ".pt":
        return path.read_bytes()
    import torch

    from . import hardware as hwlib
    from . import mlrec

    bundle = torch.load(path, map_location="cpu", weights_only=True)
    return mlrec.write_model(
        bundle, None, arch, hwlib.arch_constants(arch), weight_dtype
    )


def run_for_stage(
    out_path: Path,
    data: bytes,
    spec: Spec,
    cfg: Mapping[str, Any],
    *,
    hardware: Any,
    tw: Any = None,
    what: str,
    source: str,
    quiet: bool = False,
) -> Dict[str, Any]:
    """`evaluate_to` plus its summary on the terminal."""
    from . import ui

    rec = evaluate_to(
        out_path, data, spec, cfg, hardware=hardware, tw=tw, what=what, source=source
    )
    if not quiet:
        say = ui.ok if rec.get("status") == "ok" else ui.warn
        for line in summary_lines(rec):
            say("audit", line)
    return rec


def is_measured(rec: Optional[Mapping[str, Any]]) -> bool:
    """The record holds numbers (complete or partial)."""
    return bool(rec) and rec.get("status") in ("ok", "partial")
