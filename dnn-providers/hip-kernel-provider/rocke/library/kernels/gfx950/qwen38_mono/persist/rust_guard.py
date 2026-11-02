# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The serving guard of the Rust direct code-object backend (PLAN decision 18 (a)).

A kernel family / variant compiles with the direct backend only where a same-job paired A/B on BOTH rocke nodes
showed direct <= COMGR; everywhere else it compiles with COMGR. The evidence lives in a tuning DB
(:class:`rocke.dispatch.tuning.search.db.TuningDB` format, family :data:`FAMILY`): one record per
(key, node), ``problem = "<key>@<node>"``, ``value`` = the direct arm's time / the COMGR arm's time in that job
(< 1 is faster), ``knobs`` = the timed direct object's levers and the harness's verdict under the DESIGN 9.1 win
rule (``WIN`` / ``TIE`` / ``LOSS``, or ``HOLD``: measured but not booked, e.g. a screen and its confirmation
disagree) plus the base the verdict was measured on (decision 15).

Rule (:func:`passes`): on each of the two nodes the record's verdict is ``WIN``, or ``TIE`` with a ratio <= 1
("TIE-or-better"); both records time the same lever set. Anything missing, losing, or a tie above 1 keeps COMGR.

Keys name a kernel family and variant: ``b3.r1.tuned``, ``moe_fused.r4``, ``head.logits.r16``, ``fw.q2.r1``,
``hc.delta.attn.r1``, ``moe_lb2.<cell>`` ... (:data:`KEY_PREFIX` maps the ``RUST_LEVER_SETS`` families).

The guard is opt-in for builders (:func:`guard_kernel`, ``persist.knobs.rust_family(..., key=...)``) and, with
``ROCKE_DIRECT_GUARD=1``, also gates the family-level defaults of ``persist.knobs`` (default off until review).
It never changes IR or LLVM text: it only sets the Python-side backend mark and ``attrs["direct_levers"]`` (which
the direct backend strips), so every COMGR object is byte-identical with or without it.

CLI (records come from the paired harnesses; ratios only, absolutes stay in private-perf)::

    python -m kernels.gfx950.qwen38_mono.persist.rust_guard table [--db DB]
    python -m kernels.gfx950.qwen38_mono.persist.rust_guard put KEY NODE RATIO VERDICT --levers a+b --base B --job J
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "FAMILY",
    "CANDIDATE",
    "NODES",
    "KEY_PREFIX",
    "Selection",
    "db_path",
    "load_records",
    "make_record",
    "passes",
    "select",
    "select_family",
    "guard_kernel",
    "guard_enabled",
]

FAMILY = "qwen38_mono.codeobj"
CANDIDATE = "code_object_backend"
ARCH = "gfx950"
NODES = ("n01-09", "n07-17")
VERDICTS = ("WIN", "TIE", "LOSS", "HOLD")  # HOLD: measured, not booked (screen and confirmation disagree)
# persist.knobs.RUST_LEVER_SETS family -> the guard key prefix of its standalone kernels
KEY_PREFIX = {"b3": "b3", "moe_r1": "moe_fused.r1", "moe_r4": "moe_fused.r4", "head": "head"}
_DB_ENV = "ROCKE_DIRECT_GUARD_DB"
_GUARD_ENV = "ROCKE_DIRECT_GUARD"
_DEFAULT_DB = Path(__file__).with_name("rust_backend_db.json")


def guard_enabled() -> bool:
    """``ROCKE_DIRECT_GUARD=1``: the family-level defaults of ``persist.knobs`` consult the DB too."""
    return os.environ.get(_GUARD_ENV, "0") == "1"


def db_path() -> Path:
    """``ROCKE_DIRECT_GUARD_DB`` if set, else the DB shipped next to this module."""
    p = os.environ.get(_DB_ENV)
    return Path(p) if p else _DEFAULT_DB


def node_of(machine: str) -> str:
    """``smci355-ccs-aus-n01-09`` / ``n01-09`` -> ``n01-09``."""
    for n in NODES:
        if machine.endswith(n):
            return n
    raise ValueError(f"unknown rocke node {machine!r}; one of {NODES}")


def _levers_text(levers) -> str:
    if not levers:
        return ""
    if isinstance(levers, str):
        levers = levers.replace(",", "+").split("+")
    return "+".join(x.strip() for x in levers if x.strip())


def make_record(key: str, node: str, *, ratio: float, verdict: str, levers="", base: str, job: str,
                rounds: int = 0, audit: str = "", controls: str = "", p_value: float | None = None, note: str = ""):
    """A TunedRecord of one paired direct-vs-COMGR verdict (``ratio`` = direct / COMGR in one job)."""
    from rocke.dispatch.tuning.search.db import TunedRecord

    if verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict!r}; one of {VERDICTS}")
    node = node_of(node)
    lv = _levers_text(levers)
    ck = hashlib.sha256(lv.encode()).hexdigest()[:12]
    return TunedRecord(
        family=FAMILY, candidate=CANDIDATE, arch=ARCH, problem=f"{key}@{node}",
        tuning_id=f"{CANDIDATE}:{key}@{ck}", config_key=ck,
        knobs={"backend": "direct", "levers": lv, "verdict": verdict, "base": base, "audit": audit,
               "controls": controls, "note": note},
        value=float(ratio), unit="ratio", strategy="paired_ab", trace=job, rounds=int(rounds),
        speedup=1.0 / float(ratio), p_value=p_value, machine=f"smci355-ccs-aus-{node}",
    )


def load_records(path: str | os.PathLike | None = None) -> dict[tuple[str, str], object]:
    """``{(key, node): TunedRecord}`` of the guard's family (read-only, no lock file)."""
    from rocke.dispatch.tuning.search.db import TunedRecord

    p = Path(path) if path else db_path()
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for d in data.get("records", []):
        if d.get("family") != FAMILY or d.get("candidate") != CANDIDATE:
            continue
        r = TunedRecord.from_json(d)
        key, _, node = r.problem.rpartition("@")
        out[(key, node)] = r
    return out


def passes(rec) -> bool:
    """One node's record allows direct: ``WIN``, or ``TIE`` with ratio <= 1."""
    if rec is None or rec.knobs.get("backend") != "direct":
        return False
    v = rec.knobs.get("verdict")
    return v == "WIN" or (v == "TIE" and rec.value <= 1.0)


@dataclasses.dataclass(frozen=True)
class Selection:
    backend: str  # "direct" | "comgr"
    levers: str  # "+"-joined lever set of the timed direct object ("" for COMGR)
    reason: str
    ratios: tuple = ()  # ((node, ratio, verdict), ...)


def select(key: str, records: Mapping | None = None) -> Selection:
    """The backend of kernel ``key``: direct only when both nodes' records pass with the same lever set."""
    recs = load_records() if records is None else records
    rows = [(n, recs.get((key, n))) for n in NODES]
    ratios = tuple((n, r.value, r.knobs.get("verdict")) for n, r in rows if r is not None)
    missing = [n for n, r in rows if r is None]
    if missing:
        return Selection("comgr", "", f"{key}: no paired record on {'/'.join(missing)}", ratios)
    bad = [n for n, r in rows if not passes(r)]
    if bad:
        return Selection("comgr", "", f"{key}: direct > COMGR or a loss on {'/'.join(bad)}", ratios)
    lv = {r.knobs.get("levers", "") for _, r in rows}
    if len(lv) != 1:
        return Selection("comgr", "", f"{key}: the two nodes timed different lever sets", ratios)
    return Selection("direct", lv.pop(), f"{key}: direct <= COMGR on both nodes", ratios)


def select_family(prefix: str, records: Mapping | None = None) -> Selection:
    """Family-level guard: direct only when every recorded key under ``prefix`` (``prefix`` itself or
    ``prefix.*``) is direct with one common lever set, and at least one key is recorded."""
    recs = load_records() if records is None else records
    keys = sorted({k for k, _ in recs if k == prefix or k.startswith(prefix + ".")})
    if not keys:
        return Selection("comgr", "", f"{prefix}: no records")
    sels = [select(k, recs) for k in keys]
    bad = [s.reason for s in sels if s.backend != "direct"]
    if bad:
        return Selection("comgr", "", f"{prefix}: " + "; ".join(bad))
    lv = {s.levers for s in sels}
    if len(lv) != 1:
        return Selection("comgr", "", f"{prefix}: keys timed different lever sets")
    return Selection("direct", lv.pop(), f"{prefix}: {len(keys)} keys direct <= COMGR on both nodes")


def guard_kernel(kernel, key: str, records: Mapping | None = None):
    """Mark ``kernel`` for the backend :func:`select` gives ``key``: direct with the timed lever set as
    ``attrs["direct_levers"]``, else an explicit COMGR mark (an environment ``ROCKE_CODEOBJ_BACKEND=direct``
    then no longer moves it). Returns ``(kernel, Selection)``; IR and LLVM text never change."""
    from rocke.helpers.compile import request_code_object_backend

    sel = select(key, records)
    if sel.backend == "direct":
        if sel.levers:
            kernel.attrs["direct_levers"] = sel.levers
        else:
            kernel.attrs.pop("direct_levers", None)
    request_code_object_backend(kernel, sel.backend)
    return kernel, sel


# ----------------------------------------------------------------------------------------------- CLI
def _table(records) -> str:
    keys = sorted({k for k, _ in records})
    lines = ["| key | n01-09 | n07-17 | backend | levers | base |", "|---|---|---|---|---|---|"]
    for k in keys:
        s = select(k, records)
        cells = []
        for n in NODES:
            r = records.get((k, n))
            cells.append("-" if r is None else f"{r.value:.3f} {r.knobs.get('verdict')}")
        bases = sorted({records[(k, n)].knobs.get("base", "") for n in NODES if (k, n) in records})
        lv = s.levers if s.backend == "direct" else "-"
        lines.append(f"| `{k}` | {cells[0]} | {cells[1]} | **{s.backend}** | {lv or '(defaults)'} | {'; '.join(bases)} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rust_guard")
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    p = sub.add_parser("select")
    p.add_argument("key")
    p = sub.add_parser("put")
    for a in ("key", "node", "ratio", "verdict"):
        p.add_argument(a)
    p.add_argument("--levers", default="")
    p.add_argument("--base", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--rounds", type=int, default=0)
    p.add_argument("--audit", default="")
    p.add_argument("--controls", default="")
    p.add_argument("--note", default="")
    a = ap.parse_args(argv)
    path = Path(a.db) if a.db else db_path()
    if a.cmd == "table":
        print(_table(load_records(path)))
    elif a.cmd == "select":
        print(select(a.key, load_records(path)))
    else:
        from rocke.dispatch.tuning.search.db import TuningDB

        db = TuningDB(path)
        rec = make_record(a.key, a.node, ratio=float(a.ratio), verdict=a.verdict, levers=a.levers, base=a.base,
                          job=a.job, rounds=a.rounds, audit=a.audit, controls=a.controls, note=a.note)
        db.put(rec, force=True)
        print("stored", rec.problem, rec.value, rec.knobs["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
