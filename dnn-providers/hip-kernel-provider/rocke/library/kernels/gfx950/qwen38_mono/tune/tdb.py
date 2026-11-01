# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The monokernel's tuning DB: confirmed picks per (adapter, variant), the
signed-gain ledger, per-variant defaults and a learned selector.

Storage is the engine's :class:`~rocke.dispatch.tuning.search.db.TuningDB`
(one JSON document, locked merge, significance-gated serving): a record's
``family`` is ``qwen38_mono``, ``candidate`` the adapter name, ``problem``
the variant (``r1``, ``r4s``, ``r4g2``, ``r16``, ``delta.r1``, ``d8w8l1``,
large-batch ``b64tp2`` ...), ``value`` the confirmed ratio vs the baseline
(``unit="ratio"``), ``speedup`` / ``p_value`` the paired confirmation
against the baseline. Only significant records are served as defaults.

Next to it, ``<db>.gains.json`` keeps every measured signed gain ``1 -
ratio vs the then-current best`` per (adapter stem, knob, level): the
``prior`` of the next climb (consistent losers are skipped, promising levers
go first) and the :class:`~.strategies.GainPrior` of model-based members.

:func:`export_defaults` writes the BEST tables the stage generators and ATOM
read (knob JSON per variant; ``ATOM_Q38_SCHED`` group strings for the
schedule adapter). :class:`Selector` predicts a knob set for a variant that
was never tuned: nearest tuned variants (log-feature distance) and, with
enough records and ``lightgbm`` installed, a LightGBM model of log ratio over
(variant features, knob one-hots) cross-validated by variant.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from rocke.dispatch.tuning.search.db import TunedRecord, TuningDB
from rocke.dispatch.tuning.search.portfolio import WarmStart

from .catalog import variant_features
from .knobs import KnobAdapter

__all__ = ["FAMILY", "GainLedger", "Selector", "export_defaults", "make_record"]

FAMILY = "qwen38_mono"


def make_record(
    adapter: KnobAdapter,
    config: Mapping[str, object],
    *,
    ratio: float,
    speedup: float | None,
    p_value: float | None,
    rounds: int,
    strategy: str,
    evaluations: int,
    seed: int = 0,
    trace: str = "",
    toolchain: str = "",
    machine: str = "",
    problem: str | None = None,
) -> TunedRecord:
    """A DB record of a confirmed pick (``ratio``: its time over the
    baseline's in the confirmation)."""
    knobs = adapter.canonical_knobs(config)
    tid = adapter.tuning_id(config)
    if tid is None:
        raise ValueError(f"{adapter.stem}: config is illegal: {adapter.why_illegal(config)}")
    variant = problem or adapter.variant or "default"
    return TunedRecord(
        family=FAMILY,
        candidate=adapter.name,
        arch=adapter.arch,
        problem=variant,
        tuning_id=tid,
        config_key=tid.rsplit("@", 1)[1],
        knobs=knobs,
        value=float(ratio),
        unit="ratio",
        features=variant_features(variant),
        strategy=strategy,
        evaluations=int(evaluations),
        seed=int(seed),
        trace=trace,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        speedup=speedup,
        p_value=p_value,
        rounds=int(rounds),
        toolchain=toolchain,
        machine=machine,
    )


class GainLedger:
    """Signed gains per ``(stem, knob, level label)``, merged into a JSON
    file under a lock (concurrent tuners keep each other's entries)."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else None
        self.data: dict[str, dict[str, list[float]]] = {}
        if self.path is not None and self.path.exists():
            self.data = json.loads(self.path.read_text())

    @staticmethod
    def _k(knob: str, level: str) -> str:
        return f"{knob}={level}"

    def add(self, stem: str, gains: Mapping[str, Sequence[float]]) -> None:
        """``gains``: ``{"knob=level": [g, ...]}`` (a SignedLookback report)."""
        d = self.data.setdefault(stem, {})
        for k, v in gains.items():
            d.setdefault(k, []).extend(float(x) for x in v)

    def prior(self, stem: str) -> dict[tuple[str, str], list[float]]:
        out = {}
        for k, v in self.data.get(stem, {}).items():
            knob, _, level = k.partition("=")
            out[(knob, level)] = list(v)
        return out

    def add_pairs(self, stem: str, interactions: Mapping[str, float]) -> None:
        """Measured pairwise interaction terms (``{"a=x|b=y": log-time
        term}``, :meth:`.strategies.SignedLookback.interactions`)."""
        d = self.data.setdefault(stem + "#pairs", {})
        for k, v in interactions.items():
            d.setdefault(k, []).append(float(v))

    def add_combos(self, stem: str, combos: Mapping[str, float]) -> None:
        """Measured combinations: ``{"a=x,b=y,...": ratio vs the lookback
        base}``."""
        d = self.data.setdefault(stem + "#combos", {})
        for k, v in combos.items():
            d.setdefault(k, []).append(float(v))

    def add_verdicts(self, stem: str, verdicts: Sequence[Mapping]) -> None:
        """Base-relative verdicts (PLAN decision 15): per ``"knob=level"``,
        every measured gain with the base it was measured on (its changes from
        the climb's start) and the verdict there. A verdict holds for its base
        only; nothing here ever removes a level from a later search."""
        d = self.data.setdefault(stem + "#verdicts", {})
        for v in verdicts:
            d.setdefault(f"{v['lever']}={v['level']}", []).append(
                {"gain": float(v["gain"]), "base": ",".join(v.get("base_changes", [])), "verdict": v["verdict"]})

    def verdicts(self, stem: str) -> dict[str, list[dict]]:
        return {k: list(v) for k, v in self.data.get(stem + "#verdicts", {}).items()}

    def prior_pairs(self, stem: str) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self.data.get(stem + "#pairs", {}).items()}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        with _lock(lock):
            disk = json.loads(self.path.read_text()) if self.path.exists() else {}
            for stem, d in self.data.items():
                tgt = disk.setdefault(stem, {})
                for k, v in d.items():
                    have = tgt.get(k, [])
                    # merge as multisets: entries already on disk are not doubled
                    extra = list(v)
                    for x in have:
                        if x in extra:
                            extra.remove(x)
                    tgt[k] = have + extra
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(disk, sort_keys=True, indent=1))
            tmp.replace(self.path)
            self.data = disk


@contextlib.contextmanager
def _lock(path: Path, timeout: float = 120.0):
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - t0 > timeout:
                with contextlib.suppress(OSError):
                    os.unlink(path)
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(path)


def export_defaults(db: TuningDB, *, candidates: Sequence[str] | None = None) -> dict:
    """``{candidate: {variant: knobs}}`` of every *servable* (significant)
    record, plus ``"sched_string"``: the ``ATOM_Q38_SCHED`` value
    (``kind.variant:levers`` groups) of the schedule adapter's records."""
    out: dict[str, dict] = {}
    groups = []
    for r in sorted(db.records(), key=lambda r: (r.candidate, r.problem)):
        if r.family != FAMILY or (candidates and r.candidate not in candidates):
            continue
        if not db.significant(r):
            continue
        out.setdefault(r.candidate, {})[r.problem] = dict(r.knobs)
        if r.candidate == "sched" and r.knobs:
            body = ",".join(f"{k}={v}" for k, v in sorted(r.knobs.items()))
            groups.append(f"{r.problem}:{body}")
    if groups:
        out["sched_string"] = ";".join(groups)
    return out


class Selector:
    """Predicts a knob set for an untuned variant from tuned ones.

    ``top_k(candidate, variant, k)``: warm starts, nearest tuned variants
    first (distance over log2 of :func:`.catalog.variant_features`), then the
    LightGBM model's best predicted knob sets when it is trained.
    """

    def __init__(self, db: TuningDB, *, min_records: int = 12):
        self.db = db
        self.min_records = int(min_records)
        self.models: dict[str, object] = {}
        self._cols: dict[str, list[str]] = {}

    def _records(self, candidate: str) -> list[TunedRecord]:
        return [r for r in self.db.records() if r.family == FAMILY and r.candidate == candidate]

    @staticmethod
    def _dist(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        keys = set(a) | set(b)
        if not keys:
            return math.inf
        s = 0.0
        for k in keys:
            x, y = a.get(k, 1.0), b.get(k, 1.0)
            s += (math.log2(max(x, 1e-9)) - math.log2(max(y, 1e-9))) ** 2
        return math.sqrt(s)

    def nearest(self, candidate: str, variant: str, k: int = 3) -> list[tuple[float, TunedRecord]]:
        f = variant_features(variant)
        out = [(self._dist(f, r.features or variant_features(r.problem)), r)
               for r in self._records(candidate) if r.problem != variant]  # fmt: skip
        out.sort(key=lambda t: (t[0], t[1].value))
        return out[: int(k)]

    def _featurize(self, candidate: str, variant: str, knobs: Mapping[str, object]) -> list[float]:
        cols = self._cols[candidate]
        f = variant_features(variant)
        row = [math.log2(max(f.get(c[2:], 1.0), 1e-9)) if c.startswith("f:") else 0.0 for c in cols]
        for i, c in enumerate(cols):
            if c.startswith("k:"):
                name, _, val = c[2:].partition("=")
                row[i] = 1.0 if repr(knobs.get(name)) == val else 0.0
        return row

    def train(self, candidate: str) -> bool:
        """Fit the LightGBM model of log ratio (False when there are too few
        records or no ``lightgbm``)."""
        recs = self._records(candidate)
        if len(recs) < self.min_records:
            return False
        try:
            import lightgbm as lgb
            import numpy as np
        except ImportError:
            return False
        fcols = sorted({f"f:{k}" for r in recs for k in (r.features or variant_features(r.problem))})
        kcols = sorted({f"k:{k}={v!r}" for r in recs for k, v in r.knobs.items()})
        self._cols[candidate] = fcols + kcols
        X = np.asarray([self._featurize(candidate, r.problem, r.knobs) for r in recs])
        y = np.asarray([math.log(r.value) for r in recs])
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                  min_child_samples=2, verbose=-1)  # fmt: skip
        model.fit(X, y)
        self.models[candidate] = model
        return True

    def top_k(self, candidate: str, variant: str, k: int = 3) -> list[WarmStart]:
        out = [WarmStart(dict(r.knobs), f"neighbor:{r.problem}") for _d, r in self.nearest(candidate, variant, k)]
        model = self.models.get(candidate)
        if model is not None:
            pool = {json.dumps(r.knobs, sort_keys=True): r.knobs for r in self._records(candidate)}
            import numpy as np

            cands = list(pool.values())
            X = np.asarray([self._featurize(candidate, variant, c) for c in cands])
            pred = model.predict(X)
            for j in np.argsort(pred)[: int(k)]:
                out.append(WarmStart(dict(cands[int(j)]), "selector"))
        seen, uniq = set(), []
        for w in out:
            key = json.dumps(dict(w.config), sort_keys=True)
            if key not in seen:
                seen.add(key)
                uniq.append(w)
        return uniq
