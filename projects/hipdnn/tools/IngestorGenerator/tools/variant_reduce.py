# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reduce a variant selection to the kernels its autotune rankings justify.

Keeps each feature cell's floor (declared default plus the divisibility antichain of
its ``divides`` knobs), one kernel for every ranked graph left uncovered, and each
graph's best kernel while its gain over the best kept kernel exceeds epsilon: the
smallest grid value at or above the round-to-round noise floor that fits the budget.
Rankings are the ``WinnerCacheFile.hpp`` shards of the successful timing phases of a
``tools/sweep.py`` output. Exit 1: the kept set (eps or ``--apply-rules``) exceeds
the budget, or the applied rules together lose more of a graph's best time than the
rules eps; exit 2: unusable input; exit 3: the sweep cannot support a reduction: no
``summary.json`` listing ``expected_phases``, an expected timing or correctness phase
of the arm with no record or (timing) no winners, a failed correctness verdict, or timing
phases missing for some (round, corpus). On exit 1, and on exit 3 for a failed
correctness verdict or missing (round, corpus), the report is written and ``--out`` is
not; an incomplete sweep (the other exit 3 causes) writes neither. A sweep of fewer
than 2 timing rounds warns on stdout: its noise floor is assumed, not measured. ``reduction.single_round_epsilon``
(default 0.05) and ``reduction.drift_multiplier`` (default 2) set the noise floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_GENERATOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_GENERATOR_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codegen.config_loader import (  # noqa: E402
    DEFAULT_AOT_BUDGET_PER_ARCH,
    DEFAULT_DRIFT_MULTIPLIER,
    DEFAULT_SINGLE_ROUND_EPSILON,
)
from codegen.models import EVERY_ARCH  # noqa: E402

# WinnerCacheFile.hpp detail::WINNER_LINE_*: line, device and entry fields; version.
_LINE_FIELDS = frozenset({"v", "graph", "device", "entries"})
_DEVICE_FIELDS = frozenset({"gcn_arch_name", "warp_size", "multi_processor_count"})
_ENTRY_FIELDS = frozenset({"kernel_id", "pack_id", "dispatch_id", "time_ms"})
_LINE_FORMAT_VERSION = 1

EPSILON_GRID = tuple(i / 100 for i in range(51)) + (math.inf,)
# A healthy elapsed may be 0 (BenchmarkPlan.hpp); ratios use this floor instead.
_MIN_TIME_MS = 1e-6


class ReduceError(RuntimeError):
    """The inputs cannot be reduced as given; exit 2."""


class IncompleteSweep(ReduceError):
    """The sweep did not finish the phases this arm's reduction rests on; exit 3."""


@dataclass(frozen=True)
class Classification:
    feature: tuple
    performance: tuple
    divides: tuple

    @classmethod
    def from_reduction(cls, reduction: dict) -> "Classification":
        fields = reduction["fields"]
        knobs = list(fields.get("performance") or [])
        return cls(
            feature=tuple(fields.get("feature") or []),
            performance=tuple(k["name"] for k in knobs),
            divides=tuple(k["name"] for k in knobs if k.get("divides")),
        )


@dataclass(frozen=True)
class Variant:
    name: str
    cell: tuple
    knobs: dict = field(hash=False, compare=False)
    default: bool = False
    arches: tuple = ()


@dataclass
class Ranking:
    """One graph's winner record in one round: entries best first."""

    round: int
    corpus: str
    key: str
    entries: list
    cell: tuple

    @property
    def label(self) -> str:
        return hashlib.sha256(self.key.encode()).hexdigest()[:12]

    @property
    def best_time(self) -> float:
        return self.entries[0][1]


@dataclass
class Step:
    name: str
    gain: float
    ranking: Ranking


@dataclass
class NoiseFloor:
    rounds: int
    epsilon_min: float
    drift: float | None = None
    matched_keys: int = 0
    unmatched_keys: int = 0
    missing_pairs: list = field(default_factory=list)


@dataclass
class Rule:
    id: str
    text: str
    removed: frozenset
    graphs_losing_best: int
    worst_gain_lost: float
    cells_left_at_floor: list


@dataclass
class Correctness:
    phase: dict
    verdict: str  # "passed", "failed" or "no reference"
    winners: set


@dataclass
class Reduction:
    classification: Classification
    variants: dict
    floor: dict
    no_default_cells: list
    rankings: list
    noise: NoiseFloor
    coverage: dict
    steps: list
    budget: int
    epsilon: float | None
    rules_epsilon: float
    rules: list
    correctness: list
    arm: str
    duplicate_keys: int = 0
    empty_records: int = 0

    def keep(self, epsilon: float) -> set:
        return keep_at(self.floor, self.coverage, self.steps, epsilon)


def _phase_attempts(sweep_dir, arm=None) -> list:
    """``(attempt_dir, record)`` for the latest attempt of every phase key."""
    attempts = Path(sweep_dir) / "attempts"
    if not attempts.is_dir():
        raise ReduceError(f"{sweep_dir}: not a sweep output (no 'attempts' directory)")
    latest = {}
    for path in sorted(attempts.glob("*/phase.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            phase = record["phase"]
            identity = json.dumps(
                {k: phase[k] for k in ("kind", "corpus", "arm", "round")},
                sort_keys=True,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ReduceError(f"{path}: not a sweep phase record ({exc})") from exc
        if arm is not None and phase["arm"] != arm:
            continue
        session = record.get("session") or {}
        order = (str(session.get("started_at") or ""), path.stat().st_mtime_ns)
        if identity not in latest or order >= latest[identity][0]:
            latest[identity] = (order, path.parent, record)
    return [
        (attempt, record)
        for _, (_, attempt, record) in sorted(latest.items(), key=lambda kv: kv[0])
    ]


def load_phase_records(sweep_dir, arm=None) -> list:
    """The ``phase.json`` records of a sweep output, the latest per phase key
    (``kind``, ``corpus``, ``arm``, ``round``), optionally of one arm."""
    return [record for _, record in _phase_attempts(sweep_dir, arm)]


def _require_fields(obj, expected: frozenset, where: str) -> None:
    if not isinstance(obj, dict):
        raise ReduceError(f"{where}: expected a JSON object")
    extra = sorted(set(obj) - expected)
    if extra:
        raise ReduceError(f"{where}: {extra} not in WinnerCacheFile.hpp")
    missing = sorted(expected - set(obj))
    if missing:
        raise ReduceError(f"{where}: missing {missing}")


def read_shard(path) -> dict:
    """``{key: [entry, ...]}`` for one winner-cache shard; the last line per key wins.

    Line 1 is the version. Lines split as ``LineStore.hpp`` ``splitLineStoreLines``:
    on ``\\n``, dropping a preceding ``\\r``, ignoring an unterminated last chunk.
    """
    path = Path(path)
    records = {}
    chunks = path.read_text(encoding="utf-8").split("\n")
    lines = [chunk.removesuffix("\r") for chunk in chunks[:-1]]
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            doc = json.loads(line)
        except ValueError as exc:
            raise ReduceError(f"{where}: not JSON ({exc})") from exc
        if not isinstance(doc, dict):
            raise ReduceError(f"{where}: expected a JSON object")
        version = doc.get("v")
        if type(version) is not int or version != _LINE_FORMAT_VERSION:
            raise ReduceError(f"{where}: line format v={version!r}, expected 1")
        _require_fields(doc, _LINE_FIELDS, where)
        _require_fields(doc["device"], _DEVICE_FIELDS, f"{where} device")
        if not isinstance(doc["entries"], list):
            raise ReduceError(f"{where}: 'entries' must be a list")
        for index, entry in enumerate(doc["entries"]):
            _require_fields(entry, _ENTRY_FIELDS, f"{where} entries[{index}]")
            t = entry["time_ms"]
            if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
                raise ReduceError(f"{where} entries[{index}]: bad time_ms {t!r}")
        key = json.dumps([doc["graph"], doc["device"]], sort_keys=True)
        records[key] = doc["entries"]
    return records


def _shards(attempt: Path) -> list:
    return sorted((attempt / "winners").rglob("winners.jsonl"))


def load_selection(path):
    """``(raw, config, classification)``; requires a ``reduction:`` block."""
    import yaml
    from codegen.config_loader import ConfigError, load_config, read_yaml

    try:
        raw = read_yaml(Path(path))
        config = load_config(Path(path))
    except (OSError, ConfigError, yaml.YAMLError) as exc:
        raise ReduceError(f"{path}: {exc}") from exc
    if not config.reduction:
        raise ReduceError(f"{path}: no 'reduction' block")
    if len(config.packs) != 1:
        raise ReduceError(f"{path}: {len(config.packs)} packs, expected 1")
    return raw, config, Classification.from_reduction(config.reduction)


def load_kdp(path) -> dict:
    try:
        kdp = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = [k for k in ("id", "dispatch", "kernelDescriptors") if k not in kdp]
        if missing:
            raise KeyError(", ".join(missing))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReduceError(f"{path}: not a KDP ({exc})") from exc
    return kdp


def build_variants(config, kdp: dict, classification: Classification) -> dict:
    """``{name: Variant}`` in selection order; fields from KDP metadata, then spec."""
    pack = config.packs[0]
    selected = {k.name: k for k in pack.kernels}
    installed = {d["name"]: d for d in kdp["kernelDescriptors"]}
    if set(selected) != set(installed):
        raise ReduceError(
            f"the KDP is not built from this selection: only in the selection "
            f"{sorted(set(selected) - set(installed))[:3]}, only in the KDP "
            f"{sorted(set(installed) - set(selected))[:3]}"
        )
    variants = {}
    for name, kernel in selected.items():
        metadata = installed[name].get("metadata") or {}
        spec = kernel.kernel_source.spec

        def lookup(f):
            return metadata[f] if f in metadata else spec.get(f)

        cell = tuple(lookup(f) for f in classification.feature)
        knobs = {k: lookup(k) for k in classification.performance}
        arches = tuple(kernel.arch or pack.arch or [EVERY_ARCH])
        variants[name] = Variant(name, cell, knobs, kernel.default, arches)
    return variants


def instances(variants: dict, names) -> int:
    """Kept kernels for the most-populated arch: what the budget bounds."""
    per_arch = Counter(arch for name in names for arch in variants[name].arches)
    return max(per_arch.values(), default=0)


def _divisibility_antichain(peers: list, divides: tuple) -> list:
    """The peers whose ``divides`` values no other peer's values each divide."""

    def values(variant):
        out = tuple(variant.knobs.get(k) for k in divides)
        for knob, value in zip(divides, out):
            if type(value) is not int or value <= 0:
                raise ReduceError(
                    f"{variant.name}: {knob}={value!r}, not a positive int"
                )
        return out

    tuples = {v.name: values(v) for v in peers}
    present = set(tuples.values())
    return [
        v
        for v in peers
        if not any(
            other != tuples[v.name]
            and all(mine % theirs == 0 for mine, theirs in zip(tuples[v.name], other))
            for other in present
        )
    ]


def floor_variants(variants, classification: Classification):
    """``({name: reason}, cells_without_default)``: per cell, the defaults plus the
    antichain of the variants sharing a default's non-divides knobs (of all the
    cell's variants when it declares no default)."""
    by_cell = defaultdict(list)
    for variant in variants.values():
        by_cell[variant.cell].append(variant)
    others = [k for k in classification.performance if k not in classification.divides]
    kept = {}
    no_default = []
    for cell, members in by_cell.items():
        defaults = [v for v in members if v.default]
        for d in defaults:
            kept.setdefault(d.name, "floor: declared default")
        if not defaults:
            no_default.append(cell)
            groups = [members]
        else:
            groups = [
                [v for v in members if all(v.knobs[k] == d.knobs[k] for k in others)]
                for d in defaults
            ]
        for peers in groups:
            for v in _divisibility_antichain(peers, classification.divides):
                kept.setdefault(v.name, "floor: divisibility antichain")
    return kept, no_default


def _best_kept(ranking: Ranking, kept) -> float | None:
    times = [t for name, t in ranking.entries if name in kept]
    return min(times) if times else None


def search(rankings: list, base) -> tuple:
    """``(coverage, steps)`` from the floor ``base``: coverage adds the best entry of
    every ranking holding no kept kernel; each step then adds the best entry of the
    ranking with the largest ``t(best kept) / t(best) - 1`` (non-increasing)."""
    kept = set(base)
    coverage = {}
    for ranking in rankings:
        if _best_kept(ranking, kept) is None:
            name = ranking.entries[0][0]
            kept.add(name)
            coverage.setdefault(name, ranking)
    steps = []
    while True:
        largest = None
        for ranking in rankings:
            gain = _best_kept(ranking, kept) / ranking.best_time - 1
            candidate = ranking.entries[0][0]
            if (
                largest is None
                or gain > largest[0]
                or (gain == largest[0] and candidate < largest[1])
            ):
                largest = (gain, candidate, ranking)
        if largest is None or largest[0] <= 0:
            return coverage, steps
        gain, name, ranking = largest
        kept.add(name)
        steps.append(Step(name, gain, ranking))


def keep_at(floor, coverage, steps, epsilon: float) -> set:
    return set(floor) | set(coverage) | {s.name for s in steps if s.gain > epsilon}


def choose_epsilon(curve: list, epsilon_min: float, budget: int):
    """The smallest ``(epsilon, instances)`` epsilon >= epsilon_min within budget."""
    for epsilon, count in curve:
        if epsilon >= epsilon_min and count <= budget:
            return epsilon
    return None


def noise_floor(
    rounds: dict,
    missing_pairs: list,
    single_round_epsilon: float,
    drift_multiplier: float,
) -> NoiseFloor:
    """Median over graphs ranked in 2+ rounds of each kernel's ``(max - min) /
    median`` of ``t / t_best`` across rounds, times ``drift_multiplier``;
    ``rounds`` is ``{round: {key: Ranking}}``. One round has no drift and assumes
    ``single_round_epsilon``.
    """
    count = len(rounds)
    if count <= 1:
        return NoiseFloor(count, single_round_epsilon, missing_pairs=missing_pairs)
    presence = Counter(key for keyed in rounds.values() for key in keyed)
    matched = [key for key, n in presence.items() if n >= 2]
    if not matched:
        raise ReduceError(f"rounds share no graphs across {count} rounds")
    spreads = []
    for key in matched:
        ratios = defaultdict(list)
        for keyed in rounds.values():
            if key in keyed:
                ranking = keyed[key]
                for name, t in ranking.entries:
                    ratios[name].append(t / ranking.best_time)
        for values in ratios.values():
            if len(values) >= 2:
                spreads.append((max(values) - min(values)) / statistics.median(values))
    if not spreads:
        raise ReduceError("matched graphs share no ranked kernel across rounds")
    drift = statistics.median(spreads)
    unmatched = len(presence) - len(matched)
    return NoiseFloor(
        count,
        drift_multiplier * drift,
        drift,
        len(matched),
        unmatched,
        missing_pairs,
    )


def _cell_text(cell: tuple, classification: Classification, varying: list) -> str:
    shown = zip(classification.feature, cell)
    return "{" + ", ".join(f"{n}={v}" for n, v in shown if n in varying) + "}"


def propose_rules(variants: dict, rankings: list, protected, classification, epsilon):
    """``drop knob X`` / ``drop X=v`` / ``keep X=v only for <cells>`` rules, ordered
    by instance saving per unit of worst gain lost. A non-default value wins a ranking
    when its best entry beats every default-valued entry (every other value in a cell
    without a default) by more than epsilon, or no such entry exists. A rule never
    removes a ``protected`` kernel (floor and coverage), so any combination of rules
    keeps a kernel for every ranked graph."""
    defaults = defaultdict(lambda: defaultdict(set))
    for v in variants.values():
        if v.default:
            for knob, value in v.knobs.items():
                defaults[v.cell][knob].add(value)

    def non_default(variant, knob) -> bool:
        declared = defaults.get(variant.cell)
        return not declared or variant.knobs[knob] not in declared[knob]

    wins = defaultdict(set)
    for ranking in rankings:
        declared = defaults.get(ranking.cell)
        for knob in classification.performance:
            best_by_value = {}
            for name, t in ranking.entries:
                value = variants[name].knobs[knob]
                best_by_value[value] = min(t, best_by_value.get(value, math.inf))
            default_values = declared[knob] if declared else set()
            for value, t_value in best_by_value.items():
                if value in default_values:
                    continue
                comparators = [
                    t
                    for other, t in best_by_value.items()
                    if (other in default_values if declared else other != value)
                ]
                if not comparators or min(comparators) / t_value - 1 > epsilon:
                    wins[(knob, value)].add(ranking.cell)

    cells = sorted({v.cell for v in variants.values()}, key=repr)
    varying = [
        name
        for index, name in enumerate(classification.feature)
        if len({cell[index] for cell in cells}) > 1
    ]
    # (text, knob, values removed (empty: every non-default value), exempt cells)
    candidates = []
    for knob in classification.performance:
        values = defaultdict(set)
        for v in variants.values():
            if non_default(v, knob):
                values[v.knobs[knob]].add(v.cell)
        if not values:
            continue
        if not any(wins.get((knob, value)) for value in values):
            candidates.append((f"drop knob {knob}", knob, (), frozenset()))
            continue
        for value in sorted(
            values, key=lambda x: (x is None, type(x).__name__, repr(x))
        ):
            won = wins.get((knob, value), set())
            if not won:
                text = f"drop {knob}={value}"
                candidates.append((text, knob, (value,), frozenset()))
            elif values[value] - won:
                shown = ", ".join(
                    _cell_text(c, classification, varying)
                    for c in sorted(won, key=repr)
                )
                text = f"keep {knob}={value} only for {shown}"
                candidates.append((text, knob, (value,), frozenset(won)))

    rules = []
    for text, knob, only, exempt in candidates:
        removed = frozenset(
            v.name
            for v in variants.values()
            if non_default(v, knob)
            and (not only or v.knobs[knob] in only)
            and v.cell not in exempt
            and v.name not in protected
        )
        if removed:
            rules.append(_costed(text, removed, variants, rankings, protected))
    rules.sort(key=_rule_order)
    return rules


def _rule_order(rule: Rule) -> tuple:
    """Lossless rules first, by saving; then by saving per unit of worst gain lost."""
    w, saved = rule.worst_gain_lost, len(rule.removed)
    return (w > 0, -(saved / w) if w > 0 else -saved, rule.text)


def _rule_id(text: str) -> str:
    """Derived from the rule text alone: an id is stable for a rule's text. The text
    of ``drop`` rules does not move with ``--budget``; a ``keep ... only for <cells>``
    rule names the cells where the value wins by more than the budget-chosen eps, so
    its text, and its id, changes with ``--budget``."""
    return "R" + hashlib.sha256(text.encode()).hexdigest()[:6]


def _ranking_loss(removed, rankings) -> tuple:
    """``(sorted keys of the graphs whose best kernel is in removed, worst gain
    lost)``: the largest ``t(best remaining) / t(best) - 1`` over the rankings."""
    losing = {r.key for r in rankings if r.entries[0][0] in removed}
    worst = 0.0
    for ranking in rankings:
        remaining = [t for name, t in ranking.entries if name not in removed]
        gain = min(remaining) / ranking.best_time - 1 if remaining else math.inf
        worst = max(worst, gain)
    return sorted(losing), worst


def _costed(text, removed, variants, rankings, protected) -> Rule:
    losing, worst = _ranking_loss(removed, rankings)
    before = defaultdict(set)
    for v in variants.values():
        if v.name not in protected:
            before[v.cell].add(v.name)
    at_floor = sorted(
        (cell for cell, names in before.items() if names <= removed), key=repr
    )
    return Rule(_rule_id(text), text, removed, len(losing), worst, at_floor)


def apply_rules(variants: dict, rules: list, chosen: list) -> list:
    """The selection's kernel names, in order, left after ``chosen`` rule ids."""
    by_id = {r.id: r for r in rules}
    unknown = [r for r in chosen if r not in by_id]
    if unknown:
        raise ReduceError(
            f"--apply-rules {unknown} not proposed: {list(by_id)}; ids come from the "
            "rule text, so a rule the sweep or --budget no longer proposes is gone"
        )
    removed = set().union(*(by_id[r].removed for r in chosen))
    return [name for name in variants if name not in removed]


def _correctness_verdict(record: dict) -> str:
    """``passed``; ``no reference`` when the only failures are served graphs
    without a usable reference (``sweep.py`` ``gates.reference``); else ``failed``."""
    ledger = record.get("ledger") or []
    served = [e for e in ledger if e.get("correctness") is not None]
    compared = [
        e for e in served if (e.get("reference") or {}).get("usable") is not False
    ]
    compared_ok = all(
        isinstance(e["correctness"], dict) and e["correctness"].get("passed") is True
        for e in compared
    )
    unreferenced = len(compared) < len(served)
    if (
        record.get("status") == "success"
        and served
        and not unreferenced
        and compared_ok
    ):
        return "passed"
    failed_gates = {g for g, ok in (record.get("gates") or {}).items() if not ok}
    if unreferenced and compared_ok and failed_gates <= {"reference", "correctness"}:
        return "no reference"
    return "failed"


def _rankings_and_correctness(attempts, kdp, variants, arm):
    by_id = {str(d["id"]).lower(): d["name"] for d in kdp["kernelDescriptors"]}
    pack_id = str(kdp["id"]).lower()
    dispatch_id = str(kdp["dispatch"]).lower()

    def entries_of(raw, where):
        out = []
        for entry in raw:
            ids = (str(entry["pack_id"]).lower(), str(entry["dispatch_id"]).lower())
            name = by_id.get(str(entry["kernel_id"]).lower())
            if ids != (pack_id, dispatch_id) or name is None:
                raise ReduceError(f"stale ranking: {where}: {entry} not in the KDP")
            out.append((name, max(float(entry["time_ms"]), _MIN_TIME_MS)))
        return sorted(out, key=lambda e: (e[1], e[0]))

    timing = defaultdict(list)
    correctness = []
    for attempt, record in attempts:
        phase = record["phase"]
        if phase["arm"] != arm:
            continue
        if phase["kind"] == "timing" and record.get("status") == "success":
            timing[phase["round"]].append((phase["corpus"], attempt))
        elif phase["kind"] == "correctness":
            verdict = _correctness_verdict(record)
            winners = set()
            for shard in _shards(attempt):
                for raw in read_shard(shard).values():
                    entries = entries_of(raw, str(shard))
                    if entries:
                        winners.add(entries[0][0])
            correctness.append(Correctness(phase, verdict, winners))

    rounds = {}
    duplicates = 0
    empty = 0
    for number in sorted(timing):
        keyed = {}
        for corpus, attempt in sorted(timing[number], key=lambda x: x[0]):
            for shard in _shards(attempt):
                for key, raw in read_shard(shard).items():
                    entries = entries_of(raw, str(shard))
                    if not entries:
                        empty += 1
                        continue
                    cells = {variants[name].cell for name, _ in entries}
                    if len(cells) != 1:
                        raise ReduceError(f"{shard}: a record spans {len(cells)} cells")
                    if key in keyed:
                        duplicates += 1
                    keyed[key] = Ranking(number, corpus, key, entries, cells.pop())
        rounds[number] = keyed
    return rounds, correctness, duplicates, empty


def _missing_pairs(attempts, arm) -> list:
    timing = [(r["phase"], r.get("status")) for _, r in attempts]
    timing = [(p, s) for p, s in timing if p["kind"] == "timing" and p["arm"] == arm]
    corpora = sorted({p["corpus"] for p, _ in timing})
    numbers = sorted({p["round"] for p, _ in timing})
    present = {(p["round"], p["corpus"]) for p, s in timing if s == "success"}
    return [(n, c) for n in numbers for c in corpora if (n, c) not in present]


#: The keys every ``expected_phases`` item of ``summary.json`` carries.
_PHASE_KEYS = frozenset({"kind", "arm", "corpus", "round"})


def _require_complete_arm(sweep_dir, attempts, arm) -> None:
    """Raise `IncompleteSweep` unless ``sweep.py`` finished every timing and
    correctness phase of ``arm`` and each timing phase ranked something. The
    expected phases are the sweep's own list in ``summary.json``, which it writes
    only after its last phase; another arm's phases are not this arm's evidence."""
    summary_path = Path(sweep_dir) / "summary.json"
    try:
        expected = json.loads(summary_path.read_text(encoding="utf-8"))[
            "expected_phases"
        ]
        if not isinstance(expected, list):
            raise TypeError("expected_phases is not a list")
        for index, phase in enumerate(expected):
            if not isinstance(phase, dict) or not _PHASE_KEYS <= phase.keys():
                raise TypeError(
                    f"expected_phases[{index}] is not a mapping with "
                    f"{', '.join(sorted(_PHASE_KEYS))}"
                )
            hash((phase["corpus"], phase["round"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IncompleteSweep(
            f"{summary_path}: missing, or its 'expected_phases' is unusable ({exc}): "
            "the sweep did not finish, so which phases it owed is unknown"
        ) from exc
    if not any(p["kind"] == "timing" and p["arm"] == arm for p in expected):
        raise IncompleteSweep(
            f"{summary_path}: lists no timing phase for arm {arm}, so the sweep's "
            "claim to have finished it cannot be checked"
        )
    found = {
        (r["phase"]["kind"], r["phase"]["corpus"], r["phase"]["round"]): attempt
        for attempt, r in attempts
        if r["phase"]["arm"] == arm
    }
    problems = []
    for phase in expected:
        if phase["arm"] != arm or phase["kind"] not in (
            "timing",
            "correctness",
        ):
            continue
        label = f"{phase['kind']} {phase['corpus']} round {phase['round']}"
        attempt = found.get((phase["kind"], phase["corpus"], phase["round"]))
        if attempt is None:
            problems.append(f"{label}: no record")
        elif phase["kind"] == "timing" and not any(
            read_shard(shard) for shard in _shards(attempt)
        ):
            problems.append(f"{label}: no winners")
    if problems:
        raise IncompleteSweep(
            f"arm {arm}: the sweep is incomplete: {'; '.join(problems)}"
        )


def analyse(selection, kdp_path, sweep_dir, arm=None, budget=None) -> Reduction:
    raw, config, classification = load_selection(selection)
    kdp = load_kdp(kdp_path)
    variants = build_variants(config, kdp, classification)
    attempts = _phase_attempts(sweep_dir)
    arms = sorted(
        {r["phase"]["arm"] for _, r in attempts if r["phase"]["kind"] == "timing"}
    )
    if not arms:
        raise ReduceError(f"{sweep_dir}: no timing phase")
    if arm is None:
        if len(arms) != 1:
            raise ReduceError(f"timing arms {arms}: name one with --arm")
        arm = arms[0]
    elif arm not in arms:
        raise ReduceError(f"--arm {arm} ran no timing phase; arms: {arms}")
    _require_complete_arm(sweep_dir, attempts, arm)
    rounds, correctness, duplicates, empty = _rankings_and_correctness(
        attempts, kdp, variants, arm
    )
    if not any(rounds.values()):
        raise ReduceError(f"arm {arm} has no successful timing phase with a ranking")
    noise = noise_floor(
        rounds,
        _missing_pairs(attempts, arm),
        config.reduction.get("single_round_epsilon", DEFAULT_SINGLE_ROUND_EPSILON),
        config.reduction.get("drift_multiplier", DEFAULT_DRIFT_MULTIPLIER),
    )
    rankings = [
        ranking
        for number in sorted(rounds)
        for ranking in sorted(rounds[number].values(), key=lambda r: r.key)
    ]
    floor, no_default = floor_variants(variants, classification)
    coverage, steps = search(rankings, floor)
    if budget is None:
        budget = config.reduction.get(
            "aot_budget_per_arch", DEFAULT_AOT_BUDGET_PER_ARCH
        )
    curve = [
        (e, instances(variants, keep_at(floor, coverage, steps, e)))
        for e in EPSILON_GRID
    ]
    epsilon = choose_epsilon(curve, noise.epsilon_min, budget)
    floor_eps = next(e for e in EPSILON_GRID if e >= noise.epsilon_min)
    rules_epsilon = floor_eps if epsilon is None else epsilon
    protected = set(floor) | set(coverage)
    rules = propose_rules(variants, rankings, protected, classification, rules_epsilon)
    return Reduction(
        classification=classification,
        variants=variants,
        floor=floor,
        no_default_cells=no_default,
        rankings=rankings,
        noise=noise,
        coverage=coverage,
        steps=steps,
        budget=budget,
        epsilon=epsilon,
        rules_epsilon=rules_epsilon,
        rules=rules,
        correctness=correctness,
        arm=arm,
        duplicate_keys=duplicates,
        empty_records=empty,
    )


def write_selection(selection, names, out) -> int:
    """Write the selection restricted to ``names``, factorised and round-trip
    checked; returns its line count."""
    import factorise_config
    from codegen.config_loader import ConfigError

    raw, config, classification = load_selection(selection)
    keep = set(names)
    pack_raw = raw["packs"][0]
    kernels = []
    for kernel in config.packs[0].kernels:
        if kernel.name not in keep:
            continue
        entry = {
            "name": kernel.name,
            "kernel_source": kernel.kernel_source.authored_document(),
            "metadata": dict(kernel.metadata),
        }
        if kernel.arch:
            entry["arch"] = list(kernel.arch)
        if kernel.priority:
            entry["priority"] = kernel.priority
        if kernel.default:
            entry["default"] = True
        kernels.append(entry)
    expanded_pack = {"name": pack_raw["name"]}
    for key in ("arch", "discriminator"):
        if pack_raw.get(key):
            expanded_pack[key] = pack_raw[key]
    expanded_pack["kernels"] = kernels
    expanded = {k: v for k, v in raw.items() if k != "packs"}
    expanded["packs"] = [expanded_pack]
    # Knobs only where every kept kernel's metadata carries them: the loader
    # rejects a group's `policy_knobs` naming a field its `metadata` lacks.
    knobs = [
        knob
        for knob in classification.performance
        if all(knob in entry["metadata"] for entry in kernels)
    ]
    try:
        compact = factorise_config.factorise(
            expanded,
            knobs,
            (config.specialization or {}).get("vocabulary") or {},
        )
        factorise_config._round_trip(expanded, compact)
    except (factorise_config.FactoriseError, ConfigError) as exc:
        raise ReduceError(f"the reduced selection does not factorise: {exc}") from exc
    text = factorise_config.dump(compact)
    Path(out).write_text(text, encoding="utf-8")
    return len(text.splitlines())


def _eps(value, digits=2) -> str:
    return "inf" if value == math.inf else f"{value:.{digits}f}"


def _keep_reason(reduction: Reduction, name: str, epsilon: float) -> str:
    if name in reduction.floor:
        return reduction.floor[name]
    if name in reduction.coverage:
        r = reduction.coverage[name]
        return f"coverage: only kept kernel of graph {r.label} (r{r.round} {r.corpus})"
    step = next(
        (s for s in reduction.steps if s.name == name and s.gain > epsilon), None
    )
    if step is None:
        # Only an applied rule set keeps a kernel the eps search would not.
        return "no applied rule removes it"
    r = step.ranking
    return f"perf: graph {r.label} (r{r.round} {r.corpus}) gain {step.gain:.4f}"


def _drop_reason(name: str, rankings: list, kept: set) -> str:
    """The largest ``t(best kept) / t(name) - 1`` over the rankings holding it."""
    gains = [
        _best_kept(r, kept) / t - 1 for r in rankings for n, t in r.entries if n == name
    ]
    return f"max gain {max(gains):.4f} <= eps" if gains else "never ranked"


def _correctness_warning(r: Reduction) -> str | None:
    failed = [c.phase["corpus"] for c in r.correctness if c.verdict == "failed"]
    if not failed:
        return None
    return (
        f"correctness attempts failed for corpora {failed}; do not apply this "
        "reduction until they are investigated"
    )


def _sweep_warning(noise: NoiseFloor) -> str | None:
    if not noise.missing_pairs:
        return None
    return (
        f"timing phases missing for (round, corpus) {noise.missing_pairs}: the sweep "
        "is partial; do not apply this reduction until it is rerun"
    )


def _single_round_warning(noise: NoiseFloor) -> str | None:
    if noise.rounds >= 2:
        return None
    return (
        f"{noise.rounds} timing round: eps_min is assumed {noise.epsilon_min}, "
        "not measured; sweep at least 2 rounds"
    )


def _graph_at_device(key: str) -> str:
    """``graph@device`` for one ranking key, a JSON ``[graph, device]`` pair."""
    graph, device = json.loads(key)

    def label(doc, field):
        shown = doc.get(field) if isinstance(doc, dict) else None
        return shown if isinstance(shown, str) else json.dumps(doc, sort_keys=True)

    return f"{label(graph, 'key')}@{label(device, 'gcn_arch_name')}"


def _applied_loss(applied, rankings) -> tuple:
    """`_ranking_loss` of the union of the ``applied`` rules' removed kernels."""
    return _ranking_loss(set().union(*(rule.removed for rule in applied)), rankings)


def render_report(r: Reduction, selection, output_note: str, names, applied=()) -> str:
    """The report for the kernel set ``names`` that ``main`` writes, not whichever
    set eps alone would keep. ``applied`` are the ``--apply-rules`` rules behind it."""
    eps = r.epsilon if r.epsilon is not None else math.inf
    kept = set(names)
    noise = r.noise
    search = _eps(eps) if r.epsilon is not None else "none (over budget)"
    if applied:
        ids = ",".join(rule.id for rule in applied)
        chosen = f"chosen: rules {ids} (the eps search chose {search}; not used)"
    else:
        chosen = f"chosen eps: {search}"
    lines = [
        "# variant_reduce report",
        "",
        f"- selection `{selection}`, arm `{r.arm}`; output: {output_note}",
        f"- budget {r.budget} instances per arch; {chosen}",
        f"- kept {len(kept)} of {len(r.variants)} "
        f"({instances(r.variants, kept)} instances per arch)",
        f"- rounds {noise.rounds}, ranked graph records {len(r.rankings)}",
    ]
    for warning in (
        _correctness_warning(r),
        _sweep_warning(noise),
        _single_round_warning(noise),
    ):
        if warning:
            lines.append(f"- WARNING: {warning}")
    if applied:
        losing, worst = _applied_loss(applied, r.rankings)
        lines.append(
            f"- applied rules together: worst gain lost {_eps(worst, 4)} (bound: "
            f"rules eps {_eps(r.rules_epsilon)}); graphs losing best {len(losing)}"
            + (f": {'; '.join(map(_graph_at_device, losing))}" if losing else "")
        )
    if noise.rounds > 1:
        lines.append(
            f"- drift {noise.drift:.6f}, eps_min {noise.epsilon_min:.6f}; graph keys "
            f"matched {noise.matched_keys}, unmatched {noise.unmatched_keys}"
        )
    if noise.missing_pairs:
        lines.append(f"- missing (round, corpus): {noise.missing_pairs}")
    if r.duplicate_keys:
        lines.append(f"- keys ranked by several corpora in a round: {r.duplicate_keys}")
    if r.empty_records:
        lines.append(f"- records with no usable entry: {r.empty_records}")

    lines += ["", "## Eps curve", "", "| eps | kept |", "|---|---|"]
    for e in EPSILON_GRID:
        mark = " (chosen)" if e == r.epsilon and not applied else ""
        lines.append(f"| {_eps(e)}{mark} | {len(r.keep(e))} |")

    checked = {n for c in r.correctness if c.verdict == "passed" for n in c.winners}
    lines += ["", "## Variants", "", "| variant | kept | reason |", "|---|---|---|"]
    for name in r.variants:
        if name in kept:
            tag = "numerics checked" if name in checked else "numerics unchecked"
            reason = f"{_keep_reason(r, name, eps)}; {tag}"
        else:
            applied_ids = [rule.id for rule in applied if name in rule.removed]
            reason = (
                f"removed by {', '.join(applied_ids)}"
                if applied_ids
                else _drop_reason(name, r.rankings, kept)
            )
        lines.append(f"| {name} | {'yes' if name in kept else 'no'} | {reason} |")

    cells = sorted({v.cell for v in r.variants.values()}, key=repr)
    ranked = {x.cell for x in r.rankings}
    silent = [cell for cell in cells if cell not in ranked]
    lines += [
        "",
        f"## Feature cells ({', '.join(r.classification.feature)})",
        "",
        f"- floor only, never ranked: {len(silent)} of {len(cells)}",
        *(f"  - {cell}" for cell in silent),
        f"- no declared default: {len(r.no_default_cells)}",
        *(f"  - {cell}" for cell in r.no_default_cells),
        "",
        f"## Proposed rules (eps {_eps(r.rules_epsilon)}; never remove the floor "
        "or coverage)",
        "",
        "| id | rule | saved | graphs losing best | worst gain lost | cells at floor |",
        "|---|---|---|---|---|---|",
    ]
    for rule in r.rules:
        lines.append(
            f"| {rule.id} | {rule.text} | {len(rule.removed)} | "
            f"{rule.graphs_losing_best} | {_eps(rule.worst_gain_lost, 4)} | "
            f"{len(rule.cells_left_at_floor)} |"
        )

    lines += ["", "## Correctness attempts", ""]
    for c in r.correctness:
        verdict = "FAILED; investigate" if c.verdict == "failed" else c.verdict
        lines.append(f"- {c.phase['corpus']}: {verdict}, {len(c.winners)} winners")
    if not r.correctness:
        lines.append("- none for this arm")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", required=True, help="selection YAML")
    parser.add_argument("--kdp", required=True, help="installed KDP of the selection")
    parser.add_argument("--sweep-output", required=True, help="tools/sweep.py output")
    parser.add_argument("--arm", help="sweep arm (default: the only one)")
    parser.add_argument("--out", required=True, help="reduced selection to write")
    parser.add_argument("--report", required=True, help="report to write")
    parser.add_argument("--budget", type=int, help="instances per arch")
    parser.add_argument(
        "--apply-rules",
        default="",
        help="rule ids from the report's rule table (R and six hex digits, derived "
        "from the rule text) to apply instead of eps",
    )
    args = parser.parse_args(argv)
    chosen = [r.strip() for r in args.apply_rules.split(",") if r.strip()]
    report = Path(args.report)
    try:
        reduction = analyse(
            args.selection, args.kdp, args.sweep_output, args.arm, args.budget
        )
        blockers = [
            w
            for w in (
                _correctness_warning(reduction),
                _sweep_warning(reduction.noise),
            )
            if w
        ]
        for warning in (*blockers, _single_round_warning(reduction.noise)):
            if warning:
                print(f"WARNING: {warning}")
        applied = [rule for rule in reduction.rules if rule.id in chosen]
        if chosen:
            names = apply_rules(reduction.variants, reduction.rules, chosen)
            what = f"rules {','.join(chosen)}"
        elif reduction.epsilon is not None:
            names = sorted(reduction.keep(reduction.epsilon))
            what = f"eps {_eps(reduction.epsilon)}"
        else:
            names, what = sorted(reduction.keep(math.inf)), "floor + coverage"
        if blockers:
            note = "not written: the sweep cannot support a reduction (see WARNING)"
            report.write_text(
                render_report(reduction, args.selection, note, names, applied), "utf-8"
            )
            print(
                f"FAIL: {what} not written: {'; '.join(blockers)}; report {report}",
                file=sys.stderr,
            )
            return 3
        if applied:
            losing, worst = _applied_loss(applied, reduction.rankings)
            if worst > reduction.rules_epsilon:
                note = (
                    "not written: the applied rules together lose more than the "
                    "rules eps"
                )
                report.write_text(
                    render_report(reduction, args.selection, note, names, applied),
                    "utf-8",
                )
                print(
                    f"FAIL: {what} not written: the applied rules together lose "
                    f"{_eps(worst, 4)} of a graph's best time, over the rules eps "
                    f"{_eps(reduction.rules_epsilon)}; {len(losing)} graphs lose "
                    f"their best kernel; report {report}",
                    file=sys.stderr,
                )
                return 1
        count = instances(reduction.variants, names)
        if count > reduction.budget:
            note = f"not written: {what} exceed budget"
            report.write_text(
                render_report(reduction, args.selection, note, names, applied), "utf-8"
            )
            print(
                f"FAIL: {what} exceed budget: {count} instances > "
                f"{reduction.budget}; report {report}",
                file=sys.stderr,
            )
            return 1
        lines = write_selection(args.selection, names, args.out)
        note = f"`{args.out}` ({what}): {len(names)} kernels, {lines} lines"
        report.write_text(
            render_report(reduction, args.selection, note, names, applied), "utf-8"
        )
    except IncompleteSweep as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 3
    except ReduceError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    except (
        Exception
    ) as exc:  # noqa: BLE001 -- 1 is a gate refusal, 3 an incomplete sweep
        traceback.print_exc()
        print(f"FAIL: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"{len(names)} kernels -> {args.out}; report {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
