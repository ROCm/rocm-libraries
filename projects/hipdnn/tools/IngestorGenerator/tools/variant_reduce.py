# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reduce a variant selection to the kernels its autotune rankings justify.

Keeps each feature cell's floor (declared default plus the divisibility antichain of
its ``divides`` knobs), one kernel for every ranked graph left uncovered, and each
graph's best kernel while its gain over the best kept kernel exceeds epsilon: the
smallest grid value at or above the round-to-round noise floor that fits the budget.
Rankings are the ``WinnerCacheFile.hpp`` shards of the successful timing phases of a
``tools/sweep.py`` output: ``v:2`` whole-graph records for an unslotted engine, ``v:3``
slot records for a slotted one (its ``v:3`` program lines are validated and rank
nothing). A slotted engine reduces per slot: each slot's (graph, device) rankings,
floor, coverage, noise floor and epsilon use that slot's own kernels and the
``reduction.fields`` its own ``kmd_fields`` declare; pass ``--kdp`` once per pack. The
budget bounds every pack per arch, as ``generate.py`` counts it. The output keeps every
pack, filtered to its slot's kept kernels. Exit 1: the kept set (eps or
``--apply-rules``) exceeds the budget, or the applied rules together lose more of a
graph's best time than the rules eps (per slot, with the other slots at their reference
picks, not for the whole program); exit 2: unusable input; exit 3: the sweep cannot
support a reduction: no ``summary.json`` listing ``expected_phases``; an expected timing
or correctness phase of the arm with no record; an expected timing phase with no winners
(for a slotted engine: none for some slot); a failed correctness verdict; or timing
phases missing for some (round, corpus). On exit 1, and on exit 3 for a failed correctness verdict or missing
(round, corpus), the report is written and ``--out`` is not; an incomplete sweep (the
other exit 3 causes) writes neither. A sweep of fewer than 2 timing rounds warns on
stdout: its noise floor is assumed, not measured. ``reduction.single_round_epsilon``
(default 0.05) and ``reduction.drift_multiplier`` (default 2) set the noise floor.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import statistics
import struct
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
# A slot record is the v2 line plus `slot`; a `program` line holds the slot-to-kernel
# picks and the total time instead of `entries`.
_SLOT_LINE_FIELDS = _LINE_FIELDS | {"slot"}
_PROGRAM_LINE_FIELDS = frozenset({"v", "graph", "device", "program", "time_ms"})
_PICK_FIELDS = frozenset({"slot", "kernel_id"})
_DEVICE_FIELDS = frozenset(
    {
        "gcn_arch_name",
        "warp_size",
        "multi_processor_count",
        "total_global_mem",
        "memory_bus_width",
        "memory_clock_rate",
        "shared_mem_per_block",
    }
)
_ENTRY_FIELDS = frozenset({"kernel_id", "pack_id", "dispatch_id", "time_ms"})
# WINNER_LINE_FORMAT_VERSION: an unslotted engine's whole-graph records.
# WINNER_LINE_SLOTTED_FORMAT_VERSION: a slotted engine's slot records and program lines.
_WHOLE_GRAPH_VERSION = 2
_SLOTTED_VERSION = 3
_LINE_FORMAT_VERSIONS = (_WHOLE_GRAPH_VERSION, _SLOTTED_VERSION)
# GraphContentKey.hpp CONTENT_FIELD: the graph half of a line, the raw Graph flatbuffer.
_GRAPH_FIELDS = frozenset({"content_base64"})
# graph.fbs `table Graph`: `id: Uuid (cache_ignore)` is field 9, an inline [ubyte:16].
_GRAPH_ID_FIELD = 9
_GRAPH_ID_BYTES = 16

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
    #: A slotted engine's cells lead with the pack: a slot's packs serve different
    #: graphs (their discriminators), so no pack's kernel stands in for another's.
    by_pack: bool = False

    @property
    def cell_fields(self) -> tuple:
        """The names of a cell's values, in order."""
        return ("pack", *self.feature) if self.by_pack else self.feature

    @classmethod
    def from_reduction(cls, reduction: dict, slot_fields=None) -> "Classification":
        """``slot_fields`` are a slot's KMD field names: the classification of that
        slot, which keeps the reduction's fields the slot declares."""
        fields = reduction["fields"]

        def declared(name) -> bool:
            return slot_fields is None or name in slot_fields

        knobs = [k for k in fields.get("performance") or [] if declared(k["name"])]
        return cls(
            feature=tuple(n for n in fields.get("feature") or [] if declared(n)),
            performance=tuple(k["name"] for k in knobs),
            divides=tuple(k["name"] for k in knobs if k.get("divides")),
            by_pack=slot_fields is not None,
        )


@dataclass(frozen=True)
class Variant:
    name: str
    cell: tuple
    knobs: dict = field(hash=False, compare=False)
    default: bool = False
    arches: tuple = ()
    pack: str = ""


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
    slot: str = ""

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


def graph_identity(graph, where: str) -> str:
    """The graph half of a ranking's key: the line's Graph bytes with ``Graph.id`` zeroed.

    hipDNN mints a fresh ``Graph.id`` each time it finalizes a graph, and the C++ key
    compares content with that ``(cache_ignore)`` field skipped, but a line keeps the raw
    bytes. Every sweep round finalizes the same graph file again, so rounds only match
    once the id is cleared.
    """
    _require_fields(graph, _GRAPH_FIELDS, f"{where} graph")
    try:
        content = bytearray(base64.b64decode(graph["content_base64"], validate=True))
        table = struct.unpack_from("<I", content, 0)[0]
        vtable = table - struct.unpack_from("<i", content, table)[0]
        (vtable_size,) = struct.unpack_from("<H", content, vtable)
        slot = 4 + 2 * _GRAPH_ID_FIELD
        offset = (
            struct.unpack_from("<H", content, vtable + slot)[0]
            if slot < vtable_size
            else 0
        )
    except (ValueError, TypeError, struct.error) as exc:
        raise ReduceError(
            f"{where}: graph content is not a Graph flatbuffer ({exc})"
        ) from exc
    if offset:
        start = table + offset
        if start + _GRAPH_ID_BYTES > len(content):
            raise ReduceError(
                f"{where}: Graph.id runs past the end of the graph content"
            )
        content[start : start + _GRAPH_ID_BYTES] = bytes(_GRAPH_ID_BYTES)
    return base64.b64encode(bytes(content)).decode("ascii")


@dataclass(frozen=True)
class WinnerLine:
    """One decoded shard line (``WinnerCacheFile.hpp`` ``decodeWinnerLine``): a
    whole-graph record (v2), a slot record (v3 with ``slot``) or a program line (v3
    with ``picks``: ``(slot, kernel_id)`` in slot order, and ``time_ms``)."""

    version: int
    graph: str
    device: dict
    slot: str | None = None
    entries: list = field(default_factory=list)
    picks: tuple | None = None
    time_ms: float | None = None

    @property
    def key(self) -> str | None:
        """The ranking key, a JSON ``[graph identity, device]`` pair with the slot
        appended for a slot record; None for a program line, which ranks nothing."""
        if self.picks is not None:
            return None
        parts = [self.graph, self.device]
        if self.slot is not None:
            parts.append(self.slot)
        return json.dumps(parts, sort_keys=True)


def _check_time(t, where: str) -> None:
    if type(t) not in (int, float) or not math.isfinite(t) or t < 0:
        raise ReduceError(f"{where}: bad time_ms {t!r}")


def _decode_line(doc: dict, where: str) -> WinnerLine:
    version = doc.get("v")
    if type(version) is not int or version not in _LINE_FORMAT_VERSIONS:
        raise ReduceError(
            f"{where}: line format v={version!r}, expected one of "
            f"{list(_LINE_FORMAT_VERSIONS)}"
        )
    slotted = version == _SLOTTED_VERSION
    program = slotted and "program" in doc
    if program:
        # The decoder returns at `program` before it reads `entries`.
        _require_fields(
            {k: v for k, v in doc.items() if k != "entries"},
            _PROGRAM_LINE_FIELDS,
            where,
        )
    else:
        _require_fields(doc, _SLOT_LINE_FIELDS if slotted else _LINE_FIELDS, where)
    _require_fields(doc["device"], _DEVICE_FIELDS, f"{where} device")
    if program:
        picks = doc["program"]
        if not isinstance(picks, list) or not picks:
            raise ReduceError(f"{where}: 'program' must be a non-empty list")
        for index, pick in enumerate(picks):
            _require_fields(pick, _PICK_FIELDS, f"{where} program[{index}]")
            if any(not isinstance(pick[f], str) or not pick[f] for f in _PICK_FIELDS):
                raise ReduceError(
                    f"{where} program[{index}]: slot and kernel_id must be "
                    "non-empty strings"
                )
        _check_time(doc["time_ms"], where)
        return WinnerLine(
            version,
            graph_identity(doc["graph"], where),
            doc["device"],
            picks=tuple((p["slot"], p["kernel_id"]) for p in picks),
            time_ms=doc["time_ms"],
        )
    slot = doc.get("slot")
    if slotted and (not isinstance(slot, str) or not slot):
        raise ReduceError(f"{where}: 'slot' must be a non-empty string")
    if not isinstance(doc["entries"], list):
        raise ReduceError(f"{where}: 'entries' must be a list")
    for index, entry in enumerate(doc["entries"]):
        _require_fields(entry, _ENTRY_FIELDS, f"{where} entries[{index}]")
        _check_time(entry["time_ms"], f"{where} entries[{index}]")
    return WinnerLine(
        version,
        graph_identity(doc["graph"], where),
        doc["device"],
        slot=slot,
        entries=doc["entries"],
    )


def read_lines(path) -> list:
    """The `WinnerLine` of every line of one winner-cache shard, in file order.

    Line 1 is the version. Lines split as ``LineStore.hpp`` ``splitLineStoreLines``:
    on ``\\n``, dropping a preceding ``\\r``, ignoring an unterminated last chunk.
    """
    path = Path(path)
    out = []
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
        out.append(_decode_line(doc, where))
    return out


def read_shard(path) -> dict:
    """``{key: [entry, ...]}`` for one winner-cache shard; the last line per key wins.

    The keys are `WinnerLine.key`: a v2 record's is its ``(graph, device)``, a v3 slot
    record's adds the slot. A v3 program line is validated and ranks nothing."""
    return {line.key: line.entries for line in read_lines(path) if line.key is not None}


def _slot_of(key: str) -> str | None:
    """The slot of a ranking key; None for a whole-graph record's."""
    parts = json.loads(key)
    return parts[2] if len(parts) == 3 else None


def _ranked(shard, slots) -> dict:
    """`read_shard` of a shard swept for a selection whose slots are ``slots`` (None:
    an unslotted engine, which writes whole-graph records only; a slotted engine writes
    slot records only, for its own slots; its program lines name those slots too)."""
    lines = read_lines(shard)
    for line in lines:
        if line.picks is None:
            continue
        if slots is None:
            raise ReduceError(
                f"{shard}: a program line in the sweep of an unslotted engine"
            )
        for slot, _ in line.picks:
            if slot not in slots:
                raise ReduceError(
                    f"{shard}: a program line names slot {slot!r}, not a slot of the "
                    f"selection ({list(slots)})"
                )
    records = {line.key: line.entries for line in lines if line.key is not None}
    for key in records:
        slot = _slot_of(key)
        if slots is None and slot is not None:
            raise ReduceError(
                f"{shard}: a slot record ({slot!r}) in the sweep of an unslotted engine"
            )
        if slots is not None and slot is None:
            raise ReduceError(
                f"{shard}: a whole-graph v2 record in the sweep of a slotted engine"
            )
        if slots is not None and slot not in slots:
            raise ReduceError(
                f"{shard}: slot {slot!r} is not a slot of the selection ({list(slots)})"
            )
    return records


def _shards(attempt: Path) -> list:
    return sorted((attempt / "winners").rglob("winners.jsonl"))


def load_selection(path):
    """``(raw, config)``; requires a ``reduction:`` block, and for an unslotted engine
    a single pack."""
    import yaml
    from codegen.config_loader import ConfigError, load_config, read_yaml

    try:
        raw = read_yaml(Path(path))
        config = load_config(Path(path))
    except (OSError, ConfigError, yaml.YAMLError) as exc:
        raise ReduceError(f"{path}: {exc}") from exc
    if not config.reduction:
        raise ReduceError(f"{path}: no 'reduction' block")
    if not config.is_slotted and len(config.packs) != 1:
        raise ReduceError(f"{path}: {len(config.packs)} packs, expected 1")
    return raw, config


def classifications_of(config) -> dict:
    """``{slot: Classification}`` of the selection's reduction units: one named ``""``
    for an unslotted engine, one per slot that has a pack for a slotted one."""
    if not config.is_slotted:
        return {"": Classification.from_reduction(config.reduction)}
    return {
        slot.name: Classification.from_reduction(
            config.reduction, {f.name for f in slot.kmd_fields}
        )
        for slot in config.slots
        if any(pack.slot == slot.name for pack in config.packs)
    }


def load_kdp(path) -> dict:
    try:
        kdp = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = [k for k in ("id", "dispatch", "kernelDescriptors") if k not in kdp]
        if missing:
            raise KeyError(", ".join(missing))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReduceError(f"{path}: not a KDP ({exc})") from exc
    return kdp


def _kdp_of(pack, kdps: list, slotted: bool) -> dict:
    """The KDP built from exactly ``pack``'s kernels."""
    selected = {k.name for k in pack.kernels}
    named = [{d["name"] for d in kdp["kernelDescriptors"]} for kdp in kdps]
    for kdp, installed in zip(kdps, named):
        if installed == selected:
            return kdp
    installed = max(named, key=lambda names: len(names & selected), default=set())
    where = f" (pack {pack.name!r})" if slotted else ""
    raise ReduceError(
        f"the KDP is not built from this selection{where}: only in the selection "
        f"{sorted(selected - installed)[:3]}, only in the KDP "
        f"{sorted(installed - selected)[:3]}"
    )


def build_variants(config, kdps: list, classifications: dict) -> dict:
    """``{slot: {name: Variant}}`` in selection order, one slot (``""``) for an
    unslotted engine; fields from KDP metadata, then spec. Each pack pairs with the KDP
    built from its kernels, so ``kdps`` holds one per pack."""
    if len(kdps) != len(config.packs):
        raise ReduceError(
            f"{len(config.packs)} pack(s) need {len(config.packs)} --kdp, "
            f"got {len(kdps)}"
        )
    by_slot = {slot: {} for slot in classifications}
    for pack in config.packs:
        slot = pack.slot if config.is_slotted else ""
        classification = classifications[slot]
        kdp = _kdp_of(pack, kdps, config.is_slotted)
        kdps = [k for k in kdps if k is not kdp]
        installed = {d["name"]: d for d in kdp["kernelDescriptors"]}
        for kernel in pack.kernels:
            metadata = installed[kernel.name].get("metadata") or {}
            spec = kernel.kernel_source.spec

            def lookup(f):
                return metadata[f] if f in metadata else spec.get(f)

            cell = tuple(lookup(f) for f in classification.feature)
            if classification.by_pack:
                cell = (pack.name, *cell)
            knobs = {k: lookup(k) for k in classification.performance}
            arches = tuple(kernel.arch or pack.arch or [EVERY_ARCH])
            by_slot[slot][kernel.name] = Variant(
                kernel.name, cell, knobs, kernel.default, arches, pack.name
            )
    return by_slot


def instances(variants: dict, names) -> int:
    """Kept kernels for the most-populated (pack, arch): what the budget bounds, as
    ``generate.py`` ``warn_over_budget`` counts it."""
    per_arch = Counter(
        (variants[name].pack, arch) for name in names for arch in variants[name].arches
    )
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
    shown = zip(classification.cell_fields, cell)
    return "{" + ", ".join(f"{n}={v}" for n, v in shown if n in varying) + "}"


def propose_rules(
    variants: dict, rankings: list, protected, classification, epsilon, slot=""
):
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
        for index, name in enumerate(classification.cell_fields)
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
            rules.append(_costed(text, removed, variants, rankings, protected, slot))
    rules.sort(key=_rule_order)
    return rules


def _rule_order(rule: Rule) -> tuple:
    """Lossless rules first, by saving; then by saving per unit of worst gain lost."""
    w, saved = rule.worst_gain_lost, len(rule.removed)
    return (w > 0, -(saved / w) if w > 0 else -saved, rule.text)


def _rule_id(text: str, slot: str = "") -> str:
    """Derived from the rule text alone (and a slotted engine's slot, which can propose
    the text another proposes): an id is stable for a rule's text. The text
    of ``drop`` rules does not move with ``--budget``; a ``keep ... only for <cells>``
    rule names the cells where the value wins by more than the budget-chosen eps, so
    its text, and its id, changes with ``--budget``."""
    keyed = f"{slot}: {text}" if slot else text
    return "R" + hashlib.sha256(keyed.encode()).hexdigest()[:6]


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


def _costed(text, removed, variants, rankings, protected, slot="") -> Rule:
    losing, worst = _ranking_loss(removed, rankings)
    before = defaultdict(set)
    for v in variants.values():
        if v.name not in protected:
            before[v.cell].add(v.name)
    at_floor = sorted(
        (cell for cell, names in before.items() if names <= removed), key=repr
    )
    return Rule(_rule_id(text, slot), text, removed, len(losing), worst, at_floor)


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


def _rankings_and_correctness(attempts, kdps, variants, arm, slots):
    """``(rounds, correctness, duplicates, empty)``: ``rounds`` is ``{slot: {round:
    {key: Ranking}}}`` and the counts are ``{slot: n}``, for the ``variants`` of
    `build_variants`; ``correctness`` covers every slot. ``slots`` are the slot names of a
    slotted selection, None for an unslotted one."""
    by_id = {
        (str(kdp["id"]).lower(), str(kdp["dispatch"]).lower(), str(d["id"]).lower()): d[
            "name"
        ]
        for kdp in kdps
        for d in kdp["kernelDescriptors"]
    }

    def entries_of(raw, where, slot):
        out = []
        for entry in raw:
            ids = tuple(
                str(entry[f]).lower() for f in ("pack_id", "dispatch_id", "kernel_id")
            )
            name = by_id.get(ids)
            if name is None:
                raise ReduceError(f"stale ranking: {where}: {entry} not in the KDP")
            if name not in variants[slot]:
                raise ReduceError(f"{where}: {entry} is not a kernel of slot {slot}")
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
                for key, raw in _ranked(shard, slots).items():
                    entries = entries_of(raw, str(shard), _slot_of(key) or "")
                    if entries:
                        winners.add(entries[0][0])
            correctness.append(Correctness(phase, verdict, winners))

    rounds = {slot: {} for slot in variants}
    duplicates = dict.fromkeys(variants, 0)
    empty = dict.fromkeys(variants, 0)
    for number in sorted(timing):
        keyed = {slot: {} for slot in variants}
        for corpus, attempt in sorted(timing[number], key=lambda x: x[0]):
            for shard in _shards(attempt):
                for key, raw in _ranked(shard, slots).items():
                    slot = _slot_of(key) or ""
                    entries = entries_of(raw, str(shard), slot)
                    if not entries:
                        empty[slot] += 1
                        continue
                    cells = {variants[slot][name].cell for name, _ in entries}
                    if len(cells) != 1:
                        raise ReduceError(f"{shard}: a record spans {len(cells)} cells")
                    if key in keyed[slot]:
                        duplicates[slot] += 1
                    keyed[slot][key] = Ranking(
                        number, corpus, key, entries, cells.pop()
                    )
        for slot in variants:
            rounds[slot][number] = keyed[slot]
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


def _require_complete_arm(sweep_dir, attempts, arm, slots) -> None:
    """Raise `IncompleteSweep` unless ``sweep.py`` finished every timing and
    correctness phase of ``arm`` and each timing phase ranked something (for a slotted
    engine, in every slot of ``slots``). The expected phases are the sweep's own list in
    ``summary.json``, which it writes only after its last phase; another arm's phases
    are not this arm's evidence."""
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
        elif phase["kind"] == "timing":
            present = {
                _slot_of(key)
                for shard in _shards(attempt)
                for key in _ranked(shard, slots)
            }
            for slot in slots or [None]:
                if slot not in present:
                    problems.append(
                        f"{label}: "
                        + (f"slot {slot} has no winners" if slot else "no winners")
                    )
    if problems:
        raise IncompleteSweep(
            f"arm {arm}: the sweep is incomplete: {'; '.join(problems)}"
        )


def analyse_slots(selection, kdp_paths, sweep_dir, arm=None, budget=None) -> list:
    """One `Reduction` per slot of a slotted selection, in slot order; a single one,
    with an empty ``slot``, for an unslotted selection. ``kdp_paths`` are the installed
    KDPs, one per pack. Each slot ranks, floors, covers and sets its noise floor and
    epsilon on its own records and kernels; ``budget`` bounds every pack per arch."""
    raw, config = load_selection(selection)
    classifications = classifications_of(config)
    kdps = [load_kdp(path) for path in kdp_paths]
    variants = build_variants(config, kdps, classifications)
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
    slots = list(variants) if config.is_slotted else None
    _require_complete_arm(sweep_dir, attempts, arm, slots)
    rounds, correctness, duplicates, empty = _rankings_and_correctness(
        attempts, kdps, variants, arm, slots
    )
    missing_pairs = _missing_pairs(attempts, arm)
    if budget is None:
        budget = config.reduction.get(
            "aot_budget_per_arch", DEFAULT_AOT_BUDGET_PER_ARCH
        )
    reductions = []
    for slot, slot_variants in variants.items():
        where = f"slot {slot}: " if slot else ""
        slot_rounds = rounds[slot]
        if not any(slot_rounds.values()):
            raise ReduceError(
                f"arm {arm} has no ranking for slot {slot}"
                if slot
                else f"arm {arm} has no successful timing phase with a ranking"
            )
        try:
            noise = noise_floor(
                slot_rounds,
                missing_pairs,
                config.reduction.get(
                    "single_round_epsilon", DEFAULT_SINGLE_ROUND_EPSILON
                ),
                config.reduction.get("drift_multiplier", DEFAULT_DRIFT_MULTIPLIER),
            )
        except ReduceError as exc:
            raise ReduceError(f"{where}{exc}") from exc
        rankings = [
            ranking
            for number in sorted(slot_rounds)
            for ranking in sorted(slot_rounds[number].values(), key=lambda r: r.key)
        ]
        classification = classifications[slot]
        floor, no_default = floor_variants(slot_variants, classification)
        coverage, steps = search(rankings, floor)
        curve = [
            (e, instances(slot_variants, keep_at(floor, coverage, steps, e)))
            for e in EPSILON_GRID
        ]
        epsilon = choose_epsilon(curve, noise.epsilon_min, budget)
        floor_eps = next(e for e in EPSILON_GRID if e >= noise.epsilon_min)
        rules_epsilon = floor_eps if epsilon is None else epsilon
        protected = set(floor) | set(coverage)
        rules = propose_rules(
            slot_variants, rankings, protected, classification, rules_epsilon, slot
        )
        reductions.append(
            Reduction(
                classification=classification,
                variants=slot_variants,
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
                duplicate_keys=duplicates[slot],
                empty_records=empty[slot],
                slot=slot,
            )
        )
    return reductions


def analyse(selection, kdp_path, sweep_dir, arm=None, budget=None) -> Reduction:
    """The `Reduction` of an unslotted selection (`analyse_slots` for any selection)."""
    reductions = analyse_slots(selection, [kdp_path], sweep_dir, arm, budget)
    if reductions[0].slot:
        raise ReduceError(f"{selection}: a slotted selection reduces per slot")
    return reductions[0]


def _vocabulary(config, pack) -> dict:
    spec = config.slot_of(pack).specialization if config.is_slotted else None
    return ((spec if spec is not None else config.specialization) or {}).get(
        "vocabulary"
    ) or {}


def write_selection(selection, names, out) -> int:
    """Write the selection restricted to ``names``, every pack kept and factorised and
    round-trip checked; returns its line count. A slotted selection keeps each pack's
    ``slot``. The compact form has no per-kernel ``provenance``, so a kept kernel's is
    not carried (see the README)."""
    import factorise_config
    from codegen.config_loader import ConfigError

    raw, config = load_selection(selection)
    classifications = classifications_of(config)
    keep = set(names)
    expanded_packs = []
    compact_packs = []
    try:
        for pack_raw, pack in zip(raw["packs"], config.packs):
            kernels = []
            for kernel in pack.kernels:
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
            for key in factorise_config.PACK_HEADER_KEYS:
                if pack_raw.get(key):
                    expanded_pack[key] = pack_raw[key]
            expanded_pack["kernels"] = kernels
            expanded_packs.append(expanded_pack)
            # Knobs only where every kept kernel's metadata carries them: the loader
            # rejects a group's `policy_knobs` naming a field its `metadata` lacks.
            knobs = [
                knob
                for knob in classifications[
                    pack.slot if config.is_slotted else ""
                ].performance
                if all(knob in entry["metadata"] for entry in kernels)
            ]
            compact_packs.append(
                factorise_config.factorise_pack(
                    expanded_pack, knobs, _vocabulary(config, pack)
                )
            )
        expanded = {k: v for k, v in raw.items() if k != "packs"}
        compact = dict(expanded)
        expanded["packs"] = expanded_packs
        compact["packs"] = compact_packs
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


def _single_round_warning(noise: NoiseFloor, slot: str = "") -> str | None:
    if noise.rounds >= 2:
        return None
    where = f"slot {slot}: " if slot else ""
    return (
        f"{where}{noise.rounds} timing round: eps_min is assumed {noise.epsilon_min}, "
        "not measured; sweep at least 2 rounds"
    )


def graph_label(identity: str) -> str:
    """A short, stable name for one `graph_identity`: its content hash."""
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def _graph_at_device(key: str) -> str:
    """``graph@device`` (``/slot`` for a slot record) for one ranking key, a JSON
    ``[graph identity, device]`` pair with the slot appended for a slot record."""
    graph, device, *slot = json.loads(key)
    shown = device.get("gcn_arch_name") if isinstance(device, dict) else None
    arch = shown if isinstance(shown, str) else json.dumps(device, sort_keys=True)
    return f"{graph_label(graph)}@{arch}" + (f"/{slot[0]}" if slot else "")


def _applied_loss(applied, rankings) -> tuple:
    """`_ranking_loss` of the union of the ``applied`` rules' removed kernels."""
    return _ranking_loss(set().union(*(rule.removed for rule in applied)), rankings)


#: What a slot's loss figure does and does not bound; `_applied_loss` ranks one slot.
_SLOT_LOSS_SCOPE = (
    "the loss bounds each slot with the other slots at their reference picks, "
    "not the whole program"
)


def _budget_lines(r: Reduction, kept: set, applied) -> list:
    eps = r.epsilon if r.epsilon is not None else math.inf
    search = _eps(eps) if r.epsilon is not None else "none (over budget)"
    if applied:
        ids = ",".join(rule.id for rule in applied)
        chosen = f"chosen: rules {ids} (the eps search chose {search}; not used)"
    else:
        chosen = f"chosen eps: {search}"
    unit = "per pack per arch" if r.slot else "per arch"
    return [
        f"- budget {r.budget} instances {unit}; {chosen}",
        f"- kept {len(kept)} of {len(r.variants)} "
        f"({instances(r.variants, kept)} instances {unit})",
        f"- rounds {r.noise.rounds}, ranked graph records {len(r.rankings)}",
    ]


def _noise_lines(r: Reduction, applied, missing: bool) -> list:
    noise = r.noise
    lines = []
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
    if missing and noise.missing_pairs:
        lines.append(f"- missing (round, corpus): {noise.missing_pairs}")
    if r.duplicate_keys:
        lines.append(f"- keys ranked by several corpora in a round: {r.duplicate_keys}")
    if r.empty_records:
        lines.append(f"- records with no usable entry: {r.empty_records}")
    return lines


def _detail_lines(r: Reduction, kept: set, applied, level: int) -> list:
    """The eps curve and the variant, feature-cell and rule tables of one reduction,
    under headings ``level`` deep."""
    eps = r.epsilon if r.epsilon is not None else math.inf
    h = "#" * level
    lines = ["", f"{h} Eps curve", "", "| eps | kept |", "|---|---|"]
    for e in EPSILON_GRID:
        mark = " (chosen)" if e == r.epsilon and not applied else ""
        lines.append(f"| {_eps(e)}{mark} | {len(r.keep(e))} |")

    checked = {n for c in r.correctness if c.verdict == "passed" for n in c.winners}
    lines += ["", f"{h} Variants", "", "| variant | kept | reason |", "|---|---|---|"]
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
        f"{h} Feature cells ({', '.join(r.classification.cell_fields)})",
        "",
        f"- floor only, never ranked: {len(silent)} of {len(cells)}",
        *(f"  - {cell}" for cell in silent),
        f"- no declared default: {len(r.no_default_cells)}",
        *(f"  - {cell}" for cell in r.no_default_cells),
        "",
        f"{h} Proposed rules (eps {_eps(r.rules_epsilon)}; never remove the floor "
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
    return lines


def render_report(
    reductions: list, selection, output_note: str, names: dict, applied=None
) -> str:
    """The report for the kernel sets ``names`` (``{slot: names}``) that ``main``
    writes, not whichever sets eps alone would keep. ``applied`` maps each slot to the
    ``--apply-rules`` rules behind its set. An unslotted selection has one reduction
    with an empty slot; a slotted one gets a section per slot."""
    applied = applied or {}
    first = reductions[0]
    lines = [
        "# variant_reduce report",
        "",
        f"- selection `{selection}`, arm `{first.arm}`; output: {output_note}",
    ]
    if not first.slot:
        r = first
        kept, used = set(names[""]), applied.get("", ())
        lines += _budget_lines(r, kept, used)
        for warning in (
            _correctness_warning(r),
            _sweep_warning(r.noise),
            _single_round_warning(r.noise),
        ):
            if warning:
                lines.append(f"- WARNING: {warning}")
        lines += _noise_lines(r, used, missing=True)
        lines += _detail_lines(r, kept, used, 2)
    else:
        lines.append(
            f"- slots {', '.join(r.slot for r in reductions)}; {_SLOT_LOSS_SCOPE}; the "
            "full-corpus sweep stays the program-level check"
        )
        for warning in (_correctness_warning(first), _sweep_warning(first.noise)):
            if warning:
                lines.append(f"- WARNING: {warning}")
        if first.noise.missing_pairs:
            lines.append(f"- missing (round, corpus): {first.noise.missing_pairs}")
        for r in reductions:
            kept, used = set(names[r.slot]), applied.get(r.slot, ())
            lines += ["", f"## Slot `{r.slot}`", ""]
            lines += _budget_lines(r, kept, used)
            if warning := _single_round_warning(r.noise):
                lines.append(f"- WARNING: {warning}")
            lines += _noise_lines(r, used, missing=False)
            lines += _detail_lines(r, kept, used, 3)

    lines += ["", "## Correctness attempts", ""]
    for c in first.correctness:
        verdict = "FAILED; investigate" if c.verdict == "failed" else c.verdict
        lines.append(f"- {c.phase['corpus']}: {verdict}, {len(c.winners)} winners")
    if not first.correctness:
        lines.append("- none for this arm")
    return "\n".join(lines) + "\n"


def _check_rule_ids(rules: list, chosen: list) -> None:
    by_id = {r.id for r in rules}
    unknown = [r for r in chosen if r not in by_id]
    if unknown:
        raise ReduceError(
            f"--apply-rules {unknown} not proposed: {sorted(by_id)}; ids come from the "
            "rule text, so a rule the sweep or --budget no longer proposes is gone"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", required=True, help="selection YAML")
    parser.add_argument(
        "--kdp",
        required=True,
        action="append",
        help="installed KDP of the selection; once per pack of a slotted selection",
    )
    parser.add_argument("--sweep-output", required=True, help="tools/sweep.py output")
    parser.add_argument("--arm", help="sweep arm (default: the only one)")
    parser.add_argument("--out", required=True, help="reduced selection to write")
    parser.add_argument("--report", required=True, help="report to write")
    parser.add_argument(
        "--budget", type=int, help="instances per arch (per pack, for a slotted engine)"
    )
    parser.add_argument(
        "--apply-rules",
        default="",
        help="rule ids from the report's rule table (R and six hex digits, derived "
        "from the rule text and, for a slotted engine, its slot) to apply instead "
        "of eps; a slot with none keeps all its kernels",
    )
    args = parser.parse_args(argv)
    chosen = [r.strip() for r in args.apply_rules.split(",") if r.strip()]
    report = Path(args.report)
    try:
        reductions = analyse_slots(
            args.selection, args.kdp, args.sweep_output, args.arm, args.budget
        )
        first = reductions[0]
        blockers = [
            w for w in (_correctness_warning(first), _sweep_warning(first.noise)) if w
        ]
        singles = [_single_round_warning(r.noise, r.slot) for r in reductions]
        for warning in (*blockers, *singles):
            if warning:
                print(f"WARNING: {warning}")
        _check_rule_ids([rule for r in reductions for rule in r.rules], chosen)
        names, whats, applied = {}, {}, {}
        for r in reductions:
            ids = {rule.id for rule in r.rules}
            mine = [c for c in chosen if c in ids]
            applied[r.slot] = [rule for rule in r.rules if rule.id in mine]
            if chosen:
                names[r.slot] = apply_rules(r.variants, r.rules, mine)
                whats[r.slot] = f"rules {','.join(mine)}" if mine else "no rules"
            elif r.epsilon is not None:
                names[r.slot] = sorted(r.keep(r.epsilon))
                whats[r.slot] = f"eps {_eps(r.epsilon)}"
            else:
                names[r.slot] = sorted(r.keep(math.inf))
                whats[r.slot] = "floor + coverage"
        what = (
            "; ".join(f"slot {slot}: {w}" for slot, w in whats.items())
            if first.slot
            else whats[""]
        )

        def write_report(note):
            report.write_text(
                render_report(reductions, args.selection, note, names, applied),
                "utf-8",
            )

        if blockers:
            write_report(
                "not written: the sweep cannot support a reduction (see WARNING)"
            )
            print(
                f"FAIL: {what} not written: {'; '.join(blockers)}; report {report}",
                file=sys.stderr,
            )
            return 3
        refused = []
        for r in reductions:
            if applied[r.slot]:
                losing, worst = _applied_loss(applied[r.slot], r.rankings)
                if worst > r.rules_epsilon:
                    refused.append((r, losing, worst))
        if refused:
            write_report(
                "not written: the applied rules together lose more than the rules eps"
            )
            for r, losing, worst in refused:
                where = f"slot {r.slot}: " if r.slot else ""
                scope = f" ({_SLOT_LOSS_SCOPE})" if r.slot else ""
                print(
                    f"FAIL: {where}{whats[r.slot]} not written: the applied rules together lose "
                    f"{_eps(worst, 4)} of a graph's best time, over the rules eps "
                    f"{_eps(r.rules_epsilon)}; {len(losing)} graphs lose their best "
                    f"kernel{scope}; report {report}",
                    file=sys.stderr,
                )
            return 1
        over = [
            (r, instances(r.variants, names[r.slot]))
            for r in reductions
            if instances(r.variants, names[r.slot]) > r.budget
        ]
        if over:
            write_report(f"not written: {what} exceed budget")
            for r, count in over:
                where = f"slot {r.slot}: " if r.slot else ""
                print(
                    f"FAIL: {where}{whats[r.slot]} exceed budget: {count} instances > "
                    f"{r.budget}; report {report}",
                    file=sys.stderr,
                )
            return 1
        kept = [name for r in reductions for name in names[r.slot]]
        lines = write_selection(args.selection, kept, args.out)
        write_report(f"`{args.out}` ({what}): {len(kept)} kernels, {lines} lines")
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
    print(f"{len(kept)} kernels -> {args.out}; report {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
