#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Offline consumer for ``--emit-support-observations`` snapshots (RFC 0015 §12).

A run of the integration-test binary observes, for every graph it registers and
every engine loaded into the handle, whether that engine accepts the graph on
the machine the run happened on.  ``--emit-support-observations`` writes a
single snapshot JSON to stdout at the end of the run.  This tool reads one or
more snapshot files, compares them against the ``.support.json`` sidecars
committed to the tree, and proposes the lines the sidecars are missing.

Why an offline consumer at all
------------------------------
One CI run sees one ASIC on one OS, and usually only a shard of the bundles.
A sidecar has to hold the union across every ASIC, so no single run is in a
position to rewrite one -- it would erase the targets it did not visit.
Unioning the snapshots from every shard first, and only then diffing, is what
makes the update safe to compute.  ``--write-support-claims`` (the C++ bootstrap
path) rewrites sidecars in place from a single run precisely because it is
*not* this: it is the one-machine tool you point at an empty tree.

What it will and will not propose
---------------------------------
The merge is monotonic and optimistic, in both directions of that word:

* ``supported`` from any run wins over ``declined`` from any other.  Engines
  are allowed to be flaky and shards are allowed to be stale; a target that
  accepted the graph once can accept it, and the claim is a statement about
  capability rather than about a particular afternoon.
* ``unknown`` is discarded outright.  A query that errored, crashed, or timed
  out is not evidence of anything -- treating it as a decline is how a driver
  fault turns into a silent claim deletion.
* A committed ``supported`` claim is never removed or downgraded.  When a run
  reports ``declined`` for a cell the tree claims, that is reported as a
  conflict for a human to read and left alone.  Automatic downgrades would let
  one bad shard delete a matrix column.
* A ``declined`` observation on an unclaimed cell writes nothing.  Absence
  already means "not asserted"; there is no negative claim to record.

The output is a directory of proposed sidecars, never an edit in place.  The
committed tree is only ever read here -- the human gate is the point (RFC 0015
§12.3), and a tool that can write to the tree is a tool that can be wired into
a CI job that commits to it.

Canonical form
--------------
Proposed sidecars are serialised exactly as ``dumpCanonical()`` does in
``SupportClaims.cpp`` -- sorted keys, two-space indent, trailing newline -- and
sweep groups are rebuilt exactly as ``regroupSweepClaims()`` does in
``SupportClaimWriter.cpp``.  Both matter for the same reason: a proposal that
reformats a file it did not otherwise change buries one added platform token in
a hundred lines of churn.  A sidecar already in canonical form round-trips
through this tool untouched.

Usage
-----
    python3 harvest_support_observations.py \\
        --observations snapshot-1.json snapshot-2.json \\
        --bundles-dir integration-test-bundles \\
        --output-dir /tmp/proposed-sidecars

Add ``--dry-run`` to print the diff without writing, and ``--coverage-report``
for the per-ASIC table of how much of the tree the runs actually visited.

Exit codes
----------
0  Ran to completion (conflicts and skipped records are warnings, not failures;
   the tool's job is to report, and a non-zero exit here would block the very
   pipeline meant to surface them).
1  Could not run: no readable observation file, or the output could not be
   written.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import pathlib
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bundle_discovery import (  # noqa: E402
    SWEEP_MANIFEST_NAME,
    SWEEP_SUPPORT_NAME,
    find_graph_files,
    find_sweep_roots,
)
from engine_registry import known_engines  # noqa: E402
from verify_support_claims import (  # noqa: E402
    VALID_ENFORCEMENT_LEVELS,
    VALID_PLATFORMS,
)

INTEGRATION_TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BUNDLES_DIR = INTEGRATION_TESTS_DIR / "integration-test-bundles"

SUPPORTED = "supported"
DECLINED = "declined"
UNKNOWN = "unknown"

# Merge precedence.  Higher wins, so a single ``supported`` survives any number
# of declines and ``unknown`` never displaces an answer.
VERDICT_RANK = {UNKNOWN: 0, DECLINED: 1, SUPPORTED: 2}

SIDECAR_VERSION = 1


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def load_json(path: pathlib.Path):
    """Parse ``path`` as JSON, warning and returning None on any failure."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"{path}: failed to parse JSON ({exc}); skipping")
        return None


# --------------------------------------------------------------------------
# Observation records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationKey:
    """The full identity of one observed cell.

    Every field is part of the key.  There is no inference across any of them:
    an engine supporting a case on gfx942/linux says nothing about the same
    case on gfx90a, and a sibling case in the same sweep is a different graph.
    """

    bundle: str
    case_id: str | None
    engine: str
    arch: str
    platform: str

    @property
    def target(self) -> tuple[str, str]:
        return (self.arch, self.platform)

    @property
    def unit(self) -> tuple[str, str | None]:
        return (self.bundle, self.case_id)

    def describe(self) -> str:
        where = f"{self.bundle}[{self.case_id}]" if self.case_id else self.bundle
        return f"{where} {self.engine} {self.arch}/{self.platform}"


@dataclass
class LoadStats:
    files_read: int = 0
    files_failed: int = 0
    records_valid: int = 0
    records_malformed: int = 0


def _observation_error(obs: dict) -> str | None:
    """Return why one observation entry is unusable, or None when well-formed.

    ``arch`` and ``platform`` are validated at the snapshot level (``target``),
    not per-observation.
    """
    for name in ("bundle", "engine", "verdict"):
        value = obs.get(name)
        if not isinstance(value, str) or not value:
            return f"field '{name}' must be a non-empty string, got {value!r}"

    verdict = obs["verdict"]
    if verdict not in VERDICT_RANK:
        return f"unknown verdict {verdict!r} (expected one of {sorted(VERDICT_RANK)})"

    case_id = obs.get("case_id")
    if case_id is not None and (not isinstance(case_id, str) or not case_id):
        return f"field 'case_id' must be a non-empty string or null, got {case_id!r}"

    graph = obs.get("graph")
    if graph is not None and (not isinstance(graph, str) or not graph):
        return f"field 'graph' must be a non-empty string or null, got {graph!r}"

    # Validated for forward-compat; not consumed by the harvest pipeline today.
    level = obs.get("enforcement_level")
    if level is not None and level not in VALID_ENFORCEMENT_LEVELS:
        return (
            f"invalid enforcement_level {level!r}"
            f" (expected one of {sorted(VALID_ENFORCEMENT_LEVELS)})"
        )

    return None


def parse_snapshot(
    data: dict,
) -> tuple[list[tuple[ObservationKey, str, str | None]], list[str]]:
    """Parse a snapshot dict into ``(valid_records, per_observation_errors)``.

    Raises :class:`ValueError` for structural problems (bad schema_version,
    missing target, non-array observations).  Individual malformed observations
    are collected into the second element of the return tuple so the caller can
    log them without discarding the valid records from the same file.
    """
    if not isinstance(data, dict):
        raise ValueError(f"snapshot must be an object, got {type(data).__name__}")

    version = data.get("schema_version")
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"invalid schema_version {version!r} (expected positive int)")
    if version > 1:
        raise ValueError(
            f"schema_version {version} is newer than this tool supports (max 1); "
            "upgrade the harvest tool or regenerate with --emit-support-observations"
        )

    target = data.get("target")
    if not isinstance(target, dict):
        raise ValueError(f"'target' must be an object, got {type(target).__name__}")
    arch = target.get("arch")
    if not isinstance(arch, str) or not arch:
        raise ValueError(f"target.arch must be a non-empty string, got {arch!r}")
    platform = target.get("platform")
    if not isinstance(platform, str) or not platform:
        raise ValueError(
            f"target.platform must be a non-empty string, got {platform!r}"
        )
    if platform not in VALID_PLATFORMS:
        raise ValueError(
            f"invalid platform {platform!r} (expected one of {sorted(VALID_PLATFORMS)})"
        )

    observations = data.get("observations")
    if not isinstance(observations, list):
        raise ValueError(
            f"'observations' must be an array, got {type(observations).__name__}"
        )

    results: list[tuple[ObservationKey, str, str | None]] = []
    errors: list[str] = []
    for i, obs in enumerate(observations):
        if not isinstance(obs, dict):
            errors.append(f"observations[{i}]: must be an object")
            continue
        problem = _observation_error(obs)
        if problem is not None:
            errors.append(f"observations[{i}]: {problem}")
            continue
        key = ObservationKey(
            bundle=obs["bundle"],
            case_id=obs.get("case_id"),
            engine=obs["engine"],
            arch=arch,
            platform=platform,
        )
        results.append((key, obs["verdict"], obs.get("graph")))
    return results, errors


def load_snapshots(
    paths: list[pathlib.Path],
) -> tuple[dict[ObservationKey, str], dict[ObservationKey, str], LoadStats]:
    """Read and union every snapshot JSON file.

    Returns the merged ``key -> verdict`` map, the ``key -> graph`` hints seen
    alongside it, and the load statistics.  The union is the whole merge
    strategy: snapshots are append-only evidence, so reading them in any order,
    or reading the same one twice, gives the same answer.
    """
    merged: dict[ObservationKey, str] = {}
    graph_hints: dict[ObservationKey, str] = {}
    stats = LoadStats()

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            warn(f"{path}: cannot read snapshot ({exc}); skipping file")
            stats.files_failed += 1
            continue
        stats.files_read += 1

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            if "\n" in text.strip() and text.strip().startswith("{"):
                warn(
                    f"{path}: looks like old-format JSONL (one object per line); "
                    "this tool now expects snapshot JSON. Re-run tests with "
                    "the current binary to produce the new format."
                )
            else:
                warn(f"{path}: invalid JSON; skipping file")
            stats.files_failed += 1
            continue

        if isinstance(data, dict) and "schema_version" not in data:
            warn(
                f"{path}: JSON object has no schema_version field; "
                "this looks like a pre-snapshot observation. Re-run tests with "
                "the current binary to produce the new format."
            )
            stats.files_failed += 1
            continue

        try:
            parsed, obs_errors = parse_snapshot(data)
        except ValueError as exc:
            warn(f"{path}: {exc}; skipping file")
            stats.files_failed += 1
            continue

        for err in obs_errors:
            warn(f"{path}: {err}")
            stats.records_malformed += 1

        for key, verdict, graph in parsed:
            stats.records_valid += 1
            if VERDICT_RANK[verdict] > VERDICT_RANK.get(merged.get(key), -1):
                merged[key] = verdict
            if graph is not None:
                graph_hints[key] = graph

    return merged, graph_hints, stats


# --------------------------------------------------------------------------
# The committed bundle tree
# --------------------------------------------------------------------------


@dataclass
class BundleEntry:
    """One bundle as it exists on disk, with the sidecar it would be claimed in.

    ``graphs`` is normally a single stem, and the sidecar is derived from it.
    Nothing forbids two graph ``.json`` files sharing a directory, though, and
    then the bundle path alone no longer names one sidecar; such an observation
    is placed only if it carries the optional ``graph`` field.  Guessing would
    write one graph's support into the other's file.
    """

    bundle: str
    is_sweep: bool
    directory: pathlib.Path
    # Sweep: [] -- the sidecar is the bare support.json.  Single-graph: the
    # stems of the graphs living here, sorted.
    graphs: list[str] = field(default_factory=list)
    case_ids: set[str] = field(default_factory=set)

    def sidecar_for(self, graph: str | None) -> pathlib.Path | None:
        if self.is_sweep:
            return self.directory / SWEEP_SUPPORT_NAME
        if graph is None:
            if len(self.graphs) != 1:
                return None
            graph = self.graphs[0]
        elif graph not in self.graphs:
            return None
        return self.directory / f"{graph}.support.json"

    def units(self) -> list[tuple[str, str | None]]:
        """The claim units this bundle contributes to the coverage denominator."""
        if self.is_sweep:
            return [(self.bundle, case_id) for case_id in sorted(self.case_ids)]
        return [(self.bundle, None)]


def index_bundles(root: pathlib.Path) -> dict[str, BundleEntry]:
    """Map bundle path -> :class:`BundleEntry` for every bundle under ``root``."""
    entries: dict[str, BundleEntry] = {}
    sweep_roots = find_sweep_roots(root)

    for sweep_dir in sweep_roots:
        bundle = sweep_dir.relative_to(root).as_posix()
        entry = BundleEntry(bundle=bundle, is_sweep=True, directory=sweep_dir)
        sweep = load_json(sweep_dir / SWEEP_MANIFEST_NAME)
        cases = sweep.get("cases") if isinstance(sweep, dict) else None
        if not isinstance(cases, list):
            warn(
                f"{sweep_dir / SWEEP_MANIFEST_NAME}: malformed sweep.json; no case ids"
            )
            cases = []
        for case in cases:
            if isinstance(case, dict) and isinstance(case.get("id"), str):
                entry.case_ids.add(case["id"])
        entries[bundle] = entry

    for graph_path in find_graph_files(root, sweep_roots):
        directory = graph_path.parent
        bundle = directory.relative_to(root).as_posix()
        entry = entries.get(bundle)
        if entry is None:
            entry = BundleEntry(bundle=bundle, is_sweep=False, directory=directory)
            entries[bundle] = entry
        entry.graphs.append(graph_path.stem)

    for entry in entries.values():
        entry.graphs.sort()
    return entries


def _pairs_from_support(
    support: object, path: pathlib.Path | None = None
) -> set[tuple[str, str]]:
    """Flatten a ``{arch: [platform, ...]}`` map into ``(arch, platform)`` pairs."""
    pairs: set[tuple[str, str]] = set()
    if not isinstance(support, dict):
        if support is not None and path is not None:
            warn(f"{path}: 'support' must be an object, got {type(support).__name__}")
        return pairs
    for arch, platforms in support.items():
        if not isinstance(platforms, list):
            if path is not None:
                warn(
                    f"{path}: support[{arch!r}] must be a list,"
                    f" got {type(platforms).__name__}"
                )
            continue
        for platform in platforms:
            if isinstance(platform, str):
                pairs.add((arch, platform))
    return pairs


@dataclass
class Sidecar:
    """A sidecar's claims, flattened to the shape both shapes reduce to.

    ``cells`` is keyed the same way for both on-disk shapes -- single-graph
    files use the sole ``None`` case -- so the diff below never has to branch on
    which one it is looking at.  Only serialisation cares about the difference.
    """

    path: pathlib.Path
    is_sweep: bool
    exists: bool
    version: int = SIDECAR_VERSION
    # engine -> case id (None for single-graph) -> {(arch, platform)}
    cells: dict[str, dict[str | None, set[tuple[str, str]]]] = field(
        default_factory=dict
    )

    def claimed(
        self, engine: str, case_id: str | None, target: tuple[str, str]
    ) -> bool:
        return target in self.cells.get(engine, {}).get(case_id, ())

    def add(self, engine: str, case_id: str | None, target: tuple[str, str]) -> None:
        self.cells.setdefault(engine, {}).setdefault(case_id, set()).add(target)


def read_sidecar(path: pathlib.Path, is_sweep: bool) -> Sidecar:
    """Load a sidecar into flat cells, tolerating anything malformed in it.

    A file that cannot be parsed is read as empty rather than skipped.  Refusing
    to propose against a broken sidecar would mean the tree's least trustworthy
    files are the ones that never get fixed; proposing against an empty one
    produces a diff a reviewer can see the whole of.
    """
    sidecar = Sidecar(path=path, is_sweep=is_sweep, exists=path.is_file())
    if not sidecar.exists:
        return sidecar

    document = load_json(path)
    if not isinstance(document, dict):
        return sidecar
    if isinstance(document.get("version"), int):
        sidecar.version = document["version"]

    claims = document.get("claims")
    if not isinstance(claims, dict):
        if claims is not None:
            warn(f"{path}: 'claims' must be an object; treating as empty")
        return sidecar

    for engine, value in claims.items():
        if is_sweep:
            if not isinstance(value, list):
                warn(f"{path}: engine '{engine}' claims must be an array; ignoring")
                continue
            for group in value:
                if not isinstance(group, dict):
                    continue
                pairs = _pairs_from_support(group.get("support"), path)
                cases = group.get("cases")
                if not isinstance(cases, list):
                    continue
                for case_id in cases:
                    if isinstance(case_id, str):
                        # Union rather than replace.  Duplicate case ids across
                        # groups are already a verify_support_claims.py error,
                        # but on such a file replacing would silently drop a
                        # committed claim -- the one thing this tool must never
                        # do, however malformed the input.
                        sidecar.cells.setdefault(engine, {}).setdefault(
                            case_id, set()
                        ).update(pairs)
        else:
            pairs = _pairs_from_support(value, path)
            if pairs:
                sidecar.cells.setdefault(engine, {})[None] = pairs
    return sidecar


# --------------------------------------------------------------------------
# Serialisation, matching dumpCanonical() / regroupSweepClaims()
# --------------------------------------------------------------------------


def _arch_platform_map(pairs: set[tuple[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for arch, platform in sorted(pairs):
        grouped[arch].append(platform)
    return {arch: sorted(platforms) for arch, platforms in sorted(grouped.items())}


def render_sidecar(sidecar: Sidecar) -> str:
    """Serialise ``sidecar`` the way the C++ writer would."""
    claims: dict[str, object] = {}

    for engine, by_case in sorted(sidecar.cells.items()):
        if sidecar.is_sweep:
            # Bucket cases by identical support footprint, exactly as
            # regroupSweepClaims() does, so a tree written by the C++ writer is
            # a fixed point of this function.
            by_footprint: dict[tuple, list[str]] = collections.defaultdict(list)
            for case_id, pairs in by_case.items():
                if case_id is None or not pairs:
                    continue
                by_footprint[tuple(sorted(pairs))].append(case_id)
            groups = [
                {
                    "cases": sorted(cases),
                    "support": _arch_platform_map(set(footprint)),
                }
                for footprint, cases in by_footprint.items()
            ]
            groups.sort(key=lambda group: group["cases"][0])
            if groups:
                claims[engine] = groups
        else:
            pairs = by_case.get(None) or set()
            if pairs:
                claims[engine] = _arch_platform_map(pairs)

    document = {"version": sidecar.version, "claims": claims}
    # dumpCanonical(): nlohmann's default json is a std::map, so every key is
    # sorted; dump(2) plus a trailing newline.
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------
# Diffing observations against the tree
# --------------------------------------------------------------------------


@dataclass
class Addition:
    bundle: str
    case_id: str | None
    engine: str
    arch: str
    platform: str

    def describe(self) -> str:
        where = f"{self.bundle}[{self.case_id}]" if self.case_id else self.bundle
        return f"{where} {self.engine} {self.arch}/{self.platform}"


@dataclass
class Conflict:
    key: ObservationKey


@dataclass
class Proposal:
    """One sidecar's worth of proposed change."""

    relative_path: pathlib.PurePosixPath
    committed_path: pathlib.Path
    sidecar: Sidecar
    additions: list[Addition] = field(default_factory=list)


@dataclass
class DiffStats:
    unknown_discarded: int = 0
    already_claimed: int = 0
    declined_no_claim: int = 0
    unplaceable: int = 0


@dataclass
class HarvestResult:
    proposals: list[Proposal]
    conflicts: list[Conflict]
    stats: DiffStats
    # (arch, platform) -> units of the tree that some run observed
    observed_units: dict[tuple[str, str], set[tuple[str, str | None]]]
    targets: set[tuple[str, str]]
    total_units: int


def _resolve(
    key: ObservationKey,
    entry: BundleEntry | None,
    graph: str | None,
) -> pathlib.Path | None:
    """The sidecar this observation belongs in, or None with a warning why."""
    if entry is None:
        warn(f"{key.describe()}: no such bundle in the tree; ignoring observation")
        return None
    if entry.is_sweep:
        if key.case_id is None:
            warn(f"{key.describe()}: sweep bundle observed without a case_id; ignoring")
            return None
        if key.case_id not in entry.case_ids:
            warn(
                f"{key.describe()}: case id not found in {SWEEP_MANIFEST_NAME};"
                " ignoring observation"
            )
            return None
    elif key.case_id is not None:
        warn(
            f"{key.describe()}: case_id on a single-graph bundle; ignoring observation"
        )
        return None

    sidecar_path = entry.sidecar_for(graph)
    if sidecar_path is None:
        warn(
            f"{key.describe()}: bundle holds {len(entry.graphs)} graphs"
            f" ({', '.join(entry.graphs)}) and the record names none of them;"
            " ignoring observation"
        )
    return sidecar_path


def harvest(
    merged: dict[ObservationKey, str],
    graph_hints: dict[ObservationKey, str],
    entries: dict[str, BundleEntry],
    root: pathlib.Path,
) -> HarvestResult:
    """Turn merged observations into proposed sidecar updates."""
    stats = DiffStats()
    conflicts: list[Conflict] = []
    sidecars: dict[pathlib.Path, Sidecar] = {}
    additions: dict[pathlib.Path, list[Addition]] = collections.defaultdict(list)
    observed_units: dict[tuple[str, str], set[tuple[str, str | None]]] = (
        collections.defaultdict(set)
    )
    targets: set[tuple[str, str]] = set()

    for key in sorted(
        merged, key=lambda k: (k.bundle, k.case_id or "", k.engine, k.arch, k.platform)
    ):
        verdict = merged[key]
        targets.add(key.target)

        entry = entries.get(key.bundle)
        sidecar_path = _resolve(key, entry, graph_hints.get(key))
        if sidecar_path is None:
            stats.unplaceable += 1
            continue

        # Coverage counts the visit, not the answer.  A query that errored
        # still means a machine reached this graph, which is the question
        # "did any shard cover gfx1151/windows" is actually asking.
        observed_units[key.target].add(key.unit)

        if verdict == UNKNOWN:
            stats.unknown_discarded += 1
            continue

        sidecar = sidecars.get(sidecar_path)
        if sidecar is None:
            if entry is None:
                stats.unplaceable += 1
                continue
            sidecar = read_sidecar(sidecar_path, entry.is_sweep)
            sidecars[sidecar_path] = sidecar

        if verdict == DECLINED:
            if sidecar.claimed(key.engine, key.case_id, key.target):
                conflicts.append(Conflict(key))
            else:
                stats.declined_no_claim += 1
            continue

        if sidecar.claimed(key.engine, key.case_id, key.target):
            stats.already_claimed += 1
            continue

        sidecar.add(key.engine, key.case_id, key.target)
        additions[sidecar_path].append(
            Addition(
                bundle=key.bundle,
                case_id=key.case_id,
                engine=key.engine,
                arch=key.arch,
                platform=key.platform,
            )
        )

    proposals = [
        Proposal(
            relative_path=pathlib.PurePosixPath(path.relative_to(root).as_posix()),
            committed_path=path,
            sidecar=sidecars[path],
            additions=sorted(
                items, key=lambda a: (a.engine, a.arch, a.platform, a.case_id or "")
            ),
        )
        for path, items in sorted(additions.items())
    ]

    total_units = sum(len(entry.units()) for entry in entries.values())
    for entry in entries.values():
        sidecar_paths = (
            [entry.directory / SWEEP_SUPPORT_NAME]
            if entry.is_sweep
            else [entry.directory / f"{stem}.support.json" for stem in entry.graphs]
        )
        for path in sidecar_paths:
            if not path.is_file():
                continue
            committed = sidecars.get(path) or read_sidecar(path, entry.is_sweep)
            for by_case in committed.cells.values():
                for pairs in by_case.values():
                    targets |= pairs

    return HarvestResult(
        proposals=proposals,
        conflicts=conflicts,
        stats=stats,
        observed_units=observed_units,
        targets=targets,
        total_units=total_units,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def unified_diff(proposal: Proposal) -> str:
    """A reviewable diff of one proposed sidecar against the committed file."""
    label = proposal.relative_path.as_posix()
    if proposal.sidecar.exists:
        try:
            before = proposal.committed_path.read_text(encoding="utf-8").splitlines(
                keepends=True
            )
        except OSError:
            before = []
    else:
        before = []
    after = render_sidecar(proposal.sidecar).splitlines(keepends=True)
    lines = difflib.unified_diff(
        before,
        after,
        fromfile=f"a/{label}" + ("" if proposal.sidecar.exists else " (new file)"),
        tofile=f"b/{label}",
    )
    return "".join(lines)


def report_proposals(result: HarvestResult, stream) -> None:
    if not result.proposals:
        print("No additions to propose.", file=stream)
        return
    additions = sum(len(proposal.additions) for proposal in result.proposals)
    print(
        f"=== Proposed Additions ({additions} across"
        f" {len(result.proposals)} sidecar(s)) ===",
        file=stream,
    )
    for proposal in result.proposals:
        print(file=stream)
        for addition in proposal.additions:
            print(f"  + {addition.describe()}", file=stream)
        diff = unified_diff(proposal)
        if diff:
            print(file=stream)
            print(diff, end="" if diff.endswith("\n") else "\n", file=stream)


def report_conflicts(result: HarvestResult, stream) -> None:
    if not result.conflicts:
        return
    print(file=stream)
    print(
        f"=== Conflicts ({len(result.conflicts)}) ===\n"
        "Observed 'declined' where the tree claims support. Not downgraded --\n"
        "either the claim is stale or the run is, and only a human can say which.",
        file=stream,
    )
    for conflict in result.conflicts:
        print(f"  ! {conflict.key.describe()}", file=stream)


def report_coverage(result: HarvestResult, stream) -> None:
    """Per-ASIC coverage over claim units (a sweep case, or a whole single graph).

    Units rather than bundles: a sweep is one directory but hundreds of graphs,
    and counting it once would score a shard that reached one case of it as
    having covered the lot.
    """
    print("=== Observation Coverage ===", file=stream)
    total = result.total_units
    if total == 0:
        print("  (no bundles found)", file=stream)
        return
    labels = {target: f"{target[0]} / {target[1]}:" for target in result.targets}
    width = max((len(label) for label in labels.values()), default=0)
    digits = len(str(total))
    for target in sorted(result.targets):
        seen = len(result.observed_units.get(target, ()))
        percent = 100.0 * seen / total
        print(
            f"{labels[target]:<{width}} {seen:>{digits}} / {total} units"
            f" observed ({percent:.1f}%)",
            file=stream,
        )


def report_summary(
    load_stats: LoadStats, result: HarvestResult, distinct_cells: int, stream
) -> None:
    stats = result.stats
    additions = sum(len(proposal.additions) for proposal in result.proposals)
    rows = [
        ("snapshot files read", load_stats.files_read),
        ("snapshot files unreadable", load_stats.files_failed),
        ("records valid", load_stats.records_valid),
        ("records malformed (skipped)", load_stats.records_malformed),
        ("distinct cells after union", distinct_cells),
        ("unknown verdicts discarded", stats.unknown_discarded),
        ("observations not placeable", stats.unplaceable),
        ("already claimed (no-op)", stats.already_claimed),
        ("declined, unclaimed (no-op)", stats.declined_no_claim),
        ("additions proposed", additions),
        ("sidecars touched", len(result.proposals)),
        ("conflicts for review", len(result.conflicts)),
    ]
    width = max(len(label) for label, _ in rows)
    print("=== Harvest Summary ===", file=stream)
    for label, value in rows:
        print(f"  {label:<{width}}  {value}", file=stream)


def warn_unknown_engines(engines: set[str], context: str) -> None:
    """Flag engine names that the registry does not know (RFC 0015 §12.4)."""
    registry = known_engines()
    if registry is None:
        return
    for engine in sorted(engines - registry):
        warn(
            f"engine '{engine}' appears in {context} but is not in the engine"
            " registry -- may be retired or misspelt"
        )


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_proposals(
    result: HarvestResult, output_dir: pathlib.Path
) -> list[pathlib.Path]:
    """Mirror each proposed sidecar under ``output_dir``, never in the tree."""
    written = []
    for proposal in result.proposals:
        destination = output_dir / proposal.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_sidecar(proposal.sidecar), encoding="utf-8")
        written.append(destination)
    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Union support observations from CI runs and propose the sidecar"
            " lines the tree is missing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The committed tree is never modified. Proposals are written under"
            " --output-dir, mirroring their path in the bundle tree, for review"
            " as a pull request."
        ),
    )
    parser.add_argument(
        "--observations",
        nargs="+",
        required=True,
        type=pathlib.Path,
        metavar="JSON",
        help="snapshot JSON files emitted by --emit-support-observations",
    )
    parser.add_argument(
        "--bundles-dir",
        type=pathlib.Path,
        default=DEFAULT_BUNDLES_DIR,
        help=f"bundle tree to diff against (default: {DEFAULT_BUNDLES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="where to write proposed sidecars; required unless --dry-run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the proposed diff without writing any file",
    )
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="also print per-ASIC observation coverage",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.dry_run and args.output_dir is None:
        warn("--output-dir is required unless --dry-run is given")
        return 1

    root = args.bundles_dir.resolve()
    if not root.is_dir():
        warn(f"{args.bundles_dir}: no such bundle tree")
        return 1

    merged, graph_hints, load_stats = load_snapshots(args.observations)
    if load_stats.files_read == 0:
        warn("no snapshot file could be read")
        return 1

    if not merged:
        warn(
            f"read {load_stats.files_read} snapshot file(s) but none contained"
            f" usable observations ({load_stats.records_malformed} record(s)"
            f" malformed, {load_stats.files_failed} file(s) failed)"
        )
        return 1

    warn_unknown_engines({key.engine for key in merged}, "the observation files")

    entries = index_bundles(root)
    result = harvest(merged, graph_hints, entries, root)

    report_proposals(result, sys.stdout)
    report_conflicts(result, sys.stdout)
    print(file=sys.stdout)
    if args.coverage_report:
        report_coverage(result, sys.stdout)
        print(file=sys.stdout)
    report_summary(load_stats, result, len(merged), sys.stdout)

    if args.dry_run:
        return 0

    try:
        written = write_proposals(result, args.output_dir)
    except OSError as exc:
        warn(f"could not write proposals: {exc}")
        return 1
    print(f"\nWrote {len(written)} proposed sidecar(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
