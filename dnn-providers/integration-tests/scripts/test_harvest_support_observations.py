#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for harvest_support_observations.py.

Most tests build a small bundle tree in ``tmp_path``, hand the tool a snapshot
JSON file, and assert on the exact proposal -- no GPU, no CI data, no C++ build.

The design rules the tool exists to enforce (RFC 0015 §12.2) are all negative:
never downgrade, never delete, never infer across targets, never write to the
tree. Negative rules do not show up in a happy-path test, so each has one here
that would pass just as well against a tool that did nothing at all -- paired
with a positive test proving the tool is not, in fact, doing nothing.

``TestRealBundleTree`` runs against the committed sidecars. Its crosscheck
re-reads each file with a hand-rolled walk that shares no code with
``read_sidecar``, because the flatten/regroup round trip is the one place a
silent claim loss could hide: a bug there would corrupt every proposal
identically, and a test written in terms of the same helper would agree with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bundle_discovery import SWEEP_SUPPORT_NAME
from harvest_support_observations import (
    DECLINED,
    DEFAULT_BUNDLES_DIR,
    SUPPORTED,
    UNKNOWN,
    ObservationKey,
    harvest,
    index_bundles,
    load_snapshots,
    main,
    parse_snapshot,
    read_sidecar,
    render_sidecar,
    report_coverage,
)

GFX942 = ("gfx942", "linux")
GFX90A = ("gfx90a", "linux")
GFX1151 = ("gfx1151", "windows")

MIOPEN = "MIOPEN_ENGINE"
MLOPS = "HIP_MLOPS_ENGINE"


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _single_bundle(
    root: Path,
    relative: str,
    name: str = "Small",
    claims: dict | None = None,
) -> Path:
    directory = root / relative
    _write_json(directory / f"{name}.json", {"nodes": []})
    if claims is not None:
        _write_json(
            directory / f"{name}.support.json", {"version": 1, "claims": claims}
        )
    return directory


def _sweep_bundle(
    root: Path,
    relative: str,
    case_ids: list[str],
    claims: dict | None = None,
) -> Path:
    directory = root / relative
    _write_json(directory / "graph.template.json", {"nodes": []})
    _write_json(directory / "sweep.json", {"cases": [{"id": c} for c in case_ids]})
    if claims is not None:
        _write_json(directory / SWEEP_SUPPORT_NAME, {"version": 1, "claims": claims})
    return directory


def _record(
    bundle: str,
    engine: str = MIOPEN,
    verdict: str = SUPPORTED,
    case_id: str | None = None,
    graph: str | None = None,
    enforcement_level: str = "full",
) -> dict:
    record: dict = {
        "bundle": bundle,
        "case_id": case_id,
        "engine": engine,
        "verdict": verdict,
        "enforcement_level": enforcement_level,
    }
    if graph is not None:
        record["graph"] = graph
    return record


def _snapshot(
    path: Path,
    records: list[dict],
    target: tuple[str, str] = GFX942,
) -> Path:
    data = {
        "schema_version": 1,
        "target": {"arch": target[0], "platform": target[1]},
        "observations": records,
    }
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return path


def _run(
    root: Path,
    tmp_path: Path,
    records: list[dict],
    target: tuple[str, str] = GFX942,
    name: str = "obs.json",
):
    """Load one snapshot file and diff it against ``root``."""
    merged, hints, stats = load_snapshots([_snapshot(tmp_path / name, records, target)])
    return harvest(merged, hints, index_bundles(root), root), stats


def _run_multi(
    root: Path,
    tmp_path: Path,
    shards: list[tuple[list[dict], tuple[str, str]]],
):
    """Load multiple snapshot files (one per target) and diff against ``root``."""
    paths = []
    for i, (records, target) in enumerate(shards):
        paths.append(_snapshot(tmp_path / f"shard-{i}.json", records, target))
    merged, hints, stats = load_snapshots(paths)
    return harvest(merged, hints, index_bundles(root), root), stats


def _proposed(result, relative: str) -> dict:
    """The parsed JSON the tool would write for one sidecar."""
    for proposal in result.proposals:
        if proposal.relative_path.as_posix() == relative:
            return json.loads(render_sidecar(proposal.sidecar))
    raise AssertionError(
        f"no proposal for {relative}; got "
        f"{[p.relative_path.as_posix() for p in result.proposals]}"
    )


@pytest.fixture()
def bundle_root(tmp_path: Path) -> Path:
    root = tmp_path / "bundles"
    root.mkdir()
    return root


# --------------------------------------------------------------------------
# Record parsing
# --------------------------------------------------------------------------


class TestParseSnapshot:
    def _snap(self, records, target=GFX942, **overrides):
        data = {
            "schema_version": 1,
            "target": {"arch": target[0], "platform": target[1]},
            "observations": records,
        }
        data.update(overrides)
        return data

    def test_parses_a_well_formed_snapshot(self) -> None:
        parsed, errors = parse_snapshot(self._snap([_record("quick/A")]))
        assert not errors
        assert len(parsed) == 1
        key, verdict, graph = parsed[0]
        assert key == ObservationKey("quick/A", None, MIOPEN, "gfx942", "linux")
        assert verdict == SUPPORTED
        assert graph is None

    def test_carries_the_optional_graph_hint(self) -> None:
        parsed, errors = parse_snapshot(self._snap([_record("quick/A", graph="Small")]))
        assert not errors
        _, _, graph = parsed[0]
        assert graph == "Small"

    def test_case_id_becomes_part_of_the_key(self) -> None:
        parsed, errors = parse_snapshot(
            self._snap([_record("full/S", case_id="case_0")])
        )
        assert not errors
        key, _, _ = parsed[0]
        assert key.case_id == "case_0"
        assert key.unit == ("full/S", "case_0")

    def test_rejects_non_object_snapshot(self) -> None:
        with pytest.raises(ValueError, match="must be an object"):
            parse_snapshot([])

    def test_rejects_wrong_schema_version(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            parse_snapshot(self._snap([], schema_version=99))

    def test_rejects_zero_schema_version(self) -> None:
        with pytest.raises(ValueError, match="invalid schema_version"):
            parse_snapshot(self._snap([], schema_version=0))

    def test_rejects_string_schema_version(self) -> None:
        with pytest.raises(ValueError, match="invalid schema_version"):
            parse_snapshot(self._snap([], schema_version="1"))

    def test_future_schema_version_mentions_upgrade(self) -> None:
        with pytest.raises(ValueError, match="newer than this tool"):
            parse_snapshot(self._snap([], schema_version=2))

    def test_rejects_missing_target(self) -> None:
        with pytest.raises(ValueError, match="'target'"):
            parse_snapshot({"schema_version": 1, "observations": []})

    def test_flags_observation_missing_engine(self) -> None:
        bad_obs = {"bundle": "a", "verdict": "supported"}
        parsed, errors = parse_snapshot(self._snap([bad_obs]))
        assert not parsed
        assert any("'engine'" in e for e in errors)

    def test_flags_empty_bundle_name(self) -> None:
        bad_obs = {"bundle": "", "engine": "E", "verdict": "supported"}
        parsed, errors = parse_snapshot(self._snap([bad_obs]))
        assert not parsed
        assert any("'bundle'" in e for e in errors)

    def test_flags_an_unrecognised_verdict(self) -> None:
        parsed, errors = parse_snapshot(
            self._snap([_record("quick/A", verdict="maybe")])
        )
        assert not parsed
        assert any("unknown verdict" in e for e in errors)

    def test_rejects_a_platform_no_sidecar_may_contain(self) -> None:
        with pytest.raises(ValueError, match="invalid platform"):
            parse_snapshot(self._snap([], target=("gfx942", "freebsd")))

    def test_flags_an_unrecognised_enforcement_level(self) -> None:
        parsed, errors = parse_snapshot(
            self._snap([_record("quick/A", enforcement_level="paranoid")])
        )
        assert not parsed
        assert any("enforcement_level" in e for e in errors)

    def test_ignores_the_shape_of_provenance(self) -> None:
        """Never read here; a malformed one must not cost a real observation."""
        data = self._snap([_record("quick/A")], provenance="whatever")
        parsed, errors = parse_snapshot(data)
        assert not errors
        assert parsed[0][0].bundle == "quick/A"

    def test_good_observations_survive_alongside_bad_ones(self) -> None:
        """A single malformed observation must not discard valid siblings."""
        good = _record("quick/A")
        bad = {"bundle": "quick/B", "verdict": "supported"}  # missing engine
        parsed, errors = parse_snapshot(self._snap([good, bad]))
        assert len(parsed) == 1
        assert parsed[0][0].bundle == "quick/A"
        assert len(errors) == 1
        assert "'engine'" in errors[0]


# --------------------------------------------------------------------------
# Union across shards and runs
# --------------------------------------------------------------------------


class TestUnion:
    def test_supported_survives_a_later_decline(self, tmp_path: Path) -> None:
        """A flaky engine or a stale shard must not retract an accepted graph."""
        first = _snapshot(tmp_path / "a.json", [_record("quick/A")])
        second = _snapshot(tmp_path / "b.json", [_record("quick/A", verdict=DECLINED)])
        merged, _, _ = load_snapshots([first, second])
        assert list(merged.values()) == [SUPPORTED]

    def test_the_merge_does_not_depend_on_file_order(self, tmp_path: Path) -> None:
        first = _snapshot(tmp_path / "a.json", [_record("quick/A")])
        second = _snapshot(tmp_path / "b.json", [_record("quick/A", verdict=DECLINED)])
        forward, _, _ = load_snapshots([first, second])
        backward, _, _ = load_snapshots([second, first])
        assert forward == backward

    def test_unknown_never_displaces_an_answer(self, tmp_path: Path) -> None:
        path = _snapshot(
            tmp_path / "a.json",
            [
                _record("quick/A", verdict=DECLINED),
                _record("quick/A", verdict=UNKNOWN),
            ],
        )
        merged, _, _ = load_snapshots([path])
        assert list(merged.values()) == [DECLINED]

    def test_reading_the_same_file_twice_changes_nothing(self, tmp_path: Path) -> None:
        path = _snapshot(tmp_path / "a.json", [_record("quick/A")])
        once, _, _ = load_snapshots([path])
        twice, _, _ = load_snapshots([path, path])
        assert once == twice

    def test_targets_do_not_bleed_into_each_other(self, tmp_path: Path) -> None:
        """(bundle, case, engine, arch, platform) is the whole key."""
        gfx942 = _snapshot(tmp_path / "a.json", [_record("quick/A")], target=GFX942)
        gfx90a = _snapshot(
            tmp_path / "b.json",
            [_record("quick/A", verdict=DECLINED)],
            target=GFX90A,
        )
        merged, _, _ = load_snapshots([gfx942, gfx90a])
        assert len(merged) == 2
        assert {k.arch: v for k, v in merged.items()} == {
            "gfx942": SUPPORTED,
            "gfx90a": DECLINED,
        }

    def test_a_malformed_snapshot_costs_only_itself(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        good = _snapshot(tmp_path / "good.json", [_record("quick/A")])
        merged, _, stats = load_snapshots([bad, good])
        assert len(merged) == 1
        assert stats.files_failed == 1
        assert stats.records_valid == 1

    def test_an_unreadable_file_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        good = _snapshot(tmp_path / "a.json", [_record("quick/A")])
        merged, _, stats = load_snapshots([tmp_path / "missing.json", good])
        assert len(merged) == 1
        assert stats.files_read == 1
        assert stats.files_failed == 1

    def test_old_format_jsonl_is_detected(self, tmp_path: Path) -> None:
        old = tmp_path / "old.jsonl"
        old.write_text(
            '{"bundle":"quick/A","engine":"E","verdict":"supported"}\n'
            '{"bundle":"quick/B","engine":"E","verdict":"declined"}\n',
            encoding="utf-8",
        )
        good = _snapshot(tmp_path / "good.json", [_record("quick/A")])
        merged, _, stats = load_snapshots([old, good])
        assert len(merged) == 1
        assert stats.files_failed == 1

    def test_bad_observations_in_a_file_do_not_discard_good_ones(
        self, tmp_path: Path
    ) -> None:
        good = _record("quick/A")
        bad = {"bundle": "quick/B", "verdict": "supported"}  # missing engine
        path = tmp_path / "mixed.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "target": {"arch": "gfx942", "platform": "linux"},
                    "observations": [good, bad],
                }
            ),
            encoding="utf-8",
        )
        merged, _, stats = load_snapshots([path])
        assert len(merged) == 1
        assert stats.records_valid == 1
        assert stats.records_malformed == 1

    def test_missing_schema_version_is_detected(self, tmp_path: Path) -> None:
        bad = tmp_path / "noversion.json"
        bad.write_text(
            json.dumps({"target": {"arch": "gfx942", "platform": "linux"}}),
            encoding="utf-8",
        )
        good = _snapshot(tmp_path / "good.json", [_record("quick/A")])
        merged, _, stats = load_snapshots([bad, good])
        assert len(merged) == 1
        assert stats.files_failed == 1


# --------------------------------------------------------------------------
# The diff against committed claims
# --------------------------------------------------------------------------


class TestAdditions:
    def test_proposes_a_new_sidecar_for_an_unclaimed_bundle(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A")
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A")])
        assert _proposed(result, "quick/A/Small.support.json") == {
            "version": 1,
            "claims": {MIOPEN: {"gfx942": ["linux"]}},
        }

    def test_adds_a_target_without_disturbing_the_others(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A", claims={MIOPEN: {"gfx90a": ["linux"]}})
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A")])
        assert _proposed(result, "quick/A/Small.support.json")["claims"] == {
            MIOPEN: {"gfx90a": ["linux"], "gfx942": ["linux"]}
        }

    def test_adds_a_second_platform_to_an_existing_arch(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A", claims={MIOPEN: {"gfx1151": ["linux"]}})
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A")], target=GFX1151)
        assert _proposed(result, "quick/A/Small.support.json")["claims"] == {
            MIOPEN: {"gfx1151": ["linux", "windows"]}
        }

    def test_an_already_claimed_cell_proposes_nothing(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A", claims={MIOPEN: {"gfx942": ["linux"]}})
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A")])
        assert result.proposals == []
        assert result.stats.already_claimed == 1

    def test_a_second_engine_is_added_beside_the_first(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A", claims={MIOPEN: {"gfx942": ["linux"]}})
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A", engine=MLOPS)])
        assert _proposed(result, "quick/A/Small.support.json")["claims"] == {
            MIOPEN: {"gfx942": ["linux"]},
            MLOPS: {"gfx942": ["linux"]},
        }


class TestNeverDowngrades:
    def test_a_decline_against_a_claim_is_reported_not_applied(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A", claims={MIOPEN: {"gfx942": ["linux"]}})
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A", verdict=DECLINED)])
        assert result.proposals == []
        assert len(result.conflicts) == 1
        assert result.conflicts[0].key.engine == MIOPEN

    def test_a_decline_on_an_unclaimed_cell_writes_nothing(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        """Absence already means "not asserted"; there is no negative claim."""
        _single_bundle(bundle_root, "quick/A")
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A", verdict=DECLINED)])
        assert result.proposals == []
        assert result.conflicts == []
        assert result.stats.declined_no_claim == 1

    def test_unknown_is_not_evidence_in_either_direction(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A", claims={MIOPEN: {"gfx942": ["linux"]}})
        result, _ = _run(
            bundle_root,
            tmp_path,
            [
                _record("quick/A", verdict=UNKNOWN),
                _record("quick/A", engine=MLOPS, verdict=UNKNOWN),
            ],
        )
        assert result.proposals == []
        assert result.conflicts == []
        assert result.stats.unknown_discarded == 2

    def test_a_decline_elsewhere_does_not_block_an_addition(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        """The conflict path must not become an accidental veto."""
        _single_bundle(bundle_root, "quick/A", claims={MIOPEN: {"gfx942": ["linux"]}})
        result, _ = _run_multi(
            bundle_root,
            tmp_path,
            [
                ([_record("quick/A", verdict=DECLINED)], GFX942),
                ([_record("quick/A")], GFX90A),
            ],
        )
        assert len(result.conflicts) == 1
        assert _proposed(result, "quick/A/Small.support.json")["claims"] == {
            MIOPEN: {"gfx90a": ["linux"], "gfx942": ["linux"]}
        }

    def test_every_committed_cell_survives_the_proposal(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        """Monotonicity as a property, independent of which rule produced it."""
        committed = {
            MIOPEN: {"gfx942": ["linux"], "gfx90a": ["linux"]},
            MLOPS: {"gfx1151": ["windows"]},
        }
        _single_bundle(bundle_root, "quick/A", claims=committed)
        result, _ = _run_multi(
            bundle_root,
            tmp_path,
            [
                (
                    [
                        _record("quick/A", verdict=DECLINED),
                        _record("quick/A", engine=MLOPS, verdict=DECLINED),
                    ],
                    GFX942,
                ),
                ([_record("quick/A")], GFX1151),
            ],
        )
        proposed = _proposed(result, "quick/A/Small.support.json")["claims"]
        for engine, arch_map in committed.items():
            for arch, platforms in arch_map.items():
                assert set(platforms) <= set(proposed[engine][arch])


# --------------------------------------------------------------------------
# Sweep bundles
# --------------------------------------------------------------------------


class TestSweeps:
    def test_a_case_is_claimed_on_its_own(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _sweep_bundle(bundle_root, "full/S", ["case_0", "case_1"])
        result, _ = _run(bundle_root, tmp_path, [_record("full/S", case_id="case_0")])
        assert _proposed(result, "full/S/support.json")["claims"] == {
            MIOPEN: [{"cases": ["case_0"], "support": {"gfx942": ["linux"]}}]
        }

    def test_cases_with_the_same_footprint_share_a_group(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        """Mirrors regroupSweepClaims(): one group per distinct footprint."""
        _sweep_bundle(bundle_root, "full/S", ["case_0", "case_1", "case_2"])
        result, _ = _run_multi(
            bundle_root,
            tmp_path,
            [
                (
                    [
                        _record("full/S", case_id="case_0"),
                        _record("full/S", case_id="case_2"),
                    ],
                    GFX942,
                ),
                ([_record("full/S", case_id="case_1")], GFX90A),
            ],
        )
        assert _proposed(result, "full/S/support.json")["claims"] == {
            MIOPEN: [
                {"cases": ["case_0", "case_2"], "support": {"gfx942": ["linux"]}},
                {"cases": ["case_1"], "support": {"gfx90a": ["linux"]}},
            ]
        }

    def test_a_case_splits_out_of_its_group_when_it_gains_a_target(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _sweep_bundle(
            bundle_root,
            "full/S",
            ["case_0", "case_1"],
            claims={
                MIOPEN: [
                    {
                        "cases": ["case_0", "case_1"],
                        "support": {"gfx942": ["linux"]},
                    }
                ]
            },
        )
        result, _ = _run(
            bundle_root,
            tmp_path,
            [_record("full/S", case_id="case_1")],
            target=GFX90A,
        )
        assert _proposed(result, "full/S/support.json")["claims"] == {
            MIOPEN: [
                {"cases": ["case_0"], "support": {"gfx942": ["linux"]}},
                {
                    "cases": ["case_1"],
                    "support": {"gfx90a": ["linux"], "gfx942": ["linux"]},
                },
            ]
        }

    def test_groups_are_ordered_by_their_first_case(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _sweep_bundle(bundle_root, "full/S", ["a_case", "m_case", "z_case"])
        result, _ = _run_multi(
            bundle_root,
            tmp_path,
            [
                ([_record("full/S", case_id="z_case")], GFX942),
                ([_record("full/S", case_id="a_case")], GFX90A),
                ([_record("full/S", case_id="m_case")], GFX1151),
            ],
        )
        groups = _proposed(result, "full/S/support.json")["claims"][MIOPEN]
        assert [group["cases"][0] for group in groups] == [
            "a_case",
            "m_case",
            "z_case",
        ]

    def test_a_sibling_case_gains_nothing_from_its_neighbour(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _sweep_bundle(bundle_root, "full/S", ["case_0", "case_1"])
        result, _ = _run(bundle_root, tmp_path, [_record("full/S", case_id="case_0")])
        claimed_cases = {
            case
            for group in _proposed(result, "full/S/support.json")["claims"][MIOPEN]
            for case in group["cases"]
        }
        assert claimed_cases == {"case_0"}


# --------------------------------------------------------------------------
# Placing an observation on a sidecar
# --------------------------------------------------------------------------


class TestPlacement:
    def test_an_unknown_bundle_is_dropped_with_a_warning(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A renamed or deleted bundle leaves orphan records in old JSONL."""
        _single_bundle(bundle_root, "quick/A")
        result, _ = _run(bundle_root, tmp_path, [_record("quick/Gone")])
        assert result.proposals == []
        assert result.stats.unplaceable == 1
        assert "no such bundle" in capsys.readouterr().err

    def test_a_sweep_observed_without_a_case_id_is_dropped(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Applying it to every case would claim graphs nothing ran."""
        _sweep_bundle(bundle_root, "full/S", ["case_0"])
        result, _ = _run(bundle_root, tmp_path, [_record("full/S")])
        assert result.proposals == []
        assert "without a case_id" in capsys.readouterr().err

    def test_a_case_id_absent_from_the_sweep_is_dropped(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _sweep_bundle(bundle_root, "full/S", ["case_0"])
        result, _ = _run(bundle_root, tmp_path, [_record("full/S", case_id="case_9")])
        assert result.proposals == []
        assert "case id not found" in capsys.readouterr().err

    def test_a_case_id_on_a_single_graph_bundle_is_dropped(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _single_bundle(bundle_root, "quick/A")
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A", case_id="case_0")])
        assert result.proposals == []
        assert "case_id on a single-graph bundle" in capsys.readouterr().err

    def test_an_ambiguous_directory_needs_the_graph_field(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two graphs, one directory: guessing writes into the wrong sidecar."""
        _single_bundle(bundle_root, "quick/A", name="First")
        _single_bundle(bundle_root, "quick/A", name="Second")
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A")])
        assert result.proposals == []
        assert "names none of them" in capsys.readouterr().err

    def test_the_graph_field_resolves_the_ambiguity(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A", name="First")
        _single_bundle(bundle_root, "quick/A", name="Second")
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A", graph="Second")])
        assert [p.relative_path.as_posix() for p in result.proposals] == [
            "quick/A/Second.support.json"
        ]


# --------------------------------------------------------------------------
# Malformed committed sidecars
# --------------------------------------------------------------------------


class TestBrokenSidecars:
    def test_an_unparseable_sidecar_is_read_as_empty_not_skipped(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        """Otherwise the tree's least trustworthy files never get fixed."""
        directory = _single_bundle(bundle_root, "quick/A")
        (directory / "Small.support.json").write_text("{ broken", encoding="utf-8")
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A")])
        assert _proposed(result, "quick/A/Small.support.json")["claims"] == {
            MIOPEN: {"gfx942": ["linux"]}
        }

    def test_a_duplicated_case_id_keeps_both_footprints(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        """A verifier error, but reading it as "last group wins" would delete a
        committed claim -- the one thing this tool must never do."""
        _sweep_bundle(
            bundle_root,
            "full/S",
            ["case_0"],
            claims={
                MIOPEN: [
                    {"cases": ["case_0"], "support": {"gfx942": ["linux"]}},
                    {"cases": ["case_0"], "support": {"gfx90a": ["linux"]}},
                ]
            },
        )
        sidecar = read_sidecar(bundle_root / "full/S" / SWEEP_SUPPORT_NAME, True)
        assert sidecar.cells[MIOPEN]["case_0"] == {GFX942, GFX90A}


# --------------------------------------------------------------------------
# Canonical serialisation
# --------------------------------------------------------------------------


class TestCanonicalForm:
    def test_matches_dump_canonical(self, bundle_root: Path, tmp_path: Path) -> None:
        """Sorted keys, two-space indent, trailing newline."""
        _single_bundle(bundle_root, "quick/A")
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A")])
        text = render_sidecar(result.proposals[0].sidecar)
        assert text == (
            "{\n"
            '  "claims": {\n'
            '    "MIOPEN_ENGINE": {\n'
            '      "gfx942": [\n'
            '        "linux"\n'
            "      ]\n"
            "    }\n"
            "  },\n"
            '  "version": 1\n'
            "}\n"
        )

    def test_an_untouched_sidecar_round_trips_byte_for_byte(
        self, bundle_root: Path
    ) -> None:
        """No diff churn: a file the tool did not change must re-serialise to
        itself, or one added platform token arrives buried in a reformat."""
        _single_bundle(bundle_root, "quick/A")
        original = (
            "{\n"
            '  "claims": {\n'
            '    "MIOPEN_ENGINE": {\n'
            '      "gfx1151": [\n'
            '        "windows"\n'
            "      ],\n"
            '      "gfx90a": [\n'
            '        "linux"\n'
            "      ]\n"
            "    }\n"
            "  },\n"
            '  "version": 1\n'
            "}\n"
        )
        path = bundle_root / "quick/A/Small.support.json"
        path.write_text(original, encoding="utf-8")
        assert render_sidecar(read_sidecar(path, False)) == original

    def test_the_version_of_the_committed_file_is_preserved(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A")
        path = bundle_root / "quick/A/Small.support.json"
        path.write_text(json.dumps({"version": 2, "claims": {}}), encoding="utf-8")
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A")])
        assert _proposed(result, "quick/A/Small.support.json")["version"] == 2


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


class TestCoverage:
    def test_counts_units_not_bundles(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A sweep is one directory and many graphs. Counting the directory
        would score a shard that reached one case as having covered them all."""
        _sweep_bundle(bundle_root, "full/S", ["case_0", "case_1", "case_2"])
        _single_bundle(bundle_root, "quick/A")
        result, _ = _run(bundle_root, tmp_path, [_record("full/S", case_id="case_0")])
        assert result.total_units == 4
        report_coverage(result, __import__("sys").stdout)
        assert "1 / 4 units observed (25.0%)" in capsys.readouterr().out

    def test_repeated_engines_on_one_unit_count_once(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A")
        result, _ = _run(
            bundle_root,
            tmp_path,
            [_record("quick/A"), _record("quick/A", engine=MLOPS)],
        )
        assert len(result.observed_units[GFX942]) == 1

    def test_an_errored_query_still_counts_as_a_visit(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        """ "Did any shard reach gfx1151/windows" is a question about the run,
        not about the answer the engine gave."""
        _single_bundle(bundle_root, "quick/A")
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A", verdict=UNKNOWN)])
        assert result.observed_units[GFX942] == {("quick/A", None)}

    def test_a_target_claimed_but_unvisited_still_gets_a_row(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Zero coverage on an ASIC is the finding; omitting the row hides it."""
        _single_bundle(
            bundle_root, "quick/A", claims={MIOPEN: {"gfx1151": ["windows"]}}
        )
        result, _ = _run(bundle_root, tmp_path, [_record("quick/A")])
        report_coverage(result, __import__("sys").stdout)
        out = capsys.readouterr().out
        assert "gfx1151 / windows" in out
        assert "0 / 1 units observed (0.0%)" in out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class TestCli:
    def _args(self, bundle_root: Path, observations: Path, *extra: str) -> list[str]:
        return [
            "--observations",
            str(observations),
            "--bundles-dir",
            str(bundle_root),
            *extra,
        ]

    def test_dry_run_prints_a_diff_and_writes_nothing(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _single_bundle(bundle_root, "quick/A")
        observations = _snapshot(tmp_path / "obs.json", [_record("quick/A")])
        assert main(self._args(bundle_root, observations, "--dry-run")) == 0
        out = capsys.readouterr().out
        assert "+++ b/quick/A/Small.support.json" in out
        assert not (bundle_root / "quick/A/Small.support.json").exists()

    def test_writes_proposals_under_the_output_dir(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(bundle_root, "quick/A")
        observations = _snapshot(tmp_path / "obs.json", [_record("quick/A")])
        output = tmp_path / "proposed"
        assert (
            main(self._args(bundle_root, observations, "--output-dir", str(output)))
            == 0
        )
        assert (output / "quick/A/Small.support.json").is_file()

    def test_never_touches_the_committed_tree(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        """The human gate is the point: a tool that can write to the tree can be
        wired into a job that commits to it."""
        _single_bundle(bundle_root, "quick/A", claims={MIOPEN: {"gfx90a": ["linux"]}})
        before = {
            path: path.read_bytes() for path in bundle_root.rglob("*") if path.is_file()
        }
        observations = _snapshot(tmp_path / "obs.json", [_record("quick/A")])
        main(self._args(bundle_root, observations, "--output-dir", str(tmp_path / "p")))
        after = {
            path: path.read_bytes() for path in bundle_root.rglob("*") if path.is_file()
        }
        assert after == before

    def test_output_dir_is_required_without_dry_run(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        observations = _snapshot(tmp_path / "obs.json", [_record("quick/A")])
        assert main(self._args(bundle_root, observations)) == 1

    def test_a_missing_bundle_tree_fails(self, tmp_path: Path) -> None:
        observations = _snapshot(tmp_path / "obs.json", [_record("quick/A")])
        assert main(self._args(tmp_path / "nope", observations, "--dry-run")) == 1

    def test_no_readable_observation_file_fails(self, bundle_root: Path) -> None:
        assert (
            main(self._args(bundle_root, bundle_root / "missing.json", "--dry-run"))
            == 1
        )

    def test_conflicts_do_not_fail_the_run(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Non-zero here would block the pipeline meant to surface them."""
        _single_bundle(bundle_root, "quick/A", claims={MIOPEN: {"gfx942": ["linux"]}})
        observations = _snapshot(
            tmp_path / "obs.json", [_record("quick/A", verdict=DECLINED)]
        )
        assert main(self._args(bundle_root, observations, "--dry-run")) == 0
        assert "=== Conflicts (1) ===" in capsys.readouterr().out

    def test_coverage_report_is_opt_in(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _single_bundle(bundle_root, "quick/A")
        observations = _snapshot(tmp_path / "obs.json", [_record("quick/A")])
        main(self._args(bundle_root, observations, "--dry-run"))
        assert "Observation Coverage" not in capsys.readouterr().out
        main(self._args(bundle_root, observations, "--dry-run", "--coverage-report"))
        assert "Observation Coverage" in capsys.readouterr().out

    def test_a_retired_engine_is_flagged(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _single_bundle(bundle_root, "quick/A")
        observations = _snapshot(
            tmp_path / "obs.json", [_record("quick/A", engine="NOT_AN_ENGINE")]
        )
        main(self._args(bundle_root, observations, "--dry-run"))
        assert "NOT_AN_ENGINE" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The committed tree
# --------------------------------------------------------------------------


def _walk_sidecar_cells(root: Path) -> set[tuple[str, str, str, str, str]]:
    """Every claimed cell in the tree, read by a route that shares no code with
    ``read_sidecar`` -- a plain rglob and a literal walk of the two shapes.

    The flatten/regroup round trip is where a silent claim loss would hide, and
    a bug there would corrupt every proposal identically. A crosscheck written
    in terms of the same helper would agree with the bug.
    """
    cells = set()
    for path in root.rglob("*.json"):
        if path.name == SWEEP_SUPPORT_NAME:
            is_sweep = True
        elif path.name.endswith(".support.json"):
            is_sweep = False
        else:
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(root).as_posix()
        for engine, value in document.get("claims", {}).items():
            groups = value if is_sweep else [{"cases": [""], "support": value}]
            for group in groups:
                for case in group["cases"]:
                    for arch, platforms in group["support"].items():
                        for platform in platforms:
                            cells.add((relative, engine, case, arch, platform))
    return cells


@pytest.mark.skipif(
    not DEFAULT_BUNDLES_DIR.is_dir(),
    reason="bundle tree not present in this checkout",
)
class TestRealBundleTree:
    @staticmethod
    @pytest.fixture(scope="class")
    def entries() -> dict:
        return index_bundles(DEFAULT_BUNDLES_DIR)

    def _sidecar_paths(self, entries: dict) -> list[tuple[Path, bool]]:
        paths = []
        for entry in entries.values():
            candidates = (
                [entry.directory / SWEEP_SUPPORT_NAME]
                if entry.is_sweep
                else [entry.directory / f"{stem}.support.json" for stem in entry.graphs]
            )
            paths.extend(
                (path, entry.is_sweep) for path in candidates if path.is_file()
            )
        return paths

    def test_the_tree_has_sidecars_to_check(self, entries: dict) -> None:
        assert self._sidecar_paths(entries), "no committed sidecars found"

    def test_reading_loses_no_claim(self, entries: dict) -> None:
        loaded = set()
        for path, is_sweep in self._sidecar_paths(entries):
            relative = path.relative_to(DEFAULT_BUNDLES_DIR).as_posix()
            for engine, by_case in read_sidecar(path, is_sweep).cells.items():
                for case_id, pairs in by_case.items():
                    for arch, platform in pairs:
                        loaded.add((relative, engine, case_id or "", arch, platform))
        assert loaded == _walk_sidecar_cells(DEFAULT_BUNDLES_DIR)

    def test_every_committed_sidecar_is_a_fixed_point(self, entries: dict) -> None:
        """The C++ writer produced these files. If re-serialising one changes a
        byte, every proposal against the tree arrives full of reformat noise.
        """
        for path, is_sweep in self._sidecar_paths(entries):
            rendered = render_sidecar(read_sidecar(path, is_sweep))
            assert rendered == path.read_text(encoding="utf-8"), path

    def test_observations_matching_the_tree_propose_nothing(
        self, entries: dict, tmp_path: Path
    ) -> None:
        """Replaying the tree's own claims back as observations is a no-op.

        This is the idempotence the harvest pipeline needs: rerunning it on CI
        output that has already been merged must produce an empty diff, not a
        pull request that reasserts what is already committed.
        """
        by_target: dict[tuple[str, str], list[dict]] = {}
        for relative, engine, case, arch, platform in sorted(
            _walk_sidecar_cells(DEFAULT_BUNDLES_DIR)
        ):
            directory = Path(relative).parent.as_posix()
            by_target.setdefault((arch, platform), []).append(
                _record(
                    directory,
                    engine=engine,
                    verdict=SUPPORTED,
                    case_id=case or None,
                )
            )
        paths = [
            _snapshot(tmp_path / f"snap-{i}.json", records, target)
            for i, (target, records) in enumerate(by_target.items())
        ]
        merged, hints, _ = load_snapshots(paths)
        result = harvest(merged, hints, entries, DEFAULT_BUNDLES_DIR)
        assert result.proposals == []
        assert result.conflicts == []

    def test_the_tree_places_every_observation_it_generates(
        self, entries: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No warnings from a replay means bundle paths, case ids, and the
        single-graph/sweep split all agree between the tree and this indexer."""
        by_target: dict[tuple[str, str], list[dict]] = {}
        for relative, engine, case, arch, platform in sorted(
            _walk_sidecar_cells(DEFAULT_BUNDLES_DIR)
        ):
            directory = Path(relative).parent.as_posix()
            by_target.setdefault((arch, platform), []).append(
                _record(
                    directory,
                    engine=engine,
                    case_id=case or None,
                )
            )
        paths = [
            _snapshot(tmp_path / f"snap-{i}.json", records, target)
            for i, (target, records) in enumerate(by_target.items())
        ]
        merged, hints, _ = load_snapshots(paths)
        capsys.readouterr()
        result = harvest(merged, hints, entries, DEFAULT_BUNDLES_DIR)
        assert capsys.readouterr().err == ""
        assert result.stats.unplaceable == 0

    def test_every_claimed_engine_is_registered(self, entries: dict) -> None:
        """The renderer warns about these; a clean tree is what makes the
        warning worth reading when one does appear."""
        from engine_registry import known_engines

        registry = known_engines()
        if registry is None:
            pytest.skip("EngineNames.hpp unreadable")
        claimed = {
            engine
            for path, is_sweep in self._sidecar_paths(entries)
            for engine in read_sidecar(path, is_sweep).cells
        }
        assert claimed <= registry, sorted(claimed - registry)
