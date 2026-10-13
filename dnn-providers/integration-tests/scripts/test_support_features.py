#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the support_features package: feature coverage per op.

Three kinds of check:

- **drift** -- the registry against the schema's ``NodeAttributes`` union, and
  every catalogued op against its schema table, so an API addition fails here
  until it is catalogued or explicitly set aside;
- **join** -- ``coverage.py`` on small trees in ``tmp_path`` with a stand-in op
  module, so the counting rules are pinned without depending on any real
  catalog;
- **real tree** -- the committed bundles extract with no defect.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from support_features import Feature, FeatureError, registry
from support_features.coverage import (
    DEFAULT_BUNDLES_DIR,
    compute_coverage,
    main,
    render_markdown,
    share,
)
from support_matrix_units import ClaimUnit, collect_units

# --------------------------------------------------------------------------
# Schema reading
# --------------------------------------------------------------------------


def _schema_text(name: str) -> str:
    path = registry.SCHEMA_DIR / name
    # The schemas are the denominator. Missing, every drift check would pass
    # vacuously, so this fails instead of skipping.
    assert path.is_file(), f"schema not found: {path}"
    return re.sub(r"//[^\n]*", "", path.read_text())


def _block(text: str, keyword: str, name: str) -> str:
    match = re.search(rf"\b{keyword}\s+{name}\s*\{{(.*?)\}}", text, re.S)
    assert match, f"{keyword} {name} not found"
    return match.group(1)


def union_members(name: str, file: str = "graph.fbs") -> set[str]:
    body = _block(_schema_text(file), "union", name)
    return {member.strip() for member in body.split(",") if member.strip()}


def table_fields(file: str, table: str) -> set[str]:
    body = _block(_schema_text(file), "table", table)
    return set(re.findall(r"^\s*(\w+)\s*:", body, re.M))


def catalog_problems(module, schema_fields: set[str]) -> list[str]:
    """Every way ``module``'s catalog fails to account for its schema table.

    Each field drives at least one feature or is set aside in ``NOT_FEATURES``
    with a reason -- never both, never neither. A field doing neither is API
    surface the percentage silently leaves out of its denominator.
    """
    problems = []
    feature_fields: set[str] = set()
    ids = [feature.id for feature in module.FEATURES]
    if len(ids) != len(set(ids)):
        problems.append(f"duplicate feature ids: {sorted(ids)}")
    for feature in module.FEATURES:
        if ":" not in feature.id:
            problems.append(f"{feature.id}: id is not '<group>:<value>'")
        if not feature.fields:
            problems.append(f"{feature.id}: names no schema field")
        feature_fields |= set(feature.fields)
    set_aside = set(module.NOT_FEATURES)
    for name in sorted(feature_fields - schema_fields):
        problems.append(f"{name}: a feature names it, but the schema has no such field")
    for name in sorted(set_aside - schema_fields):
        problems.append(f"{name}: in NOT_FEATURES, but the schema has no such field")
    for name in sorted(feature_fields & set_aside):
        problems.append(f"{name}: both drives a feature and is in NOT_FEATURES")
    for name in sorted(schema_fields - feature_fields - set_aside):
        problems.append(f"{name}: neither drives a feature nor is in NOT_FEATURES")
    for name, reason in module.NOT_FEATURES.items():
        if not reason.strip():
            problems.append(f"{name}: in NOT_FEATURES with no reason")
    return problems


# --------------------------------------------------------------------------
# Drift: registry and catalogs against the schema
# --------------------------------------------------------------------------


class TestRegistry:
    def test_every_api_node_type_is_catalogued_or_pending(self) -> None:
        assert set(registry.OPS) | registry.PENDING == union_members("NodeAttributes")

    def test_no_node_type_is_both(self) -> None:
        assert not set(registry.OPS) & registry.PENDING

    def test_op_modules_are_keyed_by_their_own_node_type(self) -> None:
        for node_type, module in registry.OPS.items():
            assert module.NODE_TYPE == node_type

    def test_pending_type_has_no_module(self) -> None:
        assert registry.op_module(next(iter(registry.PENDING))) is None

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(FeatureError, match="NotAnOpAttributes"):
            registry.op_module("NotAnOpAttributes")


@pytest.mark.parametrize("node_type", sorted(registry.OPS))
def test_catalog_accounts_for_every_schema_field(node_type: str) -> None:
    module = registry.OPS[node_type]
    fields = table_fields(module.SCHEMA_FILE, module.SCHEMA_TABLE)
    assert catalog_problems(module, fields) == []


class TestCatalogProblems:
    """The exhaustiveness check itself, so a catalog passing it means something."""

    FIELDS = {"a", "b", "c"}

    def _module(self, features, not_features) -> SimpleNamespace:
        return SimpleNamespace(FEATURES=features, NOT_FEATURES=not_features)

    def test_fully_accounted_catalog_passes(self) -> None:
        module = self._module(
            (Feature("g:x", ("a",)), Feature("g:y", ("a", "b"))), {"c": "a reason"}
        )
        assert catalog_problems(module, self.FIELDS) == []

    def test_unaccounted_field_is_named(self) -> None:
        module = self._module((Feature("g:x", ("a",)),), {"b": "a reason"})
        (problem,) = catalog_problems(module, self.FIELDS)
        assert problem.startswith("c: neither")

    def test_field_in_both_lists_is_named(self) -> None:
        module = self._module(
            (Feature("g:x", ("a", "b")), Feature("g:y", ("c",))), {"a": "a reason"}
        )
        (problem,) = catalog_problems(module, self.FIELDS)
        assert problem.startswith("a: both")

    @pytest.mark.parametrize("where", ["feature", "not_feature"])
    def test_field_missing_from_schema_is_named(self, where: str) -> None:
        features = [Feature("g:x", ("a", "b", "c"))]
        not_features = {}
        if where == "feature":
            features.append(Feature("g:y", ("gone",)))
        else:
            not_features["gone"] = "a reason"
        (problem,) = catalog_problems(
            self._module(tuple(features), not_features), self.FIELDS
        )
        assert problem.startswith("gone: ")

    def test_feature_without_fields_and_bad_id_are_named(self) -> None:
        module = self._module(
            (Feature("g:x", ("a", "b", "c")), Feature("nogroup", ())), {}
        )
        assert catalog_problems(module, self.FIELDS) == [
            "nogroup: id is not '<group>:<value>'",
            "nogroup: names no schema field",
        ]

    def test_empty_reason_is_named(self) -> None:
        module = self._module((Feature("g:x", ("a", "b")),), {"c": " "})
        assert catalog_problems(module, self.FIELDS) == [
            "c: in NOT_FEATURES with no reason"
        ]


# --------------------------------------------------------------------------
# Join: coverage.py on small trees, with a stand-in op
# --------------------------------------------------------------------------

FAKE_TYPE = "SdpaAttributes"
PENDING_TYPE = "PointwiseAttributes"


def _fake_extract(node: dict, tensors_by_uid: dict) -> set[str]:
    """Features are listed on the node itself; ``tensors`` must be resolved."""
    if "boom" in node:
        raise FeatureError(node["boom"])
    assert set(tensors_by_uid) == {1}
    return set(node.get("features", ()))


FAKE_OP = SimpleNamespace(
    NODE_TYPE=FAKE_TYPE,
    SCHEMA_FILE="unused.fbs",
    SCHEMA_TABLE="Unused",
    FEATURES=(
        Feature("dtype:fp16", ("q",)),
        Feature("dtype:bf16", ("q",)),
        Feature("mask:causal", ("causal",)),
        Feature("mask:sliding", ("left", "right")),
    ),
    NOT_FEATURES={},
    extract=_fake_extract,
)


@pytest.fixture()
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "OPS", {FAKE_TYPE: FAKE_OP})
    monkeypatch.setattr(registry, "PENDING", registry.PENDING - {FAKE_TYPE})


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _node(node_type: str = FAKE_TYPE, **fields) -> dict:
    return {"type": node_type, **fields}


def _bundle(root: Path, name: str, nodes: list[dict], claims: dict | None = None):
    directory = root / "quick" / "SdpaFwd" / name
    graph = {
        "tensors": [{"uid": 1, "dims": [1], "data_type": "half"}],
        "nodes": nodes,
    }
    _write_json(directory / f"{name}.json", graph)
    if claims is not None:
        _write_json(
            directory / f"{name}.support.json", {"version": 1, "claims": claims}
        )


def _coverage(root: Path):
    errors: list[str] = []
    coverage = compute_coverage(collect_units(root, errors), errors)
    return coverage, errors


@pytest.mark.usefixtures("fake_registry")
class TestJoin:
    def test_tested_counts_every_graph_claimed_or_not(self, tmp_path: Path) -> None:
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16", "mask:causal"])])
        _bundle(
            tmp_path,
            "b",
            [_node(features=["dtype:bf16"])],
            claims={"E": {"gfx942": ["linux"]}},
        )
        coverage, errors = _coverage(tmp_path)
        assert errors == []
        op = coverage.ops[FAKE_TYPE]
        assert op.graphs == 2
        assert op.tested_count() == 3
        assert op.tested == {"dtype:fp16": 1, "mask:causal": 1, "dtype:bf16": 1}

    def test_claimed_counts_only_claimed_graphs_per_arch(self, tmp_path: Path) -> None:
        _bundle(
            tmp_path,
            "a",
            [_node(features=["dtype:fp16", "mask:causal"])],
            claims={"E1": {"gfx942": ["linux"]}, "E2": {"gfx950": ["linux"]}},
        )
        _bundle(
            tmp_path,
            "b",
            [_node(features=["dtype:bf16"])],
            claims={"E1": {"gfx942": ["linux", "windows"]}},
        )
        coverage, _ = _coverage(tmp_path)
        op = coverage.ops[FAKE_TYPE]
        assert coverage.archs == ["gfx942", "gfx950"]
        assert op.claimed_count("gfx942") == 3
        assert op.claimed_count("gfx950") == 2
        # Two engines and two platforms on one arch are still one graph.
        assert op.claimed_graphs == {"gfx942": 2, "gfx950": 1}

    def test_arch_claimed_only_by_another_op_reads_zero(self, tmp_path: Path) -> None:
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16"])])
        _bundle(
            tmp_path,
            "b",
            [_node(PENDING_TYPE)],
            claims={"E": {"gfx90a": ["linux"]}},
        )
        coverage, _ = _coverage(tmp_path)
        assert coverage.archs == ["gfx90a"]
        assert coverage.ops[FAKE_TYPE].claimed_count("gfx90a") == 0
        assert "| SdpaAttributes | 1 | 1/4 (25%) | 0/4 (0%) |" in render_markdown(
            coverage
        )

    def test_nodes_of_one_op_in_one_graph_count_once(self, tmp_path: Path) -> None:
        _bundle(
            tmp_path,
            "a",
            [
                _node(features=["dtype:fp16"]),
                _node(features=["dtype:fp16", "mask:causal"]),
            ],
        )
        op = _coverage(tmp_path)[0].ops[FAKE_TYPE]
        assert op.graphs == 1
        assert op.tested == {"dtype:fp16": 1, "mask:causal": 1}

    def test_fused_graph_counts_toward_each_op(self, tmp_path: Path) -> None:
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16"]), _node(PENDING_TYPE)])
        coverage, _ = _coverage(tmp_path)
        assert coverage.ops[FAKE_TYPE].graphs == 1
        assert coverage.pending_graphs[PENDING_TYPE] == 1

    def test_pending_types_are_listed_even_with_no_graphs(self, tmp_path: Path) -> None:
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16"])])
        coverage, _ = _coverage(tmp_path)
        assert set(coverage.pending_graphs) == registry.PENDING
        assert "CustomOpAttributes (0)" in render_markdown(coverage)

    def test_extract_error_is_reported_with_its_location(self, tmp_path: Path) -> None:
        _bundle(tmp_path, "a", [_node(boom="head dim 96 is not catalogued")])
        coverage, errors = _coverage(tmp_path)
        assert errors == ["SdpaFwd/a: SdpaAttributes: head dim 96 is not catalogued"]
        assert coverage.ops[FAKE_TYPE].graphs == 0

    def test_uncatalogued_feature_from_extract_is_reported(
        self, tmp_path: Path
    ) -> None:
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16", "mask:alibi"])])
        coverage, (error,) = _coverage(tmp_path)
        assert "catalog does not list: ['mask:alibi']" in error
        assert coverage.ops[FAKE_TYPE].graphs == 0

    def test_unknown_node_type_is_reported(self, tmp_path: Path) -> None:
        _bundle(tmp_path, "a", [_node("NotAnOpAttributes")])
        _, (error,) = _coverage(tmp_path)
        assert "unknown node type 'NotAnOpAttributes'" in error


class TestMarkdown:
    def test_share(self) -> None:
        assert share(1, 3) == "1/3 (33%)"
        assert share(0, 0) == "-"

    def test_per_feature_table_lists_every_catalogued_feature(
        self, tmp_path: Path, fake_registry: None
    ) -> None:
        _bundle(
            tmp_path,
            "a",
            [_node(features=["dtype:fp16"])],
            claims={"E": {"gfx942": ["linux"]}},
        )
        document = render_markdown(_coverage(tmp_path)[0])
        section = document.split("## SdpaAttributes", 1)[1]
        assert "| *(all graphs)* | 1 | 1 |" in section
        assert "| dtype:fp16 | 1 | 1 |" in section
        assert "| mask:sliding | 0 | 0 |" in section


@pytest.mark.usefixtures("fake_registry")
class TestCli:
    def test_missing_directory_exits_two(self, tmp_path: Path) -> None:
        assert main(["--bundles-dir", str(tmp_path / "absent")]) == 2

    def test_report_goes_to_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16"])])
        assert main(["--bundles-dir", str(tmp_path)]) == 0
        assert capsys.readouterr().out.startswith("# API feature coverage per op")

    def test_defect_exits_one_and_reports_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16"])])
        _bundle(tmp_path, "b", [_node(boom="bad")])
        assert main(["--bundles-dir", str(tmp_path)]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "error: 1 defect(s)" in captured.err


# --------------------------------------------------------------------------
# Real tree
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def units() -> list[ClaimUnit]:
    return collect_units(DEFAULT_BUNDLES_DIR)


class TestRealBundleTree:
    def test_every_graph_extracts_without_defect(self, units: list[ClaimUnit]) -> None:
        errors: list[str] = []
        compute_coverage(units, errors)
        assert errors == []

    def test_every_graph_is_counted_for_some_node_type(
        self, units: list[ClaimUnit]
    ) -> None:
        coverage = compute_coverage(units)
        counted = sum(op.graphs for op in coverage.ops.values()) + sum(
            coverage.pending_graphs.values()
        )
        assert units
        assert counted >= len(units)
