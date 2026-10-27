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
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from support_features import (
    Feature,
    FeatureError,
    batchnorm,
    batchnorm_backward,
    batchnorm_inference,
    batchnorm_inference_variance_ext,
    batchnorm_training,
    block_scale_dequantize,
    block_scale_quantize,
    convolution,
    convolution_bwd,
    convolution_fwd,
    convolution_wrw,
    custom_op,
    gdn,
    kda,
    layernorm,
    layernorm_backward,
    matmul,
    moe_grouped_matmul,
    moe_grouped_matmul_bwd,
    pointwise,
    reduction,
    registry,
    resample_bwd,
    resample_fwd,
    rmsnorm,
    rmsnorm_backward,
    sdpa,
    scalar_source,
    sdpa_backward,
)
from support_features.coverage import (
    DEFAULT_BUNDLES_DIR,
    compute_coverage,
    main,
    render_markdown,
    resolve_node_type,
    share,
)
from support_features.feature_matrix import (
    DATA_CENTER,
    DEFAULT_OVERVIEW_PATH,
    FAMILIES,
    GPUS,
    compute_feature_claims,
    group_of,
    heading_anchor,
    percent_cell,
)
from support_features.feature_matrix import main as feature_matrix_main
from support_features.feature_matrix import render as render_feature_matrix
from support_matrix_units import (
    DTYPE_LABELS,
    ClaimUnit,
    collect_units,
    node_label,
    node_type_label,
)

# --------------------------------------------------------------------------
# Schema reading
# --------------------------------------------------------------------------


def _schema_text(name: str) -> str:
    path = registry.SCHEMA_DIR / name
    # The schemas are the denominator. Missing, every drift check would pass
    # vacuously, so this fails instead of skipping.
    assert path.is_file(), f"schema not found: {path}"
    return re.sub(r"//[^\n]*", "", path.read_text(encoding="utf-8"))


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
    def test_every_api_node_type_is_catalogued(self) -> None:
        assert set(registry.OPS) == union_members("NodeAttributes")

    def test_op_modules_are_keyed_by_their_own_node_type(self) -> None:
        for node_type, module in registry.OPS.items():
            assert module.NODE_TYPE == node_type

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(FeatureError, match="NotAnOpAttributes"):
            registry.op_module("NotAnOpAttributes")


@pytest.mark.parametrize("node_type", sorted(registry.OPS))
def test_catalog_accounts_for_every_schema_field(node_type: str) -> None:
    module = registry.OPS[node_type]
    fields = table_fields(module.SCHEMA_FILE, module.SCHEMA_TABLE)
    assert catalog_problems(module, fields) == []


class TestDataTypeLabels:
    """The matrix's dtype labels, and every catalog's, against the schema."""

    def test_serializer_writes_every_schema_enumerator(self) -> None:
        assert set(_serialized_enum("DataType")) == _schema_enum(
            "data_types.fbs", "DataType"
        )

    def test_every_serialized_dtype_has_a_label(self) -> None:
        serialized = set(_serialized_enum("DataType").values()) - {"unset"}
        assert set(DTYPE_LABELS) == serialized

    def test_labels_are_distinct(self) -> None:
        # Two dtypes on one label would merge their rows: FNUZ fp8 into OCP.
        assert len(set(DTYPE_LABELS.values())) == len(DTYPE_LABELS)

    @pytest.mark.parametrize(
        "name, mapping",
        [
            ("block_scale_dequantize.DTYPES", block_scale_dequantize.DTYPES),
            (
                "block_scale_dequantize.SCALE_DTYPES",
                block_scale_dequantize.SCALE_DTYPES,
            ),
            ("block_scale_quantize.DTYPES", block_scale_quantize.DTYPES),
            ("sdpa.DTYPES", sdpa.DTYPES),
            ("sdpa_backward.DTYPES", sdpa_backward.DTYPES),
        ],
    )
    def test_catalog_map_agrees_with_the_matrix(self, name: str, mapping) -> None:
        # A catalog names a dtype as the matrix does, so one view's fp4e2m1
        # is the other's.
        assert {raw: DTYPE_LABELS.get(raw) for raw in mapping} == mapping, name

    @pytest.mark.parametrize(
        "name, labels",
        [
            ("convolution.DTYPES", convolution.DTYPES),
            ("pointwise._PORT_DTYPES", pointwise._PORT_DTYPES),
        ],
    )
    def test_catalog_labels_are_matrix_labels(self, name: str, labels) -> None:
        assert set(labels) <= set(DTYPE_LABELS.values()), name


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
OTHER_TYPE = "CustomOpAttributes"


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

# A second op, for graphs that fuse two or claim an arch for only one.
OTHER_OP = SimpleNamespace(
    NODE_TYPE=OTHER_TYPE,
    SCHEMA_FILE="unused.fbs",
    SCHEMA_TABLE="Unused",
    FEATURES=(Feature("data:empty", ("data",)), Feature("data:present", ("data",))),
    NOT_FEATURES={},
    extract=_fake_extract,
)


@pytest.fixture()
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "OPS", {FAKE_TYPE: FAKE_OP, OTHER_TYPE: OTHER_OP})


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
            [_node(OTHER_TYPE, features=["data:empty"])],
            claims={"E": {"gfx90a": ["linux"]}},
        )
        coverage, _ = _coverage(tmp_path)
        assert coverage.archs == ["gfx90a"]
        assert coverage.ops[FAKE_TYPE].claimed_count("gfx90a") == 0
        assert coverage.ops[OTHER_TYPE].claimed_count("gfx90a") == 1
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
        _bundle(
            tmp_path,
            "a",
            [
                _node(features=["dtype:fp16"]),
                _node(OTHER_TYPE, features=["data:present"]),
            ],
        )
        coverage, errors = _coverage(tmp_path)
        assert errors == []
        assert coverage.ops[FAKE_TYPE].tested == {"dtype:fp16": 1}
        assert coverage.ops[OTHER_TYPE].tested == {"data:present": 1}

    def test_op_with_no_graphs_still_has_a_row(self, tmp_path: Path) -> None:
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16"])])
        coverage, _ = _coverage(tmp_path)
        assert coverage.ops[OTHER_TYPE].graphs == 0
        assert "| CustomOpAttributes | 0 | 0/2 (0%) |" in render_markdown(coverage)

    def test_error_in_one_op_leaves_the_other_counted(self, tmp_path: Path) -> None:
        _bundle(
            tmp_path,
            "a",
            [_node(features=["dtype:fp16"]), _node(OTHER_TYPE, boom="bad payload")],
        )
        coverage, errors = _coverage(tmp_path)
        assert errors == ["SdpaFwd/a: CustomOpAttributes: bad payload"]
        assert coverage.ops[FAKE_TYPE].graphs == 1
        assert coverage.ops[OTHER_TYPE].graphs == 0

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


@pytest.mark.usefixtures("fake_registry")
class TestDrillDown:
    """``--graphs``: the graphs behind one feature, and behind one cell."""

    @staticmethod
    def _tree(root: Path) -> None:
        _bundle(root, "unclaimed", [_node(features=["dtype:fp16"])])
        _bundle(
            root,
            "both",
            [_node(features=["dtype:fp16", "mask:causal"])],
            claims={"E1": {"gfx942": ["linux"]}, "E2": {"gfx950": ["windows"]}},
        )
        _bundle(
            root,
            "other",
            [_node(features=["dtype:bf16"])],
            claims={"E1": {"gfx942": ["linux"]}},
        )

    def _run(self, root: Path, capsys, *args: str) -> tuple[int, str, str]:
        code = main(["--bundles-dir", str(root), "--graphs", *args])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    def test_no_filter_lists_every_graph_exercising_the_feature(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._tree(tmp_path)
        code, out, _ = self._run(tmp_path, capsys, "Sdpa", "dtype:fp16")
        assert code == 0
        assert out == (
            "SdpaFwd/both [quick]: E1 gfx942/linux, E2 gfx950/windows\n"
            "SdpaFwd/unclaimed [quick]: not claimed\n"
            "2 graph(s)\n"
        )

    @pytest.mark.parametrize(
        ("filters", "expected"),
        [
            (["--arch", "gfx942"], ["SdpaFwd/both [quick]: E1 gfx942/linux"]),
            (["--engine", "E2"], ["SdpaFwd/both [quick]: E2 gfx950/windows"]),
            (["--arch", "gfx942", "--platform", "windows"], []),
            (["--engine", "E2", "--arch", "gfx942"], []),
        ],
    )
    def test_filters_keep_only_graphs_claimed_where_all_hold(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        filters: list[str],
        expected: list[str],
    ) -> None:
        self._tree(tmp_path)
        code, out, _ = self._run(tmp_path, capsys, "Sdpa", "dtype:fp16", *filters)
        assert code == 0
        assert out.splitlines() == [*expected, f"{len(expected)} graph(s)"]

    def test_list_agrees_with_the_per_feature_counts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._tree(tmp_path)
        op = _coverage(tmp_path)[0].ops[FAKE_TYPE]
        for feature in op.features:
            _, out, _ = self._run(tmp_path, capsys, "SdpaAttributes", feature)
            assert out.splitlines()[-1] == f"{op.tested[feature]} graph(s)"
            for arch in ("gfx942", "gfx950"):
                _, out, _ = self._run(tmp_path, capsys, "Sdpa", feature, "--arch", arch)
                assert out.splitlines()[-1] == f"{op.claimed[arch][feature]} graph(s)"

    def test_unknown_op_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = self._run(tmp_path, capsys, "Attention", "dtype:fp16")
        assert (code, out) == (2, "")
        assert "unknown op 'Attention'" in err

    def test_uncatalogued_feature_exits_two_and_lists_the_catalog(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = self._run(tmp_path, capsys, "Sdpa", "dtype:fp32")
        assert (code, out) == (2, "")
        assert "no feature 'dtype:fp32'" in err
        assert "'mask:sliding'" in err

    def test_filter_without_graphs_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--bundles-dir", str(tmp_path), "--arch", "gfx942"]) == 2
        assert "need --graphs" in capsys.readouterr().err

    def test_defect_exits_one_and_lists_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._tree(tmp_path)
        _bundle(tmp_path, "bad", [_node(boom="bad")])
        code, out, err = self._run(tmp_path, capsys, "Sdpa", "dtype:fp16")
        assert (code, out) == (1, "")
        assert "error: 1 defect(s)" in err


# --------------------------------------------------------------------------
# Feature matrix: feature_matrix.py on small trees, with the stand-in ops
# --------------------------------------------------------------------------


def _feature_claims(root: Path):
    errors: list[str] = []
    claims = compute_feature_claims(collect_units(root, errors), errors)
    assert errors == []
    return claims


def _row(document: str, section: str, op: str) -> list[str]:
    """Cells of the row for ``op`` in the table under heading ``section``."""
    body = document.split(f"\n{section}\n", 1)[1].split("\n#", 1)[0]
    (line,) = [ln for ln in body.splitlines() if ln.startswith(f"| **[{op}]")]
    return [cell.strip() for cell in line.strip("|").split("|")]


@pytest.mark.usefixtures("fake_registry")
class TestFeatureClaims:
    def test_any_engine_unions_engines_per_target(self, tmp_path: Path) -> None:
        _bundle(
            tmp_path,
            "a",
            [_node(features=["dtype:fp16"])],
            claims={"E1": {"gfx942": ["linux"]}},
        )
        _bundle(
            tmp_path,
            "b",
            [_node(features=["mask:causal"])],
            claims={"E2": {"gfx942": ["linux"]}},
        )
        claims = _feature_claims(tmp_path)
        target = ("gfx942", "linux")
        assert claims.on_target([FAKE_TYPE], target) == 2
        assert claims.engine_on_target(FAKE_TYPE, "E1", target) == 1
        assert claims.engine_on_target(FAKE_TYPE, "E2", target) == 1

    def test_platforms_and_archs_stay_apart(self, tmp_path: Path) -> None:
        _bundle(
            tmp_path,
            "a",
            [_node(features=["dtype:fp16"])],
            claims={"E": {"gfx942": ["windows"]}},
        )
        _bundle(
            tmp_path,
            "b",
            [_node(features=["dtype:bf16", "mask:causal"])],
            claims={"E": {"gfx942": ["linux"], "gfx1100": ["linux"]}},
        )
        claims = _feature_claims(tmp_path)
        assert claims.on_target([FAKE_TYPE], ("gfx942", "windows")) == 1
        assert claims.on_target([FAKE_TYPE], ("gfx942", "linux")) == 2
        assert claims.on_target([FAKE_TYPE], ("gfx1100", "linux")) == 2
        # GPU order, then Linux before Windows.
        assert claims.targets == [
            ("gfx942", "linux"),
            ("gfx942", "windows"),
            ("gfx1100", "linux"),
        ]

    def test_tested_includes_unclaimed_graphs(self, tmp_path: Path) -> None:
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16", "mask:sliding"])])
        _bundle(
            tmp_path,
            "b",
            [_node(features=["dtype:fp16"])],
            claims={"E": {"gfx942": ["linux"]}},
        )
        claims = _feature_claims(tmp_path)
        assert claims.tested[FAKE_TYPE] == {"dtype:fp16", "mask:sliding"}
        assert claims.on_target([FAKE_TYPE], ("gfx942", "linux")) == 1

    def test_fused_graph_counts_toward_each_op(self, tmp_path: Path) -> None:
        _bundle(
            tmp_path,
            "a",
            [
                _node(features=["dtype:fp16"]),
                _node(OTHER_TYPE, features=["data:empty"]),
            ],
            claims={"E": {"gfx90a": ["linux"]}},
        )
        claims = _feature_claims(tmp_path)
        target = ("gfx90a", "linux")
        assert claims.engine_on_target(FAKE_TYPE, "E", target) == 1
        assert claims.engine_on_target(OTHER_TYPE, "E", target) == 1
        # Feature ids are per op: the totals add, they never merge.
        assert claims.total([FAKE_TYPE, OTHER_TYPE]) == 6

    def test_group_counts_a_graph_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two ops of the Attention group in one graph, one in another.
        backward = "SdpaBackwardAttributes"
        monkeypatch.setitem(registry.OPS, backward, OTHER_OP)
        _bundle(
            tmp_path,
            "a",
            [_node(features=["dtype:fp16"]), _node(backward, features=["data:empty"])],
        )
        _bundle(tmp_path, "b", [_node(features=["dtype:bf16"])])
        claims = _feature_claims(tmp_path)
        assert claims.group_graphs["Attention"] == 2


class TestFeatureMatrixNames:
    def test_every_real_op_lands_in_a_named_group(self) -> None:
        groups = {group_of(t) for t in registry.OPS}
        assert groups == {
            "Attention",
            "Batchnorm",
            "Block scale",
            "Convolution",
            "CustomOp",
            "Linear attention",
            "Layernorm",
            "Matmul",
            "Pointwise",
            "Reduction",
            "Resample",
            "RMSNorm",
        }

    def test_rmsnorm_backward_is_rmsnorm_not_pointwise(self) -> None:
        assert group_of("RMSNormBackwardAttributes") == "RMSNorm"
        assert group_of("MoeGroupedMatmulBwdAttributes") == "Matmul"

    def test_every_gpu_is_in_a_dashboard_family(self) -> None:
        assert {gpu.family for gpu in GPUS.values()} == set(FAMILIES)

    def test_gfx1250_is_data_center_despite_its_gfx1_name(self) -> None:
        assert GPUS["gfx1250"].family == DATA_CENTER

    def test_node_type_label_drops_the_attributes_token(self) -> None:
        assert node_type_label("SdpaAttributes") == "Sdpa"
        assert (
            node_type_label("BatchnormInferenceAttributesVarianceExt")
            == "BatchnormInferenceVarianceExt"
        )

    @pytest.mark.parametrize("node_type", sorted(registry.OPS))
    def test_every_view_names_an_op_alike(self, node_type: str) -> None:
        """The name the matrix and feature matrix show is one --graphs takes."""
        label = node_type_label(node_type)
        assert node_label({"type": node_type}) == label
        assert resolve_node_type(label) == node_type

    @pytest.mark.parametrize(
        ("count", "total", "cell"),
        [
            (0, 4, "❌"),
            (4, 4, "✅"),
            (3, 4, "🟡 75%"),
            (2, 3, "🟡 66%"),
            (1, 300, "🟡 1%"),
            (299, 300, "🟡 99%"),
        ],
    )
    def test_percent_cell(self, count: int, total: int, cell: str) -> None:
        assert percent_cell(count, total) == cell


@pytest.mark.usefixtures("fake_registry")
class TestFeatureMatrixMarkdown:
    def _document(self, tmp_path: Path) -> str:
        _bundle(
            tmp_path,
            "a",
            [_node(features=["dtype:fp16", "mask:causal"])],
            claims={"E1": {"gfx942": ["linux"]}, "E2": {"gfx1100": ["windows"]}},
        )
        _bundle(
            tmp_path,
            "b",
            [_node(features=["dtype:bf16"])],
            claims={"E1": {"gfx1100": ["windows"]}},
        )
        return render_feature_matrix(_feature_claims(tmp_path))

    def test_gpu_targets_name_the_hardware(self, tmp_path: Path) -> None:
        document = self._document(tmp_path)
        section = document.split("\n## GPU targets\n", 1)[1]
        assert "| Target | Architecture | Products |\n" in section
        assert (
            '| <a id="gpu-gfx942-linux"></a>**gfx942-linux** | CDNA 3 | '
            "Instinct MI300 series |\n"
        ) in section

    def test_dashboard_cells_are_feature_shares(self, tmp_path: Path) -> None:
        document = self._document(tmp_path)
        rdna = _row(document, "### RDNA (consumer)", "Attention")
        # Op group (options), gfx1100-windows.
        assert rdna == ["**[Attention](#group-attention)** (4)", "🟡 75%"]
        cdna = _row(document, "### CDNA (data center)", "Attention")
        assert cdna[1] == "🟡 50%"
        # An op no graph exercises reads ❌, not a blank.
        assert _row(document, "### CDNA (data center)", "CustomOp") == [
            "**[CustomOp](#group-customop)** (2)",
            "❌",
        ]

    def test_op_groups_count_graphs_and_features(self, tmp_path: Path) -> None:
        document = self._document(tmp_path)
        section = document.split("\n## Op groups\n", 1)[1]
        assert '<a id="group-attention"></a>**Attention** | 2 | 4 |' in section
        assert '<a id="group-customop"></a>**CustomOp** | 0 | 2 |' in section

    def test_op_groups_name_an_op_no_graph_holds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The group's share is all Sdpa's; SdpaBackward is in no graph.
        monkeypatch.setitem(registry.OPS, "SdpaBackwardAttributes", OTHER_OP)
        document = self._document(tmp_path)
        assert _row(document, "### CDNA (data center)", "Attention")[1] == "🟡 33%"
        section = document.split("\n## Op groups\n", 1)[1]
        assert (
            '| <a id="group-attention"></a>**Attention** | 2 | 6 | '
            "[Sdpa](#op-sdpa) (4), "
            "[SdpaBackward](#op-sdpabackward) (2, **no test graph**) |"
        ) in section

    def test_dashboard_puts_most_claimed_target_first(self, tmp_path: Path) -> None:
        _bundle(
            tmp_path,
            "a",
            [_node(features=["dtype:fp16"])],
            claims={"E": {"gfx90a": ["linux"]}},
        )
        _bundle(
            tmp_path,
            "b",
            [_node(features=["dtype:bf16", "mask:causal"])],
            claims={"E": {"gfx950": ["linux"]}},
        )
        document = render_feature_matrix(_feature_claims(tmp_path))
        cdna = document.split("### CDNA (data center)", 1)[1]
        header = next(ln for ln in cdna.splitlines() if ln.startswith("| Op group"))
        assert header.index("gfx950-linux") < header.index("gfx90a-linux")
        # The engine tables keep GPU order.
        engine = document.split("\n### E\n", 1)[1]
        header = next(
            ln for ln in engine.splitlines() if ln.startswith("| Op (options) |")
        )
        assert header.index("gfx90a") < header.index("gfx950")

    def test_engine_tables_count_only_that_engine(self, tmp_path: Path) -> None:
        document = self._document(tmp_path)
        # Columns: Op (options), gfx942 L, gfx1100 W.
        assert _row(document, "### E1", "Sdpa") == [
            "**[Sdpa](#op-sdpa)** (4)",
            "🟡 50%",
            "🟡 25%",
        ]
        assert _row(document, "### E2", "Sdpa")[1:] == ["❌", "🟡 50%"]

    def test_engine_table_omits_ops_it_never_claims(self, tmp_path: Path) -> None:
        document = self._document(tmp_path)
        e1 = document.split("\n### E1\n", 1)[1].split("\n#", 1)[0]
        assert "CustomOp" not in e1

    def test_every_feature_is_listed_per_op(self, tmp_path: Path) -> None:
        document = self._document(tmp_path)
        section = document.split("\n## Options per op\n", 1)[1]
        assert (
            '| <a id="op-sdpa"></a>**Sdpa** (4) | '
            "`dtype:fp16`, `dtype:bf16`, `mask:causal` | `mask:sliding` |"
        ) in section
        assert (
            '| <a id="op-customop"></a>**CustomOp** (2) | — | '
            "`data:empty`, `data:present` |"
        ) in section

    def test_legend_defines_every_column(self, tmp_path: Path) -> None:
        document = self._document(tmp_path)
        summary = document.split("## Summary by architecture", 1)[1]
        summary = summary.split("\n### ", 1)[0]
        for column in ("Op group (options)", "GPU columns"):
            assert f"- **{column}**:" in summary
        # The shares count options singly; the legend must say so.
        assert "Combinations aren't measured" in summary
        engines = document.split("## By engine", 1)[1].split("\n### ", 1)[0]
        for column in ("Op (options)", "GPU columns"):
            assert f"- **{column}**:" in engines

    def test_anchors_resolve(self, tmp_path: Path) -> None:
        document = self._document(tmp_path)
        targets = set(re.findall(r"\]\(#([^)]+)\)", document))
        defined = set(re.findall(r'<a id="([^"]+)"></a>', document))
        defined |= {
            heading_anchor(line.lstrip("#").strip())
            for line in document.splitlines()
            if line.startswith("#")
        }
        assert {
            "gpu-targets",
            "e1",
            "group-attention",
            "op-sdpa",
            "options-per-op",
        } <= targets
        assert targets <= defined

    def test_unknown_arch_is_a_defect(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Two graphs claim it; it is reported once, naming the first.
        for name in ("a", "b"):
            _bundle(
                tmp_path,
                name,
                [_node(features=["dtype:fp16"])],
                claims={"E": {"gfx9999": ["linux"], "gfx942": ["linux"]}},
            )
        errors: list[str] = []
        compute_feature_claims(collect_units(tmp_path, errors), errors)
        assert len(errors) == 1
        assert "'gfx9999' is not in GPUS" in errors[0]
        output = tmp_path / "out.md"
        argv = ["--bundles-dir", str(tmp_path), "--output", str(output)]
        assert feature_matrix_main(argv) == 1
        assert not output.exists()
        assert "error: 1 defect(s); nothing rendered" in capsys.readouterr().err

    def test_gfx1250_lands_in_the_data_center_table(self, tmp_path: Path) -> None:
        _bundle(
            tmp_path,
            "a",
            [_node(features=["dtype:fp16"])],
            claims={"E": {"gfx1250": ["linux"]}},
        )
        document = render_feature_matrix(_feature_claims(tmp_path))
        rdna = document.split("### RDNA (consumer)", 1)[1].split("\n### ", 1)[0]
        cdna = document.split("### CDNA (data center)", 1)[1].split("\n## ", 1)[0]
        assert "_No support claimed yet._" in rdna
        assert "gfx1250-linux" in cdna


@pytest.mark.usefixtures("fake_registry")
class TestFeatureMatrixCli:
    def test_missing_directory_exits_two(self, tmp_path: Path) -> None:
        assert feature_matrix_main(["--bundles-dir", str(tmp_path / "absent")]) == 2

    def test_output_writes_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16"])])
        output = tmp_path / "out.md"
        argv = ["--bundles-dir", str(tmp_path), "--output", str(output)]
        assert feature_matrix_main(argv) == 0
        assert output.read_text(encoding="utf-8").startswith(
            "# Support matrix by API option"
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == f"wrote {output}\n"

    def test_dash_output_is_stdout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # As render_support_matrix.py reads it, not a file named "-".
        _bundle(tmp_path, "a", [_node(features=["dtype:fp16"])])
        monkeypatch.chdir(tmp_path)
        argv = ["--bundles-dir", str(tmp_path), "--output", "-"]
        assert feature_matrix_main(argv) == 0
        captured = capsys.readouterr()
        assert captured.out.startswith("# Support matrix by API option")
        assert captured.err == ""
        assert not (tmp_path / "-").exists()

    def test_defect_exits_one_and_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _bundle(tmp_path, "a", [_node(boom="bad")])
        output = tmp_path / "out.md"
        argv = ["--bundles-dir", str(tmp_path), "--output", str(output)]
        assert feature_matrix_main(argv) == 1
        assert not output.exists()
        assert "error: 1 defect(s); nothing rendered" in capsys.readouterr().err


class TestStdoutEncoding:
    """Both CLIs print emoji, so stdout must not depend on the console locale.

    A Windows console defaults to cp1252, which has no code point for them;
    writing through ``sys.stdout`` there raised UnicodeEncodeError. Forcing
    cp1252 reproduces that on any host.
    """

    @pytest.mark.parametrize("script", ["feature_matrix.py", "coverage.py"])
    def test_stdout_is_utf8_under_a_cp1252_console(self, script: str) -> None:
        env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
        env.pop("PYTHONUTF8", None)
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("support_features") / script),
                *(["--output", "-"] if script == "feature_matrix.py" else []),
            ],
            capture_output=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        assert result.stderr == b""
        document = result.stdout.decode("utf-8")
        assert document.startswith("# ")


# --------------------------------------------------------------------------
# SdpaAttributes
# --------------------------------------------------------------------------


def _bhsd_strides(dims: list[int]) -> list[int]:
    b, h, s, d = dims
    return [h * s * d, s * d, d, 1]


def _bshd_strides(dims: list[int]) -> list[int]:
    b, h, s, d = dims
    return [s * h * d, d, h * d, 1]


def _sdpa(
    *,
    q_dims=(2, 8, 128, 128),
    k_dims=(2, 8, 128, 128),
    v_dims=None,
    dtype="bfloat16",
    o_dtype=None,
    strides=_bhsd_strides,
    ragged=False,
    inputs=None,
    outputs=None,
    **attrs,
) -> tuple[dict, dict]:
    """An SDPA node and its tensors, with every port and attribute unset.

    v takes k's dims unless ``v_dims`` says otherwise, and o takes q's dims
    and q's dtype unless ``o_dtype`` says otherwise. ``ragged`` packs q by
    per-batch offsets.
    """
    tensors = {
        uid: {
            "uid": uid,
            "dims": list(dims),
            "strides": strides(list(dims)),
            "data_type": dtype,
        }
        for uid, dims in ((1, q_dims), (2, k_dims), (8, v_dims or k_dims), (3, q_dims))
    }
    tensors[3]["data_type"] = o_dtype or dtype
    if ragged:
        tensors[1]["ragged_offset_tensor_uid"] = 30
    node = {
        "type": "SdpaAttributes",
        "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 8}
        | (inputs or {}),
        "outputs": {"o_tensor_uid": 3} | (outputs or {}),
        "attributes": {
            "causal_mask": False,
            "causal_mask_bottom_right": False,
            "left_bound": None,
            "right_bound": None,
            "diagonal_alignment": "TOP_LEFT",
            "attn_scale_value": None,
        }
        | attrs,
    }
    return node, tensors


class TestSdpaExtract:
    def _extract(self, **kwargs) -> set[str]:
        return sdpa.extract(*_sdpa(**kwargs))

    def _group(self, group: str, **kwargs) -> str:
        (feature,) = {f for f in self._extract(**kwargs) if f.startswith(group + ":")}
        return feature

    def test_plain_graph(self) -> None:
        assert self._extract() == {
            "dtype:bf16",
            "head_dim:128",
            "heads:mha",
            "layout:bhsd",
            "mask:none",
            "seq:q_eq_kv",
            "scale:default",
        }

    @pytest.mark.parametrize(
        "attrs, expected",
        [
            ({"causal_mask": True}, "mask:causal_top_left"),
            ({"causal_mask_bottom_right": True}, "mask:causal_bottom_right"),
            ({"right_bound": 0}, "mask:causal_top_left"),
            (
                {"right_bound": 0, "diagonal_alignment": "BOTTOM_RIGHT"},
                "mask:causal_bottom_right",
            ),
            ({"left_bound": -1, "right_bound": 0}, "mask:causal_top_left"),
            ({"left_bound": 64, "right_bound": 0}, "mask:sliding_window"),
            ({"left_bound": 64}, "mask:sliding_window"),
            ({"left_bound": -1, "right_bound": -1}, "mask:none"),
            # The deprecated flag wins over bounds describing a window.
            ({"causal_mask": True, "left_bound": 64}, "mask:causal_top_left"),
        ],
    )
    def test_mask_follows_get_mask_type(self, attrs: dict, expected: str) -> None:
        assert self._group("mask", **attrs) == expected

    def test_both_deprecated_causal_flags_raise(self) -> None:
        with pytest.raises(FeatureError, match="both"):
            self._extract(causal_mask=True, causal_mask_bottom_right=True)

    @pytest.mark.parametrize(
        "k_heads, expected", [(8, "heads:mha"), (2, "heads:gqa"), (1, "heads:mqa")]
    )
    def test_heads(self, k_heads: int, expected: str) -> None:
        assert self._group("heads", k_dims=(2, k_heads, 128, 128)) == expected

    @pytest.mark.parametrize(
        "q_seq, expected",
        [(128, "seq:q_eq_kv"), (64, "seq:q_ne_kv"), (1, "seq:decode")],
    )
    def test_seq(self, q_seq: int, expected: str) -> None:
        assert self._group("seq", q_dims=(2, 8, q_seq, 128)) == expected

    @pytest.mark.parametrize("q_seq", [128, 1])
    def test_bshd_layout(self, q_seq: int) -> None:
        layout = self._group("layout", q_dims=(2, 8, q_seq, 128), strides=_bshd_strides)
        assert layout == "layout:bshd"

    def test_unknown_layout_raises(self) -> None:
        with pytest.raises(FeatureError, match="stride order"):
            self._extract(strides=lambda dims: [1, 2, 3, 4])

    @pytest.mark.parametrize(
        "strides", [_bshd_strides, _bhsd_strides, lambda dims: [1, 2, 3, 4]]
    )
    def test_ragged_layout_ignores_strides(self, strides) -> None:
        assert self._group("layout", ragged=True, strides=strides) == "layout:ragged"

    @pytest.mark.parametrize(
        "qk, v, expected",
        [
            (64, 64, "head_dim:64"),
            (192, 192, "head_dim:192"),
            # A narrower v is its own kernel, not a 192.
            (192, 128, "head_dim:192x128"),
        ],
    )
    def test_head_dim_reads_q_and_v(self, qk: int, v: int, expected: str) -> None:
        head_dim = self._group(
            "head_dim",
            q_dims=(2, 8, 128, qk),
            k_dims=(2, 8, 128, qk),
            v_dims=(2, 8, 128, v),
        )
        assert head_dim == expected

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"q_dims": (2, 8, 128, 96)}, "head dim 96 x 128"),
            # Each head dim is catalogued, the pair is not.
            ({"v_dims": (2, 8, 128, 64)}, "head dim 128 x 64"),
            ({"dtype": "float"}, "dtype 'float'"),
            # No engine runs it.
            ({"dtype": "fp8_e5m2"}, "dtype 'fp8_e5m2'"),
            ({"dtype": ["bfloat16"]}, r"dtype \['bfloat16'\]"),
            (
                {"q_dims": (8, 128, 128), "strides": lambda dims: [1] * len(dims)},
                "q / k / v rank 3 / 4 / 4",
            ),
            (
                {"v_dims": (2, 8, 128), "strides": lambda dims: [1] * len(dims)},
                "q / k / v rank 4 / 4 / 3",
            ),
        ],
    )
    def test_uncatalogued_value_raises(self, kwargs: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            self._extract(**kwargs)

    @pytest.mark.parametrize(
        "dtype, expected",
        [
            ("fp8_e4m3", "dtype:fp8e4m3"),
            # Not folded into OCP fp8: the two are not bit-compatible.
            ("fp8_e4m3_fnuz", "dtype:fp8e4m3_fnuz"),
        ],
    )
    def test_fp8_dtype(self, dtype: str, expected: str) -> None:
        assert self._group("dtype", dtype=dtype) == expected

    @pytest.mark.parametrize(
        "o_dtype, mixed", [("bfloat16", True), ("fp8_e4m3", False)]
    )
    def test_mixed_output(self, o_dtype: str, mixed: bool) -> None:
        features = self._extract(dtype="fp8_e4m3", o_dtype=o_dtype)
        assert ("precision:mixed_output" in features) == mixed

    @pytest.mark.parametrize("uid", [1, 3])
    def test_virtual_q_or_o_is_not_mixed(self, uid: int) -> None:
        node, tensors = _sdpa(dtype="fp8_e4m3", o_dtype="bfloat16")
        tensors[uid]["virtual"] = True
        assert "precision:mixed_output" not in sdpa.extract(node, tensors)

    @pytest.mark.parametrize(
        "uid, dtype, match",
        [
            (2, "half", "k / v dtype 'half' / 'bfloat16' is not q's"),
            (8, "half", "k / v dtype 'bfloat16' / 'half' is not q's"),
            (3, "float", "o dtype 'float'"),
            (3, ["bfloat16"], r"o dtype \['bfloat16'\]"),
        ],
    )
    def test_uncatalogued_tensor_dtype_raises(
        self, uid: int, dtype: object, match: str
    ) -> None:
        node, tensors = _sdpa()
        tensors[uid]["data_type"] = dtype
        with pytest.raises(FeatureError, match=match):
            sdpa.extract(node, tensors)

    def test_unbound_o_raises(self) -> None:
        node, tensors = _sdpa(outputs={"o_tensor_uid": None})
        with pytest.raises(FeatureError, match="o_tensor_uid is not bound"):
            sdpa.extract(node, tensors)

    def test_scale(self) -> None:
        assert self._group("scale", attn_scale_value=0.125) == "scale:fixed"
        assert (
            self._group("scale", attn_scale_value=0.125, inputs={"scale_tensor_uid": 9})
            == "scale:runtime"
        )

    def test_optional_ports(self) -> None:
        features = self._extract(
            inputs={
                "seq_len_q_tensor_uid": 10,
                "attn_mask_tensor_uid": 11,
                "seed_tensor_uid": 12,
                "page_table_k_tensor_uid": 13,
                "block_mask_tensor_uid": 14,
                "sink_token_tensor_uid": 15,
            },
            outputs={
                "stats_tensor_uid": 20,
                "max_tensor_uid": 21,
                "rng_dump_tensor_uid": 22,
                "amax_o_tensor_uid": 23,
            },
            alibi_mask=True,
        )
        assert {
            "varlen:seq_lens",
            "bias:additive",
            "bias:alibi",
            "dropout:on",
            "kv:paged",
            "input:block_mask",
            "input:sink_token",
            "out:stats",
            "out:max_sum_exp",
            "out:rng_dump",
            "fp8:amax",
        } <= features

    def test_null_ports_are_absent(self) -> None:
        features = self._extract(
            inputs={"attn_mask_tensor_uid": None}, outputs={"stats_tensor_uid": None}
        )
        assert "bias:additive" not in features
        assert "out:stats" not in features

    def test_every_feature_is_reachable(self) -> None:
        """No catalogued feature is one extract() can never return."""
        reached = set()
        for kwargs in (
            {},
            {"ragged": True},
            {"dtype": "half", "q_dims": (2, 8, 1, 64), "k_dims": (2, 1, 128, 64)},
            {
                "dtype": "fp8_e4m3",
                "o_dtype": "bfloat16",
                "q_dims": (2, 8, 64, 192),
                "k_dims": (2, 2, 128, 192),
            },
            {
                "dtype": "fp8_e4m3_fnuz",
                "strides": _bshd_strides,
                "causal_mask": True,
                "q_dims": (2, 8, 128, 192),
                "k_dims": (2, 8, 128, 192),
                "v_dims": (2, 8, 128, 128),
            },
            {"causal_mask_bottom_right": True, "attn_scale_value": 0.5},
            {"left_bound": 8, "right_bound": 8, "inputs": {"scale_tensor_uid": 9}},
            {
                "padding_mask": True,
                "alibi_mask": True,
                "dropout_probability": 0.1,
                "generate_stats": True,
                "inputs": {
                    "attn_mask_tensor_uid": 11,
                    "page_table_v_tensor_uid": 13,
                    "block_mask_tensor_uid": 14,
                    "sink_token_tensor_uid": 15,
                },
                "outputs": {
                    "sum_exp_tensor_uid": 21,
                    "rng_dump_tensor_uid": 22,
                    "amax_s_tensor_uid": 23,
                },
            },
        ):
            reached |= self._extract(**kwargs)
        assert reached == {feature.id for feature in sdpa.FEATURES}


# --------------------------------------------------------------------------
# SdpaBackwardAttributes
# --------------------------------------------------------------------------


def _sdpa_backward(**kwargs) -> tuple[dict, dict]:
    """A backward SDPA node: the forward builder's, rewired.

    Backward serializes its scalars under "parameters", not "attributes".
    """
    node, tensors = _sdpa(**kwargs)
    node["type"] = "SdpaBackwardAttributes"
    node["parameters"] = node.pop("attributes")
    node["inputs"] |= {"o_tensor_uid": 3, "do_tensor_uid": 3, "stats_tensor_uid": 4}
    del node["outputs"]["o_tensor_uid"]
    node["outputs"] |= {"dq_tensor_uid": 5, "dk_tensor_uid": 6, "dv_tensor_uid": 7}
    return node, tensors


class TestSdpaBackwardExtract:
    def _extract(self, **kwargs) -> set[str]:
        return sdpa_backward.extract(*_sdpa_backward(**kwargs))

    def _group(self, group: str, **kwargs) -> str:
        (feature,) = {f for f in self._extract(**kwargs) if f.startswith(group + ":")}
        return feature

    def test_plain_graph(self) -> None:
        assert self._extract() == {
            "dtype:bf16",
            "head_dim:128",
            "heads:mha",
            "layout:bhsd",
            "mask:none",
            "seq:q_eq_kv",
            "scale:default",
        }

    def test_scalars_are_read_from_parameters(self) -> None:
        node, tensors = _sdpa_backward(causal_mask=True, attn_scale_value=0.5)
        assert {"mask:causal_top_left", "scale:fixed"} <= sdpa_backward.extract(
            node, tensors
        )
        # The forward key carries nothing for backward.
        node["attributes"] = node.pop("parameters")
        assert {"mask:none", "scale:default"} <= sdpa_backward.extract(node, tensors)

    @pytest.mark.parametrize(
        "attrs, expected",
        [
            ({"right_bound": 0}, "mask:causal_top_left"),
            (
                {"right_bound": 0, "diagonal_alignment": "BOTTOM_RIGHT"},
                "mask:causal_bottom_right",
            ),
            ({"left_bound": 64, "right_bound": 64}, "mask:sliding_window"),
        ],
    )
    def test_mask(self, attrs: dict, expected: str) -> None:
        assert self._group("mask", **attrs) == expected

    @pytest.mark.parametrize(
        "q_seq, expected", [(1, "seq:q_ne_kv"), (128, "seq:q_eq_kv")]
    )
    def test_one_query_position_is_not_decode(self, q_seq: int, expected: str) -> None:
        assert self._group("seq", q_dims=(2, 8, q_seq, 128)) == expected

    def test_fp8_raises(self) -> None:
        with pytest.raises(FeatureError, match="dtype 'fp8_e4m3'"):
            self._extract(dtype="fp8_e4m3")

    def test_kv_dtype_not_q_raises(self) -> None:
        node, tensors = _sdpa_backward()
        tensors[8]["data_type"] = "half"
        with pytest.raises(FeatureError, match="is not q's"):
            sdpa_backward.extract(node, tensors)

    def test_ragged_raises(self) -> None:
        # Not counted as its bshd stride order.
        with pytest.raises(FeatureError, match="ragged q"):
            self._extract(ragged=True, strides=_bshd_strides)

    def test_optional_ports(self) -> None:
        features = self._extract(
            inputs={"attn_mask_tensor_uid": 11, "dropout_mask_tensor_uid": 12},
            outputs={"dbias_tensor_uid": 20},
        )
        assert {"bias:additive", "dropout:on", "out:dbias"} <= features

    def test_null_ports_are_absent(self) -> None:
        features = self._extract(outputs={"dbias_tensor_uid": None})
        assert "out:dbias" not in features

    def test_every_feature_is_reachable(self) -> None:
        """No catalogued feature is one extract() can never return."""
        reached = set()
        for kwargs in (
            {},
            {"dtype": "half", "q_dims": (2, 8, 64, 64), "k_dims": (2, 1, 128, 64)},
            {"q_dims": (2, 8, 128, 192), "k_dims": (2, 2, 128, 192)},
            {
                "strides": _bshd_strides,
                "causal_mask": True,
                "q_dims": (2, 8, 128, 192),
                "k_dims": (2, 8, 128, 192),
                "v_dims": (2, 8, 128, 128),
            },
            {"causal_mask_bottom_right": True, "attn_scale_value": 0.5},
            {"left_bound": 8, "right_bound": 8, "inputs": {"scale_tensor_uid": 9}},
            {
                "padding_mask": True,
                "alibi_mask": True,
                "dropout_probability": 0.1,
                "inputs": {"attn_mask_tensor_uid": 11},
                "outputs": {"dbias_tensor_uid": 20},
            },
        ):
            reached |= self._extract(**kwargs)
        assert reached == {feature.id for feature in sdpa_backward.FEATURES}


# --------------------------------------------------------------------------
# PointwiseAttributes
# --------------------------------------------------------------------------

# Every key the serializer writes under "inputs" besides the ports
# (utilities/json/PointwiseAttributes.hpp), null unless a test sets it.
_POINTWISE_SCALARS = (
    "relu_lower_clip",
    "relu_upper_clip",
    "relu_lower_clip_slope",
    "axis_tensor_uid",
    "swish_beta",
    "elu_alpha",
    "softplus_beta",
)


def _pointwise(
    mode: str = "relu_fwd",
    *,
    n_inputs: int | None = None,
    dtype: str = "float",
    in_dtypes: tuple[str, ...] | None = None,
    out_dtype: str | None = None,
    dims: tuple[int, ...] = (2, 3, 4),
    in_dims: tuple[tuple[int, ...], ...] | None = None,
    **params,
) -> tuple[dict, dict]:
    """A pointwise node: in_0.. on uids 1.., out_0 on uid 9.

    ``n_inputs`` defaults to the mode's arity; a test passes it to bind the
    wrong number.
    """
    n = pointwise.ARITY.get(mode, 1) if n_inputs is None else n_inputs
    in_dtypes = in_dtypes or (dtype,) * n
    in_dims = in_dims or (dims,) * n
    tensors = {
        uid: {"uid": uid, "dims": list(d), "data_type": t}
        for uid, d, t in zip(range(1, n + 1), in_dims, in_dtypes)
    }
    tensors[9] = {"uid": 9, "dims": list(dims), "data_type": out_dtype or dtype}
    ports = ("in_0_tensor_uid", "in_1_tensor_uid", "in_2_tensor_uid")
    inputs = {"operation": mode}
    inputs |= {name: None for name in _POINTWISE_SCALARS}
    inputs |= {port: (i + 1 if i < n else None) for i, port in enumerate(ports)}
    node = {
        "type": "PointwiseAttributes",
        "inputs": inputs | params,
        "outputs": {"out_0_tensor_uid": 9},
    }
    return node, tensors


def _schema_enum(file: str, name: str) -> set[str]:
    match = re.search(
        rf"\benum\s+{name}\s*:\s*\w+\s*\{{(.*?)\}}", _schema_text(file), re.S
    )
    assert match, f"enum {name} not found"
    return set(re.findall(r"^\s*(\w+)\s*=", match.group(1), re.M))


class TestPointwiseExtract:
    def _extract(self, mode: str = "relu_fwd", **kwargs) -> set[str]:
        return pointwise.extract(*_pointwise(mode, **kwargs))

    def test_plain_graph(self) -> None:
        assert self._extract() == {"mode:relu_fwd", "dtype:fp32"}

    def test_every_schema_mode_has_an_arity(self) -> None:
        modes = _schema_enum("pointwise_attributes.fbs", "PointwiseMode")
        assert set(pointwise.ARITY) == {m.lower() for m in modes} - {"unset"}

    def test_arity_lists_are_disjoint(self) -> None:
        unary, binary = set(pointwise.UNARY), set(pointwise.BINARY)
        assert not unary & binary
        assert not (unary | binary) & set(pointwise.TERNARY)

    @pytest.mark.parametrize("mode", sorted(pointwise.ARITY))
    def test_every_mode_extracts_at_its_arity(self, mode: str) -> None:
        assert f"mode:{mode}" in self._extract(mode)

    @pytest.mark.parametrize(
        "mode, n_inputs",
        [("relu_fwd", 2), ("add", 1), ("add", 3), ("binary_select", 2)],
    )
    def test_wrong_arity_raises(self, mode: str, n_inputs: int) -> None:
        with pytest.raises(FeatureError, match=f"{mode} takes"):
            self._extract(mode, n_inputs=n_inputs)

    def test_input_bound_to_the_wrong_port_raises(self) -> None:
        node, tensors = _pointwise("add")
        inputs = node["inputs"]
        inputs["in_2_tensor_uid"] = inputs.pop("in_1_tensor_uid")
        with pytest.raises(FeatureError, match="add takes 2"):
            pointwise.extract(node, tensors)

    @pytest.mark.parametrize("mode", ["unset", "relu", None])
    def test_uncatalogued_mode_raises(self, mode) -> None:
        with pytest.raises(FeatureError, match="pointwise mode"):
            self._extract(mode)

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        assert self._extract(dtype=dtype) == {"mode:relu_fwd", expected}

    @pytest.mark.parametrize(
        "mode, dtype", [("relu_fwd", "int32"), ("logical_not", "boolean")]
    )
    def test_uncatalogued_data_dtype_raises(self, mode: str, dtype: str) -> None:
        with pytest.raises(FeatureError, match="in_0 dtype"):
            self._extract(mode, dtype=dtype)

    def test_uncatalogued_port_dtype_raises(self) -> None:
        with pytest.raises(FeatureError, match="int32"):
            self._extract("add", in_dtypes=("float", "int32"))

    @pytest.mark.parametrize(
        "mode, kwargs",
        [
            ("relu_fwd", {"out_dtype": "bfloat16"}),
            ("add", {"in_dtypes": ("float", "half")}),
            ("relu_bwd", {"in_dtypes": ("half", "half")}),
        ],
    )
    def test_mixed_precision(self, mode: str, kwargs: dict) -> None:
        assert "precision:mixed" in self._extract(mode, **kwargs)

    def test_bool_output_is_not_mixed_precision(self) -> None:
        features = self._extract("cmp_lt", out_dtype="boolean")
        assert features == {"mode:cmp_lt", "dtype:fp32"}

    def test_broadcast(self) -> None:
        features = self._extract("add", in_dims=((2, 3, 4), (1, 3, 1)))
        assert "broadcast:in" in features
        assert "broadcast:in" not in self._extract("add")

    @pytest.mark.parametrize(
        "mode, param, value, expected",
        [
            ("relu_fwd", "relu_lower_clip", 0.0, set()),
            ("relu_fwd", "relu_lower_clip", 0.1, {"relu:lower_clip"}),
            ("relu_bwd", "relu_lower_clip", 0.1, {"relu:lower_clip"}),
            ("relu_fwd", "relu_upper_clip", 6.0, {"relu:upper_clip"}),
            ("relu_fwd", "relu_lower_clip_slope", 0.0, set()),
            ("relu_fwd", "relu_lower_clip_slope", 0.01, {"relu:leaky"}),
            ("swish_fwd", "swish_beta", 1.0, set()),
            ("swish_bwd", "swish_beta", 1.5, {"swish:beta"}),
            ("elu_fwd", "elu_alpha", 1.0, set()),
            ("elu_bwd", "elu_alpha", 0.5, {"elu:alpha"}),
            ("softplus_fwd", "softplus_beta", 1.0, set()),
            ("softplus_bwd", "softplus_beta", 2.0, {"softplus:beta"}),
            ("gen_index", "axis_tensor_uid", 1, set()),
        ],
    )
    def test_parameter_counts_only_off_its_default(
        self, mode: str, param: str, value, expected: set[str]
    ) -> None:
        dtype = "half" if mode.endswith("_bwd") else "float"
        features = self._extract(mode, dtype=dtype, **{param: value})
        assert features - {f"mode:{mode}", "dtype:fp16", "dtype:fp32"} == expected

    @pytest.mark.parametrize(
        "mode, param, value",
        [
            ("relu_fwd", "swish_beta", 1.0),
            ("add", "relu_upper_clip", 6.0),
            ("relu_fwd", "axis_tensor_uid", 0),
            ("swish_fwd", "elu_alpha", 0.5),
        ],
    )
    def test_parameter_on_a_mode_that_ignores_it_raises(
        self, mode: str, param: str, value
    ) -> None:
        with pytest.raises(FeatureError, match=f"{param} is set, but {mode}"):
            self._extract(mode, **{param: value})

    def test_scalars_are_read_from_inputs(self) -> None:
        node, tensors = _pointwise(relu_upper_clip=6.0)
        assert "relu:upper_clip" in pointwise.extract(node, tensors)
        node["parameters"] = {"relu_upper_clip": node["inputs"].pop("relu_upper_clip")}
        assert "relu:upper_clip" not in pointwise.extract(node, tensors)

    def test_every_feature_is_reachable(self) -> None:
        """No catalogued feature is one extract() can never return."""
        reached = set()
        for mode in pointwise.ARITY:
            reached |= self._extract(mode)
        for kwargs in (
            {"dtype": "half", "relu_lower_clip": 0.1, "relu_upper_clip": 6.0},
            {"dtype": "bfloat16", "relu_lower_clip_slope": 0.01},
            {"out_dtype": "half"},
        ):
            reached |= self._extract(**kwargs)
        reached |= self._extract("add", in_dims=((2, 3, 4), (1, 3, 1)))
        reached |= self._extract("swish_fwd", swish_beta=1.5)
        reached |= self._extract("elu_fwd", elu_alpha=0.5)
        reached |= self._extract("softplus_fwd", softplus_beta=2.0)
        assert reached == {feature.id for feature in pointwise.FEATURES}


# --------------------------------------------------------------------------
# Convolution (shared core) and the convolution ops
# --------------------------------------------------------------------------


def _layout_strides(dims: list[int], layout: str) -> list[int]:
    """Packed strides for ``dims`` in channels-first or channels-last order."""
    rank = len(dims)
    order = list(range(rank)) if layout == "nchw" else [0, *range(2, rank), 1]
    strides = [0] * rank
    step = 1
    for dim in reversed(order):
        strides[dim] = step
        step *= dims[dim]
    return strides


# Op module -> its ports, and the shape role its serializer writes under
# "outputs" (the other two go under "inputs";
# utilities/json/Convolution*Attributes.hpp). Written out here rather than read
# from module.PORTS, so _conv builds what the API binds, not what the module
# believes: a module with two ports swapped then fails the binding tests.
_CONV_BINDING = {
    convolution_fwd: (
        convolution.Ports(
            "x_tensor_uid", "w_tensor_uid", "y_tensor_uid", "x_tensor_uid"
        ),
        "response",
    ),
    convolution_bwd: (
        convolution.Ports(
            "dx_tensor_uid", "w_tensor_uid", "dy_tensor_uid", "dy_tensor_uid"
        ),
        "activation",
    ),
    convolution_wrw: (
        convolution.Ports(
            "x_tensor_uid", "dw_tensor_uid", "dy_tensor_uid", "x_tensor_uid"
        ),
        "filter",
    ),
}
CONV_MODULES = tuple(_CONV_BINDING)


def _conv(
    module=convolution_fwd,
    *,
    spatial: tuple[int, ...] = (8, 8),
    channels: int = 4,
    filters: int = 8,
    groups: int = 1,
    layout: str = "nchw",
    layouts: dict[str, str] | None = None,
    dtype: str = "float",
    dtypes: dict[str, str] | None = None,
    virtual: tuple[str, ...] = (),
    conv_mode: str | None = "CROSS_CORRELATION",
    kernel: int = 3,
    **params,
) -> tuple[dict, dict]:
    """A conv node: activation on uid 1, filter on uid 2, response on uid 3.

    ``layouts`` / ``dtypes`` override ``layout`` / ``dtype`` per role;
    ``virtual`` lists the roles whose tensor is virtual. Parameters default to
    no padding, unit stride and unit dilation; the response's spatial dims
    are those the API computes from them.
    """
    zeros, ones = [0] * len(spatial), [1] * len(spatial)
    parameters = {
        "conv_mode": conv_mode,
        "pre_padding": zeros,
        "post_padding": zeros,
        "stride": ones,
        "dilation": ones,
    } | params
    try:
        response = [
            (size + pre + post - (dil * (kernel - 1) + 1)) // step + 1
            for size, pre, post, step, dil in zip(
                spatial,
                *(parameters[name] for name in convolution.PARAMETERS),
                strict=True,
            )
        ]
    except (TypeError, ValueError, ZeroDivisionError):
        # Parameters a test makes invalid: they raise before the response.
        response = list(spatial)
    dims = {
        "activation": [2, channels, *spatial],
        "filter": [filters, channels // groups, *(kernel for _ in spatial)],
        "response": [2, filters, *response],
    }
    layouts = {role: layout for role in dims} | (layouts or {})
    dtypes = {role: dtype for role in dims} | (dtypes or {})
    tensors = {
        uid: {
            "uid": uid,
            "dims": dims[role],
            "strides": _layout_strides(dims[role], layouts[role]),
            "data_type": dtypes[role],
            "virtual": role in virtual,
        }
        for uid, role in enumerate(dims, start=1)
    }
    ports, output = _CONV_BINDING[module]
    inputs = {getattr(ports, r): uid for uid, r in enumerate(dims, 1) if r != output}
    node = {
        "type": module.NODE_TYPE,
        "inputs": inputs,
        "outputs": {getattr(ports, output): list(dims).index(output) + 1},
        "parameters": parameters,
    }
    return node, tensors


class TestConvolutionLayouts:
    @pytest.mark.parametrize(
        "dims, strides, expected",
        [
            ([2, 4, 8, 8], [256, 64, 8, 1], {"nchw"}),
            ([2, 4, 8, 8], [256, 1, 32, 4], {"nhwc"}),
            ([2, 4, 8], [32, 8, 1], {"nchw"}),
            ([2, 4, 8], [32, 1, 4], {"nhwc"}),
            ([2, 4, 2, 8, 8], [512, 128, 64, 8, 1], {"nchw"}),
            ([2, 4, 2, 8, 8], [512, 1, 256, 32, 4], {"nhwc"}),
            # C = 1: the strides cannot place it, so both layouts fit.
            ([2, 1, 8, 8], [64, 64, 8, 1], {"nchw", "nhwc"}),
            ([2, 1, 8, 8], [64, 1, 8, 1], {"nchw", "nhwc"}),
            # W = 1 in 1D: N, C remain, in the same order in both layouts.
            ([2, 4, 1], [4, 1, 4], {"nchw", "nhwc"}),
            # Padded (non-packed) strides still read by their order.
            ([2, 4, 8, 8], [512, 100, 10, 1], {"nchw"}),
        ],
    )
    def test_tensor_layouts(self, dims, strides, expected) -> None:
        tensor = {"dims": dims, "strides": strides}
        assert convolution.tensor_layouts(tensor) == expected

    @pytest.mark.parametrize(
        "strides, match",
        [
            (None, "do not match"),
            ([], "do not match"),
            ([64, 8, 1], "do not match"),
            ([256, 1, 1, 8], "overlap"),
            # C outermost, N innermost: neither layout.
            ([1, 128, 16, 2], "not a catalogued layout"),
            # W before H.
            ([256, 64, 1, 8], "not a catalogued layout"),
        ],
    )
    def test_bad_strides_raise(self, strides, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            convolution.tensor_layouts({"dims": [2, 4, 8, 8], "strides": strides})


@pytest.mark.parametrize("module", CONV_MODULES, ids=lambda m: m.NODE_TYPE)
class TestConvolutionOpBinding:
    """Each conv op binds the shared core to its own three tensors.

    The core's behavior is tested once, on ConvolutionFwdAttributes, below;
    this checks what differs per op: which schema field plays which role.
    """

    PLAIN = {"dtype:fp32", "spatial:2d", "layout:nchw", "mode:cross_correlation"}

    def test_plain_graph(self, module) -> None:
        assert module.extract(*_conv(module)) == self.PLAIN

    def test_ports_are_the_schema_tensor_fields(self, module) -> None:
        fields = table_fields(module.SCHEMA_FILE, module.SCHEMA_TABLE)
        ports = module.PORTS
        assert {ports.activation, ports.filter, ports.response} == {
            f for f in fields if f.endswith("_tensor_uid")
        }

    def test_ports_bind_the_roles_the_api_does(self, module) -> None:
        expected, output = _CONV_BINDING[module]
        assert module.PORTS == expected
        # The op's dtype is that of an input, not of what it computes.
        assert expected.data != getattr(expected, output)

    def test_parameters_are_the_schema_fields(self, module) -> None:
        fields = table_fields(module.SCHEMA_FILE, module.SCHEMA_TABLE)
        assert {f for f in fields if not f.endswith("_tensor_uid")} == {
            *convolution.PARAMETERS,
            "conv_mode",
        }

    def test_dtype_is_the_data_ports(self, module) -> None:
        ports, _ = _CONV_BINDING[module]
        data_role = next(
            r for r in ports._fields[:3] if getattr(ports, r) == ports.data
        )
        others = {r: "half" for r in ports._fields[:3] if r != data_role}
        assert "dtype:fp32" in module.extract(*_conv(module, dtypes=others))
        assert "dtype:fp16" in module.extract(
            *_conv(module, dtypes={data_role: "half"})
        )

    def test_groups_read_the_activation_and_filter(self, module) -> None:
        # 4 activation channels, 2 per filter: 2 groups. Read off the response
        # (2 channels) instead, it would be 1 group and no feature.
        features = module.extract(*_conv(module, channels=4, filters=2, groups=2))
        assert "groups:grouped" in features

    def test_every_feature_is_reachable(self, module) -> None:
        """No catalogued feature is one extract() can never return."""
        reached = module.extract(*_conv(module))
        for kwargs in (
            {"spatial": (8,), "layout": "nhwc", "dtype": "half"},
            {"spatial": (4, 8, 8), "dtype": "bfloat16", "groups": 2},
            {"groups": 4, "dtypes": {"filter": "half"}},
            {"stride": [2, 2], "dilation": [2, 2]},
            {"pre_padding": [1, 1], "post_padding": [1, 1]},
            {"pre_padding": [1, 1], "post_padding": [0, 0]},
            {"conv_mode": "CONVOLUTION"},
        ):
            reached |= module.extract(*_conv(module, **kwargs))
        assert reached == {feature.id for feature in module.FEATURES}


class TestConvolutionFwdExtract:
    PLAIN = {"dtype:fp32", "spatial:2d", "layout:nchw", "mode:cross_correlation"}

    def _extract(self, **kwargs) -> set[str]:
        return convolution_fwd.extract(*_conv(convolution_fwd, **kwargs))

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    def test_every_schema_mode_is_catalogued(self) -> None:
        modes = _schema_enum("convolution_common.fbs", "ConvMode")
        assert set(convolution.MODES) == modes - {"UNSET"}

    @pytest.mark.parametrize(
        "spatial, expected",
        [((8,), "spatial:1d"), ((8, 8), "spatial:2d"), ((4, 8, 8), "spatial:3d")],
    )
    @pytest.mark.parametrize("layout", convolution.LAYOUTS)
    def test_spatial_and_layout(self, spatial, expected: str, layout: str) -> None:
        features = self._extract(spatial=spatial, layout=layout)
        assert features == {
            "dtype:fp32",
            expected,
            f"layout:{layout}",
            "mode:cross_correlation",
        }

    def test_a_tensor_that_cannot_tell_defers_to_the_others(self) -> None:
        # C = 1 leaves x and w ambiguous; y (K = 8) is channels last.
        assert "layout:nhwc" in self._extract(channels=1, layout="nhwc")
        features = self._extract(channels=1, layouts={"activation": "nhwc"})
        assert "layout:nchw" in features

    def test_no_tensor_that_tells_adds_no_layout(self) -> None:
        features = self._extract(channels=1, filters=1, spatial=(1, 1), kernel=1)
        assert not {f for f in features if f.startswith("layout:")}

    def test_disagreeing_layouts_raise(self) -> None:
        with pytest.raises(FeatureError, match="layouts disagree"):
            self._extract(layouts={"filter": "nhwc"})

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        assert self._extract(dtype=dtype) == self.PLAIN - {"dtype:fp32"} | {expected}

    def test_dtype_is_the_data_ports(self) -> None:
        features = self._extract(dtype="float", dtypes={"activation": "half"})
        assert {"dtype:fp16", "precision:mixed"} <= features
        assert "dtype:fp32" not in features

    def test_mixed_precision_counts_only_non_virtual_tensors(self) -> None:
        mixed = {"response": "float"}
        assert "precision:mixed" in self._extract(dtype="half", dtypes=mixed)
        features = self._extract(dtype="half", dtypes=mixed, virtual=("response",))
        assert "precision:mixed" not in features

    @pytest.mark.parametrize(
        "dtypes, virtual, match",
        [
            ({"activation": "int8"}, (), "x_tensor_uid dtype"),
            ({"response": "int32"}, (), r"\['int32'\]"),
            # A virtual tensor does not count toward precision, but its dtype
            # is still checked.
            ({"response": "int32"}, ("response",), r"\['int32'\]"),
        ],
    )
    def test_uncatalogued_dtype_raises(self, dtypes, virtual, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            self._extract(dtypes=dtypes, virtual=virtual)

    @pytest.mark.parametrize(
        "channels, filters, groups, expected",
        [
            (4, 8, 1, set()),
            (4, 8, 2, {"groups:grouped"}),
            (4, 8, 4, {"groups:depthwise"}),
            (4, 4, 4, {"groups:depthwise"}),
        ],
    )
    def test_groups(self, channels, filters, groups, expected) -> None:
        features = self._extract(channels=channels, filters=filters, groups=groups)
        assert features - self.PLAIN == expected

    @pytest.mark.parametrize(
        "w_channels, filters, match",
        # 4 activation channels; the last: 4 groups of 1, 6 filters.
        [(3, 8, "groups of 3"), (0, 8, "groups of 0"), (1, 6, "6 filters")],
    )
    def test_channels_that_do_not_split_raise(
        self, w_channels: int, filters: int, match: str
    ) -> None:
        node, tensors = _conv(channels=4, filters=filters)
        tensors[2]["dims"][1] = w_channels
        with pytest.raises(FeatureError, match=match):
            convolution_fwd.extract(node, tensors)

    @pytest.mark.parametrize(
        "params, expected",
        [
            ({"stride": [1, 2]}, {"stride:strided"}),
            ({"dilation": [2, 1]}, {"dilation:dilated"}),
            ({"pre_padding": [1, 1], "post_padding": [1, 1]}, {"padding:symmetric"}),
            ({"pre_padding": [1, 0], "post_padding": [1, 0]}, {"padding:symmetric"}),
            ({"pre_padding": [1, 1], "post_padding": [0, 1]}, {"padding:asymmetric"}),
            ({"pre_padding": [0, 0], "post_padding": [0, 1]}, {"padding:asymmetric"}),
        ],
    )
    def test_parameters(self, params: dict, expected: set[str]) -> None:
        assert self._extract(**params) - self.PLAIN == expected

    def test_mode(self) -> None:
        features = self._extract(conv_mode="CONVOLUTION")
        assert features == self.PLAIN - {"mode:cross_correlation"} | {
            "mode:convolution"
        }

    @pytest.mark.parametrize("conv_mode", ["UNSET", None, "cross_correlation"])
    def test_uncatalogued_mode_raises(self, conv_mode) -> None:
        with pytest.raises(FeatureError, match="conv_mode"):
            self._extract(conv_mode=conv_mode)

    @pytest.mark.parametrize(
        "params, match",
        [
            ({"stride": [1]}, "stride .* does not have 2 entries"),
            ({"dilation": None}, "dilation None does not have 2 entries"),
            ({"pre_padding": [0, 0, 0]}, "pre_padding .* does not have 2"),
            ({"pre_padding": [-1, 0]}, "negative padding"),
            ({"post_padding": [0, -1]}, "negative padding"),
            ({"stride": [0, 1]}, "below 1"),
            ({"dilation": [1, 0]}, "below 1"),
        ],
    )
    def test_invalid_parameters_raise(self, params: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            self._extract(**params)

    def test_parameters_are_read_from_parameters(self) -> None:
        node, tensors = _conv(stride=[2, 2])
        node["inputs"]["stride"] = node["parameters"].pop("stride")
        with pytest.raises(FeatureError, match="stride None"):
            convolution_fwd.extract(node, tensors)

    def test_tensor_ranks_that_differ_raise(self) -> None:
        node, tensors = _conv()
        tensors[2]["dims"].append(3)
        tensors[2]["strides"].append(1)
        with pytest.raises(FeatureError, match="ranks differ"):
            convolution_fwd.extract(node, tensors)

    @pytest.mark.parametrize(
        "spatial, kernel, params, response",
        [
            # (8 - 3) // 1 + 1.
            ((8, 8), 3, {}, [6, 6]),
            # (8 + 1 + 1 - 3) // 2 + 1: the last window is cut off.
            (
                (8, 8),
                3,
                {"pre_padding": [1, 1], "post_padding": [1, 1], "stride": [2, 2]},
                [4, 4],
            ),
            # A 3-tap filter dilated by 2 spans 5: (8 - 5) // 1 + 1.
            ((8, 8), 3, {"dilation": [2, 1]}, [4, 6]),
            # A filter the size of the activation: one output.
            ((3, 3), 3, {}, [1, 1]),
            ((5,), 1, {"stride": [2]}, [3]),
        ],
    )
    def test_response_dims_follow_the_api(
        self, spatial, kernel: int, params: dict, response: list[int]
    ) -> None:
        node, tensors = _conv(spatial=spatial, kernel=kernel, **params)
        assert tensors[3]["dims"][2:] == response
        assert convolution_fwd.extract(node, tensors)

    @pytest.mark.parametrize(
        "role, index, value, match",
        [
            ("response", 0, 3, "not batch 2 by 8 filters"),
            ("response", 1, 4, "not batch 2 by 8 filters"),
            ("response", 2, 7, "in spatial dim 0"),
            ("response", 3, 5, "in spatial dim 1"),
            # The activation's batch, not the response's, is the one that
            # disagrees: the check is symmetric.
            ("activation", 0, 3, "not batch 3 by 8 filters"),
        ],
    )
    def test_response_dims_that_do_not_follow_raise(
        self, role: str, index: int, value: int, match: str
    ) -> None:
        node, tensors = _conv()
        uid = {"activation": 1, "filter": 2, "response": 3}[role]
        tensors[uid]["dims"][index] = value
        with pytest.raises(FeatureError, match=match):
            convolution_fwd.extract(node, tensors)

    def test_a_filter_wider_than_the_padded_activation_raises(self) -> None:
        # A 5-tap filter over 3 + 1 padding: no window fits.
        node, tensors = _conv(spatial=(3, 3), kernel=5, pre_padding=[1, 1])
        with pytest.raises(FeatureError, match="in spatial dim 0"):
            convolution_fwd.extract(node, tensors)

    @pytest.mark.parametrize("spatial, n", [((), 0), ((2, 2, 2, 2), 4)])
    def test_uncatalogued_rank_raises(self, spatial, n: int) -> None:
        with pytest.raises(FeatureError, match=f"{n} spatial dims"):
            self._extract(spatial=spatial)


# --------------------------------------------------------------------------
# Scalar operands
# --------------------------------------------------------------------------


def _scalar(uid: int, source: str = "fixed", dims: list[int] | None = None) -> dict:
    """A scalar as the serializer writes it: baked, runtime, or in a buffer."""
    dims = [1] if dims is None else dims
    tensor = {"uid": uid, "data_type": "double", "dims": dims, "virtual": False}
    tensor["strides"] = [1] * len(dims)
    if source == "fixed":
        tensor |= {"value": 1e-5, "value_type": "Float64Value"}
    elif source == "runtime":
        tensor["is_runtime_pass_by_value"] = True
    return tensor


class TestScalarSource:
    @pytest.mark.parametrize(
        "tensor, expected",
        [
            (_scalar(1, "fixed"), "fixed"),
            (_scalar(1, "runtime"), "runtime"),
            # A runtime scalar may carry a default; the caller still passes it.
            (_scalar(1, "fixed") | {"is_runtime_pass_by_value": True}, "runtime"),
            (_scalar(1, "fixed") | {"is_runtime_pass_by_value": False}, "fixed"),
            (_scalar(1, "fixed", dims=[1, 1, 1, 1]), "fixed"),
        ],
    )
    def test_source(self, tensor: dict, expected: str) -> None:
        assert scalar_source(tensor, "epsilon_tensor_uid") == expected

    @pytest.mark.parametrize(
        "tensor, match",
        [
            (_scalar(1, "buffer"), "neither a baked constant nor runtime"),
            (_scalar(1, "fixed", dims=[2]), r"\[2\] are not one scalar"),
            (_scalar(1, "fixed", dims=[]), r"\[\] are not one scalar"),
            (_scalar(1, "fixed", dims=[1, 0]), "not one scalar"),
        ],
    )
    def test_not_a_pass_by_value_scalar_raises(self, tensor: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            scalar_source(tensor, "epsilon_tensor_uid")


# --------------------------------------------------------------------------
# Batch normalization (shared core) and the batch-norm ops
# --------------------------------------------------------------------------


def _channel(uid: int, dims: list[int], dtype: str = "float") -> dict:
    return {
        "uid": uid,
        "dims": dims,
        "strides": _layout_strides(dims, "nchw"),
        "data_type": dtype,
        "virtual": False,
    }


# The BatchnormAttributes ports, as utilities/json/BatchnormAttributes.hpp
# writes them. Written out here rather than read from the module, so _bn
# builds what the API binds, not what the module believes.
_BN_X, _BN_Y = "x_tensor_uid", "y_tensor_uid"
_BN_RUNNING = (
    "prev_running_mean_tensor_uid",
    "prev_running_variance_tensor_uid",
    "next_running_mean_tensor_uid",
    "next_running_variance_tensor_uid",
)


def _bn(
    *,
    batch: int = 2,
    spatial: tuple[int, ...] = (8, 8),
    channels: int = 4,
    layout: str = "nchw",
    y_layout: str | None = None,
    dtype: str = "float",
    y_dtype: str | None = None,
    y_virtual: bool = False,
    affine: list[int] | None = None,
    bias: list[int] | None = None,
    affine_dtype: str = "float",
    epsilon: str = "fixed",
    running_stats: bool = False,
    momentum: str | None = None,
    saved: tuple[str, ...] = ("mean", "inv_variance"),
    peer_stats: int = 0,
) -> tuple[dict, dict]:
    """A BatchnormAttributes node.

    ``x`` on uid 1, ``y`` on 2, scale and bias on 3 and 4, epsilon on 5, the
    saved statistics on 6 and 7, the running statistics on 8 to 11, momentum
    on 12, peer statistics from 13. ``affine`` is the scale shape (default
    ``[1, C, 1, ...]``), ``bias`` the bias shape (default: the scale's).
    """
    dims = [batch, channels, *spatial]
    per_channel = [1, channels, *(1 for _ in spatial)]
    affine = per_channel if affine is None else affine

    def activation(uid: int, dt: str, lay: str, virtual: bool) -> dict:
        strides = _layout_strides(dims, lay)
        return {
            "uid": uid,
            "dims": list(dims),
            "strides": strides,
            "data_type": dt,
            "virtual": virtual,
        }

    tensors = {
        1: activation(1, dtype, layout, False),
        2: activation(2, y_dtype or dtype, y_layout or layout, y_virtual),
        3: _channel(3, affine, affine_dtype),
        4: _channel(4, affine if bias is None else bias, affine_dtype),
        5: _scalar(5, epsilon),
        6: _channel(6, per_channel),
        7: _channel(7, per_channel),
        **{uid: _channel(uid, per_channel) for uid in range(8, 12)},
        **{uid: _channel(uid, per_channel) for uid in range(13, 13 + peer_stats)},
    }
    if momentum is not None:
        tensors[12] = _scalar(12, momentum)
    running = dict(zip(_BN_RUNNING, range(8, 12)))
    inputs = {
        _BN_X: 1,
        "scale_tensor_uid": 3,
        "bias_tensor_uid": 4,
        "epsilon_tensor_uid": 5,
        "peer_stats_tensor_uid": list(range(13, 13 + peer_stats)),
        "prev_running_mean_tensor_uid": None,
        "prev_running_variance_tensor_uid": None,
        "momentum_tensor_uid": 12 if momentum is not None else None,
    }
    outputs = {
        _BN_Y: 2,
        "mean_tensor_uid": 6 if "mean" in saved else None,
        "inv_variance_tensor_uid": 7 if "inv_variance" in saved else None,
        "next_running_mean_tensor_uid": None,
        "next_running_variance_tensor_uid": None,
    }
    if running_stats:
        for port, uid in running.items():
            (inputs if port.startswith("prev_") else outputs)[port] = uid
    node = {"type": "BatchnormAttributes", "inputs": inputs, "outputs": outputs}
    return node, tensors


class TestBatchnormTrainingBinding:
    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _bn()
        assert set(batchnorm.bound(node)) == table_fields(
            batchnorm_training.SCHEMA_FILE, batchnorm_training.SCHEMA_TABLE
        )

    def test_ports_are_the_schema_fields(self) -> None:
        m = batchnorm_training
        ports = {
            m.X,
            m.Y,
            *m.AFFINE,
            m.EPSILON,
            m.MOMENTUM,
            *m.RUNNING_STATS,
            *m.SAVED_STATS,
            m.PEER_STATS,
        }
        assert ports == table_fields(m.SCHEMA_FILE, m.SCHEMA_TABLE)

    def test_every_feature_is_reachable(self) -> None:
        """No catalogued feature is one extract() can never return."""
        extract = batchnorm_training.extract
        reached = extract(*_bn())
        for kwargs in (
            {"spatial": (), "dtype": "half", "y_dtype": "float"},
            {"spatial": (8,), "layout": "nhwc", "dtype": "bfloat16"},
            {"spatial": (4, 8, 8), "affine": [4, 1]},
            {"epsilon": "runtime", "running_stats": True, "momentum": "fixed"},
            {"running_stats": True, "momentum": "runtime", "peer_stats": 2},
        ):
            reached |= extract(*_bn(**kwargs))
        assert reached == {feature.id for feature in batchnorm_training.FEATURES}


class TestBatchnormTrainingExtract:
    PLAIN = {
        "dtype:fp32",
        "spatial:2d",
        "layout:nchw",
        "affine:full_rank",
        "epsilon:fixed",
        "out:mean",
        "out:inv_variance",
    }

    def _extract(self, **kwargs) -> set[str]:
        return batchnorm_training.extract(*_bn(**kwargs))

    def _raises(self, match: str, **kwargs) -> None:
        with pytest.raises(FeatureError, match=match):
            self._extract(**kwargs)

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "spatial, expected",
        [((8,), "spatial:1d"), ((8, 8), "spatial:2d"), ((4, 8, 8), "spatial:3d")],
    )
    @pytest.mark.parametrize("layout", convolution.LAYOUTS)
    def test_spatial_and_layout(self, spatial, expected: str, layout: str) -> None:
        features = self._extract(spatial=spatial, layout=layout)
        spatial_and_layout = {expected, f"layout:{layout}"}
        assert (
            features == self.PLAIN - {"spatial:2d", "layout:nchw"} | spatial_and_layout
        )

    def test_spatial_none_adds_no_layout(self) -> None:
        # [N, C] is the same in both layouts.
        features = self._extract(spatial=())
        assert features == self.PLAIN - {"spatial:2d", "layout:nchw"} | {"spatial:none"}

    def test_disagreeing_layouts_raise(self) -> None:
        self._raises("layouts disagree", y_layout="nhwc")

    def test_uncatalogued_rank_raises(self) -> None:
        self._raises("4 spatial dims", spatial=(2, 2, 2, 2))

    def test_fewer_than_two_dims_raises(self) -> None:
        node, tensors = _bn(spatial=())
        for uid in (1, 2):
            tensors[uid] |= {"dims": [4], "strides": [1]}
        with pytest.raises(FeatureError, match="-1 spatial dims"):
            batchnorm_training.extract(node, tensors)

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    def test_dtype_is_x(self) -> None:
        features = self._extract(dtype="half", y_dtype="float")
        assert {"dtype:fp16", "precision:mixed"} <= features
        assert "dtype:fp32" not in features

    def test_mixed_precision_counts_only_non_virtual_tensors(self) -> None:
        mixed = {"dtype": "half", "y_dtype": "float"}
        assert "precision:mixed" in self._extract(**mixed)
        assert "precision:mixed" not in self._extract(**mixed, y_virtual=True)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"dtype": "int8"}, "x_tensor_uid dtype"),
            ({"y_dtype": "int32"}, r"\['int32'\]"),
            # A virtual y does not count toward precision, but its dtype is
            # still checked.
            ({"y_dtype": "int32", "y_virtual": True}, r"\['int32'\]"),
        ],
    )
    def test_uncatalogued_dtype_raises(self, kwargs: dict, match: str) -> None:
        self._raises(match, **kwargs)

    def test_activation_shapes_that_differ_raise(self) -> None:
        node, tensors = _bn()
        tensors[2]["dims"] = [2, 4, 8, 4]
        with pytest.raises(FeatureError, match="activation shapes differ"):
            batchnorm_training.extract(node, tensors)

    @pytest.mark.parametrize(
        "batch, spatial", [(1, ()), (1, (1,)), (1, (1, 1)), (1, (1, 1, 1))]
    )
    def test_one_value_per_channel_raises(self, batch: int, spatial) -> None:
        self._raises("one value per channel", batch=batch, spatial=spatial)

    @pytest.mark.parametrize("batch, spatial", [(2, ()), (1, (2,)), (1, (1, 2))])
    def test_more_than_one_value_per_channel(self, batch: int, spatial) -> None:
        assert "affine:full_rank" in self._extract(batch=batch, spatial=spatial)

    @pytest.mark.parametrize(
        "spatial, affine",
        [
            ((8, 8), [4]),
            ((8, 8), [4, 1]),
            ((8, 8), [4, 1, 1]),
            ((8,), [4, 1]),
            ((4, 8, 8), [4, 1, 1, 1]),
        ],
    )
    def test_reduced_rank_affine(self, spatial, affine: list[int]) -> None:
        features = self._extract(spatial=spatial, affine=affine)
        assert "affine:reduced_rank" in features
        assert "affine:full_rank" not in features

    @pytest.mark.parametrize(
        "affine",
        [
            # Per-activation mode: the API rejects a spatial dim above 1.
            [1, 4, 8, 8],
            [1, 4, 8, 1],
            # The API takes [1, C] for 4D x; no provider reads it.
            [1, 4],
            [1, 4, 1],
            # [C, 1, ...] at the rank of x is neither shape.
            [4, 1, 1, 1],
            [1, 3, 1, 1],
            [3],
            [],
        ],
    )
    def test_uncatalogued_affine_shape_raises(self, affine: list[int]) -> None:
        self._raises("are neither", affine=affine)

    def test_scale_and_bias_shapes_that_differ_raise(self) -> None:
        self._raises("scale dims .* != bias dims", affine=[4, 1], bias=[1, 4, 1, 1])

    @pytest.mark.parametrize("affine_dtype", ["half", "bfloat16", "double"])
    def test_non_fp32_affine_raises(self, affine_dtype: str) -> None:
        self._raises("scale_tensor_uid dtype", affine_dtype=affine_dtype)

    def test_runtime_epsilon(self) -> None:
        features = self._extract(epsilon="runtime")
        assert features == self.PLAIN - {"epsilon:fixed"} | {"epsilon:runtime"}

    def test_epsilon_in_a_buffer_raises(self) -> None:
        self._raises("epsilon_tensor_uid is neither", epsilon="buffer")

    @pytest.mark.parametrize("momentum", ["fixed", "runtime"])
    def test_running_stats(self, momentum: str) -> None:
        features = self._extract(running_stats=True, momentum=momentum)
        assert features - self.PLAIN == {"running_stats:on", f"momentum:{momentum}"}

    def test_momentum_in_a_buffer_raises(self) -> None:
        self._raises(
            "momentum_tensor_uid is neither", running_stats=True, momentum="buffer"
        )

    @pytest.mark.parametrize("unbound", _BN_RUNNING)
    def test_partial_running_stats_raise(self, unbound: str) -> None:
        node, tensors = _bn(running_stats=True, momentum="fixed")
        side = "inputs" if unbound.startswith("prev_") else "outputs"
        assert node[side][unbound] is not None
        node[side][unbound] = None
        with pytest.raises(FeatureError, match="without the rest"):
            batchnorm_training.extract(node, tensors)

    def test_momentum_without_running_stats_raises(self) -> None:
        self._raises("momentum without running statistics", momentum="fixed")

    def test_running_stats_without_momentum_raise(self) -> None:
        self._raises("running statistics without momentum", running_stats=True)

    @pytest.mark.parametrize("uid", [6, 7, 8, 9, 10, 11])
    @pytest.mark.parametrize(
        "change, match",
        [
            ({"dims": [1, 4, 8, 8]}, r"are not \[1, 4, 1, ...\]"),
            ({"dims": [4, 1, 1, 1]}, r"are not \[1, 4, 1, ...\]"),
            ({"dims": [4]}, r"are not \[1, 4, 1, ...\]"),
            ({"data_type": "half"}, "dtype 'half' is not catalogued"),
        ],
    )
    def test_statistics_are_fp32_and_channel_only(
        self, uid: int, change: dict, match: str
    ) -> None:
        node, tensors = _bn(running_stats=True, momentum="fixed")
        tensors[uid] |= change
        with pytest.raises(FeatureError, match=match):
            batchnorm_training.extract(node, tensors)

    @pytest.mark.parametrize(
        "saved, expected",
        [
            ((), set()),
            (("mean",), {"out:mean"}),
            (("inv_variance",), {"out:inv_variance"}),
        ],
    )
    def test_saved_stats_are_independent(self, saved, expected: set[str]) -> None:
        features = self._extract(saved=saved)
        assert features - (self.PLAIN - {"out:mean", "out:inv_variance"}) == expected

    def test_peer_stats(self) -> None:
        assert "peer_stats:on" not in self._extract(peer_stats=0)
        assert "peer_stats:on" in self._extract(peer_stats=2)

    @pytest.mark.parametrize(
        "port",
        [
            "x_tensor_uid",
            "y_tensor_uid",
            "scale_tensor_uid",
            "bias_tensor_uid",
            "epsilon_tensor_uid",
        ],
    )
    def test_unbound_required_port_raises(self, port: str) -> None:
        node, tensors = _bn()
        side = "outputs" if port == _BN_Y else "inputs"
        node[side][port] = None
        with pytest.raises(FeatureError, match=f"required port {port}"):
            batchnorm_training.extract(node, tensors)


def _bn_bwd(
    *,
    batch: int = 2,
    spatial: tuple[int, ...] = (8, 8),
    channels: int = 4,
    layout: str = "nchw",
    dtype: str = "float",
    dy_dtype: str | None = None,
    dy_virtual: bool = False,
    saved_stats: bool = True,
    peer_stats: int = 0,
) -> tuple[dict, dict]:
    """A BatchnormBackwardAttributes node.

    ``dy`` on uid 1, ``x`` on 2, ``dx`` on 3, scale on 4, dscale and dbias on
    5 and 6, the saved mean and inverse variance on 7 and 8, peer statistics
    from 9. Ports as utilities/json/BatchnormBackwardAttributes.hpp writes
    them, written out rather than read from the module.
    """
    dims = [batch, channels, *spatial]
    per_channel = [1, channels, *(1 for _ in spatial)]
    strides = _layout_strides(dims, layout)
    tensors = {
        uid: {
            "uid": uid,
            "dims": list(dims),
            "strides": list(strides),
            "data_type": dt,
            "virtual": virtual,
        }
        for uid, dt, virtual in (
            (1, dy_dtype or dtype, dy_virtual),
            (2, dtype, False),
            (3, dtype, False),
        )
    }
    tensors |= {uid: _channel(uid, per_channel) for uid in range(4, 9 + peer_stats)}
    inputs = {
        "dy_tensor_uid": 1,
        "x_tensor_uid": 2,
        "mean_tensor_uid": 7 if saved_stats else None,
        "inv_variance_tensor_uid": 8 if saved_stats else None,
        "scale_tensor_uid": 4,
        "peer_stats_tensor_uid": list(range(9, 9 + peer_stats)),
    }
    outputs = {"dx_tensor_uid": 3, "dscale_tensor_uid": 5, "dbias_tensor_uid": 6}
    node = {"type": "BatchnormBackwardAttributes", "inputs": inputs, "outputs": outputs}
    return node, tensors


class TestBatchnormBackwardExtract:
    PLAIN = {"dtype:fp32", "spatial:2d", "layout:nchw", "saved_stats:on"}

    def _extract(self, **kwargs) -> set[str]:
        return batchnorm_backward.extract(*_bn_bwd(**kwargs))

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _bn_bwd()
        assert set(batchnorm.bound(node)) == table_fields(
            batchnorm_backward.SCHEMA_FILE, batchnorm_backward.SCHEMA_TABLE
        )

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"spatial": (), "dtype": "half", "dy_dtype": "float"},
            {"spatial": (8,), "layout": "nhwc", "dtype": "bfloat16"},
            {"spatial": (4, 8, 8), "peer_stats": 2},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {feature.id for feature in batchnorm_backward.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "spatial, expected",
        [((), "spatial:none"), ((8,), "spatial:1d"), ((4, 8, 8), "spatial:3d")],
    )
    def test_spatial(self, spatial, expected: str) -> None:
        assert expected in self._extract(spatial=spatial)

    def test_layout(self) -> None:
        assert "layout:nhwc" in self._extract(layout="nhwc")

    def test_dtype_is_x_not_a_virtual_dy(self) -> None:
        # Fused pointwise backward -> batch norm backward: dy is the virtual
        # fp32 gradient, the caller's tensors are fp16.
        features = self._extract(dtype="half", dy_dtype="float", dy_virtual=True)
        assert features == self.PLAIN - {"dtype:fp32"} | {"dtype:fp16"}

    def test_a_non_virtual_dy_of_another_dtype_is_mixed(self) -> None:
        features = self._extract(dtype="half", dy_dtype="float")
        assert {"dtype:fp16", "precision:mixed"} <= features

    def test_recomputed_stats(self) -> None:
        assert self._extract(saved_stats=False) == self.PLAIN - {"saved_stats:on"}

    @pytest.mark.parametrize("unbound", ["mean_tensor_uid", "inv_variance_tensor_uid"])
    def test_partial_saved_stats_raise(self, unbound: str) -> None:
        node, tensors = _bn_bwd()
        node["inputs"][unbound] = None
        with pytest.raises(FeatureError, match="without the rest"):
            batchnorm_backward.extract(node, tensors)

    def test_peer_stats(self) -> None:
        assert "peer_stats:on" not in self._extract(peer_stats=0)
        assert "peer_stats:on" in self._extract(peer_stats=1)

    def test_one_value_per_channel_raises(self) -> None:
        with pytest.raises(FeatureError, match="one value per channel"):
            self._extract(batch=1, spatial=(1, 1))

    def test_activation_shapes_that_differ_raise(self) -> None:
        node, tensors = _bn_bwd()
        tensors[3]["dims"] = [2, 4, 8, 4]
        with pytest.raises(FeatureError, match="activation shapes differ"):
            batchnorm_backward.extract(node, tensors)

    @pytest.mark.parametrize("uid", [4, 5, 6, 7, 8])
    @pytest.mark.parametrize(
        "change, match",
        [
            # Per-activation, and the reduced-rank shape forward training
            # takes: the backward API takes neither.
            ({"dims": [1, 4, 8, 8]}, r"are not \[1, 4, 1, ...\]"),
            ({"dims": [4, 1]}, r"are not \[1, 4, 1, ...\]"),
            ({"data_type": "half"}, "dtype 'half' is not catalogued"),
        ],
    )
    def test_per_channel_tensors_are_fp32_and_channel_only(
        self, uid: int, change: dict, match: str
    ) -> None:
        node, tensors = _bn_bwd()
        tensors[uid] |= change
        with pytest.raises(FeatureError, match=match):
            batchnorm_backward.extract(node, tensors)

    @pytest.mark.parametrize(
        "side, port",
        [
            ("inputs", "dy_tensor_uid"),
            ("inputs", "x_tensor_uid"),
            ("inputs", "scale_tensor_uid"),
            ("outputs", "dx_tensor_uid"),
            ("outputs", "dscale_tensor_uid"),
            ("outputs", "dbias_tensor_uid"),
        ],
    )
    def test_unbound_required_port_raises(self, side: str, port: str) -> None:
        node, tensors = _bn_bwd()
        node[side][port] = None
        with pytest.raises(FeatureError, match=f"required port {port}"):
            batchnorm_backward.extract(node, tensors)


def _bn_inf(
    *,
    batch: int = 2,
    spatial: tuple[int, ...] = (8, 8),
    channels: int = 4,
    layout: str = "nchw",
    dtype: str = "float",
    y_dtype: str | None = None,
    y_virtual: bool = False,
    epsilon: str | None = None,
) -> tuple[dict, dict]:
    """A BatchnormInferenceAttributes node, or with ``epsilon`` a
    BatchnormInferenceAttributesVarianceExt node.

    ``x`` on uid 1, ``y`` on 2, scale and bias on 3 and 4, the mean and
    inverse variance (the variance, with ``epsilon``) on 5 and 6, epsilon on
    7. Ports as utilities/json/BatchnormInference*.hpp write them, written out
    rather than read from the module.
    """
    dims = [batch, channels, *spatial]
    strides = _layout_strides(dims, layout)
    tensors = {
        uid: {
            "uid": uid,
            "dims": list(dims),
            "strides": list(strides),
            "data_type": dt,
            "virtual": virtual,
        }
        for uid, dt, virtual in ((1, dtype, False), (2, y_dtype or dtype, y_virtual))
    }
    per_channel = [1, channels, *(1 for _ in spatial)]
    tensors |= {uid: _channel(uid, per_channel) for uid in range(3, 7)}
    inputs = {
        "x_tensor_uid": 1,
        "scale_tensor_uid": 3,
        "bias_tensor_uid": 4,
        "mean_tensor_uid": 5,
        "inv_variance_tensor_uid": 6,
    }
    node_type = "BatchnormInferenceAttributes"
    if epsilon is not None:
        del inputs["inv_variance_tensor_uid"]
        inputs |= {"variance_tensor_uid": 6, "epsilon_tensor_uid": 7}
        tensors[7] = _scalar(7, epsilon)
        node_type = "BatchnormInferenceAttributesVarianceExt"
    node = {"type": node_type, "inputs": inputs, "outputs": {"y_tensor_uid": 2}}
    return node, tensors


class TestBatchnormInferenceExtract:
    PLAIN = {"dtype:fp32", "spatial:2d", "layout:nchw"}

    def _extract(self, **kwargs) -> set[str]:
        return batchnorm_inference.extract(*_bn_inf(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            batchnorm_inference.extract(node, tensors)

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _bn_inf()
        assert set(batchnorm.bound(node)) == table_fields(
            batchnorm_inference.SCHEMA_FILE, batchnorm_inference.SCHEMA_TABLE
        )

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"spatial": (), "dtype": "half", "y_dtype": "float"},
            {"spatial": (8,), "layout": "nhwc", "dtype": "bfloat16"},
            {"spatial": (4, 8, 8)},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {feature.id for feature in batchnorm_inference.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "spatial, expected",
        [((8,), "spatial:1d"), ((8, 8), "spatial:2d"), ((4, 8, 8), "spatial:3d")],
    )
    @pytest.mark.parametrize("layout", convolution.LAYOUTS)
    def test_spatial_and_layout(self, spatial, expected: str, layout: str) -> None:
        features = self._extract(spatial=spatial, layout=layout)
        assert features == {"dtype:fp32", expected, f"layout:{layout}"}

    def test_spatial_none_adds_no_layout(self) -> None:
        assert self._extract(spatial=()) == {"dtype:fp32", "spatial:none"}

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    def test_a_virtual_y_is_not_mixed(self) -> None:
        # Batch norm inference -> fused pointwise: y is the virtual fp32
        # intermediate, the caller's x is fp16.
        features = self._extract(dtype="half", y_dtype="float", y_virtual=True)
        assert features == self.PLAIN - {"dtype:fp32"} | {"dtype:fp16"}

    def test_a_non_virtual_y_of_another_dtype_is_mixed(self) -> None:
        features = self._extract(dtype="half", y_dtype="float")
        assert features == self.PLAIN - {"dtype:fp32"} | {
            "dtype:fp16",
            "precision:mixed",
        }

    def test_uncatalogued_dtype_raises(self) -> None:
        self._raises(*_bn_inf(dtype="int8"), "x_tensor_uid dtype")

    def test_one_value_per_channel_is_allowed(self) -> None:
        # The statistics are given, not computed from the batch.
        assert "spatial:2d" in self._extract(batch=1, spatial=(1, 1))

    def test_activation_shapes_that_differ_raise(self) -> None:
        node, tensors = _bn_inf()
        tensors[2]["dims"] = [2, 4, 8, 4]
        self._raises(node, tensors, "activation shapes differ")

    @pytest.mark.parametrize("uid", [3, 4, 5, 6])
    @pytest.mark.parametrize(
        "change, match",
        [
            ({"dims": [1, 4, 8, 8]}, r"are not \[1, 4, 1, ...\]"),
            ({"dims": [4, 1]}, r"are not \[1, 4, 1, ...\]"),
            ({"dims": [4]}, r"are not \[1, 4, 1, ...\]"),
            ({"data_type": "half"}, "dtype 'half' is not catalogued"),
        ],
    )
    def test_per_channel_tensors_are_fp32_and_channel_only(
        self, uid: int, change: dict, match: str
    ) -> None:
        node, tensors = _bn_inf()
        tensors[uid] |= change
        self._raises(node, tensors, match)

    @pytest.mark.parametrize(
        "side, port",
        [
            ("inputs", "x_tensor_uid"),
            ("inputs", "scale_tensor_uid"),
            ("inputs", "bias_tensor_uid"),
            ("inputs", "mean_tensor_uid"),
            ("inputs", "inv_variance_tensor_uid"),
            ("outputs", "y_tensor_uid"),
        ],
    )
    def test_unbound_required_port_raises(self, side: str, port: str) -> None:
        node, tensors = _bn_inf()
        node[side][port] = None
        self._raises(node, tensors, f"required port {port}")


class TestBatchnormInferenceVarianceExtExtract:
    """What differs from BatchnormInferenceAttributes; the rest is shared."""

    PLAIN = {"dtype:fp32", "spatial:2d", "layout:nchw", "epsilon:fixed"}

    def _extract(self, **kwargs) -> set[str]:
        kwargs.setdefault("epsilon", "fixed")
        return batchnorm_inference_variance_ext.extract(*_bn_inf(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            batchnorm_inference_variance_ext.extract(node, tensors)

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _bn_inf(epsilon="fixed")
        m = batchnorm_inference_variance_ext
        assert set(batchnorm.bound(node)) == table_fields(m.SCHEMA_FILE, m.SCHEMA_TABLE)

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"spatial": (), "dtype": "half", "y_dtype": "float"},
            {"spatial": (8,), "layout": "nhwc", "dtype": "bfloat16"},
            {"spatial": (4, 8, 8), "epsilon": "runtime"},
        ):
            reached |= self._extract(**kwargs)
        m = batchnorm_inference_variance_ext
        assert reached == {feature.id for feature in m.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    def test_shares_the_inference_features(self) -> None:
        kwargs = {"spatial": (8,), "layout": "nhwc", "dtype": "half"}
        assert self._extract(**kwargs) - {"epsilon:fixed"} == (
            batchnorm_inference.extract(*_bn_inf(**kwargs))
        )

    def test_runtime_epsilon(self) -> None:
        features = self._extract(epsilon="runtime")
        assert features == self.PLAIN - {"epsilon:fixed"} | {"epsilon:runtime"}

    def test_epsilon_in_a_buffer_raises(self) -> None:
        self._raises(*_bn_inf(epsilon="buffer"), "epsilon_tensor_uid is neither")

    @pytest.mark.parametrize(
        "change, match",
        [
            ({"dims": [1, 4, 8, 8]}, r"are not \[1, 4, 1, ...\]"),
            ({"data_type": "half"}, "dtype 'half' is not catalogued"),
        ],
    )
    def test_variance_is_fp32_and_channel_only(self, change: dict, match: str) -> None:
        node, tensors = _bn_inf(epsilon="fixed")
        tensors[6] |= change
        self._raises(node, tensors, f"variance_tensor_uid.*{match}")

    @pytest.mark.parametrize(
        "port", ["variance_tensor_uid", "epsilon_tensor_uid", "x_tensor_uid"]
    )
    def test_unbound_required_port_raises(self, port: str) -> None:
        node, tensors = _bn_inf(epsilon="fixed")
        node["inputs"][port] = None
        self._raises(node, tensors, f"required port {port}")


# --------------------------------------------------------------------------
# Layer normalization
# --------------------------------------------------------------------------


def _ln(
    *,
    dims: tuple[int, ...] = (2, 3, 4, 8),
    count: object = 1,
    layout: str = "nchw",
    y_layout: str | None = None,
    dtype: str = "float",
    y_dtype: str | None = None,
    y_virtual: bool = False,
    affine: list[int] | None = None,
    affine_dtype: str | None = None,
    phase: object = 2,
    stats_dtype: str | None = None,
    epsilon: str = "fixed",
) -> tuple[dict, dict]:
    """A LayernormAttributes node.

    ``x`` on uid 1, ``y`` on 2, scale and bias on 3 and 4, epsilon on 5, the
    mean and inverse variance on 6 and 7, bound for training (phase 2) only.
    ``affine`` is the scale and bias shape (default: ``x`` with 1 on the dims
    not normalized). Ports as utilities/json/LayernormAttributes.hpp writes
    them, written out rather than read from the module.
    """
    dims, normalized = list(dims), count if type(count) is int else 1
    kept = len(dims) - normalized

    def tensor(uid: int, shape: list[int], dt: str, lay: str = "nchw") -> dict:
        return {
            "uid": uid,
            "dims": list(shape),
            "strides": _layout_strides(list(shape), lay),
            "data_type": dt,
            "virtual": False,
        }

    affine = [1] * kept + dims[kept:] if affine is None else affine
    stats = dims[:kept] + [1] * normalized
    tensors = {
        1: tensor(1, dims, dtype, layout),
        2: tensor(2, dims, y_dtype or dtype, y_layout or layout),
        3: tensor(3, affine, affine_dtype or dtype),
        4: tensor(4, affine, affine_dtype or dtype),
        5: _scalar(5, epsilon),
        6: tensor(6, stats, stats_dtype or dtype),
        7: tensor(7, stats, stats_dtype or dtype),
    }
    tensors[2]["virtual"] = y_virtual
    training = phase == 2
    node = {
        "type": "LayernormAttributes",
        "inputs": {
            "x_tensor_uid": 1,
            "scale_tensor_uid": 3,
            "bias_tensor_uid": 4,
            "epsilon_tensor_uid": 5,
        },
        "outputs": {
            "y_tensor_uid": 2,
            "mean_tensor_uid": 6 if training else None,
            "inv_variance_tensor_uid": 7 if training else None,
        },
        "normalized_dim_count": count,
        "forward_phase": phase,
    }
    return node, tensors


class TestLayernormExtract:
    PLAIN = {
        "dtype:fp32",
        "dims:4",
        "layout:nchw",
        "norm:last_dim",
        "affine:full_rank",
        "phase:training",
        "epsilon:fixed",
    }

    def _extract(self, **kwargs) -> set[str]:
        return layernorm.extract(*_ln(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            layernorm.extract(node, tensors)

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _ln()
        written = set(batchnorm.bound(node)) | set(node) - {"type", "inputs", "outputs"}
        assert written == table_fields(layernorm.SCHEMA_FILE, layernorm.SCHEMA_TABLE)

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"dims": (8,), "dtype": "half", "y_dtype": "float", "phase": 1},
            {"dims": (2, 8), "dtype": "bfloat16", "affine_dtype": "float"},
            {"dims": (2, 3, 8), "count": 3, "stats_dtype": "bfloat16"},
            {"dims": (2, 3, 4, 4, 8), "count": 2, "layout": "nhwc"},
            {"affine": [4, 8], "count": 2, "epsilon": "runtime"},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {feature.id for feature in layernorm.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"y_dtype": "float"}, {"precision:mixed_io"}),
            ({"affine_dtype": "float"}, {"precision:mixed_affine"}),
            ({"stats_dtype": "float"}, {"precision:mixed_stats"}),
            ({"stats_dtype": "bfloat16"}, {"precision:mixed_stats"}),
            # The common mixed recipe: fp16 activations, fp32 parameters and
            # statistics; each role is counted.
            (
                {"y_dtype": "float", "affine_dtype": "float", "stats_dtype": "float"},
                {
                    "precision:mixed_io",
                    "precision:mixed_affine",
                    "precision:mixed_stats",
                },
            ),
        ],
    )
    def test_each_precision_role_is_its_own_feature(
        self, kwargs: dict, expected: set[str]
    ) -> None:
        features = self._extract(dtype="half", **kwargs)
        assert features == self.PLAIN - {"dtype:fp32"} | {"dtype:fp16"} | expected

    def test_a_virtual_y_is_not_mixed(self) -> None:
        features = self._extract(dtype="half", y_dtype="float", y_virtual=True)
        assert "precision:mixed_io" not in features

    def test_inference_has_no_statistics_to_mix(self) -> None:
        features = self._extract(dtype="half", stats_dtype="float", phase=1)
        assert "precision:mixed_stats" not in features

    @pytest.mark.parametrize(
        "dims, layouts",
        [
            ((8,), set()),
            ((2, 8), set()),
            ((2, 3, 8), {"layout:nchw"}),
            ((2, 3, 4, 8), {"layout:nchw"}),
            ((2, 3, 4, 4, 8), {"layout:nchw"}),
        ],
    )
    def test_dims(self, dims: tuple[int, ...], layouts: set[str]) -> None:
        # Below rank 3, channels first and channels last are one order.
        features = self._extract(dims=dims)
        assert {f for f in features if f.startswith(("dims:", "layout:"))} == {
            f"dims:{len(dims)}",
            *layouts,
        }

    @pytest.mark.parametrize("dims", [(2, 3, 8), (2, 3, 4, 8), (2, 3, 4, 4, 8)])
    def test_channels_last(self, dims: tuple[int, ...]) -> None:
        assert "layout:nhwc" in self._extract(dims=dims, layout="nhwc")

    def test_rank_above_five_raises(self) -> None:
        self._raises(*_ln(dims=(2, 2, 2, 2, 2, 8)), "rank 6 is not catalogued")

    def test_layouts_that_disagree_raise(self) -> None:
        self._raises(*_ln(y_layout="nhwc"), "layouts disagree")

    def test_activation_shapes_that_differ_raise(self) -> None:
        node, tensors = _ln()
        tensors[2]["dims"] = [2, 3, 8, 4]
        self._raises(node, tensors, "activation shapes differ")

    @pytest.mark.parametrize(
        "count, expected",
        [(1, "norm:last_dim"), (2, "norm:multi_dim"), (3, "norm:multi_dim")],
    )
    def test_norm(self, count: int, expected: str) -> None:
        features = self._extract(count=count)
        assert features == self.PLAIN - {"norm:last_dim"} | {expected}

    def test_every_dim_normalized_is_all_dims_and_full_rank(self) -> None:
        # The scale is x's shape; full and reduced rank are one shape.
        features = self._extract(count=4)
        assert features == self.PLAIN - {"norm:last_dim"} | {"norm:all_dims"}

    def test_a_vector_is_all_dims(self) -> None:
        assert "norm:all_dims" in self._extract(dims=(8,))

    @pytest.mark.parametrize("affine, count", [([8], 1), ([4, 8], 2), ([3, 4, 8], 3)])
    def test_reduced_rank_affine(self, affine: list[int], count: int) -> None:
        features = self._extract(affine=affine, count=count)
        assert "affine:reduced_rank" in features
        assert "affine:full_rank" not in features

    def test_a_normalized_dim_of_one_keeps_the_count(self) -> None:
        # x [2, 3, 1, 8] over its last 3: scale [1, 3, 1, 8]. The 1 is a
        # normalized dim, not a batch dim.
        features = self._extract(dims=(2, 3, 1, 8), count=3)
        assert "norm:multi_dim" in features

    @pytest.mark.parametrize(
        "affine, count",
        [
            ([1, 1, 4, 8], 1),  # the count says one dim; the scale two
            ([1, 1, 1, 8], 2),  # broadcast along a normalized dim
            ([1, 3, 1, 1], 1),  # the scale on a batch dim
            ([8, 1], 1),  # reduced rank, channel-first as batch norm's
            ([1, 8], 1),  # one-padded below the rank of x
            ([2, 3, 4, 8], 3),  # per batch index
        ],
    )
    def test_affine_not_the_normalized_dims_raises(
        self, affine: list[int], count: int
    ) -> None:
        self._raises(*_ln(affine=affine, count=count), "are neither")

    def test_scale_and_bias_shapes_that_differ_raise(self) -> None:
        node, tensors = _ln()
        tensors[4]["dims"] = [8]
        self._raises(node, tensors, "affine shapes differ")

    @pytest.mark.parametrize("count", [0, -1, 5, None, True, 1.0, "1"])
    def test_count_not_a_number_of_trailing_dims_raises(self, count: object) -> None:
        self._raises(*_ln(count=count), "normalized_dim_count")

    def test_missing_count_raises(self) -> None:
        node, tensors = _ln()
        del node["normalized_dim_count"]
        self._raises(node, tensors, "normalized_dim_count None")

    def test_inference(self) -> None:
        features = self._extract(phase=1)
        assert features == self.PLAIN - {"phase:training"} | {"phase:inference"}

    @pytest.mark.parametrize("phase", [0, 3, None, "TRAINING", True, 2.0])
    def test_uncatalogued_phase_raises(self, phase: object) -> None:
        self._raises(*_ln(phase=phase), "forward_phase .* is not catalogued")

    def test_missing_phase_raises(self) -> None:
        # The deserializer reads a missing phase as NOT_SET.
        node, tensors = _ln()
        del node["forward_phase"]
        self._raises(node, tensors, "forward_phase None")

    @pytest.mark.parametrize(
        "unbound", [("mean_tensor_uid",), ("inv_variance_tensor_uid",), ()]
    )
    def test_training_without_both_statistics_raises(self, unbound) -> None:
        node, tensors = _ln()
        for port in unbound or ("mean_tensor_uid", "inv_variance_tensor_uid"):
            node["outputs"][port] = None
        self._raises(node, tensors, "training binds the statistics")

    @pytest.mark.parametrize(
        "port, uid", [("mean_tensor_uid", 6), ("inv_variance_tensor_uid", 7)]
    )
    def test_inference_with_statistics_raises(self, port: str, uid: int) -> None:
        node, tensors = _ln(phase=1)
        node["outputs"][port] = uid
        self._raises(node, tensors, "inference binds the statistics")

    @pytest.mark.parametrize("uid", [6, 7])
    @pytest.mark.parametrize(
        "stats",
        [
            [2, 3, 4, 8],  # not reduced
            [1, 1, 1, 1],  # reduced over every dim
            [2, 3, 1, 8],  # reduced over the wrong dim
            [2, 3, 4],  # the batch dims only
        ],
    )
    def test_statistics_not_x_reduced_raise(self, uid: int, stats: list[int]) -> None:
        node, tensors = _ln()
        tensors[uid]["dims"] = stats
        self._raises(node, tensors, r"dims .* are not \[2, 3, 4, 1\]")

    @pytest.mark.parametrize(
        "uid, match",
        [
            (1, "x_tensor_uid dtype 'int8'"),
            (2, "y_tensor_uid dtype 'int8'"),
            (3, "scale_tensor_uid dtype 'int8'"),
            (4, "bias_tensor_uid dtype 'int8'"),
            (6, "mean_tensor_uid dtype 'int8'"),
            (7, "inv_variance_tensor_uid dtype 'int8'"),
        ],
    )
    def test_uncatalogued_dtype_raises(self, uid: int, match: str) -> None:
        node, tensors = _ln()
        tensors[uid]["data_type"] = "int8"
        self._raises(node, tensors, match)

    @pytest.mark.parametrize("uid, role", [(4, "affine"), (7, "statistics")])
    def test_a_role_split_in_dtype_raises(self, uid: int, role: str) -> None:
        # The provider requires each role to agree within itself.
        node, tensors = _ln()
        tensors[uid]["data_type"] = "half"
        self._raises(node, tensors, f"{role} dtypes differ")

    def test_runtime_epsilon(self) -> None:
        features = self._extract(epsilon="runtime")
        assert features == self.PLAIN - {"epsilon:fixed"} | {"epsilon:runtime"}

    def test_epsilon_in_a_buffer_raises(self) -> None:
        self._raises(*_ln(epsilon="buffer"), "epsilon_tensor_uid is neither")

    @pytest.mark.parametrize(
        "side, port",
        [
            ("inputs", "x_tensor_uid"),
            ("inputs", "scale_tensor_uid"),
            ("inputs", "bias_tensor_uid"),
            ("inputs", "epsilon_tensor_uid"),
            ("outputs", "y_tensor_uid"),
        ],
    )
    def test_unbound_required_port_raises(self, side: str, port: str) -> None:
        node, tensors = _ln()
        node[side][port] = None
        self._raises(node, tensors, f"required port {port}")


def _ln_bwd(
    *,
    dims: tuple[int, ...] = (2, 3, 4, 8),
    count: object = 1,
    layout: str = "nchw",
    dx_layout: str | None = None,
    dtype: str = "float",
    dy_dtype: str | None = None,
    dx_dtype: str | None = None,
    affine: list[int] | None = None,
    affine_dtype: str | None = None,
    stats: object = "padded",
    stats_dtype: str | None = None,
    epsilon: str = "fixed",
) -> tuple[dict, dict]:
    """A LayernormBackwardAttributes node.

    ``x`` on uid 1, ``dy`` on 2, ``dx`` on 3, scale, dscale and dbias on 4, 5
    and 6, the mean and inverse variance on 7 and 8, epsilon on 9.
    ``affine`` is the shape of scale and its gradients (default: ``x`` with 1
    on the dims not normalized). ``stats`` is "padded" (``x`` with 1 on the
    normalized dims), "batch" (the dims before them only) or None (unbound:
    recomputed); ``epsilon`` "default" leaves it unbound. Ports as
    utilities/json/LayernormBackwardAttributes.hpp writes them, written out
    rather than read from the module.
    """
    dims, normalized = list(dims), count if type(count) is int else 1
    kept = len(dims) - normalized

    def tensor(uid: int, shape: list[int], dt: str, lay: str = "nchw") -> dict:
        return {
            "uid": uid,
            "dims": list(shape),
            "strides": _layout_strides(list(shape), lay),
            "data_type": dt,
            "virtual": False,
        }

    affine = [1] * kept + dims[kept:] if affine is None else affine
    stats_dims = dims[:kept] + ([1] * normalized if stats == "padded" else [])
    tensors = {
        1: tensor(1, dims, dtype, layout),
        2: tensor(2, dims, dy_dtype or dtype, layout),
        3: tensor(3, dims, dx_dtype or dtype, dx_layout or layout),
        4: tensor(4, affine, affine_dtype or dtype),
        5: tensor(5, affine, affine_dtype or dtype),
        6: tensor(6, affine, affine_dtype or dtype),
        7: tensor(7, stats_dims, stats_dtype or dtype),
        8: tensor(8, stats_dims, stats_dtype or dtype),
        9: _scalar(9, epsilon),
    }
    saved = stats is not None
    node = {
        "type": "LayernormBackwardAttributes",
        "inputs": {
            "dy_tensor_uid": 2,
            "x_tensor_uid": 1,
            "scale_tensor_uid": 4,
            "mean_tensor_uid": 7 if saved else None,
            "inv_variance_tensor_uid": 8 if saved else None,
            "epsilon_tensor_uid": None if epsilon == "default" else 9,
        },
        "outputs": {
            "dx_tensor_uid": 3,
            "dscale_tensor_uid": 5,
            "dbias_tensor_uid": 6,
        },
        "normalized_dim_count": count,
    }
    return node, tensors


class TestLayernormBackwardExtract:
    PLAIN = {
        "dtype:fp32",
        "dims:4",
        "layout:nchw",
        "norm:last_dim",
        "affine:full_rank",
        "stats:saved",
    }
    RECOMPUTED = PLAIN - {"stats:saved"} | {"stats:recomputed", "epsilon:fixed"}

    def _extract(self, **kwargs) -> set[str]:
        return layernorm_backward.extract(*_ln_bwd(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            layernorm_backward.extract(node, tensors)

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _ln_bwd()
        written = set(batchnorm.bound(node)) | set(node) - {"type", "inputs", "outputs"}
        assert written == table_fields(
            layernorm_backward.SCHEMA_FILE, layernorm_backward.SCHEMA_TABLE
        )

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"dims": (8,), "dtype": "half", "dy_dtype": "float", "stats": None},
            {"dims": (2, 8), "dtype": "bfloat16", "affine_dtype": "float"},
            {"dims": (2, 3, 8), "count": 3, "stats_dtype": "bfloat16"},
            {"dims": (2, 3, 4, 4, 8), "count": 2, "layout": "nhwc"},
            {"affine": [4, 8], "count": 2, "stats": "batch"},
            {"stats": None, "epsilon": "runtime"},
            {"stats": None, "epsilon": "default"},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {feature.id for feature in layernorm_backward.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"dy_dtype": "float"}, {"precision:mixed_io"}),
            ({"dx_dtype": "float"}, {"precision:mixed_io"}),
            ({"affine_dtype": "float"}, {"precision:mixed_affine"}),
            ({"stats_dtype": "float"}, {"precision:mixed_stats"}),
            ({"stats_dtype": "bfloat16"}, {"precision:mixed_stats"}),
            # The bundles' mixed recipe: fp16 x and dx, fp32 dy, parameters
            # and statistics; each role is counted.
            (
                {"dy_dtype": "float", "affine_dtype": "float", "stats_dtype": "float"},
                {
                    "precision:mixed_io",
                    "precision:mixed_affine",
                    "precision:mixed_stats",
                },
            ),
        ],
    )
    def test_each_precision_role_is_its_own_feature(
        self, kwargs: dict, expected: set[str]
    ) -> None:
        features = self._extract(dtype="half", **kwargs)
        assert features == self.PLAIN - {"dtype:fp32"} | {"dtype:fp16"} | expected

    def test_recomputed_statistics_are_not_mixed(self) -> None:
        features = self._extract(dtype="half", stats_dtype="float", stats=None)
        assert "precision:mixed_stats" not in features

    @pytest.mark.parametrize(
        "dims, layouts",
        [
            ((8,), set()),
            ((2, 8), set()),
            ((2, 3, 8), {"layout:nchw"}),
            ((2, 3, 4, 8), {"layout:nchw"}),
            ((2, 3, 4, 4, 8), {"layout:nchw"}),
        ],
    )
    def test_dims(self, dims: tuple[int, ...], layouts: set[str]) -> None:
        features = self._extract(dims=dims)
        assert {f for f in features if f.startswith(("dims:", "layout:"))} == {
            f"dims:{len(dims)}",
            *layouts,
        }

    def test_channels_last(self) -> None:
        assert "layout:nhwc" in self._extract(layout="nhwc")

    def test_rank_above_five_raises(self) -> None:
        self._raises(*_ln_bwd(dims=(2, 2, 2, 2, 2, 8)), "rank 6 is not catalogued")

    def test_layouts_that_disagree_raise(self) -> None:
        self._raises(*_ln_bwd(dx_layout="nhwc"), "layouts disagree")

    @pytest.mark.parametrize("uid", [2, 3])
    def test_activation_shapes_that_differ_raise(self, uid: int) -> None:
        node, tensors = _ln_bwd()
        tensors[uid]["dims"] = [2, 3, 8, 4]
        self._raises(node, tensors, "activation shapes differ")

    @pytest.mark.parametrize(
        "count, expected",
        [(1, "norm:last_dim"), (2, "norm:multi_dim"), (3, "norm:multi_dim")],
    )
    def test_norm(self, count: int, expected: str) -> None:
        features = self._extract(count=count)
        assert features == self.PLAIN - {"norm:last_dim"} | {expected}

    def test_every_dim_normalized_is_all_dims_and_full_rank(self) -> None:
        # The scale is x's shape, so the statistics are one-padded.
        features = self._extract(count=4)
        assert features == self.PLAIN - {"norm:last_dim"} | {"norm:all_dims"}

    @pytest.mark.parametrize("affine, count", [([8], 1), ([4, 8], 2), ([3, 4, 8], 3)])
    def test_reduced_rank_affine_takes_batch_statistics(
        self, affine: list[int], count: int
    ) -> None:
        features = self._extract(affine=affine, count=count, stats="batch")
        assert features == self.PLAIN - {"affine:full_rank", "norm:last_dim"} | {
            "affine:reduced_rank",
            "norm:last_dim" if count == 1 else "norm:multi_dim",
        }

    def test_reduced_rank_affine_with_padded_statistics_raises(self) -> None:
        # The API: both one-padded or neither.
        self._raises(*_ln_bwd(affine=[8]), r"dims \[2, 3, 4, 1\] are not \[2, 3, 4\]")

    def test_full_rank_affine_with_batch_statistics_raises(self) -> None:
        self._raises(
            *_ln_bwd(stats="batch"), r"dims \[2, 3, 4\] are not \[2, 3, 4, 1\]"
        )

    @pytest.mark.parametrize(
        "affine, count",
        [
            ([1, 1, 4, 8], 1),
            ([1, 1, 1, 8], 2),
            ([1, 3, 1, 1], 1),
            ([2, 3, 4, 8], 3),
        ],
    )
    def test_affine_not_the_normalized_dims_raises(
        self, affine: list[int], count: int
    ) -> None:
        self._raises(*_ln_bwd(affine=affine, count=count), "are neither")

    @pytest.mark.parametrize("uid", [5, 6])
    def test_gradient_shapes_that_differ_from_scale_raise(self, uid: int) -> None:
        node, tensors = _ln_bwd()
        tensors[uid]["dims"] = [8]
        self._raises(node, tensors, "affine shapes differ")

    @pytest.mark.parametrize("count", [0, -1, 5, None, True, 1.0, "1"])
    def test_count_not_a_number_of_trailing_dims_raises(self, count: object) -> None:
        self._raises(*_ln_bwd(count=count), "normalized_dim_count")

    def test_missing_count_raises(self) -> None:
        node, tensors = _ln_bwd()
        del node["normalized_dim_count"]
        self._raises(node, tensors, "normalized_dim_count None")

    def test_recomputed_statistics(self) -> None:
        assert self._extract(stats=None) == self.RECOMPUTED

    @pytest.mark.parametrize("port", ["mean_tensor_uid", "inv_variance_tensor_uid"])
    def test_one_statistic_raises(self, port: str) -> None:
        node, tensors = _ln_bwd()
        node["inputs"][port] = None
        self._raises(node, tensors, "not both")

    @pytest.mark.parametrize("uid", [7, 8])
    @pytest.mark.parametrize(
        "stats",
        [
            [2, 3, 4, 8],  # not reduced
            [1, 1, 1, 1],  # reduced over every dim
            [2, 3, 1, 8],  # reduced over the wrong dim
            [2, 3, 4],  # the batch dims only, with a full-rank affine
        ],
    )
    def test_statistics_not_x_reduced_raise(self, uid: int, stats: list[int]) -> None:
        node, tensors = _ln_bwd()
        tensors[uid]["dims"] = stats
        self._raises(node, tensors, r"dims .* are not \[2, 3, 4, 1\]")

    @pytest.mark.parametrize(
        "uid, match",
        [
            (1, "x_tensor_uid dtype 'int8'"),
            (2, "dy_tensor_uid dtype 'int8'"),
            (3, "dx_tensor_uid dtype 'int8'"),
            (4, "scale_tensor_uid dtype 'int8'"),
            (5, "dscale_tensor_uid dtype 'int8'"),
            (6, "dbias_tensor_uid dtype 'int8'"),
            (7, "mean_tensor_uid dtype 'int8'"),
            (8, "inv_variance_tensor_uid dtype 'int8'"),
        ],
    )
    def test_uncatalogued_dtype_raises(self, uid: int, match: str) -> None:
        node, tensors = _ln_bwd()
        tensors[uid]["data_type"] = "int8"
        self._raises(node, tensors, match)

    @pytest.mark.parametrize(
        "uid, role", [(5, "affine"), (6, "affine"), (8, "statistics")]
    )
    def test_a_role_split_in_dtype_raises(self, uid: int, role: str) -> None:
        node, tensors = _ln_bwd()
        tensors[uid]["data_type"] = "half"
        self._raises(node, tensors, f"{role} dtypes differ")

    @pytest.mark.parametrize("epsilon", ["default", "fixed", "runtime"])
    def test_epsilon_counts_only_when_recomputed(self, epsilon: str) -> None:
        # Saved statistics leave epsilon unused.
        assert self._extract(epsilon=epsilon) == self.PLAIN
        features = self._extract(epsilon=epsilon, stats=None)
        assert features == self.RECOMPUTED - {"epsilon:fixed"} | {f"epsilon:{epsilon}"}

    @pytest.mark.parametrize("stats", ["padded", None])
    def test_epsilon_in_a_buffer_raises(self, stats: object) -> None:
        # Unused or not, the API rejects it.
        self._raises(*_ln_bwd(epsilon="buffer", stats=stats), "epsilon_tensor_uid is")

    @pytest.mark.parametrize(
        "side, port",
        [
            ("inputs", "dy_tensor_uid"),
            ("inputs", "x_tensor_uid"),
            ("inputs", "scale_tensor_uid"),
            ("outputs", "dx_tensor_uid"),
            ("outputs", "dscale_tensor_uid"),
            ("outputs", "dbias_tensor_uid"),
        ],
    )
    def test_unbound_required_port_raises(self, side: str, port: str) -> None:
        node, tensors = _ln_bwd()
        node[side][port] = None
        self._raises(node, tensors, f"required port {port}")


def _rms(
    *,
    dims: tuple[int, ...] = (2, 3, 4, 8),
    scale: list[int] | None = None,
    layout: str = "nchw",
    y_layout: str | None = None,
    dtype: str = "float",
    y_dtype: str | None = None,
    y_virtual: bool = False,
    affine_dtype: str | None = None,
    bias: bool = True,
    phase: object = "TRAINING",
    stats_dtype: str | None = None,
    epsilon: str = "fixed",
) -> tuple[dict, dict]:
    """An RMSNormAttributes node.

    ``x`` on uid 1, ``y`` on 2, scale and bias on 3 and 4, epsilon on 5,
    ``inv_rms`` on 6. ``scale`` is the scale and bias shape (default: the
    last dim of ``x``, 1 elsewhere); ``inv_rms`` is ``x`` with 1 where the
    scale is not 1, as the provider derives it. Ports as
    utilities/json/RMSNormAttributes.hpp writes them: an unbound optional
    port, like an unset phase, is left out.
    """
    dims = list(dims)

    def tensor(uid: int, shape: list[int], dt: str, lay: str = "nchw") -> dict:
        return {
            "uid": uid,
            "dims": list(shape),
            "strides": _layout_strides(list(shape), lay),
            "data_type": dt,
            "virtual": False,
        }

    scale = [1] * (len(dims) - 1) + dims[-1:] if scale is None else scale
    inv_rms = [1 if s != 1 else d for s, d in zip(scale, dims)]
    tensors = {
        1: tensor(1, dims, dtype, layout),
        2: tensor(2, dims, y_dtype or dtype, y_layout or layout),
        3: tensor(3, scale, affine_dtype or dtype),
        4: tensor(4, scale, affine_dtype or dtype),
        5: _scalar(5, epsilon),
        6: tensor(6, inv_rms, stats_dtype or dtype),
    }
    tensors[2]["virtual"] = y_virtual
    node = {
        "type": "RMSNormAttributes",
        "inputs": {"x_tensor_uid": 1, "scale_tensor_uid": 3, "epsilon_tensor_uid": 5},
        "outputs": {"y_tensor_uid": 2},
        "forward_phase": phase,
    }
    if bias:
        node["inputs"]["bias_tensor_uid"] = 4
    if phase == "TRAINING":
        node["outputs"]["inv_rms_tensor_uid"] = 6
    return node, tensors


class TestRMSNormExtract:
    PLAIN = {
        "dtype:fp32",
        "dims:4",
        "layout:nchw",
        "norm:last_dim",
        "bias:bound",
        "phase:training",
        "epsilon:fixed",
    }

    def _extract(self, **kwargs) -> set[str]:
        return rmsnorm.extract(*_rms(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            rmsnorm.extract(node, tensors)

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _rms()
        written = set(batchnorm.bound(node)) | set(node) - {"type", "inputs", "outputs"}
        assert written == table_fields(rmsnorm.SCHEMA_FILE, rmsnorm.SCHEMA_TABLE)

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"dims": (2, 8), "dtype": "half", "y_dtype": "float", "bias": False},
            {"dims": (2, 3, 8), "dtype": "bfloat16", "phase": "INFERENCE"},
            {"dtype": "half", "affine_dtype": "float", "stats_dtype": "float"},
            {"dims": (2, 3, 4, 4, 8), "scale": [1, 1, 1, 4, 8], "layout": "nhwc"},
            {"epsilon": "runtime"},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {feature.id for feature in rmsnorm.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"y_dtype": "float"}, {"precision:mixed_io"}),
            ({"affine_dtype": "float"}, {"precision:mixed_affine"}),
            ({"stats_dtype": "float"}, {"precision:mixed_stats"}),
            ({"stats_dtype": "bfloat16"}, {"precision:mixed_stats"}),
            # The bundles' mixed recipe: fp16 x, fp32 y, scale, bias and
            # inv_rms; each role is counted.
            (
                {"y_dtype": "float", "affine_dtype": "float", "stats_dtype": "float"},
                {
                    "precision:mixed_io",
                    "precision:mixed_affine",
                    "precision:mixed_stats",
                },
            ),
        ],
    )
    def test_each_precision_role_is_its_own_feature(
        self, kwargs: dict, expected: set[str]
    ) -> None:
        features = self._extract(dtype="half", **kwargs)
        assert features == self.PLAIN - {"dtype:fp32"} | {"dtype:fp16"} | expected

    def test_a_virtual_y_is_not_mixed(self) -> None:
        features = self._extract(dtype="half", y_dtype="float", y_virtual=True)
        assert "precision:mixed_io" not in features

    def test_inference_has_no_statistics_to_mix(self) -> None:
        features = self._extract(dtype="half", stats_dtype="float", phase="INFERENCE")
        assert "precision:mixed_stats" not in features

    @pytest.mark.parametrize(
        "dims, layouts",
        [
            ((2, 8), set()),
            ((2, 3, 8), {"layout:nchw"}),
            ((2, 3, 4, 8), {"layout:nchw"}),
            ((2, 3, 4, 4, 8), {"layout:nchw"}),
        ],
    )
    def test_dims(self, dims: tuple[int, ...], layouts: set[str]) -> None:
        # Below rank 3, channels first and channels last are one order.
        features = self._extract(dims=dims)
        assert {f for f in features if f.startswith(("dims:", "layout:"))} == {
            f"dims:{len(dims)}",
            *layouts,
        }

    @pytest.mark.parametrize("dims", [(2, 3, 8), (2, 3, 4, 8), (2, 3, 4, 4, 8)])
    def test_channels_last(self, dims: tuple[int, ...]) -> None:
        assert "layout:nhwc" in self._extract(dims=dims, layout="nhwc")

    @pytest.mark.parametrize("dims", [(8,), (2, 2, 2, 2, 2, 8)])
    def test_rank_outside_two_to_five_raises(self, dims: tuple[int, ...]) -> None:
        # The API's floor is 2.
        self._raises(*_rms(dims=dims), f"rank {len(dims)} is not catalogued")

    def test_layouts_that_disagree_raise(self) -> None:
        self._raises(*_rms(y_layout="nhwc"), "layouts disagree")

    def test_activation_shapes_that_differ_raise(self) -> None:
        node, tensors = _rms()
        tensors[2]["dims"] = [2, 3, 8, 4]
        self._raises(node, tensors, "activation shapes differ")

    @pytest.mark.parametrize(
        "dims, scale, expected",
        [
            ((2, 3, 4, 8), [1, 1, 1, 8], "norm:last_dim"),
            ((2, 3, 4, 8), [1, 1, 4, 8], "norm:multi_dim"),
            ((2, 3, 4, 8), [1, 3, 4, 8], "norm:all_but_batch"),
            ((2, 3, 4, 4, 8), [1, 1, 4, 4, 8], "norm:multi_dim"),
            # Rank 2 has one dim to normalize: all but the batch.
            ((2, 8), [1, 8], "norm:all_but_batch"),
            # A batch of one: scale is x, but the batch dim is not normalized.
            ((1, 3, 4, 8), [1, 3, 4, 8], "norm:all_but_batch"),
            # x's 1 joins the trailing run: the API normalizes it too.
            ((2, 3, 1, 8), [1, 3, 1, 8], "norm:all_but_batch"),
            ((2, 3, 1, 8), [1, 1, 1, 8], "norm:multi_dim"),
            # The per-channel scale fits when the spatial dims are 1.
            ((2, 3, 1, 1), [1, 3, 1, 1], "norm:all_but_batch"),
        ],
    )
    def test_norm(self, dims: tuple[int, ...], scale: list[int], expected: str) -> None:
        features = self._extract(dims=dims, scale=scale)
        assert {f for f in features if f.startswith("norm:")} == {expected}

    @pytest.mark.parametrize(
        "scale, match",
        [
            ([8], "not the rank of x"),
            ([4, 8], "not the rank of x"),
            ([1, 1, 1, 1, 8], "not the rank of x"),
            ([2, 3, 4, 8], "not 1 then a trailing run"),  # per batch index
            ([1, 3, 1, 1], "not 1 then a trailing run"),  # matches no trailing dim
            ([1, 3, 1, 8], "not 1 then a trailing run"),  # a sandwich
            ([1, 5, 4, 8], "not 1 then a trailing run"),  # a leading non-match
            ([1, 1, 4, 1], "not 1 then a trailing run"),  # broadcast on the last
        ],
    )
    def test_scale_not_a_normalized_shape_raises(
        self, scale: list[int], match: str
    ) -> None:
        self._raises(*_rms(scale=scale), match)

    def test_scale_and_bias_shapes_that_differ_raise(self) -> None:
        node, tensors = _rms()
        tensors[4]["dims"] = [1, 1, 4, 8]
        self._raises(node, tensors, "affine shapes differ")

    def test_no_bias(self) -> None:
        features = self._extract(bias=False)
        assert features == self.PLAIN - {"bias:bound"} | {"bias:none"}

    def test_a_null_bias_is_no_bias(self) -> None:
        node, tensors = _rms()
        node["inputs"]["bias_tensor_uid"] = None
        assert "bias:none" in rmsnorm.extract(node, tensors)

    def test_bias_uid_zero_is_bound(self) -> None:
        node, tensors = _rms()
        tensors[0] = {**tensors.pop(4), "uid": 0}
        node["inputs"]["bias_tensor_uid"] = 0
        assert "bias:bound" in rmsnorm.extract(node, tensors)

    def test_inference(self) -> None:
        features = self._extract(phase="INFERENCE")
        assert features == self.PLAIN - {"phase:training"} | {"phase:inference"}

    @pytest.mark.parametrize(
        "phase", ["NOT_SET", "training", "", 1, 2, None, True, ["TRAINING"]]
    )
    def test_uncatalogued_phase_raises(self, phase: object) -> None:
        node, tensors = _rms()
        node["forward_phase"] = phase
        self._raises(node, tensors, "forward_phase .* is not catalogued")

    def test_missing_phase_raises(self) -> None:
        # The serializer leaves an unset (NOT_SET) phase out.
        node, tensors = _rms()
        del node["forward_phase"]
        self._raises(node, tensors, "forward_phase None")

    def test_training_without_inv_rms_raises(self) -> None:
        node, tensors = _rms()
        del node["outputs"]["inv_rms_tensor_uid"]
        self._raises(node, tensors, "training does not bind inv_rms")

    def test_inference_with_inv_rms_raises(self) -> None:
        node, tensors = _rms(phase="INFERENCE")
        node["outputs"]["inv_rms_tensor_uid"] = 6
        self._raises(node, tensors, "inference binds inv_rms")

    @pytest.mark.parametrize(
        "inv_rms",
        [
            [2, 3, 4, 8],  # not reduced
            [1, 1, 1, 1],  # reduced over every dim
            [2, 3, 1, 8],  # reduced over the wrong dim
            [2, 3, 4],  # the leading dims only
        ],
    )
    def test_inv_rms_not_x_reduced_raises(self, inv_rms: list[int]) -> None:
        node, tensors = _rms()
        tensors[6]["dims"] = inv_rms
        self._raises(node, tensors, r"dims .* are not \[2, 3, 4, 1\]")

    def test_inv_rms_follows_the_normalized_dims(self) -> None:
        # The builder derives inv_rms as the provider does; the module reads
        # it from the scale's trailing run. The two agree.
        features = self._extract(dims=(2, 3, 1, 8), scale=[1, 3, 1, 8])
        assert "phase:training" in features

    @pytest.mark.parametrize(
        "uid, match",
        [
            (1, "x_tensor_uid dtype 'int8'"),
            (2, "y_tensor_uid dtype 'int8'"),
            (3, "scale_tensor_uid dtype 'int8'"),
            (4, "bias_tensor_uid dtype 'int8'"),
            (6, "inv_rms_tensor_uid dtype 'int8'"),
        ],
    )
    def test_uncatalogued_dtype_raises(self, uid: int, match: str) -> None:
        node, tensors = _rms()
        tensors[uid]["data_type"] = "int8"
        self._raises(node, tensors, match)

    def test_an_affine_split_in_dtype_raises(self) -> None:
        # The provider requires scale and bias to agree.
        node, tensors = _rms()
        tensors[4]["data_type"] = "half"
        self._raises(node, tensors, "affine dtypes differ")

    def test_runtime_epsilon(self) -> None:
        features = self._extract(epsilon="runtime")
        assert features == self.PLAIN - {"epsilon:fixed"} | {"epsilon:runtime"}

    def test_epsilon_in_a_buffer_raises(self) -> None:
        self._raises(*_rms(epsilon="buffer"), "epsilon_tensor_uid is neither")

    @pytest.mark.parametrize(
        "side, port",
        [
            ("inputs", "x_tensor_uid"),
            ("inputs", "scale_tensor_uid"),
            ("inputs", "epsilon_tensor_uid"),
            ("outputs", "y_tensor_uid"),
        ],
    )
    def test_unbound_required_port_raises(self, side: str, port: str) -> None:
        node, tensors = _rms()
        node[side][port] = None
        self._raises(node, tensors, f"required port {port}")


def _rms_bwd(
    *,
    dims: tuple[int, ...] = (2, 3, 4, 8),
    scale: list[int] | None = None,
    layout: str = "nchw",
    dy_layout: str | None = None,
    dtype: str = "float",
    dy_dtype: str | None = None,
    dx_dtype: str | None = None,
    dy_virtual: bool = False,
    affine_dtype: str | None = None,
    dbias: bool = True,
    stats_dtype: str | None = None,
) -> tuple[dict, dict]:
    """An RMSNormBackwardAttributes node.

    ``x`` on uid 1, ``dy`` on 2, ``dx`` on 3, scale, dscale and dbias on 4 to
    6, ``inv_rms`` on 7. ``scale`` and ``inv_rms`` are as in ``_rms``. Ports
    as utilities/json/RMSNormBackwardAttributes.hpp writes them: an unbound
    dbias is left out.
    """
    dims = list(dims)

    def tensor(uid: int, shape: list[int], dt: str, lay: str = "nchw") -> dict:
        return {
            "uid": uid,
            "dims": list(shape),
            "strides": _layout_strides(list(shape), lay),
            "data_type": dt,
            "virtual": False,
        }

    scale = [1] * (len(dims) - 1) + dims[-1:] if scale is None else scale
    inv_rms = [1 if s != 1 else d for s, d in zip(scale, dims)]
    tensors = {
        1: tensor(1, dims, dtype, layout),
        2: tensor(2, dims, dy_dtype or dtype, dy_layout or layout),
        3: tensor(3, dims, dx_dtype or dtype, layout),
        4: tensor(4, scale, affine_dtype or dtype),
        5: tensor(5, scale, affine_dtype or dtype),
        6: tensor(6, scale, affine_dtype or dtype),
        7: tensor(7, inv_rms, stats_dtype or dtype),
    }
    tensors[2]["virtual"] = dy_virtual
    node = {
        "type": "RMSNormBackwardAttributes",
        "inputs": {
            "dy_tensor_uid": 2,
            "x_tensor_uid": 1,
            "scale_tensor_uid": 4,
            "inv_rms_tensor_uid": 7,
        },
        "outputs": {"dx_tensor_uid": 3, "dscale_tensor_uid": 5},
    }
    if dbias:
        node["outputs"]["dbias_tensor_uid"] = 6
    return node, tensors


class TestRMSNormBackwardExtract:
    PLAIN = {
        "dtype:fp32",
        "dims:4",
        "layout:nchw",
        "norm:last_dim",
        "dbias:bound",
    }

    def _extract(self, **kwargs) -> set[str]:
        return rmsnorm_backward.extract(*_rms_bwd(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            rmsnorm_backward.extract(node, tensors)

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _rms_bwd()
        written = set(batchnorm.bound(node)) | set(node) - {"type", "inputs", "outputs"}
        assert written == table_fields(
            rmsnorm_backward.SCHEMA_FILE, rmsnorm_backward.SCHEMA_TABLE
        )

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"dims": (2, 8), "dtype": "half", "dy_dtype": "float", "dbias": False},
            {"dims": (2, 3, 8), "dtype": "bfloat16"},
            {"dtype": "half", "affine_dtype": "float", "stats_dtype": "float"},
            {"dims": (2, 3, 4, 4, 8), "scale": [1, 1, 1, 4, 8], "layout": "nhwc"},
            {"scale": [1, 3, 4, 8]},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {feature.id for feature in rmsnorm_backward.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"dy_dtype": "float"}, {"precision:mixed_io"}),
            ({"dx_dtype": "float"}, {"precision:mixed_io"}),
            ({"affine_dtype": "float"}, {"precision:mixed_affine"}),
            ({"stats_dtype": "float"}, {"precision:mixed_stats"}),
            # The bundles' mixed recipes: fp16 x and dx, and fp32 dy, affine
            # and inv_rms; or fp16 dy, dx and x with fp32 affine and inv_rms.
            (
                {"dy_dtype": "float", "affine_dtype": "float", "stats_dtype": "float"},
                {
                    "precision:mixed_io",
                    "precision:mixed_affine",
                    "precision:mixed_stats",
                },
            ),
            (
                {"affine_dtype": "float", "stats_dtype": "float"},
                {"precision:mixed_affine", "precision:mixed_stats"},
            ),
        ],
    )
    def test_each_precision_role_is_its_own_feature(
        self, kwargs: dict, expected: set[str]
    ) -> None:
        features = self._extract(dtype="half", **kwargs)
        assert features == self.PLAIN - {"dtype:fp32"} | {"dtype:fp16"} | expected

    def test_fp32_x_with_a_half_dy_is_mixed(self) -> None:
        # The bundles' other recipe: fp32 x and dx, a bf16 or fp16 dy.
        features = self._extract(dy_dtype="bfloat16")
        assert features == self.PLAIN | {"precision:mixed_io"}

    def test_a_virtual_dy_is_not_mixed(self) -> None:
        features = self._extract(dtype="half", dy_dtype="float", dy_virtual=True)
        assert "precision:mixed_io" not in features

    @pytest.mark.parametrize(
        "dims, layouts",
        [
            ((2, 8), set()),
            ((2, 3, 8), {"layout:nchw"}),
            ((2, 3, 4, 8), {"layout:nchw"}),
            ((2, 3, 4, 4, 8), {"layout:nchw"}),
        ],
    )
    def test_dims(self, dims: tuple[int, ...], layouts: set[str]) -> None:
        features = self._extract(dims=dims)
        assert {f for f in features if f.startswith(("dims:", "layout:"))} == {
            f"dims:{len(dims)}",
            *layouts,
        }

    @pytest.mark.parametrize("dims", [(2, 3, 8), (2, 3, 4, 8), (2, 3, 4, 4, 8)])
    def test_channels_last(self, dims: tuple[int, ...]) -> None:
        assert "layout:nhwc" in self._extract(dims=dims, layout="nhwc")

    @pytest.mark.parametrize("dims", [(8,), (2, 2, 2, 2, 2, 8)])
    def test_rank_outside_two_to_five_raises(self, dims: tuple[int, ...]) -> None:
        self._raises(*_rms_bwd(dims=dims), f"rank {len(dims)} is not catalogued")

    def test_layouts_that_disagree_raise(self) -> None:
        self._raises(*_rms_bwd(dy_layout="nhwc"), "layouts disagree")

    @pytest.mark.parametrize("uid", [2, 3])
    def test_activation_shapes_that_differ_raise(self, uid: int) -> None:
        node, tensors = _rms_bwd()
        tensors[uid]["dims"] = [2, 3, 8, 4]
        self._raises(node, tensors, "activation shapes differ")

    @pytest.mark.parametrize(
        "dims, scale, expected",
        [
            ((2, 3, 4, 8), [1, 1, 1, 8], "norm:last_dim"),
            ((2, 3, 4, 8), [1, 1, 4, 8], "norm:multi_dim"),
            ((2, 3, 4, 8), [1, 3, 4, 8], "norm:all_but_batch"),
            ((2, 3, 4, 4, 8), [1, 1, 4, 4, 8], "norm:multi_dim"),
            ((2, 3, 4, 4, 8), [1, 3, 4, 4, 8], "norm:all_but_batch"),
        ],
    )
    def test_norm(self, dims: tuple[int, ...], scale: list[int], expected: str) -> None:
        features = self._extract(dims=dims, scale=scale)
        assert {f for f in features if f.startswith("norm:")} == {expected}

    @pytest.mark.parametrize(
        "scale, match",
        [
            ([8], "not the rank of x"),
            ([2, 3, 4, 8], "not 1 then a trailing run"),
            ([1, 3, 1, 8], "not 1 then a trailing run"),
        ],
    )
    def test_scale_not_a_normalized_shape_raises(
        self, scale: list[int], match: str
    ) -> None:
        self._raises(*_rms_bwd(scale=scale), match)

    @pytest.mark.parametrize("uid", [5, 6])
    def test_affine_shapes_that_differ_raise(self, uid: int) -> None:
        node, tensors = _rms_bwd()
        tensors[uid]["dims"] = [1, 1, 4, 8]
        self._raises(node, tensors, "affine shapes differ")

    @pytest.mark.parametrize("uid", [5, 6])
    def test_an_affine_split_in_dtype_raises(self, uid: int) -> None:
        # The provider requires scale, dscale and dbias to agree.
        node, tensors = _rms_bwd()
        tensors[uid]["data_type"] = "half"
        self._raises(node, tensors, "affine dtypes differ")

    def test_no_dbias(self) -> None:
        features = self._extract(dbias=False)
        assert features == self.PLAIN - {"dbias:bound"} | {"dbias:none"}

    def test_no_dbias_leaves_its_tensor_out_of_the_affine_dtype(self) -> None:
        node, tensors = _rms_bwd(dbias=False)
        tensors[6]["data_type"] = "half"
        assert rmsnorm_backward.extract(node, tensors) == self.PLAIN - {
            "dbias:bound"
        } | {"dbias:none"}

    def test_a_null_dbias_is_no_dbias(self) -> None:
        node, tensors = _rms_bwd()
        node["outputs"]["dbias_tensor_uid"] = None
        assert "dbias:none" in rmsnorm_backward.extract(node, tensors)

    def test_dbias_uid_zero_is_bound(self) -> None:
        node, tensors = _rms_bwd()
        tensors[0] = {**tensors.pop(6), "uid": 0}
        node["outputs"]["dbias_tensor_uid"] = 0
        assert "dbias:bound" in rmsnorm_backward.extract(node, tensors)

    @pytest.mark.parametrize(
        "inv_rms",
        [
            [2, 3, 4, 8],  # not reduced
            [1, 1, 1, 1],  # reduced over every dim
            [2, 3, 1, 8],  # reduced over the wrong dim
            [2, 3, 4],  # the leading dims only
        ],
    )
    def test_inv_rms_not_x_reduced_raises(self, inv_rms: list[int]) -> None:
        node, tensors = _rms_bwd()
        tensors[7]["dims"] = inv_rms
        self._raises(node, tensors, r"dims .* are not \[2, 3, 4, 1\]")

    def test_inv_rms_follows_the_normalized_dims(self) -> None:
        features = self._extract(dims=(2, 3, 1, 8), scale=[1, 3, 1, 8])
        assert "norm:all_but_batch" in features

    @pytest.mark.parametrize(
        "uid, match",
        [
            (1, "x_tensor_uid dtype 'int8'"),
            (2, "dy_tensor_uid dtype 'int8'"),
            (3, "dx_tensor_uid dtype 'int8'"),
            (4, "scale_tensor_uid dtype 'int8'"),
            (5, "dscale_tensor_uid dtype 'int8'"),
            (6, "dbias_tensor_uid dtype 'int8'"),
            (7, "inv_rms_tensor_uid dtype 'int8'"),
        ],
    )
    def test_uncatalogued_dtype_raises(self, uid: int, match: str) -> None:
        node, tensors = _rms_bwd()
        tensors[uid]["data_type"] = "int8"
        self._raises(node, tensors, match)

    @pytest.mark.parametrize(
        "side, port",
        [
            ("inputs", "dy_tensor_uid"),
            ("inputs", "x_tensor_uid"),
            ("inputs", "scale_tensor_uid"),
            ("inputs", "inv_rms_tensor_uid"),
            ("outputs", "dx_tensor_uid"),
            ("outputs", "dscale_tensor_uid"),
        ],
    )
    def test_unbound_required_port_raises(self, side: str, port: str) -> None:
        node, tensors = _rms_bwd()
        node[side][port] = None
        self._raises(node, tensors, f"required port {port}")


# --------------------------------------------------------------------------
# MatmulAttributes
# --------------------------------------------------------------------------


def _matrix_strides(dims: list[int], layout: str) -> list[int]:
    """Packed strides for ``dims``, the last two dims in ``layout`` order."""
    *batch, rows, cols = dims
    inner = [cols, 1] if layout == "row_major" else [1, rows]
    strides = []
    step = rows * cols
    for d in reversed(batch):
        strides.insert(0, step)
        step *= d
    return strides + inner


def _matmul(
    *,
    batch_a: tuple[int, ...] = (3,),
    batch_b: tuple[int, ...] | None = None,
    m: int = 16,
    k: int = 32,
    n: int = 128,
    dtype: str = "float",
    b_dtype: str | None = None,
    c_dtype: str | None = None,
    a_layout: str = "row_major",
    b_layout: str = "row_major",
    c_layout: str = "row_major",
    a_virtual: bool = False,
    b_virtual: bool = False,
    c_virtual: bool = False,
) -> tuple[dict, dict]:
    """A MatmulAttributes node: ``a`` on uid 1, ``b`` on 2, ``c`` on 3.

    ``c`` is what the frontend infers: the larger of each batch dim, then
    ``M`` and ``N``. Ports as utilities/json/MatmulAttributes.hpp writes them.
    """
    batch_b = batch_a if batch_b is None else batch_b

    def tensor(uid: int, dims: list[int], dt: str, lay: str, virtual: bool) -> dict:
        return {
            "uid": uid,
            "dims": dims,
            "strides": _matrix_strides(dims, lay),
            "data_type": dt,
            "virtual": virtual,
        }

    batch_c = [max(x, y) for x, y in zip(batch_a, batch_b)]
    tensors = {
        1: tensor(1, [*batch_a, m, k], dtype, a_layout, a_virtual),
        2: tensor(2, [*batch_b, k, n], b_dtype or dtype, b_layout, b_virtual),
        3: tensor(3, [*batch_c, m, n], c_dtype or dtype, c_layout, c_virtual),
    }
    node = {
        "type": "MatmulAttributes",
        "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
        "outputs": {"c_tensor_uid": 3},
    }
    return node, tensors


class TestMatmulExtract:
    PLAIN = {
        "dtype:fp32",
        "batch:single",
        "a_layout:row_major",
        "b_layout:row_major",
        "c_layout:row_major",
    }

    def _extract(self, **kwargs) -> set[str]:
        return matmul.extract(*_matmul(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            matmul.extract(node, tensors)

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _matmul()
        written = set(batchnorm.bound(node)) | set(node) - {"type", "inputs", "outputs"}
        assert written == table_fields(matmul.SCHEMA_FILE, matmul.SCHEMA_TABLE)

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"batch_a": (), "dtype": "half", "b_dtype": "float", "c_dtype": "float"},
            {"batch_a": (1, 1), "batch_b": (2, 3), "dtype": "bfloat16"},
            {"batch_a": (3,), "batch_b": (1,), "a_layout": "col_major"},
            {"b_layout": "col_major", "c_layout": "col_major"},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {feature.id for feature in matmul.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"b_dtype": "float"}, {"precision:mixed_inputs"}),
            ({"c_dtype": "float"}, {"precision:mixed_output"}),
            (
                {"b_dtype": "float", "c_dtype": "float"},
                {"precision:mixed_inputs", "precision:mixed_output"},
            ),
            # b matches c, not a: still both.
            (
                {"b_dtype": "bfloat16", "c_dtype": "bfloat16"},
                {"precision:mixed_inputs", "precision:mixed_output"},
            ),
        ],
    )
    def test_each_precision_role_is_its_own_feature(
        self, kwargs: dict, expected: set[str]
    ) -> None:
        features = self._extract(dtype="half", **kwargs)
        assert features == self.PLAIN - {"dtype:fp32"} | {"dtype:fp16"} | expected

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"b_dtype": "float", "b_virtual": True},
            {"b_dtype": "float", "a_virtual": True},
            {"c_dtype": "float", "a_virtual": True},
        ],
    )
    def test_a_virtual_tensor_is_not_mixed(self, kwargs: dict) -> None:
        features = self._extract(dtype="half", **kwargs)
        assert not {f for f in features if f.startswith("precision:")}

    def test_a_virtual_c_is_not_mixed_and_has_no_layout(self) -> None:
        # The fused recipe: fp16 a and b into a virtual fp32 c, then a bias.
        features = self._extract(dtype="half", c_dtype="float", c_virtual=True)
        assert features == self.PLAIN - {"dtype:fp32", "c_layout:row_major"} | {
            "dtype:fp16"
        }

    def test_the_block_scale_recipe_is_not_mixed(self) -> None:
        # Virtual fp32 a and b out of two dequantizes, a fp16 c.
        features = self._extract(
            c_dtype="half", a_virtual=True, b_virtual=True, b_layout="col_major"
        )
        assert features == self.PLAIN - {"b_layout:row_major"} | {"b_layout:col_major"}

    @pytest.mark.parametrize(
        "batch, expected",
        [
            ((), "batch:none"),
            ((3,), "batch:single"),
            ((1,), "batch:single"),
            ((2, 3), "batch:multi"),
            ((1, 1), "batch:multi"),
            ((2, 1, 3), "batch:multi"),
        ],
    )
    def test_batch(self, batch: tuple[int, ...], expected: str) -> None:
        features = self._extract(batch_a=batch)
        assert features == self.PLAIN - {"batch:single"} | {expected}

    @pytest.mark.parametrize(
        "batch_a, batch_b, expected",
        [
            ((3,), (3,), set()),
            ((1,), (1,), set()),
            ((2, 3), (2, 3), set()),
            # The bundles' recipes.
            ((1,), (3,), {"broadcast:a"}),
            ((3,), (1,), {"broadcast:b"}),
            ((1, 1), (2, 3), {"broadcast:a"}),
            ((2, 3), (1, 1), {"broadcast:b"}),
            # One dim each way; a dim that divides rather than is 1.
            ((3, 1), (1, 3), {"broadcast:a", "broadcast:b"}),
            ((2,), (4,), {"broadcast:a"}),
            ((2, 3), (2, 1), {"broadcast:b"}),
        ],
    )
    def test_broadcast(
        self, batch_a: tuple[int, ...], batch_b: tuple[int, ...], expected: set[str]
    ) -> None:
        features = self._extract(batch_a=batch_a, batch_b=batch_b)
        assert {f for f in features if f.startswith("broadcast:")} == expected

    @pytest.mark.parametrize("port", ["a", "b", "c"])
    @pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
    def test_col_major(self, port: str, batch: tuple[int, ...]) -> None:
        features = self._extract(batch_a=batch, **{f"{port}_layout": "col_major"})
        layouts = {f for f in features if f.startswith(f"{port}_layout:")}
        assert layouts == {f"{port}_layout:col_major"}

    def test_a_matrix_with_a_dim_of_one_is_row_major(self) -> None:
        # Both last strides are 1; getTrans takes the last first.
        node, tensors = _matmul(m=1, a_layout="col_major")
        assert tensors[1]["strides"][-2:] == [1, 1]
        assert "a_layout:row_major" in matmul.extract(node, tensors)

    @pytest.mark.parametrize("uid", [1, 2, 3])
    def test_no_unit_stride_raises(self, uid: int) -> None:
        node, tensors = _matmul()
        tensors[uid]["strides"][-1] = 2
        tensors[uid]["strides"][-2] *= 2
        self._raises(node, tensors, "no unit stride")

    def test_the_layout_of_a_virtual_c_is_not_read(self) -> None:
        node, tensors = _matmul(c_virtual=True)
        tensors[3]["strides"][-1] = 2
        tensors[3]["strides"][-2] *= 2
        assert "c_layout:row_major" not in matmul.extract(node, tensors)

    @pytest.mark.parametrize(
        "a_dims, b_dims",
        [
            ([16], [16]),  # one rank, below 2
            ([16], [16, 8]),
            ([16, 32], [3, 32, 8]),
            ([3, 16, 32], [32, 8]),
        ],
    )
    def test_ranks_the_api_rejects_raise(self, a_dims: list, b_dims: list) -> None:
        node, tensors = _matmul()
        tensors[1]["dims"], tensors[2]["dims"] = a_dims, b_dims
        self._raises(node, tensors, "not one rank")

    def test_k_that_differs_raises(self) -> None:
        node, tensors = _matmul()
        tensors[2]["dims"] = [3, 16, 128]
        self._raises(node, tensors, "K differs")

    @pytest.mark.parametrize("batch_a, batch_b", [((2,), (3,)), ((2, 4), (2, 6))])
    def test_batch_dims_that_do_not_divide_raise(
        self, batch_a: tuple[int, ...], batch_b: tuple[int, ...]
    ) -> None:
        self._raises(*_matmul(batch_a=batch_a, batch_b=batch_b), "do not broadcast")

    @pytest.mark.parametrize("uid", [1, 2])
    def test_a_dim_below_one_raises(self, uid: int) -> None:
        # Before the divisibility check, which would divide by it.
        node, tensors = _matmul()
        tensors[uid]["dims"][0] = 0
        self._raises(node, tensors, "dim below 1")

    @pytest.mark.parametrize(
        "c_dims",
        [
            [16, 128],  # the rank of c
            [1, 16, 128],  # a batch dim
            [3, 32, 128],  # M
            [3, 16, 32],  # N
            [3, 128, 16],  # transposed
        ],
    )
    def test_c_dims_the_api_would_not_infer_raise(self, c_dims: list) -> None:
        node, tensors = _matmul()
        tensors[3]["dims"] = c_dims
        self._raises(node, tensors, "c_tensor_uid dims")

    @pytest.mark.parametrize("uid", [1, 2, 3])
    @pytest.mark.parametrize("dtype", ["fp8_e4m3", None])
    def test_uncatalogued_dtype_raises(self, uid: int, dtype: str | None) -> None:
        node, tensors = _matmul()
        tensors[uid]["data_type"] = dtype
        self._raises(node, tensors, "is not catalogued")

    def test_uid_zero_is_a_binding(self) -> None:
        node, tensors = _matmul()
        tensors[0] = {**tensors.pop(1), "uid": 0}
        node["inputs"]["a_tensor_uid"] = 0
        assert matmul.extract(node, tensors) == self.PLAIN

    @pytest.mark.parametrize(
        "side, port",
        [
            ("inputs", "a_tensor_uid"),
            ("inputs", "b_tensor_uid"),
            ("outputs", "c_tensor_uid"),
        ],
    )
    def test_unbound_required_port_raises(self, side: str, port: str) -> None:
        node, tensors = _matmul()
        node[side][port] = None
        self._raises(node, tensors, f"required port {port}")


# --------------------------------------------------------------------------
# BlockScaleDequantizeAttributes
# --------------------------------------------------------------------------


def _bsdq(
    *,
    dims: tuple[int, ...] = (32, 128),
    block_size: tuple[int, ...] = (32,),
    scale_dims: list[int] | None = None,
    dtype: str = "fp8_e4m3",
    scale_dtype: str = "fp8_e8m0",
    negative: bool | None = False,
) -> tuple[dict, dict]:
    """A BlockScaleDequantizeAttributes node: ``x`` on uid 1, ``scale`` on 2
    and a virtual fp32 ``y`` on 0, as the bundles' MX graphs bind it.

    ``scale`` is ``ceil(x / block)`` on each dim unless ``scale_dims`` says
    otherwise; ``negative=None`` leaves ``is_negative_scale`` out. Ports as
    utilities/json/BlockScaleDequantizeAttributes.hpp writes them.
    """
    dims = list(dims)
    if scale_dims is None:
        padded = (1,) * (len(dims) - len(block_size)) + tuple(block_size)
        scale_dims = [math.ceil(d / b) for d, b in zip(dims, padded)]

    def tensor(uid: int, shape: list[int], dt: str, virtual: bool) -> dict:
        strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
        return {
            "uid": uid,
            "dims": shape,
            "strides": strides,
            "data_type": dt,
            "virtual": virtual,
        }

    tensors = {
        1: tensor(1, dims, dtype, False),
        2: tensor(2, scale_dims, scale_dtype, False),
        0: tensor(0, list(dims), "float", True),
    }
    node = {
        "type": "BlockScaleDequantizeAttributes",
        "inputs": {"x_tensor_uid": 1, "scale_tensor_uid": 2},
        "outputs": {"y_tensor_uid": 0},
        "block_size": list(block_size),
    }
    if negative is not None:
        node["is_negative_scale"] = negative
    return node, tensors


class TestBlockScaleDequantizeExtract:
    PLAIN = {
        "dtype:fp8e4m3",
        "scale_dtype:fp8e8m0",
        "block:last_dim",
        "block_length:32",
        "fit:exact",
        "scale:multiply",
    }

    def _extract(self, **kwargs) -> set[str]:
        return block_scale_dequantize.extract(*_bsdq(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            block_scale_dequantize.extract(node, tensors)

    def _group(self, features: set[str], group: str) -> set[str]:
        return {f for f in features if f.startswith(f"{group}:")}

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _bsdq()
        written = set(batchnorm.bound(node)) | set(node) - {"type", "inputs", "outputs"}
        assert written == table_fields(
            block_scale_dequantize.SCHEMA_FILE, block_scale_dequantize.SCHEMA_TABLE
        )

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"dtype": "fp8_e5m2", "block_size": (32, 1)},
            {"dtype": "fp6_e2m3", "block_size": (16, 16)},
            {"dtype": "fp6_e3m2", "dims": (30, 100)},
            {"dtype": "fp4_e2m1", "block_size": (1,)},
            {"dtype": "float", "scale_dtype": "float", "negative": True},
            {"dtype": "half"},
            {"dtype": "bfloat16"},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {f.id for f in block_scale_dequantize.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "dtype, expected",
        [
            ("fp8_e4m3", "dtype:fp8e4m3"),
            ("fp8_e5m2", "dtype:fp8e5m2"),
            ("fp6_e2m3", "dtype:fp6e2m3"),
            ("fp6_e3m2", "dtype:fp6e3m2"),
            ("fp4_e2m1", "dtype:fp4e2m1"),
            ("float", "dtype:fp32"),
            ("half", "dtype:fp16"),
            ("bfloat16", "dtype:bf16"),
        ],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp8e4m3"} | {expected}

    @pytest.mark.parametrize(
        "scale_dtype, expected",
        [("fp8_e8m0", "scale_dtype:fp8e8m0"), ("float", "scale_dtype:fp32")],
    )
    def test_scale_dtype(self, scale_dtype: str, expected: str) -> None:
        features = self._extract(scale_dtype=scale_dtype)
        assert self._group(features, "scale_dtype") == {expected}

    @pytest.mark.parametrize(
        "dtype",
        # FNUZ fp8 is not OCP, which MX requires.
        ["fp8_e4m3_fnuz", "fp8_e5m2_fnuz", "fp8_e8m0", "int8", None],
    )
    def test_uncatalogued_x_dtype_raises(self, dtype: str | None) -> None:
        node, tensors = _bsdq()
        tensors[1]["data_type"] = dtype
        self._raises(node, tensors, "x_tensor_uid dtype .* is not catalogued")

    @pytest.mark.parametrize("dtype", ["half", "fp8_e4m3", ["float"], None])
    def test_uncatalogued_scale_dtype_raises(self, dtype) -> None:
        node, tensors = _bsdq()
        tensors[2]["data_type"] = dtype
        self._raises(node, tensors, "scale_tensor_uid dtype .* is not catalogued")

    @pytest.mark.parametrize(
        "dims, block_size, expected",
        [
            # The bundles' recipes: the K dim of a, then of b.
            ((32, 128), (32,), "block:last_dim"),
            ((128, 128), (32, 1), "block:other_dim"),
            ((1, 1, 32, 128), (32,), "block:last_dim"),
            ((1, 1, 128, 32), (32, 1), "block:other_dim"),
            # Right-aligned: a leading entry reaches the first dim.
            ((4, 8, 16), (2, 1, 1), "block:other_dim"),
            ((32, 128), (1,), "block:element"),
            ((32, 128), (1, 1), "block:element"),
            ((32, 128), (32, 32), "block:tile"),
            ((2, 32, 128), (2, 1, 32), "block:tile"),
        ],
    )
    def test_block(
        self, dims: tuple[int, ...], block_size: tuple[int, ...], expected: str
    ) -> None:
        features = self._extract(dims=dims, block_size=block_size)
        assert self._group(features, "block") == {expected}

    @pytest.mark.parametrize(
        "dims, block_size, scale_dims",
        [
            ((32, 128), (32,), [32, 4]),
            ((128, 128), (32, 1), [4, 128]),
            ((1, 128, 32), (32, 1), [1, 4, 32]),
            ((30, 100), (32,), [30, 4]),
            ((16, 128), (32, 1), [1, 128]),
        ],
    )
    def test_scale_is_one_per_block(
        self, dims: tuple[int, ...], block_size: tuple[int, ...], scale_dims: list
    ) -> None:
        # Literal shapes, not the builder's ceil: the bundles', and ragged.
        node, tensors = _bsdq(dims=dims, block_size=block_size, scale_dims=scale_dims)
        assert block_scale_dequantize.extract(node, tensors)

    @pytest.mark.parametrize(
        "block_size, expected",
        [
            ((32,), {"block_length:32"}),
            ((32, 1), {"block_length:32"}),
            ((32, 32), {"block_length:32"}),
            ((16,), {"block_length:other"}),
            ((16, 1), {"block_length:other"}),
            ((32, 16), {"block_length:other"}),
            ((1,), set()),
        ],
    )
    def test_block_length(self, block_size: tuple[int, ...], expected: set) -> None:
        features = self._extract(dims=(64, 128), block_size=block_size)
        assert self._group(features, "block_length") == expected

    @pytest.mark.parametrize(
        "dims, block_size, expected",
        [
            ((32, 128), (32,), {"fit:exact"}),
            ((128, 128), (32, 1), {"fit:exact"}),
            ((30, 100), (32,), {"fit:ragged"}),
            # A block longer than its dim.
            ((16, 128), (32, 1), {"fit:ragged"}),
            # One blocked dim ragged, the other whole; either way round.
            ((64, 40), (32, 32), {"fit:ragged"}),
            ((40, 64), (32, 32), {"fit:ragged"}),
            # An unblocked dim is always whole.
            ((30, 128), (32,), {"fit:exact"}),
            ((30, 100), (1,), set()),
        ],
    )
    def test_fit(
        self, dims: tuple[int, ...], block_size: tuple[int, ...], expected: set
    ) -> None:
        features = self._extract(dims=dims, block_size=block_size)
        assert self._group(features, "fit") == expected

    @pytest.mark.parametrize(
        "negative, expected",
        [
            (False, "scale:multiply"),
            (None, "scale:multiply"),  # left out: the reader's false
            (True, "scale:negative_exponent"),
        ],
    )
    def test_scale(self, negative: bool | None, expected: str) -> None:
        assert self._group(self._extract(negative=negative), "scale") == {expected}

    @pytest.mark.parametrize("value", [1, 0, "false", None])
    def test_a_negative_scale_that_is_not_a_bool_raises(self, value) -> None:
        node, tensors = _bsdq()
        node["is_negative_scale"] = value
        self._raises(node, tensors, "is not a bool")

    @pytest.mark.parametrize("virtual", [False, None])
    def test_a_y_that_is_not_virtual_raises(self, virtual: bool | None) -> None:
        node, tensors = _bsdq()
        tensors[0]["virtual"] = virtual
        self._raises(node, tensors, "y_tensor_uid is not virtual")

    @pytest.mark.parametrize("dims", [[], [32, 0], [0, 128]])
    def test_x_dims_the_api_rejects_raise(self, dims: list) -> None:
        node, tensors = _bsdq()
        tensors[1]["dims"] = tensors[0]["dims"] = dims
        self._raises(node, tensors, "not rank >= 1")

    @pytest.mark.parametrize("y_dims", [[32, 64], [128, 32], [1, 32, 128]])
    def test_y_dims_that_are_not_x_dims_raise(self, y_dims: list) -> None:
        node, tensors = _bsdq()
        tensors[0]["dims"] = y_dims
        self._raises(node, tensors, "y_tensor_uid dims")

    @pytest.mark.parametrize("block_size", [[], None, 32, [32, 1, 1]])
    def test_block_size_entries_the_api_rejects_raise(self, block_size) -> None:
        node, tensors = _bsdq()
        node["block_size"] = block_size
        self._raises(node, tensors, "not 1 to 2 entries")

    def test_a_missing_block_size_raises(self) -> None:
        # The reader takes it as empty, which the API rejects.
        node, tensors = _bsdq()
        del node["block_size"]
        self._raises(node, tensors, "not 1 to 2 entries")

    @pytest.mark.parametrize("block_size", [[0], [32, -1], [True], [32.0], ["32"]])
    def test_block_size_values_the_api_rejects_raise(self, block_size: list) -> None:
        node, tensors = _bsdq()
        node["block_size"] = block_size
        self._raises(node, tensors, "not an int above 0")

    @pytest.mark.parametrize(
        "scale_dims",
        [
            [32, 5],  # ceil + 1
            [32, 3],  # ceil - 1
            [4, 32],  # transposed
            [4],  # the blocked dim only
            [1, 32, 4],  # a rank above x
            [32, 128],  # one per element
        ],
    )
    def test_scale_dims_that_are_not_one_per_block_raise(self, scale_dims) -> None:
        self._raises(*_bsdq(scale_dims=scale_dims), "scale_tensor_uid dims")

    def test_a_ragged_scale_rounds_up(self) -> None:
        # [30, 100] in blocks of 32 on the last dim: 4 blocks, not 3.
        node, tensors = _bsdq(dims=(30, 100), scale_dims=[30, 3])
        self._raises(node, tensors, "scale_tensor_uid dims")

    def test_uid_zero_is_a_binding(self) -> None:
        node, tensors = _bsdq()
        assert node["outputs"]["y_tensor_uid"] == 0
        assert block_scale_dequantize.extract(node, tensors) == self.PLAIN

    @pytest.mark.parametrize(
        "side, port",
        [
            ("inputs", "x_tensor_uid"),
            ("inputs", "scale_tensor_uid"),
            ("outputs", "y_tensor_uid"),
        ],
    )
    def test_unbound_required_port_raises(self, side: str, port: str) -> None:
        node, tensors = _bsdq()
        node[side][port] = None
        self._raises(node, tensors, f"required port {port}")


def _bsq(
    *,
    dims: tuple[int, ...] = (32, 128),
    block_size: int | None = 32,
    axis: int | None = None,
    scale_dims: list[int] | None = None,
    dtype: str = "float",
    out_dtype: str = "fp8_e4m3",
    scale_dtype: str = "fp8_e8m0",
    transpose: bool | None = False,
) -> tuple[dict, dict]:
    """A BlockScaleQuantizeAttributes node: ``x`` on uid 1, ``y`` on 2 and
    ``scale`` on 3, as the CPU reference executor's float -> MX path takes it.

    ``scale`` is ``x`` with the blocked dim divided by ``block_size`` unless
    ``scale_dims`` says otherwise; ``block_size=None``, ``axis=None`` and
    ``transpose=None`` leave the key out. Ports as
    utilities/json/BlockScaleQuantizeAttributes.hpp writes them.
    """
    dims = list(dims)
    if scale_dims is None:
        scale_dims = list(dims)
        blocked = len(dims) - 1 if axis is None else axis
        scale_dims[blocked] //= block_size

    def tensor(uid: int, shape: list[int], dt: str) -> dict:
        strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
        return {
            "uid": uid,
            "dims": shape,
            "strides": strides,
            "data_type": dt,
            "virtual": False,
        }

    tensors = {
        1: tensor(1, dims, dtype),
        2: tensor(2, list(dims), out_dtype),
        3: tensor(3, scale_dims, scale_dtype),
    }
    node = {
        "type": "BlockScaleQuantizeAttributes",
        "inputs": {"x_tensor_uid": 1},
        "outputs": {"y_tensor_uid": 2, "scale_tensor_uid": 3},
    }
    for key, value in (
        ("block_size", block_size),
        ("axis", axis),
        ("transpose", transpose),
    ):
        if value is not None:
            node[key] = value
    return node, tensors


class TestBlockScaleQuantizeExtract:
    PLAIN = {
        "dtype:fp32",
        "out_dtype:fp8e4m3",
        "scale_dtype:fp8e8m0",
        "axis:last_dim",
        "block_length:32",
        "transpose:off",
    }

    def _extract(self, **kwargs) -> set[str]:
        return block_scale_quantize.extract(*_bsq(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            block_scale_quantize.extract(node, tensors)

    def _group(self, features: set[str], group: str) -> set[str]:
        return {f for f in features if f.startswith(f"{group}:")}

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _bsq(axis=1)
        written = set(batchnorm.bound(node)) | set(node) - {"type", "inputs", "outputs"}
        assert written == table_fields(
            block_scale_quantize.SCHEMA_FILE, block_scale_quantize.SCHEMA_TABLE
        )

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"dtype": "half", "out_dtype": "fp8_e5m2", "axis": 0},
            {"dtype": "bfloat16", "out_dtype": "fp6_e2m3", "transpose": True},
            {"out_dtype": "fp6_e3m2", "block_size": 16},
            {"out_dtype": "fp4_e2m1"},
            {"out_dtype": "float", "scale_dtype": "float"},
            {"out_dtype": "half", "scale_dtype": "float"},
            {"out_dtype": "bfloat16", "scale_dtype": "float"},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {f.id for f in block_scale_quantize.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "out_dtype, expected",
        [
            ("fp8_e4m3", "out_dtype:fp8e4m3"),
            ("fp8_e5m2", "out_dtype:fp8e5m2"),
            ("fp6_e2m3", "out_dtype:fp6e2m3"),
            ("fp6_e3m2", "out_dtype:fp6e3m2"),
            ("fp4_e2m1", "out_dtype:fp4e2m1"),
            ("float", "out_dtype:fp32"),
            ("half", "out_dtype:fp16"),
            ("bfloat16", "out_dtype:bf16"),
        ],
    )
    def test_out_dtype(self, out_dtype: str, expected: str) -> None:
        features = self._extract(out_dtype=out_dtype)
        assert features == self.PLAIN - {"out_dtype:fp8e4m3"} | {expected}

    @pytest.mark.parametrize(
        "scale_dtype, expected",
        [("fp8_e8m0", "scale_dtype:fp8e8m0"), ("float", "scale_dtype:fp32")],
    )
    def test_scale_dtype(self, scale_dtype: str, expected: str) -> None:
        features = self._extract(scale_dtype=scale_dtype)
        assert self._group(features, "scale_dtype") == {expected}

    @pytest.mark.parametrize(
        "dtype", ["fp8_e4m3", "fp4_e2m1", "int8", "fp32", ["float"], None]
    )
    def test_uncatalogued_x_dtype_raises(self, dtype) -> None:
        node, tensors = _bsq()
        tensors[1]["data_type"] = dtype
        self._raises(node, tensors, "x_tensor_uid dtype .* is not catalogued")

    @pytest.mark.parametrize(
        "dtype",
        # FNUZ fp8 is not OCP, which MX requires.
        ["fp8_e4m3_fnuz", "fp8_e5m2_fnuz", "fp8_e8m0", "int8", None],
    )
    def test_uncatalogued_y_dtype_raises(self, dtype: str | None) -> None:
        node, tensors = _bsq()
        tensors[2]["data_type"] = dtype
        self._raises(node, tensors, "y_tensor_uid dtype .* is not catalogued")

    @pytest.mark.parametrize("dtype", ["half", "bfloat16", "fp8_e4m3", None])
    def test_uncatalogued_scale_dtype_raises(self, dtype: str | None) -> None:
        node, tensors = _bsq()
        tensors[3]["data_type"] = dtype
        self._raises(node, tensors, "scale_tensor_uid dtype .* is not catalogued")

    @pytest.mark.parametrize(
        "dims, axis, expected",
        [
            ((32, 128), None, "axis:last_dim"),
            ((32, 128), 1, "axis:last_dim"),
            ((128,), None, "axis:last_dim"),
            ((128,), 0, "axis:last_dim"),
            ((128, 64), 0, "axis:other_dim"),
            ((2, 4, 64, 32), 2, "axis:other_dim"),
            ((2, 4, 32, 64), 3, "axis:last_dim"),
        ],
    )
    def test_axis(self, dims: tuple[int, ...], axis: int | None, expected: str) -> None:
        features = self._extract(dims=dims, axis=axis)
        assert self._group(features, "axis") == {expected}

    @pytest.mark.parametrize("axis", [-1, 2, True, 1.0, "1", None])
    def test_an_axis_the_api_rejects_raises(self, axis) -> None:
        # Present: None is a null the serializer never writes.
        node, tensors = _bsq()
        node["axis"] = axis
        self._raises(node, tensors, "axis .* is not a dim of rank 2")

    @pytest.mark.parametrize(
        "dims, axis, scale_dims",
        [
            ((32, 128), None, [32, 4]),
            ((32, 128), 1, [32, 4]),
            ((128, 64), 0, [4, 64]),
            ((2, 64, 32), 1, [2, 2, 32]),
        ],
    )
    def test_scale_is_one_per_block_on_the_axis(
        self, dims: tuple[int, ...], axis: int | None, scale_dims: list
    ) -> None:
        # Literal shapes, not the builder's division.
        node, tensors = _bsq(dims=dims, axis=axis, scale_dims=scale_dims)
        assert block_scale_quantize.extract(node, tensors)

    @pytest.mark.parametrize(
        "block_size, expected",
        [
            (32, "block_length:32"),
            (16, "block_length:other"),
            (64, "block_length:other"),
            (128, "block_length:other"),  # the whole dim: one scale a row
            (1, "block_length:other"),  # a scale per element
        ],
    )
    def test_block_length(self, block_size: int, expected: str) -> None:
        features = self._extract(block_size=block_size)
        assert self._group(features, "block_length") == {expected}

    @pytest.mark.parametrize("block_size", [0, -32, True, 32.0, "32", [32], None])
    def test_a_block_size_the_api_rejects_raises(self, block_size) -> None:
        node, tensors = _bsq()
        node["block_size"] = block_size
        self._raises(node, tensors, "block_size .* is not an int above 0")

    def test_a_missing_block_size_raises(self) -> None:
        # The serializer leaves a 0 out, which the API rejects.
        node, tensors = _bsq()
        del node["block_size"]
        self._raises(node, tensors, "block_size None is not an int above 0")

    @pytest.mark.parametrize(
        "dims, axis",
        [((30, 100), None), ((32, 100), 1), ((30, 128), 0), ((32, 16), None)],
    )
    def test_a_block_that_does_not_divide_its_dim_raises(
        self, dims: tuple[int, ...], axis: int | None
    ) -> None:
        node, tensors = _bsq(dims=dims, axis=axis, scale_dims=[1, 1])
        self._raises(node, tensors, "does not divide")

    def test_only_the_blocked_dim_must_divide(self) -> None:
        assert self._extract(dims=(30, 128)) == self.PLAIN

    @pytest.mark.parametrize(
        "scale_dims",
        [
            [32, 5],
            [32, 3],
            [4, 128],  # blocked on the wrong dim
            [1, 4],
            [4],  # the blocked dim only
            [1, 32, 4],  # a rank above x
            [32, 128],  # one per element
        ],
    )
    def test_scale_dims_that_are_not_one_per_block_raise(self, scale_dims) -> None:
        self._raises(*_bsq(scale_dims=scale_dims), "scale_tensor_uid dims")

    @pytest.mark.parametrize(
        "transpose, expected",
        [
            (False, "transpose:off"),
            (None, "transpose:off"),  # left out: the reader's false
            (True, "transpose:on"),
        ],
    )
    def test_transpose(self, transpose: bool | None, expected: str) -> None:
        features = self._extract(transpose=transpose)
        assert self._group(features, "transpose") == {expected}

    @pytest.mark.parametrize("value", [1, 0, "true", None])
    def test_a_transpose_that_is_not_a_bool_raises(self, value) -> None:
        node, tensors = _bsq()
        node["transpose"] = value
        self._raises(node, tensors, "transpose .* is not a bool")

    @pytest.mark.parametrize("dims", [[], [32, 0], [0, 128]])
    def test_x_dims_the_api_rejects_raise(self, dims: list) -> None:
        node, tensors = _bsq()
        tensors[1]["dims"] = tensors[2]["dims"] = dims
        self._raises(node, tensors, "not rank >= 1")

    @pytest.mark.parametrize("y_dims", [[32, 64], [128, 32], [1, 32, 128]])
    def test_y_dims_that_are_not_x_dims_raise(self, y_dims: list) -> None:
        node, tensors = _bsq()
        tensors[2]["dims"] = y_dims
        self._raises(node, tensors, "y_tensor_uid dims")

    def test_uid_zero_is_a_binding(self) -> None:
        node, tensors = _bsq()
        tensors[0] = {**tensors.pop(3), "uid": 0}
        node["outputs"]["scale_tensor_uid"] = 0
        assert block_scale_quantize.extract(node, tensors) == self.PLAIN

    @pytest.mark.parametrize(
        "side, port",
        [
            ("inputs", "x_tensor_uid"),
            ("outputs", "y_tensor_uid"),
            ("outputs", "scale_tensor_uid"),
        ],
    )
    def test_unbound_required_port_raises(self, side: str, port: str) -> None:
        node, tensors = _bsq()
        node[side][port] = None
        self._raises(node, tensors, f"required port {port}")


def _red(
    *,
    mode: object = "add",
    dims: tuple[int, ...] = (2, 8, 4, 4),
    y_dims: tuple[int, ...] = (2, 8, 1, 1),
    dtype: str = "float",
    y_dtype: str | None = None,
    layout: str = "nchw",
    y_layout: str | None = None,
    deterministic: bool | None = False,
) -> tuple[dict, dict]:
    """A ReductionAttributes node: ``x`` on uid 0 and ``y`` on uid 1, as the
    quick bundle binds them, with packed ``layout`` strides.

    ``y`` takes ``x``'s dtype and layout unless told otherwise;
    ``deterministic=None`` leaves the key out. Ports at the top of the node,
    as utilities/json/ReductionAttributes.hpp writes them.
    """

    def tensor(uid: int, shape: tuple[int, ...], dt: str, lay: str) -> dict:
        return {
            "uid": uid,
            "dims": list(shape),
            "strides": _layout_strides(list(shape), lay),
            "data_type": dt,
            "virtual": False,
        }

    tensors = {
        0: tensor(0, dims, dtype, layout),
        1: tensor(1, y_dims, y_dtype or dtype, y_layout or layout),
    }
    node = {
        "type": "ReductionAttributes",
        "compute_data_type": "float",
        "mode": mode,
        "in_tensor_uid": 0,
        "out_tensor_uid": 1,
    }
    if deterministic is not None:
        node["is_deterministic"] = deterministic
    return node, tensors


def _serialized_reduction_modes() -> dict[str, str]:
    """ReductionMode enumerator -> the name the serializer writes for it."""
    path = (
        registry.REPO_ROOT
        / "projects/hipdnn/flatbuffers_sdk/include/hipdnn_flatbuffers_sdk"
        / "utilities/json/ReductionAttributes.hpp"
    )
    # Missing, the mode check would pass vacuously, as for the schemas.
    assert path.is_file(), f"serializer not found: {path}"
    header = path.read_text(encoding="utf-8")
    pairs = re.findall(r'\{ReductionMode::(\w+),\s*"(\w+)"\}', header)
    assert pairs, "the serializer's ReductionMode names were not found"
    return dict(pairs)


class TestReductionExtract:
    PLAIN = {
        "mode:add",
        "dtype:fp32",
        "dims:4",
        "reduce:trailing",
        "layout:nchw",
        "deterministic:off",
    }

    def _extract(self, **kwargs) -> set[str]:
        return reduction.extract(*_red(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            reduction.extract(node, tensors)

    def _group(self, features: set[str], group: str) -> set[str]:
        return {f for f in features if f.startswith(f"{group}:")}

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _red()
        written = set(node) - {"type", "compute_data_type"}
        assert written == table_fields(reduction.SCHEMA_FILE, reduction.SCHEMA_TABLE)

    def test_the_serializer_names_every_schema_mode(self) -> None:
        names = _serialized_reduction_modes()
        assert set(names) == _schema_enum(reduction.SCHEMA_FILE, "ReductionMode")
        assert set(reduction.MODES) == set(names.values()) - {"not_set"}

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"mode": "mul", "dtype": "half", "y_dtype": "float"},
            {"mode": "min", "dtype": "bfloat16", "layout": "nhwc"},
            {"mode": "max", "dims": (4,), "y_dims": (1,)},
            {"mode": "amax", "dims": (4, 8), "y_dims": (1, 8)},
            {"mode": "avg", "dims": (2, 8, 4), "y_dims": (2, 1, 4)},
            {"mode": "norm1", "dims": (2, 8, 4, 4, 4), "y_dims": (1, 1, 1, 1, 1)},
            {"mode": "norm2", "deterministic": True},
            {"mode": "mul_no_zeros"},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {f.id for f in reduction.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    def test_uid_zero_is_a_binding(self) -> None:
        node, tensors = _red()
        assert node["in_tensor_uid"] == 0
        assert reduction.extract(node, tensors) == self.PLAIN

    @pytest.mark.parametrize("mode", reduction.MODES)
    def test_mode(self, mode: str) -> None:
        assert self._group(self._extract(mode=mode), "mode") == {f"mode:{mode}"}

    @pytest.mark.parametrize(
        "mode", ["not_set", "ADD", "min_op", "sum", "", 1, ["add"], None]
    )
    def test_a_mode_that_is_not_catalogued_raises(self, mode) -> None:
        self._raises(*_red(mode=mode), "mode .* is not catalogued")

    def test_a_missing_mode_raises(self) -> None:
        node, tensors = _red()
        del node["mode"]
        self._raises(node, tensors, "mode None is not catalogued")

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "dtype, y_dtype", [("half", "float"), ("bfloat16", "float")]
    )
    def test_a_y_of_another_dtype_is_mixed_output(
        self, dtype: str, y_dtype: str
    ) -> None:
        features = self._extract(dtype=dtype, y_dtype=y_dtype)
        assert "precision:mixed_output" in features
        assert self._group(features, "dtype") == {
            f"dtype:{layernorm.catalogued_dtype({'data_type': dtype}, 'x')}"
        }

    @pytest.mark.parametrize("port_uid", [0, 1])
    def test_a_virtual_tensor_is_not_mixed_output(self, port_uid: int) -> None:
        node, tensors = _red(dtype="half", y_dtype="float")
        tensors[port_uid]["virtual"] = True
        assert "precision:mixed_output" not in reduction.extract(node, tensors)

    @pytest.mark.parametrize(
        "dtype, y_dtype",
        # Pairs the integration tests' GPU reference takes, but no plan.
        [
            ("float", "half"),
            ("float", "bfloat16"),
            ("half", "bfloat16"),
            ("bfloat16", "half"),
        ],
    )
    @pytest.mark.parametrize("virtual", [False, True])
    def test_a_pair_the_cpu_reference_does_not_register_raises(
        self, dtype: str, y_dtype: str, virtual: bool
    ) -> None:
        node, tensors = _red(dtype=dtype, y_dtype=y_dtype)
        tensors[1]["virtual"] = virtual
        self._raises(node, tensors, "has no plan")

    @pytest.mark.parametrize("uid, port", [(0, "in_tensor_uid"), (1, "out_tensor_uid")])
    @pytest.mark.parametrize("dtype", ["int8", "fp8_e4m3", "double", None])
    def test_an_uncatalogued_dtype_raises(self, uid: int, port: str, dtype) -> None:
        node, tensors = _red()
        tensors[uid]["data_type"] = dtype
        self._raises(node, tensors, f"{port} dtype .* is not catalogued")

    @pytest.mark.parametrize(
        "dims, y_dims, expected",
        [
            ((4,), (1,), "dims:1"),
            ((4, 8), (4, 1), "dims:2"),
            ((2, 8, 4), (2, 8, 1), "dims:3"),
            ((2, 8, 4, 4), (2, 8, 1, 1), "dims:4"),
            ((2, 8, 4, 4, 4), (2, 8, 1, 1, 1), "dims:5"),
        ],
    )
    def test_dims(
        self, dims: tuple[int, ...], y_dims: tuple[int, ...], expected: str
    ) -> None:
        features = self._extract(dims=dims, y_dims=y_dims)
        assert self._group(features, "dims") == {expected}

    def test_a_rank_beyond_the_gpu_reference_raises(self) -> None:
        node, tensors = _red(dims=(2, 2, 2, 2, 2, 2), y_dims=(2, 2, 2, 2, 2, 1))
        self._raises(node, tensors, "rank 6 is not catalogued")

    @pytest.mark.parametrize(
        "dims, y_dims, expected",
        # The quick bundle's shapes first, then the rest.
        [
            ((1, 16, 8, 8), (1, 16, 1, 1), "reduce:trailing"),
            ((2, 3, 16, 8), (2, 3, 1, 1), "reduce:trailing"),
            ((4, 8, 4, 4), (1, 8, 4, 4), "reduce:leading"),
            ((2, 3, 4, 4), (1, 1, 1, 1), "reduce:all"),
            ((1, 1, 32, 32), (1, 1, 1, 1), "reduce:all"),
            ((4,), (1,), "reduce:all"),
            ((2, 8, 4, 4), (2, 1, 4, 4), "reduce:interleaved"),
            ((2, 8, 4, 4), (1, 8, 1, 1), "reduce:interleaved"),
            ((2, 8, 4, 4), (2, 8, 4, 1), "reduce:trailing"),
            ((2, 8, 4, 4), (1, 1, 4, 4), "reduce:leading"),
            # A dim of 1 in both is neither reduced nor kept.
            ((1, 8, 4, 4), (1, 8, 1, 1), "reduce:trailing"),
            ((2, 1, 4, 4), (2, 1, 1, 1), "reduce:trailing"),
            ((2, 8, 1, 4), (1, 8, 1, 4), "reduce:leading"),
            ((2, 1, 4), (1, 1, 1), "reduce:all"),
        ],
    )
    def test_reduce(
        self, dims: tuple[int, ...], y_dims: tuple[int, ...], expected: str
    ) -> None:
        features = self._extract(dims=dims, y_dims=y_dims)
        assert self._group(features, "reduce") == {expected}

    @pytest.mark.parametrize(
        "dims, y_dims, match",
        [
            ((2, 8, 4, 4), (2, 8, 4), "are not the rank of"),
            ((2, 8, 4), (2, 8, 1, 1), "are not the rank of"),
            ((2, 8, 4, 4), (2, 8, 2, 1), "or 1"),  # y not x and not 1
            ((2, 8, 4, 4), (2, 8, 8, 1), "or 1"),  # y above x
            ((2, 8, 4, 4), (2, 8, 4, 4), "reduce no dim"),
            ((1, 1, 1), (1, 1, 1), "reduce no dim"),
            ((), (), "not rank >= 1"),
            ((2, 0, 4), (1, 0, 4), "not rank >= 1"),
            ((2, 8, 4), (2, 8, 0), "not rank >= 1"),
        ],
    )
    def test_shapes_the_api_rejects_raise(
        self, dims: tuple[int, ...], y_dims: tuple[int, ...], match: str
    ) -> None:
        node, tensors = _red()
        tensors[0]["dims"], tensors[1]["dims"] = list(dims), list(y_dims)
        self._raises(node, tensors, match)

    @pytest.mark.parametrize(
        "dims, y_dims, layout, expected",
        [
            ((2, 8, 4, 4), (2, 8, 1, 1), "nchw", {"layout:nchw"}),
            ((2, 8, 4, 4), (2, 8, 1, 1), "nhwc", {"layout:nhwc"}),
            # x tells; y, with one live dim, does not.
            ((1, 16, 8, 8), (1, 16, 1, 1), "nhwc", {"layout:nhwc"}),
            # Neither tells: one live dim pair, or rank 2.
            ((1, 1, 32, 32), (1, 1, 1, 1), "nchw", set()),
            ((4, 8), (1, 8), "nchw", set()),
            ((2, 8, 4, 4, 4), (2, 8, 1, 1, 1), "nhwc", {"layout:nhwc"}),
        ],
    )
    def test_layout(
        self,
        dims: tuple[int, ...],
        y_dims: tuple[int, ...],
        layout: str,
        expected: set[str],
    ) -> None:
        features = self._extract(dims=dims, y_dims=y_dims, layout=layout)
        assert self._group(features, "layout") == expected

    def test_x_and_y_in_different_layouts_raise(self) -> None:
        node, tensors = _red(y_dims=(1, 8, 4, 4), layout="nchw", y_layout="nhwc")
        self._raises(node, tensors, "layouts of in_tensor_uid and out_tensor_uid")

    @pytest.mark.parametrize("uid", [0, 1])
    def test_strides_no_layout_describes_raise(self, uid: int) -> None:
        node, tensors = _red(y_dims=(2, 8, 4, 1))
        tensors[uid]["strides"] = list(reversed(tensors[uid]["strides"]))
        self._raises(node, tensors, "not a catalogued layout|overlap")

    @pytest.mark.parametrize(
        "deterministic, expected",
        [
            (False, "deterministic:off"),
            (None, "deterministic:off"),  # left out: the reader's false
            (True, "deterministic:on"),
        ],
    )
    def test_deterministic(self, deterministic: bool | None, expected: str) -> None:
        features = self._extract(deterministic=deterministic)
        assert self._group(features, "deterministic") == {expected}

    @pytest.mark.parametrize("value", [1, 0, "true", None])
    def test_a_deterministic_that_is_not_a_bool_raises(self, value) -> None:
        node, tensors = _red()
        node["is_deterministic"] = value
        self._raises(node, tensors, "is_deterministic .* is not a bool")

    @pytest.mark.parametrize("port", ["in_tensor_uid", "out_tensor_uid"])
    @pytest.mark.parametrize("unbound", ["null", "missing"])
    def test_unbound_required_port_raises(self, port: str, unbound: str) -> None:
        node, tensors = _red()
        if unbound == "null":
            node[port] = None
        else:
            del node[port]
        self._raises(node, tensors, f"required port {port}")

    def test_ports_under_inputs_and_outputs_are_not_read(self) -> None:
        # The serializer writes them at the top; a node shaped like the other
        # ops' is not a Reduction node.
        node, tensors = _red()
        node["inputs"] = {"in_tensor_uid": node.pop("in_tensor_uid")}
        node["outputs"] = {"out_tensor_uid": node.pop("out_tensor_uid")}
        self._raises(node, tensors, "required port in_tensor_uid")


def _rsf(
    *,
    mode: object = "maxpool",
    padding_mode: object = "neg_inf_pad",
    dims: tuple[int, ...] = (2, 3, 8, 8),
    window: tuple[int, ...] = (2, 2),
    stride: tuple[int, ...] = (2, 2),
    pre: tuple[int, ...] | None = None,
    post: tuple[int, ...] | None = None,
    dtype: str = "float",
    y_dtype: str | None = None,
    layout: str = "nchw",
    y_layout: str | None = None,
    index: bool = True,
    generate: bool | None = True,
) -> tuple[dict, dict]:
    """A ResampleFwdAttributes node: ``x`` on uid 0, ``y`` on uid 1 and, with
    ``index``, an int32 index on uid 2 of ``y``'s dims and strides, as the
    API infers it. Packed ``layout`` strides.

    Padding defaults to 0 in each spatial dim; ``y`` takes the dims the
    pooling gives, and ``x``'s dtype and layout unless told otherwise;
    ``generate=None`` leaves ``generate_index`` out. Ports at the top of the
    node, as utilities/json/ResampleFwdAttributes.hpp writes them.
    """
    spatial = len(dims) - 2
    pre = pre if pre is not None else (0,) * spatial
    post = post if post is not None else (0,) * spatial
    y_dims = [
        *dims[:2],
        *(
            (d + p + q - w) // s + 1
            for d, p, q, w, s in zip(dims[2:], pre, post, window, stride)
        ),
    ]

    def tensor(uid: int, shape: list[int], dt: str, lay: str) -> dict:
        return {
            "uid": uid,
            "dims": list(shape),
            "strides": _layout_strides(list(shape), lay),
            "data_type": dt,
            "virtual": False,
        }

    tensors = {
        0: tensor(0, list(dims), dtype, layout),
        1: tensor(1, y_dims, y_dtype or dtype, y_layout or layout),
    }
    node = {
        "type": "ResampleFwdAttributes",
        "compute_data_type": "float",
        "x_tensor_uid": 0,
        "y_tensor_uid": 1,
        "pre_padding": list(pre),
        "post_padding": list(post),
        "stride": list(stride),
        "window": list(window),
        "resample_mode": mode,
        "padding_mode": padding_mode,
    }
    if index:
        tensors[2] = {**tensors[1], "uid": 2, "data_type": "int32"}
        node["index_tensor_uid"] = 2
    if generate is not None:
        node["generate_index"] = generate
    return node, tensors


def _serialized_enum(name: str, header: str = "Common.hpp") -> dict[str, str]:
    """``name`` enumerator -> the name utilities/json/``header`` writes."""
    path = (
        registry.REPO_ROOT
        / "projects/hipdnn/flatbuffers_sdk/include/hipdnn_flatbuffers_sdk"
        / "utilities/json"
        / header
    )
    # Missing, the mode check would pass vacuously, as for the schemas.
    assert path.is_file(), f"serializer not found: {path}"
    pairs = re.findall(
        rf'\{{{name}::(\w+),\s*"(\w+)"\}}', path.read_text(encoding="utf-8")
    )
    assert pairs, f"the serializer's {name} names were not found"
    return dict(pairs)


# avgpool takes no index: the provider rejects one.
_AVG = {"index": False, "generate": False}


class TestResampleFwdExtract:
    PLAIN = {
        "mode:maxpool",
        "dtype:fp32",
        "spatial:2d",
        "layout:nchw",
        "stride:strided",
        "index:generated",
    }

    def _extract(self, **kwargs) -> set[str]:
        return resample_fwd.extract(*_rsf(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            resample_fwd.extract(node, tensors)

    def _group(self, features: set[str], group: str) -> set[str]:
        return {f for f in features if f.startswith(f"{group}:")}

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _rsf()
        written = set(node) - {"type", "compute_data_type"}
        assert written == table_fields(
            resample_fwd.SCHEMA_FILE, resample_fwd.SCHEMA_TABLE
        )

    @pytest.mark.parametrize(
        "enum, names, unset",
        [
            ("ResampleMode", resample_fwd.MODES, "not_set"),
            ("PaddingMode", resample_fwd.PADDING_MODES, "padding_not_set"),
        ],
    )
    def test_the_serializer_names_every_schema_enumerator(
        self, enum: str, names: tuple[str, ...], unset: str
    ) -> None:
        serialized = _serialized_enum(enum)
        assert set(serialized) == _schema_enum("resample_common.fbs", enum)
        assert set(names) == set(serialized.values()) - {unset}

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"mode": "avgpool_exclude_padding", "dtype": "half", **_AVG},
            {"mode": "avgpool_include_padding", "dtype": "bfloat16", **_AVG},
            {"pre": (1, 1), "post": (1, 1), "window": (3, 3)},
            {"pre": (1, 0), "padding_mode": "zero_pad", "layout": "nhwc"},
            {"dims": (2, 3, 8, 8, 8), "window": (2, 2, 2), "stride": (2, 2, 2)},
            {"stride": (1, 3), **_AVG, "mode": "avgpool_exclude_padding"},
            {"index": False, "generate": None},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {f.id for f in resample_fwd.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    def test_uid_zero_is_a_binding(self) -> None:
        node, tensors = _rsf()
        assert node["x_tensor_uid"] == 0
        assert resample_fwd.extract(node, tensors) == self.PLAIN

    @pytest.mark.parametrize("mode", resample_fwd.MODES)
    def test_mode(self, mode: str) -> None:
        kwargs = {} if mode == "maxpool" else _AVG
        features = self._extract(mode=mode, **kwargs)
        assert self._group(features, "mode") == {f"mode:{mode}"}

    @pytest.mark.parametrize(
        "mode", ["not_set", "MAXPOOL", "avgpool", "", 1, ["maxpool"], None]
    )
    def test_a_mode_that_is_not_catalogued_raises(self, mode) -> None:
        self._raises(*_rsf(mode=mode), "resample_mode .* is not catalogued")

    @pytest.mark.parametrize(
        "padding_mode",
        ["padding_not_set", "NEG_INF_PAD", "zero", "", 2, ["zero_pad"], None],
    )
    def test_a_padding_mode_that_is_not_catalogued_raises(self, padding_mode) -> None:
        self._raises(
            *_rsf(padding_mode=padding_mode), "padding_mode .* is not catalogued"
        )

    @pytest.mark.parametrize("key", ["resample_mode", "padding_mode"])
    def test_a_missing_mode_raises(self, key: str) -> None:
        node, tensors = _rsf()
        del node[key]
        self._raises(node, tensors, f"{key} None is not catalogued")

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "dtype, y_dtype",
        [("half", "float"), ("float", "bfloat16"), ("bfloat16", "half")],
    )
    def test_a_y_of_another_dtype_raises(self, dtype: str, y_dtype: str) -> None:
        self._raises(*_rsf(dtype=dtype, y_dtype=y_dtype), "is not that of x_tensor_uid")

    @pytest.mark.parametrize("uid, port", [(0, "x_tensor_uid"), (1, "y_tensor_uid")])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "fp8_e4m3", "double", None])
    def test_an_uncatalogued_dtype_raises(self, uid: int, port: str, dtype) -> None:
        node, tensors = _rsf()
        tensors[uid]["data_type"] = dtype
        self._raises(node, tensors, f"{port} dtype .* is not catalogued")

    @pytest.mark.parametrize(
        "dims, window, stride, expected",
        [
            ((2, 3, 8, 8), (2, 2), (2, 2), "spatial:2d"),
            ((2, 3, 4, 8, 8), (2, 2, 2), (2, 2, 2), "spatial:3d"),
        ],
    )
    def test_spatial(
        self,
        dims: tuple[int, ...],
        window: tuple[int, ...],
        stride: tuple[int, ...],
        expected: str,
    ) -> None:
        features = self._extract(dims=dims, window=window, stride=stride)
        assert self._group(features, "spatial") == {expected}

    @pytest.mark.parametrize(
        "dims, window, stride",
        [
            # The API takes rank 3 (1D); the provider and the CPU reference
            # do not.
            ((2, 3, 8), (2,), (2,)),
            ((2, 3, 4, 4, 4, 4), (2, 2, 2, 2), (2, 2, 2, 2)),
        ],
    )
    def test_a_rank_the_provider_rejects_raises(
        self, dims: tuple[int, ...], window: tuple[int, ...], stride: tuple[int, ...]
    ) -> None:
        node, tensors = _rsf(dims=dims, window=window, stride=stride)
        self._raises(node, tensors, f"rank {len(dims)} is not catalogued")

    def test_x_and_y_of_different_ranks_raise(self) -> None:
        node, tensors = _rsf()
        tensors[1]["dims"] = [2, 3, 4, 4, 1]
        self._raises(node, tensors, "are not the rank of x_tensor_uid")

    @pytest.mark.parametrize(
        "name", ["pre_padding", "post_padding", "stride", "window"]
    )
    @pytest.mark.parametrize(
        "values", [[2], [2, 2, 2], [], None, "2", [2, 2.0], [True, 2]]
    )
    def test_a_parameter_list_of_the_wrong_shape_raises(
        self, name: str, values
    ) -> None:
        node, tensors = _rsf()
        node[name] = values
        self._raises(node, tensors, f"{name} .* does not have 2 int entries")

    @pytest.mark.parametrize(
        "name", ["pre_padding", "post_padding", "stride", "window"]
    )
    def test_a_missing_parameter_list_raises(self, name: str) -> None:
        node, tensors = _rsf()
        del node[name]
        self._raises(node, tensors, f"{name} None does not have")

    @pytest.mark.parametrize("name", ["pre_padding", "post_padding"])
    def test_negative_padding_raises(self, name: str) -> None:
        node, tensors = _rsf()
        node[name] = [0, -1]
        self._raises(node, tensors, "negative padding")

    @pytest.mark.parametrize("name", ["stride", "window"])
    def test_a_stride_or_window_below_1_raises(self, name: str) -> None:
        node, tensors = _rsf()
        node[name] = [2, 0]
        # Not "below 1" alone: the spatial check's message says that too.
        self._raises(node, tensors, r"stride \[.*\] / window \[.*\] below 1")

    @pytest.mark.parametrize(
        "y_dims, match",
        [
            ([1, 3, 4, 4], "do not keep N and C"),
            ([2, 6, 4, 4], "do not keep N and C"),
            ([2, 3, 4, 5], r"spatial dims \[4, 5\] are not \[4, 4\]"),
            ([2, 3, 3, 4], r"spatial dims \[3, 4\] are not \[4, 4\]"),
        ],
    )
    def test_a_y_the_pooling_does_not_give_raises(
        self, y_dims: list[int], match: str
    ) -> None:
        node, tensors = _rsf()
        tensors[1]["dims"] = y_dims
        self._raises(node, tensors, match)

    def test_a_window_beyond_the_padded_x_raises(self) -> None:
        # (2 + 0 + 0 - 4) // 1 + 1 = -1: no y cell.
        node, tensors = _rsf(dims=(2, 3, 2, 2), window=(4, 4), stride=(1, 1))
        self._raises(node, tensors, "or one is below 1")

    @pytest.mark.parametrize(
        "dims, layout, expected",
        [
            ((2, 3, 8, 8), "nchw", {"layout:nchw"}),
            ((2, 3, 8, 8), "nhwc", {"layout:nhwc"}),
            ((2, 3, 4, 8, 8), "nhwc", {"layout:nhwc"}),
            # C of 1: no tensor tells.
            ((2, 1, 8, 8), "nhwc", set()),
        ],
    )
    def test_layout(
        self, dims: tuple[int, ...], layout: str, expected: set[str]
    ) -> None:
        spatial = len(dims) - 2
        features = self._extract(
            dims=dims, layout=layout, window=(2,) * spatial, stride=(2,) * spatial
        )
        assert self._group(features, "layout") == expected

    def test_x_and_y_in_different_layouts_raise(self) -> None:
        node, tensors = _rsf(layout="nchw", y_layout="nhwc")
        self._raises(node, tensors, "layouts of x_tensor_uid, y_tensor_uid")

    def test_an_index_in_another_layout_raises(self) -> None:
        node, tensors = _rsf()
        tensors[2]["strides"] = _layout_strides(tensors[2]["dims"], "nhwc")
        self._raises(node, tensors, "layouts of .*index_tensor_uid disagree")

    @pytest.mark.parametrize("uid", [0, 1, 2])
    def test_strides_no_layout_describes_raise(self, uid: int) -> None:
        node, tensors = _rsf()
        tensors[uid]["strides"] = list(reversed(tensors[uid]["strides"]))
        self._raises(node, tensors, "not a catalogued layout|overlap")

    @pytest.mark.parametrize(
        "pre, post, expected",
        [
            ((0, 0), (0, 0), set()),
            ((1, 1), (1, 1), {"padding:symmetric"}),
            ((1, 0), (1, 0), {"padding:symmetric"}),
            ((0, 1), (0, 1), {"padding:symmetric"}),
            ((1, 0), (0, 1), {"padding:asymmetric"}),
            ((0, 0), (1, 0), {"padding:asymmetric"}),
            ((0, 1), (0, 0), {"padding:asymmetric"}),
        ],
    )
    def test_padding(
        self, pre: tuple[int, ...], post: tuple[int, ...], expected: set[str]
    ) -> None:
        features = self._extract(pre=pre, post=post)
        assert self._group(features, "padding") == expected

    @pytest.mark.parametrize(
        "window, stride, expected",
        [
            ((2, 2), (2, 2), {"stride:strided"}),
            ((1, 1), (1, 1), set()),
            ((2, 2), (1, 1), {"window:overlapping"}),
            ((3, 3), (2, 2), {"stride:strided", "window:overlapping"}),
            ((2, 2), (3, 3), {"stride:strided", "window:gapped"}),
            ((1, 2), (2, 1), {"stride:strided", "window:overlapping", "window:gapped"}),
        ],
    )
    def test_stride_and_window(
        self, window: tuple[int, ...], stride: tuple[int, ...], expected: set[str]
    ) -> None:
        features = self._extract(window=window, stride=stride)
        assert (
            self._group(features, "stride") | self._group(features, "window")
            == expected
        )

    @pytest.mark.parametrize(
        "pre, post, window, stride, reached",
        [
            ((0, 0), (0, 0), (2, 2), (2, 2), False),
            ((1, 0), (0, 0), (2, 2), (2, 2), True),
            ((0, 1), (0, 0), (2, 2), (2, 2), True),
            # x 8: (8 + 1 - 2) // 2 + 1 = 4 windows, the last ending on x[7].
            ((0, 0), (1, 0), (2, 2), (2, 2), False),
            # (8 + 1 - 3) // 2 + 1 = 4 windows, the last ending on x[8], padding.
            ((0, 0), (1, 0), (3, 2), (2, 2), True),
            ((0, 0), (0, 1), (2, 3), (2, 1), True),
        ],
    )
    @pytest.mark.parametrize("padding_mode", resample_fwd.PADDING_MODES)
    def test_padding_mode_counts_under_maxpool_when_a_window_reaches_padding(
        self,
        pre: tuple[int, ...],
        post: tuple[int, ...],
        window: tuple[int, ...],
        stride: tuple[int, ...],
        reached: bool,
        padding_mode: str,
    ) -> None:
        features = self._extract(
            pre=pre, post=post, window=window, stride=stride, padding_mode=padding_mode
        )
        expected = {f"padding_mode:{padding_mode}"} if reached else set()
        assert self._group(features, "padding_mode") == expected

    @pytest.mark.parametrize(
        "mode", ["avgpool_exclude_padding", "avgpool_include_padding"]
    )
    @pytest.mark.parametrize("padding_mode", resample_fwd.PADDING_MODES)
    def test_an_average_reads_no_padding_mode_and_no_index(
        self, mode: str, padding_mode: str
    ) -> None:
        features = self._extract(
            mode=mode, padding_mode=padding_mode, pre=(1, 1), post=(1, 1), **_AVG
        )
        assert not self._group(features, "padding_mode")
        assert not self._group(features, "index")

    @pytest.mark.parametrize("generate", [False, None])
    def test_maxpool_without_an_index(self, generate: bool | None) -> None:
        features = self._extract(index=False, generate=generate)
        assert features == self.PLAIN - {"index:generated"} | {"index:none"}

    @pytest.mark.parametrize(
        "mode", ["maxpool", "avgpool_exclude_padding", "avgpool_include_padding"]
    )
    def test_generate_index_without_an_index_raises(self, mode: str) -> None:
        node, tensors = _rsf(mode=mode, index=False, generate=True)
        self._raises(node, tensors, "generate_index is on but index_tensor_uid")

    @pytest.mark.parametrize(
        "mode", ["avgpool_exclude_padding", "avgpool_include_padding"]
    )
    @pytest.mark.parametrize("generate", [True, False])
    def test_an_index_under_an_average_raises(self, mode: str, generate: bool) -> None:
        node, tensors = _rsf(mode=mode, generate=generate)
        self._raises(node, tensors, f"index_tensor_uid is bound under .* {mode}")

    @pytest.mark.parametrize("generate", [False, None])
    def test_an_index_left_ungenerated_raises(self, generate: bool | None) -> None:
        # The provider leaves it unwritten; the CPU reference fills it.
        node, tensors = _rsf(generate=generate)
        self._raises(node, tensors, "index_tensor_uid is bound but generate_index")

    @pytest.mark.parametrize("value", [1, 0, "true", None])
    def test_a_generate_index_that_is_not_a_bool_raises(self, value) -> None:
        node, tensors = _rsf()
        node["generate_index"] = value
        self._raises(node, tensors, "generate_index .* is not a bool")

    @pytest.mark.parametrize("dtype", ["int8", "int64", "float", None])
    def test_an_index_that_is_not_int32_raises(self, dtype) -> None:
        # The API takes int8; the provider and the CPU reference do not.
        node, tensors = _rsf()
        tensors[2]["data_type"] = dtype
        self._raises(node, tensors, "index_tensor_uid dtype .* is not catalogued")

    def test_an_index_not_of_y_dims_raises(self) -> None:
        node, tensors = _rsf()
        tensors[2]["dims"] = [2, 3, 8, 8]
        tensors[2]["strides"] = _layout_strides([2, 3, 8, 8], "nchw")
        self._raises(node, tensors, "index_tensor_uid dims .* are not y_tensor_uid")

    def test_uid_zero_is_an_index_binding(self) -> None:
        node, tensors = _rsf()
        tensors = {
            0: {**tensors[2], "uid": 0},
            1: tensors[1],
            2: {**tensors[0], "uid": 2},
        }
        node["x_tensor_uid"], node["index_tensor_uid"] = 2, 0
        assert resample_fwd.extract(node, tensors) == self.PLAIN

    @pytest.mark.parametrize(
        "kwargs, expected",
        # The bundles' shapes, in each layout.
        [
            (
                {
                    "mode": "avgpool_include_padding",
                    "padding_mode": "zero_pad",
                    "dims": (1, 2, 4, 5, 3),
                    "window": (2, 2, 2),
                    "stride": (1, 2, 1),
                    "pre": (1, 0, 1),
                    "post": (0, 1, 0),
                    **_AVG,
                },
                {
                    "mode:avgpool_include_padding",
                    "spatial:3d",
                    "padding:asymmetric",
                    "stride:strided",
                    "window:overlapping",
                },
            ),
            (
                {
                    "mode": "avgpool_exclude_padding",
                    "padding_mode": "zero_pad",
                    "dims": (2, 3, 7, 5),
                    "window": (3, 2),
                    "stride": (2, 1),
                    "pre": (1, 0),
                    "post": (0, 1),
                    **_AVG,
                },
                {
                    "mode:avgpool_exclude_padding",
                    "spatial:2d",
                    "padding:asymmetric",
                    "stride:strided",
                    "window:overlapping",
                },
            ),
            (
                {
                    "mode": "avgpool_include_padding",
                    "padding_mode": "zero_pad",
                    "dims": (2, 3, 7, 5),
                    "window": (2, 3),
                    "stride": (1, 2),
                    "pre": (0, 1),
                    "post": (1, 0),
                    **_AVG,
                },
                {
                    "mode:avgpool_include_padding",
                    "spatial:2d",
                    "padding:asymmetric",
                    "stride:strided",
                    "window:overlapping",
                },
            ),
            (
                {
                    "dims": (1, 2, 4, 5, 3),
                    "window": (2, 2, 2),
                    "stride": (1, 2, 1),
                },
                {
                    "mode:maxpool",
                    "spatial:3d",
                    "stride:strided",
                    "window:overlapping",
                    "index:generated",
                },
            ),
            (
                {"dims": (2, 3, 8, 6)},
                {"mode:maxpool", "spatial:2d", "stride:strided", "index:generated"},
            ),
            (
                {
                    "padding_mode": "zero_pad",
                    "dims": (2, 3, 8, 6),
                    "window": (3, 3),
                    "pre": (1, 1),
                    "post": (1, 1),
                },
                {
                    "mode:maxpool",
                    "padding_mode:zero_pad",
                    "spatial:2d",
                    "padding:symmetric",
                    "stride:strided",
                    "window:overlapping",
                    "index:generated",
                },
            ),
        ],
    )
    @pytest.mark.parametrize("layout", ["nchw", "nhwc"])
    def test_bundle_shapes(self, kwargs: dict, expected: set[str], layout: str) -> None:
        features = self._extract(**kwargs, layout=layout)
        assert features == expected | {"dtype:fp32", f"layout:{layout}"}

    @pytest.mark.parametrize("port", ["x_tensor_uid", "y_tensor_uid"])
    @pytest.mark.parametrize("unbound", ["null", "missing"])
    def test_unbound_required_port_raises(self, port: str, unbound: str) -> None:
        node, tensors = _rsf()
        if unbound == "null":
            node[port] = None
        else:
            del node[port]
        self._raises(node, tensors, f"required port {port}")

    def test_a_null_index_is_unbound(self) -> None:
        node, tensors = _rsf(generate=False)
        node["index_tensor_uid"] = None
        assert resample_fwd.extract(node, tensors) == self.PLAIN - {
            "index:generated"
        } | {"index:none"}

    def test_ports_under_inputs_and_outputs_are_not_read(self) -> None:
        # The serializer writes them at the top, as for reduction.
        node, tensors = _rsf()
        node["inputs"] = {"x_tensor_uid": node.pop("x_tensor_uid")}
        node["outputs"] = {"y_tensor_uid": node.pop("y_tensor_uid")}
        self._raises(node, tensors, "required port x_tensor_uid")


def _rsb(*, index: bool = True, **kwargs) -> tuple[dict, dict]:
    """A ResampleBwdAttributes node: ``_rsf``'s graph read backward, its
    ``x`` as ``dx`` (uid 0), its ``y`` as ``dy`` (uid 1) and, with ``index``,
    the int32 index on uid 2 of ``dy``'s dims and strides. ``y_dtype`` and
    ``y_layout`` set ``dy``'s. No ``generate_index``: the schema has none.
    """
    fwd, tensors = _rsf(index=index, generate=None, **kwargs)
    node = {
        "type": "ResampleBwdAttributes",
        "compute_data_type": "float",
        "dy_tensor_uid": fwd.pop("y_tensor_uid"),
        "dx_tensor_uid": fwd.pop("x_tensor_uid"),
    }
    node.update(
        (key, value)
        for key, value in fwd.items()
        if key not in ("type", "compute_data_type")
    )
    return node, tensors


class TestResampleBwdExtract:
    PLAIN = {
        "mode:maxpool",
        "dtype:fp32",
        "spatial:2d",
        "layout:nchw",
        "stride:strided",
    }
    # The average modes take no index: the provider rejects one.
    AVG = {"index": False}

    def _extract(self, **kwargs) -> set[str]:
        return resample_bwd.extract(*_rsb(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            resample_bwd.extract(node, tensors)

    def _group(self, features: set[str], group: str) -> set[str]:
        return {f for f in features if f.startswith(f"{group}:")}

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _rsb()
        written = set(node) - {"type", "compute_data_type"}
        assert written == table_fields(
            resample_bwd.SCHEMA_FILE, resample_bwd.SCHEMA_TABLE
        )

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"mode": "avgpool_exclude_padding", "dtype": "half", **self.AVG},
            {"mode": "avgpool_include_padding", "dtype": "bfloat16", **self.AVG},
            {"pre": (1, 1), "post": (1, 1), "window": (3, 3)},
            {"pre": (1, 0), "layout": "nhwc"},
            {"dims": (2, 3, 8, 8, 8), "window": (2, 2, 2), "stride": (2, 2, 2)},
            {"stride": (1, 3)},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {f.id for f in resample_bwd.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    def test_uid_zero_is_a_binding(self) -> None:
        node, tensors = _rsb()
        assert node["dx_tensor_uid"] == 0
        assert resample_bwd.extract(node, tensors) == self.PLAIN

    @pytest.mark.parametrize("mode", resample_fwd.MODES)
    def test_mode(self, mode: str) -> None:
        kwargs = {} if mode == "maxpool" else self.AVG
        features = self._extract(mode=mode, **kwargs)
        assert self._group(features, "mode") == {f"mode:{mode}"}

    @pytest.mark.parametrize(
        "mode", ["not_set", "MAXPOOL", "avgpool", "", 1, ["maxpool"], None]
    )
    def test_a_mode_that_is_not_catalogued_raises(self, mode) -> None:
        self._raises(*_rsb(mode=mode), "resample_mode .* is not catalogued")

    @pytest.mark.parametrize(
        "padding_mode",
        ["padding_not_set", "NEG_INF_PAD", "zero", "", 2, ["zero_pad"], None],
    )
    def test_a_padding_mode_that_is_not_catalogued_raises(self, padding_mode) -> None:
        # Not a feature, but the provider and the CPU reference reject these.
        self._raises(
            *_rsb(padding_mode=padding_mode), "padding_mode .* is not catalogued"
        )

    @pytest.mark.parametrize("key", ["resample_mode", "padding_mode"])
    def test_a_missing_mode_raises(self, key: str) -> None:
        node, tensors = _rsb()
        del node[key]
        self._raises(node, tensors, f"{key} None is not catalogued")

    @pytest.mark.parametrize("mode", resample_fwd.MODES)
    def test_padding_mode_drives_no_feature(self, mode: str) -> None:
        kwargs = {
            "mode": mode,
            "pre": (1, 1),
            "post": (1, 1),
            "window": (3, 3),
            **({} if mode == "maxpool" else self.AVG),
        }
        neg_inf, zero = (
            self._extract(padding_mode=padding_mode, **kwargs)
            for padding_mode in resample_fwd.PADDING_MODES
        )
        assert neg_inf == zero
        assert not self._group(neg_inf, "padding_mode")

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "dy_dtype, dx_dtype",
        [("half", "float"), ("float", "bfloat16"), ("bfloat16", "half")],
    )
    def test_a_dx_of_another_dtype_raises(self, dy_dtype: str, dx_dtype: str) -> None:
        node, tensors = _rsb(dtype=dx_dtype, y_dtype=dy_dtype)
        self._raises(node, tensors, "dx_tensor_uid dtype .* is not that of dy_tensor")

    @pytest.mark.parametrize("uid, port", [(1, "dy_tensor_uid"), (0, "dx_tensor_uid")])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "fp8_e4m3", "double", None])
    def test_an_uncatalogued_dtype_raises(self, uid: int, port: str, dtype) -> None:
        node, tensors = _rsb()
        tensors[uid]["data_type"] = dtype
        self._raises(node, tensors, f"{port} dtype .* is not catalogued")

    @pytest.mark.parametrize(
        "dims, window, stride, expected",
        [
            ((2, 3, 8, 8), (2, 2), (2, 2), "spatial:2d"),
            ((2, 3, 4, 8, 8), (2, 2, 2), (2, 2, 2), "spatial:3d"),
        ],
    )
    def test_spatial(
        self,
        dims: tuple[int, ...],
        window: tuple[int, ...],
        stride: tuple[int, ...],
        expected: str,
    ) -> None:
        features = self._extract(dims=dims, window=window, stride=stride)
        assert self._group(features, "spatial") == {expected}

    @pytest.mark.parametrize(
        "dims, window, stride",
        [
            # The API takes rank 3 (1D); the provider and the CPU reference
            # do not.
            ((2, 3, 8), (2,), (2,)),
            ((2, 3, 4, 4, 4, 4), (2, 2, 2, 2), (2, 2, 2, 2)),
        ],
    )
    def test_a_rank_the_provider_rejects_raises(
        self, dims: tuple[int, ...], window: tuple[int, ...], stride: tuple[int, ...]
    ) -> None:
        node, tensors = _rsb(dims=dims, window=window, stride=stride)
        self._raises(node, tensors, f"dx_tensor_uid .*: rank {len(dims)} is not")

    def test_dy_and_dx_of_different_ranks_raise(self) -> None:
        node, tensors = _rsb()
        tensors[1]["dims"] = [2, 3, 4, 4, 1]
        self._raises(node, tensors, "dy_tensor_uid .* are not the rank of dx_tensor")

    @pytest.mark.parametrize(
        "name", ["pre_padding", "post_padding", "stride", "window"]
    )
    @pytest.mark.parametrize(
        "values", [[2], [2, 2, 2], [], None, "2", [2, 2.0], [True, 2]]
    )
    def test_a_parameter_list_of_the_wrong_shape_raises(
        self, name: str, values
    ) -> None:
        node, tensors = _rsb()
        node[name] = values
        self._raises(node, tensors, f"{name} .* does not have 2 int entries")

    @pytest.mark.parametrize("name", ["pre_padding", "post_padding"])
    def test_negative_padding_raises(self, name: str) -> None:
        node, tensors = _rsb()
        node[name] = [0, -1]
        self._raises(node, tensors, "negative padding")

    @pytest.mark.parametrize("name", ["stride", "window"])
    def test_a_stride_or_window_below_1_raises(self, name: str) -> None:
        node, tensors = _rsb()
        node[name] = [2, 0]
        self._raises(node, tensors, r"stride \[.*\] / window \[.*\] below 1")

    @pytest.mark.parametrize(
        "dy_dims, match",
        [
            ([1, 3, 4, 4], "do not keep N and C"),
            ([2, 6, 4, 4], "do not keep N and C"),
            ([2, 3, 4, 5], r"dy_tensor_uid spatial dims \[4, 5\] are not \[4, 4\]"),
            ([2, 3, 3, 4], r"dy_tensor_uid spatial dims \[3, 4\] are not \[4, 4\]"),
        ],
    )
    def test_a_dy_the_pooling_does_not_give_raises(
        self, dy_dims: list[int], match: str
    ) -> None:
        node, tensors = _rsb(index=False, mode="avgpool_exclude_padding")
        tensors[1]["dims"] = dy_dims
        self._raises(node, tensors, match)

    def test_a_window_beyond_the_padded_dx_raises(self) -> None:
        # (2 + 0 + 0 - 4) // 1 + 1 = -1: no dy cell.
        node, tensors = _rsb(dims=(2, 3, 2, 2), window=(4, 4), stride=(1, 1))
        self._raises(node, tensors, "or one is below 1")

    @pytest.mark.parametrize(
        "dims, layout, expected",
        [
            ((2, 3, 8, 8), "nchw", {"layout:nchw"}),
            ((2, 3, 8, 8), "nhwc", {"layout:nhwc"}),
            ((2, 3, 4, 8, 8), "nhwc", {"layout:nhwc"}),
            # C of 1: no tensor tells.
            ((2, 1, 8, 8), "nhwc", set()),
        ],
    )
    def test_layout(
        self, dims: tuple[int, ...], layout: str, expected: set[str]
    ) -> None:
        spatial = len(dims) - 2
        features = self._extract(
            dims=dims, layout=layout, window=(2,) * spatial, stride=(2,) * spatial
        )
        assert self._group(features, "layout") == expected

    def test_dy_and_dx_in_different_layouts_raise(self) -> None:
        node, tensors = _rsb(index=False, mode="avgpool_exclude_padding")
        tensors[1]["strides"] = _layout_strides(tensors[1]["dims"], "nhwc")
        self._raises(node, tensors, "layouts of dy_tensor_uid, dx_tensor_uid disagree")

    def test_an_index_in_another_layout_raises(self) -> None:
        node, tensors = _rsb()
        tensors[2]["strides"] = _layout_strides(tensors[2]["dims"], "nhwc")
        self._raises(node, tensors, "layouts of .*index_tensor_uid disagree")

    @pytest.mark.parametrize("uid", [0, 1, 2])
    def test_strides_no_layout_describes_raise(self, uid: int) -> None:
        node, tensors = _rsb()
        tensors[uid]["strides"] = list(reversed(tensors[uid]["strides"]))
        self._raises(node, tensors, "not a catalogued layout|overlap")

    @pytest.mark.parametrize(
        "pre, post, expected",
        [
            ((0, 0), (0, 0), set()),
            ((1, 1), (1, 1), {"padding:symmetric"}),
            ((0, 1), (0, 1), {"padding:symmetric"}),
            ((1, 0), (0, 1), {"padding:asymmetric"}),
            ((0, 0), (0, 1), {"padding:asymmetric"}),
        ],
    )
    def test_padding(
        self, pre: tuple[int, ...], post: tuple[int, ...], expected: set[str]
    ) -> None:
        features = self._extract(pre=pre, post=post)
        assert self._group(features, "padding") == expected

    @pytest.mark.parametrize(
        "window, stride, expected",
        [
            ((2, 2), (2, 2), {"stride:strided"}),
            ((1, 1), (1, 1), set()),
            ((2, 2), (1, 1), {"window:overlapping"}),
            ((2, 2), (3, 3), {"stride:strided", "window:gapped"}),
            ((1, 2), (2, 1), {"stride:strided", "window:overlapping", "window:gapped"}),
        ],
    )
    def test_stride_and_window(
        self, window: tuple[int, ...], stride: tuple[int, ...], expected: set[str]
    ) -> None:
        features = self._extract(window=window, stride=stride)
        assert (
            self._group(features, "stride") | self._group(features, "window")
            == expected
        )

    @pytest.mark.parametrize("unbound", ["null", "missing"])
    def test_maxpool_without_an_index_raises(self, unbound: str) -> None:
        # Unlike the forward: the index is how maxpool's gradient finds x.
        node, tensors = _rsb()
        if unbound == "null":
            node["index_tensor_uid"] = None
        else:
            del node["index_tensor_uid"]
        self._raises(node, tensors, "resample_mode maxpool requires index_tensor_uid")

    @pytest.mark.parametrize(
        "mode", ["avgpool_exclude_padding", "avgpool_include_padding"]
    )
    def test_an_index_under_an_average_raises(self, mode: str) -> None:
        node, tensors = _rsb(mode=mode)
        self._raises(node, tensors, f"index_tensor_uid is bound under .* {mode}")

    @pytest.mark.parametrize("dtype", ["int8", "int64", "float", None])
    def test_an_index_that_is_not_int32_raises(self, dtype) -> None:
        # The API takes int8; the provider and the CPU reference do not.
        node, tensors = _rsb()
        tensors[2]["data_type"] = dtype
        self._raises(node, tensors, "index_tensor_uid dtype .* is not catalogued")

    def test_an_index_not_of_dy_dims_raises(self) -> None:
        node, tensors = _rsb()
        tensors[2]["dims"] = [2, 3, 8, 8]
        tensors[2]["strides"] = _layout_strides([2, 3, 8, 8], "nchw")
        self._raises(node, tensors, "index_tensor_uid dims .* are not dy_tensor_uid")

    def test_uid_zero_is_an_index_binding(self) -> None:
        node, tensors = _rsb()
        tensors = {
            0: {**tensors[2], "uid": 0},
            1: tensors[1],
            2: {**tensors[0], "uid": 2},
        }
        node["dx_tensor_uid"], node["index_tensor_uid"] = 2, 0
        assert resample_bwd.extract(node, tensors) == self.PLAIN

    @pytest.mark.parametrize("port", ["dy_tensor_uid", "dx_tensor_uid"])
    @pytest.mark.parametrize("unbound", ["null", "missing"])
    def test_unbound_required_port_raises(self, port: str, unbound: str) -> None:
        node, tensors = _rsb()
        if unbound == "null":
            node[port] = None
        else:
            del node[port]
        self._raises(node, tensors, f"required port {port}")

    def test_ports_under_inputs_and_outputs_are_not_read(self) -> None:
        # The serializer writes them at the top, as for the forward.
        node, tensors = _rsb()
        node["inputs"] = {"dy_tensor_uid": node.pop("dy_tensor_uid")}
        node["outputs"] = {"dx_tensor_uid": node.pop("dx_tensor_uid")}
        self._raises(node, tensors, "required port dy_tensor_uid")


# --------------------------------------------------------------------------
# MoeGroupedMatmulAttributes
# --------------------------------------------------------------------------


def _moe(
    *,
    mode: str = "none",
    experts: int = 2,
    batch: int = 1,
    tokens: int = 8,
    k: int = 16,
    n: int = 32,
    top_k: int | None = None,
    routed: int | None = None,
    dtype: str = "float",
    weight_dtype: str | None = None,
    out_dtype: str | None = None,
    weight_layout: str = "row_major",
    token_virtual: bool = False,
    output_virtual: bool = False,
) -> tuple[dict, dict]:
    """A MoeGroupedMatmulAttributes node: ``token`` on uid 1, ``weight`` on 2,
    the offset on 3, the index on 4, the ks on 5, ``output`` on 6.

    ``gather`` binds an index of ``routed`` rows (``tokens`` unless told
    otherwise), ``scatter`` an index and a ks of ``tokens`` rows; ``top_k``
    is 1 under ``scatter`` and 0 otherwise unless told. ``output`` has the
    rows the mode routes to. Ports as
    utilities/json/MoeGroupedMatmulAttributes.hpp writes them.
    """
    if top_k is None:
        top_k = 1 if mode == "scatter" else 0
    rows = tokens if routed is None else routed

    def tensor(uid: int, dims: list[int], dt: str, virtual: bool = False) -> dict:
        return {
            "uid": uid,
            "dims": dims,
            "strides": _matrix_strides(dims, "row_major"),
            "data_type": dt,
            "virtual": virtual,
        }

    weight = tensor(2, [experts, k, n], weight_dtype or dtype)
    weight["strides"] = _matrix_strides(weight["dims"], weight_layout)
    tensors = {
        1: tensor(1, [1, tokens, k], dtype, token_virtual),
        2: weight,
        3: tensor(3, [batch * experts, 1, 1], "int32"),
        4: tensor(4, [1, rows, 1], "int32"),
        5: tensor(5, [1, rows, 1], "int32"),
        6: tensor(
            6,
            [1, rows if mode == "gather" else tokens, n],
            out_dtype or dtype,
            output_virtual,
        ),
    }
    inputs = {
        "token_tensor_uid": 1,
        "weight_tensor_uid": 2,
        "first_token_offset_tensor_uid": 3,
    }
    if mode in ("gather", "scatter"):
        inputs["token_index_tensor_uid"] = 4
    if mode == "scatter":
        inputs["token_ks_tensor_uid"] = 5
    node = {
        "type": "MoeGroupedMatmulAttributes",
        "inputs": inputs,
        "outputs": {"output_tensor_uid": 6},
        "mode": mode,
        "top_k": top_k,
    }
    return node, tensors


class TestMoeGroupedMatmulExtract:
    PLAIN = {"mode:none", "dtype:fp32", "batch:single", "weight_layout:row_major"}

    def _extract(self, **kwargs) -> set[str]:
        return moe_grouped_matmul.extract(*_moe(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            moe_grouped_matmul.extract(node, tensors)

    @staticmethod
    def _group(features: set[str], group: str) -> set[str]:
        return {f for f in features if f.startswith(f"{group}:")}

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _moe(mode="scatter")
        written = set(batchnorm.bound(node)) | set(node) - {"type", "inputs", "outputs"}
        assert written == table_fields(
            moe_grouped_matmul.SCHEMA_FILE, moe_grouped_matmul.SCHEMA_TABLE
        )

    def test_the_serializer_names_every_schema_enumerator(self) -> None:
        # No NOT_SET to leave out: the schema's default is NONE.
        serialized = _serialized_enum(
            "MoeGroupedMatmulMode", "MoeGroupedMatmulAttributes.hpp"
        )
        assert set(serialized) == _schema_enum(
            moe_grouped_matmul.SCHEMA_FILE, "MoeGroupedMatmulMode"
        )
        assert set(moe_grouped_matmul.MODES) == set(serialized.values())

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"mode": "gather", "dtype": "half", "out_dtype": "float", "batch": 2},
            {"mode": "scatter", "dtype": "bfloat16", "weight_layout": "col_major"},
            {"mode": "scatter", "top_k": 2},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {f.id for f in moe_grouped_matmul.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "dtype, out_dtype",
        [
            ("half", "float"),
            ("bfloat16", "float"),
            ("float", "half"),
            ("float", "bfloat16"),
        ],
    )
    def test_mixed_output(self, dtype: str, out_dtype: str) -> None:
        # The CPU reference's pairs: the other dtype fp32.
        features = self._extract(dtype=dtype, out_dtype=out_dtype)
        assert "precision:mixed_output" in features

    @pytest.mark.parametrize("virtual", ["token_virtual", "output_virtual"])
    def test_a_virtual_tensor_is_not_mixed(self, virtual: str) -> None:
        features = self._extract(dtype="half", out_dtype="float", **{virtual: True})
        assert features == self.PLAIN - {"dtype:fp32"} | {"dtype:fp16"}

    @pytest.mark.parametrize("output_virtual", [False, True])
    @pytest.mark.parametrize(
        "dtype, out_dtype", [("half", "bfloat16"), ("bfloat16", "half")]
    )
    def test_an_output_pair_with_no_plan_raises(
        self, dtype: str, out_dtype: str, output_virtual: bool
    ) -> None:
        # The signature key reads the dtype of a virtual output too.
        node, tensors = _moe(
            dtype=dtype, out_dtype=out_dtype, output_virtual=output_virtual
        )
        self._raises(node, tensors, "output_tensor_uid dtype .* has no plan")

    def test_a_weight_not_of_token_dtype_raises(self) -> None:
        node, tensors = _moe(dtype="half", weight_dtype="float")
        self._raises(node, tensors, "weight_tensor_uid dtype .* is not that of")

    @pytest.mark.parametrize(
        "kwargs, port",
        [
            ({"dtype": "int8"}, "token_tensor_uid"),
            ({"weight_dtype": "double"}, "weight_tensor_uid"),
            ({"out_dtype": "int32"}, "output_tensor_uid"),
        ],
    )
    def test_an_uncatalogued_dtype_raises(self, kwargs: dict, port: str) -> None:
        node, tensors = _moe(**kwargs)
        self._raises(node, tensors, f"{port} dtype .* is not catalogued")

    @pytest.mark.parametrize(
        "mode, expected",
        [("none", "mode:none"), ("gather", "mode:gather"), ("scatter", "mode:scatter")],
    )
    def test_mode(self, mode: str, expected: str) -> None:
        assert self._group(self._extract(mode=mode), "mode") == {expected}

    @pytest.mark.parametrize("mode", ["NONE", "not_set", None, 0, "missing"])
    def test_an_uncatalogued_mode_raises(self, mode) -> None:
        # The reader requires mode, so a missing one is not NONE.
        node, tensors = _moe()
        if mode == "missing":
            del node["mode"]
        else:
            node["mode"] = mode
        self._raises(node, tensors, "mode .* is not catalogued")

    @pytest.mark.parametrize(
        "batch, expected", [(1, "batch:single"), (2, "batch:multi"), (3, "batch:multi")]
    )
    @pytest.mark.parametrize("experts", [1, 2, 4])
    def test_batch(self, batch: int, expected: str, experts: int) -> None:
        # Read on G against E, not on G alone.
        features = self._extract(batch=batch, experts=experts)
        assert self._group(features, "batch") == {expected}

    @pytest.mark.parametrize(
        "top_k, expected", [(1, {"top_k:one"}), (2, {"top_k:multi"})]
    )
    def test_top_k_under_scatter(self, top_k: int, expected: set[str]) -> None:
        features = self._extract(mode="scatter", top_k=top_k)
        assert self._group(features, "top_k") == expected

    @pytest.mark.parametrize("mode", ["none", "gather"])
    def test_no_top_k_feature_off_scatter(self, mode: str) -> None:
        assert not self._group(self._extract(mode=mode), "top_k")

    @pytest.mark.parametrize("top_k", [0, -1, 3])
    def test_a_scatter_top_k_off_1_to_experts_raises(self, top_k: int) -> None:
        node, tensors = _moe(mode="scatter", top_k=top_k)
        self._raises(node, tensors, f"top_k {top_k} is not from 1 to 2 experts")

    @pytest.mark.parametrize("top_k", [1, -1])
    @pytest.mark.parametrize("mode", ["none", "gather"])
    def test_a_top_k_off_scatter_raises(self, mode: str, top_k: int) -> None:
        node, tensors = _moe(mode=mode, top_k=top_k)
        self._raises(node, tensors, f"top_k {top_k} is not 0 under mode {mode}")

    @pytest.mark.parametrize("top_k", [True, False, 1.0, "1", None, "missing"])
    def test_a_top_k_that_is_not_an_int_raises(self, top_k) -> None:
        node, tensors = _moe(mode="scatter")
        if top_k == "missing":
            del node["top_k"]
        else:
            node["top_k"] = top_k
        self._raises(node, tensors, "top_k .* is not an int")

    @pytest.mark.parametrize("port", ["token_index_tensor_uid", "token_ks_tensor_uid"])
    def test_none_binding_routing_raises(self, port: str) -> None:
        # The frontend takes an index under NONE; the backend does not.
        node, tensors = _moe()
        node["inputs"][port] = 4
        self._raises(node, tensors, "mode none binds")

    @pytest.mark.parametrize(
        "drop, add",
        [("token_index_tensor_uid", None), (None, "token_ks_tensor_uid")],
    )
    def test_gather_not_binding_an_index_alone_raises(self, drop, add) -> None:
        node, tensors = _moe(mode="gather")
        if drop:
            node["inputs"][drop] = None
        if add:
            node["inputs"][add] = 5
        self._raises(node, tensors, "mode gather does not bind token_index")

    @pytest.mark.parametrize("port", ["token_index_tensor_uid", "token_ks_tensor_uid"])
    def test_scatter_not_binding_both_raises(self, port: str) -> None:
        node, tensors = _moe(mode="scatter")
        del node["inputs"][port]
        self._raises(node, tensors, "mode scatter does not bind both")

    @pytest.mark.parametrize(
        "mode, uid", [("gather", 4), ("scatter", 4), ("scatter", 5)]
    )
    @pytest.mark.parametrize("dtype", ["int64", "int8", "float", None])
    def test_routing_not_int32_raises(self, mode: str, uid: int, dtype) -> None:
        node, tensors = _moe(mode=mode)
        tensors[uid]["data_type"] = dtype
        self._raises(node, tensors, "token_(index|ks)_tensor_uid dtype .* is not int32")

    @pytest.mark.parametrize("uid", [4, 5])
    @pytest.mark.parametrize("dims", [[8], [1, 8], [2, 8, 1], [1, 8, 2], [1, 0, 1]])
    def test_routing_not_1_rows_1_raises(self, uid: int, dims: list[int]) -> None:
        node, tensors = _moe(mode="scatter")
        tensors[uid]["dims"] = dims
        self._raises(node, tensors, "token_(index|ks)_tensor_uid dims .* rows, 1")

    @pytest.mark.parametrize("uid, rows", [(4, 5), (5, 5)])
    def test_scatter_index_and_ks_rows_disagreeing_raises(
        self, uid: int, rows: int
    ) -> None:
        node, tensors = _moe(mode="scatter")
        tensors[uid]["dims"] = [1, rows, 1]
        self._raises(node, tensors, "rows .* are not token_tensor_uid rows 8")

    def test_scatter_routing_not_token_rows_raises(self) -> None:
        # The API checks only that index and ks agree; the CPU reference
        # bounds them by token's rows.
        node, tensors = _moe(mode="scatter", routed=5)
        self._raises(node, tensors, "index_tensor_uid rows 5 and .* rows 5 are not")

    @pytest.mark.parametrize("routed", [5, 8, 13])
    def test_gather_output_rows_are_the_index_rows(self, routed: int) -> None:
        features = self._extract(mode="gather", routed=routed)
        assert self._group(features, "mode") == {"mode:gather"}

    def test_gather_output_of_token_rows_raises(self) -> None:
        node, tensors = _moe(mode="gather", routed=5)
        tensors[6]["dims"] = [1, 8, 32]
        self._raises(node, tensors, r"output_tensor_uid dims .* are not \[1, 5, 32\]")

    @pytest.mark.parametrize("mode", ["none", "scatter"])
    @pytest.mark.parametrize("dims", [[1, 8, 16], [1, 7, 32], [2, 8, 32]])
    def test_an_output_of_the_wrong_dims_raises(self, mode: str, dims) -> None:
        node, tensors = _moe(mode=mode)
        tensors[6]["dims"] = dims
        self._raises(node, tensors, r"output_tensor_uid dims .* are not \[1, 8, 32\]")

    @pytest.mark.parametrize(
        "uid, port",
        [
            (1, "token_tensor_uid"),
            (2, "weight_tensor_uid"),
            (3, "first_token_offset_tensor_uid"),
            (6, "output_tensor_uid"),
        ],
    )
    @pytest.mark.parametrize("dims", [[8, 16], [1, 1, 8, 16], [1, 0, 16], []])
    def test_a_required_tensor_not_rank_3_raises(self, uid, port, dims) -> None:
        node, tensors = _moe()
        tensors[uid]["dims"] = dims
        self._raises(node, tensors, f"{port} dims .*: not rank 3 with dims >= 1")

    @pytest.mark.parametrize("dims", [[2, 8, 16], [1, 8, 15]])
    def test_a_token_not_1_tokens_k_raises(self, dims: list[int]) -> None:
        node, tensors = _moe()
        tensors[1]["dims"] = dims
        self._raises(
            node, tensors, r"token_tensor_uid dims .* are not \[1, tokens, 16\]"
        )

    @pytest.mark.parametrize("dims", [[2, 2, 1], [2, 1, 2], [3, 1, 1], [1, 1, 1]])
    def test_an_offset_not_a_multiple_of_experts_raises(self, dims) -> None:
        node, tensors = _moe()
        tensors[3]["dims"] = dims
        self._raises(node, tensors, "first_token_offset_tensor_uid dims .* 2 experts")

    @pytest.mark.parametrize("dtype", ["int64", "float", None])
    def test_an_offset_not_int32_raises(self, dtype) -> None:
        node, tensors = _moe()
        tensors[3]["data_type"] = dtype
        self._raises(
            node, tensors, "first_token_offset_tensor_uid dtype .* is not int32"
        )

    @pytest.mark.parametrize(
        "layout, expected",
        [
            ("row_major", "weight_layout:row_major"),
            ("col_major", "weight_layout:col_major"),
        ],
    )
    def test_weight_layout(self, layout: str, expected: str) -> None:
        features = self._extract(weight_layout=layout)
        assert self._group(features, "weight_layout") == {expected}

    def test_col_major_is_cudnn_documented_strides(self) -> None:
        # {K*N, 1, K}: the CPU reference's column path.
        _, tensors = _moe(weight_layout="col_major")
        assert tensors[2]["strides"] == [16 * 32, 1, 16]

    def test_a_weight_with_no_unit_stride_raises(self) -> None:
        node, tensors = _moe()
        tensors[2]["strides"] = [2 * 16 * 32, 64, 2]
        self._raises(node, tensors, "weight_tensor_uid strides .* no unit stride")

    def test_uid_zero_is_a_binding(self) -> None:
        node, tensors = _moe()
        tensors[0] = {**tensors.pop(1), "uid": 0}
        node["inputs"]["token_tensor_uid"] = 0
        assert moe_grouped_matmul.extract(node, tensors) == self.PLAIN

    @pytest.mark.parametrize("port", moe_grouped_matmul.REQUIRED)
    @pytest.mark.parametrize("unbound", ["null", "missing"])
    def test_unbound_required_port_raises(self, port: str, unbound: str) -> None:
        node, tensors = _moe()
        ports = node["outputs" if port == "output_tensor_uid" else "inputs"]
        if unbound == "null":
            ports[port] = None
        else:
            del ports[port]
        self._raises(node, tensors, f"required port {port}")

    def test_ports_at_the_top_of_the_node_are_not_read(self) -> None:
        # Unlike reduction and resample: the serializer nests them.
        node, tensors = _moe()
        node.update(node.pop("inputs"))
        self._raises(node, tensors, "required port token_tensor_uid")


# --------------------------------------------------------------------------
# MoeGroupedMatmulBwdAttributes
# --------------------------------------------------------------------------


def _moe_bwd(
    *,
    experts: int = 2,
    rows: int = 8,
    k: int = 16,
    n: int = 32,
    dtype: str = "float",
    doutput_dtype: str | None = None,
    dweight_dtype: str | None = None,
    dweight_layout: str = "col_major",
    token_virtual: bool = False,
    dweight_virtual: bool = False,
) -> tuple[dict, dict]:
    """A MoeGroupedMatmulBwdAttributes node: ``doutput`` on uid 1, ``token``
    on 2, the offset on 3, ``dweight`` on 4. ``dweight`` has the strides
    the frontend infers unless told. Ports as
    utilities/json/MoeGroupedMatmulBwdAttributes.hpp writes them.
    """

    def tensor(uid: int, dims: list[int], dt: str, virtual: bool = False) -> dict:
        return {
            "uid": uid,
            "dims": dims,
            "strides": _matrix_strides(dims, "row_major"),
            "data_type": dt,
            "virtual": virtual,
        }

    dweight = tensor(4, [experts, k, n], dweight_dtype or dtype, dweight_virtual)
    dweight["strides"] = _matrix_strides(dweight["dims"], dweight_layout)
    tensors = {
        1: tensor(1, [1, rows, n], doutput_dtype or dtype),
        2: tensor(2, [1, rows, k], dtype, token_virtual),
        3: tensor(3, [experts, 1, 1], "int32"),
        4: dweight,
    }
    node = {
        "type": "MoeGroupedMatmulBwdAttributes",
        "inputs": {
            "doutput_tensor_uid": 1,
            "token_tensor_uid": 2,
            "first_token_offset_tensor_uid": 3,
        },
        "outputs": {"dweight_tensor_uid": 4},
    }
    return node, tensors


class TestMoeGroupedMatmulBwdExtract:
    PLAIN = {"dtype:fp32", "dweight_layout:col_major"}

    def _extract(self, **kwargs) -> set[str]:
        return moe_grouped_matmul_bwd.extract(*_moe_bwd(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            moe_grouped_matmul_bwd.extract(node, tensors)

    def test_builder_binds_every_schema_field(self) -> None:
        node, _ = _moe_bwd()
        assert set(batchnorm.bound(node)) == table_fields(
            moe_grouped_matmul_bwd.SCHEMA_FILE, moe_grouped_matmul_bwd.SCHEMA_TABLE
        )

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"dtype": "half", "dweight_dtype": "float", "dweight_layout": "row_major"},
            {"dtype": "bfloat16"},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {f.id for f in moe_grouped_matmul_bwd.FEATURES}

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "dtype, expected",
        [("float", "dtype:fp32"), ("half", "dtype:fp16"), ("bfloat16", "dtype:bf16")],
    )
    def test_dtype(self, dtype: str, expected: str) -> None:
        features = self._extract(dtype=dtype)
        assert features == self.PLAIN - {"dtype:fp32"} | {expected}

    @pytest.mark.parametrize(
        "dtype, dweight_dtype",
        [
            ("half", "float"),
            ("bfloat16", "float"),
            ("float", "half"),
            ("float", "bfloat16"),
        ],
    )
    def test_mixed_output(self, dtype: str, dweight_dtype: str) -> None:
        # The CPU reference's pairs: the other dtype fp32.
        features = self._extract(dtype=dtype, dweight_dtype=dweight_dtype)
        assert "precision:mixed_output" in features

    @pytest.mark.parametrize("virtual", ["token_virtual", "dweight_virtual"])
    def test_a_virtual_tensor_is_not_mixed(self, virtual: str) -> None:
        features = self._extract(dtype="half", dweight_dtype="float", **{virtual: True})
        assert features == self.PLAIN - {"dtype:fp32"} | {"dtype:fp16"}

    @pytest.mark.parametrize("dweight_virtual", [False, True])
    @pytest.mark.parametrize(
        "dtype, dweight_dtype", [("half", "bfloat16"), ("bfloat16", "half")]
    )
    def test_a_dweight_pair_with_no_plan_raises(
        self, dtype: str, dweight_dtype: str, dweight_virtual: bool
    ) -> None:
        # The signature key reads the dtype of a virtual dweight too.
        node, tensors = _moe_bwd(
            dtype=dtype, dweight_dtype=dweight_dtype, dweight_virtual=dweight_virtual
        )
        self._raises(node, tensors, "dweight_tensor_uid dtype .* has no plan")

    @pytest.mark.parametrize(
        "dtype, doutput_dtype", [("half", "float"), ("float", "half")]
    )
    def test_a_doutput_not_of_token_dtype_raises(
        self, dtype: str, doutput_dtype: str
    ) -> None:
        node, tensors = _moe_bwd(dtype=dtype, doutput_dtype=doutput_dtype)
        self._raises(node, tensors, "doutput_tensor_uid dtype .* is not that of")

    @pytest.mark.parametrize(
        "kwargs, port",
        [
            ({"dtype": "int8"}, "token_tensor_uid"),
            ({"doutput_dtype": "double"}, "doutput_tensor_uid"),
            ({"dweight_dtype": "int32"}, "dweight_tensor_uid"),
        ],
    )
    def test_an_uncatalogued_dtype_raises(self, kwargs: dict, port: str) -> None:
        node, tensors = _moe_bwd(**kwargs)
        self._raises(node, tensors, f"{port} dtype .* is not catalogued")

    @pytest.mark.parametrize(
        "layout, expected",
        [
            ("row_major", "dweight_layout:row_major"),
            ("col_major", "dweight_layout:col_major"),
        ],
    )
    def test_dweight_layout(self, layout: str, expected: str) -> None:
        features = self._extract(dweight_layout=layout)
        assert features == {"dtype:fp32", expected}

    def test_col_major_is_the_frontend_inferred_strides(self) -> None:
        # MoeGroupedMatmulBwdNode: [K*N, 1, K] for [E, K, N].
        _, tensors = _moe_bwd()
        assert tensors[4]["strides"] == [16 * 32, 1, 16]

    def test_a_dweight_with_no_unit_stride_has_no_layout(self) -> None:
        # The CPU reference reads any strides: no layout, not an error.
        node, tensors = _moe_bwd()
        tensors[4]["strides"] = [1, 2 * 32, 2]
        assert moe_grouped_matmul_bwd.extract(node, tensors) == {"dtype:fp32"}

    @pytest.mark.parametrize(
        "uid, port",
        [
            (1, "doutput_tensor_uid"),
            (2, "token_tensor_uid"),
            (3, "first_token_offset_tensor_uid"),
            (4, "dweight_tensor_uid"),
        ],
    )
    @pytest.mark.parametrize("dims", [[8, 16], [1, 1, 8, 16], [1, 0, 16], []])
    def test_a_tensor_not_rank_3_raises(self, uid, port, dims) -> None:
        node, tensors = _moe_bwd()
        tensors[uid]["dims"] = dims
        self._raises(node, tensors, f"{port} dims .*: not rank 3 with dims >= 1")

    @pytest.mark.parametrize(
        "uid, port",
        [
            (1, "doutput_tensor_uid"),
            (2, "token_tensor_uid"),
            (3, "first_token_offset_tensor_uid"),
            (4, "dweight_tensor_uid"),
        ],
    )
    @pytest.mark.parametrize("ragged_uid", [0, 7])
    def test_a_ragged_tensor_raises(self, uid, port, ragged_uid: int) -> None:
        # CHECK_NO_RAGGED_TENSORS; uid 0 is a ragged offset too.
        node, tensors = _moe_bwd()
        tensors[uid]["ragged_offset_tensor_uid"] = ragged_uid
        self._raises(node, tensors, f"{port} is ragged")

    def test_a_null_ragged_offset_is_not_ragged(self) -> None:
        node, tensors = _moe_bwd()
        tensors[2]["ragged_offset_tensor_uid"] = None
        assert moe_grouped_matmul_bwd.extract(node, tensors) == self.PLAIN

    def test_a_token_not_1_rows_k_raises(self) -> None:
        node, tensors = _moe_bwd()
        tensors[2]["dims"] = [2, 8, 16]
        self._raises(node, tensors, r"token_tensor_uid dims .* are not \[1, rows, K\]")

    @pytest.mark.parametrize("dims", [[2, 8, 32], [1, 7, 32], [1, 9, 32]])
    def test_a_doutput_not_1_token_rows_raises(self, dims: list[int]) -> None:
        node, tensors = _moe_bwd()
        tensors[1]["dims"] = dims
        self._raises(node, tensors, r"doutput_tensor_uid dims .* are not \[1, 8, N\]")

    @pytest.mark.parametrize("dims", [[2, 2, 1], [2, 1, 2]])
    def test_an_offset_not_e_1_1_raises(self, dims: list[int]) -> None:
        node, tensors = _moe_bwd()
        tensors[3]["dims"] = dims
        self._raises(
            node, tensors, r"first_token_offset_tensor_uid dims .* are not \[E, 1, 1\]"
        )

    @pytest.mark.parametrize("dtype", ["int64", "float", None])
    def test_an_offset_not_int32_raises(self, dtype) -> None:
        node, tensors = _moe_bwd()
        tensors[3]["data_type"] = dtype
        self._raises(
            node, tensors, "first_token_offset_tensor_uid dtype .* is not int32"
        )

    @pytest.mark.parametrize(
        "dims", [[1, 16, 32], [4, 16, 32], [2, 15, 32], [2, 16, 31], [2, 32, 16]]
    )
    def test_a_dweight_not_e_k_n_raises(self, dims: list[int]) -> None:
        node, tensors = _moe_bwd()
        tensors[4]["dims"] = dims
        self._raises(node, tensors, r"dweight_tensor_uid dims .* are not \[2, 16, 32\]")

    @pytest.mark.parametrize("experts", [1, 3, 4])
    def test_any_expert_count(self, experts: int) -> None:
        # The offset is E, not a multiple of it as for the forward.
        assert self._extract(experts=experts) == self.PLAIN

    def test_an_offset_of_a_multiple_of_experts_raises(self) -> None:
        # The forward's batch: the backward has none.
        node, tensors = _moe_bwd()
        tensors[3]["dims"] = [4, 1, 1]
        self._raises(node, tensors, r"dweight_tensor_uid dims .* are not \[4, 16, 32\]")

    def test_uid_zero_is_a_binding(self) -> None:
        node, tensors = _moe_bwd()
        tensors[0] = {**tensors.pop(1), "uid": 0}
        node["inputs"]["doutput_tensor_uid"] = 0
        assert moe_grouped_matmul_bwd.extract(node, tensors) == self.PLAIN

    @pytest.mark.parametrize("port", moe_grouped_matmul_bwd.REQUIRED)
    @pytest.mark.parametrize("unbound", ["null", "missing"])
    def test_unbound_required_port_raises(self, port: str, unbound: str) -> None:
        node, tensors = _moe_bwd()
        ports = node["outputs" if port == "dweight_tensor_uid" else "inputs"]
        if unbound == "null":
            ports[port] = None
        else:
            del ports[port]
        self._raises(node, tensors, f"required port {port}")

    def test_ports_at_the_top_of_the_node_are_not_read(self) -> None:
        node, tensors = _moe_bwd()
        node.update(node.pop("inputs"))
        self._raises(node, tensors, "required port doutput_tensor_uid")


# --------------------------------------------------------------------------
# CustomOpAttributes
# --------------------------------------------------------------------------


def _custom_op(
    *,
    inputs: tuple = (1, 2),
    outputs: tuple = (3,),
    data: tuple = (0xDE, 0xAD),
    op_id: object = "test.op",
) -> tuple[dict, dict]:
    """A CustomOpAttributes node over tensors 0 to 4, its fields at the top
    of the node as utilities/json/CustomOpAttributes.hpp writes them.
    """
    tensors = {
        uid: {"uid": uid, "dims": [2, 3], "strides": [3, 1], "data_type": "float"}
        for uid in range(5)
    }
    node = {
        "type": "CustomOpAttributes",
        "custom_op_id": op_id,
        "input_tensor_uids": list(inputs),
        "output_tensor_uids": list(outputs),
        "data": list(data),
    }
    return node, tensors


class TestCustomOpExtract:
    PLAIN = {"inputs:multi", "outputs:one", "data:present"}

    def _extract(self, **kwargs) -> set[str]:
        return custom_op.extract(*_custom_op(**kwargs))

    def _raises(self, node: dict, tensors: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            custom_op.extract(node, tensors)

    def test_builder_writes_every_schema_field(self) -> None:
        node, _ = _custom_op()
        assert set(node) - {"type"} == table_fields(
            custom_op.SCHEMA_FILE, custom_op.SCHEMA_TABLE
        )

    def test_registry_maps_the_node_type_here(self) -> None:
        assert registry.op_module("CustomOpAttributes") is custom_op

    def test_every_feature_is_reachable(self) -> None:
        reached = self._extract()
        for kwargs in (
            {"inputs": (), "outputs": (), "data": ()},
            {"inputs": (1,), "outputs": (3, 4)},
        ):
            reached |= self._extract(**kwargs)
        assert reached == {f.id for f in custom_op.FEATURES}

    def test_plain(self) -> None:
        assert self._extract() == self.PLAIN

    @pytest.mark.parametrize(
        "count, value", [(0, "none"), (1, "one"), (2, "multi"), (3, "multi")]
    )
    @pytest.mark.parametrize("port", ["inputs", "outputs"])
    def test_arity(self, port: str, count: int, value: str) -> None:
        # One port varies; the other stays at a count none of these share.
        uids = tuple(range(1, count + 1))
        other = "outputs" if port == "inputs" else "inputs"
        features = self._extract(**{port: uids, other: (4,)})
        assert f"{port}:{value}" in features
        assert f"{other}:one" in features

    def test_repeated_input_counts_each_time(self) -> None:
        # x * x: one tensor, read twice, is two inputs.
        assert "inputs:multi" in self._extract(inputs=(1, 1))

    def test_uid_zero_is_a_tensor(self) -> None:
        assert "inputs:one" in self._extract(inputs=(0,))
        assert "outputs:one" in self._extract(outputs=(0,))

    @pytest.mark.parametrize(
        "data, value",
        [((), "empty"), ((0,), "present"), ((255,), "present"), ((1, 2), "present")],
    )
    def test_data(self, data: tuple, value: str) -> None:
        assert f"data:{value}" in self._extract(data=data)

    def test_undotted_id_is_taken(self) -> None:
        # The schema documents "<plugin>.<operation>"; nothing enforces it.
        assert self._extract(op_id="rope") == self.PLAIN

    @pytest.mark.parametrize("op_id", ["", None, 5, ["test.op"]])
    def test_bad_custom_op_id_raises(self, op_id: object) -> None:
        node, tensors = _custom_op(op_id=op_id)
        self._raises(node, tensors, "custom_op_id .* is not a non-empty string")

    def test_missing_custom_op_id_raises(self) -> None:
        node, tensors = _custom_op()
        del node["custom_op_id"]
        self._raises(node, tensors, "custom_op_id None is not a non-empty string")

    @pytest.mark.parametrize("port", [custom_op.INPUTS, custom_op.OUTPUTS])
    @pytest.mark.parametrize("value", ["missing", None, 1, {"x": 1}, "12"])
    def test_uid_list_that_is_not_a_list_raises(self, port: str, value: object) -> None:
        node, tensors = _custom_op()
        if value == "missing":
            del node[port]
        else:
            node[port] = value
        self._raises(node, tensors, f"{port} .* is not a list of uids")

    @pytest.mark.parametrize("port", [custom_op.INPUTS, custom_op.OUTPUTS])
    @pytest.mark.parametrize("uid", [True, 1.0, "1", None])
    def test_uid_that_is_not_an_int_raises(self, port: str, uid: object) -> None:
        # True and 1.0 would find tensor 1 if looked up.
        node, tensors = _custom_op()
        node[port] = [2, uid]
        self._raises(node, tensors, f"{port} uid .* is not an int")

    @pytest.mark.parametrize("port", [custom_op.INPUTS, custom_op.OUTPUTS])
    @pytest.mark.parametrize("uid", [5, -1])
    def test_uid_not_in_the_graph_raises(self, port: str, uid: int) -> None:
        node, tensors = _custom_op()
        node[port] = [2, uid]
        self._raises(node, tensors, f"{port} uid {uid} is not a tensor of the graph")

    @pytest.mark.parametrize("port", [custom_op.INPUTS, custom_op.OUTPUTS])
    def test_uid_zero_not_in_the_graph_raises(self, port: str) -> None:
        # Valid only as a binding; a graph without tensor 0 has no uid 0.
        node, tensors = _custom_op()
        del tensors[0]
        node[port] = [0]
        self._raises(node, tensors, f"{port} uid 0 is not a tensor of the graph")

    def test_lists_under_inputs_and_outputs_are_not_read(self) -> None:
        node, tensors = _custom_op()
        node["inputs"] = {custom_op.INPUTS: node.pop(custom_op.INPUTS)}
        self._raises(node, tensors, "input_tensor_uids None is not a list of uids")

    # "" has no character to fail the byte check: only the list check rejects it.
    @pytest.mark.parametrize("data", ["missing", None, "", "dead", 7, {"0": 1}])
    def test_data_that_is_not_a_list_raises(self, data: object) -> None:
        node, tensors = _custom_op()
        if data == "missing":
            del node["data"]
        else:
            node["data"] = data
        self._raises(node, tensors, "data .* is not a list of bytes")

    @pytest.mark.parametrize("byte", [256, -1, True, 1.0, "a", None])
    def test_data_that_is_not_bytes_raises(self, byte: object) -> None:
        node, tensors = _custom_op(data=(0, byte))
        self._raises(node, tensors, "data .* is not a list of bytes")


# --------------------------------------------------------------------------
# GdnAttributes and KdaAttributes
# --------------------------------------------------------------------------

_LA_INPUTS = (
    "q_tensor_uid",
    "k_tensor_uid",
    "v_tensor_uid",
    "g_tensor_uid",
    "beta_tensor_uid",
    "cu_seqlens_tensor_uid",
    "initial_state_tensor_uid",
    "a_log_tensor_uid",
    "dt_bias_tensor_uid",
    "state_indices_tensor_uid",
)
_LA_OUTPUTS = ("o_tensor_uid", "final_state_tensor_uid", "state_checkpoints_tensor_uid")
_LA_REQUIRED = {
    "q_tensor_uid": 1,
    "k_tensor_uid": 2,
    "v_tensor_uid": 3,
    "g_tensor_uid": 4,
    "beta_tensor_uid": 5,
    "cu_seqlens_tensor_uid": 6,
    "o_tensor_uid": 7,
}


def _linear_attention(
    node_type: str,
    *,
    dtype: str = "half",
    ports: tuple[str, ...] = (),
    **attributes,
) -> tuple[dict, dict]:
    """A GdnAttributes / KdaAttributes node as the serializers write it: every
    port under ``inputs`` / ``outputs`` (``null`` unless required or in
    ``ports``), every scalar under ``attributes``. Only ``q``'s dtype is read.
    """
    uids = {**_LA_REQUIRED, **{port: 10 + i for i, port in enumerate(ports)}}
    scalars = {
        "scale": None,
        "output_final_state": False,
        "use_qk_l2norm": False,
        "checkpoint_every_n_tokens": 0,
        "use_beta_sigmoid": False,
        "allow_neg_eigval": False,
        "safe_gate": False,
        "gate_domain": "log",
        "batch_invariant": False,
        "overwrite_initial_state": False,
    }
    if node_type == "KdaAttributes":
        scalars["gate_lower_bound"] = None
    node = {
        "type": node_type,
        "inputs": {port: uids.get(port) for port in _LA_INPUTS},
        "outputs": {port: uids.get(port) for port in _LA_OUTPUTS},
        "attributes": {**scalars, **attributes},
    }
    tensors = {
        uid: {"uid": uid, "dims": [1], "data_type": "half"} for uid in uids.values()
    }
    tensors[1]["data_type"] = dtype
    return node, tensors


@pytest.mark.parametrize("module", [gdn, kda], ids=["gdn", "kda"])
class TestLinearAttentionExtract:
    def _build(self, module, **kwargs) -> tuple[dict, dict]:
        return _linear_attention(module.NODE_TYPE, **kwargs)

    def _extract(self, module, **kwargs) -> set[str]:
        return module.extract(*self._build(module, **kwargs))

    def _raises(self, module, match: str, **kwargs) -> None:
        with pytest.raises(FeatureError, match=match):
            self._extract(module, **kwargs)

    def test_builder_names_every_schema_field(self, module) -> None:
        node, _ = self._build(module)
        written = set(node["inputs"]) | set(node["outputs"]) | set(node["attributes"])
        assert written == table_fields(module.SCHEMA_FILE, module.SCHEMA_TABLE)

    def test_plain_graph(self, module) -> None:
        assert self._extract(module) == {
            "dtype:fp16",
            "gate:log",
            "beta:direct",
            "scale:default",
        }

    def test_every_feature_is_reachable(self, module) -> None:
        lower_bound = {"gate_lower_bound": -3.0} if module is kda else {}
        graphs = (
            {"dtype": "float", "gate_domain": "linear", "scale": 0.5},
            {"dtype": "bfloat16", "use_beta_sigmoid": True, "use_qk_l2norm": True},
            {
                "use_beta_sigmoid": True,
                "allow_neg_eigval": True,
                "batch_invariant": True,
            },
            {
                "safe_gate": True,
                "ports": ("a_log_tensor_uid", "dt_bias_tensor_uid"),
                **lower_bound,
            },
            {
                "ports": (
                    "initial_state_tensor_uid",
                    "final_state_tensor_uid",
                    "state_checkpoints_tensor_uid",
                ),
                "output_final_state": True,
                "checkpoint_every_n_tokens": 64,
                "overwrite_initial_state": True,
            },
            {
                "ports": ("initial_state_tensor_uid", "state_indices_tensor_uid"),
            },
        )
        reached: set[str] = set()
        for kwargs in graphs:
            reached |= self._extract(module, **kwargs)
        assert reached == {f.id for f in module.FEATURES}

    def test_safe_gate_replaces_the_domain_feature(self, module) -> None:
        features = self._extract(module, safe_gate=True)
        assert "gate:safe" in features
        assert not features & {"gate:log", "gate:linear"}

    def test_state_modes_combine(self, module) -> None:
        features = self._extract(
            module,
            ports=("initial_state_tensor_uid", "final_state_tensor_uid"),
            output_final_state=True,
            overwrite_initial_state=True,
        )
        assert {"state:initial", "state:final", "state:in_place"} <= features

    @pytest.mark.parametrize("port", sorted(_LA_REQUIRED))
    def test_unbound_required_port_raises(self, module, port: str) -> None:
        node, tensors = self._build(module)
        for section in ("inputs", "outputs"):
            if port in node[section]:
                node[section][port] = None
        with pytest.raises(FeatureError, match=f"required port {port}"):
            module.extract(node, tensors)

    def test_uncatalogued_dtype_raises(self, module) -> None:
        self._raises(module, "dtype", dtype="double")

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"gate_domain": "exp"}, "gate_domain"),
            ({"safe_gate": True, "gate_domain": "linear"}, "invalid with safe_gate"),
            ({"ports": ("a_log_tensor_uid",)}, "requires safe_gate"),
            ({"ports": ("dt_bias_tensor_uid",)}, "requires safe_gate"),
            ({"allow_neg_eigval": True}, "requires use_beta_sigmoid"),
            ({"output_final_state": True}, "disagree"),
            ({"ports": ("final_state_tensor_uid",)}, "disagree"),
            ({"checkpoint_every_n_tokens": 8}, "disagree"),
            ({"ports": ("state_checkpoints_tensor_uid",)}, "disagree"),
            ({"checkpoint_every_n_tokens": -1}, "negative"),
            ({"ports": ("state_indices_tensor_uid",)}, "requires initial_state"),
            (
                {
                    "ports": (
                        "initial_state_tensor_uid",
                        "state_indices_tensor_uid",
                        "state_checkpoints_tensor_uid",
                    ),
                    "checkpoint_every_n_tokens": 8,
                },
                "cannot combine",
            ),
            ({"overwrite_initial_state": True}, "requires an initial and a final"),
            (
                {
                    "ports": ("initial_state_tensor_uid",),
                    "overwrite_initial_state": True,
                },
                "requires an initial and a final",
            ),
        ],
    )
    def test_graph_the_api_would_reject_raises(
        self, module, kwargs: dict, match: str
    ) -> None:
        self._raises(module, match, **kwargs)


class TestKdaGateLowerBound:
    def test_needs_safe_gate(self) -> None:
        node, tensors = _linear_attention("KdaAttributes", gate_lower_bound=-2.0)
        with pytest.raises(FeatureError, match="gate_lower_bound requires safe_gate"):
            kda.extract(node, tensors)

    def test_is_a_feature_with_safe_gate(self) -> None:
        node, tensors = _linear_attention(
            "KdaAttributes", safe_gate=True, gate_lower_bound=-2.0
        )
        assert "gate_lower_bound:set" in kda.extract(node, tensors)


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
        counted = sum(op.graphs for op in coverage.ops.values())
        assert units
        assert counted >= len(units)

    def test_feature_matrix_knows_every_claimed_arch(
        self, units: list[ClaimUnit]
    ) -> None:
        # An arch missing from GPUS would fail the render; catch it here first.
        errors: list[str] = []
        claims = compute_feature_claims(units, errors)
        assert errors == []
        assert claims.targets

    def test_committed_overview_stays_renderable(self, units: list[ClaimUnit]) -> None:
        """GitHub stops rendering markdown well past a few hundred KB."""
        document = render_feature_matrix(compute_feature_claims(units))
        size = len(document.encode("utf-8"))
        assert (
            size < 500_000
        ), f"overview is {size} bytes; split it before it stops rendering on GitHub"

    def test_committed_overview_is_current(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """SUPPORT_MATRIX_OVERVIEW.md matches a fresh render of the committed tree.

        The check-support-matrix-overview hook cannot guard this on its own:
        the top-level pre-commit exclude hides integration-test-bundles/ from
        every hook, so a commit that changes only bundle data never runs it.
        This test runs on any change under dnn-providers/, whichever file in
        the tree moved -- a sidecar, a sweep manifest, a whole bundle.
        """
        status = feature_matrix_main(
            ["--output", str(DEFAULT_OVERVIEW_PATH), "--check"]
        )
        assert status == 0, capsys.readouterr().err
