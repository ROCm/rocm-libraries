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
from support_features import (
    Feature,
    FeatureError,
    pointwise,
    registry,
    sdpa,
    sdpa_backward,
)
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

    def test_pending_type_has_no_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Its own PENDING, so the test outlives the real list emptying.
        monkeypatch.setattr(registry, "PENDING", frozenset({"LaterAttributes"}))
        assert registry.op_module("LaterAttributes") is None

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
PENDING_TYPE = "CustomOpAttributes"


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
    # A fixed PENDING, not the real one, which shrinks with every op catalogued.
    monkeypatch.setattr(registry, "PENDING", frozenset({PENDING_TYPE}))


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
        assert set(coverage.pending_graphs) == {PENDING_TYPE}
        assert f"{PENDING_TYPE} (0)" in render_markdown(coverage)

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
    dtype="bfloat16",
    strides=_bhsd_strides,
    inputs=None,
    outputs=None,
    **attrs,
) -> tuple[dict, dict]:
    """An SDPA node and its tensors, with every port and attribute unset."""
    tensors = {
        uid: {
            "uid": uid,
            "dims": list(dims),
            "strides": strides(list(dims)),
            "data_type": dtype,
        }
        for uid, dims in ((1, q_dims), (2, k_dims))
    }
    node = {
        "type": "SdpaAttributes",
        "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 2}
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
        "kwargs, match",
        [
            ({"q_dims": (2, 8, 128, 96)}, "head dim 96"),
            ({"dtype": "float"}, "dtype 'float'"),
            (
                {"q_dims": (8, 128, 128), "strides": lambda dims: [1] * len(dims)},
                "rank",
            ),
        ],
    )
    def test_uncatalogued_value_raises(self, kwargs: dict, match: str) -> None:
        with pytest.raises(FeatureError, match=match):
            self._extract(**kwargs)

    def test_fp8_dtype_label(self) -> None:
        assert self._group("dtype", dtype="fp8_e4m3") == "dtype:fp8e4m3"

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
            {"dtype": "half", "q_dims": (2, 8, 1, 64), "k_dims": (2, 1, 128, 64)},
            {
                "dtype": "fp8_e4m3",
                "q_dims": (2, 8, 64, 192),
                "k_dims": (2, 2, 128, 192),
            },
            {"dtype": "fp8_e5m2", "strides": _bshd_strides, "causal_mask": True},
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
            {"strides": _bshd_strides, "causal_mask": True},
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
