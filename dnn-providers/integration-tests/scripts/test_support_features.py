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
    convolution,
    convolution_bwd,
    convolution_fwd,
    convolution_wrw,
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
    **params,
) -> tuple[dict, dict]:
    """A conv node: activation on uid 1, filter on uid 2, response on uid 3.

    ``layouts`` / ``dtypes`` override ``layout`` / ``dtype`` per role;
    ``virtual`` lists the roles whose tensor is virtual. Parameters default to
    no padding, unit stride and unit dilation.
    """
    dims = {
        "activation": [2, channels, *spatial],
        "filter": [filters, channels // groups, *(3 for _ in spatial)],
        "response": [2, filters, *spatial],
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
    zeros, ones = [0] * len(spatial), [1] * len(spatial)
    parameters = {
        "conv_mode": conv_mode,
        "pre_padding": zeros,
        "post_padding": zeros,
        "stride": ones,
        "dilation": ones,
    } | params
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

    PLAIN = {"dtype:fp32", "rank:2d", "layout:nchw", "mode:cross_correlation"}

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
    PLAIN = {"dtype:fp32", "rank:2d", "layout:nchw", "mode:cross_correlation"}

    def _extract(self, **kwargs) -> set[str]:
        return convolution_fwd.extract(*_conv(convolution_fwd, **kwargs))

    def test_plain_graph(self) -> None:
        assert self._extract() == self.PLAIN

    def test_every_schema_mode_is_catalogued(self) -> None:
        modes = _schema_enum("convolution_common.fbs", "ConvMode")
        assert set(convolution.MODES) == modes - {"UNSET"}

    @pytest.mark.parametrize(
        "spatial, expected",
        [((8,), "rank:1d"), ((8, 8), "rank:2d"), ((4, 8, 8), "rank:3d")],
    )
    @pytest.mark.parametrize("layout", convolution.LAYOUTS)
    def test_rank_and_layout(self, spatial, expected: str, layout: str) -> None:
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
        features = self._extract(channels=1, filters=1, spatial=(1, 1))
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

    @pytest.mark.parametrize("spatial, n", [((), 0), ((2, 2, 2, 2), 4)])
    def test_uncatalogued_rank_raises(self, spatial, n: int) -> None:
        with pytest.raises(FeatureError, match=f"{n} spatial dims"):
            self._extract(spatial=spatial)


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
