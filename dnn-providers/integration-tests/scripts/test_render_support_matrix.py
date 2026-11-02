#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for render_support_matrix.py and the two modules it drives,
support_matrix_units.py and support_matrix_markdown.py.

Most tests build a two-bundle tree in ``tmp_path`` so the expected output is
small enough to assert on exactly. ``TestRealBundleTree`` then runs against the
committed bundles and re-counts every overview cell by a route that shares no
code with the renderer -- the renderer aggregates through ``ClaimUnit``, the
crosscheck reads the sidecars straight off disk. Agreement on all cells means
the aggregation is not quietly dropping or double-counting claims.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from render_support_matrix import (
    DEFAULT_BUNDLES_DIR,
    DEFAULT_MATRIX_PATH,
    main,
)
from support_matrix_markdown import (
    FULL,
    LEGEND_OVERVIEW,
    NONE,
    PARTIAL,
    REGEN_COMMAND,
    REGEN_OVERVIEW_COMMAND,
    render_markdown,
)
from support_matrix_units import (
    NO_LAYOUT,
    ClaimUnit,
    collect_units,
    dtypes_of,
    layout_from_name,
    layout_from_strides,
    placeholder_field,
    shape_tags_of,
    variant_of,
)

GFX942 = ("gfx942", "linux")
GFX90A = ("gfx90a", "linux")
# Seconds a CLI under test may run; rendering the real tree takes a few.
SUBPROCESS_TIMEOUT = 120


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _tensor(uid: int, dims: list[int], strides: list[int] | None = None) -> dict:
    tensor = {"uid": uid, "dims": dims, "data_type": "half"}
    if strides is not None:
        tensor["strides"] = strides
    return tensor


def _graph(nodes: list[dict] | None = None, **overrides) -> dict:
    graph = {
        "io_data_type": "half",
        "compute_data_type": "float",
        # Batch 1, unit stride, no padding: the untagged baseline, so tests
        # about other axes are not perturbed by a shape tag they never asked
        # for. Tests that want a tag set the shape themselves.
        "tensors": [_tensor(1, [1, 32, 16, 16], [8192, 256, 16, 1])],
        # The primary node type is what the family is read off, so the default
        # has to agree with the ``Conv`` directories these fixtures build under.
        "nodes": nodes if nodes is not None else [{"type": "ConvAttributes"}],
    }
    graph.update(overrides)
    return graph


def _single_bundle(
    root: Path,
    relative: str,
    name: str,
    graph: dict | None = None,
    claims: dict | None = None,
) -> Path:
    directory = root / relative
    _write_json(directory / f"{name}.json", graph if graph is not None else _graph())
    if claims is not None:
        _write_json(
            directory / f"{name}.support.json", {"version": 1, "claims": claims}
        )
    return directory


def _sweep_bundle(
    root: Path,
    relative: str,
    cases: list[dict],
    template: dict | None = None,
    claims: dict | None = None,
) -> Path:
    directory = root / relative
    _write_json(
        directory / "graph.template.json",
        template if template is not None else _graph(),
    )
    _write_json(directory / "sweep.json", {"cases": cases})
    if claims is not None:
        _write_json(directory / "support.json", {"version": 1, "claims": claims})
    return directory


@pytest.fixture()
def bundle_root(tmp_path: Path) -> Path:
    return tmp_path


# --------------------------------------------------------------------------
# Metadata extraction
# --------------------------------------------------------------------------


class TestLayoutInference:
    @pytest.mark.parametrize(
        ("case_id", "expected"),
        [
            ("8_32_16_16_fp16_nchw", "NCHW"),
            ("8_32_16_16_fp16_nhwc", "NHWC"),
            ("bn_fwd_ncl", "NCL"),
            ("attn_bshd_causal", "BSHD"),
        ],
    )
    def test_token_in_case_id_wins(self, case_id: str, expected: str) -> None:
        assert layout_from_name(case_id) == expected

    def test_last_token_wins(self) -> None:
        """A path can mention several; the most specific one comes last."""
        assert layout_from_name("nchw/variants/case_nhwc") == "NHWC"

    def test_no_token_yields_none(self) -> None:
        assert layout_from_name("8_32_16_16_fp16") is None

    def test_channel_first_strides(self) -> None:
        graph = _graph(tensors=[_tensor(1, [8, 32, 16, 16], [8192, 256, 16, 1])])
        assert layout_from_strides(graph) == "NCHW"

    def test_channel_last_strides(self) -> None:
        graph = _graph(tensors=[_tensor(1, [8, 32, 16, 16], [8192, 1, 512, 32])])
        assert layout_from_strides(graph) == "NHWC"

    def test_rank_selects_the_label_family(self) -> None:
        graph = _graph(tensors=[_tensor(1, [8, 32, 16], [512, 16, 1])])
        assert layout_from_strides(graph) == "NCL"

    def test_unranked_layout_is_not_none_sentinel(self, bundle_root: Path) -> None:
        """A rank-2 graph has no NCHW-family layout; that is not "unsupported".

        NO_LAYOUT has to stay distinct from NONE or a cell reads
        ``✅ NCHW, —``, which looks like a partial result.
        """
        graph = _graph(tensors=[_tensor(1, [128, 256], [256, 1])])
        _single_bundle(
            bundle_root,
            "quick/Matmul/Default",
            "Mm",
            graph,
            {"E": {"gfx942": ["linux"]}},
        )
        units = collect_units(bundle_root)
        assert [u.layout for u in units] == [NO_LAYOUT]
        assert NO_LAYOUT != NONE


class TestVariant:
    def test_single_node_is_bare(self) -> None:
        assert variant_of(_graph(), []) == "(bare)"

    def test_fusion_lists_trailing_nodes(self) -> None:
        graph = _graph(
            nodes=[
                {"type": "ConvFpropAttributes"},
                {"type": "PointwiseAttributes", "inputs": {"operation": "relu_fwd"}},
            ]
        )
        assert variant_of(graph, []) == " + Pointwise:RELU_FWD"

    def test_relu_clips_are_named(self) -> None:
        graph = _graph(
            nodes=[
                {"type": "ConvFpropAttributes"},
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "relu_fwd",
                        "relu_lower_clip_slope": 0.01,
                    },
                },
            ]
        )
        assert variant_of(graph, []) == " + Pointwise:RELU_FWD[lower_clip_slope]"

    def test_scenario_tags_replace_bare(self) -> None:
        # A tag already says the graph is a single node of some flavour, so
        # "[causal](bare)" would be saying it twice.
        assert variant_of(_graph(), ["[causal]"]) == "[causal]"

    def test_scenario_tags_prefix_a_fusion_chain(self) -> None:
        graph = _graph(
            nodes=[
                {"type": "ConvFpropAttributes"},
                {"type": "PointwiseAttributes", "inputs": {"operation": "relu_fwd"}},
            ]
        )
        assert variant_of(graph, ["[causal]"]) == "[causal] + Pointwise:RELU_FWD"


class TestShapeTags:
    """Tags read off shapes and parameters rather than off the bundle path.

    The fusion chain alone cannot separate a strided grouped convolution from a
    plain one, so an engine that supports the second and refuses the first has
    nowhere to say so. These tags give the refusal its own row.
    """

    def test_batch_one_is_untagged(self) -> None:
        assert shape_tags_of(_graph()) == set()

    def test_batch_above_one_is_multi_batch(self) -> None:
        graph = _graph(tensors=[_tensor(1, [8, 32, 16, 16], [8192, 256, 16, 1])])
        assert shape_tags_of(graph) == {"multi_batch"}

    def test_rank_two_has_no_batch_axis(self) -> None:
        # Dim 0 of a matmul operand is a row count, not a batch.
        graph = _graph(tensors=[_tensor(1, [64, 32], [32, 1])])
        assert shape_tags_of(graph) == set()

    def test_stride_dilation_and_padding_are_read_off_parameters(self) -> None:
        graph = _graph(
            nodes=[
                {
                    "type": "ConvFpropAttributes",
                    "parameters": {
                        "stride": [2, 2],
                        "dilation": [2, 1],
                        "pre_padding": [1, 1],
                        "post_padding": [0, 0],
                    },
                }
            ]
        )
        assert shape_tags_of(graph) == {"stride", "dilation", "padding"}

    def test_neutral_parameters_are_not_tags(self) -> None:
        graph = _graph(
            nodes=[
                {
                    "type": "ConvFpropAttributes",
                    "parameters": {
                        "stride": [1, 1],
                        "dilation": [1, 1],
                        "pre_padding": [0, 0],
                        "post_padding": [0, 0],
                    },
                }
            ]
        )
        assert shape_tags_of(graph) == set()

    def test_grouped_is_a_filter_spanning_fewer_channels(self) -> None:
        graph = _graph(
            tensors=[
                _tensor(1, [1, 32, 16, 16], [8192, 256, 16, 1]),
                _tensor(2, [64, 8, 3, 3], [72, 9, 3, 1]),
            ],
            nodes=[
                {
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                }
            ],
        )
        assert shape_tags_of(graph) == {"grouped"}

    def test_ungrouped_convolution_carries_no_tag(self) -> None:
        graph = _graph(
            tensors=[
                _tensor(1, [1, 32, 16, 16], [8192, 256, 16, 1]),
                # Kept smaller than x in volume so x stays the representative
                # tensor and the batch axis read below is the input's.
                _tensor(2, [16, 32, 3, 3], [288, 9, 3, 1]),
            ],
            nodes=[
                {
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                }
            ],
        )
        assert shape_tags_of(graph) == set()

    # Data-grad and weight-grad graphs as quick/ConvolutionBwd/Default and
    # quick/ConvolutionWrw/Default wire them: the activation or the filter is
    # the node's output, so a lookup through inputs alone finds nothing. The
    # filter is kept smaller than the activations in volume so it never
    # becomes the representative tensor and reads as multi_batch.
    def _bwd(self, dx_channels: int, w_channels: int) -> dict:
        return _graph(
            tensors=[
                _tensor(0, [1, dx_channels, 32, 32], [dx_channels * 1024, 1024, 32, 1]),
                _tensor(1, [1, 16, 32, 32], [16384, 1024, 32, 1]),
                _tensor(2, [16, w_channels, 3, 3], [w_channels * 9, 9, 3, 1]),
            ],
            nodes=[
                {
                    "type": "ConvolutionBwdAttributes",
                    "inputs": {"dy_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"dx_tensor_uid": 0},
                }
            ],
        )

    def _wrw(self, x_channels: int, dw_channels: int) -> dict:
        return _graph(
            tensors=[
                _tensor(0, [16, dw_channels, 3, 3], [dw_channels * 9, 9, 3, 1]),
                _tensor(1, [1, 16, 32, 32], [16384, 1024, 32, 1]),
                _tensor(2, [1, x_channels, 32, 32], [x_channels * 1024, 1024, 32, 1]),
            ],
            nodes=[
                {
                    "type": "ConvolutionWrwAttributes",
                    "inputs": {"dy_tensor_uid": 1, "x_tensor_uid": 2},
                    "outputs": {"dw_tensor_uid": 0},
                }
            ],
        )

    def test_grouped_data_grad_reads_dx_off_outputs(self) -> None:
        assert shape_tags_of(self._bwd(dx_channels=32, w_channels=8)) == {"grouped"}

    def test_ungrouped_data_grad_carries_no_tag(self) -> None:
        assert shape_tags_of(self._bwd(dx_channels=32, w_channels=32)) == set()

    def test_grouped_weight_grad_reads_dw_off_outputs(self) -> None:
        assert shape_tags_of(self._wrw(x_channels=32, dw_channels=8)) == {"grouped"}

    def test_ungrouped_weight_grad_carries_no_tag(self) -> None:
        assert shape_tags_of(self._wrw(x_channels=32, dw_channels=32)) == set()

    def test_pooling_parameters_sit_on_the_node_itself(self) -> None:
        # quick/ResampleFwd/Default keeps stride and padding as top-level node
        # keys rather than under "parameters" as convolution does.
        graph = _graph(
            nodes=[
                {
                    "type": "ResampleFwdAttributes",
                    "resample_mode": "maxpool",
                    "window": [3, 3],
                    "stride": [2, 2],
                    "pre_padding": [1, 1],
                    "post_padding": [1, 1],
                }
            ]
        )
        assert shape_tags_of(graph) == {"stride", "padding"}

    def test_neutral_pooling_parameters_are_not_tags(self) -> None:
        graph = _graph(
            nodes=[
                {
                    "type": "ResampleFwdAttributes",
                    "window": [2, 2],
                    "stride": [1, 1],
                    "pre_padding": [0, 0],
                    "post_padding": [0, 0],
                }
            ]
        )
        assert shape_tags_of(graph) == set()

    def test_several_tags_share_one_bracket(self) -> None:
        graph = _graph(
            tensors=[_tensor(1, [8, 32, 16, 16], [8192, 256, 16, 1])],
            nodes=[{"type": "ConvFpropAttributes", "parameters": {"stride": [2, 2]}}],
        )
        assert variant_of(graph, []) == "[multi_batch,stride]"


class TestDtypes:
    def test_all_three_reported(self) -> None:
        graph = _graph(intermediate_data_type="float")
        assert dtypes_of(graph) == "[io=fp16, compute=fp32, intermediate=fp32]"

    def test_falls_back_to_tensor_dtype(self) -> None:
        graph = _graph(io_data_type="unset")
        assert dtypes_of(graph) == "[io=fp16, compute=fp32]"

    def test_unknown_token_passes_through(self) -> None:
        graph = _graph(io_data_type="some_future_type", compute_data_type="unset")
        assert dtypes_of(graph) == "[io=some_future_type]"

    def test_nothing_known(self) -> None:
        graph = {"tensors": [], "nodes": []}
        assert dtypes_of(graph) == "[unspecified]"


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


class TestCollection:
    def test_bundle_without_sidecar_still_counts(self, bundle_root: Path) -> None:
        """No sidecar means no claims -- but the graph is still in the tree.

        It contributes nothing to a numerator and everything to the
        denominator. Dropping it outright, as this once did, deletes a wholly
        unclaimed family from the document instead of showing it as dashes.
        """
        _single_bundle(bundle_root, "quick/Conv/Default", "ConvFwd")
        (unit,) = collect_units(bundle_root)
        assert unit.op == "Conv"
        assert unit.claims == {}

    def test_sweep_without_sidecar_counts_every_case(self, bundle_root: Path) -> None:
        _sweep_bundle(
            bundle_root, "quick/Conv/Sweep", [{"id": i} for i in ("a", "b", "c")]
        )
        units = collect_units(bundle_root)
        assert [u.case_id for u in units] == ["a", "b", "c"]
        assert all(u.claims == {} for u in units)

    def test_wholly_unclaimed_family_still_gets_a_row(self, bundle_root: Path) -> None:
        """The regression that motivated the rule: silence has to be visible."""
        _single_bundle(
            bundle_root,
            "quick/Conv/Default",
            "ConvFwd",
            claims={"MIOPEN_ENGINE": {"gfx942": ["linux"]}},
        )
        _single_bundle(
            bundle_root,
            "quick/Reduction/Default",
            "Reduce",
            graph=_graph(nodes=[{"type": "ReductionAttributes"}]),
        )
        assert "**Reduction**" in render_markdown(collect_units(bundle_root), 0)

    def test_single_graph_bundle(self, bundle_root: Path) -> None:
        _single_bundle(
            bundle_root,
            "quick/Conv/Default",
            "ConvFwd",
            claims={"MIOPEN_ENGINE": {"gfx942": ["linux"], "gfx90a": ["linux"]}},
        )
        (unit,) = collect_units(bundle_root)
        assert unit.op == "Conv"
        assert unit.primary == "Conv"
        assert unit.tiers == ["quick"]
        assert unit.bundle == "Conv/Default"
        assert unit.case_id is None
        assert unit.label == "Default"
        assert unit.claims["MIOPEN_ENGINE"] == {GFX942, GFX90A}

    def test_op_comes_from_the_path_and_the_chain_from_the_graph(
        self, bundle_root: Path
    ) -> None:
        """Rows file under the op directory, wherever the graph starts.

        ``ConvPointwise`` holds graphs whose first node is ``Conv``. The row
        stays under ``ConvPointwise`` -- the directory is the one thing a
        reader can find in the tree -- and the Variant column names the first
        node so the fusion chain still reads as a whole.
        """
        claims = {"MIOPEN_ENGINE": {"gfx942": ["linux"]}}
        _single_bundle(bundle_root, "quick/Conv/Default", "ConvFwd", claims=claims)
        _single_bundle(
            bundle_root,
            "quick/ConvPointwise/Relu",
            "ConvFwdRelu",
            graph=_graph(
                nodes=[
                    {"type": "ConvAttributes"},
                    {
                        "type": "PointwiseAttributes",
                        "inputs": {"operation": "relu_fwd"},
                    },
                ]
            ),
            claims=claims,
        )
        units = sorted(collect_units(bundle_root), key=lambda u: u.op)
        assert [(u.op, u.primary, u.variant_label) for u in units] == [
            ("Conv", "Conv", "(bare)"),
            ("ConvPointwise", "Conv", "Conv + Pointwise:RELU_FWD"),
        ]

    def test_a_bare_graph_under_another_op_is_named_by_its_node(
        self, bundle_root: Path
    ) -> None:
        _single_bundle(
            bundle_root,
            "quick/ConvPointwise/Default",
            "ConvFwd",
            claims={"MIOPEN_ENGINE": {"gfx942": ["linux"]}},
        )
        (unit,) = collect_units(bundle_root)
        assert unit.variant_label == "Conv"

    def test_attributes_is_stripped_wherever_it_appears(
        self, bundle_root: Path
    ) -> None:
        """``Attributes`` is a schema wart, and not always the trailing token."""
        _single_bundle(
            bundle_root,
            "quick/BatchnormInferenceVarianceExt/Default",
            "Bn",
            graph=_graph(nodes=[{"type": "BatchnormInferenceAttributesVarianceExt"}]),
            claims={"MIOPEN_ENGINE": {"gfx942": ["linux"]}},
        )
        (unit,) = collect_units(bundle_root)
        assert unit.primary == "BatchnormInferenceVarianceExt"
        assert unit.variant_label == "(bare)"

    def test_graph_without_nodes_keeps_its_claims(self, bundle_root: Path) -> None:
        """Malformed, but its claims should not vanish from the matrix."""
        _single_bundle(
            bundle_root,
            "quick/Conv/Default",
            "ConvFwd",
            graph=_graph(nodes=[]),
            claims={"MIOPEN_ENGINE": {"gfx942": ["linux"]}},
        )
        (unit,) = collect_units(bundle_root)
        assert unit.op == "Conv"
        assert unit.claims["MIOPEN_ENGINE"] == {GFX942}

    def test_sweep_bundle_produces_one_unit_per_case(self, bundle_root: Path) -> None:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "a_nchw"}, {"id": "b_nhwc"}, {"id": "c_nchw"}],
            claims={
                "MIOPEN_ENGINE": [
                    {"cases": ["a_nchw"], "support": {"gfx942": ["linux"]}}
                ]
            },
        )
        units = collect_units(bundle_root)
        assert [u.case_id for u in units] == ["a_nchw", "b_nhwc", "c_nchw"]
        assert units[0].claims["MIOPEN_ENGINE"] == {GFX942}
        # Unclaimed cases still exist -- they are the denominator.
        assert units[1].claims == {}
        assert units[2].claims == {}

    def test_template_substitution(self, bundle_root: Path) -> None:
        template = _graph(io_data_type="${case.dtype}")
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[
                {"id": "half_nchw", "values": {"dtype": "half"}},
                {"id": "bf16_nchw", "values": {"dtype": "bfloat16"}},
            ],
            template=template,
            claims={"E": [{"cases": ["half_nchw"], "support": {"gfx942": ["linux"]}}]},
        )
        units = collect_units(bundle_root)
        assert units[0].dtypes == "[io=fp16, compute=fp32]"
        assert units[1].dtypes == "[io=bf16, compute=fp32]"

    def test_per_tensor_spec_fills_that_tensors_placeholders(
        self, bundle_root: Path
    ) -> None:
        template = _graph(
            tensors=[
                {
                    "uid": 1,
                    "dims": "${case.dims}",
                    "strides": "${case.strides}",
                    "data_type": "half",
                }
            ]
        )
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[
                {
                    "id": "case_a",
                    "values": {
                        # Inside tensor 1, its own spec wins over these.
                        "dims": [1, 1, 1, 1],
                        "strides": [8192, 256, 16, 1],
                        "tensors": [
                            {
                                "uid": 1,
                                "dims": [8, 32, 16, 16],
                                "strides": [8192, 1, 512, 32],
                            }
                        ],
                    },
                }
            ],
            template=template,
            claims={"E": [{"cases": ["case_a"], "support": {"gfx942": ["linux"]}}]},
        )
        (unit,) = collect_units(bundle_root)
        assert unit.graph["tensors"][0]["dims"] == [8, 32, 16, 16]
        assert unit.layout == "NHWC"  # from the spec's strides, not the top level

    def test_other_placeholders_in_a_tensor_fall_back_to_the_top_level(
        self, bundle_root: Path
    ) -> None:
        template = _graph(
            tensors=[{**_tensor(1, [1, 32, 16, 16]), "virtual": "${case.virtual}"}]
        )
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "case_a", "values": {"virtual": True}}],
            template=template,
        )
        (unit,) = collect_units(bundle_root)
        assert unit.graph["tensors"][0]["virtual"] is True

    def test_case_tensor_spec_does_not_overwrite_a_literal(
        self, bundle_root: Path
    ) -> None:
        """A spec only feeds placeholders; the harness warns it is unused."""
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[
                {
                    "id": "case_a",
                    "values": {"tensors": [{"uid": 1, "dims": [8, 32, 16, 16]}]},
                }
            ],
        )
        (unit,) = collect_units(bundle_root)
        assert unit.graph["tensors"][0]["dims"] == [1, 32, 16, 16]

    def test_placeholder_resolving_to_null_is_kept(self, bundle_root: Path) -> None:
        template = _graph(
            tensors=[{**_tensor(1, [1, 32, 16, 16]), "virtual": "${case.virtual}"}]
        )
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "case_a", "values": {"virtual": None}}],
            template=template,
        )
        errors: list[str] = []
        (unit,) = collect_units(bundle_root, errors)
        assert errors == []
        assert unit.graph["tensors"][0]["virtual"] is None

    def test_any_object_with_a_uid_scopes_its_placeholders(
        self, bundle_root: Path
    ) -> None:
        template = _graph(
            nodes=[{"type": "ConvAttributes", "x": {"uid": 1, "dims": "${case.dims}"}}]
        )
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[
                {
                    "id": "case_a",
                    "values": {"tensors": [{"uid": 1, "dims": [2, 2]}]},
                }
            ],
            template=template,
        )
        (unit,) = collect_units(bundle_root)
        assert unit.graph["nodes"][0]["x"] == {"uid": 1, "dims": [2, 2]}

    def test_resolved_value_is_not_expanded_again(self, bundle_root: Path) -> None:
        template = _graph(
            tensors=[{**_tensor(1, [1, 32, 16, 16]), "name": "${case.name}"}]
        )
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "case_a", "values": {"name": "${case.other}", "other": "x"}}],
            template=template,
        )
        (unit,) = collect_units(bundle_root)
        assert unit.graph["tensors"][0]["name"] == "${case.other}"

    @pytest.mark.parametrize(
        ("value", "field"),
        [
            ("${case.dims}", "dims"),
            ("${case.a.b}", "a.b"),
            ("${case.a-b c}", "a-b c"),
            ("${case.x}", "x"),
            ("${case.}", None),
            ("${case.x", None),
            ("x${case.y}", None),
            ("${case.y} ", None),
            (7, None),
        ],
    )
    def test_placeholder_is_a_whole_string(self, value: object, field: str) -> None:
        assert placeholder_field(value) == field

    def test_tensor_patch_sets_and_removes_keys(self, bundle_root: Path) -> None:
        """How a case makes a scalar runtime pass-by-value; see the README."""
        scalar = {**_tensor(2, [1], [1]), "value_type": "float", "value": 1.0}
        template = _graph(
            tensors=[_tensor(1, [1, 32, 16, 16], [8192, 256, 16, 1]), scalar]
        )
        patch = {
            "uid": 2,
            "set": {"is_runtime_pass_by_value": True},
            "remove": ["value_type", "value"],
        }
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "fixed"}, {"id": "runtime", "tensor_patches": [patch]}],
            template=template,
        )
        fixed, runtime = collect_units(bundle_root)
        assert fixed.graph["tensors"][1] == scalar
        assert runtime.graph["tensors"][1] == {
            "uid": 2,
            "dims": [1],
            "strides": [1],
            "data_type": "half",
            "is_runtime_pass_by_value": True,
        }
        assert runtime.digest != fixed.digest

    def test_tensor_patch_applies_after_case_tensors(self, bundle_root: Path) -> None:
        template = _graph(
            tensors=[{**_tensor(1, [1, 32, 16, 16]), "data_type": "${case.data_type}"}]
        )
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[
                {
                    "id": "case_a",
                    "values": {"tensors": [{"uid": 1, "data_type": "float"}]},
                    "tensor_patches": [{"uid": 1, "set": {"data_type": "double"}}],
                }
            ],
            template=template,
        )
        (unit,) = collect_units(bundle_root)
        assert unit.graph["tensors"][0]["data_type"] == "double"

    @pytest.mark.parametrize(
        ("case", "message"),
        [
            (
                {"values": {"tensors": [{"uid": 9, "dims": [1]}]}},
                "case tensor uid 9 not present in template graph",
            ),
            (
                {"values": {"tensors": [{"uid": 1}, {"uid": 1}]}},
                "duplicate case tensor uid 1",
            ),
            (
                {"values": {"tensors": [{"uid": "1"}]}},
                "case tensor missing integer uid",
            ),
            (
                {"values": {"tensors": [{"uid": True}]}},
                "case tensor missing integer uid",
            ),
            ({"values": {"tensors": {"uid": 1}}}, "values.tensors must be an array"),
            (
                {"tensor_patches": [{"uid": 9, "set": {"virtual": True}}]},
                "tensor_patch uid 9 not found in expanded graph",
            ),
            (
                {"tensor_patches": [{"set": {"virtual": True}}]},
                "tensor_patch entry is missing an integer uid",
            ),
            (
                {"golden": {"dir": "golden/bad"}},
                "golden.path is required when golden is present",
            ),
            (
                {"golden": "golden/bad"},
                "golden.path is required when golden is present",
            ),
        ],
    )
    def test_case_the_harness_cannot_load_is_an_error(
        self, bundle_root: Path, case: dict, message: str
    ) -> None:
        """INVALID_SWEEP_CASE: the harness skips the case, so it is no unit.

        Two such cases sat in the tree unnoticed, because the harness only
        logs the skip; reading them as graphs put two never-run cases in the
        denominators.
        """
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "good"}, {"id": "bad", **case}],
            claims={"E": [{"cases": ["good"], "support": {"gfx942": ["linux"]}}]},
        )
        errors: list[str] = []
        units = collect_units(bundle_root, errors)
        assert [u.case_id for u in units] == ["good"]
        assert len(errors) == 1
        assert f"case 'bad': {message}" in errors[0]

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            (
                {"tensors": [{"uid": 1, "dims": [1, 32, 16, 16]}]},
                "missing placeholder value '${case.dtype}'",
            ),
            (
                # A top-level dims never stands in for a tensor's own.
                {"dtype": "half", "dims": [1, 32, 16, 16]},
                "missing per-tensor placeholder value '${case.dims}' "
                "for tensor uid 1",
            ),
            (
                {"dtype": "half", "tensors": [{"uid": 1, "strides": [1, 1, 1, 1]}]},
                "missing per-tensor placeholder value '${case.dims}' "
                "for tensor uid 1",
            ),
        ],
    )
    def test_case_missing_a_placeholder_value_is_an_error(
        self, bundle_root: Path, values: dict, message: str
    ) -> None:
        """The harness throws rather than leave the field unset."""
        template = _graph(
            io_data_type="${case.dtype}",
            tensors=[{**_tensor(1, [1, 32, 16, 16]), "dims": "${case.dims}"}],
        )
        good = {"dtype": "half", "tensors": [{"uid": 1, "dims": [1, 32, 16, 16]}]}
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[
                {"id": "good", "values": good},
                {"id": "bad", "values": values},
            ],
            template=template,
        )
        errors: list[str] = []
        units = collect_units(bundle_root, errors)
        assert [u.case_id for u in units] == ["good"]
        assert len(errors) == 1
        assert f"case 'bad': {message}" in errors[0]

    @pytest.mark.parametrize("uid", [None, True, "1"])
    def test_template_tensor_without_an_integer_uid_fails_every_case(
        self, bundle_root: Path, uid: object
    ) -> None:
        tensor = _tensor(1, [1, 32, 16, 16])
        if uid is None:
            del tensor["uid"]
        else:
            tensor["uid"] = uid
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "a"}, {"id": "b"}],
            template=_graph(tensors=[tensor]),
        )
        errors: list[str] = []
        assert collect_units(bundle_root, errors) == []
        assert len(errors) == 2
        assert all("template tensor missing integer uid" in e for e in errors)

    def test_unknown_case_id_is_an_error(self, bundle_root: Path) -> None:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "real"}],
            claims={"E": [{"cases": ["ghost"], "support": {"gfx942": ["linux"]}}]},
        )
        errors: list[str] = []
        units = collect_units(bundle_root, errors)
        assert [u.case_id for u in units] == ["real"]
        assert units[0].claims == {}
        assert any("unknown case 'ghost'" in e for e in errors)

    def test_malformed_graph_is_an_error(self, bundle_root: Path) -> None:
        directory = bundle_root / "quick" / "Conv" / "Default"
        directory.mkdir(parents=True)
        (directory / "ConvFwd.json").write_text("{not json", encoding="utf-8")
        (directory / "ConvFwd.support.json").write_text(
            '{"version": 1, "claims": {}}', encoding="utf-8"
        )
        errors: list[str] = []
        assert collect_units(bundle_root, errors) == []
        assert any("ConvFwd.json: invalid JSON" in e for e in errors)

    def test_non_utf8_graph_is_an_error_not_a_crash(self, bundle_root: Path) -> None:
        directory = bundle_root / "quick" / "Conv" / "Default"
        directory.mkdir(parents=True)
        (directory / "ConvFwd.json").write_bytes(b"\xff\xfe{}")
        errors: list[str] = []
        assert collect_units(bundle_root, errors) == []
        assert any("ConvFwd.json: not valid UTF-8" in e for e in errors)

    @pytest.mark.parametrize("document", ["null", "[]", "7"])
    def test_non_object_graph_is_an_error(
        self, bundle_root: Path, document: str
    ) -> None:
        directory = bundle_root / "quick" / "Conv" / "Default"
        directory.mkdir(parents=True)
        (directory / "ConvFwd.json").write_text(document, encoding="utf-8")
        errors: list[str] = []
        assert collect_units(bundle_root, errors) == []
        assert errors == [f"{directory / 'ConvFwd.json'}: not a JSON object"]

    @pytest.mark.parametrize("document", ["null", "[]"])
    def test_non_object_sweep_template_is_an_error(
        self, bundle_root: Path, document: str
    ) -> None:
        # A ``null`` template used to be dropped with no error at all: the
        # parse succeeded, and the non-object check only looked at non-None.
        sweep = _sweep_bundle(bundle_root, "quick/Conv/Sweep", cases=[{"id": "a"}])
        (sweep / "graph.template.json").write_text(document, encoding="utf-8")
        errors: list[str] = []
        assert collect_units(bundle_root, errors) == []
        assert errors == [f"{sweep / 'graph.template.json'}: not a JSON object"]

    @pytest.mark.parametrize("document", ["null", "[]"])
    def test_non_object_sidecar_is_an_error_and_contributes_nothing(
        self, bundle_root: Path, document: str
    ) -> None:
        _single_bundle(bundle_root, "quick/Conv/Default", "ConvFwd")
        sidecar = bundle_root / "quick" / "Conv" / "Default" / "ConvFwd.support.json"
        sidecar.write_text(document, encoding="utf-8")
        errors: list[str] = []
        (unit,) = collect_units(bundle_root, errors)
        assert unit.claims == {}
        assert len(errors) == 1
        assert "top-level value must be an object" in errors[0]

    @pytest.mark.parametrize(
        "cases, message",
        [
            ([{"id": "a"}, {"id": "a"}], "duplicate case id 'a'"),
            ([{"id": ""}], "empty id"),
            ([{"values": {}}], "no string id"),
            ([{"id": 7}], "no string id"),
            (["a"], "no string id"),
        ],
    )
    def test_bad_sweep_manifest_ids_are_errors(
        self, bundle_root: Path, cases: list, message: str
    ) -> None:
        # readSweepCaseIds() throws on each of these, so the harness registers
        # none of the sweep's cases; the matrix must not count them either.
        _sweep_bundle(bundle_root, "quick/Conv/Sweep", cases=cases, claims=None)
        errors: list[str] = []
        assert collect_units(bundle_root, errors) == []
        assert any(message in e for e in errors)

    def test_sweep_manifest_without_cases_is_an_error(self, bundle_root: Path) -> None:
        sweep = _sweep_bundle(bundle_root, "quick/Conv/Sweep", cases=[], claims=None)
        _write_json(sweep / "sweep.json", {"variables": {}})
        errors: list[str] = []
        assert collect_units(bundle_root, errors) == []
        assert any("missing cases[] array" in e for e in errors)

    @pytest.mark.parametrize(
        "sidecar, message",
        [
            ({"version": 2, "claims": {}}, "version must be 1"),
            ({"version": 1, "claims": []}, "claims must be an object"),
            ({"version": 1, "claims": {"E": {"gfx942": ["macos"]}}}, "macos"),
            (
                {"version": 1, "claims": {"E": {"gfx942": "linux"}}},
                "arch 'gfx942' value must be an array",
            ),
        ],
    )
    def test_invalid_single_graph_sidecar_is_an_error_and_contributes_nothing(
        self, bundle_root: Path, sidecar: dict, message: str
    ) -> None:
        directory = bundle_root / "quick" / "Conv" / "Default"
        _write_json(directory / "ConvFwd.json", _graph())
        _write_json(directory / "ConvFwd.support.json", sidecar)
        errors: list[str] = []
        (unit,) = collect_units(bundle_root, errors)
        assert unit.claims == {}  # still in the denominator, claimless
        assert errors and all("ConvFwd.support.json" in e for e in errors)
        assert any(message in e for e in errors)

    def test_unreadable_graph_shape_is_isolated_to_its_bundle(
        self, bundle_root: Path
    ) -> None:
        bad = bundle_root / "quick" / "Conv" / "Bad"
        # A string uid passes every JSON check yet cannot be negated when the
        # representative tensor is chosen.
        _write_json(
            bad / "ConvFwd.json", _graph(tensors=[_tensor("x", [1, 32, 16, 16])])
        )
        _write_json(
            bad / "ConvFwd.support.json",
            {"version": 1, "claims": {"E": {"gfx942": ["linux"]}}},
        )
        _single_bundle(
            bundle_root, "quick/Conv/Good", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        errors: list[str] = []
        units = collect_units(bundle_root, errors)
        assert [u.bundle for u in units] == ["Conv/Good"]
        assert any("cannot read bundle" in e and "Bad" in e for e in errors)

    def test_graphs_inside_a_sweep_are_not_separate_units(
        self, bundle_root: Path
    ) -> None:
        sweep = _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "a"}],
            claims={"E": [{"cases": ["a"], "support": {"gfx942": ["linux"]}}]},
        )
        _write_json(sweep / "golden" / "a" / "tensors.json", {"nodes": []})
        assert len(collect_units(bundle_root)) == 1


# --------------------------------------------------------------------------
# Tiers: an attribute, not an identity
# --------------------------------------------------------------------------


# The tier list in a traceability entry: "Conv/Sweep (quick+full, 3 case(s))".
_TIER_TOKEN = re.compile(r"\([\w-]+(?:\+[\w-]+)*, (?=\d+ case\(s\)\))")


def _render(root: Path) -> tuple[str, list[ClaimUnit]]:
    errors: list[str] = []
    units = collect_units(root, errors)
    assert errors == []
    return render_markdown(units, -1), units


def _without_tiers(units: list[ClaimUnit]) -> list[ClaimUnit]:
    return [replace(u, tiers=[]) for u in sorted(units, key=ClaimUnit.sort_key)]


def _move_sweep_case(root: Path, source: str, target: str, case_id: str) -> None:
    """What moving one case to another tier does: split it out of a sweep
    into the same path under another tier, claims and golden data with it."""
    src, dst = root / source, root / target
    manifest = json.loads((src / "sweep.json").read_text(encoding="utf-8"))
    moving = [c for c in manifest["cases"] if c["id"] == case_id]
    manifest["cases"] = [c for c in manifest["cases"] if c["id"] != case_id]
    _write_json(src / "sweep.json", manifest)
    _write_json(dst / "sweep.json", {"cases": moving})
    _write_json(
        dst / "graph.template.json",
        json.loads((src / "graph.template.json").read_text(encoding="utf-8")),
    )

    sidecar = json.loads((src / "support.json").read_text(encoding="utf-8"))
    moved: dict = {}
    for engine, groups in sidecar["claims"].items():
        for group in groups:
            if case_id in group["cases"]:
                group["cases"].remove(case_id)
                moved.setdefault(engine, []).append(
                    {"cases": [case_id], "support": group["support"]}
                )
        sidecar["claims"][engine] = [g for g in groups if g["cases"]]
    _write_json(src / "support.json", sidecar)
    _write_json(dst / "support.json", {"version": 1, "claims": moved})

    (dst / "golden").mkdir(parents=True, exist_ok=True)
    (src / "golden" / case_id).rename(dst / "golden" / case_id)


class TestTierMoves:
    """Bundles move between tiers.

    The matrix is about which engine supports which graph on which target;
    which CI lane runs the graph is a different question. A move must
    therefore change the ``tiers`` attribute and nothing else.
    """

    @pytest.fixture()
    def tree(self, bundle_root: Path) -> Path:
        cases = [{"id": "a_nchw"}, {"id": "b_nhwc"}, {"id": "c_nchw"}]
        claims = {
            "MIOPEN_ENGINE": [
                {"cases": ["a_nchw", "b_nhwc"], "support": {"gfx942": ["linux"]}},
                {"cases": ["c_nchw"], "support": {"gfx90a": ["linux"]}},
            ],
            "HIP_MLOPS_ENGINE": [
                {"cases": ["b_nhwc"], "support": {"gfx942": ["linux"]}}
            ],
        }
        sweep = _sweep_bundle(bundle_root, "quick/Conv/Sweep", cases, claims=claims)
        for case in cases:
            _write_json(sweep / "golden" / case["id"] / "tensors.json", {})
        # A second topology reusing a case id: (op, case id) alone is not an
        # identity, so the path below the tier has to be part of it.
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Grouped",
            [{"id": "a_nchw"}],
            template=_graph(
                nodes=[{"type": "ConvAttributes", "parameters": {"stride": [2, 2]}}]
            ),
            claims={
                "MIOPEN_ENGINE": [
                    {"cases": ["a_nchw"], "support": {"gfx942": ["linux"]}}
                ]
            },
        )
        _single_bundle(
            bundle_root,
            "quick/Batchnorm/Default",
            "Bn",
            graph=_graph(nodes=[{"type": "BatchnormAttributes"}]),
            claims={"MIOPEN_ENGINE": {"gfx942": ["linux"]}},
        )
        return bundle_root

    def test_moving_bundles_changes_only_the_tier(self, tree: Path) -> None:
        markdown_before, units_before = _render(tree)

        _move_sweep_case(tree, "quick/Conv/Sweep", "standard/Conv/Sweep", "b_nhwc")
        (tree / "full").mkdir()
        (tree / "quick" / "Batchnorm").rename(tree / "full" / "Batchnorm")
        markdown_after, units_after = _render(tree)

        # The move did happen, and is visible exactly where it should be.
        tiers = {(u.bundle, u.case_id): u.tiers for u in units_after}
        assert tiers[("Conv/Sweep", "b_nhwc")] == ["standard"]
        assert tiers[("Conv/Sweep", "a_nchw")] == ["quick"]
        assert tiers[("Batchnorm/Default", None)] == ["full"]

        assert _without_tiers(units_after) == _without_tiers(units_before)
        # Only the traceability comments name tiers.
        assert _TIER_TOKEN.sub("(", markdown_after) != markdown_after
        assert _strip_tier_comments(markdown_after) == _strip_tier_comments(
            markdown_before
        )

    def test_a_copy_in_two_tiers_is_one_unit(self, tree: Path) -> None:
        _sweep_bundle(
            tree,
            "full/Conv/Sweep",
            [{"id": "c_nchw"}],
            claims={
                "MIOPEN_ENGINE": [
                    {"cases": ["c_nchw"], "support": {"gfx942": ["linux"]}}
                ]
            },
        )
        units = [u for u in collect_units(tree) if u.bundle == "Conv/Sweep"]
        (copy,) = [u for u in units if u.case_id == "c_nchw"]
        assert copy.tiers == ["full", "quick"]
        # Claims union: gfx90a from the quick copy, gfx942 from the full one.
        assert copy.claims["MIOPEN_ENGINE"] == {GFX942, GFX90A}
        assert len(units) == 3  # counted once, not twice

    def test_copies_holding_different_graphs_are_an_error(self, tree: Path) -> None:
        _sweep_bundle(
            tree,
            "full/Conv/Sweep",
            [{"id": "c_nchw"}],
            template=_graph(io_data_type="bfloat16"),
        )
        errors: list[str] = []
        units = collect_units(tree, errors)
        assert any(
            "Conv/Sweep case 'c_nchw'" in e and "different graphs" in e for e in errors
        ), errors
        assert len([u for u in units if u.case_id == "c_nchw"]) == 1

    def test_a_new_tier_needs_no_code_change(self, bundle_root: Path) -> None:
        _single_bundle(
            bundle_root, "nightly/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        (unit,) = collect_units(bundle_root)
        assert (unit.op, unit.tiers, unit.bundle) == ("Conv", ["nightly"], "Conv/D")

    @pytest.mark.parametrize(
        "relative", [".", "quick"], ids=["at-root", "directly-under-a-tier"]
    )
    def test_a_bundle_too_shallow_for_tier_and_op_is_an_error(
        self, bundle_root: Path, relative: str
    ) -> None:
        _single_bundle(bundle_root, relative, "C", claims={"E": {"gfx942": ["linux"]}})
        errors: list[str] = []
        assert collect_units(bundle_root, errors) == []
        assert any("must sit under <tier>/<op>/" in e for e in errors), errors

    def test_orphaned_sweep_sidecar_is_an_error(self, bundle_root: Path) -> None:
        """A half-finished move: claims left where no sweep reads them."""
        _write_json(
            bundle_root / "quick" / "Conv" / "Sweep" / "support.json",
            {"version": 1, "claims": {}},
        )
        errors: list[str] = []
        collect_units(bundle_root, errors)
        assert any("orphaned support.json" in e for e in errors), errors

    def test_orphaned_single_graph_sidecar_is_an_error(self, bundle_root: Path) -> None:
        _write_json(
            bundle_root / "quick" / "Conv" / "D" / "Gone.support.json",
            {"version": 1, "claims": {"E": {"gfx942": ["linux"]}}},
        )
        errors: list[str] = []
        collect_units(bundle_root, errors)
        assert any(
            "Gone.support.json" in e and "no sibling graph Gone.json" in e
            for e in errors
        ), errors

    def test_sweep_sidecar_with_single_graph_shape_is_an_error(
        self, bundle_root: Path
    ) -> None:
        _sweep_bundle(bundle_root, "quick/Conv/Sweep", [{"id": "a"}])
        _write_json(
            bundle_root / "quick" / "Conv" / "Sweep" / "support.json",
            {"version": 1, "claims": {"E": {"gfx942": ["linux"]}}},
        )
        errors: list[str] = []
        (unit,) = collect_units(bundle_root, errors)
        assert unit.claims == {}
        assert errors and all("support.json" in e for e in errors)

    def test_single_graph_sidecar_with_sweep_shape_is_an_error(
        self, bundle_root: Path
    ) -> None:
        _single_bundle(
            bundle_root,
            "quick/Conv/D",
            "C",
            claims={"E": [{"cases": ["a"], "support": {"gfx942": ["linux"]}}]},
        )
        errors: list[str] = []
        (unit,) = collect_units(bundle_root, errors)
        assert unit.claims == {}
        assert errors and all("C.support.json" in e for e in errors)

    def test_unknown_sweep_sidecar_version_is_an_error(self, bundle_root: Path) -> None:
        _sweep_bundle(bundle_root, "quick/Conv/Sweep", [{"id": "a"}])
        _write_json(
            bundle_root / "quick" / "Conv" / "Sweep" / "support.json",
            {"version": 2, "claims": {}},
        )
        errors: list[str] = []
        collect_units(bundle_root, errors)
        assert any("version must be 1" in e for e in errors), errors

    def test_any_error_fails_the_run(self, bundle_root: Path, tmp_path: Path) -> None:
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        _write_json(
            bundle_root / "quick" / "Conv" / "D" / "Gone.support.json",
            {"version": 1, "claims": {}},
        )
        out = tmp_path / "out" / "m.md"
        assert main(["--bundles-dir", str(bundle_root), "--output", str(out)]) == 1
        assert not out.exists()


def _strip_tier_comments(markdown: str) -> str:
    return _TIER_TOKEN.sub("(", markdown)


# --------------------------------------------------------------------------
# Cell rules
# --------------------------------------------------------------------------


def _unit(layout: str, engines: dict[str, set]) -> ClaimUnit:
    return ClaimUnit(
        op="Conv",
        primary="Conv",
        variant="(bare)",
        dtypes="[io=fp16]",
        layout=layout,
        tiers=["quick"],
        bundle="Conv/Default",
        case_id=None,
        claims=engines,
    )


def _body(document: str) -> str:
    """Everything from the first target section on.

    The quick key at the top spells out the cell marks by example, so it
    contains the same literals the cell tests look for. Asserting against the
    whole document would pass on the key alone.
    """
    return document.split("### Overview", 1)[1]


class TestCells:
    def test_all_supported_shows_layouts(self) -> None:
        units = [_unit("NCHW", {"E": {GFX942}}), _unit("NHWC", {"E": {GFX942}})]
        document = render_markdown(units, 0)
        assert f"{FULL} 2/2 NCHW, NHWC" in _body(document)

    def test_partial_support_marks_and_counts(self) -> None:
        units = [_unit("NCHW", {"E": {GFX942}}), _unit("NHWC", {})]
        document = render_markdown(units, 0)
        assert f"{PARTIAL} 1/2" in document  # overview
        assert f"{PARTIAL} 1/2 NCHW" in document  # per-op

    def test_layouts_never_stand_in_for_the_count(self) -> None:
        """A single-layout family must still distinguish its rows.

        Sdpa is BHSD throughout, so a layout-only cell renders a variant
        claimed 1 of 2 identically to one claimed 2 of 2 -- the two rows a
        reader most needs told apart.
        """
        thin = replace(_unit("BHSD", {"E": {GFX942}}), variant="(bare)")
        thin_gap = replace(_unit("BHSD", {}), variant="(bare)")
        whole = replace(_unit("BHSD", {"E": {GFX942}}), variant="[multi_batch]")
        body = _body(render_markdown([thin, thin_gap, whole], 0))
        assert f"{PARTIAL} 1/2 BHSD" in body
        assert f"{FULL} 1/1 BHSD" in body

    def test_no_support_is_a_dash(self) -> None:
        units = [_unit("NCHW", {"E": set()}), _unit("NHWC", {"OTHER": {GFX942}})]
        document = render_markdown(units, 0)
        assert f"| {NONE} |" in _body(document)

    def test_no_layout_sorts_last(self) -> None:
        units = [
            _unit(NO_LAYOUT, {"E": {GFX942}}),
            _unit("NCHW", {"E": {GFX942}}),
        ]
        assert f"NCHW, {NO_LAYOUT}" in _body(render_markdown(units, 0))

    def test_bare_variant_sorts_first(self) -> None:
        """Every fused variant starts with ``" + "``, which sorts below ``(``."""
        fused = replace(_unit("NCHW", {"E": {GFX942}}), variant=" + Pointwise:ADD")
        bare = _unit("NCHW", {"E": {GFX942}})
        body = _body(render_markdown([fused, bare], 0))
        assert body.index("| `(bare)` |") < body.index("| ` + Pointwise:ADD` |")


# --------------------------------------------------------------------------
# Markdown document
# --------------------------------------------------------------------------


class TestMarkdown:
    @pytest.fixture()
    def document(self, bundle_root: Path) -> str:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "a_nchw"}, {"id": "b_nhwc"}],
            claims={
                "MIOPEN_ENGINE": [
                    {"cases": ["a_nchw", "b_nhwc"], "support": {"gfx942": ["linux"]}}
                ]
            },
        )
        _single_bundle(
            bundle_root,
            "full/Batchnorm/Default",
            "Bn",
            graph=_graph(nodes=[{"type": "BatchnormAttributes"}]),
            claims={"HIP_MLOPS_ENGINE": {"gfx90a": ["linux"]}},
        )
        return render_markdown(collect_units(bundle_root), 0)

    def test_has_the_expected_skeleton(self, document: str) -> None:
        assert document.startswith("# Combined Engine Support Matrix\n")
        assert "gfx90a / linux" in document
        assert "gfx942 / linux" in document
        assert "### Overview" in document
        assert "🔎 per-(variant, dtype) detail" in document

    def test_names_the_regeneration_command(self, document: str) -> None:
        assert "scripts/render_support_matrix.py" in document
        assert "Do not hand-edit" in document

    def test_regeneration_command_names_the_cli(self) -> None:
        # The markdown module spells out the CLI's path itself; a rename that
        # left it behind would publish a command that no longer runs.
        repo_root = Path(__file__).resolve().parents[3]
        _, script, *_ = REGEN_COMMAND.split()
        assert (repo_root / script).resolve() == Path(__file__).with_name(
            "render_support_matrix.py"
        ).resolve()

    def test_quick_key_precedes_the_first_target(self, document: str) -> None:
        assert document.index(f"`{FULL} ") < document.index("### Overview")

    def test_reading_guide_follows_the_last_target(self, document: str) -> None:
        assert document.index("Reading guide</b></big>") > document.rindex(
            "### Overview"
        )

    def test_reading_guide_defines_every_cell_form(self, document: str) -> None:
        guide = document.split("Reading guide</b></big>", 1)[1]
        for token in (FULL, PARTIAL, NONE, NO_LAYOUT):
            assert f"`{token}" in guide, f"reading guide does not explain {token!r}"

    def test_reading_guide_says_a_dash_is_unclaimed_not_unsupported(
        self, document: str
    ) -> None:
        """The one reading that would invert the document's meaning."""
        guide = document.split("Reading guide</b></big>", 1)[1]
        assert "*unclaimed*" in guide
        assert "not the same as *known unsupported*" in guide

    def test_arch_headings_are_the_bare_target(self, document: str) -> None:
        """Headings carry exactly what the sidecars key on, nothing looked up."""
        assert "<b>gfx90a / linux</b>" in document
        assert "<b>gfx942 / linux</b>" in document

    def test_any_arch_renders_without_code_changes(self) -> None:
        unit = _unit("NCHW", {"E": {("gfx1337", "linux")}})
        assert "gfx1337 / linux</b></big>" in render_markdown([unit], 0)

    def test_engine_columns_are_alphabetical(self, document: str) -> None:
        header = next(ln for ln in document.splitlines() if ln.startswith("| Op |"))
        columns = [c.strip() for c in header.strip("|").split("|")[1:]]
        assert columns == sorted(columns)
        assert columns == ["HIP_MLOPS_ENGINE", "MIOPEN_ENGINE"]

    def test_ops_are_alphabetical(self, document: str) -> None:
        ops = re.findall(r"^\| \*\*(\w+)\*\* \|", document, re.MULTILINE)
        # Two target sections, each listing every op in order.
        assert ops == ["Batchnorm", "Conv", "Batchnorm", "Conv"]

    def test_dtypes_line_is_present(self, document: str) -> None:
        assert "_Dtypes observed: [io=fp16, compute=fp32]_" in document

    def test_comments_never_interrupt_a_table(self, document: str) -> None:
        """A comment between two rows terminates the table in GFM."""
        lines = document.splitlines()
        for index, line in enumerate(lines[:-1]):
            if line.startswith("<!--"):
                following = lines[index + 1]
                assert not following.startswith(
                    "| "
                ), f"comment at line {index + 1} is followed by a table row"

    def test_traceability_names_bundle_and_count(self, document: str) -> None:
        assert "bundles: Conv/Sweep (quick, 2 case(s))" in document

    def test_traceability_is_recorded_once_for_the_document(
        self, document: str
    ) -> None:
        """Provenance does not vary by target, so repeating it per target was
        pure duplication -- and duplication that grew with every target added,
        on a document that stops rendering on GitHub past a few hundred KB."""
        assert document.count("<!-- row: ") == len(
            {line for line in document.splitlines() if line.startswith("<!-- row: ")}
        )

    def test_case_ids_are_omitted_by_default(self, document: str) -> None:
        """Inlining 5k ids costs several times the visible document."""
        assert "a_nchw," not in document
        assert ": a_nchw" not in document

    def test_max_case_ids_inlines_and_truncates(self, bundle_root: Path) -> None:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": f"case_{i}"} for i in range(5)],
            claims={
                "E": [
                    {
                        "cases": [f"case_{i}" for i in range(5)],
                        "support": {"gfx942": ["linux"]},
                    }
                ]
            },
        )
        units = collect_units(bundle_root)
        assert "case_0, case_1, … +3 more" in render_markdown(units, 2)
        assert "case_0, case_1, case_2, case_3, case_4 -->" in render_markdown(
            units, -1
        )

    def test_empty_tree_renders_a_stub(self) -> None:
        document = render_markdown([], 0)
        assert "_No claim-bearing bundles found._" in document

    def test_output_is_deterministic(self, bundle_root: Path, document: str) -> None:
        assert render_markdown(collect_units(bundle_root), 0) == document


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class TestCli:
    def test_missing_directory_exits_two(self, tmp_path: Path) -> None:
        assert main(["--bundles-dir", str(tmp_path / "absent")]) == 2

    def test_output_file_is_written(self, bundle_root: Path, tmp_path: Path) -> None:
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        out = tmp_path / "nested" / "SUPPORT_MATRIX.md"
        assert main(["--bundles-dir", str(bundle_root), "--output", str(out)]) == 0
        assert out.read_text(encoding="utf-8").startswith(
            "# Combined Engine Support Matrix"
        )

    def test_check_passes_on_a_fresh_file(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "SUPPORT_MATRIX.md"
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        main(["--bundles-dir", str(bundle_root), "--output", str(out)])
        assert (
            main(["--bundles-dir", str(bundle_root), "--output", str(out), "--check"])
            == 0
        )

    def test_check_fails_on_a_stale_file(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "SUPPORT_MATRIX.md"
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        main(["--bundles-dir", str(bundle_root), "--output", str(out)])
        _single_bundle(
            bundle_root,
            "quick/Batchnorm/D",
            "B",
            graph=_graph(nodes=[{"type": "BatchnormAttributes"}]),
            claims={"E": {"gfx942": ["linux"]}},
        )
        assert (
            main(["--bundles-dir", str(bundle_root), "--output", str(out), "--check"])
            == 1
        )
        assert "is stale" in capsys.readouterr().err

    def test_check_fails_when_the_file_is_missing(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        assert (
            main(
                [
                    "--bundles-dir",
                    str(bundle_root),
                    "--output",
                    str(tmp_path / "absent.md"),
                    "--check",
                ]
            )
            == 1
        )
        assert "cannot read" in capsys.readouterr().err

    def test_check_does_not_write(self, bundle_root: Path, tmp_path: Path) -> None:
        out = tmp_path / "absent.md"
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        main(["--bundles-dir", str(bundle_root), "--output", str(out), "--check"])
        assert not out.exists()

    def test_rendering_is_idempotent(self, bundle_root: Path, tmp_path: Path) -> None:
        out = tmp_path / "SUPPORT_MATRIX.md"
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        main(["--bundles-dir", str(bundle_root), "--output", str(out)])
        first = out.read_text(encoding="utf-8")
        main(["--bundles-dir", str(bundle_root), "--output", str(out)])
        assert out.read_text(encoding="utf-8") == first


# --------------------------------------------------------------------------
# Overview-only mode
# --------------------------------------------------------------------------


class TestOverviewOnly:
    @pytest.fixture()
    def overview(self, bundle_root: Path) -> str:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "a_nchw"}, {"id": "b_nhwc"}],
            claims={
                "MIOPEN_ENGINE": [
                    {"cases": ["a_nchw", "b_nhwc"], "support": {"gfx942": ["linux"]}}
                ]
            },
        )
        _single_bundle(
            bundle_root,
            "full/Batchnorm/Default",
            "Bn",
            graph=_graph(nodes=[{"type": "BatchnormAttributes"}]),
            claims={"HIP_MLOPS_ENGINE": {"gfx90a": ["linux"]}},
        )
        return render_markdown(collect_units(bundle_root), 0, overview_only=True)

    def test_no_details_tags(self, overview: str) -> None:
        assert "<details>" not in overview
        assert "</details>" not in overview
        assert "<summary>" not in overview

    def test_no_reading_guide(self, overview: str) -> None:
        assert "Reading guide" not in overview

    def test_no_traceability_comments(self, overview: str) -> None:
        assert "<!-- row:" not in overview

    def test_no_overview_subheading(self, overview: str) -> None:
        assert "### Overview" not in overview

    def test_target_uses_plain_heading(self, overview: str) -> None:
        assert "## gfx942 / linux" in overview or "## gfx90a / linux" in overview

    def test_legend_is_overview_variant(self, overview: str) -> None:
        assert LEGEND_OVERVIEW in overview
        assert "Expand a row" not in overview

    def test_regen_command_says_overview_only(self, overview: str) -> None:
        assert "--overview-only" in overview

    def test_no_per_op_detail(self, overview: str) -> None:
        # The pointer to the full matrix may name the detail; the overview
        # itself must not carry any of it.
        assert "per-(variant, dtype) detail" not in overview
        assert "<details>" not in overview
        assert "<!--" not in overview

    def test_overview_table_is_still_present(self, overview: str) -> None:
        assert "| Op |" in overview
        assert "Batchnorm" in overview
        assert "Conv" in overview

    def test_empty_tree_overview(self) -> None:
        document = render_markdown([], 0, overview_only=True)
        assert "_No claim-bearing bundles found._" in document
        assert "<details>" not in document


class TestOverviewOnlyCli:
    def test_overview_only_default_output_path(
        self, bundle_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        # Stay inside this test's tmp_path; its parent is shared by other tests.
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out = out_dir / "SUPPORT_MATRIX_GRAPH_OVERVIEW.md"
        monkeypatch.setattr(
            "render_support_matrix.DEFAULT_MATRIX_PATH",
            out_dir / "SUPPORT_MATRIX.md",
        )
        monkeypatch.setattr("render_support_matrix.DEFAULT_OVERVIEW_PATH", out)
        monkeypatch.setattr("render_support_matrix.INTEGRATION_TESTS_DIR", out_dir)
        assert main(["--bundles-dir", str(bundle_root), "--overview-only"]) == 0
        assert out.exists()
        assert not (out_dir / "SUPPORT_MATRIX.md").exists()
        content = out.read_text(encoding="utf-8")
        assert "<details>" not in content

    def test_overview_only_explicit_output(
        self, bundle_root: Path, tmp_path: Path
    ) -> None:
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        out = tmp_path / "custom_overview.md"
        assert (
            main(
                [
                    "--bundles-dir",
                    str(bundle_root),
                    "--overview-only",
                    "--output",
                    str(out),
                ]
            )
            == 0
        )
        assert out.exists()
        assert "<details>" not in out.read_text(encoding="utf-8")

    def test_check_overview_only_uses_correct_regen_command(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        out = tmp_path / "overview.md"
        main(
            [
                "--bundles-dir",
                str(bundle_root),
                "--overview-only",
                "--output",
                str(out),
            ]
        )
        _single_bundle(
            bundle_root,
            "quick/Batchnorm/D",
            "B",
            graph=_graph(nodes=[{"type": "BatchnormAttributes"}]),
            claims={"E": {"gfx942": ["linux"]}},
        )
        rc = main(
            [
                "--bundles-dir",
                str(bundle_root),
                "--overview-only",
                "--output",
                str(out),
                "--check",
            ]
        )
        assert rc == 1
        assert "--overview-only" in capsys.readouterr().err


# --------------------------------------------------------------------------
# --check edge cases
# --------------------------------------------------------------------------


class TestCheckEdgeCases:
    def test_check_stdout_returns_two(self, bundle_root: Path) -> None:
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        assert (
            main(["--bundles-dir", str(bundle_root), "--output", "-", "--check"]) == 2
        )


# --------------------------------------------------------------------------
# Sidecar edge cases
# --------------------------------------------------------------------------


class TestSidecarEdgeCases:
    def test_empty_claims_produces_claimless_unit(self, bundle_root: Path) -> None:
        _single_bundle(
            bundle_root,
            "quick/Conv/Default",
            "C",
            claims={},
        )
        units = collect_units(bundle_root)
        assert len(units) == 1
        assert units[0].claims == {}

    def test_case_id_in_two_groups_of_one_engine_is_an_error(
        self, bundle_root: Path
    ) -> None:
        # SupportClaims.cpp rejects the whole sidecar for this, so no case of
        # the sweep carries a claim the harness would act on.
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "c0"}, {"id": "c1"}],
            claims={
                "E": [
                    {"cases": ["c0", "c1"], "support": {"gfx942": ["linux"]}},
                    {"cases": ["c0"], "support": {"gfx90a": ["linux"]}},
                ]
            },
        )
        errors: list[str] = []
        units = collect_units(bundle_root, errors)
        assert [u.claims for u in units] == [{}, {}]
        assert any("duplicate case id 'c0'" in e for e in errors)

    def test_same_case_id_under_two_engines_is_fine(self, bundle_root: Path) -> None:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "c0"}],
            claims={
                "E": [{"cases": ["c0"], "support": {"gfx942": ["linux"]}}],
                "F": [{"cases": ["c0"], "support": {"gfx90a": ["linux"]}}],
            },
        )
        errors: list[str] = []
        (unit,) = collect_units(bundle_root, errors)
        assert errors == []
        assert unit.claims == {"E": {GFX942}, "F": {GFX90A}}

    def test_main_refuses_to_write_past_a_defect(
        self, bundle_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "real"}],
            claims={"E": [{"cases": ["ghost"], "support": {"gfx942": ["linux"]}}]},
        )
        output = tmp_path / "out.md"
        rc = main(["--bundles-dir", str(bundle_root), "--output", str(output)])
        assert rc == 1
        assert not output.exists()
        assert "nothing rendered" in capsys.readouterr().err

    def test_units_with_no_targets_shows_message(self, bundle_root: Path) -> None:
        _single_bundle(bundle_root, "quick/Conv/D", "C", claims={})
        units = collect_units(bundle_root)
        md = render_markdown(units, 0)
        assert "1 bundle(s) found, none carrying any support claim" in md
        assert "| Op |" not in md


class TestCheckRegenCommand:
    @pytest.mark.parametrize(
        ("flags", "command"),
        [([], REGEN_COMMAND), (["--overview-only"], REGEN_OVERVIEW_COMMAND)],
        ids=["full", "overview"],
    )
    def test_check_names_the_command_for_the_variant(
        self,
        bundle_root: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        flags: list[str],
        command: str,
    ) -> None:
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        argv = ["--bundles-dir", str(bundle_root), "--output", str(tmp_path / "m.md")]
        assert main([*argv, *flags, "--check"]) == 1
        assert f"regenerate it with: {command}\n" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The committed tree
# --------------------------------------------------------------------------


def _below_tier(directory: Path, root: Path) -> tuple[str, ...]:
    """A bundle directory's path with the tier dropped: ``(op, ...rest)``.

    Re-derived here rather than imported: the tier is the first directory and
    the op the second, and nothing else about the path matters.
    """
    return directory.relative_to(root).parts[1:]


_COMPANION_KINDS = {"meta", "support"}


def _is_graph_file(path: Path) -> bool:
    """Re-decide what counts as a graph rather than importing the shared rule.

    ``bundle_discovery`` holds the version the renderer uses. Importing it here
    would make the crosscheck agree with the renderer by construction on the
    question that turned out to matter most: which files exist at all.
    """
    if path.suffix != ".json" or path.name in ("graph.template.json", "sweep.json"):
        return False
    stem = path.stem
    return stem not in _COMPANION_KINDS and stem.rsplit(".", 1)[-1] not in (
        _COMPANION_KINDS
    )


def _recount_overview(root: Path) -> dict:
    """Re-derive every overview cell by walking graphs, not sidecars.

    Shares no code *and no premise* with the renderer. It enumerates the graphs
    the tree holds and counts a bundle's cases whether or not a sidecar
    mentions them.

    An earlier version iterated ``*.support.json``, which is how the renderer
    once found its work too. Two independent implementations of the same wrong
    premise agree on every cell, so this crosscheck could not have caught the
    denominator dropping unclaimed graphs -- and did not.

    A graph held in several tiers is one graph: everything is keyed by the
    path below the tier, so a copy adds its claims and not its count.
    """
    # identity -> op, and identity -> {(engine, target)}
    ops: dict[tuple, str] = {}
    claimed: dict[tuple, set[tuple[str, str]]] = {}

    sweeps = sorted(
        d
        for d in root.rglob("*")
        if d.is_dir()
        and (d / "sweep.json").is_file()
        and (d / "graph.template.json").is_file()
    )

    for sweep_dir in sweeps:
        where = _below_tier(sweep_dir, root)
        cases = json.loads((sweep_dir / "sweep.json").read_text(encoding="utf-8")).get(
            "cases", []
        )
        ids = {c["id"] for c in cases if isinstance(c, dict) and "id" in c}
        for case_id in ids:
            ops[(where, case_id)] = where[0]
            claimed.setdefault((where, case_id), set())

        support = sweep_dir / "support.json"
        if not support.is_file():
            continue
        claims = json.loads(support.read_text(encoding="utf-8")).get("claims", {})
        for engine, groups in claims.items():
            for group in groups:
                for case_id in group.get("cases", []):
                    if case_id not in ids:
                        continue
                    for arch, platforms in group.get("support", {}).items():
                        for platform in platforms:
                            claimed[(where, case_id)].add(
                                (engine, f"{arch}/{platform}")
                            )

    for graph in sorted(root.rglob("*.json")):
        if not _is_graph_file(graph):
            continue
        if any(sweep == graph.parent or sweep in graph.parents for sweep in sweeps):
            continue  # Already counted case-by-case above.
        identity = (_below_tier(graph.parent, root), graph.name)
        ops[identity] = identity[0][0]
        claimed.setdefault(identity, set())

        support = graph.with_name(f"{graph.stem}.support.json")
        if not support.is_file():
            continue  # Counted in the denominator; contributes to no numerator.
        claims = json.loads(support.read_text(encoding="utf-8")).get("claims", {})
        for engine, arch_map in claims.items():
            for arch, platforms in arch_map.items():
                for platform in platforms:
                    claimed[identity].add((engine, f"{arch}/{platform}"))

    totals: dict[str, int] = {}
    hits: dict[tuple[str, str, str], int] = {}
    for identity, op in ops.items():
        totals[op] = totals.get(op, 0) + 1
        for engine, target in claimed[identity]:
            hits[(op, engine, target)] = hits.get((op, engine, target), 0) + 1
    return {"totals": totals, "hits": hits}


@pytest.fixture(scope="module")
def units() -> list[ClaimUnit]:
    """Walked once: the committed tree is thousands of cases across 60+ sidecars."""
    return collect_units(DEFAULT_BUNDLES_DIR)


_CELL_COUNT = re.compile(rf"[{FULL}{PARTIAL}] (\d+)/(\d+)")


def _counts(cells: list[str]) -> list[int]:
    """The claimed count each cell reports, with ``NONE`` reading as zero."""
    out = []
    for cell in cells:
        match = _CELL_COUNT.search(cell)
        out.append(0 if match is None else int(match[1]))
    return out


def _row_counts(table: str) -> list[list[int]]:
    """One claimed-count list per body row of a pipe table, header dropped."""
    rows = [ln for ln in table.splitlines() if ln.startswith("| ")]
    return [_counts(ln.strip().strip("|").split("|")[1:]) for ln in rows[1:]]


def _op_blocks(document: str) -> list[tuple[list[int], str, str]]:
    """Per op block: the overview counts, the variant table, the detail table.

    The overview counts come from the pipe-table row that sits just above the
    ``<details>`` block (e.g. ``| **Batchnorm** | — | 🟡 528/840 | ... |``).
    The variant and detail tables are the two nested ``<details>`` inside it.

    Splits by target section first so that ops with the same name in
    different ``arch / platform`` sections are paired correctly.
    """
    blocks = []
    for target_section in document.split("### Overview")[1:]:
        overview_rows: dict[str, list[int]] = {}
        for line in target_section.splitlines():
            if line.startswith("| **") and "** |" in line:
                cells = line.strip().strip("|").split("|")
                name = cells[0].strip().strip("*").strip()
                overview_rows[name] = _counts(cells[1:])

        for chunk in target_section.split("<summary>📂 <b>")[1:]:
            name = chunk.split("</b>")[0]
            body = chunk.split("\n", 1)[1].split("</details>\n</details>")[0]
            variants, _, detail = body.partition("<summary>🔎")
            blocks.append((overview_rows.get(name, []), variants, detail))
    return blocks


@pytest.mark.skipif(
    not DEFAULT_BUNDLES_DIR.is_dir(), reason="bundle tree not present in this checkout"
)
class TestRealBundleTree:
    def test_the_tree_has_claims(self, units: list[ClaimUnit]) -> None:
        assert units, "no claim-bearing bundles found in the committed tree"

    def test_renders_without_warnings(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A warning here means a real sidecar the renderer cannot make sense of."""
        collect_units(DEFAULT_BUNDLES_DIR)
        assert capsys.readouterr().err == ""

    def test_every_unit_is_fully_populated(self, units: list[ClaimUnit]) -> None:
        for unit in units:
            assert unit.op and unit.primary and unit.variant
            assert unit.dtypes and unit.layout
            assert unit.tiers and unit.tiers == sorted(set(unit.tiers))
            # The op is the first segment below the tier, and no tier name
            # leaks into the identity.
            assert unit.bundle.split("/")[0] == unit.op

    def test_overview_cells_match_an_independent_recount(
        self, units: list[ClaimUnit]
    ) -> None:
        expected = _recount_overview(DEFAULT_BUNDLES_DIR)

        totals: dict[str, int] = {}
        hits: dict[tuple[str, str, str], int] = {}
        for unit in units:
            totals[unit.op] = totals.get(unit.op, 0) + 1
            for engine, pairs in unit.claims.items():
                for arch, platform in pairs:
                    key = (unit.op, engine, f"{arch}/{platform}")
                    hits[key] = hits.get(key, 0) + 1

        assert totals == expected["totals"]
        assert hits == expected["hits"]

    def test_every_zoom_level_adds_up(self, units: list[ClaimUnit]) -> None:
        """An op's variant counts sum to its overview count, and its
        per-(variant, dtype) counts sum to the same total.

        This is the property that makes three tables worth printing instead of
        one. Each level partitions the same graphs, so any level disagreeing
        with another means a row is being counted twice or dropped -- which is
        how the denominator bug hid for as long as it did.
        """
        blocks = _op_blocks(render_markdown(units, 0))
        assert blocks, "no op blocks in the rendered document"
        for overview, variants, detail in blocks:
            for level, table in (("variant", variants), ("detail", detail)):
                summed = [sum(col) for col in zip(*_row_counts(table))]
                assert summed == overview, f"{level} rows do not sum to the overview"

    def test_real_tree_renders(self, tmp_path: Path) -> None:
        """The same check the pre-commit hook runs.

        The matrix is generated rather than committed, so there is no freshness
        to verify -- what is left worth catching is a sidecar in the real tree
        that the renderer cannot resolve at all.
        """
        assert main(["--output", str(tmp_path / "SUPPORT_MATRIX.md")]) == 0

    def test_real_tree_renders_deterministically(self, tmp_path: Path) -> None:
        """Two renders of one tree must agree byte for byte.

        Nobody diffs a committed copy any more, so an unstable ordering would
        no longer announce itself as a spurious diff -- it would just quietly
        make two contributors' matrices disagree.
        """
        first, second = tmp_path / "a.md", tmp_path / "b.md"
        assert main(["--output", str(first)]) == 0
        assert main(["--output", str(second)]) == 0
        assert first.read_bytes() == second.read_bytes()

    def test_stdout_survives_a_consumer_that_closes_early(self) -> None:
        """`--output - | head` must exit quietly, not print a traceback.

        Piping into a pager or `head` is the whole reason '-' exists, and every
        one of those consumers closes the pipe before the document has drained.
        Uses a Python one-liner instead of `head` for cross-platform support.
        """
        script = Path(__file__).with_name("render_support_matrix.py")
        # stderr goes to a file, not a pipe: nothing drains a pipe until the
        # renderer exits, so a renderer that reports errors would fill it and
        # block forever instead of failing the test.
        with tempfile.TemporaryFile() as stderr:
            renderer = subprocess.Popen(
                [sys.executable, str(script), "--output", "-"],
                stdout=subprocess.PIPE,
                stderr=stderr,
            )
            assert renderer.stdout is not None
            # Read one line then close — equivalent to `head -1` but works on
            # Windows where `head` is not available.
            reader = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(sys.stdin.buffer.readline()); sys.exit(0)",
                ],
                stdin=renderer.stdout,
                stdout=subprocess.PIPE,
            )
            renderer.stdout.close()
            reader.communicate(timeout=SUBPROCESS_TIMEOUT)
            assert renderer.wait(timeout=SUBPROCESS_TIMEOUT) == 0
            stderr.seek(0)
            assert stderr.read() == b""

    def test_write_stdout_survives_a_consumer_that_closes_early(self) -> None:
        """The guard itself, for every CLI that shares it.

        The feature CLIs' documents fit in a pipe buffer today, so only a
        document larger than any buffer is sure to hit the closed pipe.
        """
        # stderr goes to a file for the reason given in the test above.
        with tempfile.TemporaryFile() as stderr:
            writer = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "from support_matrix_units import write_stdout; "
                    "write_stdout('✅ line\\n' * 1_000_000)",
                    str(Path(__file__).parent),
                ],
                stdout=subprocess.PIPE,
                stderr=stderr,
            )
            assert writer.stdout is not None
            reader = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(sys.stdin.buffer.readline()); sys.exit(0)",
                ],
                stdin=writer.stdout,
                stdout=subprocess.PIPE,
            )
            writer.stdout.close()
            line, _ = reader.communicate(timeout=SUBPROCESS_TIMEOUT)
            assert line == "✅ line\n".encode("utf-8")
            assert writer.wait(timeout=SUBPROCESS_TIMEOUT) == 0
            stderr.seek(0)
            assert stderr.read() == b""
