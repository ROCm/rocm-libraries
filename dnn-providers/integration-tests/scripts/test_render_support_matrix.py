#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for render_support_matrix.py.

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
from dataclasses import replace
from pathlib import Path

import pytest
from render_support_matrix import (
    DEFAULT_BUNDLES_DIR,
    DEFAULT_MATRIX_PATH,
    DEFAULT_OVERVIEW_PATH,
    FULL,
    LEGEND_OVERVIEW,
    NO_LAYOUT,
    NONE,
    PARTIAL,
    REGEN_COMMAND,
    REGEN_OVERVIEW_COMMAND,
    ClaimUnit,
    collect_units,
    dtypes_of,
    layout_from_name,
    layout_from_strides,
    main,
    render_json,
    render_markdown,
    shape_tags_of,
    variant_of,
)

GFX942 = ("gfx942", "linux")
GFX90A = ("gfx90a", "linux")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


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
        assert unit.family == "Conv"
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
        assert unit.family == "Conv"
        assert unit.tier == "quick"
        assert unit.bundle == "quick/Conv/Default"
        assert unit.case_id is None
        assert unit.label == "Default"
        assert unit.claims["MIOPEN_ENGINE"] == {GFX942, GFX90A}

    def test_fused_bundle_folds_into_the_unfused_family(
        self, bundle_root: Path
    ) -> None:
        """The reason family comes from the graph and not the directory.

        ``ConvPointwise`` is the same op as ``Conv`` with a node appended;
        filing it as its own family would strand the fused rows in a section a
        reader has to already know exists.
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
        units = collect_units(bundle_root)
        assert {u.family for u in units} == {"Conv"}
        assert sorted(u.variant for u in units) == [
            " + Pointwise:RELU_FWD",
            "(bare)",
        ]

    def test_attributes_is_stripped_wherever_it_appears(
        self, bundle_root: Path
    ) -> None:
        """``Attributes`` is a schema wart, and not always the trailing token."""
        _single_bundle(
            bundle_root,
            "quick/BatchnormInferenceAttributesVarianceExt/Default",
            "Bn",
            graph=_graph(nodes=[{"type": "BatchnormInferenceAttributesVarianceExt"}]),
            claims={"MIOPEN_ENGINE": {"gfx942": ["linux"]}},
        )
        (unit,) = collect_units(bundle_root)
        assert unit.family == "BatchnormInferenceVarianceExt"

    def test_misfiled_bundle_warns_but_keeps_its_claims(
        self, bundle_root: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """The directory no longer decides the family, but it should still agree."""
        _single_bundle(
            bundle_root,
            "quick/Matmul/Default",
            "NotAMatmul",
            graph=_graph(nodes=[{"type": "ConvAttributes"}]),
            claims={"MIOPEN_ENGINE": {"gfx942": ["linux"]}},
        )
        (unit,) = collect_units(bundle_root)
        assert unit.family == "Conv"
        assert "misfiled" in capsys.readouterr().err

    def test_graph_without_nodes_falls_back_to_the_directory(
        self, bundle_root: Path
    ) -> None:
        """Malformed, but its claims should not vanish from the matrix."""
        _single_bundle(
            bundle_root,
            "quick/Conv/Default",
            "ConvFwd",
            graph=_graph(nodes=[]),
            claims={"MIOPEN_ENGINE": {"gfx942": ["linux"]}},
        )
        (unit,) = collect_units(bundle_root)
        assert unit.family == "Conv"

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

    def test_unresolved_placeholder_reads_as_unset(self, bundle_root: Path) -> None:
        template = _graph(io_data_type="${case.missing}")
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "a_nchw", "values": {}}],
            template=template,
            claims={"E": [{"cases": ["a_nchw"], "support": {"gfx942": ["linux"]}}]},
        )
        (unit,) = collect_units(bundle_root)
        assert unit.dtypes == "[io=fp16, compute=fp32]"  # falls back to the tensor

    def test_per_uid_tensor_override(self, bundle_root: Path) -> None:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[
                {
                    "id": "case_a",
                    "values": {
                        "tensors": [
                            {
                                "uid": 1,
                                "dims": [8, 32, 16, 16],
                                "strides": [8192, 1, 512, 32],
                            }
                        ]
                    },
                }
            ],
            claims={"E": [{"cases": ["case_a"], "support": {"gfx942": ["linux"]}}]},
        )
        (unit,) = collect_units(bundle_root)
        assert unit.layout == "NHWC"  # from the overridden strides, not the template

    def test_unknown_case_id_warns_but_does_not_crash(
        self, bundle_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "real"}],
            claims={"E": [{"cases": ["ghost"], "support": {"gfx942": ["linux"]}}]},
        )
        units = collect_units(bundle_root)
        assert [u.case_id for u in units] == ["real"]
        assert units[0].claims == {}
        assert "unknown case 'ghost'" in capsys.readouterr().err

    def test_malformed_json_warns_and_skips(
        self, bundle_root: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory = bundle_root / "quick" / "Conv" / "Default"
        directory.mkdir(parents=True)
        (directory / "ConvFwd.json").write_text("{not json")
        (directory / "ConvFwd.support.json").write_text('{"version": 1, "claims": {}}')
        assert collect_units(bundle_root) == []
        assert "failed to parse JSON" in capsys.readouterr().err

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
# Cell rules
# --------------------------------------------------------------------------


def _unit(layout: str, engines: dict[str, set]) -> ClaimUnit:
    return ClaimUnit(
        family="Conv",
        variant="(bare)",
        dtypes="[io=fp16]",
        layout=layout,
        tier="quick",
        bundle="quick/Conv/Default",
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
        assert f"{PARTIAL} 1/2 NCHW" in document  # per-family

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

    def test_quick_key_precedes_the_first_target(self, document: str) -> None:
        assert document.index(f"`{FULL}`") < document.index("### Overview")

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

    def test_arch_headings_carry_the_marketing_name(self, document: str) -> None:
        assert "gfx90a / linux — MI200 series (MI210/MI250/MI250X)" in document
        assert "gfx942 / linux — MI300 series (MI300A/MI300X/MI325X)" in document

    def test_unknown_arch_renders_bare(self) -> None:
        """A new gfx target must not have to wait on the name table."""
        unit = _unit("NCHW", {"E": {("gfx1337", "linux")}})
        assert "gfx1337 / linux</b></big>" in render_markdown([unit], 0)

    def test_engine_columns_are_alphabetical(self, document: str) -> None:
        header = next(
            ln for ln in document.splitlines() if ln.startswith("| Op family")
        )
        columns = [c.strip() for c in header.strip("|").split("|")[1:]]
        assert columns == sorted(columns)
        assert columns == ["HIP_MLOPS_ENGINE", "MIOPEN_ENGINE"]

    def test_families_are_alphabetical(self, document: str) -> None:
        families = re.findall(r"^\| \*\*(\w+)\*\* \|", document, re.MULTILINE)
        # Two target sections, each listing every family in order.
        assert families == ["Batchnorm", "Conv", "Batchnorm", "Conv"]

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
        assert "bundles: quick/Conv/Sweep (quick, 2 case(s))" in document

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
# JSON document
# --------------------------------------------------------------------------


class TestJson:
    @pytest.fixture()
    def document(self, bundle_root: Path) -> dict:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "a_nchw"}, {"id": "b_nhwc"}],
            claims={
                "MIOPEN_ENGINE": [
                    {"cases": ["a_nchw"], "support": {"gfx942": ["linux"]}}
                ]
            },
        )
        return json.loads(render_json(collect_units(bundle_root)))

    def test_shape(self, document: dict) -> None:
        assert document["version"] == 1
        assert document["engines"] == ["MIOPEN_ENGINE"]
        assert document["targets"] == [
            {"id": "gfx942/linux", "arch": "gfx942", "platform": "linux"}
        ]

    def test_one_record_per_claim_unit(self, document: dict) -> None:
        assert [u["case_id"] for u in document["units"]] == ["a_nchw", "b_nhwc"]

    def test_record_keeps_what_markdown_aggregates_away(self, document: dict) -> None:
        record = document["units"][0]
        assert record["tier"] == "quick"
        assert record["bundle"] == "quick/Conv/Sweep"
        assert record["layout"] == "NCHW"
        assert record["claims"] == {"MIOPEN_ENGINE": ["gfx942/linux"]}

    def test_unclaimed_units_are_present_with_no_claims(self, document: dict) -> None:
        assert document["units"][1]["claims"] == {}

    def test_units_are_sorted(self, document: dict) -> None:
        keys = [
            (u["family"], u["variant"], u["dtypes"], u["layout"], u["case_id"])
            for u in document["units"]
        ]
        assert keys == sorted(keys)


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
        assert out.read_text().startswith("# Combined Engine Support Matrix")

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

    def test_json_format(self, bundle_root: Path, tmp_path: Path) -> None:
        out = tmp_path / "matrix.json"
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        main(
            [
                "--bundles-dir",
                str(bundle_root),
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )
        assert json.loads(out.read_text())["engines"] == ["E"]

    def test_rendering_is_idempotent(self, bundle_root: Path, tmp_path: Path) -> None:
        out = tmp_path / "SUPPORT_MATRIX.md"
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        main(["--bundles-dir", str(bundle_root), "--output", str(out)])
        first = out.read_text()
        main(["--bundles-dir", str(bundle_root), "--output", str(out)])
        assert out.read_text() == first


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
        assert "How to read a cell:" in overview
        assert "Expand a row" not in overview

    def test_regen_command_says_overview_only(self, overview: str) -> None:
        assert "--overview-only" in overview

    def test_no_per_family_detail(self, overview: str) -> None:
        assert "per-(variant, dtype) detail" not in overview
        assert "per-variant" not in overview.lower().split("how to read")[0]

    def test_overview_table_is_still_present(self, overview: str) -> None:
        assert "| Op family" in overview
        assert "Batchnorm" in overview
        assert "Conv" in overview

    def test_empty_tree_overview(self) -> None:
        document = render_markdown([], 0, overview_only=True)
        assert "_No claim-bearing bundles found._" in document
        assert "<details>" not in document


class TestOverviewOnlyCli:
    def test_overview_only_default_output_path(
        self, bundle_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        out = bundle_root.parent / "SUPPORT_MATRIX_OVERVIEW.md"
        monkeypatch.setattr(
            "render_support_matrix.DEFAULT_MATRIX_PATH",
            bundle_root.parent / "SUPPORT_MATRIX.md",
        )
        monkeypatch.setattr("render_support_matrix.DEFAULT_OVERVIEW_PATH", out)
        monkeypatch.setattr(
            "render_support_matrix.INTEGRATION_TESTS_DIR", bundle_root.parent
        )
        assert main(["--bundles-dir", str(bundle_root), "--overview-only"]) == 0
        assert out.exists()
        content = out.read_text()
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
        assert "<details>" not in out.read_text()

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

    def test_check_json_format(self, bundle_root: Path) -> None:
        import tempfile

        out = Path(tempfile.mkdtemp()) / "matrix.json"
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        main(
            [
                "--bundles-dir",
                str(bundle_root),
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )
        assert (
            main(
                [
                    "--bundles-dir",
                    str(bundle_root),
                    "--format",
                    "json",
                    "--output",
                    str(out),
                    "--check",
                ]
            )
            == 0
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

    def test_overlapping_sweep_case_ids_merge_targets(self, bundle_root: Path) -> None:
        _sweep_bundle(
            bundle_root,
            "quick/Conv/Sweep",
            cases=[{"id": "c0"}, {"id": "c1"}],
            claims={
                "E": [
                    {"cases": ["c0"], "support": {"gfx942": ["linux"]}},
                    {"cases": ["c0"], "support": {"gfx90a": ["linux"]}},
                ]
            },
        )
        units = collect_units(bundle_root)
        c0 = next(u for u in units if u.case_id == "c0")
        assert c0.claims["E"] == {GFX942, GFX90A}

    def test_units_with_no_targets_shows_message(self, bundle_root: Path) -> None:
        _single_bundle(bundle_root, "quick/Conv/D", "C", claims={})
        units = collect_units(bundle_root)
        md = render_markdown(units, 0)
        assert "1 bundle(s) found, none carrying any support claim" in md
        assert "Op family" not in md


class TestCheckRegenCommand:
    def test_check_json_uses_json_regen_command(self, bundle_root: Path) -> None:
        import tempfile

        out = Path(tempfile.mkdtemp()) / "matrix.json"
        _single_bundle(
            bundle_root, "quick/Conv/D", "C", claims={"E": {"gfx942": ["linux"]}}
        )
        main(
            [
                "--bundles-dir",
                str(bundle_root),
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )
        out.write_text("stale", encoding="utf-8")
        import io
        from contextlib import redirect_stderr

        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = main(
                [
                    "--bundles-dir",
                    str(bundle_root),
                    "--format",
                    "json",
                    "--output",
                    str(out),
                    "--check",
                ]
            )
        assert rc == 1
        assert "--format json" in buf.getvalue()


# --------------------------------------------------------------------------
# The committed tree
# --------------------------------------------------------------------------


def _family(graph_file: Path, root: Path) -> str:
    """Re-derive a bundle's family without going through the renderer.

    Same rule, spelled out independently: the primary node's type with the
    ``Attributes`` wart removed, falling back to the directory for a graph
    that declares no nodes.
    """
    nodes = json.loads(graph_file.read_text()).get("nodes") or []
    if nodes and isinstance(nodes[0], dict) and nodes[0].get("type"):
        return nodes[0]["type"].replace("Attributes", "")
    return graph_file.parent.relative_to(root).parts[1]


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
    the tree holds, re-reads each primary node type, and counts a bundle's
    cases whether or not a sidecar mentions them.

    An earlier version iterated ``*.support.json``, which is how the renderer
    once found its work too. Two independent implementations of the same wrong
    premise agree on every cell, so this crosscheck could not have caught the
    denominator dropping unclaimed graphs -- and did not.
    """
    totals: dict[str, int] = {}
    hits: dict[tuple[str, str, str], int] = {}

    sweeps = sorted(
        d
        for d in root.rglob("*")
        if d.is_dir()
        and (d / "sweep.json").is_file()
        and (d / "graph.template.json").is_file()
    )

    for sweep_dir in sweeps:
        family = _family(sweep_dir / "graph.template.json", root)
        cases = json.loads((sweep_dir / "sweep.json").read_text()).get("cases", [])
        ids = {c["id"] for c in cases if isinstance(c, dict) and "id" in c}
        totals[family] = totals.get(family, 0) + len(ids)

        support = sweep_dir / "support.json"
        if not support.is_file():
            continue
        claims = json.loads(support.read_text()).get("claims", {})
        for engine, groups in claims.items():
            claimed: dict[str, set] = {}
            for group in groups:
                for case_id in group.get("cases", []):
                    if case_id not in ids:
                        continue
                    for arch, platforms in group.get("support", {}).items():
                        for platform in platforms:
                            claimed.setdefault(f"{arch}/{platform}", set()).add(case_id)
            for target, matched in claimed.items():
                hits[(family, engine, target)] = hits.get(
                    (family, engine, target), 0
                ) + len(matched)

    for graph in sorted(root.rglob("*.json")):
        if not _is_graph_file(graph):
            continue
        if any(sweep == graph.parent or sweep in graph.parents for sweep in sweeps):
            continue  # Already counted case-by-case above.
        family = _family(graph, root)
        totals[family] = totals.get(family, 0) + 1

        support = graph.with_name(f"{graph.stem}.support.json")
        if not support.is_file():
            continue  # Counted in the denominator; contributes to no numerator.
        claims = json.loads(support.read_text()).get("claims", {})
        for engine, arch_map in claims.items():
            for arch, platforms in arch_map.items():
                for platform in platforms:
                    key = (family, engine, f"{arch}/{platform}")
                    hits[key] = hits.get(key, 0) + 1

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


def _family_blocks(document: str) -> list[tuple[list[int], str, str]]:
    """Per family block: the overview counts, the variant table, the detail table.

    The overview counts come from the pipe-table row that sits just above the
    ``<details>`` block (e.g. ``| **Batchnorm** | — | 🟡 528/840 | ... |``).
    The variant and detail tables are the two nested ``<details>`` inside it.

    Splits by target section first so that families with the same name in
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
            assert unit.family and unit.variant and unit.dtypes and unit.layout
            assert unit.tier in {"quick", "standard", "full"}, unit.tier
            assert unit.bundle.startswith(unit.tier + "/")

    def test_overview_cells_match_an_independent_recount(
        self, units: list[ClaimUnit]
    ) -> None:
        expected = _recount_overview(DEFAULT_BUNDLES_DIR)

        totals: dict[str, int] = {}
        hits: dict[tuple[str, str, str], int] = {}
        for unit in units:
            totals[unit.family] = totals.get(unit.family, 0) + 1
            for engine, pairs in unit.claims.items():
                for arch, platform in pairs:
                    key = (unit.family, engine, f"{arch}/{platform}")
                    hits[key] = hits.get(key, 0) + 1

        assert totals == expected["totals"]
        assert hits == expected["hits"]

    def test_every_zoom_level_adds_up(self, units: list[ClaimUnit]) -> None:
        """A family's variant counts sum to its overview count, and its
        per-(variant, dtype) counts sum to the same total.

        This is the property that makes three tables worth printing instead of
        one. Each level partitions the same graphs, so any level disagreeing
        with another means a row is being counted twice or dropped -- which is
        how the denominator bug hid for as long as it did.
        """
        blocks = _family_blocks(render_markdown(units, 0))
        assert blocks, "no family blocks in the rendered document"
        for overview, variants, detail in blocks:
            for level, table in (("variant", variants), ("detail", detail)):
                summed = [sum(col) for col in zip(*_row_counts(table))]
                assert summed == overview, f"{level} rows do not sum to the overview"

    def test_markdown_stays_renderable(self, units: list[ClaimUnit]) -> None:
        """GitHub stops rendering markdown well past a few hundred KB."""
        size = len(render_markdown(units, 0).encode("utf-8"))
        assert size < 500_000, (
            f"matrix is {size} bytes; trim the traceability comments or split "
            "the document before it stops rendering on GitHub"
        )

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
        one of those consumers closes the pipe before 420 KB has drained.
        Uses a Python one-liner instead of `head` for cross-platform support.
        """
        script = Path(__file__).with_name("render_support_matrix.py")
        renderer = subprocess.Popen(
            [sys.executable, str(script), "--output", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
        reader.communicate()
        assert renderer.wait() == 0
        assert renderer.stderr is not None
        assert renderer.stderr.read() == b""
