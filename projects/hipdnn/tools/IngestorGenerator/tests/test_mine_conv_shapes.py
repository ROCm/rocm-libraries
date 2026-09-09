# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Protect source semantics and coverage denominators at the mining boundary."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "mine_conv_shapes.py"
_SPEC = importlib.util.spec_from_file_location("mine_conv_shapes", _SCRIPT)
miner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(miner)


def _case(**overrides):
    return {
        "case_id": "F001",
        "model": "example",
        "op": "conv2d",
        "layout": "NHWC",
        "N": 2,
        "Cin": 32,
        "Cout": 32,
        "H": 14,
        "W": 14,
        "Kh": 3,
        "Kw": 3,
        "stride": 1,
        "pad": 1,
        "dilation": 1,
        "groups": 1,
        "dtype": "fp16",
        **overrides,
    }


def _dense(dims, *, channels_last=True):
    order = [0, *range(2, len(dims)), 1] if channels_last else list(range(len(dims)))
    strides = [0] * len(dims)
    stride = 1
    for axis in reversed(order):
        strides[axis] = stride
        stride *= dims[axis]
    return strides


def _graph(
    *,
    groups=1,
    spatial=(14, 14),
    kernel=(3, 3),
    pre=(1, 1),
    post=None,
    stride=(1, 1),
    dilation=(1, 1),
    channels_last=True,
):
    post = pre if post is None else post
    out = [
        (i + p + q - d * (f - 1) - 1) // s + 1
        for i, f, p, q, s, d in zip(spatial, kernel, pre, post, stride, dilation)
    ]
    dims = ([2, 32, *spatial], [32, 32 // groups, *kernel], [2, 32, *out])
    tensors = [
        {
            "uid": uid,
            "name": name,
            "dims": shape,
            "strides": _dense(shape, channels_last=channels_last),
            "data_type": "half",
            "virtual": False,
        }
        for uid, name, shape in zip(
            (11, 22, 33), ("arbitrary_a", "arbitrary_b", "arbitrary_c"), dims
        )
    ]
    return {
        "name": "misleading_nchw_stride7",
        "compute_data_type": "float",
        "io_data_type": "half",
        "metadata": {"source": "recorded model graph", "model": "model-a"},
        "tensors": tensors,
        "nodes": [
            {
                "name": "conv",
                "type": "ConvolutionFwdAttributes",
                "compute_data_type": "float",
                "inputs": {"x_tensor_uid": 11, "w_tensor_uid": 22},
                "outputs": {"y_tensor_uid": 33},
                "parameters": {
                    "conv_mode": "CROSS_CORRELATION",
                    "pre_padding": list(pre),
                    "post_padding": list(post),
                    "stride": list(stride),
                    "dilation": list(dilation),
                },
            }
        ],
    }


def _read_cases(tmp_path, rows):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(rows))
    return miner.from_rocke_cases(
        path, source_uri="https://example.org/rev/bench_cases_conv.json"
    )


def _read_graph(tmp_path, graph):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    return miner.from_graph_corpus(path)[0]


def test_graph_and_case_resolve_same_request_without_tensor_name_or_filename_heuristics(
    tmp_path,
):
    case = _read_cases(tmp_path, [_case()])[0]
    graph = _read_graph(tmp_path, _graph())
    assert graph["oracle_request"] == case["oracle_request"]
    assert graph["oracle_request"]["layout"] == "NHWC"
    assert graph["oracle_request"]["stride_h"] == 1
    assert graph["kernel_support"] == "pending"


@pytest.mark.parametrize(
    "fields", [{}, {"is_override_shape_enabled": False}], ids=["absent", "disabled"]
)
def test_static_graph_shape_flag_keeps_the_exact_request(tmp_path, fields):
    graph = {**_graph(), **fields}
    record = _read_graph(tmp_path, graph)
    expected = _read_cases(tmp_path, [_case()])[0]["oracle_request"]
    assert record["status"] == "request"
    assert record["oracle_request"] == expected
    assert record["reasons"] == []
    assert record["provenance"]["attributes"] == graph


def test_shape_overrides_cannot_become_a_fixed_shape_catalog_request(tmp_path):
    graph = {**_graph(), "is_override_shape_enabled": True}
    record = _read_graph(tmp_path, graph)
    assert record["status"] == "outside_contract"
    assert record["oracle_request"] is None
    assert record["kernel_support"] == "not_representable"
    assert any("shape overrides" in reason for reason in record["reasons"])
    assert record["provenance"]["attributes"] == graph
    requests, report = miner.build_corpus([record])
    assert requests == []
    assert report["counts"]["source_records"] == 1
    assert report["counts"]["outside_contract"] == 1
    assert report["counts"]["unique_requests"] == 0


@pytest.mark.parametrize("value", [None, 0, 1, 0.0, "false", [], {}])
def test_nonboolean_shape_override_flags_fail_with_source(tmp_path, value):
    graph = {**_graph(), "is_override_shape_enabled": value}
    with pytest.raises(
        miner.MiningError, match="graph.json.*is_override_shape_enabled.*boolean"
    ):
        _read_graph(tmp_path, graph)


def test_rectangular_stride_dilation_and_padding_are_not_collapsed(tmp_path):
    record = _read_graph(
        tmp_path,
        _graph(
            spatial=(19, 17), kernel=(3, 5), stride=(2, 3), dilation=(2, 1), pre=(2, 1)
        ),
    )
    req = record["oracle_request"]
    assert {
        k: req[k]
        for k in (
            "Hi",
            "Wi",
            "Y",
            "X",
            "stride_h",
            "stride_w",
            "dilation_h",
            "dilation_w",
            "pad_h",
            "pad_w",
        )
    } == {
        "Hi": 19,
        "Wi": 17,
        "Y": 3,
        "X": 5,
        "stride_h": 2,
        "stride_w": 3,
        "dilation_h": 2,
        "dilation_w": 1,
        "pad_h": 2,
        "pad_w": 1,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "f16_typo"),
        ("layout", "CHNW"),
        ("op", "conv2d_unknown"),
        ("conv_mode", "transpose"),
    ],
)
def test_unknown_case_categories_fail_with_source(tmp_path, field, value):
    with pytest.raises(miner.MiningError, match="cases.json.*unknown"):
        _read_cases(tmp_path, [_case(**{field: value})])


@pytest.mark.parametrize("dtype", ["fp16", "half", "float16", "torch.float16"])
def test_known_dtype_aliases_normalize_without_changing_source_attributes(
    tmp_path, dtype
):
    record = _read_cases(tmp_path, [_case(dtype=dtype)])[0]
    assert record["oracle_request"]["dtype"] == "fp16"
    assert record["provenance"]["attributes"]["dtype"] == dtype


def test_groups_are_inferred_from_filter_channels_and_retained_for_family_oracle(
    tmp_path,
):
    record = _read_graph(tmp_path, _graph(groups=4))
    assert record["status"] == "outside_contract"
    assert record["oracle_request"]["G"] == 4
    assert record["kernel_support"] == "pending"
    assert "integration contract covers groups=1" in record["reasons"][0]
    shapes, report = miner.build_corpus([record])
    assert shapes == []
    assert report["counts"]["source_records"] == 1


def test_3d_channels_last_maps_to_dispatcher_depth_fields_without_claiming_unsupported(
    tmp_path,
):
    record = _read_graph(
        tmp_path,
        _graph(
            spatial=(5, 14, 14),
            kernel=(3, 3, 3),
            pre=(1, 1, 1),
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
        ),
    )
    assert record["status"] == "outside_contract"
    assert record["oracle_request"]["Di"] == 5
    assert record["oracle_request"]["Z"] == 3
    assert record["oracle_request"]["layout"] == "NHWC"
    assert record["kernel_support"] == "pending"
    assert record["provenance"]["attributes"]["tensors"][0]["dims"] == [
        2,
        32,
        5,
        14,
        14,
    ]


def test_nchw_layout_is_observed_from_strides_and_not_transposed(tmp_path):
    record = _read_graph(tmp_path, _graph(channels_last=False))
    assert record["status"] == "outside_contract"
    assert record["oracle_request"]["layout"] == "NCHW"


def test_asymmetric_padding_does_not_become_symmetric_request(tmp_path):
    graph = _graph(pre=(0, 1), post=(2, 1))
    record = _read_graph(tmp_path, graph)
    assert record["status"] == "outside_contract"
    assert record["oracle_request"] is None
    assert record["provenance"]["attributes"]["nodes"][0]["parameters"][
        "post_padding"
    ] == [2, 1]


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda g: g["tensors"][1].update(data_type="bfloat16"), "uniform"),
        (lambda g: g["tensors"][0]["strides"].__setitem__(2, 500), "layouts cannot"),
        (lambda g: g["nodes"][0]["parameters"].update(alpha=2), "uninterpreted"),
        (lambda g: g["nodes"][0].update(compute_data_type="half"), "fp32 accumulation"),
        (lambda g: g["tensors"][2].update(virtual=True), "non-virtual"),
        (
            lambda g: g["tensors"][0].update(ragged_offset=99),
            "uninterpreted tensor attributes",
        ),
        (
            lambda g: g["nodes"][0]["inputs"].update(bias_tensor_uid=22),
            "uninterpreted convolution tensor bindings",
        ),
    ],
)
def test_semantics_not_expressible_by_request_remain_explicit_exclusions(
    tmp_path, mutate, reason
):
    graph = _graph()
    mutate(graph)
    record = _read_graph(tmp_path, graph)
    assert record["status"] == "outside_contract"
    assert record["oracle_request"] is None
    assert any(reason in text for text in record["reasons"])


def test_fused_and_backward_graphs_are_counted_without_mining_plain_forward_requests(
    tmp_path,
):
    fused = _graph()
    fused["nodes"].append({"name": "relu", "type": "PointwiseAttributes"})
    backward = _graph()
    backward["nodes"][0]["type"] = "ConvolutionBwdAttributes"
    records = [_read_graph(tmp_path, graph) for graph in (fused, backward)]
    shapes, report = miner.build_corpus(records)
    assert shapes == []
    assert report["counts"]["outside_contract"] == 1
    assert report["counts"]["non_forward"] == 1


def test_all_duplicate_provenance_survives_and_inputs_are_not_mutated(tmp_path):
    records = _read_cases(
        tmp_path, [_case(case_id="F001", model="a"), _case(case_id="F002", model="b")]
    )
    records.append(_read_graph(tmp_path, _graph()))
    original = copy.deepcopy(records)
    shapes, report = miner.build_corpus(records)
    assert len(shapes) == 1
    assert report["counts"]["merged_request_records"] == 2
    provenance = shapes[0]["_provenance"]
    assert provenance["uri"] == "https://example.org/rev/bench_cases_conv.json"
    assert provenance["also"][0]["case_id"] == "F002"
    assert provenance["also"][1]["attributes"]["metadata"]["model"] == "model-a"
    assert len(provenance["sha256"]) == 64
    assert records == original


@pytest.mark.parametrize(
    "field,value", [("stride", 0), ("N", 2.5), ("groups", True), ("dilation", [1])]
)
def test_invalid_dimensions_and_attributes_are_not_coerced(tmp_path, field, value):
    with pytest.raises(miner.MiningError):
        _read_cases(tmp_path, [_case(**{field: value})])


def test_output_dimensions_are_checked_against_attributes(tmp_path):
    graph = _graph()
    graph["tensors"][2]["dims"][2] = 13
    with pytest.raises(miner.MiningError, match="output tensor dims disagree"):
        _read_graph(tmp_path, graph)


def test_equivalent_scalar_and_vector_aliases_are_allowed_but_conflicts_fail(tmp_path):
    record = _read_cases(tmp_path, [_case(strides=[1, 1], pads_before=[1, 1])])[0]
    assert record["status"] == "request"
    with pytest.raises(miner.MiningError, match="conflicting aliases"):
        _read_cases(tmp_path, [_case(strides=[1, 2])])


def test_workload_tarball_is_read_without_extracting_and_keeps_member_provenance(
    tmp_path,
):
    path = tmp_path / "conv.tar.gz"
    blob = json.dumps(_graph()).encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("conv/fprop/recorded.json")
        member.size = len(blob)
        archive.addfile(member, io.BytesIO(blob))
    record = miner.from_graph_corpus(path, source_uri="https://example.org/dvc/object")[
        0
    ]
    assert record["provenance"]["path"] == "conv/fprop/recorded.json"
    assert (
        record["provenance"]["uri"]
        == "https://example.org/dvc/object#conv/fprop/recorded.json"
    )
    assert len(record["provenance"]["archive_sha256"]) == 64
    assert not (tmp_path / "conv").exists()


def test_cli_always_writes_complete_inventory_beside_request_list(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([_case(), _case(case_id="F002", groups=4)]))
    out = tmp_path / "requests.json"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--rocke-cases", str(cases), "--out", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(out.read_text())) == 1
    report = json.loads(out.with_name("requests.report.json").read_text())
    assert report["counts"]["source_records"] == 2
    assert report["records"][1]["oracle_request"]["G"] == 4
    assert "Kernel support pending" in result.stdout


def test_cli_no_source_is_an_error(tmp_path):
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--out", str(tmp_path / "out.json")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "at least one source" in result.stderr
