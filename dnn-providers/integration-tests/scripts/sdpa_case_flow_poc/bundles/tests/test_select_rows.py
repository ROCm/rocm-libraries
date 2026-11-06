# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bundle row selection (select_rows.py).

The unit tests classify a few rows built in memory with a small class-to-value table. The
recorded-count test runs the selection on the committed levels.csv with the class-to-value
table built from the committed model.
"""
import json
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import select_rows  # noqa: E402

CLASS_VALUES = {
    "ragged_tensors": {
        "padded": [],
        "ragged": ["Q", "K", "V", "O"],
        "ragged_q_side": ["Q", "O"],
        "ragged_kv_side": ["K", "V"],
    },
    "paged_sides": {"present": ["K", "V"], "k_only": ["K"], "v_only": ["V"]},
    "page_tables": {"separate": 2, "shared": 1},
    "packed_page_table": {"dense": False, "packed": True},
    "head_dim": {"d64": 64, "d128": 128},
}

STRIDE_AXES = ("strides_q", "strides_k", "strides_v", "strides_o", "strides_stats")

BASE_ROW = {
    "case_key": "lv-1",
    "level": "standard",
    "batch_packing": "padded",
    "ragged_offsets": "-",
    "paged_kv": "absent",
    "page_table_storage": "-",
    "page_table_sharing": "-",
    "sequence_lengths": "uniform",
    "tensor_overlap": "separate",
    "dtype_q": "fp16",
    "dtype_k_vs_q": "same",
    "dtype_v_vs_q": "same",
    "matmul_input_precision": "fp16",
    "block_mask": "absent",
    "bias": "absent",
    "attention_sinks": "absent",
    "scale_source": "attribute_value",
    "dropout": "absent",
    "dropout_rng_state": "-",
    "dropout_pattern_output": "-",
    "stats_output": "not_requested",
    "max_sumexp_output": "not_requested",
    "amax_s": "not_requested",
    "amax_o": "not_requested",
    "descale_q": "absent",
    "descale_k": "absent",
    "descale_v": "absent",
    "s_quantization": "absent",
    "scale_o": "absent",
    **{axis: "packed" for axis in STRIDE_AXES},
}


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "class-to-value.json"
    path.write_text(json.dumps(CLASS_VALUES))
    return select_rows.Model(path)


def row(**changes):
    out = dict(BASE_ROW)
    out.update(changes)
    return out


@pytest.mark.parametrize("axis", STRIDE_AXES)
def test_zero_stride_row_is_selected(model, axis):
    assert select_rows.classify([row(**{axis: "zero_broadcast"})], model) == [[]]


def test_zero_stride_row_keeps_its_content_reasons(model):
    reasons = select_rows.classify(
        [row(strides_q="zero_broadcast", ragged_offsets="prefix_sum_of_lengths")],
        model,
    )
    assert reasons == [["ragged_offsets"]]


def test_random_dropout_without_rng_dump_is_still_excluded(model):
    reasons = select_rows.classify(
        [
            row(
                dropout="random",
                dropout_rng_state="caller_supplied",
                dropout_pattern_output="not_requested",
            )
        ],
        model,
    )
    assert reasons == [["dropout_without_rng_dump"]]


@pytest.fixture(scope="module")
def committed_selection(tmp_path_factory):
    from sdpa_case_flow_poc import class_values
    from sdpa_case_flow_poc.model import load_model

    levels = select_rows.PLAN_DIR / "levels.csv"
    if select_rows.sha256(levels) != select_rows.expected_sha256():
        pytest.skip("levels.csv does not match levels.csv.sha256")
    path = tmp_path_factory.mktemp("cv") / "class-to-value.json"
    path.write_bytes(
        class_values.render(
            class_values.build(load_model(select_rows.PLAN_DIR), select_rows.PLAN_DIR)
        )
    )
    model = select_rows.Model(path)
    _, _, rows = select_rows.read_rows(levels)
    reasons = select_rows.classify(rows, model)
    return rows, reasons, select_rows.summarise(rows, reasons, model), model


def test_recorded_counts_hold_for_the_committed_csv(committed_selection):
    _, _, summary, _ = committed_selection
    assert select_rows.check_expected(summary) == []


def test_every_expressible_zero_stride_row_of_the_committed_csv_is_selected(
    committed_selection,
):
    rows, reasons, _, model = committed_selection
    zero = [
        (r, w)
        for r, w in zip(rows, reasons)
        if any(r[a] == "zero_broadcast" for a in STRIDE_AXES)
    ]
    expressible = [
        (r, w)
        for r, w in zero
        if not select_rows.content_reasons(r, model)
        and not select_rows.extra_reasons(r)
    ]
    assert len(expressible) == 54
    assert all(not w for _, w in expressible)
