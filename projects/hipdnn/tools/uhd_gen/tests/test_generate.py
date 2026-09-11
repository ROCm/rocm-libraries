# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collection must never associate a timing with an unaddressed candidate."""
import copy

import pytest

pytest.importorskip("pandas")
from uhd_gen.generate import collect_graph


def _page():
    return {"engine_id": 7, "graph_id": "graph", "device_id": "board", "device_arch": "gfx942",
            "problem_features": {"attention.query.dims[2]": 128}, "device_features": {"device.cu_count": 120},
            "engine_descriptor_id": "13ab344f-4818-4772-bb8e-8e1441fec82c", "engine_name": "hipkernel:test",
            "candidates": [{"id": "kernel-a", "knob_settings": {"tile_m": 32}, "kernel_features": {"kernel.tile_m": 32}}],
            "total_count": 1, "next_offset": None}


def _timing(page):
    candidate = page["candidates"][0]
    return {key: copy.deepcopy(page[key]) for key in
            ("engine_id", "graph_id", "device_id", "device_arch", "engine_descriptor_id", "engine_name", "problem_features", "device_features")} | {
        "results": [{"candidate_id": candidate["id"], "knob_settings": candidate["knob_settings"],
                     "kernel_features": candidate["kernel_features"], "succeeded": True,
                     "is_valid": True, "robust_time_ms": 2.5, "min_time_ms": 2.0, "avg_time_ms": 2.6}]}


def _collect(monkeypatch, tmp_path, responses):
    iterator = iter(responses)
    monkeypatch.setattr("uhd_gen.generate._run_json", lambda *args: next(iterator))
    return collect_graph(["hipdnn_bench", "--graph", "graph.json", "--engine-id", "7"], {}, tmp_path, [],
                         engine_descriptor_id="13ab344f-4818-4772-bb8e-8e1441fec82c")


@pytest.mark.parametrize("mutation", ["candidate", "device", "arch", "knobs", "kernel_features", "provenance"])
def test_timing_must_resolve_exact_enrolled_identity(monkeypatch, tmp_path, mutation):
    page = _page()
    timed = _timing(page)
    if mutation == "provenance":
        page["engine_descriptor_id"] = "not-the-trained-engine"
    elif mutation == "candidate":
        timed["results"][0]["candidate_id"] = "kernel-b"
    elif mutation == "device":
        timed["device_id"] = "other-board"
    elif mutation == "arch":
        page["device_arch"] = "gfx942:sramecc+:xnack-"
        timed["device_arch"] = "gfx942:sramecc-:xnack-"
    elif mutation == "knobs":
        timed["results"][0]["knob_settings"] = {"tile_m": 64}
    else:
        timed["results"][0]["kernel_features"] = {"kernel.tile_m": 64}
    with pytest.raises(ValueError):
        _collect(monkeypatch, tmp_path, [page, timed])


def test_incomplete_enumeration_is_not_a_training_corpus(monkeypatch, tmp_path):
    page = _page()
    page["total_count"] = 2
    with pytest.raises(ValueError, match="truncation"):
        _collect(monkeypatch, tmp_path, [page])


def test_ambiguous_enrolled_tuple_is_rejected_before_timing(monkeypatch, tmp_path):
    page = _page()
    duplicate = copy.deepcopy(page["candidates"][0])
    duplicate["id"] = "kernel-b"
    page["candidates"].append(duplicate)
    page["total_count"] = 2
    with pytest.raises(ValueError, match="unique"):
        _collect(monkeypatch, tmp_path, [page])


def test_feature_suffixed_device_uses_bare_training_architecture(monkeypatch, tmp_path):
    page = _page()
    page["device_arch"] = "gfx942:sramecc+:xnack-"
    rows, _ = _collect(monkeypatch, tmp_path, [page, _timing(page)])
    assert {row["arch"] for row in rows} == {"gfx942"}
