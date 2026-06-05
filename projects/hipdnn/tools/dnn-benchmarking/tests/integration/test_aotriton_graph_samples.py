# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for AOTriton-derived SDPA graph samples."""

from pathlib import Path
from typing import Any, Dict, Iterable

import pytest

from dnn_benchmarking.graph import GraphLoader


AOTRITON_GRAPH_DIR = Path(__file__).parent.parent.parent / "graphs" / "aotriton_test"
AOTRITON_GRAPH_FILES = sorted(AOTRITON_GRAPH_DIR.glob("*.json"))


def _referenced_tensor_uids(graph_json: Dict[str, Any]) -> Iterable[int]:
    for node in graph_json.get("nodes", []):
        for section in ("inputs", "outputs"):
            for value in node.get(section, {}).values():
                if isinstance(value, int):
                    yield value


def _tensor_by_uid(graph_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    tensors = graph_json.get("tensors", [])
    return {tensor["uid"]: tensor for tensor in tensors}


@pytest.mark.parametrize(
    "graph_path", AOTRITON_GRAPH_FILES, ids=lambda path: path.name
)
def test_aotriton_sdpa_graph_sample_loads(graph_path: Path) -> None:
    """AOTriton-derived samples are loadable hipDNN SDPA graph JSON."""
    loader = GraphLoader()
    graph_json = loader.load_json(graph_path)

    loader.validate(graph_json)
    tensor_infos = loader.extract_tensor_info(graph_json)

    assert tensor_infos
    assert any(tensor_info.is_output for tensor_info in tensor_infos)
    assert all(node.get("type") == "SdpaAttributes" for node in graph_json["nodes"])

    tensor_uids = [tensor["uid"] for tensor in graph_json["tensors"]]
    assert len(tensor_uids) == len(set(tensor_uids))
    assert set(_referenced_tensor_uids(graph_json)).issubset(set(tensor_uids))


def test_aotriton_sdpa_graph_shapes_are_consistent() -> None:
    """AOTriton-derived SDPA tensors obey frontend shape invariants."""
    loader = GraphLoader()

    for graph_path in AOTRITON_GRAPH_FILES:
        graph_json = loader.load_json(graph_path)
        tensors = _tensor_by_uid(graph_json)
        node = graph_json["nodes"][0]
        inputs = node["inputs"]
        outputs = node["outputs"]

        q = tensors[inputs["q_tensor_uid"]]
        k = tensors[inputs["k_tensor_uid"]]
        v = tensors[inputs["v_tensor_uid"]]
        o = tensors[outputs["o_tensor_uid"]]

        q_dims = q["dims"]
        k_dims = k["dims"]
        v_dims = v["dims"]
        o_dims = o["dims"]

        assert len(q_dims) == len(k_dims) == len(v_dims) == len(o_dims) == 4
        assert q_dims[0] == k_dims[0] == v_dims[0]
        assert q_dims[1] % k_dims[1] == 0
        assert q_dims[1] % v_dims[1] == 0
        assert q_dims[3] == k_dims[3]
        assert k_dims[2] == v_dims[2]
        assert o_dims == [q_dims[0], q_dims[1], q_dims[2], v_dims[3]]

        mask_uid = inputs.get("attn_mask_tensor_uid")
        if mask_uid is not None:
            mask_dims = tensors[mask_uid]["dims"]
            assert len(mask_dims) <= 4
            assert mask_dims[-1] in (k_dims[2], 1)
            if len(mask_dims) >= 2:
                assert mask_dims[-2] in (q_dims[2], 1)

        seq_len_q_uid = inputs.get("seq_len_q_tensor_uid")
        seq_len_kv_uid = inputs.get("seq_len_kv_tensor_uid")
        if seq_len_q_uid is not None or seq_len_kv_uid is not None:
            assert seq_len_q_uid is not None
            assert seq_len_kv_uid is not None
            seq_len_q = tensors[seq_len_q_uid]
            seq_len_kv = tensors[seq_len_kv_uid]
            assert seq_len_q["data_type"] == "int32"
            assert seq_len_kv["data_type"] == "int32"
            assert seq_len_q["dims"] == [q_dims[0]]
            assert seq_len_kv["dims"] == [q_dims[0]]
