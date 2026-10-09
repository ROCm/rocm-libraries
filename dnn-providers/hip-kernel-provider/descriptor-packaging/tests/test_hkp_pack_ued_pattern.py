"""Packager side of the UED `graph_match` contract shared with the runtime loader.

The fixture corpus and the JSON Schema live in the hipDNN plugin SDK. The
runtime's gtests read the same files, so the packer and the loader agree on
every fixture and on the schema's member lists.
"""

import json
from pathlib import Path

import pytest

from hkp_pack.descriptors import (
    GRAPH_MATCH_ARMS,
    UED_KEYS,
    UED_REQUIRED_KEYS,
    load_flat_input,
)
from hkp_pack.errors import HkpPackError
from hkp_pack.graph_pattern import (
    NODE_KEYS,
    NODE_REQUIRED_KEYS,
    OP_SCHEMA_REGISTRY_ENV,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PLUGIN_SDK = _REPO_ROOT / "projects" / "hipdnn" / "plugin_sdk"
FIXTURE_DIR = _PLUGIN_SDK / "tests" / "ingestor" / "ued_fixtures"
SCHEMA_PATH = _PLUGIN_SDK / "schemas" / "universal_engine_descriptor.schema.json"

_FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))
assert _FIXTURES, f"no UED fixtures under {FIXTURE_DIR}"

# The error each invalid fixture must raise. Matching on it stops an invalid
# fixture from passing because of an unrelated defect.
_EXPECTED_ERROR = {
    "bad_kind": "is not 'op'",
    "both_arms": "exactly one of 'nodes' or 'native'",
    "duplicate_json_key": "duplicate key 'in_0'",
    "duplicate_node_id": "node 'pw' id is not unique",
    "empty_nodes": "'nodes' must be a nonempty array",
    "graph_input_bound_by_two_operands": "binds '\\$x', already bound by node 'first'",
    "neither_arm": "exactly one of 'nodes' or 'native'",
    "one_of_optionality_disagreement": "disagree on operand 'mean'",
    "one_of_single_member": "needs at least two opcodes",
    "one_of_unknown_key": "op has unknown key 'mode'",
    "one_of_unknown_member": "op 'pointwise_relu' is not in the op-schema registry",
    "optional_produced_variable": "marks '\\$conv_out' optional, but node 'conv'",
    "optional_required_operand": "operand 'in_0' is bound with '\\?'",
    "optional_result": "result 'out_0' binding .* takes no '\\?'",
    "reserved_node_id": "node 'graph' id is a reserved root",
    "reserved_variable": "binds reserved root '\\$kernel'",
    "unknown_edge_name": "operand 'A' is not declared by op 'pointwise'",
    "unknown_node_key": "node 'pointwise' has unknown key 'inputs'",
    "unknown_opcode": "op 'pointwise_add' is not in the op-schema registry",
    "variable_bound_by_two_results": "binds '\\$y', already bound by node 'first'",
    "variable_equals_node_id": "'\\$pointwise' collides with node id",
    "wrong_direction_edge": "operand 'out_0' is declared by op 'pointwise' as a result",
}

_NODES_UED = {
    "version": "1.0",
    "id": "8a4a9d0e-4a3f-5b7c-9d1e-2f3a4b5c6d7e",
    "name": "test_fixture:nodes",
    "metadata": "0b1c2d3e-4f5a-5b6c-8d7e-9f0a1b2c3d4e",
    "graph_match": {
        "nodes": [
            {
                "kind": "op",
                "id": "pointwise",
                "op": "pointwise",
                "operands": {"in_0": "$input_a", "in_1": "$input_b"},
                "results": {"out_0": "$output"},
            }
        ]
    },
}


def _ued_root(tmp_path, text, metadata):
    """A root holding one UED and the KMD its `metadata` names."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "fixture.ued.json").write_text(text, encoding="utf-8")
    kmd = {"version": "1.0", "id": metadata, "name": "fixture", "fields": []}
    (root / "fixture.kmd.json").write_text(json.dumps(kmd), encoding="utf-8")
    return root


@pytest.mark.quick
@pytest.mark.parametrize("fixture", _FIXTURES, ids=lambda path: path.stem)
def test_ued_fixture_validity_matches_the_runtime(tmp_path, fixture):
    case = json.loads(fixture.read_text(encoding="utf-8"))
    text = case["ued_text"] if "ued_text" in case else json.dumps(case["ued"])
    root = _ued_root(tmp_path, text, json.loads(text)["metadata"])

    if case["valid"]:
        load_flat_input(root)
    else:
        with pytest.raises(HkpPackError, match=_EXPECTED_ERROR[fixture.stem]):
            load_flat_input(root)


@pytest.mark.quick
def test_schema_member_lists_equal_the_packer_constants():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    graph_match = schema["properties"]["graph_match"]
    node = graph_match["properties"]["nodes"]["items"]

    assert set(schema["properties"]) == set(UED_KEYS)
    assert set(schema["required"]) == set(UED_REQUIRED_KEYS)
    assert set(graph_match["properties"]) == set(GRAPH_MATCH_ARMS)
    assert [arm["required"] for arm in graph_match["oneOf"]] == [
        [arm] for arm in GRAPH_MATCH_ARMS
    ]
    assert set(node["properties"]) == set(NODE_KEYS)
    assert set(node["required"]) == set(NODE_REQUIRED_KEYS)


@pytest.mark.quick
def test_every_schema_property_records_a_supported_added_in_version():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    versions = schema["properties"]["version"]["enum"]
    missing = []

    def walk(node, path):
        if isinstance(node, dict):
            for name, prop in node.get("properties", {}).items():
                if prop.get("addedInVersion") not in versions:
                    missing.append(f"{path}/{name}")
            for key, child in node.items():
                walk(child, f"{path}/{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}/{index}")

    walk(schema, "")
    assert not missing


@pytest.mark.quick
def test_duplicate_json_key_names_the_key_and_the_file(tmp_path):
    text = json.dumps(_NODES_UED)
    duplicated = text.replace('"name": ', '"name": "test_fixture:first", "name": ', 1)
    root = _ued_root(tmp_path, duplicated, _NODES_UED["metadata"])

    with pytest.raises(HkpPackError, match="fixture.ued.json has duplicate key 'name'"):
        load_flat_input(root)


@pytest.mark.quick
@pytest.mark.parametrize("source", ["argument", "environment"])
def test_nodes_pattern_with_missing_registry_names_its_path(
    tmp_path, monkeypatch, source
):
    missing = tmp_path / "absent" / "op_schema_registry.json"
    root = _ued_root(tmp_path, json.dumps(_NODES_UED), _NODES_UED["metadata"])
    kwargs = {}
    if source == "argument":
        kwargs["op_schema_registry"] = missing
    else:
        monkeypatch.setenv(OP_SCHEMA_REGISTRY_ENV, str(missing))

    with pytest.raises(HkpPackError, match="cannot read op-schema registry") as exc:
        load_flat_input(root, **kwargs)
    assert str(missing) in str(exc.value)


@pytest.mark.quick
def test_native_ued_loads_without_a_registry(tmp_path):
    native = dict(_NODES_UED, graph_match={"native": "hipkernel.pointwise.graph_match"})
    root = _ued_root(tmp_path, json.dumps(native), native["metadata"])

    load_flat_input(root, op_schema_registry=tmp_path / "absent.json")


@pytest.mark.quick
@pytest.mark.parametrize(
    "key, field",
    [("operands", "input_tensor_uids"), ("results", "output_tensor_uids")],
)
def test_an_edge_naming_a_registry_edge_list_is_undeclared(tmp_path, key, field):
    # custom_op's uid vectors are edge lists. The runtime indexes them, but a
    # pattern never binds them.
    node = {"kind": "op", "id": "custom", "op": "custom_op", key: {field: "$uids"}}
    ued = dict(_NODES_UED, graph_match={"nodes": [node]})
    root = _ued_root(tmp_path, json.dumps(ued), ued["metadata"])

    with pytest.raises(
        HkpPackError, match=f"{key[:-1]} '{field}' is not declared by op 'custom_op'"
    ):
        load_flat_input(root)


@pytest.mark.quick
def test_a_registry_with_a_malformed_edge_list_names_the_op(tmp_path):
    registry = tmp_path / "op_schema_registry.json"
    entry = {
        "operands": [{"name": "in_0", "optional": False}],
        "results": [{"name": "out_0", "optional": False}],
        "operand_lists": [{"field": "in_0_tensor_uids"}],
        "result_lists": [],
    }
    registry.write_text(json.dumps({"ops": {"pointwise": entry}}), encoding="utf-8")
    root = _ued_root(tmp_path, json.dumps(_NODES_UED), _NODES_UED["metadata"])

    with pytest.raises(HkpPackError, match="op 'pointwise' operand_lists must be"):
        load_flat_input(root, op_schema_registry=registry)
