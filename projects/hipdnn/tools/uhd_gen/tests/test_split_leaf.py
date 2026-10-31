# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The split/leaf `model.bin` encoding: what the converter writes, and what the offline
verifier refuses, reason for reason, as TreeDataAdapter refuses it."""
from __future__ import annotations

import struct

import pytest

pytest.importorskip("flatbuffers")
lgb = pytest.importorskip("lightgbm")
np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import (  # noqa: E402
    GbdtModel,
    GbdtModelT,
)
from hipdnn_flatbuffers_sdk.data_objects.Int32Array import Int32ArrayT  # noqa: E402
from hipdnn_flatbuffers_sdk.data_objects.IntegerArray import IntegerArray  # noqa: E402
from hipdnn_flatbuffers_sdk.data_objects.RealArray import RealArray  # noqa: E402
from uhd_gen.artifact import (  # noqa: E402
    _check_trees,
    threshold_table,
    verify_tree_artifact,
)
from uhd_gen.lgbm_to_flatbuffer import build_gbdt_model  # noqa: E402
from uhd_gen.tests.model_bin import write_model  # noqa: E402


def _booster(trees: int = 60, leaves: int = 31, features: int = 6) -> lgb.Booster:
    rng = np.random.default_rng(5)
    X = rng.random((3000, features))
    y = X @ rng.random(features) + rng.normal(0, 0.05, len(X))
    return lgb.train(
        {"objective": "regression", "num_leaves": leaves, "verbose": -1},
        lgb.Dataset(X, label=y),
        num_boost_round=trees,
    )


def _model(buffer: bytes) -> GbdtModelT:
    return GbdtModelT.InitFromPackedBuf(bytearray(buffer), 0)


def _leaf(value: float) -> dict:
    return {"leaf_value": value}


def _split(feature: int, threshold: float, left: dict, right: dict) -> dict:
    return {
        "split_feature": feature,
        "threshold": threshold,
        "default_left": True,
        "decision_type": "<=",
        "left_child": left,
        "right_child": right,
    }


def _dump(*roots: dict, features: int = 2) -> dict:
    return {
        "tree_info": [{"tree_structure": root} for root in roots],
        "max_feature_idx": features - 1,
        "objective": "regression",
    }


def test_split_leaf_is_the_default_and_a_fraction_of_the_node_array_size():
    dump = _booster().dump_model()
    node_arrays = build_gbdt_model(dump, "sha256:x", encoding="node_arrays")
    split_leaf = build_gbdt_model(dump, "sha256:x")
    assert all(tree.leavesType != 0 for tree in _model(split_leaf).trees)
    assert len(split_leaf) < 0.3 * len(node_arrays), (len(split_leaf), len(node_arrays))


def test_each_array_takes_the_narrowest_type_that_holds_it():
    """Six features and 31-leaf trees: features and children fit a byte, signed for the
    children (a leaf is ~index); the threshold ids are as wide as the table needs."""
    model = _model(build_gbdt_model(_booster().dump_model(), "sha256:x"))
    tree = model.trees[0]
    assert tree.splitFeaturesType == IntegerArray.UInt8Array
    assert tree.splitLeftType == tree.splitRightType == IntegerArray.Int8Array
    expected_ids = (
        IntegerArray.UInt8Array
        if len(model.thresholdTable.values) <= 256
        else IntegerArray.UInt16Array
    )
    assert tree.splitThresholdsType == expected_ids
    assert tree.leavesType == RealArray.Float32Array


@pytest.mark.parametrize(
    ("thresholds", "table_type"),
    [
        pytest.param((0.5, 2.25), RealArray.Float32Array, id="every-value-a-float32"),
        pytest.param((0.1, 2.25), RealArray.Float64Array, id="one-value-not-a-float32"),
    ],
)
def test_thresholds_are_stored_exactly(thresholds, table_type):
    """0.1 is not a float32: the whole table stays float64 rather than move a split."""
    low, high = thresholds
    dump = _dump(
        _split(0, low, _leaf(1.0), _split(1, high, _leaf(2.0), _leaf(3.0))),
    )
    model = _model(build_gbdt_model(dump, "sha256:x"))
    assert model.thresholdTableType == table_type
    assert [float(v) for v in model.thresholdTable.values] == sorted(thresholds)


def test_routing_flags_are_stored_only_when_they_vary():
    dump = _dump(_split(0, 0.5, _leaf(1.0), _leaf(2.0)))
    uniform = _model(build_gbdt_model(dump, "sha256:x")).trees[0]
    assert uniform.splitDefaultLeft is None and uniform.defaultLeftFill
    assert uniform.splitDecisionLte is None

    mixed = _dump(
        _split(
            0,
            0.5,
            {**_split(1, 0.25, _leaf(1.0), _leaf(2.0)), "default_left": False},
            {**_split(1, 0.75, _leaf(3.0), _leaf(4.0)), "decision_type": "<"},
        )
    )
    tree = _model(build_gbdt_model(mixed, "sha256:x")).trees[0]
    assert [bool(v) for v in tree.splitDefaultLeft] == [True, False, True]
    assert [bool(v) for v in tree.splitDecisionLte] == [True, True, False]


def test_a_tree_sending_every_missing_value_right_fills_false():
    dump = _dump({**_split(0, 0.5, _leaf(1.0), _leaf(2.0)), "default_left": False})
    tree = _model(build_gbdt_model(dump, "sha256:x")).trees[0]
    assert tree.splitDefaultLeft is None and not tree.defaultLeftFill


def test_a_single_leaf_tree_is_its_leaf_alone():
    model = _model(build_gbdt_model(_dump(_leaf(4.5)), "sha256:x"))
    tree = model.trees[0]
    assert tree.splitLeftType == IntegerArray.NONE
    assert [float(v) for v in tree.leaves.values] == [4.5]
    assert model.thresholdTableType == RealArray.NONE


@pytest.mark.parametrize(
    "keyword", [{"encoding": "packed"}, {"leaf_precision": "float16"}]
)
def test_an_unknown_encoding_choice_is_refused(keyword):
    with pytest.raises(ValueError, match="must be one of"):
        build_gbdt_model(_dump(_leaf(1.0)), "sha256:x", **keyword)


# ---- The offline verifier refuses what TreeDataAdapter refuses ---------------------


def _two_splits() -> GbdtModelT:
    dump = _dump(_split(0, 0.5, _split(1, 1.5, _leaf(1.0), _leaf(2.0)), _leaf(3.0)))
    return _model(build_gbdt_model(dump, "sha256:x"))


def test_the_unbroken_model_verifies(tmp_path):
    verify_tree_artifact(write_model(_two_splits(), tmp_path / "model.bin"), None)


def _with(change) -> GbdtModelT:
    model = _two_splits()
    change(model, model.trees[0])
    return model


def _int32(values) -> Int32ArrayT:
    array = Int32ArrayT()
    array.values = list(values)
    return array


def _set(tree, field: str, values) -> None:
    setattr(tree, f"{field}Type", IntegerArray.Int32Array)
    setattr(tree, field, _int32(values))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param(
            lambda m, t: _set(t, "splitLeft", [2, ~0]),
            "child index outside the tree",
            id="split-index-past-the-splits",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitRight", [~0, ~3]),
            "child index outside the tree",
            id="leaf-index-past-the-leaves",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitLeft", [0, ~0]),
            "cycle in child indices",
            id="cycle",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitThresholds", [0, 2]),
            "split threshold index outside the threshold table",
            id="threshold-id",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitFeatures", [0, 2]),
            "split feature outside the declared feature count",
            id="feature",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitFeatures", [0, -1]),
            "split feature outside the declared feature count",
            id="negative-feature",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitThresholds", [0, -1]),
            "split threshold index outside the threshold table",
            id="negative-threshold-id",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitFeatures", [0]),
            "split arrays do not hold one entry per split",
            id="short-array",
        ),
        pytest.param(
            lambda m, t: setattr(t, "splitDecisionLte", [True]),
            "split arrays do not hold one entry per split",
            id="short-decision-lte",
        ),
        pytest.param(
            lambda m, t: setattr(t.leaves, "values", [1.0]),
            "a single-leaf tree has split arrays",
            id="single-leaf-with-splits",
        ),
        pytest.param(
            lambda m, t: setattr(t.leaves, "values", []),
            "missing nodes or a required node array",
            id="no-leaves",
        ),
        pytest.param(
            lambda m, t: setattr(t, "splitDefaultLeft", [True]),
            "split arrays do not hold one entry per split",
            id="short-flags",
        ),
        pytest.param(
            lambda m, t: setattr(t, "splitFeaturesType", 9),
            "split_features is stored as integer array type 9, which this runtime "
            "cannot read",
            id="unknown-integer-tag",
        ),
        pytest.param(
            lambda m, t: (
                setattr(t, "splitLeftType", 9),
                setattr(t, "splitRightType", 9),
            ),
            "split children are stored in a type this runtime cannot read",
            id="unknown-children-tag",
        ),
        pytest.param(
            lambda m, t: setattr(t, "leavesType", 7),
            "leaves are stored in a type this runtime cannot read",
            id="unknown-leaves-tag",
        ),
        pytest.param(
            lambda m, t: setattr(m, "thresholdTableType", 7),
            "threshold_table is stored as real array type 7, which this runtime "
            "cannot read",
            id="unknown-table-tag",
        ),
        pytest.param(
            lambda m, t: (
                setattr(m, "thresholdTableType", 0),
                setattr(m, "thresholdTable", None),
            ),
            "split threshold index outside the threshold table",
            id="no-table",
        ),
    ],
)
def test_a_malformed_split_leaf_tree_is_refused_with_the_runtimes_reason(
    tmp_path, change, reason
):
    path = write_model(_with(change), tmp_path / "model.bin")
    with pytest.raises(ValueError, match=reason.replace("(", r"\(")):
        verify_tree_artifact(path, None)


def test_a_non_finite_threshold_or_leaf_is_refused(tmp_path):
    model = _two_splits()
    model.thresholdTable.values = np.asarray([0.5, np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match="split threshold is not finite"):
        verify_tree_artifact(write_model(model, tmp_path / "a.bin"), None)

    model = _two_splits()
    model.trees[0].leaves.values = np.asarray([1.0, np.nan, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="leaf prediction is not finite"):
        verify_tree_artifact(write_model(model, tmp_path / "b.bin"), None)


def test_repeated_split_leaf_trees_cannot_claim_more_nodes_than_the_buffer_holds():
    """Every entry may point at one tree table, so the bytes each claims are bounded together."""
    model = _two_splits()
    table = threshold_table(model, "model")
    _check_trees([model.trees[0]] * 4, table, 2, "model", 0, 64)
    with pytest.raises(
        ValueError, match="ensemble has more nodes than the model buffer can hold"
    ):
        _check_trees([model.trees[0]] * 100, table, 2, "model", 0, 64)


# ---- The verifier port refuses the typed arrays the runtime's verifier refuses ------
#
# Edits of a real converter output (60 trees: UInt16Array threshold ids, Float32Array
# leaves): a union member, or its `values`, copied to the end of the buffer and pointed at.


def _u32(data, at):
    return struct.unpack_from("<I", data, at)[0]


def _field(data, table, slot):
    vtable = table - struct.unpack_from("<i", data, table)[0]
    return table + struct.unpack_from("<H", data, vtable + slot)[0]


def _union_member(data, owner, slot):
    at = _field(data, owner, slot + 2)
    return at + _u32(data, at)


def _first_tree(data):
    at = _field(data, _u32(data, 0), 4)
    first = at + _u32(data, at) + 4
    return first + _u32(data, first)


def _append_at(data, residue, width=4):
    """Pad `data` to an offset of `residue` mod 4 (its elements aligned to `width` when the
    offset itself is aligned), and return it."""
    while len(data) % 4 != residue or (residue == 0 and (len(data) + 4) % width):
        data.append(0)
    return len(data)


def _move_values(data, owner, slot, width, residue):
    """The member's `values` vector, copied to an offset of `residue` mod 4."""
    member = _union_member(data, owner, slot)
    at = _field(data, member, 4)
    vector = at + _u32(data, at)
    blob = bytes(data[vector : vector + 4 + _u32(data, vector) * width])
    moved = _append_at(data, residue, width)
    data += blob
    struct.pack_into("<I", data, at, moved - at)


def _move_member(data, owner, slot, width, residue):
    """The member table itself, copied to an offset of `residue` mod 4; its vector after."""
    at = _field(data, owner, slot + 2)
    member = at + _u32(data, at)
    vtable = member - struct.unpack_from("<i", data, member)[0]
    values_at = _field(data, member, 4)
    vector = values_at + _u32(data, values_at)
    blob = bytes(data[vector : vector + 4 + _u32(data, vector) * width])
    moved = _append_at(data, residue)
    data += bytes(values_at - member + 4)
    struct.pack_into("<i", data, moved, moved - vtable)
    copied = _append_at(data, 0, width)
    data += blob
    moved_values_at = moved + values_at - member
    struct.pack_into("<I", data, moved_values_at, copied - moved_values_at)
    struct.pack_into("<I", data, at, moved - at)


def _claim_past_the_end(data, owner, slot, width):
    """The member's `values` length, one element more than the buffer holds."""
    member = _union_member(data, owner, slot)
    at = _field(data, member, 4)
    vector = at + _u32(data, at)
    struct.pack_into("<I", data, vector, (len(data) - vector - 4) // width + 1)


#: (owner, union slot, element width of the converter output's member)
LEAVES = (_first_tree, 40, 4)
THRESHOLD_IDS = (_first_tree, 22, 2)
TABLE = (lambda data: _u32(data, 0), 30, None)


@pytest.fixture(scope="module")
def converted():
    data = bytearray(build_gbdt_model(_booster().dump_model(), "sha256:x"))
    model = GbdtModel.GetRootAs(data, 0)
    assert model.Trees(0).SplitThresholdsType() == IntegerArray.UInt16Array
    assert model.Trees(0).LeavesType() == RealArray.Float32Array
    table_width = 4 if model.ThresholdTableType() == RealArray.Float32Array else 8
    return bytes(data), table_width


def _edited(converted, edit, member, *args):
    data, table_width = converted
    data = bytearray(data)
    owner, slot, width = member
    edit(data, owner(data), slot, width or table_width, *args)
    return data


def _verify(tmp_path, data):
    path = tmp_path / "model.bin"
    path.write_bytes(bytes(data))
    return verify_tree_artifact(path, None)


@pytest.mark.parametrize(
    "member", [LEAVES, THRESHOLD_IDS, TABLE], ids=["leaves", "ids", "table"]
)
def test_a_moved_but_aligned_member_still_verifies(tmp_path, converted, member):
    _verify(tmp_path, _edited(converted, _move_values, member, 0))


@pytest.mark.parametrize(
    ("edit", "member", "reason"),
    [
        pytest.param(
            _move_values,
            LEAVES,
            r"trees\[0\]\.leaves\.values length at \d+ is not 4-byte aligned",
            id="leaves-values",
        ),
        pytest.param(
            _move_values,
            THRESHOLD_IDS,
            r"trees\[0\]\.split_thresholds\.values length at \d+ is not 4-byte aligned",
            id="threshold-id-values",
        ),
        pytest.param(
            _move_member,
            THRESHOLD_IDS,
            r"trees\[0\]\.split_thresholds vtable offset at \d+ is not 4-byte aligned",
            id="threshold-id-member",
        ),
        pytest.param(
            _move_values,
            TABLE,
            r"threshold_table\.values length at \d+ is not 4-byte aligned",
            id="threshold-table-values",
        ),
    ],
)
def test_a_misaligned_member_is_refused(tmp_path, converted, edit, member, reason):
    with pytest.raises(ValueError, match=reason):
        _verify(tmp_path, _edited(converted, edit, member, 2))


@pytest.mark.parametrize(
    ("member", "what"),
    [
        pytest.param(LEAVES, r"trees\[0\]\.leaves\.values", id="leaves"),
        # Two bytes an id: a length that one-byte ids would fit is still refused.
        pytest.param(THRESHOLD_IDS, r"trees\[0\]\.split_thresholds\.values", id="ids"),
        pytest.param(TABLE, r"threshold_table\.values", id="table"),
    ],
)
def test_a_member_claiming_past_the_buffer_is_refused(
    tmp_path, converted, member, what
):
    with pytest.raises(ValueError, match=f"{what}.* lies outside"):
        _verify(tmp_path, _edited(converted, _claim_past_the_end, member))
