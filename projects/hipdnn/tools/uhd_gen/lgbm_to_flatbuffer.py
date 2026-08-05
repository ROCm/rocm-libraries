#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convert LightGBM model to FlatBuffer GbdtModel format.

The output format matches hipdnn_flatbuffers_sdk.data_objects.GbdtModel,
which is read by TreeDataAdapter at runtime.
"""
from __future__ import annotations

import logging
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flatbuffers
import lightgbm as lgb

logger = logging.getLogger(__name__)

# File identifier for GbdtModel FlatBuffers
GBDT_MODEL_FILE_IDENTIFIER = b"HGBM"


def convert(
    lgbm_path: str | Path,
    features_hash: str,
    output_path: str | Path,
    num_training_samples: int | None = None,
) -> None:
    """Convert LightGBM model to FlatBuffer GbdtModel.

    Args:
        lgbm_path: Path to .lgbm model file.
        features_hash: SHA-256 hash of feature specification.
        output_path: Output path for .bin FlatBuffer file.
        num_training_samples: Optional number of training samples for metadata.
    """
    model = lgb.Booster(model_file=str(lgbm_path))
    model_json = model.dump_model()

    buffer = build_gbdt_model(model_json, features_hash, num_training_samples)

    with open(output_path, "wb") as f:
        f.write(buffer)

    logger.info(
        "Converted %s to %s (%d bytes, %d trees)",
        lgbm_path,
        output_path,
        len(buffer),
        len(model_json["tree_info"]),
    )


def build_gbdt_model(
    model_json: dict[str, Any],
    features_hash: str,
    num_training_samples: int | None = None,
) -> bytes:
    """Build FlatBuffer GbdtModel from LightGBM model JSON.

    Args:
        model_json: Output of lgb.Booster.dump_model().
        features_hash: SHA-256 hash of feature specification.
        num_training_samples: Optional number of training samples.

    Returns:
        FlatBuffer bytes for GbdtModel.
    """
    builder = flatbuffers.Builder(1024 * 1024)

    tree_offsets = []
    for tree_info in model_json["tree_info"]:
        tree_offset = _build_tree(builder, tree_info["tree_structure"])
        tree_offsets.append(tree_offset)

    trees_vec = _create_offset_vector(builder, tree_offsets)

    hash_offset = builder.CreateString(features_hash)
    framework_offset = builder.CreateString("lightgbm")
    training_date_offset = builder.CreateString(
        datetime.now(timezone.utc).isoformat()
    )
    objective_offset = builder.CreateString(
        model_json.get("objective", {}).get("name", "regression")
    )

    learning_rate = 1.0
    if "objective" in model_json and "config" in model_json["objective"]:
        learning_rate = float(
            model_json["objective"]["config"].get("learning_rate", 1.0)
        )

    _start_gbdt_model(builder)
    _add_gbdt_model_trees(builder, trees_vec)
    _add_gbdt_model_num_features(builder, model_json["max_feature_idx"] + 1)
    _add_gbdt_model_features_hash(builder, hash_offset)
    _add_gbdt_model_base_score(builder, 0.0)
    _add_gbdt_model_learning_rate(builder, learning_rate)
    _add_gbdt_model_framework(builder, framework_offset)
    _add_gbdt_model_training_date(builder, training_date_offset)
    if num_training_samples is not None:
        _add_gbdt_model_num_training_samples(builder, num_training_samples)
    _add_gbdt_model_training_objective(builder, objective_offset)
    model_offset = _end_gbdt_model(builder)

    builder.Finish(model_offset, file_identifier=GBDT_MODEL_FILE_IDENTIFIER)
    return bytes(builder.Output())


def _build_tree(builder: flatbuffers.Builder, root_node: dict[str, Any]) -> int:
    """Build GbdtTree FlatBuffer from LightGBM tree structure.

    LightGBM stores trees as nested dicts. We flatten to arrays for
    cache-friendly traversal matching TreeDataAdapter's evaluation.
    """
    nodes: list[dict[str, Any]] = []
    _flatten_tree(root_node, nodes)

    feature_indices = [n.get("split_feature", -1) for n in nodes]
    thresholds = [n.get("threshold", 0.0) for n in nodes]
    left_children = [n.get("left_idx", -1) for n in nodes]
    right_children = [n.get("right_idx", -1) for n in nodes]
    leaf_values = [n.get("leaf_value", 0.0) for n in nodes]
    default_left = [n.get("default_left", True) for n in nodes]

    fi_vec = _create_int32_vector(builder, feature_indices)
    th_vec = _create_double_vector(builder, thresholds)
    lc_vec = _create_int32_vector(builder, left_children)
    rc_vec = _create_int32_vector(builder, right_children)
    lv_vec = _create_double_vector(builder, leaf_values)
    dl_vec = _create_bool_vector(builder, default_left)

    _start_gbdt_tree(builder)
    _add_gbdt_tree_feature_indices(builder, fi_vec)
    _add_gbdt_tree_thresholds(builder, th_vec)
    _add_gbdt_tree_left_children(builder, lc_vec)
    _add_gbdt_tree_right_children(builder, rc_vec)
    _add_gbdt_tree_leaf_values(builder, lv_vec)
    _add_gbdt_tree_default_left(builder, dl_vec)
    return _end_gbdt_tree(builder)


def _flatten_tree(node: dict[str, Any], nodes: list[dict[str, Any]]) -> int:
    """DFS flatten LightGBM tree to node array with child indices.

    Returns the index of this node in the nodes list.
    """
    current_idx = len(nodes)

    if "leaf_value" in node:
        nodes.append({
            "leaf_value": node["leaf_value"],
            "left_idx": -1,
            "right_idx": -1,
        })
    else:
        nodes.append({})
        left_idx = _flatten_tree(node["left_child"], nodes)
        right_idx = _flatten_tree(node["right_child"], nodes)
        nodes[current_idx] = {
            "split_feature": node["split_feature"],
            "threshold": node["threshold"],
            "default_left": node.get("default_left", True),
            "left_idx": left_idx,
            "right_idx": right_idx,
        }

    return current_idx


def _create_int32_vector(builder: flatbuffers.Builder, values: list[int]) -> int:
    """Create FlatBuffer vector of int32."""
    builder.StartVector(4, len(values), 4)
    for v in reversed(values):
        builder.PrependInt32(v)
    return builder.EndVector()


def _create_double_vector(builder: flatbuffers.Builder, values: list[float]) -> int:
    """Create FlatBuffer vector of double (float64)."""
    builder.StartVector(8, len(values), 8)
    for v in reversed(values):
        builder.PrependFloat64(v)
    return builder.EndVector()


def _create_bool_vector(builder: flatbuffers.Builder, values: list[bool]) -> int:
    """Create FlatBuffer vector of bool."""
    builder.StartVector(1, len(values), 1)
    for v in reversed(values):
        builder.PrependBool(v)
    return builder.EndVector()


def _create_offset_vector(builder: flatbuffers.Builder, offsets: list[int]) -> int:
    """Create FlatBuffer vector of offsets (for nested tables)."""
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


# GbdtTree table helpers (inline since we don't have generated Python bindings)
def _start_gbdt_tree(builder: flatbuffers.Builder) -> None:
    builder.StartObject(6)


def _add_gbdt_tree_feature_indices(builder: flatbuffers.Builder, vec: int) -> None:
    builder.PrependUOffsetTRelativeSlot(0, vec, 0)


def _add_gbdt_tree_thresholds(builder: flatbuffers.Builder, vec: int) -> None:
    builder.PrependUOffsetTRelativeSlot(1, vec, 0)


def _add_gbdt_tree_left_children(builder: flatbuffers.Builder, vec: int) -> None:
    builder.PrependUOffsetTRelativeSlot(2, vec, 0)


def _add_gbdt_tree_right_children(builder: flatbuffers.Builder, vec: int) -> None:
    builder.PrependUOffsetTRelativeSlot(3, vec, 0)


def _add_gbdt_tree_leaf_values(builder: flatbuffers.Builder, vec: int) -> None:
    builder.PrependUOffsetTRelativeSlot(4, vec, 0)


def _add_gbdt_tree_default_left(builder: flatbuffers.Builder, vec: int) -> None:
    builder.PrependUOffsetTRelativeSlot(5, vec, 0)


def _end_gbdt_tree(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()


# GbdtModel table helpers
def _start_gbdt_model(builder: flatbuffers.Builder) -> None:
    builder.StartObject(9)


def _add_gbdt_model_trees(builder: flatbuffers.Builder, vec: int) -> None:
    builder.PrependUOffsetTRelativeSlot(0, vec, 0)


def _add_gbdt_model_num_features(builder: flatbuffers.Builder, n: int) -> None:
    builder.PrependInt32Slot(1, n, 0)


def _add_gbdt_model_features_hash(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(2, s, 0)


def _add_gbdt_model_base_score(builder: flatbuffers.Builder, v: float) -> None:
    builder.PrependFloat64Slot(3, v, 0.0)


def _add_gbdt_model_learning_rate(builder: flatbuffers.Builder, v: float) -> None:
    builder.PrependFloat64Slot(4, v, 1.0)


def _add_gbdt_model_framework(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(5, s, 0)


def _add_gbdt_model_training_date(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(6, s, 0)


def _add_gbdt_model_num_training_samples(builder: flatbuffers.Builder, n: int) -> None:
    builder.PrependInt64Slot(7, n, 0)


def _add_gbdt_model_training_objective(builder: flatbuffers.Builder, s: int) -> None:
    builder.PrependUOffsetTRelativeSlot(8, s, 0)


def _end_gbdt_model(builder: flatbuffers.Builder) -> int:
    return builder.EndObject()
