#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convert a LightGBM model to the FlatBuffer GbdtModel read by TreeDataAdapter.

Written through the flatc-generated object API, so fields are matched by name.

Trees are written in the split/leaf encoding (gbdt_model.fbs) unless `encoding="node_arrays"`
asks for the original one, which runtimes without split/leaf support still read. In the
split/leaf encoding every array is stored in the narrowest type that holds its values exactly,
and the model's union tags say which, so the reader needs no out-of-band schema:

* split features, children and threshold indices as int8/uint8, int16/uint16 or int32;
* thresholds once each, in one model-wide table, float32 when every one round-trips exactly
  and float64 otherwise, so a split compares against exactly the value LightGBM chose;
* leaf values as float32 (`leaf_precision="float64"` keeps them exact): a leaf's relative
  error is at most 2**-24, far below the accuracy of any measured throughput;
* the per-split routing flags only when they are not all the default.

For a 500-tree, 127-leaf model this is about a third of the node-array size.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flatbuffers
import lightgbm as lgb
import numpy as np

# Importing the package puts `_generated/` on sys.path; see uhd_gen/__init__.py.
import uhd_gen  # noqa: F401
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT
from hipdnn_flatbuffers_sdk.data_objects.GbdtGroup import GbdtGroupT
from hipdnn_flatbuffers_sdk.data_objects.GbdtTree import GbdtTreeT
from hipdnn_flatbuffers_sdk.data_objects.IntegerArray import IntegerArray
from hipdnn_flatbuffers_sdk.data_objects.RealArray import RealArray
from hipdnn_flatbuffers_sdk.data_objects.Int8Array import Int8ArrayT
from hipdnn_flatbuffers_sdk.data_objects.UInt8Array import UInt8ArrayT
from hipdnn_flatbuffers_sdk.data_objects.Int16Array import Int16ArrayT
from hipdnn_flatbuffers_sdk.data_objects.UInt16Array import UInt16ArrayT
from hipdnn_flatbuffers_sdk.data_objects.Int32Array import Int32ArrayT
from hipdnn_flatbuffers_sdk.data_objects.Float32Array import Float32ArrayT
from hipdnn_flatbuffers_sdk.data_objects.Float64Array import Float64ArrayT

logger = logging.getLogger(__name__)

# File identifier for GbdtModel FlatBuffers
GBDT_MODEL_FILE_IDENTIFIER = b"HGBM"

#: Tree encodings `build_gbdt_model` writes (gbdt_model.fbs).
SPLIT_LEAF = "split_leaf"
NODE_ARRAYS = "node_arrays"
ENCODINGS = (SPLIT_LEAF, NODE_ARRAYS)
LEAF_PRECISIONS = ("float32", "float64")

#: Integer array types, narrowest first: (union tag, object class, numpy dtype).
_INTEGER_TYPES = (
    (IntegerArray.UInt8Array, UInt8ArrayT, np.uint8),
    (IntegerArray.Int8Array, Int8ArrayT, np.int8),
    (IntegerArray.UInt16Array, UInt16ArrayT, np.uint16),
    (IntegerArray.Int16Array, Int16ArrayT, np.int16),
    (IntegerArray.Int32Array, Int32ArrayT, np.int32),
)

#: Reproducible-builds stamp source: converting the same model twice must give the same
#: bytes, since RFC 0019.13 §10.5 records a content hash over them.
SOURCE_DATE_EPOCH = "SOURCE_DATE_EPOCH"


def resolve_training_date(training_date: str | None = None) -> str | None:
    """The stamp to record, or None when there is no truthful one.

    Never wall-clock time: that would make every conversion's bytes differ.
    """
    if training_date is not None:
        return training_date
    epoch = os.environ.get(SOURCE_DATE_EPOCH, "").strip()
    if not epoch:
        return None
    try:
        seconds = int(epoch)
    except ValueError:
        raise ValueError(
            f"{SOURCE_DATE_EPOCH} must be an integer count of seconds; got {epoch!r}"
        ) from None
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def convert(
    lgbm_path: str | Path,
    features_hash: str,
    output_path: str | Path,
    num_training_samples: int | None = None,
    training_arches: list[str] | None = None,
    model_version: str | None = None,
    training_date: str | None = None,
    group_by_feature_index: int = -1,
    group_models: list[tuple[float, "lgb.Booster"]] | None = None,
    encoding: str = SPLIT_LEAF,
    leaf_precision: str = "float32",
) -> str:
    """Convert a LightGBM model file to FlatBuffer GbdtModel and write it.

    `training_arches` feeds RFC 0019 §9.2 out-of-distribution detection.
    `group_models` is one `(grouping value, booster)` per layer-2 group; None writes a
    single-layer model. Returns the bare hex SHA-256 of the bytes written.
    """
    model = lgb.Booster(model_file=str(lgbm_path))
    model_json = model.dump_model()

    buffer = build_gbdt_model(
        model_json,
        features_hash,
        num_training_samples,
        training_arches=training_arches,
        model_version=model_version,
        training_date=training_date,
        group_by_feature_index=group_by_feature_index,
        groups=(
            None
            if not group_models
            else [(value, booster.dump_model()) for value, booster in group_models]
        ),
        encoding=encoding,
        leaf_precision=leaf_precision,
    )

    Path(output_path).write_bytes(buffer)
    digest = hashlib.sha256(buffer).hexdigest()

    logger.info(
        "Converted %s to %s (%d bytes, %d trees, sha256:%s)",
        lgbm_path,
        output_path,
        len(buffer),
        len(model_json["tree_info"]),
        digest,
    )
    return digest


def build_gbdt_model(
    model_json: dict[str, Any],
    features_hash: str,
    num_training_samples: int | None = None,
    training_arches: list[str] | None = None,
    model_version: str | None = None,
    training_date: str | None = None,
    group_by_feature_index: int = -1,
    groups: list[tuple[float, dict[str, Any]]] | None = None,
    encoding: str = SPLIT_LEAF,
    leaf_precision: str = "float32",
) -> bytes:
    """Build FlatBuffer GbdtModel bytes from `lgb.Booster.dump_model()` output.

    `groups` is one `(grouping value, dumped ensemble)` per group, in layer-1 order;
    None writes a single-layer model. `encoding` and `leaf_precision` are as in the
    module docstring.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    if leaf_precision not in LEAF_PRECISIONS:
        raise ValueError(
            f"leaf_precision must be one of {LEAF_PRECISIONS}, got {leaf_precision!r}"
        )
    structures = [info["tree_structure"] for info in model_json["tree_info"]]
    group_structures = [
        (value, [info["tree_structure"] for info in group_json["tree_info"]])
        for value, group_json in (groups or [])
    ]

    model = GbdtModelT()
    if encoding == SPLIT_LEAF:
        table = _threshold_table(
            structures + [tree for _, trees in group_structures for tree in trees]
        )
        # Keyed by the float64 value LightGBM chose; a float32 table holds it exactly.
        index = {float(value): position for position, value in enumerate(table.values)}
        build = lambda root: _build_split_leaf_tree(
            root, index, leaf_precision
        )  # noqa: E731
        if len(table.values):
            model.thresholdTableType = (
                RealArray.Float32Array
                if isinstance(table, Float32ArrayT)
                else RealArray.Float64Array
            )
            model.thresholdTable = table
    else:
        build = _build_tree
    model.trees = [build(root) for root in structures]
    model.numFeatures = model_json["max_feature_idx"] + 1
    model.featuresHash = features_hash
    model.baseScore = 0.0

    # LightGBM folds the shrinkage into the dumped leaf values, so the ensemble is
    # summed with a unit rate. TreeDataAdapter::score() matches this by not applying
    # a rate of its own; the field is provenance only.
    model.learningRate = 1.0

    model.framework = "lightgbm"
    model.trainingObjective = _objective_name(model_json)

    stamp = resolve_training_date(training_date)
    if stamp is not None:
        model.trainingDate = stamp

    if num_training_samples is not None:
        model.numTrainingSamples = num_training_samples
    if training_arches:
        model.trainingArches = list(training_arches)
    if model_version:
        model.modelVersion = model_version

    # Layer 2, when one ensemble was trained per group; absent means single-layer.
    if groups:
        model.groupByFeatureIndex = group_by_feature_index
        model.groups = []
        for value, trees in group_structures:
            group = GbdtGroupT()
            group.value = float(value)
            group.trees = [build(root) for root in trees]
            model.groups.append(group)

    builder = flatbuffers.Builder(1024 * 1024)
    builder.Finish(model.Pack(builder), file_identifier=GBDT_MODEL_FILE_IDENTIFIER)
    return bytes(builder.Output())


def _objective_name(model_json: dict[str, Any]) -> str:
    """The bare objective name from a LightGBM 2.x-4.x model dump."""
    objective = model_json.get("objective")

    if isinstance(objective, str):
        # "binary sigmoid:1" -> "binary"; a bare "regression" is unchanged.
        name = objective.split()[0] if objective.split() else ""
        return name or "regression"

    if isinstance(objective, dict):
        return str(objective.get("name") or "regression")

    return "regression"


def _build_tree(root_node: dict[str, Any]) -> GbdtTreeT:
    """Flatten a LightGBM nested tree into the parallel arrays TreeDataAdapter reads."""
    nodes: list[dict[str, Any]] = []
    _flatten_tree(root_node, nodes)

    tree = GbdtTreeT()
    tree.featureIndices = [n.get("split_feature", -1) for n in nodes]
    tree.thresholds = [n.get("threshold", 0.0) for n in nodes]
    tree.leftChildren = [n.get("left_idx", -1) for n in nodes]
    tree.rightChildren = [n.get("right_idx", -1) for n in nodes]
    tree.leafValues = [n.get("leaf_value", 0.0) for n in nodes]
    tree.defaultLeft = [n.get("default_left", True) for n in nodes]
    tree.decisionLte = [n.get("decision_lte", True) for n in nodes]
    return tree


def _integers(values) -> tuple[int, Any]:
    """(union tag, object) of the narrowest integer array type holding every value."""
    array = np.asarray(values, dtype=np.int64)
    low = int(array.min()) if len(array) else 0
    high = int(array.max()) if len(array) else 0
    for tag, cls, dtype in _INTEGER_TYPES:
        limits = np.iinfo(dtype)
        if limits.min <= low and high <= limits.max:
            typed = cls()
            typed.values = array.astype(dtype)
            return tag, typed
    raise ValueError(f"values {low}..{high} do not fit an int32 array")


def _threshold_table(roots: list[dict[str, Any]]):
    """The distinct split thresholds, sorted: float32 when every one is exactly a float32."""
    found: set[float] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if "leaf_value" not in node:
            found.add(float(node["threshold"]))
            stack += [node["left_child"], node["right_child"]]
    values = np.asarray(sorted(found), dtype=np.float64)
    narrow = values.astype(np.float32)
    if np.array_equal(narrow.astype(np.float64), values):
        table = Float32ArrayT()
        table.values = narrow
    else:
        table = Float64ArrayT()
        table.values = values
    return table


def _build_split_leaf_tree(
    root: dict[str, Any], threshold_index: dict[float, int], leaf_precision: str
) -> GbdtTreeT:
    """One tree in the split/leaf encoding: splits in preorder from 0, leaves in the order
    a preorder walk meets them; a child is its split index, or ~leaf for a leaf."""
    features: list[int] = []
    thresholds: list[int] = []
    lefts: list[int] = []
    rights: list[int] = []
    default_left: list[bool] = []
    decision_lte: list[bool] = []
    leaves: list[float] = []

    def walk(node: dict[str, Any]) -> int:
        if "leaf_value" in node:
            leaves.append(float(node["leaf_value"]))
            return ~(len(leaves) - 1)
        split = len(features)
        features.append(int(node["split_feature"]))
        thresholds.append(threshold_index[float(node["threshold"])])
        default_left.append(bool(node.get("default_left", True)))
        # Categorical "==" splits are treated as "<=", as in the node-array encoding.
        decision_lte.append(node.get("decision_type", "<=") in ("<=", "=="))
        lefts.append(0)
        rights.append(0)
        lefts[split] = walk(node["left_child"])
        rights[split] = walk(node["right_child"])
        return split

    walk(root)
    tree = GbdtTreeT()
    if features:
        tree.splitFeaturesType, tree.splitFeatures = _integers(features)
        tree.splitThresholdsType, tree.splitThresholds = _integers(thresholds)
        # Both children in one type, so a split's two sides read alike.
        tag, joined = _integers(lefts + rights)
        tree.splitLeftType, tree.splitLeft = tag, type(joined)()
        tree.splitLeft.values = joined.values[: len(lefts)]
        tree.splitRightType, tree.splitRight = tag, type(joined)()
        tree.splitRight.values = joined.values[len(lefts) :]
        if all(default_left) or not any(default_left):
            tree.defaultLeftFill = bool(default_left[0])
        else:
            tree.splitDefaultLeft = default_left
        if not all(decision_lte):
            tree.splitDecisionLte = decision_lte
    if leaf_precision == "float32":
        tree.leavesType, tree.leaves = RealArray.Float32Array, Float32ArrayT()
        tree.leaves.values = np.asarray(leaves, dtype=np.float32)
    else:
        tree.leavesType, tree.leaves = RealArray.Float64Array, Float64ArrayT()
        tree.leaves.values = np.asarray(leaves, dtype=np.float64)
    return tree


def _flatten_tree(node: dict[str, Any], nodes: list[dict[str, Any]]) -> int:
    """DFS-flatten a LightGBM tree into `nodes`; returns this node's index."""
    current_idx = len(nodes)

    if "leaf_value" in node:
        nodes.append(
            {
                "leaf_value": node["leaf_value"],
                "left_idx": -1,
                "right_idx": -1,
            }
        )
    else:
        nodes.append({})
        left_idx = _flatten_tree(node["left_child"], nodes)
        right_idx = _flatten_tree(node["right_child"], nodes)
        # Categorical "==" splits are treated as "<=".
        decision_type = node.get("decision_type", "<=")
        use_lte = decision_type in ("<=", "==")
        nodes[current_idx] = {
            "split_feature": node["split_feature"],
            "threshold": node["threshold"],
            "default_left": node.get("default_left", True),
            "decision_lte": use_lte,
            "left_idx": left_idx,
            "right_idx": right_idx,
        }

    return current_idx
