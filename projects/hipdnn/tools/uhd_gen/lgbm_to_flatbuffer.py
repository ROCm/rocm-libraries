#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Convert LightGBM model to FlatBuffer GbdtModel format.

The output format is hipdnn_flatbuffers_sdk.data_objects.GbdtModel, read at runtime
by TreeDataAdapter.

Written through the flatc-generated object API. The hand-rolled vtable this
replaced happened to match gbdt_model.fbs, but only by coincidence -- the sibling
writer in uhd_to_flatbuffer.py did not, and shipped unloadable buffers for as long
as it existed. Field identity here is a name, so a schema addition renumbers
nothing.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flatbuffers
import lightgbm as lgb

# Importing the package puts `_generated/` on sys.path; see uhd_gen/__init__.py.
import uhd_gen  # noqa: F401
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT
from hipdnn_flatbuffers_sdk.data_objects.GbdtTree import GbdtTreeT

logger = logging.getLogger(__name__)

# File identifier for GbdtModel FlatBuffers
GBDT_MODEL_FILE_IDENTIFIER = b"HGBM"

def convert(
    lgbm_path: str | Path,
    features_hash: str,
    output_path: str | Path,
    num_training_samples: int | None = None,
    training_arches: list[str] | None = None,
    model_version: str | None = None,
) -> None:
    """Convert LightGBM model to FlatBuffer GbdtModel.

    Args:
        lgbm_path: Path to .lgbm model file.
        features_hash: SHA-256 hash of feature specification.
        output_path: Output path for .bin FlatBuffer file.
        num_training_samples: Optional number of training samples for metadata.
        training_arches: Optional list of GPU architectures the model was trained
            on (e.g., ["gfx942", "gfx1100"]). Used for RFC 0019 §9.2 out-of-distribution
            detection at runtime.
        model_version: Optional semantic version (e.g., "1.0.0").
    """
    model = lgb.Booster(model_file=str(lgbm_path))
    model_json = model.dump_model()

    buffer = build_gbdt_model(
        model_json,
        features_hash,
        num_training_samples,
        training_arches=training_arches,
        model_version=model_version,
    )

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
    training_arches: list[str] | None = None,
    model_version: str | None = None,
) -> bytes:
    """Build FlatBuffer GbdtModel from LightGBM model JSON.

    Args:
        model_json: Output of lgb.Booster.dump_model().
        features_hash: SHA-256 hash of feature specification.
        num_training_samples: Optional number of training samples.
        training_arches: GPU architectures the model was trained on.
        model_version: Semantic version string.

    Returns:
        FlatBuffer bytes for GbdtModel.
    """
    model = GbdtModelT()
    model.trees = [
        _build_tree(tree_info["tree_structure"]) for tree_info in model_json["tree_info"]
    ]
    model.numFeatures = model_json["max_feature_idx"] + 1
    model.featuresHash = features_hash
    model.baseScore = 0.0

    # LightGBM folds the shrinkage into the dumped leaf values, so the ensemble is
    # summed with a unit rate. TreeDataAdapter::score() matches this by not applying
    # a rate of its own; the field is provenance only.
    model.learningRate = 1.0

    model.framework = "lightgbm"
    model.trainingDate = datetime.now(timezone.utc).isoformat()
    model.trainingObjective = _objective_name(model_json)

    if num_training_samples is not None:
        model.numTrainingSamples = num_training_samples
    if training_arches:
        model.trainingArches = list(training_arches)
    if model_version:
        model.modelVersion = model_version

    builder = flatbuffers.Builder(1024 * 1024)
    builder.Finish(model.Pack(builder), file_identifier=GBDT_MODEL_FILE_IDENTIFIER)
    return bytes(builder.Output())


def _objective_name(model_json: dict[str, Any]) -> str:
    """Extract the bare objective name from a LightGBM model dump.

    LightGBM 4.x reports `objective` as a string, sometimes carrying trailing
    parameters (e.g. "binary sigmoid:1"); only the leading token names the
    objective. Older 2.x/3.x dumps used a {"name": ..., "config": ...} mapping.
    Both are accepted so an artifact produced against either version converts.
    """
    objective = model_json.get("objective")

    if isinstance(objective, str):
        # "binary sigmoid:1" -> "binary"; a bare "regression" is unchanged.
        name = objective.split()[0] if objective.split() else ""
        return name or "regression"

    if isinstance(objective, dict):
        return str(objective.get("name") or "regression")

    return "regression"


def _build_tree(root_node: dict[str, Any]) -> GbdtTreeT:
    """Build a GbdtTreeT from a LightGBM tree structure.

    LightGBM stores trees as nested dicts. We flatten to parallel arrays for
    cache-friendly traversal matching TreeDataAdapter's evaluation.
    """
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
        # LightGBM decision_type: "<=" means go left if feature <= threshold (default)
        # decision_type can be "==" for categorical, but we treat those as <=
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
