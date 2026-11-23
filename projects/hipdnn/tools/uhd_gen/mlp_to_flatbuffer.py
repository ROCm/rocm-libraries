#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write a fitted `MlpEnsemble` as the FlatBuffer MlpModel read by MlpDataAdapter.

Written through the flatc-generated object API, so fields are matched by name. The bytes are
a pure function of the model and the stamps given, so RFC 0019.13 §10.5's content hash is
reproducible.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import flatbuffers
import numpy as np

# Importing the package puts `_generated/` on sys.path; see uhd_gen/__init__.py.
import uhd_gen  # noqa: F401
from hipdnn_flatbuffers_sdk.data_objects.MlpInput import MlpInputT
from hipdnn_flatbuffers_sdk.data_objects.MlpLayer import MlpLayerT
from hipdnn_flatbuffers_sdk.data_objects.MlpModel import MlpModelT
from hipdnn_flatbuffers_sdk.data_objects.MlpNetwork import MlpNetworkT

from .artifact import MLP_MODEL_IDENTIFIER
from .lgbm_to_flatbuffer import resolve_training_date
from .mlp import MlpEnsemble

logger = logging.getLogger(__name__)

#: `MlpModel.framework`: how the networks were fitted.
FRAMEWORK = "uhd_gen.mlp"


def build_mlp_model(
    ensemble: MlpEnsemble,
    metric: str,
    features_hash: str,
    num_training_samples: int | None = None,
    training_arches: list[str] | None = None,
    model_version: str | None = None,
    training_date: str | None = None,
) -> bytes:
    """FlatBuffer MlpModel bytes for `ensemble`, which predicts the log of `metric`."""
    model = MlpModelT()
    model.numFeatures = int(ensemble.num_features)
    model.metric = metric
    model.featuresHash = features_hash
    model.inputs = []
    for slot, mean, scale in zip(
        ensemble.input_slots, ensemble.input_mean, ensemble.input_scale
    ):
        entry = MlpInputT()
        entry.feature = int(slot)
        entry.mean = float(mean)
        entry.scale = float(scale)
        model.inputs.append(entry)
    model.missingFlags = [int(slot) for slot in ensemble.missing_slots]
    model.networks = []
    for layers in ensemble.networks:
        network = MlpNetworkT()
        network.layers = []
        for weights, biases in layers:
            layer = MlpLayerT()
            layer.outputs, layer.inputs = (int(n) for n in weights.shape)
            # Row-major, one row per output, as the schema fixes.
            layer.weights = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
            layer.biases = np.ascontiguousarray(biases, dtype=np.float32)
            network.layers.append(layer)
        model.networks.append(network)
    model.targetMean = float(ensemble.target_mean)
    model.targetScale = float(ensemble.target_scale)

    model.framework = FRAMEWORK
    stamp = resolve_training_date(training_date)
    if stamp is not None:
        model.trainingDate = stamp
    if num_training_samples is not None:
        model.numTrainingSamples = int(num_training_samples)
    if training_arches:
        model.trainingArches = list(training_arches)
    if model_version:
        model.modelVersion = model_version

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=MLP_MODEL_IDENTIFIER)
    return bytes(builder.Output())


def write_mlp_model(
    ensemble: MlpEnsemble,
    metric: str,
    features_hash: str,
    output_path: str | Path,
    num_training_samples: int | None = None,
    training_arches: list[str] | None = None,
    model_version: str | None = None,
    training_date: str | None = None,
) -> str:
    """Write the model and return the bare hex SHA-256 of the bytes written."""
    buffer = build_mlp_model(
        ensemble,
        metric,
        features_hash,
        num_training_samples,
        training_arches=training_arches,
        model_version=model_version,
        training_date=training_date,
    )
    Path(output_path).write_bytes(buffer)
    digest = hashlib.sha256(buffer).hexdigest()
    logger.info(
        "Wrote %s (%d bytes, %d network(s), sha256:%s)",
        output_path,
        len(buffer),
        len(ensemble.networks),
        digest,
    )
    return digest
