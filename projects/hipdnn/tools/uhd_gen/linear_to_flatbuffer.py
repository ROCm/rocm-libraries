#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write a fitted `LinearRanker` as the FlatBuffer LinearModel read by LinearDataAdapter.

Written through the flatc-generated object API, so fields are matched by name. The bytes are
a pure function of the model and the stamps given, so RFC 0019.13 §10.5's content hash is
reproducible.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import flatbuffers

# Importing the package puts `_generated/` on sys.path; see uhd_gen/__init__.py.
import uhd_gen  # noqa: F401
from hipdnn_flatbuffers_sdk.data_objects.LinearBasisTerm import LinearBasisTermT
from hipdnn_flatbuffers_sdk.data_objects.LinearConfig import LinearConfigT
from hipdnn_flatbuffers_sdk.data_objects.LinearModel import LinearModelT

from .artifact import LINEAR_MODEL_IDENTIFIER
from .lgbm_to_flatbuffer import resolve_training_date
from .rellin import LinearRanker

logger = logging.getLogger(__name__)

#: `LinearModel.framework`: how the coefficients were fitted.
FRAMEWORK = "rellin"


def _config(key, coefficients, floor) -> LinearConfigT:
    entry = LinearConfigT()
    entry.key = [float(value) for value in key]
    entry.coefficients = [float(value) for value in coefficients]
    entry.floor = float(floor)
    return entry


def build_linear_model(
    ranker: LinearRanker,
    features_hash: str,
    num_training_samples: int | None = None,
    training_arches: list[str] | None = None,
    model_version: str | None = None,
    training_date: str | None = None,
) -> bytes:
    """FlatBuffer LinearModel bytes for `ranker`."""
    model = LinearModelT()
    model.numFeatures = int(ranker.num_features)
    model.featuresHash = features_hash
    model.keyFeatures = [int(slot) for slot in ranker.key_features]
    model.familyFeatures = [int(slot) for slot in ranker.family_features]
    model.basis = []
    for j in range(ranker.terms):
        term = LinearBasisTermT()
        term.feature = int(ranker.basis_features[j])
        term.transform = int(ranker.transforms[j])
        term.mean = float(ranker.mean[j])
        term.scale = float(ranker.scale[j])
        term.lo = float(ranker.lo[j])
        term.hi = float(ranker.hi[j])
        model.basis.append(term)
    model.configs = [
        _config(key, coefficients, floor)
        for key, coefficients, floor in zip(
            ranker.config_keys, ranker.config_coefficients, ranker.config_floors
        )
    ]
    model.families = [
        _config(key, coefficients, floor)
        for key, coefficients, floor in zip(
            ranker.family_keys, ranker.family_coefficients, ranker.family_floors
        )
    ]
    model.global_ = _config([], ranker.global_coefficients, ranker.global_floor)

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
    builder.Finish(model.Pack(builder), file_identifier=LINEAR_MODEL_IDENTIFIER)
    return bytes(builder.Output())


def write_linear_model(
    ranker: LinearRanker,
    features_hash: str,
    output_path: str | Path,
    num_training_samples: int | None = None,
    training_arches: list[str] | None = None,
    model_version: str | None = None,
    training_date: str | None = None,
) -> str:
    """Write the model and return the bare hex SHA-256 of the bytes written."""
    buffer = build_linear_model(
        ranker,
        features_hash,
        num_training_samples,
        training_arches=training_arches,
        model_version=model_version,
        training_date=training_date,
    )
    Path(output_path).write_bytes(buffer)
    digest = hashlib.sha256(buffer).hexdigest()
    logger.info(
        "Wrote %s (%d bytes, %d configs, sha256:%s)",
        output_path,
        len(buffer),
        len(ranker.config_floors),
        digest,
    )
    return digest
