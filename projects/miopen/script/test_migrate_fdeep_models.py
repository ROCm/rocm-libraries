#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run with FDEEP_SOURCE pointing to the frugally-deep source checkout."""

import base64
import copy
import json
import os
from pathlib import Path
import unittest

import keras
import numpy as np

from migrate_fdeep_models import load_exporter, migrate


def encoded(values):
    return [base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")]


def dense_model():
    return {
        "architecture": {
            "class_name": "Functional",
            "config": {
                "name": "legacy",
                "layers": [
                    {
                        "class_name": "InputLayer",
                        "name": "input",
                        "config": {"name": "input", "batch_input_shape": [None, 2]},
                        "inbound_nodes": [],
                    },
                    {
                        "class_name": "Dense",
                        "name": "dense",
                        "config": {"name": "dense", "units": 1, "use_bias": True},
                        "inbound_nodes": [[["input", 0, 0, {}]]],
                    },
                ],
                "input_layers": [["input", 0, 0]],
                "output_layers": [["dense", 0, 0]],
            },
        },
        "image_data_format": "channels_last",
        "input_shapes": [[2]],
        "output_shapes": [[1]],
        "trainable_params": {
            "dense": {"weights": encoded([2, 3]), "bias": encoded([4])},
        },
        "hash": "preserved-model-identifier",
        "tests": [
            {
                "inputs": [{"shape": [2], "values": encoded([5, 6])}],
                "outputs": [{"shape": [1], "values": encoded([32])}],
            }
        ],
    }


class MigrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exporter = load_exporter(Path(os.environ["FDEEP_SOURCE"]))

    def test_reexport_preserves_artifacts_and_keras_predictions(self):
        original = dense_model()
        snapshot = copy.deepcopy(original)
        result = migrate(original, self.exporter)
        self.assertEqual(original, snapshot)
        self.assertEqual(
            {key: value for key, value in result.items() if key != "architecture"},
            {key: value for key, value in original.items() if key != "architecture"},
        )
        graph = result["architecture"]
        self.assertIn("batch_shape", graph["config"]["layers"][0]["config"])
        self.assertIsInstance(graph["config"]["layers"][1]["inbound_nodes"][0], dict)
        model = keras.models.model_from_json(json.dumps(graph))
        self.assertEqual(model.output_shape, (None, 1))
        self.assertEqual(result, migrate(original, self.exporter))

    def test_rejects_reference_prediction_changes(self):
        data = dense_model()
        data["tests"][0]["outputs"][0]["values"] = encoded([0])
        with self.assertRaises(AssertionError):
            migrate(data, self.exporter)

    def test_rejects_unsupported_connections(self):
        data = dense_model()
        data["architecture"]["config"]["layers"][1]["inbound_nodes"][0][0][3] = {
            "mask": True
        }
        with self.assertRaisesRegex(ValueError, "Unsupported connection arguments"):
            migrate(data, self.exporter)

    def test_rejects_unconsumed_weights(self):
        data = dense_model()
        data["trainable_params"]["unknown"] = {}
        with self.assertRaisesRegex(ValueError, "Unconsumed"):
            migrate(data, self.exporter)

    def test_rejects_changed_shapes(self):
        data = dense_model()
        data["input_shapes"] = [[3]]
        with self.assertRaisesRegex(ValueError, "shapes changed"):
            migrate(data, self.exporter)


if __name__ == "__main__":
    unittest.main()
