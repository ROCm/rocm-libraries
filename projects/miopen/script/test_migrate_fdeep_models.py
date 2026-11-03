#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run with FDEEP_SOURCE pointing to the frugally-deep source checkout (needs keras, tensorflow)."""

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reference_lstm(x, h, c, kernel, recurrent, bias):
    units = h.shape[0]
    for step in x:
        z = step @ kernel + h @ recurrent + bias
        i, f, g, o = (z[k * units : (k + 1) * units] for k in range(4))
        c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
        h = sigmoid(o) * np.tanh(c)
    return h, c


def lstm_initial_state_model():
    rng = np.random.default_rng(0)
    steps, features, units = 3, 2, 4
    kernel = rng.normal(size=(features, 4 * units)).astype("<f4")
    recurrent = rng.normal(size=(units, 4 * units)).astype("<f4")
    bias = rng.normal(size=(4 * units,)).astype("<f4")
    x = rng.normal(size=(steps, features)).astype("<f4")
    h0 = rng.normal(size=(units,)).astype("<f4")
    c0 = rng.normal(size=(units,)).astype("<f4")
    h, c = reference_lstm(
        x.astype(np.float64),
        h0.astype(np.float64),
        c0.astype(np.float64),
        kernel,
        recurrent,
        bias,
    )

    def tensor(values):
        values = np.asarray(values)
        return {"shape": list(values.shape), "values": encoded(values.reshape(-1))}

    def input_layer(name, shape):
        return {
            "class_name": "InputLayer",
            "name": name,
            "config": {"name": name, "batch_input_shape": [None, *shape]},
            "inbound_nodes": [],
        }

    return {
        "architecture": {
            "class_name": "Functional",
            "config": {
                "name": "legacy_lstm",
                "layers": [
                    input_layer("x", [steps, features]),
                    input_layer("h0", [units]),
                    input_layer("c0", [units]),
                    {
                        "class_name": "LSTM",
                        "name": "lstm",
                        "config": {
                            "name": "lstm",
                            "units": units,
                            "activation": "tanh",
                            "recurrent_activation": "sigmoid",
                            "use_bias": True,
                            "return_sequences": False,
                            "return_state": True,
                            "time_major": False,
                            "implementation": 2,
                        },
                        "inbound_nodes": [
                            [["x", 0, 0, {}], ["h0", 0, 0, {}], ["c0", 0, 0, {}]]
                        ],
                    },
                ],
                "input_layers": [["x", 0, 0], ["h0", 0, 0], ["c0", 0, 0]],
                "output_layers": [["lstm", 0, 0], ["lstm", 0, 1], ["lstm", 0, 2]],
            },
        },
        "image_data_format": "channels_last",
        "input_shapes": [[steps, features], [units], [units]],
        "output_shapes": [[units], [units], [units]],
        "trainable_params": {
            "lstm": {
                "weights": encoded(kernel),
                "recurrent_weights": encoded(recurrent),
                "bias": encoded(bias),
            },
        },
        "hash": "preserved-lstm-identifier",
        "tests": [
            {
                "inputs": [tensor(x), tensor(h0), tensor(c0)],
                "outputs": [tensor(h), tensor(h), tensor(c)],
            }
        ],
    }


@unittest.skipUnless(
    os.environ.get("FDEEP_SOURCE"),
    "FDEEP_SOURCE must point to the frugally-deep source checkout",
)
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

    def test_lstm_initial_state_order(self):
        data = lstm_initial_state_model()
        result = migrate(data, self.exporter)
        node = result["architecture"]["config"]["layers"][3]["inbound_nodes"][0]
        self.assertIn("initial_state", node["kwargs"])
        swapped = copy.deepcopy(data)
        connections = swapped["architecture"]["config"]["layers"][3]["inbound_nodes"][0]
        connections[1], connections[2] = connections[2], connections[1]
        with self.assertRaises(AssertionError):
            migrate(swapped, self.exporter)

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
