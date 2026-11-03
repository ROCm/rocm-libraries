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


def functional(name, layers, inputs, outputs):
    return {
        "class_name": "Functional",
        "config": {
            "name": name,
            "layers": layers,
            "input_layers": [[layer, 0, 0] for layer in inputs],
            "output_layers": [[layer, 0, 0] for layer in outputs],
        },
    }


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


def embedding_model():
    table = np.arange(10, dtype="<f4").reshape(5, 2) / 4 - 1
    indices = np.array([4, 0, 2], dtype="<f4")
    embedding = {
        "class_name": "Embedding",
        "name": "embedding",
        "config": {
            "name": "embedding",
            "batch_input_shape": [None, 3],
            "input_dim": 5,
            "output_dim": 2,
            "input_length": 3,
            "mask_zero": False,
        },
        "inbound_nodes": [[["indices", 0, 0, {}]]],
    }
    return {
        "architecture": functional(
            "legacy_embedding",
            [input_layer("indices", [3]), embedding],
            ["indices"],
            ["embedding"],
        ),
        "image_data_format": "channels_last",
        "input_shapes": [[3]],
        "output_shapes": [[3, 2]],
        "trainable_params": {"embedding": {"weights": encoded(table)}},
        "hash": "preserved-embedding-identifier",
        "tests": [
            {
                "inputs": [tensor(indices)],
                "outputs": [tensor(table[indices.astype(int)])],
            }
        ],
    }


def layer_norm_residual_model():
    gamma = np.array([1, 2, 0.5, -1], dtype="<f4")
    beta = np.array([0.1, 0, -0.2, 0.3], dtype="<f4")
    x = np.array([1, -2, 3, 0.5], dtype="<f4")
    epsilon = 1e-3
    normalized = (x - x.mean()) / np.sqrt(x.var() + epsilon) * gamma + beta
    expected = np.maximum(normalized, 0) + x
    layers = [
        input_layer("x", [4]),
        {
            "class_name": "LayerNormalization",
            "name": "norm",
            "config": {
                "name": "norm",
                "axis": [1],
                "epsilon": epsilon,
                "center": True,
                "scale": True,
            },
            "inbound_nodes": [[["x", 0, 0, {}]]],
        },
        {
            "class_name": "ReLU",
            "name": "relu",
            "config": {"name": "relu"},
            "inbound_nodes": [[["norm", 0, 0, {}]]],
        },
        {
            "class_name": "Add",
            "name": "add",
            "config": {"name": "add"},
            "inbound_nodes": [[["relu", 0, 0, {}], ["x", 0, 0, {}]]],
        },
    ]
    return {
        "architecture": functional("legacy_norm", layers, ["x"], ["add"]),
        "image_data_format": "channels_last",
        "input_shapes": [[4]],
        "output_shapes": [[4]],
        "trainable_params": {"norm": {"gamma": encoded(gamma), "beta": encoded(beta)}},
        "hash": "preserved-norm-identifier",
        "tests": [{"inputs": [tensor(x)], "outputs": [tensor(expected)]}],
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

    def test_embedding_and_layer_norm_residual_graphs(self):
        for data in (embedding_model(), layer_norm_residual_model()):
            with self.subTest(model=data["hash"]):
                result = migrate(data, self.exporter)
                self.assertEqual(result["tests"], data["tests"])
                self.assertEqual(result["trainable_params"], data["trainable_params"])
                keras.models.model_from_json(json.dumps(result["architecture"]))

    def test_rejects_unsupported_graphs(self):
        def layer(data, index=1):
            return data["architecture"]["config"]["layers"][index]

        cases = {
            "Only bundled Functional": lambda d: d["architecture"].update(
                class_name="Sequential"
            ),
            "Unsupported layer type": lambda d: layer(d).update(class_name="Conv2D"),
            "Invalid legacy connection": lambda d: layer(d).update(
                inbound_nodes=[[["input"]]]
            ),
            "Unexpected inputs": lambda d: layer(d).update(inbound_nodes=[[]]),
            "Unexpected weights": lambda d: d["trainable_params"]["dense"].pop("bias"),
        }
        for message, mutate in cases.items():
            with self.subTest(message=message):
                data = dense_model()
                mutate(data)
                with self.assertRaisesRegex(ValueError, message):
                    migrate(data, self.exporter)

    def test_rejects_changed_lstm_semantics(self):
        for key, value in (
            ("time_major", True),
            ("activation", "relu"),
            ("recurrent_activation", "hard_sigmoid"),
        ):
            with self.subTest(option=key):
                data = lstm_initial_state_model()
                data["architecture"]["config"]["layers"][3]["config"][key] = value
                with self.assertRaises(ValueError):
                    migrate(data, self.exporter)

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
