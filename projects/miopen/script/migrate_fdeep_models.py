#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import base64
import copy
import importlib.util
import json
from pathlib import Path

import keras
import numpy as np


def decode(values):
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError("Expected base64-encoded float32 weights")
    return np.frombuffer(base64.b64decode("".join(values), validate=True), dtype="<f4")


def rebuild(data):
    architecture = data["architecture"]
    if architecture["class_name"] != "Functional":
        raise ValueError("Only bundled Functional models are supported")
    config = architecture["config"]
    tensors = {}
    layers = {}
    weight_keys = {
        "Dense": lambda layer: ["weights"] + (["bias"] if layer.use_bias else []),
        "Embedding": lambda layer: ["weights"],
        "LSTM": lambda layer: ["weights", "recurrent_weights"]
        + (["bias"] if layer.use_bias else []),
        "LayerNormalization": lambda layer: (["gamma"] if layer.scale else [])
        + (["beta"] if layer.center else []),
        "ReLU": lambda layer: [],
        "Add": lambda layer: [],
    }

    def resolve(connection):
        if not isinstance(connection, list) or len(connection) not in (3, 4):
            raise ValueError(f"Invalid legacy connection: {connection}")
        if len(connection) == 4 and connection[3] != {}:
            raise ValueError(f"Unsupported connection arguments: {connection}")
        return tensors[tuple(connection[:3])]

    for entry in config["layers"]:
        kind = entry["class_name"]
        options = copy.deepcopy(entry["config"])
        name = options["name"]
        if kind == "InputLayer":
            options["batch_shape"] = options.pop("batch_input_shape")
            layer = keras.layers.InputLayer.from_config(options)
            tensors[(name, 0, 0)] = layer.output
            continue
        if kind not in weight_keys:
            raise ValueError(f"Unsupported layer type: {kind}")
        if kind == "Embedding":
            options.pop("batch_input_shape", None)
            options.pop("input_length", None)
        if kind == "LSTM":
            if options.pop("time_major", False):
                raise ValueError("time_major LSTM is not supported")
            options.pop("implementation", None)
            if (
                options["activation"] != "tanh"
                or options["recurrent_activation"] != "sigmoid"
            ):
                raise ValueError(
                    "Only unchanged tanh/sigmoid LSTM semantics are supported"
                )
        layer = getattr(keras.layers, kind).from_config(options)
        layers[name] = layer
        for index, node in enumerate(entry["inbound_nodes"]):
            inputs = [resolve(connection) for connection in node]
            if kind == "LSTM" and len(inputs) == 3:
                outputs = layer(inputs[0], initial_state=inputs[1:])
            elif kind == "Add":
                outputs = layer(inputs)
            elif len(inputs) == 1:
                outputs = layer(inputs[0])
            else:
                raise ValueError(f"Unexpected inputs for {name}: {len(inputs)}")
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            for tensor_index, output in enumerate(outputs):
                tensors[(name, index, tensor_index)] = output
        keys = weight_keys[kind](layer)
        params = data["trainable_params"].get(name, {})
        if set(keys) != set(params) or len(keys) != len(layer.weights):
            raise ValueError(f"Unexpected weights for {name}")
        layer.set_weights(
            [
                decode(params[key]).reshape(tuple(weight.shape))
                for key, weight in zip(keys, layer.weights)
            ]
        )
    if set(data["trainable_params"]) != {
        name for name, layer in layers.items() if layer.weights
    }:
        raise ValueError("Unconsumed trainable parameters")
    return keras.Model(
        inputs=[resolve(connection) for connection in config["input_layers"]],
        outputs=[resolve(connection) for connection in config["output_layers"]],
        name=config["name"],
    )


def migrate(data, exporter):
    keras.backend.clear_session()
    keras.utils.set_random_seed(889)
    model = rebuild(data)
    params = exporter.get_all_weights(model, "")
    if params != data["trainable_params"]:
        raise ValueError("Re-export changed packed weights")
    input_shapes = [list(tensor.shape[1:]) for tensor in model.inputs]
    output_shapes = [list(tensor.shape[1:]) for tensor in model.outputs]
    if input_shapes != data["input_shapes"] or output_shapes != data["output_shapes"]:
        raise ValueError("Model input/output shapes changed")
    for test in data.get("tests", []):
        inputs = [
            decode(tensor["values"]).reshape((1, *tensor["shape"]))
            for tensor in test["inputs"]
        ]
        outputs = model(inputs, training=False)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        if len(outputs) != len(test["outputs"]):
            raise ValueError("Model output count changed")
        for actual, expected in zip(outputs, test["outputs"]):
            np.testing.assert_allclose(
                np.asarray(actual).reshape(-1),
                decode(expected["values"]),
                atol=1e-4,
                rtol=1e-4,
            )
    result = copy.deepcopy(data)
    result["architecture"] = json.loads(model.to_json())
    return result


def load_exporter(source):
    spec = importlib.util.spec_from_file_location(
        "fdeep_exporter", source / "keras_export" / "convert_model.py"
    )
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    return exporter


def main():
    parser = argparse.ArgumentParser(
        description="Migrate bundled model graphs to Keras 3 without retraining."
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fdeep-source", required=True, type=Path)
    args = parser.parse_args()
    if args.source_dir.resolve() == args.output_dir.resolve():
        parser.error(
            "Use a separate output directory to preserve the original artifacts"
        )
    exporter = load_exporter(args.fdeep_source)
    paths = sorted(
        path for path in args.source_dir.glob("*.model") if "metadata" not in path.name
    )
    if not paths:
        parser.error("No bundled models found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        data = json.loads(path.read_text())
        result = migrate(data, exporter)
        destination = args.output_dir / path.name
        destination.write_text(
            json.dumps(result, allow_nan=False, separators=(",", ":"))
        )
        print(
            f"Migrated {path.name}: weights and {len(data.get('tests', []))} reference tests preserved",
            flush=True,
        )


if __name__ == "__main__":
    main()
