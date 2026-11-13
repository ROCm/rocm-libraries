# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`mlp_data`: the numpy fit, its artifact, the offline scorer and the train/promote paths."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("flatbuffers")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("lightgbm")

from uhd_gen.__main__ import main  # noqa: E402
from uhd_gen.artifact import (  # noqa: E402
    MAX_MLP_ARTIFACT_BYTES,
    MLP_MODEL_IDENTIFIER,
    load_mlp_artifact,
    mlp_features_hash,
    verify_mlp_artifact,
)
from uhd_gen.evaluate import _mlp_scorer, load_model  # noqa: E402
from uhd_gen.mlp import (  # noqa: E402
    LEARNING_RATE,
    PATIENCE,
    MlpEnsemble,
    _fit_member,
    _forward,
    _initial_layers,
    _validation_split,
    fit_mlp,
)
from uhd_gen.mlp_to_flatbuffer import build_mlp_model, write_mlp_model  # noqa: E402

NAN = float("nan")
FEATURES_HASH = "sha256:0123456789abcdef"
CPP_TEST = (
    Path(__file__).resolve().parents[3]
    / "plugin_sdk/tests/heuristics/uhd/TestMlpDataAdapter.cpp"
)


# --------------------------------------------------------------------------------------
# The shared fixture: TestMlpDataAdapter.cpp embeds these bytes and these scores
# --------------------------------------------------------------------------------------


def _weights(rows: int, columns: int, salt: int) -> np.ndarray:
    """Exact binary fractions in [-1, 1], a function of position only."""
    return np.array(
        [
            [((o * 7 + i * 3 + salt) % 9 - 4) / 4 for i in range(columns)]
            for o in range(rows)
        ],
        dtype=np.float32,
    )


def _biases(count: int, salt: int) -> np.ndarray:
    return np.array(
        [((o * 5 + salt) % 7 - 3) / 16 for o in range(count)], dtype=np.float32
    )


def fixture_ensemble() -> MlpEnsemble:
    """4 features; inputs over slots 0, 1 and 3, a missing flag for slot 1 (slot 2 is
    unread); network input width 4; two networks, 4 -> 5 -> 1 and 4 -> 3 -> 2 -> 1."""
    return MlpEnsemble(
        num_features=4,
        input_slots=np.array([0, 1, 3], dtype=np.int64),
        input_mean=np.array([2.0, 1.0, 1.0]),
        input_scale=np.array([6.0, 5.0, 8.0]),
        missing_slots=np.array([1], dtype=np.int64),
        networks=[
            [
                (_weights(5, 4, 1), _biases(5, 2)),
                (_weights(1, 5, 3), _biases(1, 4)),
            ],
            [
                (_weights(3, 4, 5), _biases(3, 6)),
                (_weights(2, 3, 7), _biases(2, 1)),
                (_weights(1, 2, 2), _biases(1, 3)),
            ],
        ],
        target_mean=3.0,
        target_scale=4.0,
    )


FIXTURE_ROWS = np.array(
    [
        [10, 3, 0, 7],  # ordinary
        [0.5, 1.5, 5, 0.25],  # small values
        [100, NAN, 1, 20],  # missing in an input that has a flag: z reads 0, flag 1
        [7, 2, NAN, NAN],  # missing in an input with no flag; slot 2 is not read
        [-5, -2, 0, -30],  # negatives: signed log
        [1e9, 1e7, 3, 1e12],  # large magnitudes
        [0, 0, 0, 0],  # zeros
        [-1e9, NAN, 2, 1e-3],  # a large negative beside a missing value
    ]
)


def fixture_bytes() -> bytes:
    return build_mlp_model(
        fixture_ensemble(),
        "tflops",
        FEATURES_HASH,
        100,
        ["gfx942"],
        "1.0.0",
        "2026-01-01T00:00:00+00:00",
    )


def direct_score(ensemble: MlpEnsemble, row) -> float:
    """The documented semantics (mlp_model.fbs), one row at a time, in float64."""
    values = []
    for slot, mean, scale in zip(
        ensemble.input_slots, ensemble.input_mean, ensemble.input_scale
    ):
        x = float(row[slot])
        t = 0.0 if math.isnan(x) else math.copysign(math.log1p(abs(x)), x)
        z = (t - mean) / scale
        if not math.isfinite(z):
            return NAN
        values.append(z)
    values += [
        1.0 if math.isnan(float(row[slot])) else 0.0 for slot in ensemble.missing_slots
    ]

    outputs = []
    for layers in ensemble.networks:
        a = np.array(values)
        for index, (weights, biases) in enumerate(layers):
            a = weights.astype(np.float64) @ a + biases.astype(np.float64)
            if index + 1 < len(layers):
                a = a / (1.0 + np.exp(-a))
        outputs.append(float(a[0]))
    return ensemble.target_mean + ensemble.target_scale * (sum(outputs) / len(outputs))


def test_the_predictor_is_the_documented_formula():
    ensemble = fixture_ensemble()
    expected = [direct_score(ensemble, row) for row in FIXTURE_ROWS]
    # Float32 members against the float64 reading of the same weights.
    np.testing.assert_allclose(
        ensemble.predict(FIXTURE_ROWS), expected, rtol=0, atol=1e-5
    )
    # The rows differ, so none of the rules is dead.
    assert len(set(np.round(expected, 4))) == len(expected)


def test_a_row_the_runtime_would_decline_scores_nan():
    ensemble = fixture_ensemble()
    rows = np.array(
        [[1.0, 2.0, 3.0, 4.0], [math.inf, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, -math.inf]]
    )
    scores = ensemble.predict(rows)
    assert np.isfinite(scores[0]) and np.isnan(scores[1:]).all()
    # An infinity in a slot no input reads does not matter.
    assert np.isfinite(ensemble.predict(np.array([[1.0, 2.0, math.inf, 4.0]]))).all()
    with pytest.raises(ValueError, match="4 columns"):
        ensemble.predict(np.zeros((2, 3)))


def test_the_cpp_test_embeds_this_fixture_and_these_scores():
    """TestMlpDataAdapter.cpp must test the bytes the writer produces, scored as here."""
    if not CPP_TEST.exists():
        pytest.skip("the C++ tests are not in this checkout")
    text = CPP_TEST.read_text(encoding="utf-8")
    # Whitespace-tolerant: clang-format may break a declaration before its `=`.
    hex_literal = re.search(r"FIXTURE_HEX\s*=\s*((?:\s*\"[0-9a-f]+\"\s*)+);", text)
    digest = re.search(r'FIXTURE_SHA256\s*=\s*"([0-9a-f]{64})"', text)
    scores = re.search(r"FIXTURE_SCORES\s*=\s*\{(.*?)\};", text, re.S)
    assert hex_literal and digest and scores
    embedded = bytes.fromhex(
        "".join(re.findall(r'"([0-9a-f]+)"', hex_literal.group(1)))
    )
    assert embedded == fixture_bytes()
    assert digest.group(1) == hashlib.sha256(embedded).hexdigest()
    recorded = [float(v) for v in re.findall(r"[-0-9.e+]+", scores.group(1))]
    np.testing.assert_allclose(
        recorded, fixture_ensemble().predict(FIXTURE_ROWS), rtol=1e-6, atol=0
    )


# --------------------------------------------------------------------------------------
# Writer and verifier
# --------------------------------------------------------------------------------------


def test_the_artifact_round_trips_and_scores_identically(tmp_path):
    ensemble = fixture_ensemble()
    path = tmp_path / "model.bin"
    digest = write_mlp_model(
        ensemble,
        "tflops",
        FEATURES_HASH,
        path,
        100,
        ["gfx942"],
        "1.0.0",
        "2026-01-01T00:00:00+00:00",
    )
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes()[4:8] == MLP_MODEL_IDENTIFIER
    loaded, metric = load_mlp_artifact(path, digest, feature_count=4)
    assert metric == "tflops"
    assert mlp_features_hash(path.read_bytes()) == FEATURES_HASH
    np.testing.assert_array_equal(
        loaded.predict(FIXTURE_ROWS), ensemble.predict(FIXTURE_ROWS)
    )
    assert verify_mlp_artifact(path, digest) == path.read_bytes()
    # The writer is a pure function of the model and its stamps.
    assert fixture_bytes() == fixture_bytes()


def _mutated(mutate) -> bytes:
    """The fixture, rewritten through the object API after `mutate(model)`."""
    import flatbuffers
    from hipdnn_flatbuffers_sdk.data_objects.MlpModel import MlpModelT

    model = MlpModelT.InitFromPackedBuf(bytearray(fixture_bytes()), 0)
    mutate(model)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=MLP_MODEL_IDENTIFIER)
    return bytes(builder.Output())


def _layer(model, network=0, layer=0):
    return model.networks[network].layers[layer]


def _with_value(array, index, value):
    copy_ = np.array(array, dtype=np.float32)
    copy_[index] = value
    return copy_


def _wide_input(model):
    """513 inputs: one more than the network input may hold."""
    model.inputs = [model.inputs[0] for _ in range(513)]


def _wide_layer(model):
    first, second = _layer(model, 0, 0), _layer(model, 0, 1)
    first.outputs, second.inputs = 513, 513
    first.weights = np.zeros(4 * 513, dtype=np.float32)
    first.biases = np.zeros(513, dtype=np.float32)
    second.weights = np.zeros(513, dtype=np.float32)


def _deep(model):
    """9 layers: 4 -> 1 -> ... -> 1."""
    layers = []
    for index in range(9):
        layer = type(_layer(model))()
        layer.inputs, layer.outputs = (4 if index == 0 else 1), 1
        layer.weights = np.zeros(layer.inputs, dtype=np.float32)
        layer.biases = np.zeros(1, dtype=np.float32)
        layers.append(layer)
    model.networks[0].layers = layers


def _many_networks(model):
    model.networks = [model.networks[0] for _ in range(65)]


REFUSALS = [
    ("no features", lambda m: setattr(m, "numFeatures", 0), "num_features must be"),
    (
        "too many features",
        lambda m: setattr(m, "numFeatures", 4097),
        "num_features must be",
    ),
    ("no metric", lambda m: setattr(m, "metric", ""), "names no metric"),
    ("metric absent", lambda m: setattr(m, "metric", None), "names no metric"),
    ("input too wide", _wide_input, "between 1 and 512 values"),
    (
        "no network input",
        lambda m: (setattr(m, "inputs", []), setattr(m, "missingFlags", [])),
        "between 1 and 512 values",
    ),
    (
        "input slot out of range",
        lambda m: setattr(m.inputs[1], "feature", 4),
        "input 1 reads a slot outside the 4 features",
    ),
    (
        "negative input slot",
        lambda m: setattr(m.inputs[0], "feature", -1),
        "input 0 reads a slot outside",
    ),
    (
        "nan mean",
        lambda m: setattr(m.inputs[2], "mean", NAN),
        "input 2 needs a finite mean and a positive scale",
    ),
    (
        "zero scale",
        lambda m: setattr(m.inputs[0], "scale", 0.0),
        "positive scale",
    ),
    (
        "negative scale",
        lambda m: setattr(m.inputs[0], "scale", -1.0),
        "positive scale",
    ),
    (
        "infinite scale",
        lambda m: setattr(m.inputs[0], "scale", math.inf),
        "positive scale",
    ),
    (
        "missing flag out of range",
        lambda m: setattr(m, "missingFlags", [9]),
        "missing flag 0 reads a slot outside",
    ),
    (
        "zero target scale",
        lambda m: setattr(m, "targetScale", 0.0),
        "target_scale positive",
    ),
    (
        "infinite target mean",
        lambda m: setattr(m, "targetMean", math.inf),
        "target_mean must be finite",
    ),
    (
        "no networks",
        lambda m: setattr(m, "networks", []),
        "networks must hold between 1 and 64",
    ),
    ("too many networks", _many_networks, "networks must hold between 1 and 64"),
    (
        "network without layers",
        lambda m: setattr(m.networks[1], "layers", []),
        "network 1 must hold between 1 and 8 layers",
    ),
    ("too many layers", _deep, "network 0 must hold between 1 and 8 layers"),
    (
        "layer of no outputs",
        lambda m: setattr(_layer(m, 0, 0), "outputs", 0),
        "network 0 layer 0 must declare positive inputs and outputs",
    ),
    (
        "first layer ignores the input width",
        lambda m: setattr(_layer(m, 0, 0), "inputs", 3),
        "network 0 layer 0 takes 3 inputs, but 4 are available",
    ),
    (
        "layer does not follow its predecessor",
        lambda m: setattr(_layer(m, 1, 1), "inputs", 2),
        "network 1 layer 1 takes 2 inputs, but 3 are available",
    ),
    ("layer too wide", _wide_layer, "width outside [1, 512]"),
    (
        "last layer of two outputs",
        lambda m: setattr(_layer(m, 1, 2), "outputs", 2),
        "does not have exactly one output",
    ),
    (
        "weights absent",
        lambda m: setattr(_layer(m, 0, 0), "weights", None),
        "network 0 layer 0 weights must hold at most 20 values",
    ),
    (
        "too many weights",
        lambda m: setattr(_layer(m, 0, 1), "weights", np.zeros(6, dtype=np.float32)),
        "network 0 layer 1 weights must hold at most 5 values",
    ),
    (
        "too few weights",
        lambda m: setattr(_layer(m, 0, 0), "weights", np.zeros(19, dtype=np.float32)),
        "inputs*outputs weights and outputs biases",
    ),
    (
        "too few biases",
        lambda m: setattr(_layer(m, 1, 0), "biases", np.zeros(2, dtype=np.float32)),
        "inputs*outputs weights and outputs biases",
    ),
    (
        "too many biases",
        lambda m: setattr(_layer(m, 1, 0), "biases", np.zeros(4, dtype=np.float32)),
        "biases must hold at most 3 values",
    ),
    (
        "nan weight",
        lambda m: setattr(
            _layer(m, 0, 0), "weights", _with_value(_layer(m, 0, 0).weights, 3, NAN)
        ),
        "weights holds a value that is not finite",
    ),
    (
        "infinite bias",
        lambda m: setattr(
            _layer(m, 1, 1), "biases", _with_value(_layer(m, 1, 1).biases, 0, math.inf)
        ),
        "biases holds a value that is not finite",
    ),
]


@pytest.mark.parametrize(
    "mutate, reason", [pytest.param(m, r, id=n) for n, m, r in REFUSALS]
)
def test_an_inconsistent_model_is_refused_for_the_reason_the_runtime_gives(
    tmp_path, mutate, reason
):
    path = tmp_path / "model.bin"
    path.write_bytes(_mutated(mutate))
    with pytest.raises(ValueError, match=re.escape(reason)):
        load_mlp_artifact(path, None)


def test_the_limits_themselves_are_accepted(tmp_path):
    """The runtime's bounds are inclusive: 512 inputs, 8 layers, 64 networks."""

    def at_the_limits(model):
        model.numFeatures = 4096
        model.inputs = [model.inputs[0] for _ in range(512)]
        model.missingFlags = []
        model.networks = [model.networks[0] for _ in range(64)]
        first, second = _layer(model, 0, 0), _layer(model, 0, 1)
        first.inputs = 512
        first.weights = np.zeros(5 * 512, dtype=np.float32)
        model.inputs[0].feature = 4095

    path = tmp_path / "model.bin"
    path.write_bytes(_mutated(at_the_limits))
    ensemble, _ = load_mlp_artifact(path, None, feature_count=4096)
    assert len(ensemble.networks) == 64 and ensemble.input_width == 512


def test_a_buffer_the_runtime_would_not_open_is_refused_in_its_order(tmp_path):
    path = tmp_path / "model.bin"
    good = fixture_bytes()

    path.write_bytes(good)
    with pytest.raises(ValueError, match="model hash mismatch"):
        load_mlp_artifact(path, "0" * 64)
    with pytest.raises(ValueError, match="num_features 4 differs"):
        load_mlp_artifact(path, None, feature_count=5)

    for other in (b"TABL", b"HLIN", b"HGBM"):
        path.write_bytes(good[:4] + other + good[8:])
        with pytest.raises(ValueError, match="file identifier"):
            load_mlp_artifact(path, None)

    path.write_bytes(good[: len(good) // 2])
    with pytest.raises(ValueError, match="verifier refuses"):
        load_mlp_artifact(path, None)

    path.write_bytes(b"")
    with pytest.raises(ValueError, match="artifact size"):
        load_mlp_artifact(path, None)

    path.write_bytes(good[:4] + MLP_MODEL_IDENTIFIER + bytes(MAX_MLP_ARTIFACT_BYTES))
    with pytest.raises(ValueError, match="artifact size"):
        load_mlp_artifact(path, None)

    # A root offset pointing past the end of the buffer.
    corrupt = bytearray(good)
    corrupt[0:4] = (len(good) + 100).to_bytes(4, "little")
    path.write_bytes(bytes(corrupt))
    with pytest.raises(ValueError, match="verifier refuses"):
        load_mlp_artifact(path, None)


# --------------------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------------------


def _curve(rows=600, seed=1):
    """Two log-uniform inputs, one with 10% missing, and a smooth target in them."""
    rng = np.random.default_rng(seed)
    a = np.exp(rng.uniform(0, 8, rows))
    b = np.exp(rng.uniform(0, 6, rows))
    c = rng.choice([1.0, 2.0, 4.0, 8.0], rows)
    c[rng.random(rows) < 0.1] = NAN
    matrix = np.column_stack([a, b, c, np.full(rows, 3.0)])
    y = (
        0.5 * np.log1p(a)
        + 0.3 * np.log1p(b)
        + 0.2 * np.log1p(np.nan_to_num(c))
        + 0.01 * rng.standard_normal(rows)
    )
    return matrix, y


def _same_networks(left: MlpEnsemble, right: MlpEnsemble) -> bool:
    return all(
        np.array_equal(w1, w2) and np.array_equal(b1, b2)
        for n1, n2 in zip(left.networks, right.networks)
        for (w1, b1), (w2, b2) in zip(n1, n2)
    )


def test_the_fit_learns_a_smooth_target_and_is_a_function_of_its_seed():
    matrix, y = _curve()
    ensemble, fit = fit_mlp(matrix, y, epochs=40, members=2)
    error = np.abs(ensemble.predict(matrix) - y).mean()
    assert error < 0.1 * np.abs(y - y.mean()).mean()
    assert fit["hidden"] == [64, 64] and fit["members"] == 2 and fit["rows"] == 600
    assert [m["seed"] for m in fit["member_fits"]] == [0, 1]

    again, _ = fit_mlp(matrix, y, epochs=40, members=2)
    assert _same_networks(ensemble, again)
    other, _ = fit_mlp(matrix, y, epochs=40, members=2, seed=5)
    assert not _same_networks(ensemble, other)
    # The members are different networks, not one repeated.
    assert not np.array_equal(ensemble.networks[0][0][0], ensemble.networks[1][0][0])


def test_inputs_are_the_varying_columns_and_flags_the_partly_missing_ones():
    rng = np.random.default_rng(0)
    rows = 60
    matrix = np.column_stack(
        [
            np.exp(rng.uniform(0, 5, rows)),  # 0: varies
            np.full(rows, 7.0),  # 1: constant
            np.where(np.arange(rows) % 4 == 0, NAN, rng.uniform(1, 9, rows)),  # 2
            np.full(rows, NAN),  # 3: never present
            np.where(np.arange(rows) % 3 == 0, NAN, 5.0),  # 4: constant where present
        ]
    )
    ensemble, fit = fit_mlp(
        matrix, rng.normal(size=rows), epochs=2, members=1, hidden=(4,)
    )
    assert ensemble.input_slots.tolist() == [0, 2, 4]
    assert ensemble.missing_slots.tolist() == [2, 4]
    assert fit["input_slots"] == [0, 2, 4] and fit["missing_flag_slots"] == [2, 4]
    assert ensemble.input_width == 5
    assert [w.shape for w, _ in ensemble.networks[0]] == [(4, 5), (1, 4)]


def test_the_target_is_standardized_and_a_constant_target_still_has_a_scale():
    matrix, y = _curve(rows=80)
    ensemble, _ = fit_mlp(matrix, y, epochs=1, members=1, hidden=(3,))
    assert ensemble.target_mean == pytest.approx(y.mean())
    assert ensemble.target_scale == pytest.approx(y.std() + 1e-9)
    flat, _ = fit_mlp(matrix, np.full(80, 2.0), epochs=1, members=1, hidden=(3,))
    assert flat.target_scale == pytest.approx(1e-9)


def test_a_member_stops_when_validation_stops_improving_and_keeps_its_best():
    rng = np.random.default_rng(3)
    matrix = rng.uniform(0, 10, size=(150, 3))
    # Noise unrelated to the features: the validation loss bottoms out early.
    _, fit = fit_mlp(matrix, rng.normal(size=150), epochs=300, members=2, hidden=(32,))
    for member in fit["member_fits"]:
        assert member["epochs_run"] < 300
        assert member["epochs_run"] == member["best_epoch"] + PATIENCE + 1
        assert member["validation_rows"] == 22 and member["train_rows"] == 128


def test_a_corpus_beyond_max_rows_is_subsampled_with_the_seed():
    matrix, y = _curve(rows=200)
    ensemble, fit = fit_mlp(matrix, y, epochs=2, members=1, hidden=(4,), max_rows=50)
    assert (fit["rows"], fit["rows_available"]) == (50, 200)
    again, _ = fit_mlp(matrix, y, epochs=2, members=1, hidden=(4,), max_rows=50)
    assert _same_networks(ensemble, again)
    whole, fit = fit_mlp(matrix, y, epochs=2, members=1, hidden=(4,), max_rows=500)
    assert fit["rows"] == 200


def test_a_member_keeps_the_weights_of_its_best_epoch_not_its_last():
    rng = np.random.default_rng(3)
    inputs = rng.uniform(0, 10, size=(150, 3)).astype(np.float32)
    target = rng.normal(size=150).astype(np.float32)
    layers, fit = _fit_member(inputs, target, (32,), epochs=300, seed=3)
    # The run went on PATIENCE epochs past its best, so the last weights are not the best.
    assert fit["epochs_run"] == fit["best_epoch"] + PATIENCE + 1
    valid, _ = _validation_split(len(inputs), np.random.default_rng(3), None)
    held_out = float(np.mean(np.abs(_forward(layers, inputs[valid]) - target[valid])))
    assert held_out == pytest.approx(fit["validation_l1"], rel=1e-5)


def test_a_corpus_beyond_max_rows_keeps_a_seeded_sample_in_row_order():
    matrix, y = _curve(rows=200)
    seed = 7
    kept = np.sort(np.random.default_rng(seed).choice(200, size=50, replace=False))
    options = {"epochs": 2, "members": 1, "hidden": (4,), "seed": seed}
    sampled, _ = fit_mlp(matrix, y, max_rows=50, **options)
    reference, _ = fit_mlp(matrix[kept], y[kept], **options)
    assert _same_networks(sampled, reference)
    assert sampled.target_mean == pytest.approx(y[kept].mean())


def _l1_gradient(layers, inputs, target):
    """d(mean |network - target|)/d(each parameter) by central differences in float64,
    ordered weights then biases per layer."""
    wide = [[p.astype(np.float64) for p in layer] for layer in layers]

    def loss():
        a = inputs.astype(np.float64)
        for index, (weights, biases) in enumerate(wide):
            a = a @ weights.T + biases
            if index + 1 < len(wide):
                a = a / (1.0 + np.exp(-a))
        return float(np.mean(np.abs(a[:, 0] - target)))

    gradient = []
    for layer in wide:
        for array in layer:
            for position in np.ndindex(array.shape):
                original = array[position]
                array[position] = original + 1e-6
                above = loss()
                array[position] = original - 1e-6
                below = loss()
                array[position] = original
                gradient.append((above - below) / 2e-6)
    return np.array(gradient)


def test_a_first_step_descends_the_l1_loss_through_every_silu_layer():
    """Adam's first update is a step of `-LEARNING_RATE * sign(gradient)`, so the direction
    each weight and bias moves is the sign of the backward pass's gradient. One minibatch,
    one epoch: the update from the known initial weights must descend the true mean absolute
    error, whose gradient is taken numerically."""
    rng = np.random.default_rng(11)
    inputs = rng.normal(size=(40, 3)).astype(np.float32)
    target = rng.normal(size=40).astype(np.float32)
    hidden, seed = (4, 3), 5

    split = np.random.default_rng(seed)
    _, train = _validation_split(len(inputs), split, None)
    initial = _initial_layers([3, *hidden, 1], split)
    trained, _ = _fit_member(inputs, target, hidden, epochs=1, seed=seed)

    gradient = _l1_gradient(initial, inputs[train], target[train])
    moved = np.concatenate(
        [
            (after - before).ravel()
            for (w1, b1), (w0, b0) in zip(trained, initial)
            for after, before in ((w1, w0), (b1, b0))
        ]
    )
    decisive = np.abs(gradient) > 1e-4
    assert decisive.sum() > 0.8 * len(gradient)
    assert np.all(np.sign(moved[decisive]) == -np.sign(gradient[decisive]))
    assert np.all(np.abs(moved[decisive]) < 2 * LEARNING_RATE)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"hidden": (8,) * 8}, "8 hidden layer"),
        ({"hidden": (513,)}, "hidden widths"),
        ({"hidden": (0,)}, "hidden widths"),
        ({"members": 0}, "members must be"),
        ({"members": 65}, "members must be"),
        ({"epochs": 0}, "epochs must be"),
        ({"max_rows": -1}, "max_rows"),
    ],
)
def test_a_network_the_runtime_would_not_load_is_refused_before_fitting(
    options, message
):
    matrix, y = _curve(rows=40)
    with pytest.raises(ValueError, match=message):
        fit_mlp(matrix, y, **options)


def test_data_the_runtime_could_not_score_is_refused():
    matrix, y = _curve(rows=40)
    inf = matrix.copy()
    inf[3, 0] = math.inf
    with pytest.raises(ValueError, match="finite or NaN"):
        fit_mlp(inf, y, epochs=1)
    with pytest.raises(ValueError, match="target must be finite"):
        fit_mlp(matrix, np.where(np.arange(40) == 2, NAN, y), epochs=1)
    with pytest.raises(ValueError, match="at least 2 rows"):
        fit_mlp(matrix[:1], y[:1], epochs=1)
    wide = np.random.default_rng(0).uniform(1, 9, size=(20, 513))
    with pytest.raises(ValueError, match="513 values"):
        fit_mlp(wide, np.zeros(20), epochs=1)


# --------------------------------------------------------------------------------------
# The offline scorer
# --------------------------------------------------------------------------------------


def test_the_offline_scorer_is_the_direct_evaluation_in_the_metrics_units(tmp_path):
    ensemble = fixture_ensemble()
    path = tmp_path / "model.bin"
    digest = write_mlp_model(ensemble, "tflops", FEATURES_HASH, path)
    features = ["f0", "f1", "f2", "f3"]
    # The shared feature path refuses non-finite values, so only the finite rows go through it.
    rows = FIXTURE_ROWS[~np.isnan(FIXTURE_ROWS).any(axis=1)]
    frame = pd.DataFrame(rows, columns=features)
    log_scores = np.array([direct_score(ensemble, row) for row in rows])

    scorer = _mlp_scorer(
        path, features, model_hash=digest, metric="tflops", score_transform="log"
    )
    np.testing.assert_allclose(scorer(frame), np.exp(log_scores), rtol=1e-5)
    raw = _mlp_scorer(path, features, score_transform="identity")
    np.testing.assert_allclose(raw(frame), log_scores, rtol=0, atol=1e-5)

    with pytest.raises(ValueError, match="model hash mismatch"):
        _mlp_scorer(path, features, model_hash="0" * 64)
    with pytest.raises(
        ValueError, match="predicts 'tflops' but the descriptor scores 'time'"
    ):
        _mlp_scorer(path, features, metric="time")
    with pytest.raises(ValueError, match="does not match the shipped model"):
        _mlp_scorer(path, features, expected_hash="sha256:ffffffffffffffff")


# --------------------------------------------------------------------------------------
# train, evaluate and promote
# --------------------------------------------------------------------------------------

#: Mixed into the L1 corpus a UED-backed engine is measured on (test_immediate.measurement).
TRAIN_OPTIONS = ["--mlp-epochs", "150", "--mlp-members", "2", "--mlp-hidden", "16,16"]


def _l1_corpus(path, count=60, metric="tflops"):
    """Rows of an engine whose time grows smoothly with the graph's flops and nodes."""
    from uhd_gen.tests.test_immediate import measurement

    rows = []
    for index in range(count):
        row = measurement(graph=f"graph-{index}", metric=metric)
        flops = float(2e9 * (index + 1))
        nodes = float(1 + index % 5)
        row["features"].update({"graph.flops": flops, "graph.nodes": nodes})
        row["avgTimeMs"] = (index + 1) ** 0.8 * (1 + 0.1 * nodes)
        rows.append(row)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _train(tmp_path, evaluator, *extra, metric="tflops", output="model"):
    corpus = tmp_path / "corpus.json"
    if not corpus.exists():
        _l1_corpus(corpus, metric=metric)
    code = main(
        [
            "train",
            "--role",
            "predict_engine",
            "--adapter",
            "mlp_data",
            "--input",
            str(corpus),
            "--features",
            "graph.flops",
            "graph.nodes",
            "--metric",
            metric,
            "--feature-evaluator",
            evaluator,
            "--output-dir",
            str(tmp_path / output),
            "--training-arches",
            "gfx942",
            *TRAIN_OPTIONS,
            *extra,
        ]
    )
    return code, tmp_path / output


@pytest.mark.parametrize("metric, objective", [("tflops", "max"), ("time", "min")])
def test_train_emits_a_valid_mlp_data_uhd(tmp_path, evaluator, metric, objective):
    from uhd_gen.immediate import read_corpus, validate_model
    from uhd_gen.promote import _validate_descriptor

    code, output = _train(tmp_path, evaluator, metric=metric)
    assert code == 0
    descriptor_path = output / "heuristic.uhd.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert descriptor["adapter"] == "mlp_data"
    assert descriptor["objective"] == objective
    assert descriptor["score"] == {
        "metric": metric,
        "calibrated": True,
        "transform": "log",
    }
    assert "tree_data" not in descriptor
    body = descriptor["mlp_data"]
    assert body["artifact"] == "model.bin"
    artifact = output / "model.bin"
    assert body["hash"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    # The validators every consumer runs accept it.
    _validate_descriptor(descriptor, descriptor_path)
    assert validate_model(descriptor).name == metric

    # The runtime's checks, in its order, against the descriptor's own signature.
    ensemble, recorded = load_mlp_artifact(
        artifact, body["hash"], feature_count=len(descriptor["features_signature"])
    )
    assert recorded == metric
    assert mlp_features_hash(artifact.read_bytes()) == descriptor["features_hash"]
    assert len(ensemble.networks) == 2
    assert [w.shape for w, _ in ensemble.networks[0]] == [(16, 2), (16, 16), (1, 16)]

    manifest = json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter"] == "mlp_data"
    assert manifest["score_transform"] == "log"
    assert manifest["mlp_fit"]["members"] == 2 and manifest["mlp_fit"]["hidden"] == [
        16,
        16,
    ]
    assert manifest["role"] == "predict_engine"
    for key in ("num_trees", "feature_importance", "training_options", "early_exit"):
        assert key not in manifest
    assert not (output / "model.lgbm").exists()

    # Evaluated in the metric's units, as the runtime's estimate is read.
    bundle = load_model(output, feature_evaluator=evaluator)
    frame = read_corpus(tmp_path / "corpus.json")
    truth = frame["tflops"] if metric == "tflops" else frame["avgTimeMs"]
    error = np.abs(np.log(bundle.scorer(frame)) - np.log(truth.to_numpy(dtype=float)))
    assert np.median(error) < 0.15


def test_train_is_deterministic_in_its_seed(tmp_path, evaluator, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1767225600")
    code, first = _train(tmp_path, evaluator, "--mlp-epochs", "5", output="a")
    assert code == 0
    code, second = _train(tmp_path, evaluator, "--mlp-epochs", "5", output="b")
    assert code == 0
    assert (first / "model.bin").read_bytes() == (second / "model.bin").read_bytes()
    code, third = _train(
        tmp_path, evaluator, "--mlp-epochs", "5", "--mlp-seed", "9", output="c"
    )
    assert code == 0
    assert (first / "model.bin").read_bytes() != (third / "model.bin").read_bytes()


def test_evaluate_reports_the_mlp_models_predictions(tmp_path, evaluator):
    code, output = _train(tmp_path, evaluator, "--mlp-epochs", "20")
    assert code == 0
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "evaluate",
                "--input",
                str(tmp_path / "corpus.json"),
                "--model-dir",
                str(output),
                "--eval-fraction",
                "1",
                "--feature-evaluator",
                evaluator,
                "--output",
                str(report),
            ]
        )
        == 0
    )
    scored = json.loads(report.read_text(encoding="utf-8"))["metrics"][
        "prediction_coverage"
    ]
    assert scored["scored_rows"] == scored["rows"] == 60


def test_evaluate_refuses_an_artifact_of_another_metric_than_the_descriptor_scores(
    tmp_path, evaluator
):
    code, output = _train(tmp_path, evaluator, "--mlp-epochs", "5", metric="tflops")
    assert code == 0
    load_model(output, feature_evaluator=evaluator)
    path = output / "heuristic.uhd.json"
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    descriptor["score"]["metric"] = "time"
    descriptor["objective"] = "min"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(
        ValueError, match="predicts 'tflops' but the descriptor scores 'time'"
    ):
        load_model(output, feature_evaluator=evaluator)


def test_the_offline_scorer_refuses_an_artifact_whose_arity_is_not_the_signatures(
    tmp_path,
):
    ensemble = fixture_ensemble()
    path = tmp_path / "model.bin"
    write_mlp_model(ensemble, "tflops", FEATURES_HASH, path)
    features = ["f0", "f1", "f2", "f3"]
    _mlp_scorer(path, features, signature=list(range(ensemble.num_features)))
    with pytest.raises(ValueError, match="num_features 4 differs from the 3-slot"):
        _mlp_scorer(path, features[:3], signature=[0, 1, 2])


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--group-by-feature", "graph.nodes"], "tree_data option"),
        (["--mlp-hidden", "16,x"], "comma-separated integers"),
        (["--mlp-hidden", "16,0"], "widths must be positive"),
        (["--mlp-members", "65"], "members must be"),
        (["--mlp-epochs", "0"], "epochs must be"),
    ],
)
def test_train_refuses_what_an_mlp_cannot_be(
    tmp_path, evaluator, caplog, extra, message
):
    corpus = tmp_path / "corpus.json"
    _l1_corpus(corpus)
    arguments = [
        "train",
        "--role",
        "predict_engine",
        "--adapter",
        "mlp_data",
        "--input",
        str(corpus),
        "--features",
        "graph.flops",
        "graph.nodes",
        "--metric",
        "tflops",
        "--feature-evaluator",
        evaluator,
        "--output-dir",
        str(tmp_path / "model"),
        *TRAIN_OPTIONS,
    ]
    # A later --role replaces the earlier one.
    code = main([*arguments, *extra])
    assert code == 1
    assert message in caplog.text
    assert not (tmp_path / "model").exists()


def test_the_mlp_options_belong_to_the_mlp_adapter(tmp_path, evaluator, caplog):
    corpus = tmp_path / "corpus.json"
    _l1_corpus(corpus)
    code = main(
        [
            "train",
            "--role",
            "predict_engine",
            "--input",
            str(corpus),
            "--features",
            "graph.flops",
            "--feature-evaluator",
            evaluator,
            "--output-dir",
            str(tmp_path / "model"),
            "--mlp-epochs",
            "5",
            "--mlp-seed",
            "1",
        ]
    )
    assert code == 1
    assert "--mlp-epochs, --mlp-seed apply only to --adapter mlp_data" in caplog.text


def _installed_tree(tmp_path):
    """A descriptor tree whose UED is the one `_l1_corpus` is measured under."""
    from uhd_gen.tests.test_immediate import KMD, UED

    root = tmp_path / "descriptors"
    root.mkdir()
    (root / "engine.ued.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "id": UED,
                "name": "provider:engine7",
                "metadata": KMD,
            }
        ),
        encoding="utf-8",
    )
    (root / "metadata.kmd.json").write_text(
        json.dumps({"version": "1.0", "id": KMD}), encoding="utf-8"
    )
    return root


def test_promote_installs_an_mlp_data_uhd_the_runtime_would_load(tmp_path, evaluator):
    from uhd_gen.immediate import read_corpus
    from uhd_gen.promote import _apply, build_plan

    code, output = _train(tmp_path, evaluator)
    assert code == 0
    root = _installed_tree(tmp_path)
    plan = build_plan(
        output, root, role="predict_engine", arch="gfx942", feature_evaluator=evaluator
    )
    _apply(plan)
    installed_dir = plan.destination_descriptor.parent
    installed = json.loads(plan.destination_descriptor.read_text(encoding="utf-8"))
    assert installed["adapter"] == "mlp_data"
    ued = json.loads((root / "engine.ued.json").read_text(encoding="utf-8"))
    assert ued["predict_engine"]["gfx942"] == [installed["id"]]

    # An installed model has no manifest; it scores as it did before it moved.
    frame = read_corpus(tmp_path / "corpus.json")
    before = load_model(output, feature_evaluator=evaluator).scorer(frame)
    after = load_model(installed_dir, feature_evaluator=evaluator)
    assert after.role == "predict_engine"
    np.testing.assert_array_equal(after.scorer(frame), before)


def test_promote_refuses_an_mlp_data_uhd_the_runtime_would_refuse(tmp_path, evaluator):
    from uhd_gen.promote import PromoteError, build_plan

    code, output = _train(tmp_path, evaluator)
    assert code == 0
    root = _installed_tree(tmp_path)
    document = json.loads((output / "heuristic.uhd.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))
    artifact = (output / "model.bin").read_bytes()

    def broken(name, edit):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "heuristic.uhd.json").write_text(
            json.dumps(edit(copy.deepcopy(document))), encoding="utf-8"
        )
        (directory / "model.bin").write_bytes(artifact)
        (directory / "train_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return directory

    def plan(directory):
        return build_plan(
            directory,
            root,
            role="predict_engine",
            arch="gfx942",
            feature_evaluator=evaluator,
        )

    plan(broken("same", lambda d: d))

    # The artifact names the metric it predicts; the descriptor must score the same one.
    def other_metric(d):
        d["objective"], d["score"]["metric"] = "min", "time"
        return d

    with pytest.raises(
        PromoteError, match="predicts 'tflops' but the UHD scores 'time'"
    ):
        plan(broken("metric", other_metric))

    def no_metric(d):
        d["score"].pop("metric")
        d["score"]["calibrated"] = False
        return d

    with pytest.raises(PromoteError, match="mlp_data requires score.metric"):
        plan(broken("bare", no_metric))

    def not_log(d):
        d["score"]["transform"] = "log1p"
        return d

    with pytest.raises(PromoteError, match="score.transform log"):
        plan(broken("log1p", not_log))

    def wide(d):
        d["features_signature"] = [*d["features_signature"], "$graph.nodes"]
        return d

    with pytest.raises(PromoteError, match="features_hash|num_features"):
        plan(broken("wide", wide))


def test_a_slotted_engines_l1_estimate_may_be_mlp_data(tmp_path, evaluator):
    """A slot ranker must be metric-less, but the engine-level L1 estimate names a metric,
    so it can be mlp_data: it records every slot's KMD (UHD 1.1) and installs at the engine.
    """
    from uhd_gen.immediate import validate_model
    from uhd_gen.promote import _apply, _validate_descriptor, build_plan
    from uhd_gen.provenance import snapshot_provenance
    from uhd_gen.tests.test_promote_slots import KMD_DKDV, KMD_DQ, KMD_PREP, _tree

    tree = _tree(tmp_path / "descriptors")
    provenance = {
        **snapshot_provenance(tree, arch="gfx942"),
        "selector_revision": "provider-1",
    }
    corpus = _l1_corpus(tmp_path / "corpus.json")
    rows = json.loads(corpus.read_text(encoding="utf-8"))
    for row in rows:
        row["engine_name"] = row["binding"]["engine"] = "test:slotted"
        row["binding"]["selector_revision"] = "provider-1"
        row["binding"]["trained_against"] = provenance
    corpus.write_text(json.dumps(rows), encoding="utf-8")

    code, output = _train(tmp_path, evaluator)
    assert code == 0
    descriptor_path = output / "heuristic.uhd.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert descriptor["version"] == "1.1"
    assert descriptor["adapter"] == "mlp_data"
    assert descriptor["score"]["metric"] == "tflops"
    assert {
        entry["slot"]: entry["id"] for entry in descriptor["trained_against"]["kmds"]
    } == {
        "prep": KMD_PREP,
        "dkdv": KMD_DKDV,
        "dq": KMD_DQ,
    }
    # The validators every consumer runs accept it.
    _validate_descriptor(descriptor, descriptor_path)
    assert validate_model(descriptor).name == "tflops"

    original = json.loads((tree / "engine.ued.json").read_text(encoding="utf-8"))
    plan = build_plan(
        output, tree, role="predict_engine", arch="gfx942", feature_evaluator=evaluator
    )
    _apply(plan)
    ued = json.loads((tree / "engine.ued.json").read_text(encoding="utf-8"))
    assert plan.slot is None
    assert ued["predict_engine"] == {"gfx942": [descriptor["id"]]}
    assert ued["slots"] == original["slots"]


def _probe_corpus(path, count=40):
    """An engine no UED describes, bound by selector revision alone."""
    rows = []
    for index in range(count):
        flops = 1e12 + index * 1e10
        rows.append(
            {
                "engine_id": 1,
                "engine_name": "probe:A",
                "graph_id": f"g{index}",
                "graph_key": f"g{index}",
                "device_id": "board",
                "arch": "gfx942",
                "binding": {
                    "engine": "probe:A",
                    "role": "predict_engine",
                    "metric": "tflops",
                    "arch": "gfx942",
                    "selector_revision": "probe-1",
                    "trained_against": {"selector_revision": "probe-1"},
                },
                "metric": "tflops",
                "features": {
                    "graph.flops": flops,
                    "graph.nodes[0].x.dims[0]": index + 1,
                },
                "avgTimeMs": 1 + index * 0.1,
                "robustMeanMs": 1 + index * 0.1,
                "is_valid": True,
                "selection_mode": "immediate",
                "timing_statistic": "robustMeanMs",
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "incumbent, incoming", [("tree_data", "mlp_data"), ("mlp_data", "tree_data")]
)
def test_a_model_replaces_the_installed_one_of_its_slot_across_adapters(
    tmp_path, evaluator, incumbent, incoming
):
    from uhd_gen.promote import _apply, build_plan
    from uhd_gen.tests.test_promote import _tree

    identity = "d7b3a1c0-5f7e-4b0c-9d0e-1a2b3c4d5e6f"
    corpus = _probe_corpus(tmp_path / "corpus.json")
    tree = _tree(tmp_path / "tree")

    def train(adapter):
        output = tmp_path / adapter
        options = (
            TRAIN_OPTIONS
            if adapter == "mlp_data"
            else ["--num-boost-round", "2", "--early-stopping", "1"]
        )
        assert (
            main(
                [
                    "train",
                    "--role",
                    "predict_engine",
                    "--engine",
                    "probe:A",
                    "--adapter",
                    adapter,
                    "--input",
                    str(corpus),
                    "--features",
                    "graph.flops",
                    "graph.nodes[0].x.dims[0]",
                    "--metric",
                    "tflops",
                    "--uhd-id",
                    identity,
                    "--feature-evaluator",
                    evaluator,
                    "--output-dir",
                    str(output),
                    *options,
                ]
            )
            == 0
        )
        return output

    def promote(model):
        return build_plan(
            model,
            tree,
            "probe:A",
            role="predict_engine",
            arch="gfx942",
            uhd_ids={"tflops": identity},
            feature_evaluator=evaluator,
        )

    # The incumbent ships beside the provider, outside promote's own layout.
    shipped = tree / "shipped"
    shutil.copytree(train(incumbent), shipped)
    plan = promote(train(incoming))
    assert plan.destination_descriptor == shipped / "heuristic.uhd.json"
    assert any("replacing" in warning for warning in plan.warnings)
    _apply(plan)
    installed = json.loads(plan.destination_descriptor.read_text(encoding="utf-8"))
    assert installed["adapter"] == incoming
    assert incumbent not in installed
    artifact = shipped / installed[incoming]["artifact"]
    assert (
        hashlib.sha256(artifact.read_bytes()).hexdigest() == installed[incoming]["hash"]
    )
    if incoming == "mlp_data":
        load_mlp_artifact(artifact, installed[incoming]["hash"], feature_count=2)


def _catalog_frame():
    """The linear_data catalog corpus, its two kernel knobs named as the promote tree's KMD."""
    from uhd_gen.tests.test_linear_model import _catalog

    return _catalog(problems=30).rename(
        columns={"kernel.tile": "kernel.block_size", "kernel.stages": "kernel.tile_m"}
    )


def _train_catalog(
    tmp_path, evaluator, provenance, *extra, frame=None, metric="tflops"
):
    corpus = tmp_path / "catalog.csv"
    (_catalog_frame() if frame is None else frame).to_csv(corpus, index=False)
    snapshot = tmp_path / "provenance.json"
    snapshot.write_text(json.dumps(provenance), encoding="utf-8")
    output = tmp_path / "catalog_model"
    code = main(
        [
            "train",
            "--role",
            "sort_kernel_catalog",
            "--adapter",
            "mlp_data",
            "--input",
            str(corpus),
            "--provenance",
            str(snapshot),
            "--features",
            "kernel.block_size",
            "kernel.tile_m",
            "q.size",
            "q.size2",
            *(["--metric", metric] if metric else []),
            "--group-by",
            "benchmark",
            "device",
            "--feature-evaluator",
            evaluator,
            "--output-dir",
            str(output),
            "--training-arches",
            "gfx942",
            *TRAIN_OPTIONS,
            *extra,
        ]
    )
    return code, output


def _scored_fraction_of_best(frame, scores):
    """Mean over problems of the picked candidate's TFLOPS over the problem's best."""
    picked = (
        frame.assign(score=scores).sort_values("score").groupby("benchmark").tail(1)
    )
    best = frame.groupby("benchmark")["tflops"].max()
    return float((picked.set_index("benchmark")["tflops"] / best).mean())


def test_a_catalog_trained_mlp_data_model_ranks_and_is_evaluated_as_a_catalog_ranker(
    tmp_path, evaluator
):
    from uhd_gen.tests.test_promote import _tree
    from uhd_gen.provenance import snapshot_provenance

    provenance = snapshot_provenance(_tree(tmp_path / "tree"), arch="gfx942")
    frame = _catalog_frame()
    code, output = _train_catalog(tmp_path, evaluator, provenance, frame=frame)
    assert code == 0

    descriptor = json.loads((output / "heuristic.uhd.json").read_text(encoding="utf-8"))
    assert descriptor["adapter"] == "mlp_data"
    assert descriptor["objective"] == "max"
    assert descriptor["score"] == {
        "metric": "tflops",
        "calibrated": False,
        "transform": "log",
    }
    artifact = output / "model.bin"
    _, metric = load_mlp_artifact(
        artifact,
        descriptor["mlp_data"]["hash"],
        feature_count=len(descriptor["features_signature"]),
    )
    assert metric == "tflops"
    manifest = json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))
    assert "role" not in manifest and manifest["group_by"] == ["benchmark", "device"]
    # A problem's candidates validate together, never against their own siblings.
    fit = manifest["mlp_fit"]
    assert fit["validation_split"] == "group" and fit["validation_groups"] == 30
    for member in fit["member_fits"]:
        assert member["validation_rows"] % 4 == 0
        assert member["train_rows"] + member["validation_rows"] == 120

    bundle = load_model(output, feature_evaluator=evaluator)
    assert bundle.role != "predict_engine"
    assert _scored_fraction_of_best(frame, bundle.scorer(frame)) > 0.9

    report = tmp_path / "report.json"
    code = main(
        [
            "evaluate",
            "--input",
            str(tmp_path / "catalog.csv"),
            "--model-dir",
            str(output),
            "--eval-fraction",
            "1",
            "--feature-evaluator",
            evaluator,
            "--output",
            str(report),
        ]
    )
    assert code == 0
    metrics = json.loads(report.read_text(encoding="utf-8"))["metrics"]
    assert metrics["problems_scored"] == 30
    assert metrics["top1_regret"]["mean"] < 0.1


def test_a_catalog_trained_mlp_data_model_installs_into_the_catalog_slot(
    tmp_path, evaluator
):
    from uhd_gen.promote import _apply, build_plan
    from uhd_gen.provenance import snapshot_provenance
    from uhd_gen.tests.test_promote import _read, _tree

    tree = _tree(tmp_path / "tree")
    frame = _catalog_frame()
    code, output = _train_catalog(
        tmp_path, evaluator, snapshot_provenance(tree, arch="gfx942"), frame=frame
    )
    assert code == 0
    plan = build_plan(
        output,
        tree,
        role="sort_kernel_catalog",
        arch="gfx942",
        feature_evaluator=evaluator,
    )
    _apply(plan)
    installed = _read(plan.destination_descriptor)
    assert installed["adapter"] == "mlp_data"
    assert (
        installed["id"]
        in _read(tree / "engine.ued.json")["sort_kernel_catalog"]["gfx942"]
    )
    # An installed model has no manifest; it scores as it did before it moved.
    before = load_model(output, feature_evaluator=evaluator).scorer(frame)
    after = load_model(plan.destination_descriptor.parent, feature_evaluator=evaluator)
    assert after.role == "sort_kernel_catalog"
    np.testing.assert_array_equal(after.scorer(frame), before)


def test_a_catalog_mlp_data_model_names_a_metric_because_its_artifact_does(
    tmp_path, evaluator, caplog
):
    from uhd_gen.provenance import snapshot_provenance
    from uhd_gen.tests.test_promote import _tree

    provenance = snapshot_provenance(_tree(tmp_path / "tree"), arch="gfx942")
    code, output = _train_catalog(
        tmp_path, evaluator, provenance, "--target", "q.size", metric=None
    )
    assert code == 1 and "pass --metric" in caplog.text
    assert not output.exists()


def test_a_grouped_validation_split_never_separates_a_groups_rows():
    from uhd_gen.mlp import VALIDATION_FRACTION, _validation_split

    groups = np.repeat(np.arange(20), 5)
    held = {}
    for seed in range(8):
        valid, train = _validation_split(
            len(groups), np.random.default_rng(seed), groups
        )
        assert sorted([*valid, *train]) == list(range(len(groups)))
        assert not set(groups[valid]) & set(groups[train])
        assert len(set(groups[valid])) == round(VALIDATION_FRACTION * 20) == 3
        # Every row of a held-out group is held out with it.
        assert len(valid) == 3 * 5
        held[seed] = tuple(sorted(set(groups[valid])))
    assert len(set(held.values())) > 1, "the held-out groups follow the seed"
    again, _ = _validation_split(len(groups), np.random.default_rng(3), groups)
    assert tuple(sorted(set(groups[again]))) == held[3]


def test_a_grouped_fit_holds_out_groups_and_the_row_fit_is_unchanged():
    matrix, y = _curve(rows=200)
    groups = np.repeat(np.arange(40), 5)
    grouped, fit = fit_mlp(matrix, y, epochs=3, members=2, hidden=(4,), groups=groups)
    assert (fit["validation_split"], fit["validation_groups"]) == ("group", 40)
    for member in fit["member_fits"]:
        assert member["validation_rows"] == 6 * 5 and member["train_rows"] == 34 * 5
    plain, plain_fit = fit_mlp(matrix, y, epochs=3, members=2, hidden=(4,))
    assert (plain_fit["validation_split"], plain_fit["validation_groups"]) == (
        "row",
        None,
    )
    assert plain_fit["member_fits"][0]["validation_rows"] == 30
    assert not _same_networks(grouped, plain)
    # A subsample keeps each row's group, and a lone problem has nothing to validate on.
    _, sampled = fit_mlp(
        matrix, y, epochs=1, members=1, hidden=(4,), groups=groups, max_rows=100
    )
    assert sampled["rows"] == 100
    with pytest.raises(ValueError, match="at least 2 problems"):
        fit_mlp(matrix, y, epochs=1, members=1, groups=np.zeros(200, dtype=int))
