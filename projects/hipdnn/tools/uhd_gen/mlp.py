# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`mlp_data`: an ensemble of small fully connected networks estimating one engine's metric.

The model of `MlpDataAdapter` (mlp_model.fbs), fitted and evaluated in numpy so training adds
no dependency. A raw feature row x (NaN = missing) becomes

    t = sign(x) * log1p(|x|)           (NaN -> 0)
    z = (t - mean) / scale              per input slot
    input = [z..., isnan(x[m]) for each missing-flag slot m]

and every member is `Linear -> SiLU` per hidden layer and a final one-output `Linear`. The score
is `target_mean + target_scale * mean over members`, the predicted log of the metric. Prediction
runs in float32 with the ensemble mean in double, as the runtime does.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Limits `MlpDataAdapter::loadFromBuffer` enforces; an artifact beyond them is refused.
MAX_FEATURES = 4096
MAX_WIDTH = 512
MAX_LAYERS = 8
MAX_MEMBERS = 64

DEFAULT_HIDDEN = (64, 64)
DEFAULT_MEMBERS = 3
DEFAULT_EPOCHS = 300
DEFAULT_SEED = 0

#: The recipe the model was developed with.
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
BATCH_SIZE = 256
VALIDATION_FRACTION = 0.15
PATIENCE = 25
#: Added to the target's spread so a constant target still has a positive scale.
TARGET_SCALE_EPSILON = 1e-9

Layer = tuple[np.ndarray, np.ndarray]


@dataclass
class MlpEnsemble:
    """Fitted networks and the transforms around them.

    `networks[m]` is member m's layers, each `(weights, biases)` as float32 with weights
    `[outputs, inputs]`; a member's last layer has one output.
    """

    num_features: int
    input_slots: np.ndarray
    input_mean: np.ndarray
    input_scale: np.ndarray
    missing_slots: np.ndarray
    networks: list[list[Layer]]
    target_mean: float
    target_scale: float

    @property
    def input_width(self) -> int:
        return len(self.input_slots) + len(self.missing_slots)

    def network_input(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The float32 network input of each row, and which rows are encodable.

        A row is not when a standardized input is not finite (an infinity in the feature
        row); its input is zeroed so it flows through harmlessly.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.num_features:
            raise ValueError(
                f"feature rows must have {self.num_features} columns, got shape "
                f"{matrix.shape}"
            )
        x = matrix[:, self.input_slots]
        with np.errstate(invalid="ignore"):
            t = np.where(np.isnan(x), 0.0, np.copysign(np.log1p(np.abs(x)), x))
            z = (t - self.input_mean) / self.input_scale
        finite = np.isfinite(z).all(axis=1)
        z[~finite] = 0.0
        flags = np.isnan(matrix[:, self.missing_slots])
        return np.concatenate([z, flags], axis=1).astype(np.float32), finite

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """The predicted log of the metric per row; NaN for a row the runtime declines."""
        inputs, finite = self.network_input(matrix)
        total = np.zeros(len(inputs), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for layers in self.networks:
                total += _forward(layers, inputs).astype(np.float64)
        scores = self.target_mean + self.target_scale * (total / len(self.networks))
        scores[~finite] = np.nan
        return scores


def _silu(z: np.ndarray) -> np.ndarray:
    return z / (np.float32(1.0) + np.exp(-z))


def _forward(layers: list[Layer], inputs: np.ndarray) -> np.ndarray:
    """One member's output per row, float32."""
    a = inputs
    for index, (weights, biases) in enumerate(layers):
        a = a @ weights.T + biases
        if index + 1 < len(layers):
            a = _silu(a)
    return a[:, 0]


def check_shape(input_width: int, hidden: tuple[int, ...], members: int) -> None:
    """Refuse a network the runtime would refuse to load, before any time is spent fitting."""
    if not 1 <= input_width <= MAX_WIDTH:
        raise ValueError(
            f"the network input has {input_width} values (inputs plus missing flags); the "
            f"runtime loads between 1 and {MAX_WIDTH}"
        )
    if not 1 <= len(hidden) + 1 <= MAX_LAYERS:
        raise ValueError(
            f"{len(hidden)} hidden layer(s) make {len(hidden) + 1} layers; the runtime loads "
            f"between 1 and {MAX_LAYERS}"
        )
    if any(not 1 <= width <= MAX_WIDTH for width in hidden):
        raise ValueError(
            f"hidden widths {list(hidden)} must each be between 1 and {MAX_WIDTH}"
        )
    if not 1 <= members <= MAX_MEMBERS:
        raise ValueError(f"members must be between 1 and {MAX_MEMBERS}, got {members}")


def _initial_layers(
    sizes: list[int], rng: np.random.Generator
) -> list[list[np.ndarray]]:
    """PyTorch's `nn.Linear` default: weights and biases ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    layers = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        bound = 1.0 / math.sqrt(fan_in)
        weights = rng.uniform(-bound, bound, size=(fan_out, fan_in)).astype(np.float32)
        biases = rng.uniform(-bound, bound, size=fan_out).astype(np.float32)
        layers.append([weights, biases])
    return layers


def _validation_split(
    rows: int, seed_rng: np.random.Generator, groups: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (validation, train). Without `groups`, a seeded 15% of rows are held out.
    With them (one id per row), a seeded 15% of the groups are, and every row of a held-out
    group with them, so no problem's rows straddle the split."""
    if groups is None:
        held = min(rows - 1, max(1, round(VALIDATION_FRACTION * rows)))
        order = seed_rng.permutation(rows)
        return order[:held], order[held:]
    ids = np.unique(groups)
    held = min(len(ids) - 1, max(1, round(VALIDATION_FRACTION * len(ids))))
    chosen = ids[seed_rng.permutation(len(ids))[:held]]
    is_valid = np.isin(groups, chosen)
    return np.flatnonzero(is_valid), np.flatnonzero(~is_valid)


def _fit_member(
    inputs: np.ndarray,
    target: np.ndarray,
    hidden: tuple[int, ...],
    epochs: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[list[Layer], dict]:
    """One network: AdamW on L1 loss with cosine-annealed rate, minibatches, a seeded 15%
    validation split (of rows, or of `groups` when given), early stopping on its L1 and the
    best weights restored."""
    rng = np.random.default_rng(seed)
    rows = len(inputs)
    valid, train = _validation_split(rows, rng, groups)
    held = len(valid)
    x_train, y_train = inputs[train], target[train]
    x_valid, y_valid = inputs[valid], target[valid]

    layers = _initial_layers([inputs.shape[1], *hidden, 1], rng)
    params = [p for layer in layers for p in layer]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    one = np.float32(1.0)

    def forward_all(x: np.ndarray) -> np.ndarray:
        return _forward([(w, b) for w, b in layers], x)

    best_loss, best_epoch, best_params = math.inf, -1, None
    step = 0
    epochs_run = 0
    with np.errstate(over="ignore"):
        for epoch in range(epochs):
            rate = 0.5 * LEARNING_RATE * (1.0 + math.cos(math.pi * epoch / epochs))
            shuffled = rng.permutation(len(x_train))
            for begin in range(0, len(shuffled), BATCH_SIZE):
                batch = shuffled[begin : begin + BATCH_SIZE]
                x, y = x_train[batch], y_train[batch]
                # Forward, keeping the pre-activations and activations.
                activations = [x]
                pre = []
                a = x
                for index, (weights, biases) in enumerate(layers):
                    z = a @ weights.T + biases
                    pre.append(z)
                    a = _silu(z) if index + 1 < len(layers) else z
                    activations.append(a)
                # d(mean |p - y|)/dp
                grad = (np.sign(a[:, 0] - y) / np.float32(len(batch)))[:, None]
                grads = [None] * len(params)
                for index in range(len(layers) - 1, -1, -1):
                    weights = layers[index][0]
                    grads[2 * index] = grad.T @ activations[index]
                    grads[2 * index + 1] = grad.sum(axis=0)
                    if index:
                        s = one / (one + np.exp(-pre[index - 1]))
                        grad = (grad @ weights) * (
                            s * (one + pre[index - 1] * (one - s))
                        )
                step += 1
                correction1 = 1.0 - BETAS[0] ** step
                correction2 = 1.0 - BETAS[1] ** step
                for p, g, m, v in zip(params, grads, first, second):
                    p *= np.float32(1.0 - rate * WEIGHT_DECAY)
                    m *= np.float32(BETAS[0])
                    m += np.float32(1.0 - BETAS[0]) * g
                    v *= np.float32(BETAS[1])
                    v += np.float32(1.0 - BETAS[1]) * g * g
                    denominator = np.sqrt(v) / np.float32(math.sqrt(correction2))
                    denominator += np.float32(ADAM_EPSILON)
                    p -= np.float32(rate / correction1) * m / denominator
            epochs_run = epoch + 1
            loss = float(np.mean(np.abs(forward_all(x_valid) - y_valid)))
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                best_params = [p.copy() for p in params]
            elif epoch - best_epoch >= PATIENCE:
                break
    if best_params is None:
        raise ValueError(
            "the network did not reach a finite validation loss; the features or target "
            "hold values it cannot learn from"
        )
    restored = [
        (best_params[2 * i], best_params[2 * i + 1]) for i in range(len(layers))
    ]
    train_loss = float(np.mean(np.abs(_forward(restored, x_train) - y_train)))
    return restored, {
        "seed": seed,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "train_rows": int(len(train)),
        "validation_rows": int(held),
        "train_l1": train_loss,
        "validation_l1": best_loss,
    }


def fit_mlp(
    matrix: np.ndarray,
    y_log: np.ndarray,
    *,
    hidden: tuple[int, ...] = DEFAULT_HIDDEN,
    members: int = DEFAULT_MEMBERS,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
    max_rows: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[MlpEnsemble, dict]:
    """Fit an ensemble of `members` networks to `y_log`, the log of the metric, over the
    feature `matrix` (NaN = missing). Returns it and what the manifest records.

    Member k is seeded `seed + k` for its initial weights, its validation split and its
    shuffles, so the fit is a function of its arguments. `max_rows` > 0 subsamples a larger
    corpus (seeded, row order kept); 0 uses every row.

    `groups` (one id per row, e.g. the problem a catalog row measures) makes the early-stopping
    validation split hold out whole groups instead of rows, so no problem's candidates train
    the network that is validated on a sibling. Omitted, the split is of rows.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    y_log = np.asarray(y_log, dtype=np.float64)
    hidden = tuple(int(width) for width in hidden)
    if matrix.ndim != 2 or len(matrix) != len(y_log):
        raise ValueError(
            "the feature matrix and the target must have one row per sample"
        )
    if len(matrix) < 2:
        raise ValueError(
            "an MLP needs at least 2 rows: one to train on, one to validate"
        )
    if not 1 <= matrix.shape[1] <= MAX_FEATURES:
        raise ValueError(
            f"{matrix.shape[1]} features; the runtime loads between 1 and {MAX_FEATURES}"
        )
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if max_rows < 0:
        raise ValueError("max_rows must be 0 (every row) or positive")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(matrix),):
            raise ValueError("groups must hold one id per row")
    if not np.isfinite(y_log).all():
        raise ValueError("the log target must be finite")
    if np.isinf(matrix).any():
        raise ValueError(
            "feature values must be finite or NaN: the runtime declines infinities"
        )

    total_rows = len(matrix)
    if max_rows and total_rows > max_rows:
        keep = np.sort(
            np.random.default_rng(seed).choice(total_rows, size=max_rows, replace=False)
        )
        matrix, y_log = matrix[keep], y_log[keep]
        if groups is not None:
            groups = groups[keep]
    if groups is not None and len(np.unique(groups)) < 2:
        raise ValueError(
            "a grouped validation split needs at least 2 problems: one to train on, "
            "one to validate"
        )
    rows = len(matrix)

    t = np.where(np.isnan(matrix), 0.0, np.copysign(np.log1p(np.abs(matrix)), matrix))
    mean = t.mean(axis=0)
    std = t.std(axis=0)
    # Exactly constant columns carry nothing; the ptp test keeps rounding in the mean of a
    # constant column from passing for spread.
    inputs = np.flatnonzero((np.ptp(t, axis=0) > 0) & (std > 0))
    nan_rows = np.isnan(matrix).sum(axis=0)
    missing = np.flatnonzero((nan_rows > 0) & (nan_rows < rows))
    check_shape(len(inputs) + len(missing), hidden, members)

    target_mean = float(y_log.mean())
    target_scale = float(y_log.std()) + TARGET_SCALE_EPSILON
    ensemble = MlpEnsemble(
        num_features=int(matrix.shape[1]),
        input_slots=inputs.astype(np.int64),
        input_mean=mean[inputs],
        input_scale=std[inputs],
        missing_slots=missing.astype(np.int64),
        networks=[],
        target_mean=target_mean,
        target_scale=target_scale,
    )
    network_input, finite = ensemble.network_input(matrix)
    if not finite.all():
        raise ValueError("a standardized training input is not finite")
    standardized = ((y_log - target_mean) / target_scale).astype(np.float32)

    summaries = []
    for member in range(members):
        layers, summary = _fit_member(
            network_input, standardized, hidden, epochs, seed + member, groups
        )
        ensemble.networks.append(layers)
        summaries.append(summary)
        logger.info(
            "mlp member %d (seed %d): %d epochs, best %d, validation L1 %.4f (log units %.4f)",
            member,
            seed + member,
            summary["epochs_run"],
            summary["best_epoch"],
            summary["validation_l1"],
            summary["validation_l1"] * target_scale,
        )
    return ensemble, {
        "adapter": "mlp_data",
        "hidden": list(hidden),
        "members": members,
        "epochs": epochs,
        "seed": seed,
        "max_rows": max_rows,
        "validation_split": "group" if groups is not None else "row",
        "validation_groups": None if groups is None else int(len(np.unique(groups))),
        "rows": int(rows),
        "rows_available": int(total_rows),
        "input_slots": [int(slot) for slot in inputs],
        "missing_flag_slots": [int(slot) for slot in missing],
        "target_mean": target_mean,
        "target_scale": target_scale,
        "member_fits": summaries,
        "validation_l1_log": float(
            np.mean([s["validation_l1"] for s in summaries]) * target_scale
        ),
    }
