# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RelLin: the per-configuration linear ranker behind the `linear_data` adapter.

A candidate's score is its predicted log slowdown against the best candidate of the same
problem (lower is better), as a linear function of a standardized basis fitted per kernel
configuration (linear_model.fbs):

* every configuration has its own ridge regression, shrunk toward its family's and, through
  the family, toward one fitted on every row (`ridge` scales with the rows' weight);
* a configuration with fewer than `min_rows` rows is shrunk much harder, because its own rows
  say too little about it;
* the guard: every basis term is clamped to the range it took in training, and a
  configuration's prediction never falls below the smallest slowdown it showed in training, so
  a configuration that never won cannot be extrapolated into a winner.

`LinearRanker.predict` is the definition the runtime's `LinearDataAdapter` implements;
`evaluate` scores a shipped artifact with it, so the two agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: `LinearBasisTerm.transform` values (linear_model.fbs).
IDENTITY = 0
LOG1P = 1
#: A term is log-transformed when it is non-negative and spans more than this: a size-like
#: feature whose range is orders of magnitude, where the slowdown moves with its log.
LOG_RANGE_THRESHOLD = 64.0

DEFAULT_RIDGE = 1e-2
DEFAULT_MIN_ROWS = 10
#: Ridge weight, per basis column, of a configuration with too few rows to trust.
_THIN_CONFIG_PENALTY = 10.0
#: Keeps a family's solve well posed when it has few rows.
_FAMILY_JITTER = 1e-6


def apply_transform(values: np.ndarray, transform: int) -> np.ndarray:
    """`t(x)` of a basis term; NaN stays NaN."""
    if transform == LOG1P:
        return np.log1p(np.maximum(values, 0.0))
    return values


def relative_log_slowdown(
    values: np.ndarray, problems: np.ndarray, objective: str
) -> np.ndarray:
    """Each row's log slowdown against the best row of its problem, in the objective's
    direction: `log t - log min t` for a cost (`min`), `log max v - log v` for a rate (`max`).
    Zero for the best row, positive otherwise."""
    import pandas as pd

    if objective not in ("min", "max"):
        raise ValueError(f"objective must be 'min' or 'max', got {objective!r}")
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(
            "RelLin trains on log slowdown, so every target value must be finite and positive"
        )
    series = pd.Series(values)
    grouped = series.groupby(np.asarray(problems))
    if objective == "min":
        return np.log(values) - np.log(grouped.transform("min").to_numpy())
    return np.log(grouped.transform("max").to_numpy()) - np.log(values)


def _canonical_key(values: np.ndarray) -> np.ndarray:
    """One comparable row per key: the value with NaN and -0.0 folded away, then a NaN flag
    column per slot, so keys equal exactly as `LinearDataAdapter` compares them (doubles,
    every NaN equal)."""
    values = np.asarray(values, dtype=np.float64)
    missing = np.isnan(values)
    return np.concatenate([np.where(missing, 0.0, values) + 0.0, missing], axis=1)


def _key_tuples(values: np.ndarray) -> list[tuple]:
    return list(map(tuple, _canonical_key(values).tolist()))


@dataclass
class LinearRanker:
    """A fitted model, exactly as linear_model.fbs stores it."""

    num_features: int
    key_features: list[int]
    family_features: list[int]
    # One entry per basis term.
    basis_features: np.ndarray
    transforms: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    # Per entry: key values (n, width), coefficients (n, 1 + terms) and floors (n,).
    config_keys: np.ndarray
    config_coefficients: np.ndarray
    config_floors: np.ndarray
    family_keys: np.ndarray
    family_coefficients: np.ndarray
    family_floors: np.ndarray
    global_coefficients: np.ndarray
    global_floor: float = 0.0
    _indices: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def terms(self) -> int:
        return len(self.basis_features)

    def basis(self, matrix: np.ndarray) -> np.ndarray:
        """z for each row: standardized, clamped to the training range; a NaN input is 0."""
        matrix = np.asarray(matrix, dtype=np.float64)
        z = np.empty((len(matrix), self.terms))
        for j in range(self.terms):
            x = matrix[:, self.basis_features[j]]
            t = apply_transform(x, int(self.transforms[j]))
            standardized = np.clip(
                (t - self.mean[j]) / self.scale[j], self.lo[j], self.hi[j]
            )
            z[:, j] = np.where(np.isnan(x), 0.0, standardized)
        return z

    def _lookup(self, name: str, keys: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """The index in `keys` of each row's key at the `name` slots, or -1."""
        slots = self.key_features if name == "config" else self.family_features
        if not len(keys):
            return np.full(len(rows), -1, dtype=np.int64)
        index = self._indices.get(name)
        if index is None:
            index = {key: i for i, key in enumerate(_key_tuples(keys))}
            self._indices[name] = index
        wanted = _key_tuples(rows[:, slots])
        return np.fromiter(
            (index.get(key, -1) for key in wanted), dtype=np.int64, count=len(wanted)
        )

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """The predicted log slowdown of each row: `max(c . [1, z], floor)` for the entry the
        row's configuration, else its family, else the global fit selects."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.num_features:
            raise ValueError(
                f"feature matrix has width {matrix.shape[-1]}, the model expects "
                f"{self.num_features}"
            )
        z = self.basis(matrix)
        coefficients = np.tile(self.global_coefficients, (len(matrix), 1))
        floors = np.full(len(matrix), self.global_floor)
        if len(self.family_keys):
            hit = self._lookup("family", self.family_keys, matrix)
            seen = hit >= 0
            coefficients[seen] = self.family_coefficients[hit[seen]]
            floors[seen] = self.family_floors[hit[seen]]
        hit = self._lookup("config", self.config_keys, matrix)
        seen = hit >= 0
        coefficients[seen] = self.config_coefficients[hit[seen]]
        floors[seen] = self.config_floors[hit[seen]]
        scores = coefficients[:, 0] + (coefficients[:, 1:] * z).sum(axis=1)
        return np.maximum(scores, floors)


def select_basis(
    matrix: np.ndarray, key_features: list[int]
) -> tuple[list[int], list[int]]:
    """The model inputs: every non-key slot that varies in training, log1p-transformed when it
    is non-negative and spans more than `LOG_RANGE_THRESHOLD`. Returns (slots, transforms).
    """
    slots, transforms = [], []
    for slot in range(matrix.shape[1]):
        if slot in key_features:
            continue
        column = matrix[:, slot]
        column = column[~np.isnan(column)]
        if not len(column) or column.min() == column.max():
            continue
        slots.append(slot)
        transforms.append(
            LOG1P
            if column.min() >= 0 and column.max() > LOG_RANGE_THRESHOLD
            else IDENTITY
        )
    return slots, transforms


def _group(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(first row of each distinct key in key order, each row's group index)."""
    canonical = _canonical_key(keys)
    if canonical.shape[1] == 0:  # no key slots: every row is the one configuration
        return np.zeros(1, dtype=np.int64), np.zeros(len(canonical), dtype=np.int64)
    _, first, inverse = np.unique(
        canonical, axis=0, return_index=True, return_inverse=True
    )
    return first, inverse.reshape(-1)


def fit_rellin(
    matrix: np.ndarray,
    slowdown: np.ndarray,
    *,
    key_features: list[int],
    family_features: list[int] | None = None,
    basis_features: list[int],
    transforms: list[int],
    ridge: float = DEFAULT_RIDGE,
    min_rows: int = DEFAULT_MIN_ROWS,
    clip: float | None = None,
    tau: float | None = None,
) -> LinearRanker:
    """Fit the ranker on feature rows and their log slowdowns (`relative_log_slowdown`).

    `clip` caps the targets, so a configuration far slower than the best is only "bad"; `tau`
    weights rows by exp(-slowdown / tau), so the fit is exact where the winner is decided.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(slowdown, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(y) or not len(y):
        raise ValueError(
            "RelLin needs one slowdown per feature row, and at least one row"
        )
    family_features = list(family_features or [])
    if len(basis_features) != len(transforms):
        raise ValueError("every basis term needs a transform")

    weights = np.exp(-y / tau) if tau else np.ones_like(y)
    if clip:
        y = np.minimum(y, clip)

    # Standardize the basis. A term that does not vary is left unscaled (mean 0, scale 1),
    # so its clamp pins it to the one value it has.
    raw = np.column_stack(
        [
            apply_transform(matrix[:, slot], transform)
            for slot, transform in zip(basis_features, transforms)
        ]
        or [np.empty((len(y), 0))]
    )
    mean = np.zeros(raw.shape[1])
    scale = np.ones(raw.shape[1])
    for j in range(raw.shape[1]):
        column = raw[:, j][~np.isnan(raw[:, j])]
        spread = column.std() if len(column) else 0.0
        if not np.isclose(spread, 0):
            mean[j], scale[j] = column.mean(), spread
    standardized = (raw - mean) / scale
    # A missing value stands at the training mean, as the runtime reads it.
    z = np.where(np.isnan(standardized), 0.0, standardized)
    lo = z.min(axis=0) if len(z) else np.zeros(0)
    hi = z.max(axis=0) if len(z) else np.zeros(0)

    design = np.column_stack([np.ones(len(y)), z])
    width = design.shape[1]

    def solve(rows: np.ndarray, prior: np.ndarray, penalty: float) -> np.ndarray:
        d, target, weight = design[rows], y[rows], weights[rows]
        weighted = d * weight[:, None]
        system = weighted.T @ d + penalty * np.eye(width)
        return np.linalg.solve(system, weighted.T @ target + penalty * prior)

    everything = np.ones(len(y), dtype=bool)
    global_fit = solve(everything, np.zeros(width), ridge * max(1.0, weights.sum()))

    config_first, config_of = _group(matrix[:, key_features])
    if family_features:
        family_first, family_of = _group(matrix[:, family_features])
        family_fits = np.stack(
            [
                solve(
                    family_of == f,
                    global_fit,
                    ridge * max(1.0, weights[family_of == f].sum()) + _FAMILY_JITTER,
                )
                for f in range(len(family_first))
            ]
        )
    else:
        family_first = np.empty(0, dtype=np.int64)
        family_of = np.zeros(len(y), dtype=np.int64)
        family_fits = global_fit[None, :]

    config_fits = np.empty((len(config_first), width))
    config_floors = np.empty(len(config_first))
    for c in range(len(config_first)):
        rows = config_of == c
        prior = family_fits[family_of[np.argmax(rows)]]
        penalty = ridge * max(1.0, weights[rows].sum())
        if rows.sum() < min_rows:
            penalty += _THIN_CONFIG_PENALTY * width
        config_fits[c] = solve(rows, prior, penalty)
        config_floors[c] = y[rows].min()

    return LinearRanker(
        num_features=matrix.shape[1],
        key_features=list(key_features),
        family_features=family_features,
        basis_features=np.asarray(basis_features, dtype=np.int64),
        transforms=np.asarray(transforms, dtype=np.uint8),
        mean=mean,
        scale=scale,
        lo=lo,
        hi=hi,
        config_keys=matrix[config_first][:, key_features],
        config_coefficients=config_fits,
        config_floors=config_floors,
        family_keys=matrix[family_first][:, family_features],
        family_coefficients=family_fits if family_features else np.empty((0, width)),
        # An unseen configuration has no training slowdown to floor at, and slowdowns are
        # not negative: the floor of a family or the global fit is 0.
        family_floors=np.zeros(len(family_first)),
        global_coefficients=global_fit,
        global_floor=0.0,
    )
