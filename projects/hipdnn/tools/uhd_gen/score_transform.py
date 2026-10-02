# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The score transform uhd_gen trains on, and the inverses it can score with.

Mirrors `hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp`. Training uses `log`
because its inverse `exp` is positive for any finite output, so the runtime never
discards a prediction as non-positive (`log1p`/`expm1` could). Labels must be > 0.
"""

from __future__ import annotations

import numpy as np

#: The transform `train` fits and declares.
TRAINED = "log"

#: Transforms uhd_gen can invert, so older `log1p`/`identity` models still score.
INVERTIBLE = ("identity", "log1p", "log")


def forward(target: np.ndarray) -> np.ndarray:
    """Target -> the space the model is fitted in. Targets must be strictly positive."""
    values = np.asarray(target, dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("a log-space target must be finite and strictly positive")
    return np.log(values)


def inverse(raw: np.ndarray, transform: str) -> np.ndarray:
    """Model output -> value in the metric's units, as `applyInverse` computes it.

    -inf (a rejected candidate) stays -inf rather than becoming a finite value.
    """
    if transform not in INVERTIBLE and transform != "":
        raise ValueError(
            f"uhd_gen can only score {', '.join(INVERTIBLE)} transforms, not {transform!r}"
        )
    values = np.asarray(raw, dtype=np.float64)
    if transform == "log1p":
        recovered = np.expm1(values)
    elif transform == "log":
        with np.errstate(over="ignore"):
            recovered = np.exp(values)
    else:
        return values
    recovered = np.atleast_1d(np.array(recovered, dtype=np.float64))
    recovered[np.atleast_1d(values) == -np.inf] = -np.inf
    return recovered.reshape(values.shape)
