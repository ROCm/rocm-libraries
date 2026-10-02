# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The score transform uhd_gen trains on, and the inverses it can score with.

Mirrors `hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp`: a model is fitted on a
transformed target, the descriptor's `score.transform` names that transform, and the
runtime inverts it (`applyInverse`) to recover a value in `score.metric`'s units.

uhd_gen trains on `log(target)`. The runtime refuses any recovered score that is not
finite and strictly positive (`UhdKernelHeuristic::scoreFromRaw`, RFC 0019 §8.3), and the
inverse of `log` is `exp`, which is positive for every finite raw output. A model trained
this way cannot produce a score the runtime discards, whatever it extrapolates to.

It used to train on `log1p(target)`. `expm1` is negative for any raw output below zero,
and a gradient-boosted ensemble undershoots on the smallest targets it saw -- measured on
the cluster: an AITER gfx950 L1 model trained on 1600 graphs predicted negative TFLOPS
for 4 of 200 unseen graphs, and the engine answered INVALID for each. Training could only
report that after the fact. `log` removes it by construction, and it fits relative error
uniformly, which is what a throughput prediction compared across engines needs.

The labels are strictly positive by the same rule: a zero or negative measurement is no
measurement (§8.3), and `log` of one is undefined, so training refuses them.
"""

from __future__ import annotations

import numpy as np

#: The transform `train` fits and declares.
TRAINED = "log"

#: Transforms uhd_gen can invert, so a model trained before `log` still scores.
INVERTIBLE = ("identity", "log1p", "log")


def forward(target: np.ndarray) -> np.ndarray:
    """Target -> the space the model is fitted in. Targets must be strictly positive."""
    values = np.asarray(target, dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("a log-space target must be finite and strictly positive")
    return np.log(values)


def inverse(raw: np.ndarray, transform: str) -> np.ndarray:
    """Model output -> value in the metric's units, as `applyInverse` computes it.

    A raw score of -inf marks a candidate the ranking rejected; it stays -inf rather than
    becoming the transform's image of -inf (`exp` gives 0, `expm1` gives -1), which would
    be a finite value that outranks a real score.
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
