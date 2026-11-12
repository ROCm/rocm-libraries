# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TPE acquisition over every member of a finite kernel catalog."""

import math

from .surrogate import TPE


class ListedTPE(TPE):
    """Keep TPE's model and protocol, replace its sampled acquisition pool.

    All listed candidates enter the existing legality, history, pending,
    prior and build-preference path. A fresh permutation breaks acquisition
    ties without preferring the catalog's first row. The immutable starting
    order makes the permutation reproducible from the normal RNG checkpoint.
    This evaluates model scores, not the kernels themselves.

    ``gamma_schedule="sqrt"`` takes the best ``ceil(sqrt(n))`` of ``n``
    observations as the good set (at least one stays bad) instead of
    ``ceil(gamma * n)``: a broader good set than the default early in a
    search, a narrower one late.
    """

    name = "tpe_listed"

    def __init__(self, space, seed=0, *, gamma_schedule="linear", **options):
        listed = space.listed()
        if not listed:
            raise ValueError("candidate_pool=listed requires a nonempty listed space")
        if "n_candidates" in options:
            raise ValueError("candidate_pool=listed fixes the pool size; omit n_candidates")
        if gamma_schedule not in ("linear", "sqrt"):
            raise ValueError("TPE gamma_schedule must be linear or sqrt")
        super().__init__(space, seed, n_candidates=len(listed), **options)
        self.gamma_schedule = gamma_schedule
        # Keep the sampled-TPE startup for a paired acquisition comparison.
        self.rng.seed(f"{TPE.name}:{self.seed}")
        self._listed_candidates = tuple(listed)

    def _propose(self, taken, batch_pos):
        candidates = list(self._listed_candidates)
        self.rng.shuffle(candidates)
        self._candidate_iter = iter(candidates)
        gamma = self.gamma
        if self.gamma_schedule == "sqrt":
            # TPE takes ceil(gamma * n) good observations: aim at the middle
            # of the wanted count's interval (n >= 2 when TPE proposes).
            n = sum(map(math.isfinite, self.history.values()))
            self.gamma = (min(math.ceil(math.sqrt(n)), n - 1) - 0.5) / n
        try:
            return super()._propose(taken, batch_pos)
        finally:
            self._candidate_iter = None
            self.gamma = gamma

    def _sample(self, density):
        return next(self._candidate_iter)

    def _load(self, state):
        super()._load(state)
        self.uid_prefix = f"{self.name}:{self.seed}"
