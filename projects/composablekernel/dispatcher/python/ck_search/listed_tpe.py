# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TPE acquisition over every member of a finite kernel catalog."""

from .surrogate import TPE


class ListedTPE(TPE):
    """Keep TPE's model and protocol, replace its sampled acquisition pool.

    All listed candidates enter the existing legality, history, pending,
    prior and build-preference path. A fresh permutation breaks acquisition
    ties without preferring the catalog's first row. The immutable starting
    order makes the permutation reproducible from the normal RNG checkpoint.
    This evaluates model scores, not the kernels themselves.
    """

    name = "tpe_listed"

    def __init__(self, space, seed=0, **options):
        listed = space.listed()
        if not listed:
            raise ValueError("candidate_pool=listed requires a nonempty listed space")
        if "n_candidates" in options:
            raise ValueError("candidate_pool=listed fixes the pool size; omit n_candidates")
        super().__init__(space, seed, n_candidates=len(listed), **options)
        # Keep the sampled-TPE startup for a paired acquisition comparison.
        self.rng.seed(f"{TPE.name}:{self.seed}")
        self._listed_candidates = tuple(listed)

    def _propose(self, taken, batch_pos):
        candidates = list(self._listed_candidates)
        self.rng.shuffle(candidates)
        self._candidate_iter = iter(candidates)
        try:
            return super()._propose(taken, batch_pos)
        finally:
            self._candidate_iter = None

    def _sample(self, density):
        return next(self._candidate_iter)

    def _load(self, state):
        super()._load(state)
        self.uid_prefix = f"{self.name}:{self.seed}"
