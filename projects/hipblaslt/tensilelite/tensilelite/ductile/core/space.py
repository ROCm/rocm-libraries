# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from .population import Individual, Population, IndividualSet, ExceedsCapacity
from typing import Callable, Union

import os
import math
import numpy as np
import joblib
import contextlib

from ..utils import tqdm

# xdist workers are already parallel processes. Keep inner GA sampling serial to
# avoid nested parallel execution issues (state/process inheritance failures).
IS_XDIST_WORKER = bool(os.environ.get("PYTEST_XDIST_WORKER"))

if IS_XDIST_WORKER:
    JOBLIB_BACKEND = "threading"
    JOBLIB_N_JOBS_OVERRIDE = 1
else:
    JOBLIB_BACKEND = "multiprocessing"
    JOBLIB_N_JOBS_OVERRIDE = None


class MaxIterationsReached(Exception):
    pass


def sample_chunk(valid_fn, p, sizes, chunk_size, seed):
    rng = np.random.default_rng(seed)
    valid_inds = []
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            for _ in range(chunk_size):
                ind = Individual({k: rng.choice(s, p=p.get(k, None)) for k, s in sizes.items()})
                if valid_fn(ind):
                    valid_inds.append(ind)
    return valid_inds

class SearchSpace:
    def __init__(self,
                 space: dict,
                 max_iters: int = 100,
                 valid: Callable = None):
        if not (isinstance(space, dict) and len(space) > 0):
            raise ValueError("space must be a non-empty dictionary")

        self.sizes = {k: len(v) for k, v in space.items()}
        if any(v == 0 for v in self.sizes.values()):
            raise ValueError("some variables have a search space of length = 0")

        self.map = {k: v for k, v in space.items()}
        self.space = {k: list(range(len(v))) for k, v in space.items()}
        self.n_perms = math.prod(v for v in self.sizes.values())

        self.max_iters = max_iters
        self._valid = valid
        self.cache = None
        self.seed_seq = np.random.SeedSequence()
        

    def seed(self, seed: int):
        self.seed_seq = np.random.SeedSequence(seed)

    def valid(self, x):
        if self._valid:
            return self._valid(self.transform(x))
        return True

    def __contains__(self, key: str):
        return key in self.space

    def __getitem__(self, key: str):
        return self.space[key]

    def __iter__(self):
        return dict.__iter__(self.space)

    def __len__(self):
        return len(self.space)

    def size(self, key: str = None) -> int:
        if key is None:
            return len(self)
        return self.sizes[key]

    def transform(self, X: Union[Individual, Population]):
        if isinstance(X, Individual):
            return {k: self.map[k][i] for k, i in X.items}
        return [{k: self.map[k][i] for k, i in x.items} for x in X]

    def sample(self, 
               size: int, 
               p: dict = None, 
               iter_mul: int = 1,
               reuse: bool = False) -> Population:

        p = p if p else {}
        it, max_iters = 0, self.max_iters * size * iter_mul

        n_jobs = max((os.cpu_count() or 1) // 12, 1)
        if JOBLIB_N_JOBS_OVERRIDE is not None:
            n_jobs = JOBLIB_N_JOBS_OVERRIDE
        
        if reuse and self.cache:
            pop = IndividualSet(self.cache, capacity=max(size, len(self.cache)))
        else:
            pop = IndividualSet(capacity=size)

        with tqdm(total=size) as pbar:
            completed = len(pop)
            pbar.update(completed)
            while it < max_iters:
                total_to_generate = max((size - len(pop)) * 25, n_jobs * 4)
                chunk_size = total_to_generate // n_jobs
                seeds = self.seed_seq.spawn(n_jobs)
        
                res = joblib.Parallel(n_jobs=n_jobs, backend=JOBLIB_BACKEND)(
                    joblib.delayed(sample_chunk)(self.valid, p, self.sizes, chunk_size, seed) 
                    for seed in seeds
                )
                try:
                    sampled = [ind for chunk in res for ind in chunk]
                    for ind in sampled:
                        pop.add(ind)
                    it += total_to_generate
                    pbar.update(len(pop) - completed)
                    completed = len(pop)
                except ExceedsCapacity:
                    pbar.update(len(pop) - completed)
                    break 

        self.cache = list(pop)
        if len(pop) < size:
            raise MaxIterationsReached(f"max iters reached while sampling a population: {len(pop)}/{size}", len(pop))
        
        return Population(pop)

    def __repr__(self):
        return f"SearchSpace(num_vars={len(self)}, n_perms={float(self.n_perms)}, counts={self.sizes})"
