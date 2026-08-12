# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
from ..core import Population, SearchSpace, Mating, Survival
from ..utils import Logger
from ..config import DEFAULTS
from ..core.space import MaxIterationsReached
from ..core.mating import MatingExhaustedError
from typing import Callable
from pathlib import Path

import os
import pickle
import numpy as np
import random
import logging
import time


# noinspection DuplicatedCode
class GeneticAlgorithm:
    name = "GA"

    def __init__(self,
                 space: SearchSpace,
                 mating: Mating,
                 evaluate: Callable,
                 survival: Survival = Survival.get("fitness"),
                 pop_size: int = 512,
                 n_gen: int = 20,
                 soo: bool = False,
                 period: int = 5,
                 tol: float = 8e-4,
                 div_thr: float = 0.5,
                 seed: int = None,
                 verbose: int = 1,
                 log_file: str = None,
                 checkpoint_path: str = None,
                 weights: list[dict[str, list[float]]] = None,
                 weight_beta: float = 0.25):

        self.logger = Logger(self.name, log_file=log_file, verbose=verbose)

        if not isinstance(space, SearchSpace):
            raise ValueError("space must be of type SearchSpace.")
        self.space = space

        if not isinstance(mating, Mating):
            raise ValueError("mating must be of type Mating.")
        self.mating = mating

        if not isinstance(survival, Survival):
            raise ValueError("survival must be of type Survival.")
        self.survival = survival

        if not callable(evaluate):
            raise ValueError("'evaluate' must be a callable function.")
        self.evaluate = evaluate

        if not (isinstance(pop_size, int) and pop_size > 2):
            self.logger.error("pop_size must be an integer larger than 2, changing to default value.")
            pop_size = DEFAULTS["pop_size"]
        self.pop_size = pop_size

        if not (isinstance(n_gen, int) and n_gen > 0):
            self.logger.error("n_gen must be an integer larger than 0, changing to default value.")
            n_gen = DEFAULTS["n_gen"]
        self.n_gen = n_gen

        if not isinstance(soo, bool):
            self.logger.error("soo must be an boolean, changing to default value.")
            soo = DEFAULTS["soo"]
        self.soo = soo
        self.reduce_fn = np.mean if self.soo else np.max

        if period and not (isinstance(period, int) and period >= 0):
            self.logger.error("period must be an integer larger or equal to 0, changing to default value.")
            period = DEFAULTS["period"]
        self.period = period

        if not (isinstance(tol, float) and tol >= 0):
            self.logger.error("tol must be a float larger or equal to 0, changing to default value.")
            tol = DEFAULTS["tol"]
        self.tol = tol

        if self.space.n_perms < self.pop_size:
            raise ValueError(f"search space too small: only {self.space.n_perms} valid permutations "
                             f"for pop_size={self.pop_size}. Consider using the Tensile backend for "
                             f"exhaustive search instead.")

        self._pop_size = pop_size
        self._decay_type = "none"
        max_sp_sz = max(sz for sz in self.space.sizes.values())
        if max_sp_sz > pop_size:
            self.logger.warning(f"Some variables have a larger search space than pop_size. "
                                f"Increasing pop_size for the first generations.")
            self.decay = lambda sz: int(self._pop_size + (sz - self._pop_size) / 2)
            self._decay_type = "large_space"
            self.pop_size = int(max_sp_sz * 1.15)
        elif max_sp_sz < self.pop_size / 5:
            self.pop_size //= 2
            if max_sp_sz < self.pop_size / 5:
                self.pop_size //= 2
                self.space.max_iters *= 2
            self._pop_size = self.pop_size
            self.logger.info(f"New pop_size set to {self.pop_size}")
        
        if not (isinstance(div_thr, float) and (div_thr >= 0 or div_thr <= 1)):
            self.logger.error("div_thr must be a float between 0 and 1, changing to default value.")
            div_thr = DEFAULTS["div_thr"]
        self.div_thr = div_thr

        self.probs = {}
        if weights:
            if not isinstance(weights, (list, tuple)) or not all(isinstance(el, dict) for el in weights):
                raise ValueError("weights must be an iterable of dictionaries.")
            for param_weights in weights:
                k, w = next(iter(param_weights.items()))
                if k not in self.space:
                    self.logger.warning(f"Parameter {k} not found in the search space. Sampling probability ignored.")
                    continue
                if not isinstance(weights, list) or self.space.sizes[k] != len(w):
                    raise ValueError(f"Parameter {k} weights should be an iterable with the "
                                     f"same size as the parameter space.")
                w = np.array(w, dtype=np.float32)
                w = np.exp(-weight_beta * (w - w.min()))
                self.probs[k] = w / w.sum()
                
        self.stats = {}
        self._resume_state = None
        self.checkpoint_path = checkpoint_path

        random.seed(seed)
        np.random.seed(seed)
        self.space.seed(seed)
        self.seed = seed

        self.logger.info(f"Setup completed.")
        self.logger.log_lines(self.__repr__(), logging.INFO)
    
    def update(self, best, pop, old_pop, scores):
        scores[scores < 0] = -1.0
        
        pop.G = scores
        
        if best is None:
            best = pop[scores.argmax(1)].copy()
            best.F = scores.max(1)
        elif (scores.max(1) > best).any():
            mask = scores.max(1) > best
            best[mask] = pop[scores.argmax(1)][mask].copy()
            best[mask].F = scores.max(1)[mask]
        
        pop.F = self.reduce_fn(scores / best.F[..., None], axis=0)
        if old_pop.size:
            old_pop.F = self.reduce_fn(old_pop.G / best.F, axis=1)
        
        return best, best.F.mean()

    def termination(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.stats:
                self.stats[k].append(v)
            else:
                self.stats[k] = [v]
                            
        if self.period and len(self.stats.get("f_avg", [])) > self.period:
            w = slice(-self.period - 1, -1)
            tol = self.stats["f_max"][-1] * self.tol
            ma_fa = np.mean(self.stats["f_avg"][w]) + tol
            ma_fm = np.mean(self.stats["f_max"][w]) + tol
            if ma_fa >= self.stats["f_avg"][-1] and ma_fm >= self.stats["f_max"][-1]: 
                raise StopIteration(f"f_avg and f_max did not increase for the last {self.period} generations.")

        if kwargs.get('diversity', 1.0) < self.div_thr:
            self.decay = lambda sz: int(self._pop_size / 2 + (sz - self._pop_size / 2) / 1.25)
            self._decay_type = "low_diversity"
        self.pop_size = self.decay(self.pop_size) if hasattr(self, "decay") else self.pop_size

    def save(self, path: str, gen: int, best: Population, n_evals: int, old_pop: Population, pop: Population):
        state = {
            "gen": gen,
            "soo": self.soo,
            "space_map": self.space.map,
            "stats": self.stats,
            "best": best.tolist() if best is not None else None,
            "n_evals": n_evals,
            "old_pop": old_pop.tolist(),
            "pop": pop.tolist(),
            "pop_size": self.pop_size,
            "_pop_size": self._pop_size,
            "decay_type": self._decay_type,
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "wb") as fp:
            pickle.dump(state, fp)

    def load(self, checkpoint):
        if isinstance(checkpoint, (str, bytes, os.PathLike, Path)):
            with open(checkpoint, "rb") as fp:
                checkpoint = pickle.load(fp)

        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a dict or a path to a pickled checkpoint")

        required = {
            "gen", "soo", "space_map", "stats", "best", "n_evals",
            "old_pop", "pop", "pop_size", "_pop_size", "decay_type",
            "random_state", "np_random_state"
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"checkpoint missing required fields: {sorted(missing)}")

        if checkpoint["soo"] != self.soo:
            raise ValueError(f"soo mismatch. checkpoint={checkpoint['soo']} current={self.soo}")

        if checkpoint["space_map"] != self.space.map:
            raise ValueError("space.map mismatch between checkpoint and current search space")

        decay_type = checkpoint.get("decay_type", "none")
        if decay_type == "large_space":
            self.decay = lambda sz: int(self._pop_size + (sz - self._pop_size) / 2)
        elif decay_type == "low_diversity":
            self.decay = lambda sz: int(self._pop_size / 2 + (sz - self._pop_size / 2) / 1.25)
        elif decay_type == "none":
            if hasattr(self, "decay"):
                delattr(self, "decay")
        else:
            raise ValueError(f"unknown decay_type '{decay_type}' in checkpoint")

        self._decay_type = decay_type

        self.stats = checkpoint["stats"]
        self.pop_size = checkpoint["pop_size"]

        self._pop_size = checkpoint["_pop_size"]
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["np_random_state"])

        self._resume_state = checkpoint

        return self

    def optimize(self):
        start = time.time()
        if self._resume_state is not None:
            state = self._resume_state
            gen_start = int(state["gen"]) + 1
            best = Population(state["best"]) if state["best"] is not None else None
            n_evals = int(state["n_evals"])
            old_pop = Population(state["old_pop"])
            pop = Population(state["pop"])
            self._resume_state = None
            self.logger.info(f"Resuming optimization from generation {gen_start} with best F={best.F.mean():.4f}...")
        else:
            self.stats = {}
            best = None
            n_evals = 0
            old_pop = Population()
            gen_start = 1

            self.logger.info("Sampling initial population...")
            try:
                pop = self.space.sample(self.pop_size, p=self.probs)
            except MaxIterationsReached as e:
                self.logger.warning("Max iterations reached. Attempting with half the population size...")

                n_sampled = e.args[1]
                p = self.probs if n_sampled > 0 else None
                pop = self.space.sample(self.pop_size // 2, p=p, iter_mul=4, reuse=True)

        self.logger.info(f"Starting optimization...")
        try:
            for gen in range(gen_start, self.n_gen + 1):
                scores = self.evaluate(self.space.transform(pop))
                if scores.ndim == 1:
                    scores = scores[None, ...]

                best, f_max = self.update(best, pop, old_pop, scores)

                n_valid = (scores > 0).max(0).sum()
                n_evals += n_valid
                f_avg = scores[scores > 0].mean() if n_valid else -1 # Here we ignore non-valid solutions (F<=0)
                diversity = pop.diversity()
                
                self.logger.print_stats(n_gen=gen, n_evals=n_evals, diversity=diversity, f_avg=f_avg, f_max=f_max)
                
                self.termination(f_avg=f_avg, f_max=f_max, diversity=diversity)

                old_pop = self.survival(old_pop, pop, self.pop_size)
                pop = self.mating(old_pop, self.pop_size)

                if self.checkpoint_path:
                    self.save(self.checkpoint_path, gen=gen, best=best, n_evals=n_evals, old_pop=old_pop, pop=pop)
        except MatingExhaustedError as e:
            self.logger.warning(
                f"Mating stopped early: {e}. Ending optimization with current best solution."
            )
        except StopIteration as e:
            self.logger.info(f"Stop criterion reached: {e}")

        if self.soo and len(best) > 1:
            best = best[(best.G / best.F).mean(1).argmax()]
            best.F = best.G.mean()
            best = Population([best])
        
        X = self.space.transform(best)
        self.logger.info(f"Finished optimization in {(time.time() - start):.3f}s")
        self.logger.info(f"X: {X}")
        self.logger.info(f"F: {best.F.mean():.4f}")
        return X, best.F

    def __repr__(self):
        return f"GeneticAlgorithm(pop_size={self.pop_size}, n_gen={self.n_gen}, period={self.period}, " \
               f"tol={self.tol}, div_thr={self.div_thr}, soo={self.soo})\n{self.mating}\n{self.space}"
