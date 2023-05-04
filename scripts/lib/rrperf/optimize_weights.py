#!/usr/bin/env python3

"""
Use a genetic algorithm to optimize scheduler weights for rocRoller.

The output is as follows:

<output dir>:
    results_<i>.yaml:
        Results for every set of weights considered in generation i, sorted by time.
        Created after generation i is complete.
    results_<i>_all.yaml:
        Results for every set of weights for all generations <=i with valid results.
        Created after generation i is complete.

<output dir>/gen_<i>: Data files for generation i. These are created as we go.
    weights_<hash>.yaml: One set of weights
    result_<hash>.yaml: Result file from GEMM client, including performance.
"""

import argparse
import hashlib
import itertools
import math
import multiprocessing
import numpy as np
import pathlib
import random
import subprocess

# import tempfile
import yaml

import rrperf.run
import rrperf.problems

from dataclasses import dataclass, field, fields, asdict
from typing import List, Tuple

gpus = {}
mp_pool = None


def instantiate_gpus(idxs: List[int]):
    global gpus, mp_pool
    for idx in idxs:
        gpus[idx] = multiprocessing.Lock()
    if mp_pool is not None:
        mp_pool.close()
        mp_pool = None


def acquire_lock() -> Tuple[int, multiprocessing.Lock]:
    global gpus
    for i in range(100):
        for id, lock in gpus.items():
            locked = lock.acquire(False)
            if locked:
                return id, lock

    raise RuntimeError("Could not get a GPU!")


def pool() -> multiprocessing.Pool:
    global mp_pool, gpus
    if mp_pool is None:
        mp_pool = multiprocessing.Pool(len(gpus))
    return mp_pool


def random_int(max=40):
    return lambda: int(random.uniform(0, max))


def random_inv_exp(mean=100.0):
    return lambda: 1.0 / random.expovariate(mean)


def random_bool():
    return lambda: random.choice([False, True])


def fixed_value(value):
    return lambda: value


@dataclass(frozen=True, order=True, unsafe_hash=True)
class Weights:
    nops: float = field(default_factory=random_inv_exp())
    vmcnt: float = field(default_factory=random_inv_exp())
    lgkmcnt: float = field(default_factory=random_inv_exp())
    newSGPRs: float = field(default_factory=random_inv_exp())
    newVGPRs: float = field(default_factory=random_inv_exp())
    highWaterMarkSGPRs: float = field(default_factory=random_inv_exp())
    highWaterMarkVGPRs: float = field(default_factory=random_inv_exp())
    notMFMA: float = field(default_factory=random_inv_exp())
    isMFMA: float = field(default_factory=random_inv_exp())
    fractionOfSGPRs: float = field(default_factory=random_inv_exp())
    fractionOfVGPRs: float = field(default_factory=random_inv_exp())

    vmQueueLen: int = field(default_factory=random_int())
    vectorQueueSat: float = field(default_factory=random_inv_exp())
    lgkmQueueLen: int = field(default_factory=random_int())
    ldsQueueSat: float = field(default_factory=random_inv_exp())

    zeroFreeBarriers: bool = field(default_factory=random_bool())

    # It doesn't make a lot of sense to allow the optimizer to choose
    # whether to run out of registers should the opportunity arise.
    # Therefore, fix this parameter at a high value.
    outOfRegisters: float = field(default_factory=fixed_value(1e9))

    @classmethod
    def Combine(cls, inputs: list, mutation: float = 0.1):
        vals = {}
        for fld in fields(cls):
            if random.uniform(0, 1) < mutation:
                vals[fld.name] = fld.default_factory()
            else:
                vals[fld.name] = getattr(random.choice(inputs), fld.name)
        return cls(**vals)

    @property
    def short_hash(self):
        d = hashlib.shake_128()

        the_hash = hash(self)
        num_bits = the_hash.bit_length()
        num_bytes = (num_bits + 7) // 8
        the_bytes = the_hash.to_bytes(num_bytes + 1, byteorder="big", signed=True)

        d.update(the_bytes)

        return d.hexdigest(4)


def bench(thedir: pathlib.Path, problem: rrperf.problems.GEMMRun, weights: Weights):
    device, lock = acquire_lock()

    try:
        thedir.mkdir(parents=True, exist_ok=True)
        weights_file = f"weights_{weights.short_hash}.yaml"
        weights_path = thedir / weights_file
        with weights_path.open("w") as wfile:
            yaml.dump(asdict(weights), wfile)

        result_file = f"result_{weights.short_hash}.yaml"
        result_path = thedir / result_file

        build_dir = rrperf.run.get_build_dir()
        env = rrperf.run.get_arch_env(build_dir)
        env["ROCROLLER_SCHEDULER_WEIGHTS"] = str(weights_path.absolute())

        cmd = problem.command(device=device, yaml=result_path.absolute())
        print(env, "\n", " ".join(cmd))

        result = subprocess.call(cmd, env=env, cwd=build_dir)
        if result != 0:
            return math.inf, weights

        with result_path.open() as f:
            result = yaml.safe_load(f)
        if not result["correct"]:
            return math.inf, weights

        return float(np.median(result["kernelExecute"])), weights
    finally:
        lock.release()


prev_results = {}


def generation(
    output_dir: pathlib.Path, problem: rrperf.problems.GEMMRun, weights: List[Weights]
) -> List[tuple]:
    global prev_results

    old_results = list([(prev_results[w], w) for w in weights if w in prev_results])

    to_run = list([w for w in weights if w not in prev_results])

    to_run_msg = " ".join([w.short_hash for w in to_run])
    old_results_msg = " ".join(w[1].short_hash for w in old_results)
    print(f"Running {to_run_msg}.\nUsing previous results for {old_results_msg}")

    to_run_args = zip(itertools.repeat(output_dir), itertools.repeat(problem), to_run)
    new_results = pool().starmap(bench, to_run_args)
    for t, w in new_results:
        prev_results[w] = t

    return sorted(old_results + new_results)


def read_gen_results(resfile: str):
    resfile = pathlib.Path(resfile)
    with resfile.open() as f:
        data = yaml.safe_load(f)

        def res(el):
            return (el["time"], Weights(**el["weights"]))

        return list(map(res, data))


def write_generation(thedir: pathlib.Path, iter: int, results: List[tuple]):
    def tup_dict(val):
        return {"time": val[0], "hash": val[1].short_hash, "weights": asdict(val[1])}

    data = list([tup_dict(val) for val in results])
    datafile = thedir / f"results_{iter}.yaml"
    with datafile.open("w") as f:
        yaml.dump(data, f)
    print(f"Wrote {datafile.absolute()}")


def new_inputs(all_results, population, num_parents, num_random, mutation):
    if len(all_results) == 0:
        rv = set()
        while len(rv) < population:
            rv.add(Weights())
        return list(rv)

    num_children = population - num_random
    rv_len = population + num_parents

    parents = set()

    # get parents from beginning of all_results.
    i = 0
    while i < len(all_results) and len(parents) < num_parents:
        parents.add(all_results[i][1])
        i += 1
    # use new random values if there aren't enough.
    while len(parents) < num_parents:
        parents.add(Weights())
    parents = list(parents)

    # create children: Half from sets of 2 parents
    rv = set(parents)
    while len(rv) < len(parents) + (num_children // 2):
        these_parents = random.choices(parents, k=2)
        new_child = Weights.Combine(these_parents, mutation)
        rv.add(new_child)

    # create children: Half from all parents together.
    while len(rv) < len(parents) + num_children:
        new_child = Weights.Combine(parents, mutation)
        rv.add(new_child)

    # Add num_random new random weights.
    while len(rv) < rv_len:
        new_weights = Weights()
        rv.add(new_weights)

    return list(rv)


def genetic(args):
    num_children = args.population - args.num_random
    assert num_children > 0

    for i in range(args.generations):
        inputs = new_inputs(
            args.all_results,
            population=args.population,
            num_parents=args.num_parents,
            num_random=args.num_random,
            mutation=args.mutation,
        )

        gen_dir = args.output_dir / f"gen_{i}"
        results = generation(gen_dir, args.problem, inputs)

        write_generation(args.output_dir, i, results)

        ok_results = list([r for r in results if math.isfinite(r[0])])

        args.all_results = sorted(set(args.all_results + ok_results))
        write_generation(args.output_dir, f"{i}_all", args.all_results)
        args.mutation *= args.mutation_decay


def get_args(parser: argparse.ArgumentParser):
    def split_ints(x):
        if len(x) == 0:
            return []
        return list(map(int, x.split(",")))

    parser.add_argument(
        "--gpus",
        type=split_ints,
        default=[0],
        help="Comma-separated list of device ordinals of GPUs to use.",
    )

    parser.add_argument(
        "--output",
        "-o",
        dest="output_dir",
        type=pathlib.Path,
        required=True,
        help="Directory to store results.",
    )

    parser.add_argument(
        "--input",
        default=[],
        type=read_gen_results,
        dest="all_results",
        help="Start from results_*.yaml from previous run",
    )

    parser.add_argument(
        "--generations", default=100, type=int, help="Number of generations to run."
    )

    parser.add_argument(
        "--parents",
        dest="num_parents",
        default=10,
        type=int,
        help="Number of winners to use as parents every generation.",
    )

    parser.add_argument(
        "--population",
        default=32,
        type=int,
        help="Number of new configurations to try every generation "
        "(including children and random configuration).",
    )

    parser.add_argument(
        "--new",
        dest="num_random",
        default=4,
        type=int,
        help="Number of new random configurations to try every " "generation.",
    )

    parser.add_argument(
        "--mutation",
        default=0.1,
        type=float,
        help="Chance that a parameter will be newly generated instead "
        "of picking from a parent.",
    )

    parser.add_argument(
        "--mutation-decay",
        default=0.98,
        type=float,
        help="Mutation rate is multiplied by this number every" " generation.",
    )

    parser.add_argument(
        "--suite",
        dest="problem",
        type=rrperf.run.first_problem_from_suite,
        help="Benchmark suite to run. NOTE: Only the first problem from the "
        "suite will be used.",
    )


def run(args):
    """
    Use a genetic algorithm to optimize scheduler weights for rocRoller.
    """
    try:
        instantiate_gpus(args.gpus)
        genetic(args)

    finally:
        global mp_pool
        if mp_pool is not None:
            mp_pool.close()
            mp_pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_args(parser)

    args = parser.parse_args()
    run(args)
