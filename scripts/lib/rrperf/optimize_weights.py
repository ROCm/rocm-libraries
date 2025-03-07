#!/usr/bin/env python3

################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################


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


def terminate_pool():
    global mp_pool

    if mp_pool is not None:
        mp_pool.terminate()
        mp_pool.join()


def close_pool():
    global mp_pool

    if mp_pool is not None:
        mp_pool.close()
        mp_pool.join()


def random_int(max=40):
    def factory():
        return int(random.uniform(0, max))

    factory.is_variable = max > 0
    return factory


def random_inv_exp(mean=100.0):
    def factory():
        return 1.0 / random.expovariate(mean)

    factory.is_variable = True
    return factory


def random_bool():
    def factory():
        return random.choice([False, True])

    factory.is_variable = True
    return factory


def fixed_value(value):
    def factory():
        return value

    factory.is_variable = False
    return factory


@dataclass(frozen=True, order=True, unsafe_hash=True)
class Weights:
    nops: float = field(default_factory=random_inv_exp())

    vmcnt: float = field(default_factory=random_inv_exp())
    lgkmcnt: float = field(default_factory=random_inv_exp())

    vmQueueLen: int = field(
        default_factory=random_int(), metadata={"isCoefficient": False}
    )
    vectorQueueSat: float = field(default_factory=random_inv_exp())
    ldsQueueSat: float = field(default_factory=random_inv_exp())
    lgkmQueueLen: int = field(
        default_factory=random_int(), metadata={"isCoefficient": False}
    )

    # Fix the cost of a stall cycle to provide a common reference point
    # so that different randomly generated weights are of comparable magnitudes.
    stallCycles: float = field(default_factory=fixed_value(1000.0))

    notMFMA: float = field(default_factory=random_inv_exp())
    isMFMA: float = field(default_factory=random_inv_exp())

    isSMEM: float = field(default_factory=random_inv_exp())
    isSControl: float = field(default_factory=random_inv_exp())
    isSALU: float = field(default_factory=random_inv_exp())

    isVMEMRead: float = field(default_factory=random_inv_exp())
    isVMEMWrite: float = field(default_factory=random_inv_exp())
    isLDSRead: float = field(default_factory=random_inv_exp())
    isLDSWrite: float = field(default_factory=random_inv_exp())
    isVALU: float = field(default_factory=random_inv_exp())

    isACCVGPRWrite: float = field(default_factory=random_inv_exp())
    isACCVGPRRead: float = field(default_factory=random_inv_exp())

    newSGPRs: float = field(default_factory=random_inv_exp())
    newVGPRs: float = field(default_factory=random_inv_exp())
    highWaterMarkSGPRs: float = field(default_factory=random_inv_exp())
    highWaterMarkVGPRs: float = field(default_factory=random_inv_exp())
    fractionOfSGPRs: float = field(default_factory=random_inv_exp())
    fractionOfVGPRs: float = field(default_factory=random_inv_exp())

    # It doesn't make a lot of sense to allow the optimizer to choose
    # whether to run out of registers should the opportunity arise.
    # Therefore, fix this parameter at a high value.
    outOfRegisters: float = field(default_factory=fixed_value(1e9))

    zeroFreeBarriers: bool = field(
        default_factory=random_bool(), metadata={"isCoefficient": False}
    )

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


@dataclass(order=True, unsafe_hash=True)
class Result:
    # The time field must be first so that results are sorted by speed.
    time: float = field(default=math.inf)

    weights: Weights = field(default_factory=Weights)

    command: str = field(default="", compare=False)
    output: str = field(default="", compare=False)

    output_file: str = field(default="", compare=False)

    rnorm: float = field(default=math.inf)

    @property
    def passed(self):
        return math.isfinite(self.time)

    @property
    def summary(self):
        lines = [
            f"Weights: {self.weights.short_hash}",
            self.command,
            self.output,
        ]
        if self.passed:
            lines += [f"Median time: {self.time:,} ns"]
        return "\n".join(lines)

    @property
    def short_summary(self):
        return f"Weights: {self.weights.short_hash}, Median time: {self.time:,} ns"

    @property
    def dict(self):
        rv = asdict(self)
        rv["hash"] = self.weights.short_hash
        return rv

    @classmethod
    def from_dict(cls, d):
        args = {k: v for k, v in d.items() if k != "hash"}
        if "weights" in args:
            args["weights"] = Weights(**args["weights"])
        return cls(**args)


def bench_star(arg):
    return bench(*arg)


def bench(
    thedir: pathlib.Path, problem: rrperf.problems.GEMMRun, weights: Weights
) -> Result:
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
        env["OMP_NUM_THREADS"] = str(1)

        cmd = problem.command(device=device, yaml=result_path.absolute())

        result = Result(weights=weights)

        env_prefix = " ".join([f"{k}='{v}'" for k, v in env.items()])
        result.command = env_prefix + " " + " ".join(cmd)

        print(f"Launching {weights.short_hash}")

        process_result = subprocess.run(
            cmd,
            env=env,
            cwd=build_dir,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        )

        result.output = process_result.stdout.decode()

        if process_result.returncode == 0:
            with result_path.open() as f:
                result_data = yaml.safe_load(f)
            if result_data["correct"]:
                result.time = float(np.median(result_data["kernelExecute"]))
            result.rnorm = result_data["rnorm"]

        print(result.summary)

        output_file = f"output_{weights.short_hash}.yaml"
        output_path = thedir / output_file

        with output_path.open("w") as ofile:
            yaml.dump(result.dict, ofile)

        return result

    finally:
        lock.release()


def sanity_check(results: List[Result]):
    rnorms = {r.rnorm for r in results if r.passed}
    print(f"RNorms: {rnorms}")
    if len(rnorms) != 1:
        print("Differing rnorms!!!")
        raise ValueError


prev_results = {}


def split_old_new_results(weights) -> Tuple[List[Weights], List[Weights]]:
    global prev_results

    already_ran = []
    to_run = []
    for w in weights:
        if w in prev_results:
            already_ran.append(w)
        else:
            to_run.append(w)

    return already_ran, to_run


def generation(
    output_dir: pathlib.Path, problem: rrperf.problems.GEMMRun, weights: List[Weights]
) -> List[Result]:
    global prev_results

    already_ran, to_run = split_old_new_results(weights)

    old_results = list([prev_results[w] for w in already_ran])

    to_run_msg = ", ".join([w.short_hash for w in to_run])
    print(f"Running {to_run_msg}.")

    if len(old_results) > 0:
        old_results_msg = ", ".join(w.weights.short_hash for w in old_results)
        print(f"Using previous results for {old_results_msg}")

    to_run_args = zip(itertools.repeat(output_dir), itertools.repeat(problem), to_run)

    async_results = pool().imap_unordered(bench_star, to_run_args)
    new_results = []
    for r in async_results:
        new_results.append(r)
        prev_results[r.weights] = r

    return sorted(old_results + new_results)


def read_gen_results(resfile: str):
    resfile = pathlib.Path(resfile)
    with resfile.open() as f:
        data = yaml.safe_load(f)

        def res(el):
            return Result.from_dict(el)

        return list(map(res, data))


def write_generation(thedir: pathlib.Path, name, results: List[Result]):
    data = list([val.dict for val in results])
    datafile = thedir / f"results_{name}.yaml"
    with datafile.open("w") as f:
        yaml.dump(data, f)
    print(f"Wrote {datafile.absolute()}")


def new_inputs(
    all_results: List[Result], population, num_parents, num_random, mutation
):
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
        if not all_results[i].passed:
            break
        parents.add(all_results[i].weights)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
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
            sanity_check(results)

            write_generation(args.output_dir, i, results)

            args.all_results = sorted(set(args.all_results + results))
            write_generation(args.output_dir, f"{i}_all", args.all_results)
            block = "#" * 5
            print(f"{block} Generation {i} {block}")
            print_top_results(args)

            args.mutation *= args.mutation_decay

    except KeyboardInterrupt:
        pool().terminate()

    finally:
        print_bad_results(args)
        print_top_results(args)
        close_pool()


def find_most_different_outputs(results: List[Result], n: int = 5):
    n = min(n, len(results))
    if n <= 0:
        return

    import difflib

    yielded = set()

    for i in range(n):
        bestIdx = 0
        bestRatio = 1
        for j in range(len(results)):
            if j in yielded:
                continue

            for k in yielded:
                ratio = difflib.SequenceMatcher(
                    None, results[j].output, results[k].output
                ).ratio()
                if ratio < bestRatio:
                    bestIdx = j
                    bestRatio = ratio

        yield results[bestIdx]
        yielded.add(bestIdx)


def print_bad_results(args):
    bad_results = []
    for r in reversed(args.all_results):
        print(r.time)
        if r.passed:
            break
        bad_results.append(r)

    print(f"{len(bad_results)} errors.")
    write_generation(args.output_dir, "errors", bad_results)

    for r in find_most_different_outputs(bad_results):
        print(r.summary)


def print_top_results(args):
    good_results = []
    for r in args.all_results:
        if not r.passed:
            break
        good_results.append(r)

    print(f"{len(good_results)} good weights.")
    write_generation(args.output_dir, "good", good_results)

    for i in range(min(10, len(good_results))):
        print(good_results[i].short_summary)


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
        close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    get_args(parser)

    args = parser.parse_args()
    run(args)
