#!/usr/bin/env python3

# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Union, List
import argparse
import glob
import itertools
import json
import logging
import multiprocessing as mp
import os
import pathos.multiprocessing as pamp
import rich.logging
import rich.progress
import scipy.optimize
import shutil
import subprocess

parameter_spaces = {
    "device_segmented_radix_sort_keys": {
        "benchmark": "benchmark_device_segmented_radix_sort_keys",
        "types": {
            "KeyType": [
                "int64_t",
                "int",
                "short",
                "int8_t",
                "double",
                "float",
                "rocprim::half",
            ],
        },
        "params": {
            "RadixBits": [6, 7, 8],
            "BlockSize": [256],
            "ItemsPerThread": [7, 8, 13, 16, 17],
            "WarpSmallLWS": [8, 16, 32, 64],
            "WarpSmallIPT": [2, 4, 8],
            "WarpSmallBS": [256],
            "WarpPartition": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            "WarpMediumLWS": [16, 32, 64],
            "WarpMediumIPT": [4, 8, 16],
            "WarpMediumBS": [256],
        },
    },
    "device_segmented_radix_sort_pairs": {
        "benchmark": "benchmark_device_segmented_radix_sort_pairs",
        "types": {
            "KeyType": [
                "int64_t",
                "int",
                "short",
                "int8_t",
                "double",
                "float",
                "rocprim::half",
            ],
            "ValueType": [
                "int64_t",
                "int",
                "short",
                "int8_t",
            ],
        },
        "params": {
            "RadixBits": [6, 7, 8],
            "BlockSize": [256],
            "ItemsPerThread": [7, 8, 13, 16, 17],
            "WarpSmallLWS": [8, 16, 32, 64],
            "WarpSmallIPT": [2, 4, 8],
            "WarpSmallBS": [256],
            "WarpPartition": [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            "WarpMediumLWS": [16, 32, 64],
            "WarpMediumIPT": [4, 8, 16],
            "WarpMediumBS": [256],
        },
    },
}

def get_result_from_json(filename: os.PathLike) -> Union[float, int]:
    '''
    Get the result from the benchmark json.
    '''
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            return float(data['benchmarks'][0]['bytes_per_second'])
        except Exception as e:
            log.error('Could not extract \'bytes_per_second\' from JSON!')
            raise e

def merge_jsons(source_filenames: List[os.PathLike], target_filename: os.PathLike) -> None:
    '''
    Merges benchmark JSONs. This is used to collect the singular results generated from the various
    benchmark runs.
    '''
    merged = {
        'context': {},
        'benchmarks': [],
    }

    # collect jsons
    for filename in source_filenames:
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
            except json.decoder.JSONDecodeError as e:
                log.warning(f'Skipping file \'{filename}\' because of error: {e}')

            # HACK: we reuse the last context since we can only have one
            merged['context'] = data['context']
            # append benchmark data
            merged['benchmarks'].extend(data['benchmarks'])

    # write out file
    with open(target_filename, 'w') as file:
        json.dump(merged, file, indent=2)

def combine(alg_name: str, arch: str):
    '''
    combine()
    '''
    script_dir = os.path.dirname(os.path.realpath(__file__))
    result_dir = os.path.join(script_dir, 'artifacts')

    alg_space = parameter_spaces[alg_name]
    build_target = alg_space['benchmark']

    merge_jsons(
        glob.glob(os.path.join(result_dir, f'{arch}_{build_target}_*.json')),
        os.path.join(result_dir, f'{arch}_{build_target}.json'),
    )

def tune_alg(alg_name: str, arch: str, max_samples: int, num_workers: int, size: int, trials: int) -> None:
    '''
    The core tuning procedure. This tunes a single algorithm for multiple types.
    '''

    # get the context of the tuning run
    alg_space = parameter_spaces[alg_name]
    build_target = alg_space['benchmark']

    # types to tune, this can be a product of multiple types
    types = [
        dict(zip(alg_space['types'], ts))
        for ts in itertools.product(
            *[alg_space['types'][type] for type in alg_space['types']]
        )
    ]

    # generate bounds by normalizing the parameter space from discrete to real numbers (relaxation)
    bounds = dict(zip(alg_space['params'], ((0, 1) for _ in alg_space['params'])))

    # define a utility function to access parameters in 'alg_space'
    def param_from_normalized(name: str, value: float) -> str:
        '''
        Internal which maps a continious named parameter in [0; 1] to it's discrete value.
        '''

        # get the list of discrete values
        params = alg_space['params'][name]

        # get the index, make sure we're not out-of-bounds when value is 1.0
        index = min(int(value * len(params)), len(params) - 1)

        try:
            return str(params[index])
        except IndexError as e:
            log.error(
                f"Could not find parameter '{name}' at '{index}' derived from value '{value}' in {params}."
            )
            raise e

    def tune_type(type: str) -> None:
        cache = {}

        def sample(xs: List[float]) -> Union[float, int]:
            # each worker should get their own build dir
            build_dir = os.path.join(source_dir, f'build/tune-{worker_id}')

            # delete *.parallel folder
            try:
                # HACK: we just delete the benchmark folder because it's easier
                shutil.rmtree(os.path.join(build_dir, 'benchmark'))
            except FileNotFoundError:
                # if the tree doesn't exist we don't have to remove it :)
                pass

            tune_param_names = list(type.keys()) + list(alg_space['params'])
            tune_param_vals = list(type.values()) + [
                param_from_normalized(name, value)
                for (name, value) in zip(alg_space['params'].keys(), xs)
            ]

            result_id = '_'.join(tune_param_vals)
            if result_id in cache:
                log.info(f'[{worker_id}] Skipped already computed result!')
                return cache[result_id]

            result_filename = f'{arch}_{build_target}_{result_id}.json'

            tune_param_names = ';'.join(tune_param_names)
            tune_param_vals = ';'.join(tune_param_vals)

            # CMake configure
            log.info(f'[{worker_id}] Configuring: {result_id}')
            configure = subprocess.call(
                [
                    'cmake',
                    '-S',
                    '.',
                    '-B',
                    build_dir,
                    '-GNinja',
                    '-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++',
                    '-DBUILD_BENCHMARK=ON',
                    '-DBENCHMARK_CONFIG_TUNING=ON',
                    f'-DAMDGPU_TARGETS={arch}',
                    f'-DBENCHMARK_TUNE_PARAM_NAMES={tune_param_names}',
                    f'-DBENCHMARK_TUNE_PARAMS={tune_param_vals}',
                ],
                cwd=source_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if configure != 0:
                cache[result_id] = 0.0
                return 0.0

            # Build target
            log.info(f'[{worker_id}] Building: {result_id}')
            build = subprocess.call(
                ['cmake', '--build', '.', '--target', build_target],
                cwd=build_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if build != 0:
                cache[result_id] = 0.0
                result_context = {
                    'config': dict(
                        (
                            (name, param_from_normalized(name, val))
                            for name, val in zip(alg_space['params'], xs)
                        )
                    )
                }
                log.debug(json.dumps(result_context, indent=2))
                return 0.0

            # Run benchmark
            gpu_lock.acquire()
            try:
                log.info(f'[{worker_id}] Benchmarking: {result_id}')
                bench = subprocess.call(
                    [
                        os.path.join(build_dir, 'benchmark', build_target),
                        '--seed',
                        'random',  # Random is better... I think? Otherwise we might overfit.
                        '--size',
                        f'{size}',
                        '--trials',
                        f'{trials}',
                        f'--benchmark_out={result_filename}',
                    ],
                    cwd=result_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                )

                if bench != 0:
                    cache[result_id] = 0.0
                    return 0.0
                result_value = get_result_from_json(
                    os.path.join(result_dir, result_filename)
                )
                log.info(f'[{worker_id}] Completed: {result_id} @ {result_value / 1e9:.3f} GB/s')
            finally:
                gpu_lock.release()

            result_context = {
                'config': dict(
                    (
                        (name, param_from_normalized(name, val))
                        for name, val in zip(alg_space['params'], xs)
                    )
                ),
                'bytes_per_second': result_value,
            }
            log.debug(json.dumps(result_context, indent=2))

            cache[result_id] = -result_value

            # scipy.optimize does minimization, negate result for maximize
            return -result_value

        # Dual annealing is very good for tuning. See:
        # - 'Benchmarking optimization algorithms for auto-tuning GPU kernels' by Schoonhoven et al, 2022.
        # - 'A methodology for comparing optimization algorithms for auto-tuning' by Willemsen et al, 2024.
        scipy.optimize.dual_annealing(
            sample, bounds=bounds.values(), maxfun=max_samples
        )

    script_dir = os.path.dirname(os.path.realpath(__file__))
    source_dir = os.path.join(script_dir, '../..')
    result_dir = os.path.join(script_dir, 'artifacts')

    os.makedirs(result_dir, exist_ok=True)

    def pool_init(worker_ids, lock):
        global worker_id, gpu_lock
        gpu_lock = lock
        worker_id = worker_ids.get(False)

    man = mp.Manager()

    # create queue to distrubute worker ids
    worker_ids = man.Queue(num_workers)
    for i in range(num_workers):
        worker_ids.put(i)

    # 'pathos' is needed to pickle local scopes to workers
    with pamp.Pool(
        processes=num_workers,
        initializer=pool_init,
        initargs=[worker_ids, mp.Lock()],
    ) as pool:
        pool.map(tune_type, types)

    # We're done with tuning this entire algorithm, collect them into a single file!
    merge_jsons(
        glob.glob(os.path.join(result_dir, f'{arch}_{build_target}_*.json')),
        os.path.join(result_dir, f'{arch}_{build_target}.json'),
    )

parser = argparse.ArgumentParser(
    prog='autotune-search',
    description='config tuning using local search',
)

parser.add_argument('targets',metavar='TARGETS', nargs='*', help='target(s) to optimize, seperated by comma')
parser.add_argument('-a', '--arch', default='gfx942', help='architecture to target, e.g. gfx908')
parser.add_argument('-n', '--evals', default=200, help='maximum number of configs being evaluated per type per target')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
parser.add_argument('-w', '--workers', default=8, help='number of workers')
parser.add_argument('-s', '--size', default=33554432, help='input size to use for tuning')
parser.add_argument('-t', '--trials', default=3, help='number of trials per config to test')
parser.add_argument('-c', '--combine', action='store_true', help='skip tuning and combine the results of a previous run for the given targets and architecture')
parser.add_argument('-l', '--list', action='store_true', help='list available targets')

args = parser.parse_args()

if not args.targets:
    args.targets = list(parameter_spaces.keys())

if args.list:
    for target in parameter_spaces.keys():
        print(target)
    quit()

log_level = logging.INFO
if args.verbose:
    log_level = logging.DEBUG

logging.basicConfig(format='%(message)s', handlers=[rich.logging.RichHandler(rich_tracebacks=True, markup=True)], level=log_level)
log = logging.getLogger('rich')

if args.combine:
    for target in args.targets:
        combine(alg_name=target, arch=args.arch)
    quit()

for target in args.targets:
    log.info(f'Tuning {target} for {args.arch} with {int(args.evals)} max evaluations')
    tune_alg(
        alg_name=target,
        arch=args.arch,
        max_samples=int(args.evals),
        num_workers=int(args.workers),
        size=int(args.size),
        trials=int(args.trials)
    )
