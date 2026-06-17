# Benchmarking AMD Primitives

This document contains the commands needed to benchmark all AMD Primitives libraries and their NVIDIA counterparts.

The commands are formatted in code blocks. Pasting them into a fresh HIP or CUDA container will execute every step required to begin benchmarking. This guide also includes minimal VS Code dev containers for both HIP and CUDA.

The intended way to view this markdown file is to open it in VS Code, and to press the `Open Preview to the Side` button in the top-right corner. This lets VS Code collapse the code blocks, as they are wrapped in `<details>` HTML tags.

Please note that AMD Primitives their benchmarking suites are currently in a transition phase: some projects have migrated to using [primbench](https://github.com/ROCm/rocm-libraries/tree/develop/shared/primbench), while others still use Google Benchmark.

## Tips

To keep the benchmarks running on a server after disconnecting, use [screen](https://en.wikipedia.org/wiki/GNU_Screen) or [nohup](https://en.wikipedia.org/wiki/Nohup).

To copy directory outputs from a remote server to your local machine, use [scp](https://en.wikipedia.org/wiki/Secure_copy_protocol) with the `-r` flag for recursive copying. For example, the following command copies the `results` directory from the remote server into your local `Downloads` directory:

```sh
scp -r myname@servername:~/rocm-libraries/projects/hipcub/results ~/Downloads
```

## How AMD Primitives Libraries Relate to Their NVIDIA Counterparts

`roc*` libraries are AMD-native implementations:
* `rocPRIM` -> AMD-native equivalent of `CUB`
* `rocRAND` -> AMD-native equivalent of `cuRAND`
* `rocThrust` -> AMD-native equivalent of `Thrust`

`hip*` libraries are cross-platform HIP wrappers that dispatch to either the `roc*` backend (on AMD) or the original NVIDIA library (on CUDA):
* `hipCUB` -> `rocPRIM` (AMD backend) + `CUB` (NVIDIA backend)
* `hipRAND` -> `rocRAND` (AMD backend) + `cuRAND` (NVIDIA backend)

### Note

- `hipThrust` does not exist
- `hipRAND` contains no benchmarks
- `hipCUB` contains slightly different benchmarks than `rocPRIM`

## VS Code Dev Containers

<details>
<summary>HIP <code>Dockerfile</code> + <code>devcontainer.json</code></summary>

`Dockerfile`:
```Dockerfile
FROM rocm/rocm-terminal:latest
```

`devcontainer.json`:
```json
{
    "build": {
        "dockerfile": "Dockerfile"
    },
    "name": "hip-minimal",
    "runArgs": [
        "--device=/dev/kfd",
        "--device=/dev/dri"
    ]
}
```

</details>

<details>
<summary>CUDA <code>Dockerfile</code> + <code>devcontainer.json</code></summary>

`Dockerfile`:
```Dockerfile
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

RUN apt update && apt install -y git cmake ninja-build wget

RUN wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg

RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2 noble main" > /etc/apt/sources.list.d/rocm.list

RUN tee /etc/apt/preferences.d/rocm-pin-600 <<'EOF'
Package: *
Pin: origin repo.radeon.com
Pin-Priority: 600
EOF

RUN apt update && apt install -y hip-base

ENV PATH=/opt/rocm/bin:$PATH
ENV HIP_PLATFORM=nvidia
```

`devcontainer.json`:
```json
{
    "build": {
        "dockerfile": "Dockerfile"
    },
    "name": "cuda-minimal",
    "runArgs": [
        "--gpus=all"
    ]
}
```

</details>

## Running all hipCUB benchmarks

Save this file as `rocm-libraries/projects/hipcub/run_benchmarks.py` (you can use `scp` to copy files to a server through a terminal):

<details>
<summary><code>run_benchmarks.py</code></summary>

```py
#!/usr/bin/env python3

# Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
from collections import namedtuple
import json
import os
import re
import stat
import subprocess
import sys

# Added 'hot' to the BenchmarkContext namedtuple
BenchmarkContext = namedtuple('BenchmarkContext', ['gpu_architecture', 'benchmark_output_dir', 'benchmark_dir', 'benchmark_filename_regex', 'benchmark_filter_regex', 'size', 'trials', 'seed', 'skip_gathered', 'iteration_info_output_dir', 'benchmark_min_time', 'hot'])

def run_benchmarks(benchmark_context):
    def is_benchmark_executable(filename):
        if not re.match(benchmark_context.benchmark_filename_regex, filename):
            return False
        path = os.path.join(benchmark_context.benchmark_dir, filename)
        st_mode = os.stat(path).st_mode

        # we are not interested in permissions, just whether there is any execution flag set
        # and it is a regular file (S_IFREG)
        return (st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)) and (st_mode & stat.S_IFREG)

    def should_skip(results_json_path):
        if not benchmark_context.skip_gathered:
            return False

        try:
            with open(results_json_path) as f:
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        return True

    success = True
    benchmark_names = [name for name in os.listdir(benchmark_context.benchmark_dir) if is_benchmark_executable(name)]
    print('The following benchmarks will be run:\n{}'.format('\n'.join(benchmark_names)), file=sys.stderr, flush=True)
    for benchmark_name in benchmark_names:
        results_json_name = f'{benchmark_name}_{benchmark_context.gpu_architecture}.json'

        benchmark_path = os.path.join(benchmark_context.benchmark_dir, benchmark_name)
        results_json_path = os.path.join(benchmark_context.benchmark_output_dir, results_json_name)
        if should_skip(results_json_path):
            print(f'Skipping {benchmark_name}, because its results have already been gathered at {results_json_path}', file=sys.stderr, flush=True)
            continue
        args = [
            benchmark_path,
            f'--benchmark_out={results_json_path}',
            f'--benchmark_filter={benchmark_context.benchmark_filter_regex}'
        ]

        if benchmark_context.size:
            args += ['--size', benchmark_context.size]
        if benchmark_context.trials:
            args += ['--trials', benchmark_context.trials]
        if benchmark_context.seed:
            args += ['--seed', benchmark_context.seed]
        if benchmark_context.iteration_info_output_dir:
            args += ['--iteration_info_out', os.path.join(benchmark_context.iteration_info_output_dir, results_json_name)]
        if benchmark_context.benchmark_min_time:
            args += ['--benchmark_min_time', benchmark_context.benchmark_min_time]
        # Pass the --hot flag to the executable if it was set
        if benchmark_context.hot:
            args += ['--hot']

        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError as error:
            print(f'Could not run benchmark at {benchmark_path}. Error: "{error}"', file=sys.stderr, flush=True)
            success = False
    return success



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_dir',
        help='The local directory that contains the benchmark executables',
        required=True)
    parser.add_argument('--benchmark_gpu_architecture',
        help='The architecture of the currently enabled GPU',
        required=True)
    parser.add_argument('--benchmark_output_dir',
        help='The directory to write the benchmarks to',
        required=True)
    parser.add_argument('--benchmark_filename_regex',
        help='Regular expression that controls the list of benchmark executables to run',
        default=r'^benchmark',
        required=False)
    parser.add_argument('--benchmark_filter_regex',
        help='Regular expression that controls the list of benchmarks to run in each benchmark executable',
        default='',
        required=False)
    parser.add_argument('--size',
        help='Controls the number of processed items in each benchmark',
        default='',
        required=False)
    parser.add_argument('--trials',
        help='Controls the number of trial iterations for each benchmark case',
        default='',
        required=False)
    parser.add_argument('--seed',
        help='Controls the seed for random number generation for each benchmark case',
        default='',
        required=False)
    parser.add_argument('--skip_gathered',
        help='Skip running benchmarks whose JSON data has already been gathered',
        default=False,
        action='store_true',
        required=False)
    parser.add_argument('--iteration_info_output_dir',
        help='The directory to write the benchmark iteration info to',
        required=False)
    parser.add_argument('--benchmark_min_time', # TODO: Remove this option once the benchmarks don't use Google Benchmark anymore.
        help='The minimum amount of time for Google Benchmark to run a benchmark for, where the value \'0s\' means no minimum time',
        required=False)
    # Added the --hot argument to the parser
    parser.add_argument('--hot',
        help='Pass the --hot flag to the benchmark executables',
        default=False,
        action='store_true',
        required=False)

    args = parser.parse_args()

    # Included args.hot in the BenchmarkContext initialization
    benchmark_context = BenchmarkContext(
        args.benchmark_gpu_architecture,
        args.benchmark_output_dir,
        args.benchmark_dir,
        args.benchmark_filename_regex,
        args.benchmark_filter_regex,
        args.size,
        args.trials,
        args.seed,
        args.skip_gathered,
        args.iteration_info_output_dir,
        args.benchmark_min_time,
        args.hot)

    benchmark_run_successful = run_benchmarks(benchmark_context)

    return benchmark_run_successful


if __name__ == '__main__':
    success = main()
    if success:
        exit(0)
    else:
        exit(1)
```

</details>

## `roc*` Benchmark Commands

<details>
<summary>rocPRIM - HIP only</summary>

```sh
sudo apt update && \
sudo apt install -y ninja-build && \
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git && \
cd rocm-libraries && \
git sparse-checkout init --cone && \
git sparse-checkout set projects/rocprim shared/primbench && \
git checkout develop && \
cd projects/rocprim && \
mkdir build && \
cd build && \
export gfx=$(rocm_agent_enumerator | head -n1) && \
CXX=hipcc cmake -GNinja -DBUILD_BENCHMARK=ON -DGPU_TARGETS="$gfx" .. && \
ninja && \
cd .. && \
python3 .gitlab/run_benchmarks.py \
  --benchmark-executables-dir build/benchmark \
  --gpu-architecture $gfx \
  --json-out-dir build/results/json \
  --csv-out-dir build/results/csv
```

</details>

<details>
<summary>rocRAND - HIP</summary>

```sh
sudo apt update && \
sudo apt install -y ninja-build && \
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git && \
cd rocm-libraries && \
git sparse-checkout init --cone && \
git sparse-checkout set projects/rocrand shared/primbench && \
git checkout users/mynameistrez/use-primbench-in-rocrand && \
cd projects/rocrand && \
mkdir build && \
cd build && \
export gfx=$(rocm_agent_enumerator | head -n1) && \
CXX=hipcc cmake -GNinja -DBUILD_BENCHMARK=ON -DGPU_TARGETS="$gfx" .. && \
ninja benchmark_host_api benchmark_device_api && \
benchmark/benchmark_device_api \
  --json-out benchmark_rocrand_device_api.json \
  --csv-out benchmark_rocrand_device_api.csv && \
benchmark/benchmark_host_api \
  --json-out benchmark_rocrand_host_api.json \
  --csv-out benchmark_rocrand_host_api.csv
```

</details>

<details>
<summary>rocRAND - CUDA (cuRAND)</summary>

```sh
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git && \
cd rocm-libraries && \
git sparse-checkout init --cone && \
git sparse-checkout set projects/rocrand shared/primbench && \
git checkout users/mynameistrez/use-primbench-in-rocrand && \
cd projects/rocrand && \
mkdir build && \
cd build && \
cmake -GNinja -DBUILD_BENCHMARK=ON .. && \
ninja benchmark_host_api benchmark_device_api && \
benchmark/benchmark_device_api \
  --json-out benchmark_curand_device_api.json \
  --csv-out benchmark_curand_device_api.csv && \
benchmark/benchmark_host_api \
  --json-out benchmark_curand_host_api.json \
  --csv-out benchmark_curand_host_api.csv
```

</details>

<details>
<summary>rocThrust - HIP</summary>

```sh
sudo apt update && \
sudo apt install -y ninja-build && \
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git && \
cd rocm-libraries && \
git sparse-checkout init --cone && \
git sparse-checkout set projects/rocthrust && \
git checkout users/mynameistrez/fix-rocthrust-set-bytes-processed && \
cd projects/rocthrust && \
mkdir build && \
cd build && \
export gfx=$(rocm_agent_enumerator | head -n1) && \
CXX=hipcc cmake -GNinja -DBUILD_BENCHMARK=ON -DGPU_TARGETS="$gfx" .. && \
ninja $(ninja -t targets | grep '^benchmark_thrust_' | cut -d: -f1) && \
cd .. && \
mkdir build/results && \
python3 .gitlab/run_benchmarks.py \
  --benchmark_dir=build/benchmark \
  --benchmark_gpu_architecture="$gfx" \
  --benchmark_output_dir=build/results
```

</details>

<details>
<summary>rocThrust - CUDA (Thrust)</summary>

```sh
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git && \
cd rocm-libraries && \
git sparse-checkout init --cone && \
git sparse-checkout set projects/rocthrust projects/rocrand && \
git checkout users/naraenda/rocthrust-allow-grafting-thrust && \
cd projects/rocthrust && \
cmake --preset hip-nv-dev -DCCCL_DIR=/usr/local/cuda/lib64/cmake/cccl/ && \
cd build/hip-nv-dev && \
export gfx=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n 1 | tr -d '.') && \
ninja $(ninja -t targets | grep '^benchmark_thrust_' | cut -d: -f1) && \
cd ../.. && \
mkdir build/hip-nv-dev/results && \
python3 .gitlab/run_benchmarks.py \
  --benchmark_dir=build/hip-nv-dev/benchmark \
  --benchmark_gpu_architecture="$gfx" \
  --benchmark_output_dir=build/hip-nv-dev/results
```

</details>

## `hip*` Benchmark Commands

<details>
<summary>hipCUB - HIP</summary>

```sh
sudo apt update && \
sudo apt install -y ninja-build && \
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git && \
cd rocm-libraries && \
git sparse-checkout init --cone && \
git sparse-checkout set projects/hipcub && \
git checkout develop && \
cd projects/hipcub && \
mkdir build && \
cd build && \
export gfx=$(rocm_agent_enumerator | head -n1) && \
CXX=hipcc cmake -GNinja -DBUILD_BENCHMARK=ON -DGPU_TARGETS="$gfx" .. && \
ninja && \
cd ..
```

Then add `rocm-libraries/projects/hipcub/run_benchmarks.py`, as described under [Running all hipCUB benchmarks](#running-all-hipcub-benchmarks), and run it:

```sh
mkdir build/results && \
python3 run_benchmarks.py \
  --benchmark_dir=build/benchmark \
  --benchmark_gpu_architecture="$gfx" \
  --benchmark_output_dir=build/results
```

</details>

<details>
<summary>hipCUB - CUDA (CUB)</summary>

```sh
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git && \
cd rocm-libraries && \
git sparse-checkout init --cone && \
git sparse-checkout set projects/hipcub && \
git checkout develop && \
cd projects/hipcub && \
mkdir build && \
cd build && \
export gfx=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n 1 | tr -d '.') && \
cmake -GNinja -DBUILD_BENCHMARK=ON .. && \
ninja && \
cd ..
```

Then add `rocm-libraries/projects/hipcub/run_benchmarks.py`, as described under [Running all hipCUB benchmarks](#running-all-hipcub-benchmarks), and run it:

```sh
mkdir build/results && \
python3 run_benchmarks.py \
  --benchmark_dir=build/benchmark \
  --benchmark_gpu_architecture="$gfx" \
  --benchmark_output_dir=build/results
```

</details>

## Using gbench2primbench.py

rocThrust and hipCUB still use Google Benchmark, rather than primbench, meaning they output Google Benchmark JSON files. Since `grapher.py` only accepts _primbench_ JSON and CSV files, any Google Benchmark JSON files have to first be converted to primbench CSV files.

The script `gbench2primbench.py` converts a directory of old Google Benchmark JSON files, to a directory of new primbench CSV files.

<details><summary><code>--help</code> options</summary>

```
usage: gbench2primbench.py [-h] --project {rocprim,rocrand,hipcub,rocthrust} --noise-threshold-percentage NOISE_THRESHOLD_PERCENTAGE input_dir output_dir

Convert Google Benchmark JSON to primbench CSV format

positional arguments:
  input_dir             Directory containing Google Benchmark JSON files
  output_dir            Output directory for primbench CSV files

options:
  -h, --help            show this help message and exit
  --project {rocprim,rocrand,hipcub,rocthrust}
                        Project name
  --noise-threshold-percentage NOISE_THRESHOLD_PERCENTAGE
                        The noise threshold percentage, past which benchmark specializations are considered to be too noisy
```

</details>

<details>
<summary>Command for converting hipCUB</summary>

```sh
python3 gbench2primbench.py \
  --project hipcub \
  --noise-threshold-percentage 1 \
  build/results \
  converted
```

</details>

<details>
<summary>Command for converting rocThrust</summary>

```sh
python3 gbench2primbench.py \
  --project rocthrust \
  --noise-threshold-percentage 1 \
  build/results \
  converted
```

</details>

<details>
<summary><code>gbench2primbench.py</code></summary>

```sh
#!/usr/bin/env python3
"""
Convert Google Benchmark JSON files to primbench CSV format.

Reads Google Benchmark JSON output files from an input directory and converts
them to primbench CSV format, writing results to an output directory.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, cast
import re

seen_binary_search_params: Set[str] = set()

# These nested tuples generate a lookup table of scales
# that benchmark_device_histogram forgot to output
histogram_even_scales = (
    (12345,) * 4
    + (1234,) * 4
    + (5,) * 4
    + (16,)
    + (1,)
    + (1234,) * 8
    + (5,) * 4
    + (16,)
    + (1,)
    + (16,)
    + (1,)
) * 4
histogram_even_index = 0

histogram_multi_even_scales = (
    (1234,) * 4 + (5,) * 4 + (16,) + (1,) + (1234,) * 4 + (16,) + (1,) + (16,) + (1,)
) * 4
histogram_multi_even_index = 0

device_scan_deterministic_skipped_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)
device_scan_deterministic_index = 0

device_scan_skipped_indices = (8, 10)
device_scan_index = 0


def replace(params: Dict[str, Any], key: str, entries: Dict[str, str]):
    """Replaces for example an 'offset_type' its value 'int' with 'i32'."""
    for old, new in entries.items():
        if params.get(key) == old:
            params[key] = new


def transform(params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply algorithm-specific transformations to benchmark parameters."""

    # These params don't exist in hipcub
    lvl = params.get("lvl", "")
    algo = params.get("algo", "")
    name = lvl + "_" + algo

    for key in (
        "key_type",
        "value_type",
        "offset_type",
        "item_type",
        "size_type",
        "data_type",
        "flag_type",
        "input_type",
        "output_type",
    ):
        replace(
            params,
            key,
            {
                "char": "i8",
                "common::custom_type<1024,float,float>": "huge<1024,f32,f32>",
                "common::custom_type<2048,float,float>": "huge<2048,f32,f32>",
                "common::custom_type<char,double>": "custom<i8,f64>",
                "common::custom_type<char,short>": "custom<i8,i16>",
                "common::custom_type<double,double>": "custom<f64,f64>",
                "common::custom_type<float,float>": "custom<f32,f32>",
                "common::custom_type<float,int16_t>": "custom<f32,i16>",
                "common::custom_type<int,double>": "custom<i32,f64>",
                "common::custom_type<int,int>": "custom<i32,i32>",
                "common::custom_type<int64_t,double>": "custom<i64,f64>",
                "common::custom_type_copyable<char,double>": "copyable<i8,f64>",
                "common::custom_type_copyable<double,double>": "copyable<f64,f64>",
                "custom_128": "custom<i64,i64>",
                "custom_char_double": "custom<i8,f64>",
                "custom_double2": "custom<f64,f64>",
                "custom_float2": "custom<f32,f32>",
                "custom_int2": "custom<i32,i32>",
                "custom_int_double": "custom<i32,f64>",
                "custom_int_type": "custom<i32,i32>",
                "custom_longlong_double": "custom<i64,f64>",
                "custom_type<int,double>": "custom<i32,f64>",
                "double": "f64",
                "empty_type": "empty",
                "float": "f32",
                "int": "i32",
                "int16_t": "i16",
                "int32_t": "i32",
                "int64_t": "i64",
                "int8_t": "i8",
                "long long": "i64",
                "rocprim::half": "half",
                "rocprim::int128_t": "i128",
                "rocprim::uint128_t": "u128",
                "short": "i16",
                "uint32_t": "u32",
                "uint64_t": "u64",
                "uint8_t": "u8",
                "uint8_t": "u8",
                "unsigned char": "u8",
                "unsigned int": "u32",
                "unsigned long long": "u64",
            },
        )

    if name in ("device_adjacent_difference", "device_adjacent_difference_inplace"):
        if "is_left" in params:
            params["left"] = params.pop("is_left")
        params["inplace"] = algo.endswith("_inplace")

    if name == "block_radix_rank":
        params["cfg"]["method"] = params["cfg"]["method"].removeprefix(
            "rocprim::block_radix_rank_algorithm::"
        )

    if name == "block_run_length_decode":
        params["cfg"]["bs"] = params["cfg"].pop("block_size")

    # These were all in benchmark_config_dispatch
    if algo in (
        "default_stream",
        "per_thread_stream",
        "explicit_stream",
        "async_stream",
        "empty_kernel",
    ):
        params["method"] = algo

    if name == "device_adjacent_find":
        params["first_adj_pos"] = float(params["first_adj_pos"])

    # These were all in benchmark_device_batch_memcpy
    if algo in ("batch_memcpy", "batch_copy"):
        params["subalgo"] = algo

    # These were all in benchmark_device_binary_search
    if algo in ("binary_search", "lower_bound", "upper_bound"):
        params["key_type"] = params.pop("value_type")
        params["subalgo"] = algo
        params["needles_percent"] = 10

        global seen_binary_search_params
        s = str(params)
        params["sorted_needles"] = s not in seen_binary_search_params
        seen_binary_search_params.add(s)

    if name == "device_find_end":
        params["repeating"] = params.pop("value_pattern") == "repeating"

    if name == "device_find_first_of":
        params["first_occurrence"] = f"{float(params['first_occurrence']):g}"

    # These were all in benchmark_device_histogram
    if algo in (
        "histogram_even",
        "multi_histogram_even",
        "histogram_range",
        "multi_histogram_range",
    ):
        params["subalgo"] = algo.replace("histogram_", "")

    # benchmark_device_histogram forgot to output scale
    if algo == "histogram_even":
        global histogram_even_index
        params["scale"] = histogram_even_scales[histogram_even_index]
        histogram_even_index += 1
    if algo == "multi_histogram_even":
        global histogram_multi_even_index
        params["scale"] = histogram_multi_even_scales[histogram_multi_even_index]
        histogram_multi_even_index += 1

    if name == "device_memory" and params["subalgo"] == "copy":
        params["cfg"] = {"bs": 1, "ipt": 1}
        params["operation"] = "no_operation"

    if name in (
        "device_nth_element",
        "device_partial_sort_copy",
        "device_partial_sort",
    ):
        params["small_n"] = params.pop("nth") == "small"

    if algo == "partition_two_way":
        params["subalgo"] = f"two_way_{params['subalgo']}"
    if algo == "partition_three_way":
        params["subalgo"] = "three_way"

    if name == "device_run_length_encode" and "subalgo" in params:
        del params["subalgo"]

    if name == "device_search":
        params["repeating"] = params.pop("value_pattern") == "repeating"

    if algo in ("transform", "transform_pointer"):
        params["is_binary"] = params.pop("op") == "binary"

    if algo in ("read_predicate_it", "write_predicate_it", "transform_it"):
        params["subalgo"] = algo.removesuffix("_it")
        params["percent"] = params.pop("p").removeprefix("p")

    # segmented_radix_sort_keys always output "value_type: empty"
    if algo == "segmented_radix_sort" and params["value_type"] == "empty":
        del params["value_type"]

    if name == "warp_sort":
        if params["value_type"] == "empty":
            del params["value_type"]

    return params


def strip_prefixes(s: str) -> str:
    """Remove rocprim:: and common:: prefixes from string."""
    prefixes = ["rocprim::", "common::"]
    for prefix in prefixes:
        while prefix in s:
            s = s.replace(prefix, "")
    return s


def serialize(value: Any) -> str:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, Any], value)
        items: List[str] = []
        for k, v in mapping.items():
            items.append(f"{k}: {serialize(v)}")
        return "{ " + ", ".join(items) + " }"

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = cast(Sequence[Any], value)
        return "[ " + ", ".join(serialize(v) for v in seq) + " ]"

    if isinstance(value, bool):
        return str(value).lower()

    if value is None:
        return "null"

    return strip_prefixes(str(value))


def sort_dict_alphabetically(d: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new dict with keys sorted alphabetically, recursively."""
    sorted_dict: Dict[str, Any] = {}
    for key in sorted(d, key=str):
        value = d[key]
        if isinstance(value, Mapping):
            sorted_dict[key] = sort_dict_alphabetically(cast(Mapping[str, Any], value))
        else:
            sorted_dict[key] = value
    return sorted_dict


def parse_benchmark_name(name: str) -> str:
    """Extract and format benchmark parameters from JSON-formatted name."""
    name = name.removesuffix("/manual_time")
    name = name.removesuffix("/iterations:100")

    params = json.loads(name)
    params = transform(params)

    # Alphabetically sort keys
    params = sort_dict_alphabetically(params)

    blacklist = {"lvl", "algo"}

    parts: List[str] = []
    for key, value in params.items():
        # Skip blacklisted keys
        if key in blacklist:
            continue

        # Skip default configs
        if key == "cfg" and value == "default_config":
            continue

        parts.append(f"{key}: {serialize(value)}")

    return ", ".join(parts)


def convert_rocrand_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for idx, bench in enumerate(data["benchmarks"]):
        name = bench["name"]  # device_kernel<lfsr113,uniform-uint>/manual_time

        if name.startswith("device_kernel<"):
            is_device_api = True
            inner = name[len("device_kernel<") :]  # lfsr113,uniform-uint>/manual_time
        elif name.startswith("device_generate<"):
            is_device_api = False
            inner = name[
                len("device_generate<") :
            ]  # lfsr113,default,uniform-uint>/manual_time
        else:
            continue

        inner = inner.split(">", 1)[0]  # lfsr113,uniform-uint
        parts = inner.split(",")  # ['lfsr113', 'uniform-uint']

        ordering = None
        if len(parts) == 2:
            assert is_device_api
            engine, distribution_raw = parts  # lfsr113, uniform-uint
        elif len(parts) == 3:
            assert not is_device_api
            engine, ordering, distribution_raw = parts  # lfsr113, default, uniform-uint
        else:
            continue

        poisson_lambda = None

        if distribution_raw.startswith("uniform-"):
            distribution_name = "uniform"
            type_raw = distribution_raw[len("uniform-") :]  # uint
        elif distribution_raw.startswith("normal-"):
            distribution_name = "normal"
            type_raw = distribution_raw[len("normal-") :]  # float
        elif distribution_raw.startswith("log-normal-"):
            distribution_name = "log_normal"
            type_raw = distribution_raw[len("log-normal-") :]  # float
        elif distribution_raw.startswith(
            "discrete-poisson("
        ) or distribution_raw.startswith("poisson("):
            distribution_name = (
                "discrete_poisson"
                if distribution_raw.startswith("discrete-poisson(")
                else "poisson"
            )
            type_raw = "uint"
            if "lambda=" in distribution_raw:
                raw = distribution_raw.split("lambda=")[1].rstrip(")")  # 10.0
                lam = float(raw)
                poisson_lambda = str(int(lam)) if lam.is_integer() else str(lam)  # 10
        elif distribution_raw == "discrete-custom":
            distribution_name = "discrete_custom"
            type_raw = "uint"
        else:
            continue

        type_map = {
            "uchar": "u8",
            "ushort": "u16",
            "uint": "u32",
            "long-long": "u64",  # It not being "ulong-long" was a bug in old benchmark
            "ullong": "u64",
            "float": "f32",
            "double": "f64",
            "half": "half",
        }

        type_name = type_map[type_raw]  # u32

        name_parts: List[str] = []

        # Device API benchmarks do not provide a config.
        # Assume 256 blocks and 256 threads, even though they may have been overriden.
        if is_device_api:
            name_parts.append("cfg: { blocks: 256, threads: 256 }")

        name_parts.append(f"distribution: {distribution_name}")
        name_parts.append(f"engine: {engine}")

        if ordering is not None:
            if "sobol" in engine:
                name_parts.append(f"ordering: quasi_default")
            else:
                name_parts.append(f"ordering: {ordering}")
        if poisson_lambda is not None:
            name_parts.append(f"poisson_lambda: {poisson_lambda}")

        name_parts.append(f"type: {type_name}")

        result_name = ", ".join(name_parts)

        bytes_per_second = bench["bytes_per_second"]
        gib_per_second = bytes_per_second / (1024.0 * 1024.0 * 1024.0)

        results.append(
            {
                "index": idx,
                "name": result_name,
                "bytes_per_second": bytes_per_second,
                "gib_per_second": gib_per_second,
                "items_per_second": bench["items_per_second"],
                "noise_timeout": 0,
                "noise_percent": 0,
            }
        )

    return results


def convert_rocprim_json(
    data: Dict[str, Any], noise_threshold: float
) -> List[Dict[str, Any]]:
    """Load Google Benchmark JSON and convert to primbench format."""
    times_seen_adjacent_find_i32 = 0
    times_seen_adjacent_find_i16 = 0

    seen_specializations: Set[str] = set()

    results: List[Dict[str, Any]] = []
    for idx, bench in enumerate(data["benchmarks"]):
        name = parse_benchmark_name(bench["name"])

        # device_adjacent_find registers i16 and i32 specializations
        # in a group of three: 0.1, 0.5, and 0.9
        # It accidentally registered this group twice, so skip that 2nd group
        if "adjacent_find" in bench["name"]:
            if name.endswith("input_type: i16"):
                times_seen_adjacent_find_i16 += 1
                if times_seen_adjacent_find_i16 > 3:
                    continue
            if name.endswith("input_type: i32"):
                times_seen_adjacent_find_i32 += 1
                if times_seen_adjacent_find_i32 > 3:
                    continue

        # These accidentally benchmarked some specializations several times
        if (
            "find_first_of" in bench["name"] or "reduce_by_key" in bench["name"]
        ) and bench["name"] in seen_specializations:
            continue

        # The only way to tell device_scan_by_key_deterministic
        # apart from device_scan_by_key is the executable/JSON file name
        # The executable is less likely to have been renamed, so use that
        if data["context"]["executable"].endswith(
            "/benchmark_device_scan_by_key_deterministic"
        ):
            # benchmark_device_scan_by_key_deterministic accidentally benchmarked
            # specializations that had Deterministic=False, so skip those
            if "key_type: i32," not in name or "max_segment_length: 1024" in name:
                continue

        if data["context"]["executable"].endswith(
            "/benchmark_device_scan_deterministic"
        ):
            global device_scan_deterministic_index
            skipped = (
                device_scan_deterministic_index
                in device_scan_deterministic_skipped_indices
            )
            device_scan_deterministic_index += 1
            if skipped:
                continue

        if data["context"]["executable"].endswith("/benchmark_device_scan"):
            global device_scan_index
            skipped = device_scan_index in device_scan_skipped_indices
            device_scan_index += 1
            if skipped:
                continue

        seen_specializations.add(bench["name"])

        bytes_per_second = bench["bytes_per_second"]
        gib_per_second = bytes_per_second / (1024.0 * 1024.0 * 1024.0)

        result: Dict[str, Any] = {
            "index": idx,
            "name": name,
            "bytes_per_second": bytes_per_second,
            "gib_per_second": gib_per_second,
            "items_per_second": bench["items_per_second"],
            "noise_timeout": 1 if (bench["cv"] * 100) > noise_threshold else 0,
            "noise_percent": bench["cv"] * 100,
        }
        results.append(result)

    # This asserts that scale was added to all even and multi_even specializations
    if "histogram_even" in data["benchmarks"][0]["name"]:
        assert len(histogram_even_scales) == histogram_even_index
        assert len(histogram_multi_even_scales) == histogram_multi_even_index

    return results


def split_top_level(s):
    parts = []
    current = []
    depth = 0

    for ch in s:
        if ch in ("<", "[", "("):
            depth += 1
        elif ch in (">", "]", ")"):
            depth -= 1

        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append("".join(current))

    return parts


def fix_to_json(s):
    s = s.strip("{}")
    result = {}

    for part in split_top_level(s):
        if ":::" in part:
            key, value = part.split(":::", 1)
            value = "::" + value
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            continue

        key = key.strip()
        value = value.strip()

        if value.isdigit():
            value = int(value)

        if isinstance(value, str):
            value = re.sub(
                r"::hipcub::([A-Z0-9_]+)",
                lambda m: m.group(1).lower(),
                value
            )

            value = re.sub(
                r"hipcub::([A-Z0-9_]+)",
                lambda m: m.group(1).lower(),
                value
            )

        result[key] = value

    return result

def convert_hipcub_json(
    data: Dict[str, Any], noise_threshold: float
) -> List[Dict[str, Any]]:
    """Load Google Benchmark JSON and convert to hipcub format."""
    times_seen_adjacent_find_i32 = 0
    times_seen_adjacent_find_i16 = 0

    seen_specializations: Set[str] = set()

    # Hack for device_for
    device_for_f32_found = False
    device_for_f64_found = False

    results: List[Dict[str, Any]] = []
    for idx, bench in enumerate(data["benchmarks"]):
        name_str = bench["name"]

        is_block_adjacent_difference = "block_adjacent_difference" in name_str
        is_block_discontinuity = "block_discontinuity" in name_str
        is_block_radix_rank = "block_radix_rank" in name_str
        is_block_reduce = "block_reduce" in name_str
        is_block_scan = "block_scan" in name_str
        is_block_shuffle = "block_shuffle" in name_str
        is_device_for = "for_each" in name_str
        is_device_histogram = "device_histogram" in name_str or "device_multi_histogram" in name_str
        is_device_memory = "device_memory" in name_str
        is_device_merge_sort = "device_merge_sort" in name_str
        is_device_merge = "device_merge" in name_str and not is_device_merge_sort
        is_device_partition = "device_parition" in name_str # "partition" is misspelled in the old gbench code
        is_device_radix_sort = "device_radix_sort" in name_str
        is_device_reduce_by_key = "device_reduce_by_key" in name_str
        is_device_reduce = "device_reduce" in name_str and not is_device_reduce_by_key
        is_device_run_length_encode = "device_run_length_encode" in name_str or "run_length_encode_non_trivial_runs" in name_str
        is_device_scan = "device_inclusive_scan" in name_str or "device_exclusive_scan" in name_str
        is_device_segmented_radix_sort = "device_segmented_radix_sort_keys" in name_str or "device_segmented_radix_sort_pairs" in name_str
        is_device_segmented_reduce = "device_segmented_reduce" in name_str
        is_device_segmented_sort = "device_segmented_sort" in name_str
        is_device_select = "device_select" in name_str
        is_device_spmv = "device_spmv" in name_str
        is_warp_exchange = "warp_exchange" in name_str
        is_warp_merge_sort = "warp_merge_sort" in name_str
        is_warp_reduce = "warp_reduce" in name_str

        if is_warp_merge_sort:
            # The original gbench code had these two swapped, so we fix it here
            if "segmented_sort" in name_str:
                name_str = re.sub(r":segmented_sort", ":sort", name_str)
            else:
                name_str = re.sub(r":sort", ":segmented_sort", name_str)

            # Turn the subalgo into the segmented+pair bools
            warp_merge_is_segmented = "segmented_sort" in name_str
            warp_merge_is_pairs = "values" in name_str

        if is_warp_reduce:
            # Turn the subalgo into the segmented+pair bools
            warp_reduce_is_segmented = "segmented_reduce" in name_str

        if is_warp_exchange:
            if "warp_exchange_striped_to_blocked" in name_str:
                name_str = re.sub(r"<", "<op:striped_to_blocked_op,", name_str, count=1)
            elif "warp_exchange_blocked_to_striped" in name_str:
                name_str = re.sub(r"<", "<op:blocked_to_striped_op,", name_str, count=1)
            elif "warp_exchange_scatter_to_striped" in name_str:    
                name_str = re.sub(r"<", "<subalgo:scatter_to_striped,", name_str, count=1)
        
        if is_block_adjacent_difference:
            name_str = re.sub(r"subtract_left<", "subtract_left,", name_str, count=1)
            name_str = re.sub(r"subtract_right<", "subtract_right,", name_str, count=1)
            name_str = re.sub(r"subtract_left_partial_tile<", "subtract_left_partial_tile,", name_str, count=1)
            name_str = re.sub(r"subtract_right_partial_tile<", "subtract_right_partial_tile,", name_str, count=1)
        
        if is_block_discontinuity:
            name_str = re.sub(r"flag_heads<", "flag_heads,", name_str, count=1)
            name_str = re.sub(r"flag_tails<", "flag_tails,", name_str, count=1)
            name_str = re.sub(r"flag_heads_and_tails<", "flag_heads_and_tails,", name_str, count=1)

        if is_block_radix_rank:
            name_str = name_str.replace("kind", "sub_algorithm_name");

            name_str = name_str.replace("RadixRankAlgorithm::RADIX_RANK_BASIC", "basic")
            name_str = name_str.replace("RadixRankAlgorithm::RADIX_RANK_MATCH", "match")
            name_str = name_str.replace("RadixRankAlgorithm::RADIX_RANK_MEMOIZE", "memoize")

        if is_block_reduce:
            # Lowercase the enums
            name_str = name_str.replace("BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY", "block_reduce_raking_commutative_only")
            name_str = name_str.replace("BLOCK_REDUCE_RAKING", "block_reduce_raking")
            name_str = name_str.replace("BLOCK_REDUCE_WARP_REDUCTIONS", "block_reduce_warp_reductions")

        if is_block_scan:
            # Lowercase the enums
            name_str = name_str.replace("BLOCK_SCAN_RAKING_MEMOIZE", "block_scan_raking_memoize")
            name_str = name_str.replace("BLOCK_SCAN_RAKING", "block_scan_raking")
            name_str = name_str.replace("BLOCK_SCAN_WARP_SCANS", "block_scan_warp_scans")

        if is_device_histogram:
            # Get rid of the ()
            name_str = name_str.replace("(", "")
            name_str = name_str.replace(">.entropy_percent", ",entropy_percent")
            name_str = name_str.replace(">.bin_count", ",bin_count")
            name_str = name_str.replace(" bins)", ">")
            name_str = name_str.replace("%", "")

            subalgo = "even" if "device_histogram_even" in name_str else ""
            subalgo = "multi_even" if "device_multi_histogram_even" in name_str else subalgo
            subalgo = "range" if "device_histogram_range" in name_str else subalgo
            subalgo = "multi_range" if "device_multi_histogram_range" in name_str else subalgo

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

        if is_device_reduce_by_key or is_device_run_length_encode or is_device_partition or is_device_segmented_radix_sort or is_device_segmented_reduce or is_device_segmented_sort or is_device_select:
            inside = re.search(r'\((.*?)\)', name_str).group(1)
            name_str = re.sub(r'\.?\(.*?\)', '', name_str)

            pos = name_str.rfind('>')
            name_str = name_str[:pos] + f",{inside}" + name_str[pos:]

        if is_device_select:
            name_str = re.sub(r"probability:\s*([0-9]*\.?[0-9]+)f", r"probability: \1", name_str)

            subalgo = "flagged" if "device_select_flagged" in name_str else ""
            subalgo = "flagged_if" if "device_select_flagged_if" in name_str else subalgo
            subalgo = "unique" if "device_select_unique" in name_str else subalgo
            subalgo = "unique_by_key" if "device_select_unique_by_key" in name_str else subalgo
            subalgo = "if" if "device_select_if" in name_str else subalgo

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

            if "device_select_unique_by_key" in name_str:
                name_str = name_str.replace("Key data_type", "key_data_type")

        if is_device_segmented_reduce or is_device_segmented_sort:
            name_str = re.sub(r"number_of_segments:~(\d+)\s+segments", r"desired_segments: \1", name_str)

        if is_device_segmented_radix_sort:
            name_str = re.sub(r"segments:~(\d+)\s+segments", r"desired_segments: \1", name_str)

            keys = "device_segmented_radix_sort_keys" in name_str
            subalgo = "sort_keys" if keys else "sort_pairs"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

        if is_device_segmented_sort:
            keys = "device_segmented_sort_keys" in name_str
            subalgo = "sort_keys" if keys else "sort_pairs"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

        if is_device_run_length_encode:
            nontrivial = "run_length_encode_non_trivial_runs" in name_str
            subalgo = "non_trivial_runs" if nontrivial else "encode"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

            name_str = name_str.replace("run_length_encode_non_trivial_runs", "device_run_length_encode")

        if is_device_spmv:
            name_str = re.sub(r'e-(\d)f', r'e-0\1f', name_str)

        if is_device_partition:
            # The subalgo is included in the name, so properly put "subalgo: [subalgo]" before the last >
            subalgo = None
            if "flagged" in name_str:
                subalgo = "flagged"
            elif "predicate" in name_str:
                subalgo = "predicate"
            elif "three_way" in name_str:
                subalgo = "three_way"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo:{subalgo}" + name_str[pos:] 

        if is_device_reduce or is_device_segmented_reduce:
            name_str = name_str.replace("hipcub::ArgMin", "argmin")
            name_str = name_str.replace("argMin", "argmin")

        if is_device_radix_sort:
            descending = "descending" in name_str

            subalgo = "sort_keys"
            if "value_data_type" in name_str:
                subalgo = "sort_pairs"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",descending:{descending}, subalgo:{subalgo}" + name_str[pos:] 

        if is_device_merge:
            if "value_data_type" in name_str:
                name_str = name_str.replace("<", "<subalgo: merge_pairs, ", 1)
            else:
                name_str = name_str.replace("<", "<subalgo: merge_keys, ", 1)  

        if is_device_merge_sort:
            if "value_data_type" in name_str:
                name_str = name_str.replace("<", "<subalgo: sort_pairs, ", 1)
            else:
                name_str = name_str.replace("<", "<subalgo: sort_keys, ", 1)

        if is_device_scan:
            exclusive = False
            if "device_exclusive_scan" in name_str:
                exclusive = True

            subalgo = "scan"
            if "by_key" in name_str:
                subalgo = "scan_by_key"

            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",exclusive:{exclusive}, subalgo:{subalgo}" + name_str[pos:] 

        if is_device_memory and "device_memory_memcpy" in name_str:
            pos = name_str.rfind('>')  
            name_str = name_str[:pos] + f",subalgo: memcpy" + name_str[pos:] 

        name_str = re.sub(r'^[^<]*<', '', name_str)
        name_str = name_str.removesuffix("/manual_time")
        name_str = re.sub(r"/iterations:\d+$", "", name_str)
        name_str = re.sub(r'>$','', name_str)

        name_str = name_str.replace("unsigned int", "u32")
        name_str = name_str.replace("uint8_t", "u8")
        name_str = name_str.replace("int8_t", "i8")
        name_str = name_str.replace("unsigned short", "u16")
        name_str = name_str.replace("uint16_t", "u16")
        name_str = name_str.replace("int16_t", "i16")
        name_str = name_str.replace("uint32_t", "u32")
        name_str = name_str.replace("int32_t", "i32")
        name_str = name_str.replace("uint64_t", "u64")
        name_str = name_str.replace("std::int64_t", "i64")
        name_str = name_str.replace("int64_t", "i64")
        name_str = name_str.replace("unsigned long long", "u64")
        name_str = name_str.replace("custom_int_t", "custom<i32>")
        name_str = name_str.replace("custom_int_double", "custom<i32,f64>")
        name_str = name_str.replace("long long", "i64")
        name_str = re.sub(r'\bint\b', 'i32', name_str)
        name_str = re.sub(r'\b__half\b', 'f16', name_str)
        name_str = re.sub(r'\bshort\b', 'i16', name_str)
        name_str = re.sub(r'\bfloat\b', 'f32', name_str)
        name_str = re.sub(r'\bdouble\b', 'f64', name_str)
        name_str = name_str.replace("sub_algorithm_name", "subalgo")
        name_str = name_str.replace(">.", ",")

        name_str = name_str.replace("Datatype", "data_type")

        name_str = "{" + name_str + "}"

        if is_device_for:       
            # f32 and f64 are duplicated in the old gbench code
            if not device_for_f32_found and "f32" in name_str:
                device_for_f32_found = True
                continue

            if not device_for_f64_found and "f64" in name_str:
                device_for_f64_found = True
                continue

        if is_device_memory:
            # Substitute size: megabytes<i32>(x) with the computed value of x * 1024 * 1024
            name_str = re.sub(r'megabytes<(?:i32|int)>\((\d+)\)', lambda m: str(int(m.group(1)) * 1024 * 1024), name_str)
            name_str = name_str.replace("operation", "kernel_op")
            name_str = name_str.replace("method", "subalgo")

        name_str = name_str.replace("custom_double2", "custom<f64,f64>")
        name_str = name_str.replace("custom_float2", "custom<f32,f32>")
        name_str = name_str.replace("custom_char_double", "custom<i8,f64>")
        name_str = name_str.replace("custom_double_char", "custom<f64,i8>")

        name_json = fix_to_json(name_str)

        if is_warp_merge_sort:
            name_json["segmented"] = warp_merge_is_segmented
            name_json["pairs"] = warp_merge_is_pairs
            del name_json["subalgo"]

        if is_warp_reduce:
            name_json["segmented"] = warp_reduce_is_segmented
            del name_json["subalgo"]

        # In some cases method_name is just the actual algorithm name, and in others, it's a part of the subalgorithm name
        if "method_name" in name_json:
            if is_block_scan:
                # Merge subalgo and method_name into one
                name_json["subalgo"] = name_json.pop("method_name") + "(" + name_json["subalgo"] + ")"
            else:    
                name_json.pop("method_name")

        if is_block_shuffle:
            # If the subalgo is either "offset" or "rotate", then we have to include a dummy value for the graph matchmaking
            if name_json["subalgo"] == "offset" or name_json["subalgo"] == "rotate":
                name_json["items_per_thread"] = 1

        name = parse_benchmark_name(json.dumps(name_json))

        # Fix boolean capitalization
        name = re.sub(r"\bTrue\b", "true", name)
        name = re.sub(r"\bFalse\b", "false", name)

        seen_specializations.add(bench["name"])

        bytes_per_second = bench["bytes_per_second"]
        gib_per_second = bytes_per_second / (1024.0 * 1024.0 * 1024.0)

        result: Dict[str, Any] = {
            "index": idx,
            "name": name,
            "bytes_per_second": bytes_per_second,
            "gib_per_second": gib_per_second,
            "items_per_second": bench["items_per_second"],
            "noise_timeout": 1 if (bench.get("cv", 0) * 100) > noise_threshold else 0,
            "noise_percent": bench.get("cv", 0) * 100,
        }
        results.append(result)

    return results

def convert_rocthrust_json(
    data: Dict[str, Any], noise_threshold: float
) -> List[Dict[str, Any]]:
    """Load rocThrust Google Benchmark JSON and convert to primbench format.

    rocThrust encodes benchmark parameters directly as a JSON object in the
    benchmark name field, e.g.:
      {"algo":"adjacent_difference","subalgo":"basic","input_type":"int8_t",
       "elements":"1 << 16"}/min_time:0.400/manual_time

    Type names use rocThrust conventions (float32_t, float64_t,
    bench_utils::large_data, etc.) that are mapped to primbench conventions
    (f32, f64, large_data, etc.).  The noise metric is 'gpu_noise' (a
    fraction), rather than the 'cv' used by rocPRIM and hipCUB.
    """
    TYPE_MAP: Dict[str, str] = {
        "bench_utils::large_data": "large_data",
        "double": "f64",
        "float": "f32",
        "float32_t": "f32",
        "float64_t": "f64",
        "int128_t": "i128",
        "int16_t": "i16",
        "int32_t": "i32",
        "int64_t": "i64",
        "int8_t": "i8",
        "uint16_t": "u16",
        "uint32_t": "u32",
        "uint64_t": "u64",
        "uint8_t": "u8",
    }

    TYPE_FIELDS = {
        "input_type",
        "key_type",
        "output_type",
        "value_type",
    }

    results: List[Dict[str, Any]] = []

    for idx, bench in enumerate(data["benchmarks"]):
        # Skip benchmarks that failed at runtime (e.g. hipErrorOutOfMemory)
        if bench.get("error_occurred"):
            continue

        raw_name = bench["name"]

        # Strip benchmark timing suffixes, e.g. "/min_time:0.400/manual_time"
        raw_name = re.sub(r"/min_time:[0-9.]+/manual_time$", "", raw_name)
        raw_name = raw_name.removesuffix("/manual_time")
        raw_name = re.sub(r"/iterations:\d+$", "", raw_name)

        # The name is a JSON object encoding all benchmark parameters
        raw_name = re.sub(r'""([^"]+)""', r'"\1"', raw_name)
        params: Dict[str, Any] = json.loads(raw_name)

        # Apply type name transformations to fields that carry type names
        for field in TYPE_FIELDS:
            if field in params and isinstance(params[field], str):
                params[field] = TYPE_MAP.get(params[field], params[field])

        # Sort alphabetically (recursive, consistent with other converters)
        params = sort_dict_alphabetically(params)

        # Build primbench name; exclude 'algo' because it is encoded in the
        # CSV filename (mirrors the rocPRIM convention of excluding 'algo'/'lvl')
        blacklist = {"algo"}
        parts: List[str] = []
        for key, value in params.items():
            if key in blacklist:
                continue
            parts.append(f"{key}: {serialize(value)}")
        name = ", ".join(parts)

        bytes_per_second = bench["bytes_per_second"]
        gib_per_second = bytes_per_second / (1024.0 * 1024.0 * 1024.0)

        # rocThrust reports noise as 'gpu_noise' (a fraction 0-1); treat
        # absent / null values as zero noise
        gpu_noise: float = bench.get("gpu_noise") or 0.0
        noise_percent = gpu_noise * 100

        result: Dict[str, Any] = {
            "index": idx,
            "name": name,
            "bytes_per_second": bytes_per_second,
            "gib_per_second": gib_per_second,
            "items_per_second": bench["items_per_second"],
            "noise_timeout": 1 if noise_percent > noise_threshold else 0,
            "noise_percent": noise_percent,
        }
        results.append(result)

    return results


def write_csv_output(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Write results to primbench CSV format."""
    # Sort results alphabetically by name
    results = sorted(results, key=lambda x: x["name"])

    # Re-index after sorting
    for idx, result in enumerate(results):
        result["index"] = idx

    fieldnames = [
        "index",
        "name",
        "bytes_per_second",
        "gib_per_second",
        "items_per_second",
        "noise_timeout",
        "noise_percent",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            # Format floating point values with C++17 std::ofstream double precision
            row["bytes_per_second"] = f"{row['bytes_per_second']:.5e}"
            row["gib_per_second"] = f"{row['gib_per_second']:g}"
            row["items_per_second"] = f"{row['items_per_second']:.5e}"
            row["noise_percent"] = f"{row['noise_percent']:.6f}"
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Google Benchmark JSON to primbench CSV format"
    )
    parser.add_argument(
        "--project", choices=["rocprim", "rocrand", "hipcub", "rocthrust"], required=True, help="Project name"
    )
    parser.add_argument(
        "--noise-threshold-percentage",
        type=float,
        required=True,
        help="The noise threshold percentage, past which benchmark specializations "
        "are considered to be too noisy",
    )
    parser.add_argument(
        "input_dir", type=Path, help="Directory containing Google Benchmark JSON files"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for primbench CSV files"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each JSON file in input directory
    for json_path in args.input_dir.glob("*.json"):
        print(f"Converting {json_path.name}...")
        with open(json_path, "r") as f:
            data = json.load(f)

        if args.project == "rocprim":
            results = convert_rocprim_json(data, args.noise_threshold_percentage)
        elif args.project == "rocrand":
            results = convert_rocrand_json(data)
        elif args.project == "hipcub":
            results = convert_hipcub_json(data, args.noise_threshold_percentage)
        elif args.project == "rocthrust":
            results = convert_rocthrust_json(data, args.noise_threshold_percentage)
        else:
            raise ValueError(f"Missing convert function for {args.project}")

        # Output file has same stem as input, but with .csv extension
        output_file = args.output_dir / f"{json_path.stem}.csv"
        write_csv_output(results, output_file)
        print(f"Converted {json_path.name} -> {output_file.name}")


if __name__ == "__main__":
    main()
```

</details>

## Using grapher.py

The script `grapher.py` takes primbench JSON and CSV files, and generates a relative or absolute graph.

In the previous step `gbench2primbench.py` was used to convert old Google Benchmark JSON files to primbench CSV files.

The zip of the results contains both relative and absolute graphs, where the absolute graphs were generated by passing `--absolute` to `grapher.py`.

<details><summary><code>--help</code> options</summary>

```
Usage: grapher.py [-h] --output OUTPUT [--algo ALGO] [--arch ARCH] [--filter FILTER] [--absolute] input_files [input_files ...]

Positional Arguments:
  input_files      paths to input .json or .csv primbench files

Options:
  -h, --help       show this help message and exit
  --output OUTPUT  path to output .svg graph (default: None)
  --algo ALGO      algorithm name for the chart title (required if only CSV files are passed) (default: None)
  --arch ARCH      GPU arch name for the chart title (required if only CSV files are passed) (default: None)
  --filter FILTER  regex pattern of specializations to include (default: None)
  --absolute       perform absolute instead of relative comparison (default: False)
```

</details>

<details>
<summary>Command for graphing rocPRIM</summary>

```sh
python3 -m pip install pygal pandas rich_argparse scipy && \
for json in hip/results/json/*.json; do
  b=$(basename "$json" .json)
  python3 grapher.py \
    --output "graphs/$b.svg" \
    "hip/results/json/$b.json" \
    "cuda/results/json/$b.json"
done
```

</details>

<details>
<summary>Command for graphing rocRAND</summary>

```sh
apt update && \
apt install -y python3-pip && \
python3 -m pip install --break-system-packages pygal pandas rich_argparse scipy && \
for json in hip/*.json; do
  b=$(basename "$json" .json)
  python3 grapher.py \
    --output "graphs/$b.svg" \
    "hip/$b.json" \
    "cuda/$b.json"
done
```

</details>

<details>
<summary>Command for graphing hipCUB</summary>

```sh
export gfx=$(rocm_agent_enumerator 2>/dev/null | head -n1) && \
if [ -z "$gfx" ]; then
  gfx=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | tr -d '.')
fi && \
python3 -m pip install --break-system-packages pygal pandas rich_argparse scipy && \
for csv in converted_hip/*.csv; do
  b=$(basename "$csv" .csv)
  algo="${b#benchmark_}"
  algo="${algo%_gfx*}"
  python3 grapher.py \
    --output "graphs/$b.svg" \
    --algo "$algo" \
    --arch "$gfx" \
    "converted_hip/$b.csv" \
    "converted_cuda/$b.csv"
done
```

</details>

<details>
<summary>Command for graphing rocThrust</summary>

```sh
export gfx=$(rocm_agent_enumerator 2>/dev/null | head -n1) && \
if [ -z "$gfx" ]; then
  gfx=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | tr -d '.')
fi && \
apt update && \
apt install -y python3-pip && \
python3 -m pip install --break-system-packages pygal pandas rich_argparse scipy && \
for csv in converted_hip/*.csv; do
  b=$(basename "$csv" .csv)
  algo="${b#benchmark_}"
  algo="${algo%_gfx*}"
  python3 grapher.py \
    --output "graphs/$b.svg" \
    --algo "$algo" \
    --arch "$gfx" \
    "converted_hip/$b.csv" \
    "converted_cuda/$b.csv"
done
```

</details>

<details>
<summary><code>grapher.py</code></summary>

```py
#!/usr/bin/env python3
"""
Grapher for primbench results: takes JSON and CSV files containing benchmark specialization
data and generates a comparison graph showing performance metrics across all
specializations. By default, shows relative comparison (percentage change). The graph displays
specializations on the Y-axis with their throughput changes, and bytes/second (or percentage)
on the X-axis. The legend lists all input files used.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pygal  # pyright: ignore[reportMissingTypeStubs]
from pandas import DataFrame, read_csv
from pygal.style import DefaultStyle  # pyright: ignore[reportMissingTypeStubs]
from rich_argparse import ArgumentDefaultsRichHelpFormatter
from scipy import stats


def print_performance_stats(df: DataFrame, algo: str):
    """Calculates and prints the 'Good vs Bad' metrics with a 3% noise threshold."""
    old_column, new_column = df.columns[0], df.columns[-1]
    threshold = 3.0  # 3% noise threshold

    new = df[new_column]
    old = df[old_column]
    changes = ((new / old) - 1) * 100

    # Basic stats
    mean_change = np.mean(changes)
    median_change = np.median(changes)

    # Win rate (percentage of cases that improved)
    wins = np.sum(changes > 0)
    win_rate = (wins / len(changes)) * 100

    # 95% Confidence Interval
    conf_int = stats.t.interval(
        0.95, len(changes) - 1, loc=mean_change, scale=stats.sem(changes)
    )

    print("=" * 40)
    print(f"Algorithm:           {algo}")
    print(f"Mean Change:         {mean_change:+.2f}%")
    print(f"Median Change:       {median_change:+.2f}%")
    print(f"95% Conf. Int:       [{conf_int[0]:+.2f}%, {conf_int[1]:+.2f}%]")
    print(
        f"Improvement Rate:    {win_rate:.1f}% ({wins}/{len(changes)} specializations)"
    )

    # Verdict Logic with 3% threshold
    if conf_int[0] > threshold:
        print(f"Verdict:             SIGNIFICANT IMPROVEMENT (>{threshold}%)")
    elif conf_int[1] < -threshold:
        print(f"Verdict:             SIGNIFICANT REGRESSION (< -{threshold}%)")
    else:
        # If the confidence interval overlaps with the threshold, or the mean is inside it
        direction = "POSITIVE" if mean_change > 0 else "NEGATIVE"
        print(f"Verdict:             INSIGNIFICANT CHANGE (Trending {direction})")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsRichHelpFormatter)
    parser.add_argument(
        "input_files",
        nargs="+",
        type=str,
        help="paths to input .json or .csv primbench files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to output .svg graph",
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="algorithm name for the chart title (required if only CSV files are passed)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        help="GPU arch name for the chart title (required if only CSV files are passed)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="regex pattern of specializations to include",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="perform absolute instead of relative comparison",
    )
    args = parser.parse_args()

    # Validate input files
    has_json = False
    for input_file in args.input_files:
        ext = Path(input_file).suffix.lower()
        if ext not in [".json", ".csv"]:
            raise ValueError(f"Input file must be .json or .csv, got: {input_file}")
        if ext == ".json":
            has_json = True

    # Validate output file
    output_ext = Path(args.output).suffix.lower()
    if output_ext != ".svg":
        raise ValueError(f"Output file must be a .svg, got: {args.output}")

    # Determine algorithm and arch for title
    algo: Optional[str] = None
    arch: Optional[str] = None
    if has_json:
        # Extract from the first JSON file's context
        for input_file in args.input_files:
            if Path(input_file).suffix.lower() == ".json":
                with open(input_file) as f:
                    data = json.load(f)
                algo = data["context"]["general"]["algorithm"]
                arch = data["context"]["general"]["gpu"]["arch"]
                break
    else:
        # Only CSV files, so algo and arch must be provided
        if not args.algo:
            raise ValueError(
                "--algo must be passed on the command line when only CSV files are passed"
            )
        if not args.arch:
            raise ValueError(
                "--arch must be passed on the command line when only CSV files are passed"
            )
        algo = args.algo
        arch = args.arch
    assert algo
    assert arch

    # Parse all input files
    dfs: List[Tuple[str, DataFrame]] = []
    for input_file in args.input_files:
        df = parse_primbench_file(input_file)
        check_duplicates(df, input_file)
        dfs.append((input_file, df))

    # Merge all dataframes
    merged_df = merge_dataframes(dfs)

    # Apply filter if provided
    if args.filter:
        merged_df = apply_filter(merged_df, args.filter)

    # Print stats before generating chart
    print_performance_stats(merged_df, algo)

    # Generate chart
    chart = get_chart(merged_df, algo, arch, args.absolute)

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    chart.render_to_file(args.output)

    print(f"Graph saved to {args.output}")


def parse_primbench_file(file_path: str):
    """Parse a primbench JSON or CSV file and return a DataFrame."""
    ext = Path(file_path).suffix.lower()

    if ext == ".json":
        return parse_json(file_path)

    return parse_csv(file_path)


def parse_json(json_path: str):
    """Parse primbench JSON file into a DataFrame."""
    with open(json_path) as f:
        data = json.load(f)

    specializations = data["specializations"]

    rows: List[Dict[str, Union[str, float]]] = []
    for spec in specializations:
        rows.append(
            {
                "name": spec["name"],
                "bytes_per_second": spec["bytes_per_second"],
                "file": json_path,
            }
        )

    return DataFrame(rows)


def parse_csv(csv_path: str):
    """Parse primbench CSV file into a DataFrame."""
    df = read_csv(csv_path)

    # Ensure required columns exist
    required_cols = ["name", "bytes_per_second"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    df["file"] = csv_path
    return df.loc[:, ["name", "bytes_per_second", "file"]]


def check_duplicates(df: DataFrame, file_path: str):
    """Check for duplicate 'name' entries in a DataFrame and exit if any found."""
    duplicated_mask = df.duplicated(subset=["name"], keep=False)
    if duplicated_mask.any():
        duplicate_names = df.loc[duplicated_mask, "name"]
        first_duplicate = duplicate_names.iloc[0]
        exit(
            f"ERROR: duplicate parameter name found in {file_path}:\n"
            f"  {first_duplicate}"
        )


def merge_dataframes(dfs: List[Tuple[str, DataFrame]]):
    """Merge multiple DataFrames from different files."""
    merged: Optional[DataFrame] = None

    for file_path, df in dfs:
        current_df = df.copy()
        current_df["index_label"] = current_df["name"]
        current_df = current_df.set_index("index_label")
        current_df.index.name = None

        # Use filename as column name for bytes_per_second values
        renamed_df = current_df[["bytes_per_second"]]
        renamed_df.columns = [file_path]
        current_df = cast(DataFrame, renamed_df)

        if merged is None:
            merged = current_df
        else:
            merged = merged.merge(
                current_df, how="outer", left_index=True, right_index=True
            )

            nan_rows = merged[merged.isna().any(axis=1)]
            if not nan_rows.empty:
                key = nan_rows.index[0]

                row = merged.loc[[key]]

                missing_cols = row.columns[row.isna().any(axis=0)].tolist()
                present_cols = row.columns[row.notna().all(axis=0)].tolist()

                exit(
                    f"ERROR: key mismatch while merging files:\n"
                    f"  Key: {key}\n"
                    f"  Present in: {present_cols[0] if present_cols else 'NONE'}\n"
                    f"  Missing from: {missing_cols[0] if missing_cols else 'NONE'}"
                )

    assert merged is not None
    return merged.sort_index()


def apply_filter(df: DataFrame, pattern: str):
    """Filter DataFrame rows based on regex pattern matching specialization names."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    # Keep rows where the index (specialization name) matches the pattern
    mask = df.index.str.contains(regex)
    filtered_df = cast(DataFrame, df[mask])

    if filtered_df.empty:
        raise ValueError(f"No specializations matched the filter pattern: {pattern}")

    return filtered_df


def get_chart(df: DataFrame, algo: str, arch: str, absolute: bool):
    """Generate a pygal HorizontalBar chart from the merged DataFrame."""

    old_column, new_column = df.columns[0], df.columns[-1]
    old_series = df[old_column]
    new_series = df[new_column]

    percent_change = ((new_series / old_series) - 1) * 100

    # Sort DataFrame and percent_change by percent_change
    sort_order = percent_change.sort_values(ascending=False).index
    df = df.loc[sort_order]
    percent_change = percent_change.loc[sort_order]

    style = DefaultStyle(label_font_size=11)

    if absolute:
        # Absolute mode: show actual bytes/sec values
        data_df = df
        chart = pygal.HorizontalBar(
            style=style,
            x_title="Throughput (bytes/sec)",
            value_formatter=lambda x: f"{x:.2e}",
        )
        bar_count = len(df.columns)

        # Add all series normally
        for column in data_df.columns:
            chart.add(column, data_df[column].tolist())
    else:
        # Relative mode: show percentage change relative to baseline
        baseline = df.iloc[:, 0]

        relative_series = (new_series / baseline - 1) * 100

        data_df = relative_series.to_frame(name=f"{old_column} vs {new_column}")

        chart = pygal.HorizontalBar(
            # The "black" here sets the legend dot's color
            style=DefaultStyle(label_font_size=11, colors=["black"]),
            x_title="Throughput change (%)",
            value_formatter=lambda x: f"{x:+.1f}%" if x != 0 else "0%",
        )
        bar_count = 1

        # Retrieve Pygal's default colors from the style
        negative_color = style.colors[0]
        positive_color = style.colors[1]

        # Add bars with tooltips showing all bytes/sec
        bars_with_tooltip: List[Dict[str, Union[str, float]]] = []
        for _, row in df.iterrows():
            val = float(((row[new_column] / row[old_column]) - 1) * 100)

            color = positive_color if val >= 0 else negative_color

            tooltip_text = ", ".join([f"{col}: {row[col]:.2e}" for col in df.columns])

            bars_with_tooltip.append(
                {"value": val, "color": color, "label": tooltip_text}
            )

        chart.add(data_df.columns[0], bars_with_tooltip)

    # Let x_labels include percent change
    chart.x_labels = [
        f"{label}, {change:+.1f}%" for label, change in zip(df.index, percent_change)
    ]

    # Title with worst/best percent change
    chart.title = f"{algo} {arch} (worst {percent_change.min():+.1f}%, best {percent_change.max():+.1f}%)"

    # Common styling
    longest_label = max(df.index, key=len)
    chart.width = 15 * len(longest_label) + 500
    chart.height = 15 * len(df.index) * bar_count + 200
    chart.legend_at_bottom = True
    chart.truncate_legend = False
    chart.legend_at_bottom_columns = 1

    return chart


if __name__ == "__main__":
    main()
```

</details>

## Using print_noisy.py

The script takes a directory of primbench CSV files, and prints the names of any files that contain noisy specializations.

<details><summary><code>--help</code> options</summary>

```
usage: print_noisy.py [-h] csv_dir

Summarize noise from CSV files

positional arguments:
  csv_dir     Path to directory containing CSV files

options:
  -h, --help  show this help message and exit
```

</details>

<details>
<summary>Command for printing noise</summary>

```sh
python3 print_noisy.py csv_dir
```

</details>

<details>
<summary><code>print_noisy.py</code></summary>

```py
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize noise from CSV files")
    parser.add_argument(
        "csv_dir",
        type=Path,
        help="Path to directory containing CSV files",
    )
    args = parser.parse_args()

    csv_dir = args.csv_dir

    if not csv_dir.is_dir():
        raise ValueError(f"Not a directory: {csv_dir}")

    results = []

    for filepath in csv_dir.iterdir():
        if filepath.suffix != ".csv":
            continue

        noise = 0

        with filepath.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row["noise_timeout"]):
                    noise = max(noise, float(row["noise_percent"]))

        results.append({"noise": noise, "name": filepath.name})

    results.sort(key=lambda x: x["noise"], reverse=True)

    for result in results:
        if result["noise"] == 0:
            continue
        print(f"{result['noise']:.1f}%: {result['name']}")

    print("")


if __name__ == "__main__":
    main()
```

</details>

## Using print_throughput_changes.py

The script takes an old directory of primbench CSV files, and a new directory of primbench CSV files, and prints the specialization that had the largest throughput change for every CSV file.

By default the script prints the specializations that had the largest regressions, but by passing `--improvements` it instead prints the specializations that had the largest improvements.

<details><summary><code>--help</code> options</summary>

```
usage: print_throughput_changes.py [-h] [--improvements] old_csv_dir new_csv_dir

positional arguments:
  old_csv_dir     directory containing old primbench CSV files
  new_csv_dir     directory containing new primbench CSV files

options:
  -h, --help      show this help message and exit
  --improvements  print improvements instead of regressions
```

</details>

<details>
<summary>Command for printing throughput changes</summary>

```sh
python3 -m pip install pandas && \
python3 print_throughput_changes.py old_csv_dir new_csv_dir
```

</details>

<details>
<summary><code>print_throughput_changes.py</code></summary>

```py
#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from pandas import DataFrame, read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "old_csv_dir",
        type=Path,
        help="directory containing old primbench CSV files",
    )
    parser.add_argument(
        "new_csv_dir",
        type=Path,
        help="directory containing new primbench CSV files",
    )
    parser.add_argument(
        "--improvements",
        action="store_true",
        help="print improvements instead of regressions",
    )
    args = parser.parse_args()

    if not args.old_csv_dir.is_dir():
        raise ValueError(f"Not a directory: {args.old_csv_dir}")
    if not args.new_csv_dir.is_dir():
        raise ValueError(f"Not a directory: {args.new_csv_dir}")

    old_csvs = {p.name: p for p in args.old_csv_dir.glob("*.csv")}
    new_csvs = {p.name: p for p in args.new_csv_dir.glob("*.csv")}

    common_files = sorted(old_csvs.keys() & new_csvs.keys())

    if not common_files:
        raise SystemExit("ERROR: no matching CSV filenames found between directories")

    results: List[Dict[str, Any]] = []

    for filename in common_files:
        old_path = old_csvs[filename]
        new_path = new_csvs[filename]

        dfs = [
            (str(old_path), parse_csv(str(old_path))),
            (str(new_path), parse_csv(str(new_path))),
        ]

        df = merge_dataframes(dfs)

        old_column, new_column = df.columns[0], df.columns[-1]

        percent_change = ((df[new_column] / df[old_column]) - 1) * 100

        min_idx = percent_change.idxmin()
        max_idx = percent_change.idxmax()

        min_change = percent_change.loc[min_idx]
        max_change = percent_change.loc[max_idx]

        algo = filename.removeprefix("benchmark_").removesuffix(".csv")
        algo = re.sub(r"_gfx\d+$", "", algo)

        results.append(
            {
                "algo": algo,
                "min": min_change,
                "max": max_change,
                "min_spec": min_idx,
                "max_spec": max_idx,
            }
        )

    if args.improvements:
        results.sort(key=lambda x: x["max"], reverse=True)
    else:
        results.sort(key=lambda x: x["min"])

    for result in results:
        percent = result["max"] if args.improvements else result["min"]
        print(f"{percent:+.1f}%: {result['algo']}")

        specialization = result["max_spec"] if args.improvements else result["min_spec"]
        print(f"    {specialization}")


def parse_csv(csv_path: str) -> DataFrame:
    """Parse primbench CSV file into a DataFrame."""
    df = read_csv(csv_path)
    df["file"] = csv_path
    return cast(DataFrame, df[["name", "bytes_per_second", "file"]])


def merge_dataframes(dfs: List[Tuple[str, DataFrame]]) -> DataFrame:
    """Merge multiple DataFrames from different files."""
    merged: Optional[DataFrame] = None

    for file_path, df in dfs:
        df = df.copy()
        df["index_label"] = df["name"]
        df = df.set_index("index_label")
        df.index.name = None

        subset = cast(DataFrame, df[["bytes_per_second"]])
        df = subset.rename(columns={"bytes_per_second": file_path})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, how="outer", left_index=True, right_index=True)

            nan_rows = merged[merged.isna().any(axis=1)]
            if not nan_rows.empty:
                key = nan_rows.index[0]
                row = merged.loc[[key]]

                missing_cols = row.columns[row.isna().any(axis=0)].tolist()
                present_cols = row.columns[row.notna().all(axis=0)].tolist()

                raise SystemExit(
                    f"ERROR: key mismatch while merging files:\n"
                    f"  Key: {key}\n"
                    f"  Present in: {present_cols[0] if present_cols else 'NONE'}\n"
                    f"  Missing from: {missing_cols[0] if missing_cols else 'NONE'}"
                )

    assert merged is not None
    return merged.sort_index()


if __name__ == "__main__":
    main()
```

</details>