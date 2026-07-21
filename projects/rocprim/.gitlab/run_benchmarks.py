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
import json
import os
import re
import stat
import subprocess
import sys


def run_benchmarks(
    benchmark_executables_dir,
    gpu_architecture,
    json_out_dir,
    csv_out_dir,
    benchmark_executable_filter,
    skip_gathered,
    size,
    hot,
    seed,
    filter,
    dry,
    min_gpu_ms_per_batch,
    min_secs,
    noise_timeout_secs,
    batch_window_size,
    noise_tolerance_percent,
    min_gpu_temp,
    max_gpu_temp,
    max_warming_secs,
    max_cooling_secs,
    output_batches,
    spaces_per_indent,
    stream_blocking_timeout_secs,
):
    def is_benchmark_executable(filename):
        if not re.search(benchmark_executable_filter, filename):
            return False

        path = os.path.join(benchmark_executables_dir, filename)
        st_mode = os.stat(path).st_mode

        # Return True when any execution flag is set, AND it is a regular file.
        # Doesn't check permissions.
        return (st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)) and (
            st_mode & stat.S_IFREG
        )

    def should_skip(json_out_path):
        if not skip_gathered:
            return False

        try:
            with open(json_out_path) as f:
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        return True

    # Create output directories if they don't exist
    os.makedirs(json_out_dir, exist_ok=True)
    if csv_out_dir:
        os.makedirs(csv_out_dir, exist_ok=True)

    benchmark_names = [
        name
        for name in os.listdir(benchmark_executables_dir)
        if is_benchmark_executable(name)
    ]
    print(
        "The following benchmarks will be run:\n{}".format("\n".join(benchmark_names)),
        file=sys.stderr,
        flush=True,
    )

    success = True

    for benchmark_name in benchmark_names:
        results_json_name = f"{benchmark_name}_{gpu_architecture}.json"
        json_out_path = os.path.join(json_out_dir, results_json_name)

        if should_skip(json_out_path):
            print(
                f"Skipping {benchmark_name}, because its results have already been gathered at {json_out_path}",
                file=sys.stderr,
                flush=True,
            )
            continue

        benchmark_path = os.path.join(benchmark_executables_dir, benchmark_name)

        args = [benchmark_path]
        args += ["--json-out", json_out_path]

        if size:
            args += ["--size", size]
        if hot:
            args += ["--hot"]
        if seed:
            args += ["--seed", seed]
        if csv_out_dir:
            results_csv_name = f"{benchmark_name}_{gpu_architecture}.csv"
            csv_out_path = os.path.join(csv_out_dir, results_csv_name)
            args += ["--csv-out", csv_out_path]
        if filter:
            args += ["--filter", filter]
        if dry:
            args += ["--dry"]
        if min_gpu_ms_per_batch:
            args += ["--min-gpu-ms-per-batch", min_gpu_ms_per_batch]
        if min_secs:
            args += ["--min-secs", min_secs]
        if noise_timeout_secs:
            args += ["--noise-timeout-secs", noise_timeout_secs]
        if batch_window_size:
            args += ["--batch-window-size", batch_window_size]
        if noise_tolerance_percent:
            args += ["--noise-tolerance-percent", noise_tolerance_percent]
        if min_gpu_temp:
            args += ["--min-gpu-temp", min_gpu_temp]
        if max_gpu_temp:
            args += ["--max-gpu-temp", max_gpu_temp]
        if max_warming_secs:
            args += ["--max-warming-secs", max_warming_secs]
        if max_cooling_secs:
            args += ["--max-cooling-secs", max_cooling_secs]
        if output_batches:
            args += ["--output-batches"]
        if spaces_per_indent:
            args += ["--spaces-per-indent", spaces_per_indent]
        if stream_blocking_timeout_secs:
            args += ["--stream-blocking-timeout-secs", stream_blocking_timeout_secs]

        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError as error:
            print(
                f'Could not run benchmark at {benchmark_path}. Error: "{error}"',
                file=sys.stderr,
                flush=True,
            )
            success = False

    return success


def main():
    parser = argparse.ArgumentParser()

    # Python arguments
    parser.add_argument(
        "--benchmark-executables-dir",
        help="Benchmark executables directory",
        required=True,
    )
    parser.add_argument(
        "--gpu-architecture",
        help="Architecture of the currently enabled GPU",
        required=True,
    )
    parser.add_argument(
        "--json-out-dir",
        help="Directory to write JSON result files to, creating parent directories if necessary",
        required=True,
    )
    parser.add_argument(
        "--csv-out-dir",
        help="Directory to write CSV result files to, creating parent directories if necessary",
        required=False,
    )
    parser.add_argument(
        "--benchmark-executable-filter",
        help="Regular expression that controls the list of benchmark executables to run",
        default=r"^benchmark",
        required=False,
    )
    parser.add_argument(
        "--skip-gathered",
        help="Skip running benchmarks whose JSON file has already been gathered",
        default=False,
        action="store_true",
        required=False,
    )

    # primbench arguments
    parser.add_argument(
        "--size",
        help="Input size. Benchmarks decide what this represents, but it is commonly the number of bytes or items",
        default="",
        required=False,
    )
    parser.add_argument(
        "--hot",
        help="Skip clearing the GPU cache between batch iterations",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--seed", help="Seed used for input generation", default="", required=False
    )
    parser.add_argument(
        "--filter",
        help="Regex filter of specialization names to benchmark",
        default="",
        required=False,
    )
    parser.add_argument(
        "--dry",
        help="Perform a dry run. The benchmark setup is still run, and JSON and CSV files are still output, but state.run() immediately returns",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--min-gpu-ms-per-batch",
        help="Minimum duration of a batch in milliseconds (GPU time)",
        default="",
        required=False,
    )
    parser.add_argument(
        "--min-secs",
        help="Minimum total benchmark duration in seconds (wall time)",
        default="",
        required=False,
    )
    parser.add_argument(
        "--noise-timeout-secs",
        help="Maximum total benchmark duration in seconds before timing out a noisy run (wall time)",
        default="",
        required=False,
    )
    parser.add_argument(
        "--batch-window-size",
        help="Number of batch times used in the noise (coefficient of variation) window to decide early stopping",
        default="",
        required=False,
    )
    parser.add_argument(
        "--noise-tolerance-percent",
        help="Noise tolerance of batch times in percent, used to determine early benchmark stopping",
        default="",
        required=False,
    )
    parser.add_argument(
        "--min-gpu-temp",
        help="Minimum GPU temperature in °C. Too low slows benchmarks; too high increases noise",
        default="",
        required=False,
    )
    parser.add_argument(
        "--max-gpu-temp",
        help="Maximum GPU temperature in °C. Too low slows benchmarks; too high increases noise",
        default="",
        required=False,
    )
    parser.add_argument(
        "--max-warming-secs",
        help="Maximum seconds allowed for GPU warming before an error is thrown",
        default="",
        required=False,
    )
    parser.add_argument(
        "--max-cooling-secs",
        help="Maximum seconds allowed for GPU cooling before an error is thrown",
        default="",
        required=False,
    )
    parser.add_argument(
        "--output-batches",
        help="Output a `batches` array for each specialization, containing per-batch details",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--spaces-per-indent",
        help="Number of spaces per indentation level in JSON output. Set to 0 for no indentation",
        default="",
        required=False,
    )
    parser.add_argument(
        "--stream-blocking-timeout-secs",
        help="Maximum stream blocking duration in seconds before timing out. Stream is blocked while queueing kernel calls. Use `primbench::flags::sync` if kernel is synchronous",
        default="",
        required=False,
    )

    args = parser.parse_args()

    benchmark_run_successful = run_benchmarks(
        args.benchmark_executables_dir,
        args.gpu_architecture,
        args.json_out_dir,
        args.csv_out_dir,
        args.benchmark_executable_filter,
        args.skip_gathered,
        args.size,
        args.hot,
        args.seed,
        args.filter,
        args.dry,
        args.min_gpu_ms_per_batch,
        args.min_secs,
        args.noise_timeout_secs,
        args.batch_window_size,
        args.noise_tolerance_percent,
        args.min_gpu_temp,
        args.max_gpu_temp,
        args.max_warming_secs,
        args.max_cooling_secs,
        args.output_batches,
        args.spaces_per_indent,
        args.stream_blocking_timeout_secs,
    )

    return benchmark_run_successful


if __name__ == "__main__":
    success = main()
    if success:
        exit(0)
    else:
        exit(1)
