#!/usr/bin/env python3

# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
import statistics
import sys


class colors:
    OK = "\033[92m"
    FAIL = "\033[91m"
    END_COLOR = "\033[0m"


def print_results(results):
    # Store the length of the longest value in a column
    longest = {
        "name": max(len(result["name"]) for result in results),
        "noisy_permutations": max(
            len(result["noisy_permutations"]) for result in results
        ),
        "mean": max(len(result["mean"]) for result in results),
        "median": max(len(result["median"]) for result in results),
        "max": max(len(result["max"]) for result in results),
        "batch": max(len(result["batch"]) for result in results),
        "warmup": max(len(result["warmup"]) for result in results),
        "bytes": max(len(result["bytes"]) for result in results),
    }

    # The name of a column can be longer than its values
    longest = {key: max(value, len(key)) for key, value in longest.items()}

    printed = "name".ljust(longest["name"] + 1)
    printed += "noisy permutations".ljust(longest["noisy_permutations"] + 1)
    printed += "mean".ljust(longest["mean"] + 1)
    printed += "median".ljust(longest["median"] + 1)
    printed += "max".ljust(longest["max"] + 1)
    printed += "batch".ljust(longest["batch"] + 1)
    printed += "warmup".ljust(longest["warmup"] + 1)
    printed += "bytes".ljust(longest["bytes"] + 1)
    printed += "seed"
    print(printed)

    for result in results:
        printed = result["name"].ljust(longest["name"])

        printed += " "
        printed += colors.FAIL if result["noisy"] else colors.OK
        printed += (
            f'{result["noisy_permutations"].ljust(longest["noisy_permutations"])}'
        )
        printed += colors.END_COLOR

        printed += " "
        printed += colors.FAIL if result["bad_mean"] else colors.OK
        printed += result["mean"].ljust(longest["mean"])
        printed += colors.END_COLOR

        printed += " "
        printed += colors.FAIL if result["bad_median"] else colors.OK
        printed += result["median"].ljust(longest["median"])
        printed += colors.END_COLOR

        printed += " "
        printed += colors.FAIL if result["bad_max"] else colors.OK
        printed += result["max"].ljust(longest["max"])
        printed += colors.END_COLOR

        printed += " "
        printed += colors.FAIL if result["bad_batch"] else colors.OK
        printed += result["batch"].ljust(longest["batch"])
        printed += colors.END_COLOR

        printed += " "
        printed += colors.FAIL if result["bad_warmup"] else colors.OK
        printed += result["warmup"].ljust(longest["warmup"])
        printed += colors.END_COLOR

        printed += " "
        printed += colors.FAIL if result["bad_bytes"] else colors.OK
        printed += result["bytes"].ljust(longest["bytes"])
        printed += colors.END_COLOR

        printed += " "
        printed += colors.FAIL if result["seed"] == "random" else colors.OK
        printed += result["seed"]
        printed += colors.END_COLOR

        print(printed)


def get_results(benchmarks, threshold):
    def get_humanized_bytes(size):
        for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
            if size < 1024.0 or unit == "PiB":
                break
            size /= 1024.0
        return f"{size:.1f} {unit}"

    success = True

    results = []

    for benchmark in benchmarks:
        data = benchmark["data"]

        name = benchmark["name"]

        permutations = data["benchmarks"]

        cvs = [permutation["cv"] for permutation in permutations]

        # The cv (coefficient of variation) is a standard way of quantifying noise
        noises = sum(cv * 100 > threshold for cv in cvs)
        noisy = noises > 0

        if noisy:
            success = False

        context = data["context"]

        noisy_permutations = f"{noises}/{len(permutations)}"

        mean = statistics.mean(cvs)
        median = statistics.median(cvs)
        max_ = max(cvs)

        batch = context["batch_iterations"]
        warmup = context["warmup_iterations"]

        bytes_ = int(context["size"])
        seed = context["seed"]

        results.append(
            {
                "name": name,
                "noisy": noisy,
                "noisy_permutations": noisy_permutations,
                "bad_mean": mean * 100 > threshold,
                "mean": f"{mean:.1%}",
                "bad_median": median * 100 > threshold,
                "median": f"{median:.1%}",
                "bad_max": max_ * 100 > threshold,
                "max": f"{max_:.1%}",
                "bad_batch": int(batch) < 10,
                "batch": batch,
                "bad_warmup": int(warmup) < 5,
                "warmup": warmup,
                "bad_bytes": 0 < bytes_ < 128 * 1024 * 1024,  # 128 MiB
                "bytes": get_humanized_bytes(int(context["size"])),
                "seed": seed,
            }
        )

    return results, success


def load_benchmarks(benchmark_json_dir):
    def is_benchmark_json(filename):
        if not re.match(r".*\.json$", filename):
            return False
        path = os.path.join(benchmark_json_dir, filename)
        st_mode = os.stat(path).st_mode

        # we are not interested in permissions, just whether it is a regular file (S_IFREG)
        return st_mode & stat.S_IFREG

    benchmark_names = [
        name for name in os.listdir(benchmark_json_dir) if is_benchmark_json(name)
    ]

    success = True
    benchmarks = []
    for benchmark_name in benchmark_names:
        with open(os.path.join(benchmark_json_dir, benchmark_name)) as f:
            try:
                benchmarks.append({"name": benchmark_name, "data": json.load(f)})
            except json.JSONDecodeError as e:
                print(
                    f"{colors.FAIL}Failed to load {benchmark_name}{colors.END_COLOR}: {e}\n",
                    file=sys.stderr,
                )
                success = False

    return benchmarks, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise_threshold_percentage",
        help="The noise threshold percentage, past which benchmark permutations are considered to be too noisy",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--benchmark_json_dir",
        help="The directory of benchmark JSON files, which to report the noise of",
        required=True,
    )
    parser.add_argument(
        "--accept_high_noise",
        help="Don't call exit(1) when there is a noisy benchmark permutation",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    print(f"The noise threshold is {args.noise_threshold_percentage:.1f}%\n")

    benchmarks, load_success = load_benchmarks(args.benchmark_json_dir)
    results, results_success = get_results(benchmarks, args.noise_threshold_percentage)

    print_results(results)

    if not load_success:
        return False
    if args.accept_high_noise:
        return True
    return results_success


if __name__ == "__main__":
    success = main()
    if success:
        exit(0)
    else:
        exit(1)
