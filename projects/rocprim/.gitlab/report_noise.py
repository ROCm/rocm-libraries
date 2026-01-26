#!/usr/bin/env python3

# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
        "noisy_specializations": max(
            len(result["noisy_specializations"]) for result in results
        ),
        "mean": max(len(result["mean"]) for result in results),
        "median": max(len(result["median"]) for result in results),
        "max": max(len(result["max"]) for result in results),
        "bytes": max(len(result["bytes"]) for result in results),
    }

    # The name of a column can be longer than its values
    longest = {key: max(value, len(key)) for key, value in longest.items()}

    printed = "name".ljust(longest["name"] + 2)
    printed += "noisy specializations".ljust(longest["noisy_specializations"] + 2)
    printed += "mean".ljust(longest["mean"] + 2)
    printed += "median".ljust(longest["median"] + 2)
    printed += "max".ljust(longest["max"] + 2)
    printed += "bytes".ljust(longest["bytes"] + 2)
    print(printed)

    for result in results:
        printed = result["name"].ljust(longest["name"])

        printed += "  "
        printed += colors.FAIL if result["noisy"] else colors.OK
        printed += (
            f'{result["noisy_specializations"].ljust(longest["noisy_specializations"])}'
        )
        printed += colors.END_COLOR

        printed += "  "
        printed += colors.FAIL if result["bad_mean"] else colors.OK
        printed += result["mean"].ljust(longest["mean"])
        printed += colors.END_COLOR

        printed += "  "
        printed += colors.FAIL if result["bad_median"] else colors.OK
        printed += result["median"].ljust(longest["median"])
        printed += colors.END_COLOR

        printed += "  "
        printed += colors.FAIL if result["bad_max"] else colors.OK
        printed += result["max"].ljust(longest["max"])
        printed += colors.END_COLOR

        printed += "  "
        printed += colors.FAIL if result["bad_bytes"] else colors.OK
        printed += result["bytes"].ljust(longest["bytes"])
        printed += colors.END_COLOR

        print(printed)


def get_results(benchmarks, threshold):
    def get_humanized_bytes(bytes):
        for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
            if bytes < 1024.0 or unit == "PiB":
                break
            bytes /= 1024.0
        return f"{bytes:.1f} {unit}"

    success = True

    results = []

    for benchmark in benchmarks:
        data = benchmark["data"]
        name = benchmark["name"]
        context = data["context"]
        cli_settings = context["cli_settings"]
        specializations = data["specializations"]

        noise_percents = [
            specialization["noise_percent"] for specialization in specializations
        ]

        num_noisy = sum(noise_percent > threshold for noise_percent in noise_percents)
        noisy = num_noisy > 0

        if noisy:
            success = False

        noisy_specializations = f"{num_noisy}/{len(specializations)}"

        mean = statistics.mean(noise_percents)
        median = statistics.median(noise_percents)
        max_ = max(noise_percents)

        bytes_ = int(cli_settings["bytes"])

        results.append(
            {
                "name": name,
                "noisy": noisy,
                "noisy_specializations": noisy_specializations,
                "bad_mean": mean > threshold,
                "mean": f"{mean:.1f}%",
                "bad_median": median > threshold,
                "median": f"{median:.1f}%",
                "bad_max": max_ > threshold,
                "max": f"{max_:.1f}%",
                "bad_bytes": 0 < bytes_ < 128 * 1024 * 1024,  # 128 MiB
                "bytes": get_humanized_bytes(int(cli_settings["bytes"])),
            }
        )

    return results, success


def load_benchmarks(json_dir):
    def is_benchmark_json(filename):
        if not re.match(r".*\.json$", filename):
            return False
        path = os.path.join(json_dir, filename)
        st_mode = os.stat(path).st_mode

        # we are not interested in permissions, just whether it is a regular file (S_IFREG)
        return st_mode & stat.S_IFREG

    benchmark_names = [name for name in os.listdir(json_dir) if is_benchmark_json(name)]

    success = True
    benchmarks = []
    for benchmark_name in benchmark_names:
        with open(os.path.join(json_dir, benchmark_name)) as f:
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
        "--json-dir",
        help="The directory of benchmark JSON files that will be analyzed for noise",
        required=True,
    )
    parser.add_argument(
        "--noise-threshold-percentage",
        help="The noise threshold percentage, past which benchmark specializations are considered to be too noisy",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--accept-high-noise",
        help="Don't call exit(1) when there is a noisy benchmark specialization",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    print(f"The noise threshold is {args.noise_threshold_percentage:.1f}%\n")

    benchmarks, load_success = load_benchmarks(args.json_dir)
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
