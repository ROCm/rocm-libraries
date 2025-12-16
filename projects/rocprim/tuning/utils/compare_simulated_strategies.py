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

import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This script performs comparative analysis of different optimization strategies for kernel tuning simulations.
It is designed to work with Kernel Tuner output data to evaluate and visualize the effectiveness of
various tuning strategies.

Key Features:
- Compares multiple simulated tuning strategies against actual best-known performances
- Calculates relative performance metrics (as percentages) for each strategy
- Generates detailed performance statistics (mean, min, max, median, stddev)
- Creates a visual heatmap representation of strategy performances

Input Requirements:
- Simulated results should be in '../simulated_output/<strategy_name>/' directories
- Actual results should be in '../output/' directory
- JSON files must follow naming convention: containing both algorithm and architecture names
- Each JSON file should follow the JSON formats of Kernel Tuner

Output Formats:
- Console output: Detailed performance statistics for each strategy
- PNG file: Heatmap visualization showing relative performance across all configurations
- Heatmap naming: 'optimization_strategies_heatmap_<algo>_<arch>.png'

Performance Metrics:
- Values are calculated as: (best_known_time / strategy_time) * 100
- Higher percentages indicate better performance (closer to optimal solutions)
- 100% means the strategy found the optimal solution
- Values below 100% indicate suboptimal solutions

Usage:
python3 script.py --arch <architecture> --algo <algorithm>
Example: python3 script.py --arch gfx90a --algo device_merge
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare optimization strategies for specific architecture and algorithm."
    )
    parser.add_argument(
        "--arch", type=str, required=True, help="Architecture name (e.g., gfx90a)"
    )
    parser.add_argument(
        "--algo", type=str, required=True, help="Algorithm name (e.g., device_merge)"
    )
    return parser.parse_args()


def is_matching_file(filename, algo, arch):
    """
    Check if the JSON file contains both the algorithm name and architecture name.
    """
    return algo in filename and arch in filename and filename.endswith("_cache.json")


def find_best_time_from_cache(data):
    """
    Find the best (lowest) time from results or cache entries.
    """
    times = []

    # For simulated results structure
    if "results" in data:
        for result in data["results"]:
            for measurement in result.get("measurements", []):
                if measurement["name"] == "time" and measurement["unit"] == "ms":
                    time = measurement["value"]
                    if isinstance(time, (int, float)) and time > 0:
                        times.append(time)

    # For output results structure with cache
    else:
        cache = data.get("cache", {})
        for entry_key, entry_data in cache.items():
            time = entry_data.get("time", float("inf"))
            if isinstance(time, (int, float)) and time > 0:
                times.append(time)

    return min(times) if times else float("inf")


def load_json_files(simulated_output_dir, output_dir, algo, arch):
    """
    Load and process JSON files from both simulated and actual output directories.
    Returns best times for each kernel configuration.
    """
    results = {"simulated": {}, "best_times": {}}

    # First, process actual output files to get best times
    for json_file in os.listdir(output_dir):
        if json_file.endswith(".json") and is_matching_file(json_file, algo, arch):
            with open(os.path.join(output_dir, json_file), "r") as f:
                data = json.load(f)
                best_time = find_best_time_from_cache(data)
                results["best_times"][json_file] = best_time

    # Process simulated output directories (different strategies)
    for strategy in os.listdir(simulated_output_dir):
        strategy_path = os.path.join(simulated_output_dir, strategy)
        if os.path.isdir(strategy_path):
            results["simulated"][strategy] = {}
            for json_file in os.listdir(strategy_path):
                if json_file.endswith(".json") and is_matching_file(
                    json_file, algo, arch
                ):
                    with open(os.path.join(strategy_path, json_file), "r") as f:
                        data = json.load(f)
                        best_time = find_best_time_from_cache(data)
                        results["simulated"][strategy][json_file] = best_time

    if not results["best_times"]:
        raise ValueError(
            f"No matching files found for algorithm '{algo}' and architecture '{arch}' in {output_dir}"
        )

    if not any(results["simulated"].values()):
        raise ValueError(
            f"No matching files found in any strategy directory for algorithm '{algo}' and architecture '{arch}'"
        )

    return results


def calculate_performance_stats(results, algo, arch):
    """
    Calculate and print performance statistics for each strategy.
    Returns the DataFrame for the heatmap.
    """
    strategies = sorted(list(results["simulated"].keys()))
    kernels = sorted(list(results["best_times"].keys()))

    data = []
    for strategy in strategies:
        row = []
        for kernel in kernels:
            strategy_time = results["simulated"][strategy][kernel]
            best_time = results["best_times"][kernel]
            relative_perf = (best_time / strategy_time) * 100
            row.append(relative_perf)
        data.append(row)

    df = pd.DataFrame(data, index=strategies, columns=kernels)

    # Calculate summary statistics
    summary_stats = pd.DataFrame(
        {
            "Mean": df.mean(axis=1).round(1),
            "Min": df.min(axis=1).round(1),
            "Max": df.max(axis=1).round(1),
            "Median": df.median(axis=1).round(1),
            "Std Dev": df.std(axis=1).round(1),
        }
    ).sort_values("Mean", ascending=False)

    # Print performance summary
    print(f"\nOptimization Strategy Performance Summary for {algo} on {arch}")
    print("=" * 80)
    print("\nPerformance statistics (%):")
    print(summary_stats.to_string())
    print("\nNote: Higher values indicate better performance")
    print("=" * 80)

    return df


def create_performance_heatmap(df, output_filename, algo, arch):
    """
    Create a heatmap comparing different strategies' performance.
    """
    # Create figure
    plt.figure(figsize=(20, 8))

    # Sort DataFrame rows by mean performance (descending)
    df = df.reindex(df.mean(axis=1).sort_values(ascending=False).index)

    # Normalize kernel names for better display
    df.columns = [
        col.replace(f"{algo}_", "").replace("_cache.json", "") for col in df.columns
    ]

    min_val = df.min().min()
    max_val = df.max().max()

    # Create heatmap with adjusted parameters
    sns.heatmap(
        df,
        annot=True,  # Show numbers in cells
        fmt=".1f",  # Format numbers to 1 decimal place
        cmap="RdYlGn",  # Red (bad/low) to Green (good/high)
        vmin=min_val,  # Use actual minimum value
        vmax=max_val,  # Use actual maximum value
        center=None,  # Remove center to maximize color spread
        robust=True,  # Use robust quantiles for color scaling
        square=True,  # Make cells square
        cbar_kws={"label": "Performance relative to best (%)"},
    )

    plt.title(
        f"Optimization Strategies Performance for {algo} on {arch}\nValues range: {min_val:.1f}% - {max_val:.1f}%"
    )
    plt.xlabel("Kernel Configurations")
    plt.ylabel("Optimization Strategies")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_arguments()

    base_dir = "."
    simulated_output_dir = os.path.join(base_dir, "../simulated_output")
    output_dir = os.path.join(base_dir, "../output")

    results = load_json_files(simulated_output_dir, output_dir, args.algo, args.arch)

    # Calculate and print performance statistics
    df = calculate_performance_stats(results, args.algo, args.arch)

    # Create heatmap
    output_filename = f"optimization_strategies_heatmap_{args.algo}_{args.arch}.png"
    create_performance_heatmap(df, output_filename, args.algo, args.arch)


if __name__ == "__main__":
    main()
