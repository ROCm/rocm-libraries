#!/usr/bin/env python3

import argparse
import json


def filter_sarif(input_file, output_file, exclude_paths):
    with open(input_file, "r") as f:
        sarif = json.load(f)

    filtered_results = []
    for run in sarif["runs"]:
        for result in run["results"]:
            analyzed_file_path = result["locations"][0]["physicalLocation"][
                "artifactLocation"
            ]["uri"]
            if not any(analyzed_file_path.startswith(path) for path in exclude_paths):
                filtered_results.append(result)
            else:
                print(f"  excluding '{analyzed_file_path}' file.")
        run["results"] = filtered_results

    with open(output_file, "w") as f:
        json.dump(sarif, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter sarif file")

    parser.add_argument("input", type=str, help="Input sarif file")
    parser.add_argument(
        "exclude_paths_file",
        type=str,
        help="File with paths to exclude. One path per line",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output sarif file (default: <input>.filtered)"
    )

    args = parser.parse_args()

    if not args.output:
        args.output = args.input + ".filtered"

    print(f"Filtering sarif file '{args.input}' and saving it to '{args.output}' file.")

    with open(args.exclude_paths_file, "r") as f:
        exclude_paths = [line.strip() for line in f.readlines()]

    filter_sarif(args.input, args.output, exclude_paths)
