#!/bin/python3

import argparse
import csv
import os
from pathlib import Path


def parse_csv_file(file_path: Path):
    print(f"Parsing: {file_path}")
    with open(file_path, newline="") as f:
        lines = list(csv.DictReader(f))
        ds_only = [line for line in lines if "ds_" in line["Instruction"]]
        return ds_only[-1]["Latency"]


def parse_csvs_in_directory(directory):
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: {directory} is not a valid directory.")
        return

    stride_to_latency = {}

    csv_files = dir_path.glob("*.csv")
    for csv_file in csv_files:
        stride_to_latency[int(csv_file.stem)] = parse_csv_file(csv_file)

    for k, v in sorted(stride_to_latency.items()):
        print(f"{k:>5} {v:>5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="dir containing csv files")
    args = parser.parse_args()

    parse_csvs_in_directory(args.dir)


if __name__ == "__main__":
    main()
