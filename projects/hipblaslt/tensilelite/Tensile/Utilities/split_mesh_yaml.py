# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split MeshBased YAML logic files into kernel YAML + companion .csv.gz table.

Usage:
    python split_mesh_yaml.py <directory_or_file> [...]

For each MeshBased YAML found, creates a companion .csv.gz containing the
ExactLogic table and rewrites the YAML with ExactLogic set to null.

Uses the same YAML loader and data layout as LibraryIO.parseLibraryLogicFile.
"""

import argparse
import csv
import gzip
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Tensile.CustomYamlLoader import load_yaml_stream
from Tensile.LibraryIO import StrictTypeLoader

import yaml


def _get_library_type(data):
    if isinstance(data, dict):
        return data.get("LibraryType"), "dict"
    if isinstance(data, list) and len(data) > 11 and data[11]:
        return data[11], "list"
    return None, None


def _get_exact_logic(data, fmt):
    if fmt == "dict":
        return data.get("ExactLogic")
    if fmt == "list":
        return data[7] if len(data) > 7 else None
    return None


def _clear_exact_logic(data, fmt):
    if fmt == "dict":
        data["ExactLogic"] = None
    elif fmt == "list":
        data[7] = None


def _write_csv_gz(exact_logic, out_path):
    with gzip.open(out_path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "N", "batch", "K", "solutionIdx"])
        for entry in exact_logic:
            key, val = entry[0], entry[1]
            writer.writerow([key[0], key[1], key[2], key[3], val[0]])


def process_file(yaml_path):
    print(f"Processing: {os.path.basename(yaml_path)}")
    orig_size = os.path.getsize(yaml_path)

    data = load_yaml_stream(yaml_path, StrictTypeLoader)

    lib_type, fmt = _get_library_type(data)
    if lib_type != "MeshBased":
        print("  Skipping (not MeshBased)")
        return

    exact_logic = _get_exact_logic(data, fmt)
    if not exact_logic:
        print("  Skipping (ExactLogic is empty/null)")
        return

    csv_path = yaml_path + ".csv.gz"
    _write_csv_gz(exact_logic, csv_path)

    _clear_exact_logic(data, fmt)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, width=200)

    new_size = os.path.getsize(yaml_path)
    csv_size = os.path.getsize(csv_path)
    print(f"  Original:  {orig_size:>12,} bytes")
    print(f"  New YAML:  {new_size:>12,} bytes")
    print(f"  CSV.gz:    {csv_size:>12,} bytes")
    print(f"  Entries:   {len(exact_logic):>12,}")
    print(f"  Savings:   {orig_size - new_size - csv_size:>12,} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="YAML files or directories to process")
    args = parser.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".yaml"):
                    files.append(os.path.join(p, name))
        elif os.path.isfile(p):
            files.append(p)

    for f in files:
        process_file(f)


if __name__ == "__main__":
    main()
