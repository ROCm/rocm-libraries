# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split MeshBased YAML logic files into kernel YAML + companion .csv.xz table.

Usage:
    python split_mesh_yaml.py <directory_or_file> [...]

For each MeshBased YAML found, creates a companion .csv.xz containing the
ExactLogic table and rewrites the YAML with ExactLogic set to null.

Uses the same YAML loader and data layout as LibraryIO.parseLibraryLogicFile.
"""

import argparse
import csv
import lzma
import os

import yaml

from ..CustomYamlLoader import load_yaml_stream
from ..LibraryIO import StrictTypeLoader, _loadMeshTableFromCsvXz


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


def _get_solution_indices(data, fmt):
    if fmt == "dict":
        return {s["SolutionIndex"] for s in data.get("Solutions", [])}
    if fmt == "list" and len(data) > 5 and isinstance(data[5], list):
        return {s["SolutionIndex"] for s in data[5] if isinstance(s, dict)}
    return set()


def _clear_exact_logic(data, fmt):
    if fmt == "dict":
        data["ExactLogic"] = None
    elif fmt == "list":
        data[7] = None


def _write_csv_xz(exact_logic, out_path):
    for i, entry in enumerate(exact_logic):
        if len(entry[0]) != 4:
            raise ValueError(
                f"entry {i}: expected 4-element key, got {len(entry[0])}"
            )
        if len(entry[1]) < 2:
            raise ValueError(
                f"entry {i}: expected 2-element value, got {len(entry[1])}"
            )
    hasSpeed = any(entry[1][1] != 0.0 for entry in exact_logic)
    sorted_logic = sorted(exact_logic, key=lambda e: (e[0][0], e[0][1], e[0][3]))
    with lzma.open(out_path, "wt", newline="") as f:
        writer = csv.writer(f)
        header = ["M", "N", "batch", "K", "solutionIdx"]
        if hasSpeed:
            header.append("speed")
        writer.writerow(header)
        for entry in sorted_logic:
            key, val = entry[0], entry[1]
            row = [key[0], key[1], key[2], key[3], val[0]]
            if hasSpeed:
                row.append(val[1])
            writer.writerow(row)


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

    valid_indices = _get_solution_indices(data, fmt)
    if valid_indices:
        csv_indices = {entry[1][0] for entry in exact_logic}
        invalid = csv_indices - valid_indices
        if invalid:
            raise ValueError(
                f"{os.path.basename(yaml_path)}: ExactLogic references "
                f"{len(invalid)} solution index(es) not in Solutions: "
                f"{sorted(invalid)[:20]}"
            )

    csv_path = yaml_path + ".csv.xz"
    _write_csv_xz(exact_logic, csv_path)

    round_trip = _loadMeshTableFromCsvXz(csv_path)
    orig_by_key = {tuple(e[0]): e[1] for e in exact_logic}
    rt_by_key = {tuple(e[0]): e[1] for e in round_trip}
    if set(orig_by_key) != set(rt_by_key):
        missing = set(orig_by_key) - set(rt_by_key)
        extra = set(rt_by_key) - set(orig_by_key)
        raise ValueError(
            f"Round-trip key mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    for key in orig_by_key:
        orig_idx, orig_speed = orig_by_key[key]
        rt_idx, rt_speed = rt_by_key[key]
        if orig_idx != rt_idx:
            raise ValueError(
                f"Round-trip index mismatch at {list(key)}: "
                f"{orig_idx} != {rt_idx}"
            )
        if type(orig_speed) is not type(rt_speed) or orig_speed != rt_speed:
            raise ValueError(
                f"Round-trip speed mismatch at {list(key)}: "
                f"{orig_speed!r} ({type(orig_speed).__name__}) != "
                f"{rt_speed!r} ({type(rt_speed).__name__})"
            )

    _clear_exact_logic(data, fmt)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=None, width=200)

    new_size = os.path.getsize(yaml_path)
    csv_size = os.path.getsize(csv_path)
    print(f"  Original:  {orig_size:>12,} bytes")
    print(f"  New YAML:  {new_size:>12,} bytes")
    print(f"  CSV.xz:    {csv_size:>12,} bytes")
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
