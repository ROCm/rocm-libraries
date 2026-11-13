# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Adaptive rocjitsu selection; static policy belongs to test_categories.yaml."""

import argparse
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re

_REORDER_SHAPE = re.compile(r"_dim0_(\d+)_dim1_(\d+)_dim2_(\d+)_dim3_(\d+)(?:_|$)")


def listed_tests(output):
    """Read GTest's listing without parameter comments or their spacing."""
    suite = None
    names = []
    for line in output.splitlines():
        name = line.split(" #", 1)[0].strip()
        if line.startswith("  "):
            if suite and name:
                names.append(suite + name)
        else:
            suite = name if name.endswith(".") else None
    return names


def _reorder_exclusions(names, policy):
    """Keep axis extremes, singleton/nondegenerate shapes and two planes."""
    if policy["strategy"] != "axis_extremes":
        raise ValueError("Unknown tensor_reorder strategy")
    shapes = set()
    for name in names:
        if fnmatchcase(name.split(".", 1)[0], policy["suite_pattern"]):
            match = _REORDER_SHAPE.search(name)
            if match:
                shapes.add(tuple(map(int, match.groups())))
    if not shapes:
        return []
    axes = [sorted({shape[axis] for shape in shapes}) for axis in range(4)]
    low = tuple(values[0] for values in axes)
    middle = tuple(values[len(values) // 2] for values in axes)
    high = tuple(values[-1] for values in axes)
    retained = {
        low,
        middle,
        high,
        (low[0], low[1], high[2], high[3]),
        (high[0], high[1], low[2], low[3]),
    }
    for axis in range(4):
        shape = list(low)
        shape[axis] = high[axis]
        retained.add(tuple(shape))
    return [
        f"{policy['suite_pattern']}.*/*_dim0_{a}_dim1_{b}_dim2_{c}_dim3_{d}_*"
        for a, b, c, d in sorted(shapes - retained)
    ]


def reference_features(geometry):
    """Semantic classes and estimated work of a named convolution geometry."""
    p = {key: int(value) for key, value in re.findall(r"([a-z]+)(\d+)", geometry)}
    if not {"n", "c", "k", "wi", "hi", "fx", "fy", "g"} <= p.keys() or p["g"] <= 0:
        return None
    grouped = p["g"] > 1
    strided = any(p.get(axis, 1) > 1 for axis in ("sx", "sy", "sz"))
    dilated = any(p.get(axis, 1) > 1 for axis in ("dx", "dy", "dz"))
    filters = [p["fx"], p["fy"]]
    if "fz" in p:
        filters.append(p["fz"])
    features = {
        ("grouped", grouped),
        ("batch", p["n"] > 1),
        ("strided", strided),
        ("dilated", dilated),
        ("padded", any(p.get(axis, 0) > 0 for axis in ("px", "py", "pz"))),
        ("pointwise", all(value == 1 for value in filters)),
        ("anisotropic", len(set(filters)) > 1),
        ("channels", (p["k"] > p["c"]) - (p["k"] < p["c"])),
        ("group_stride", grouped, strided),
        ("group_dilation", grouped, dilated),
        ("stride_dilation", strided, dilated),
    }
    work = (
        p["n"]
        * p["c"]
        * p["k"]
        * p["wi"]
        * p["hi"]
        * p.get("di", 1)
        * p["fx"]
        * p["fy"]
        * p.get("fz", 1)
        // p["g"]
    )
    return features, work


def _reference_exclusions(names, policy):
    """Cover semantics per fixture, preferring smaller geometries."""
    if policy["strategy"] != "semantic_cover":
        raise ValueError("Unknown reference_kernels strategy")
    groups = {}
    parameters = {}
    for name in names:
        family = name.split(".", 1)[0]
        if not fnmatchcase(family, policy["suite_pattern"]) or any(
            fnmatchcase(family, pattern) for pattern in policy["protected_patterns"]
        ):
            continue
        geometry = name.rsplit("/", 1)[-1]
        if geometry not in parameters:
            parameters[geometry] = reference_features(geometry)
        if parameters[geometry] is not None:
            groups.setdefault(family, set()).add(geometry)
    excluded_2d = set()
    retained_2d = set()
    exclusions = []
    for family, geometries in groups.items():
        uncovered = set().union(*(parameters[g][0] for g in geometries))
        retained = set()
        while uncovered:
            geometry = min(
                geometries,
                key=lambda g: (-len(parameters[g][0] & uncovered), parameters[g][1], g),
            )
            retained.add(geometry)
            uncovered -= parameters[geometry][0]
        if "_2d_" in family:
            excluded_2d |= geometries
            retained_2d |= retained
        else:
            exclusions.extend(f"{family}.*/{g}" for g in sorted(geometries - retained))
    # A geometry needed by any 2D fixture is retained in all 2D fixtures.
    # Emit exact families so custom policy boundaries/protected suites cannot widen.
    for family in groups:
        if "_2d_" in family:
            exclusions.extend(
                f"{family}.*/{g}"
                for g in sorted((excluded_2d - retained_2d) & groups[family])
            )
    return exclusions


def _matches_filter(name, positive, negative):
    return any(fnmatchcase(name, p) for p in positive) and not any(
        fnmatchcase(name, p) for p in negative
    )


def build_filter(base_filter, names, representative_cases):
    """Narrow a packaged category filter without dropping its existing negatives.

    Static exclusions have already applied to the listing. Rechecking the base
    also makes selection safe for callers supplying an unfiltered inventory.
    """
    positive, _, negative = base_filter.partition("-")
    positive_patterns = (positive or "*").split(":")
    negative_patterns = negative.split(":")
    eligible = [
        name
        for name in names
        if _matches_filter(name, positive_patterns, negative_patterns)
    ]
    exclusions = []
    if "tensor_reorder" in representative_cases:
        exclusions.extend(
            _reorder_exclusions(eligible, representative_cases["tensor_reorder"])
        )
    if "reference_kernels" in representative_cases:
        exclusions.extend(
            _reference_exclusions(eligible, representative_cases["reference_kernels"])
        )
    exact = {}
    wildcard = []
    for pattern in exclusions:
        family = pattern.split(".", 1)[0]
        if any(char in family for char in "*?["):
            wildcard.append((family, pattern))
        else:
            exact.setdefault(family, []).append(pattern)
    rules = {}
    selected = []
    for name in eligible:
        family = name.split(".", 1)[0]
        if family not in rules:
            rules[family] = exact.get(family, []) + [
                pattern for prefix, pattern in wildcard if fnmatchcase(family, prefix)
            ]
        if not any(fnmatchcase(name, pattern) for pattern in rules[family]):
            selected.append(name)
    if exclusions:
        combined_negative = ":".join(([negative] if negative else []) + exclusions)
        result_filter = f"{positive}-{combined_negative}"
    else:
        result_filter = base_filter
    return result_filter, selected, exclusions


def target_version(arch: str) -> int:
    """Decode AMD's gfx major/minor/revision name into its KFD version number."""
    match = re.fullmatch(r"gfx([1-9][0-9]?)([0-9a-f])([0-9a-f])", arch)
    if not match:
        raise ValueError(f"Invalid AMDGPU architecture: {arch!r}")
    major, minor, revision = match.groups()
    return int(major) * 10000 + int(minor, 16) * 100 + int(revision, 16)


def validate_targets(targets):
    """Reject unusable policy entries before selecting an installed profile."""
    if not isinstance(targets, dict) or not targets:
        raise ValueError("rocjitsu targets must be a nonempty architecture map")
    for arch, target in targets.items():
        if not isinstance(arch, str):
            raise ValueError(f"Invalid AMDGPU architecture: {arch!r}")
        target_version(arch)
        if not isinstance(target, dict) or set(target) != {"family", "profile"}:
            raise ValueError(f"rocjitsu target {arch} must define family and profile")
        family, profile = target["family"], target["profile"]
        if not isinstance(family, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", family
        ):
            raise ValueError(f"Invalid rocjitsu family for {arch}: {family!r}")
        if (
            not isinstance(profile, str)
            or not profile.endswith(".json")
            or not all(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
                for part in profile.split("/")
            )
        ):
            raise ValueError(
                f"Invalid relative rocjitsu profile for {arch}: {profile!r}"
            )


def export_policy(source_yaml, destination_json):
    """Export configure-time YAML; the installed selector needs only stdlib."""
    import yaml

    with open(source_yaml, encoding="utf-8") as source:
        category = dict(yaml.safe_load(source)["test_categories"]["rocjitsu-quick"])
    for key in ("test_patterns", "exclude", "exclude_windows", "exclude_linux"):
        if key in category:
            category[key] = [
                item
                for entry in category[key] or []
                for item in (entry if isinstance(entry, list) else [entry])
            ]
    category["category"] = "rocjitsu-quick"
    validate_targets(category.get("targets"))
    # Missing adaptive policy is a configuration error, not a runtime fallback.
    policies = category["representative_cases"]
    for key, strategy in (
        ("tensor_reorder", "axis_extremes"),
        ("reference_kernels", "semantic_cover"),
    ):
        if policies[key]["strategy"] != strategy or not policies[key]["suite_pattern"]:
            raise ValueError(f"Invalid {key} policy")
    if not isinstance(policies["reference_kernels"]["protected_patterns"], list):
        raise ValueError("reference_kernels protected_patterns must be a list")
    Path(destination_json).write_text(
        json.dumps(category, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-policy", nargs=2, required=True, metavar=("SOURCE_YAML", "DEST_JSON")
    )
    args = parser.parse_args()
    export_policy(*args.export_policy)


if __name__ == "__main__":
    main()
