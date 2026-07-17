#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compute the Dapper union filter for a test category, honoring fallback_mode.

Self-contained (no imports from the dependency-parser package) so it can be
co-installed next to the test binary and run on a GPU runner that only has the
installed artifact. Used by the per-project gtest_runner wrapper (e.g.
run_miopen_gtest.py) that the generated CTestTestfile invokes.

Dapper is strictly subtractive: the returned positive set is always a subset of
the category's positives (or a minimal default), never a superset. The category's
negative patterns are always preserved.

The dapper JSON (produced on the builder by `main.py select`) carries:
- dapper_filter : impact-derived positive fixture patterns (may be empty)
- fallback_mode : 'union' | 'entire_category' | 'minimal'
"""

import fnmatch
import json

# Super-minimal default when there is nothing meaningful to run in this category.
DEFAULT_MINIMAL_FILTER = "CPU_HandleHipDevice_NONE*"


def split_gtest_filter_includes(filter_str):
    """Split a --gtest_filter string into (positives, negatives).

    Example: "A.*:B.*-C.*:D.*" -> (['A.*','B.*'], ['C.*','D.*']).
    A negative-only filter yields positives == ['*'] (gtest runs all then subtracts).
    """
    if not filter_str:
        return [], []
    if "-" in filter_str:
        positive_part, *negative_part = filter_str.split("-")
        positives = [p for p in positive_part.split(":") if p]
        negatives = [n for n in ":".join(negative_part).split(":") if n]
    else:
        positives = [p for p in filter_str.split(":") if p]
        negatives = []
    if not positives:
        positives = ["*"]
    return positives, negatives


def _fixed_prefix(pattern):
    """Literal portion of a wildcard pattern up to the first metacharacter."""
    for i, ch in enumerate(pattern):
        if ch in "*?[":
            return pattern[:i]
    return pattern


def patterns_overlap(dapper_pattern, category_pattern):
    """True if a dapper (prefix-style) and category (arbitrary wildcard) pattern
    could match a common fixture. Tested both directions since fnmatch needs a
    concrete string on one side."""
    return fnmatch.fnmatch(
        _fixed_prefix(dapper_pattern), category_pattern
    ) or fnmatch.fnmatch(_fixed_prefix(category_pattern), dapper_pattern)


def compute_union_filter(dapper_filter, category_filter):
    """Intersect dapper positives with category positives; keep category negatives.

    Returns the gtest filter string to run. Empty overlap -> minimal default.
    """
    dapper_positives, _ = split_gtest_filter_includes(dapper_filter)
    category_positives, category_exclude = split_gtest_filter_includes(category_filter)

    union_positives = [
        dp
        for dp in dapper_positives
        if any(patterns_overlap(dp, cp) for cp in category_positives)
    ]
    # de-dupe, preserve order
    seen = set()
    union_positives = [p for p in union_positives if not (p in seen or seen.add(p))]

    if not union_positives:
        print(
            "dapper_union: no overlap between dapper filter and category "
            f"'{category_filter}'; using minimal default '{DEFAULT_MINIMAL_FILTER}'."
        )
        union_positives = [DEFAULT_MINIMAL_FILTER]

    result = ":".join(union_positives)
    if category_exclude:
        result = result + "-" + ":".join(category_exclude)
    return result


def compute_filter(dapper_json_path, category_name, category_filter):
    """Resolve the gtest filter for a category given the dapper JSON.

    fallback_mode:
      - 'minimal'         -> minimal default (nothing test-relevant changed)
      - 'entire_category' -> the category filter as-is (unattributable change; safe)
      - 'union' (default) -> dapper impact filter intersected with the category
    Never returns a superset of the category (subtractive-only).
    """
    with open(dapper_json_path, "r") as f:
        data = json.load(f)
    dapper_filter = data.get("dapper_filter", "")
    fallback_mode = data.get("fallback_mode", "union")

    if fallback_mode == "minimal":
        final = DEFAULT_MINIMAL_FILTER
    elif fallback_mode == "entire_category":
        final = category_filter
    else:  # 'union'
        final = compute_union_filter(dapper_filter, category_filter)

    print(
        f"dapper_union: category='{category_name}' fallback_mode='{fallback_mode}' "
        f"-> --gtest_filter={final}"
    )
    return final
