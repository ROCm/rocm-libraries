# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SLO linter settings. Plain Python so it loads on Python 3.10 with no extra packages.

Owners change limits here, not in the code. Every target is a proposal until the
rocKE owners agree to it.
"""

# Folders (relative to the rocke root) whose .py files are measured.
SCOPE = (
    "library/kernels",
    "platform/python/rocke/instances",
)

# Path parts that are never measured, even inside SCOPE.
EXCLUDE_PARTS = ("tests", "__pycache__")

# Limits for NEW files. Old files are compared to their own baseline instead.
# (metric, limit, direction): "max" = must be <= limit, "min" = must be >= limit.
TARGETS = {
    "sloc": (600, "max"),  # SLO101
    "longest_fn": (120, "max"),  # SLO102
    "raw_per_100": (15.0, "max"),  # SLO103
    "spec_bools_max": (8, "max"),  # SLO105
    "overlap_max": (0.40, "max"),  # SLO106
}

# Variable names always treated as the IR builder, on top of names found by type.
BUILDER_NAMES = ("b", "bb")

# Copy-paste overlap (SLO106).
OVERLAP_MIN_SLOC = 200  # files smaller than this are not compared
OVERLAP_MIN_LINE_CHARS = 14  # shorter lines (after whitespace squeeze) are ignored
OVERLAP_MIN_SHARED = 60  # a pair must share at least this many distinct lines
OVERLAP_LIST_FROM = 0.30  # pairs at or above this overlap are listed in the report

# Which direction is "worse" for every metric the baseline tracks.
# Pattern counts (pat.*) are always "up is worse".
WORSE_IF = {
    "sloc": "up",
    "longest_fn": "up",
    "raw_per_100": "up",
    "hcr": "down",
    "spec_bools_max": "up",
    "overlap_max": "up",
}

# Report-only metrics: shown as warnings, never fail, even with --fail-on increase.
# hcr misses some helper styles, so it is not trusted as a gate yet.
REPORT_ONLY = {"hcr"}

# Ratios move a little on any edit. A change smaller than this is ignored.
TOLERANCE = {
    "raw_per_100": 0.0,
    "hcr": 0.0,
    "overlap_max": 0.0,
}
