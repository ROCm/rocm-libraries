#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""2D subset of validation_holdout problems."""

from validation_holdout import VALIDATION_PROBLEMS

VALIDATION_PROBLEMS_2D = [
    p for p in VALIDATION_PROBLEMS
    if getattr(p, "Di", 1) == 1 and getattr(p, "Z", 1) == 1
]


if __name__ == "__main__":
    print(f"Validation holdout 2D problems: {len(VALIDATION_PROBLEMS_2D)}")