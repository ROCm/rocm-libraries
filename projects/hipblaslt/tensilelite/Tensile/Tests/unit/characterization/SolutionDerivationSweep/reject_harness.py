################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Shared helpers for set-cover solution-rejection tests."""

import copy
import importlib

from Tensile.SolutionStructs.Solution import Solution
from Tensile.SolutionStructs.Utilities import reject as real_reject

solution_module = importlib.import_module("Tensile.SolutionStructs.Solution")


def apply_overrides(state, overrides):
    for key, value in overrides.items():
        if "." in key:
            top, subkey = key.split(".", 1)
            state[top][subkey] = value
        else:
            state[key] = value


def derive_with_rejections(state, isa_info_map, rocm, monkeypatch, keys):
    """Re-derive ``state`` and return selected fields plus rejection reasons."""
    state = copy.deepcopy(state)
    state["AssignedDerivedParameters"] = False
    state["AssignedProblemIndependentDerivedParameters"] = False
    rejection_reasons = []

    def record_reject(rejected_state, _print_reason=True, *args):
        rejection_reasons.append(" ".join(str(arg) for arg in args))
        return real_reject(rejected_state, False, *args)

    monkeypatch.setattr(solution_module, "reject", record_reject)
    Solution.assignDerivedParameters(state, False, False, False, isa_info_map, rocm)
    result = {key: state.get(key) for key in keys}
    result["rejections"] = rejection_reasons
    return result
