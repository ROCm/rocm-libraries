################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""Property test: registry coverage and disjointness (Finding 1.5, B5).

Asserts that every key in the four input-YAML registries is owned by
exactly one section validator. The validators are section-scoped (per
the plan's "Ownership rule (section-scoped)"), so name collisions
across registries are allowed only when each occurrence has a clear
home. This test makes the "which registry owns key X" question
explicit and fails if a future commit accidentally lands a duplicate
that breaks the ownership contract.
"""

from Tensile.Common.GlobalParameters import (
    globalParameters,
    globalParameterTypeOverrides,
    defaultAnalysisParameters,
    libraryLogicTypeOverrides,
)
from Tensile.Common.ValidParameters import (
    validParameters,
    _expectedParamTypes,
    _skipTypeCheck,
)
from Tensile.SolutionStructs.Problem import (
    _defaultProblemType,
    _expectedProblemTypeParamTypes,
)


# Names that are intentionally shared across registries with
# section-scoped semantics; see the plan's "Parameter-name collisions
# across registries" subsection (M4). Each name appears here once and
# is expected to be present in both its owning registries.
KNOWN_CROSS_SECTION_NAMES = {
    "ISA",                # globalParameters list + validParameters list (skip set)
    "CodeObjectVersion",  # globalParameters str + downstream Solution flow
    "MXScaleFormat",      # globalParameters int default + validParameters
                          # enum list; the validParameters entry is the
                          # source of truth for Solution-side state,
                          # globalParameters holds the per-run override.
    "Sparse",             # validParameters -1 sentinel (Solution-side
                          # derived parameter, populated from ProblemType)
                          # + _defaultProblemType int default (the user-
                          # authored value lives in the ProblemType
                          # section).
}


class TestRegistryCoverage:
    def test_every_validParam_has_a_type_or_skip(self):
        """Every validParameters entry is in _expectedParamTypes or _skipTypeCheck.

        The plan rule: an entry whose allowedValues is the -1 sentinel
        is free-form (omitted from the type map); everything else with
        a non-empty value list is in _expectedParamTypes. Anything
        post-processed downstream is in _skipTypeCheck. There must be
        no entry that falls in neither bucket.
        """
        for name, allowed in validParameters.items():
            if allowed == -1:
                # Free-form: type check skipped by design.
                continue
            if name in _skipTypeCheck:
                continue
            if isinstance(allowed, list) and not allowed:
                # Empty list: nothing to derive types from. Acceptable.
                continue
            assert name in _expectedParamTypes, (
                f"validParameters[{name!r}] has no entry in "
                f"_expectedParamTypes and is not in _skipTypeCheck."
            )

    def test_every_default_problem_type_key_is_typed(self):
        """Every _defaultProblemType key derives an expected-type set."""
        for name in _defaultProblemType:
            assert name in _expectedProblemTypeParamTypes, (
                f"_defaultProblemType[{name!r}] has no entry in "
                f"_expectedProblemTypeParamTypes."
            )

    def test_every_none_default_global_has_override(self):
        """Every None-defaulted globalParameters key has an override entry."""
        for k, v in globalParameters.items():
            if v is None:
                assert k in globalParameterTypeOverrides, (
                    f"globalParameters[{k!r}] = None has no entry in "
                    f"globalParameterTypeOverrides."
                )

    def test_every_library_logic_override_is_real_key(self):
        """libraryLogicTypeOverrides keys all live in defaultAnalysisParameters."""
        for k in libraryLogicTypeOverrides:
            assert k in defaultAnalysisParameters, (
                f"libraryLogicTypeOverrides[{k!r}] is not in "
                f"defaultAnalysisParameters."
            )


class TestSectionOwnership:
    """Cross-registry name collisions are all on the known-allowed list.

    The plan's M4 documents two known collisions (ISA, CodeObjectVersion).
    Any new collision should be added to KNOWN_CROSS_SECTION_NAMES with
    a clear ownership rationale, or removed by renaming.
    """

    def test_validParams_vs_global_collision_known(self):
        overlap = (set(validParameters) & set(globalParameters)) - KNOWN_CROSS_SECTION_NAMES
        assert not overlap, (
            f"Unexpected validParameters/globalParameters name collision: "
            f"{sorted(overlap)}. Either rename one side or add to "
            f"KNOWN_CROSS_SECTION_NAMES with a documented ownership rule."
        )

    def test_validParams_vs_problemtype_collision_known(self):
        overlap = (set(validParameters) & set(_defaultProblemType)) - KNOWN_CROSS_SECTION_NAMES
        assert not overlap, (
            f"Unexpected validParameters/_defaultProblemType collision: "
            f"{sorted(overlap)}."
        )

    def test_validParams_vs_analysis_collision_known(self):
        overlap = (set(validParameters) & set(defaultAnalysisParameters)) - KNOWN_CROSS_SECTION_NAMES
        assert not overlap, (
            f"Unexpected validParameters/defaultAnalysisParameters collision: "
            f"{sorted(overlap)}."
        )

    def test_global_vs_problemtype_collision_known(self):
        overlap = (set(globalParameters) & set(_defaultProblemType)) - KNOWN_CROSS_SECTION_NAMES
        assert not overlap, (
            f"Unexpected globalParameters/_defaultProblemType collision: "
            f"{sorted(overlap)}."
        )

    def test_global_vs_analysis_collision_known(self):
        overlap = (set(globalParameters) & set(defaultAnalysisParameters)) - KNOWN_CROSS_SECTION_NAMES
        assert not overlap, (
            f"Unexpected globalParameters/defaultAnalysisParameters collision: "
            f"{sorted(overlap)}."
        )

    def test_problemtype_vs_analysis_collision_known(self):
        overlap = (set(_defaultProblemType) & set(defaultAnalysisParameters)) - KNOWN_CROSS_SECTION_NAMES
        assert not overlap, (
            f"Unexpected _defaultProblemType/defaultAnalysisParameters collision: "
            f"{sorted(overlap)}."
        )
