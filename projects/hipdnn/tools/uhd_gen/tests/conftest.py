# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fixtures shared by the uhd_gen suite."""
import pytest

from uhd_gen.features import resolve_feature_evaluator


@pytest.fixture
def evaluator() -> str:
    """The shared hipdnn_uhd_features binary, without which no features_hash exists.

    RFC 0019 §6.3 leaves the digest with a single definition, and that definition is
    FeatureExtractor::computeHash in the plugin SDK. Python reaches it only by running
    this binary, so a checkout with nothing built cannot exercise anything that stamps
    or verifies a hash. Skipping states that; a Python-side fallback would instead have
    quietly tested a second implementation and reported it as coverage.

    Resolved through the tool's own lookup rather than a second copy of it, so a build
    tree the tool would find is one the suite runs against -- and a lookup that stops
    finding it turns these tests yellow instead of leaving them green against a stale
    assumption.
    """
    try:
        return resolve_feature_evaluator()
    except ValueError as error:
        pytest.skip(str(error))
