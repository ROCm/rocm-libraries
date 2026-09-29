# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fixtures shared by the uhd_gen suite."""
import os
import sys

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


_REPORTING_EVALUATOR = """\
import json, subprocess, sys
result = subprocess.run([{real!r}], input=sys.stdin.buffer.read(), capture_output=True)
if result.returncode:
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)
response = json.loads(result.stdout)
response.pop("feature_semantics_revision", None)
if {revision!r} is not None:
    response["feature_semantics_revision"] = {revision!r}
sys.stdout.write(json.dumps(response) + "\\n")
"""


@pytest.fixture
def evaluator_reporting(evaluator, tmp_path):
    """The real evaluator, reporting `revision` as its feature-semantics revision.

    What a build after a FeatureSemantics.hpp bump looks like from Python: every digest and
    value is still the real one, and only the revision differs. `None` reports none, as an
    evaluator built before the revision existed does. A wrapper executable rather than a
    patched function, so what is exercised is the protocol uhd_gen reads the revision
    through -- the same subprocess, the same response -- not a stand-in for it.
    """
    made = []

    def make(revision) -> str:
        stem = tmp_path / f"evaluator_{len(made)}"
        made.append(revision)
        script = stem.with_suffix(".py")
        script.write_text(_REPORTING_EVALUATOR.format(real=evaluator, revision=revision), encoding="utf-8")
        if os.name == "nt":
            wrapper = stem.with_suffix(".cmd")
            wrapper.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
        else:
            wrapper = stem
            wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
            wrapper.chmod(0o755)
        return str(wrapper)
    return make
