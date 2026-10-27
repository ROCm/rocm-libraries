# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fixtures shared by the uhd_gen suite."""
import os
import sys

import pytest

from uhd_gen.features import resolve_feature_evaluator


@pytest.fixture
def evaluator() -> str:
    """The hipdnn_uhd_features binary; skips when it is not built.

    FeatureExtractor::computeHash is the only definition of features_hash
    (RFC 0019 §6.3), so there is deliberately no Python fallback.
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
    """Factory for a wrapper around the real evaluator that reports `revision` instead.

    `None` reports no revision, like an evaluator built before revisions existed.
    """
    made = []

    def make(revision) -> str:
        stem = tmp_path / f"evaluator_{len(made)}"
        made.append(revision)
        script = stem.with_suffix(".py")
        script.write_text(
            _REPORTING_EVALUATOR.format(real=evaluator, revision=revision),
            encoding="utf-8",
        )
        if os.name == "nt":
            wrapper = stem.with_suffix(".cmd")
            wrapper.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
        else:
            wrapper = stem
            wrapper.write_text(
                f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
        return str(wrapper)

    return make
