# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The generate -> validate round trip, as a permanent (opt-in) regression.

This is deliberately NOT part of the default ``pytest`` run: it depends on
``hipdnn_validate_descriptors``, a C++ binary this Python tool's own test
suite does not and should not build. Point ``HIPDNN_VALIDATE_DESCRIPTORS``
at a build configured with ``HIPDNN_ENABLE_KERNEL_INGESTOR=ON`` and run
with the ``round_trip`` marker selected:

    HIPDNN_VALIDATE_DESCRIPTORS=<build-dir>/bin/hipdnn_validate_descriptors \\
        .venv/bin/python -m pytest -m round_trip

Skipped (not failed) when the env var is unset or names a nonexistent path
-- there is no default hipDNN build containing this binary (it only exists
under HIPDNN_ENABLE_KERNEL_INGESTOR=ON), so a bare `pytest` run must not
fail on a missing tool it was never asked to find.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.round_trip


def _validator_path() -> Path | None:
    raw = os.environ.get("HIPDNN_VALIDATE_DESCRIPTORS")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


@pytest.fixture
def validator():
    path = _validator_path()
    if path is None:
        pytest.skip(
            "HIPDNN_VALIDATE_DESCRIPTORS not set or does not name an existing file -- "
            "set it to <build-dir>/bin/hipdnn_validate_descriptors from a build "
            "configured with HIPDNN_ENABLE_KERNEL_INGESTOR=ON to run this test."
        )
    return path


def test_scale_add_round_trip_validates_clean(
    validator, generator, scale_add_config, tmp_path
):
    written = generator.render(scale_add_config, tmp_path)
    native_rel = next(f for f in written if f.endswith("Native.cpp"))

    result = subprocess.run(
        [
            str(validator),
            str(tmp_path / "descriptors"),
            "--expect-engine",
            scale_add_config.engine.name,
            "--native-source",
            str(tmp_path / native_rel),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert scale_add_config.engine.name in payload["engines"]
    assert payload["expected_engines_missing"] == []
    native_checks = payload["native_source_checks"]
    assert len(native_checks) == 1
    assert native_checks[0]["clean"] is True
    assert native_checks[0]["in_source_not_in_descriptors"] == []


def test_binary_ops_round_trip_validates_clean(
    validator, generator, binary_ops_config, tmp_path
):
    written = generator.render(binary_ops_config, tmp_path)
    native_rel = next(f for f in written if f.endswith("Native.cpp"))

    result = subprocess.run(
        [
            str(validator),
            str(tmp_path / "descriptors"),
            "--expect-engine",
            binary_ops_config.engine.name,
            "--native-source",
            str(tmp_path / native_rel),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert binary_ops_config.engine.name in payload["engines"]
