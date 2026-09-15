# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The validator's two permanent (opt-in) regressions: the generate -> validate
round trip, and the discriminating mutation fixtures under
``tests/fixtures/validate_descriptors/``.

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


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "validate_descriptors"
FIXTURE_ENGINE = "hipkernel:ValidateFixture"

# Each malformed bundle differs from valid/ by exactly one field. The marker is a token
# the loader can only be emitting because it reached THAT mutation -- for three of them
# the value the field was mutated to, and for duplicate_tuple the id of the kernel the
# mutation was applied to, since that diagnostic identifies the offending kernel by id
# rather than echoing the colliding value. Asserting on it is what makes the case
# discriminating: a bundle that failed for an unrelated reason -- a stray typo, a field
# lost to copy-paste -- would still exit non-zero and would still drop the engine, but
# would not name this token. See the fixtures' README for the mechanism each one trips.
MALFORMED_FIXTURES = [
    ("bad_arch", "GFX942"),
    ("dangling_uuid", "9341b3cb-3540-44f6-9066-f3695a3b6a2d"),
    ("duplicate_tuple", "4dfc8557-7e87-48d1-b512-be076419fad0"),
    ("undeclared_knob", "tile_count"),
]


def _run_validator(validator, root):
    """Validate a fixture bundle, always naming the engine it is supposed to expose.

    ``--expect-engine`` is not optional here. Every malformed bundle fails by making
    the loader DROP the engine, and a drop on its own leaves no error behind -- the
    validator would report an empty engine list and exit 0. Naming the engine is what
    turns a silent drop into a non-zero exit.
    """
    result = subprocess.run(
        [str(validator), str(root), "--expect-engine", FIXTURE_ENGINE, "--json"],
        capture_output=True,
        text=True,
    )
    return result, json.loads(result.stdout)


def test_valid_fixture_validates_clean(validator):
    """The baseline every malformed bundle is a one-field mutation of."""
    result, payload = _run_validator(validator, FIXTURE_ROOT / "valid")

    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["success"] is True
    assert FIXTURE_ENGINE in payload["engines"]
    assert payload["expected_engines_missing"] == []


@pytest.mark.parametrize(
    "name,marker", MALFORMED_FIXTURES, ids=[n for n, _ in MALFORMED_FIXTURES]
)
def test_malformed_fixture_is_rejected(validator, name, marker):
    result, payload = _run_validator(validator, FIXTURE_ROOT / name)

    assert result.returncode != 0, (
        f"{name} validated clean; it differs from valid/ by exactly one deliberate "
        f"defect and must be rejected"
    )
    assert payload["success"] is False
    assert FIXTURE_ENGINE in payload["expected_engines_missing"], (
        f"{name} failed, but {FIXTURE_ENGINE} still loaded -- the defect did not "
        f"suppress the engine, so this bundle is not testing what it claims"
    )

    diagnostics = " ".join(d["message"] for d in payload["diagnostics"])
    assert marker in diagnostics, (
        f"{name} was rejected, but no diagnostic mentions {marker!r}; the bundle may "
        f"be failing for a reason other than its one deliberate defect. Diagnostics: "
        f"{diagnostics}"
    )


def test_scale_add_round_trip_validates_clean(
    validator, generator, scale_add_config, tmp_path
):
    generator.render(scale_add_config, tmp_path)

    result = subprocess.run(
        [
            str(validator),
            str(tmp_path / scale_add_config.descriptor_dir),
            "--expect-engine",
            scale_add_config.engine.name,
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


def test_binary_ops_round_trip_validates_clean(
    validator, generator, binary_ops_config, tmp_path
):
    generator.render(binary_ops_config, tmp_path)

    result = subprocess.run(
        [
            str(validator),
            str(tmp_path / binary_ops_config.descriptor_dir),
            "--expect-engine",
            binary_ops_config.engine.name,
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert binary_ops_config.engine.name in payload["engines"]
