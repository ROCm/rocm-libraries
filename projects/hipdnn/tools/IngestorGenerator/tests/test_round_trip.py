# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The validator's two opt-in regressions: the generate -> validate round trip and
the discriminating mutation and role fixtures under
``tests/fixtures/validate_descriptors/``.

Both need ``hipdnn_validate_descriptors``, a C++ binary this Python suite does not
build -- it exists only under ``HIPDNN_ENABLE_KERNEL_INGESTOR=ON`` -- so they skip
unless pointed at one:

    HIPDNN_VALIDATE_DESCRIPTORS=<build-dir>/bin/hipdnn_validate_descriptors \\
        .venv/bin/python -m pytest -m round_trip
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
# the loader can only emit because it reached THAT mutation; for duplicate_tuple it is
# the kernel id, since that diagnostic names the kernel rather than the colliding
# value. See the fixtures' README for the mechanism each one trips.
MALFORMED_FIXTURES = [
    ("bad_arch", "GFX942"),
    ("dangling_uuid", "9341b3cb-3540-44f6-9066-f3695a3b6a2d"),
    ("duplicate_tuple", "4dfc8557-7e87-48d1-b512-be076419fad0"),
    ("undeclared_knob", "tile_count"),
]

# Each role fixture is valid/ with one role-bound UHD changed; the engine loads in every
# case. Expected: exit status, then (role, success) for every model_checks entry in report
# order. See the fixtures' README for the runtime admission each one mirrors.
ROLE_FIXTURES = [
    ("l2_static_order", 0, [("sort_kernel_catalog", True)]),
    (
        "l1_static_order",
        1,
        [("sort_kernel_catalog", True), ("predict_engine", False)],
    ),
    (
        "l1_native",
        0,
        [("sort_kernel_catalog", True), ("predict_engine", True)],
    ),
]


def _run_validator(validator, root):
    """Validate a fixture bundle, always naming the engine it should expose: malformed
    bundles fail by making the loader DROP the engine, leaving no error behind, and
    ``--expect-engine`` turns that silent drop into a non-zero exit."""
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


@pytest.mark.parametrize(
    "name,exit_code,expected_checks",
    ROLE_FIXTURES,
    ids=[name for name, _, _ in ROLE_FIXTURES],
)
def test_role_fixture_model_checks(validator, name, exit_code, expected_checks):
    result, payload = _run_validator(validator, FIXTURE_ROOT / name)

    assert result.returncode == exit_code, result.stdout + result.stderr
    assert payload["success"] is (exit_code == 0)
    assert FIXTURE_ENGINE in payload["engines"]
    assert [
        (check["role"], check["success"]) for check in payload["model_checks"]
    ] == expected_checks, payload["model_checks"]


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
