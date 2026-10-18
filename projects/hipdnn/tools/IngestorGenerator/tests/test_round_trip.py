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
    ("slotted_ranker_declares_metric", "a slot ranker must be metric-less"),
]

# Each role fixture is valid/ (or slotted_valid/) with one role-bound UHD, pack or
# provenance record changed; the engine loads in every case. Expected: exit status, then
# (role, success) for every model_checks entry in report order -- a slot ranker's role is
# spelled role@slot -- then text some reported diagnostic must contain (None: not pinned).
# See the fixtures' README for the runtime admission each one mirrors.
SLOTTED_CHECKS_OK = [
    ("sort_kernel_catalog@dkdv", True),
    ("sort_kernel_catalog@dq", True),
    ("predict_engine", True),
]
ROLE_FIXTURES = [
    ("l2_static_order", 0, [("sort_kernel_catalog", True)], None),
    ("l2_dynamic_features", 0, [("sort_kernel_catalog", True)], None),
    (
        "l1_static_order",
        1,
        [("sort_kernel_catalog", True), ("predict_engine", False)],
        None,
    ),
    (
        "l1_native",
        0,
        [("sort_kernel_catalog", True), ("predict_engine", True)],
        None,
    ),
    ("slotted_valid", 0, SLOTTED_CHECKS_OK, None),
    (
        "slotted_empty_slot",
        0,
        SLOTTED_CHECKS_OK,
        "has no loadable kernel pack",
    ),
    (
        "slotted_ranker_undeclared_field",
        1,
        [
            ("sort_kernel_catalog@dkdv", True),
            ("sort_kernel_catalog@dq", False),
            ("predict_engine", True),
        ],
        "Feature references an undeclared KMD field",
    ),
    (
        "slotted_ranker_unexposed_axis",
        1,
        [
            ("sort_kernel_catalog@dkdv", True),
            ("sort_kernel_catalog@dq", False),
            ("predict_engine", True),
        ],
        "which its UED does not expose as knobs",
    ),
    (
        "slotted_ranker_reads_workspace_token",
        1,
        [
            ("sort_kernel_catalog@dkdv", True),
            ("sort_kernel_catalog@dq", False),
            ("predict_engine", True),
        ],
        "a shared-workspace token a slotted engine reserves",
    ),
    (
        "slotted_ranker_names_other_slot",
        1,
        [
            ("sort_kernel_catalog@dkdv", False),
            ("sort_kernel_catalog@dq", True),
            ("predict_engine", True),
        ],
        "but the ranker serves slot",
    ),
    (
        "slotted_estimate_missing_slot_kmd",
        1,
        [
            ("sort_kernel_catalog@dkdv", True),
            ("sort_kernel_catalog@dq", True),
            ("predict_engine", False),
        ],
        "does not name slot",
    ),
    (
        "slotted_kernel_in_two_slots",
        1,
        [
            ("sort_kernel_catalog@dkdv", False),
            ("sort_kernel_catalog@dq", True),
            ("predict_engine", True),
        ],
        "a kernel serves at most one slot",
    ),
]


def _run_validator(validator, root, *extra):
    """Validate a fixture bundle, always naming the engine it should expose: malformed
    bundles fail by making the loader DROP the engine, leaving no error behind, and
    ``--expect-engine`` turns that silent drop into a non-zero exit."""
    result = subprocess.run(
        [
            str(validator),
            str(root),
            "--expect-engine",
            FIXTURE_ENGINE,
            "--json",
            *extra,
        ],
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


def test_multi_launch_fixture_loads_and_keeps_its_kernels(validator):
    """Kernels carrying `sources` load under the stub handler that declares their
    launch names: the engine is neither dropped nor missing, one kernel under two
    launch names included."""
    engine = "hipkernel:ValidateFixtureMultiLaunch"
    result = subprocess.run(
        [
            str(validator),
            str(FIXTURE_ROOT / "multi_launch"),
            "--expect-engine",
            engine,
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0, result.stdout + result.stderr
    assert payload["success"] is True
    assert payload["engines"] == [engine]
    assert payload["expected_engines_missing"] == []
    assert "0 dropped" in " ".join(d["message"] for d in payload["diagnostics"])


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
    "name,exit_code,expected_checks,diagnostic",
    ROLE_FIXTURES,
    ids=[fixture[0] for fixture in ROLE_FIXTURES],
)
def test_role_fixture_model_checks(
    validator, name, exit_code, expected_checks, diagnostic
):
    result, payload = _run_validator(validator, FIXTURE_ROOT / name)

    assert result.returncode == exit_code, result.stdout + result.stderr
    assert payload["success"] is (exit_code == 0)
    assert FIXTURE_ENGINE in payload["engines"]
    assert [
        (
            check["role"] + (f"@{check['slot']}" if "slot" in check else ""),
            check["success"],
        )
        for check in payload["model_checks"]
    ] == expected_checks, payload["model_checks"]
    if diagnostic is not None:
        messages = " ".join(d["message"] for d in payload["diagnostics"])
        assert diagnostic in messages, messages


@pytest.mark.parametrize(
    "bindings,exit_code,rows",
    [({"graph.batch": 8}, 0, 2), ({}, 1, None)],
    ids=["bound", "unbound"],
)
def test_a_covering_sample_requires_every_feature_to_extract(
    validator, tmp_path, bindings, exit_code, rows
):
    """Extraction is skipped only when no sample covers the engine. Once one does, every
    feature must extract -- one row per candidate kernel -- and a sample leaving a
    feature unbound fails the run instead of being skipped."""
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps([{"engine": FIXTURE_ENGINE, "arch": "gfx942", "bindings": bindings}])
    )
    result, payload = _run_validator(
        validator,
        FIXTURE_ROOT / "l2_dynamic_features",
        "--feature-samples",
        str(samples),
    )

    assert result.returncode == exit_code, result.stdout + result.stderr
    [check] = payload["model_checks"]
    assert check["success"] is (exit_code == 0), check
    assert check.get("feature_rows_checked") == rows, check
    assert "feature_extraction_skipped" not in check, check


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


def test_a_slotted_bundle_with_a_non_default_sdk_version_and_notes_validates_clean(
    validator, generator, tmp_path
):
    """The generated slotted bundle loads in the real validator, and the two engine
    members the shipped fixture leaves at their defaults survive into the UED it loads.
    """
    from tests.test_slots import _load, non_default_engine

    config = _load(tmp_path, non_default_engine)
    generator.render(config, tmp_path)
    descriptors = tmp_path / config.descriptor_dir

    result = subprocess.run(
        [
            str(validator),
            str(descriptors),
            "--expect-engine",
            config.engine.name,
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert config.engine.name in payload["engines"]
    ued = json.loads((descriptors / "split_bwd.ued.json").read_text("utf-8"))
    assert ued["sdk_version"] == "2.0.0"
    assert ued["behavior_notes"] == ["runtime_compilation"]
