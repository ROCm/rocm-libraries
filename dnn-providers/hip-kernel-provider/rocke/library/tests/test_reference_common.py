# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared reference numerics, archives, CLI, worker isolation, and test selection."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from reference_common.artifact import (
    OPERATIONS,
    manifest_operation,
    pack,
    unpack,
    validate_bundle,
)
from reference_common.numeric import (
    ErrorBudget,
    array_digest,
    decode,
    encode,
    max_abs_upper,
)
import io

_SCRIPT = Path(__file__).parent / "reference_common/artifact.py"


def _fixture(tmp_path, operation):
    bundle = tmp_path / "bundle"
    (bundle / "payload").mkdir(parents=True)
    payload = b"host archive fixture"
    (bundle / "payload/kernel.hsaco").write_bytes(payload)
    manifest = {
        "schema": 2,
        "baseline_revision": "a" * 40,
        "files": {"kernel.hsaco": hashlib.sha256(payload).hexdigest()},
    }
    # The current SDPA manifest has no explicit operation field.
    if operation != "sdpa":
        manifest["operation"] = manifest_operation(operation)
    encoded = json.dumps(manifest).encode()
    (bundle / "manifest.json").write_bytes(encoded)
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema": 2,
                "baseline_revision": manifest["baseline_revision"],
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    )
    return bundle, lock


def _run(*args, script=_SCRIPT):
    # -S disables installed packages; -I removes inherited source paths.
    return subprocess.run(
        [sys.executable, "-I", "-S", str(script), *map(str, args)],
        capture_output=True,
        text=True,
    )


def _another_operation(operation):
    """Any other registered operation, for cross-operation rejection checks."""
    return next(other for other in OPERATIONS if other != operation)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_common_cli_round_trip_without_site_packages(tmp_path, operation):
    bundle, lock = _fixture(tmp_path, operation)
    archive = tmp_path / "common.tar.gz"
    result = _run(
        "pack",
        "--operation",
        operation,
        "--bundle",
        bundle,
        "--lock",
        lock,
        "--archive",
        archive,
    )
    assert result.returncode == 0, result.stderr
    with tarfile.open(archive) as tar:
        assert all(m.name.startswith(f"{operation}_reference_bundle/") for m in tar)
    staged = tmp_path / "staged"
    result = _run(
        "unpack",
        "--operation",
        operation,
        "--bundle",
        staged,
        "--lock",
        lock,
        "--archive",
        archive,
    )
    assert result.returncode == 0, result.stderr
    result = _run(
        "validate", "--operation", operation, "--bundle", staged, "--lock", lock
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("operation", [None, "unknown", "../escape"])
def test_common_cli_requires_a_supported_explicit_operation(tmp_path, operation):
    bundle, lock = _fixture(tmp_path, "sdpa")
    args = ["validate", "--bundle", bundle, "--lock", lock]
    if operation is not None:
        args += ["--operation", operation]
    result = _run(*args)
    assert result.returncode == 2
    assert "--operation" in result.stderr


def test_operation_package_alone_enrolls_the_common_cli(tmp_path):
    """Adding <operation>_reference needs no edit to the shared artifact CLI."""
    tests = tmp_path / "tests"
    script = tests / "reference_common/artifact.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(_SCRIPT.read_bytes())
    package = tests / "gemm_reference"
    (package / "architectures").mkdir(parents=True)
    (package / "architectures/registry.json").write_text('["gfx942"]\n')
    (package / "operation.json").write_text('{"manifest_operation": "gemm-fwd"}\n')
    # A package without an architecture registry is not an operation.
    (tests / "draft_reference").mkdir()
    bundle, lock = _fixture(tmp_path, "sdpa")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["operation"] = "gemm-fwd"
    encoded = json.dumps(manifest).encode()
    (bundle / "manifest.json").write_bytes(encoded)
    expected = json.loads(lock.read_text())
    expected["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    lock.write_text(json.dumps(expected))
    # Operations come from the tree beside the script, as in an installed tree.
    for operation, returncode in [("gemm", 0), ("draft", 2), ("sdpa", 2)]:
        result = _run(
            "validate",
            "--operation",
            operation,
            "--bundle",
            bundle,
            "--lock",
            lock,
            script=script,
        )
        assert result.returncode == returncode, result.stderr


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("command", ["pack", "validate"])
def test_common_cli_rejects_wrong_operation_even_with_matching_lock(
    tmp_path, operation, command
):
    bundle, lock = _fixture(tmp_path, operation)
    other = _another_operation(operation)
    archive = tmp_path / "wrong.tar.gz"
    result = _run(
        command,
        "--operation",
        other,
        "--bundle",
        bundle,
        "--lock",
        lock,
        "--archive",
        archive,
    )
    assert result.returncode != 0
    assert "does not match" in result.stderr
    assert not archive.exists()


def test_archive_cannot_be_unpacked_as_another_operation(tmp_path):
    bundle, lock = _fixture(tmp_path, "conv")
    archive = tmp_path / "conv.tar.gz"
    pack(bundle, archive, lock, operation="conv")
    with pytest.raises(ValueError, match="invalid SDPA archive member"):
        unpack(archive, tmp_path / "wrong", lock, operation="sdpa")
    assert not (tmp_path / "wrong").exists()


def test_python_api_rejects_invalid_operation_before_writing(tmp_path):
    bundle, lock = _fixture(tmp_path, "conv")
    archive = tmp_path / "bad.tar.gz"
    with pytest.raises(ValueError, match="unsupported reference operation"):
        pack(bundle, archive, lock, operation="../escape")
    assert not archive.exists()


def test_archive_is_reproducible_and_has_no_host_metadata(tmp_path, operation):
    bundle, lock = _fixture(tmp_path, operation)
    first, second = (tmp_path / "first.tar.gz", tmp_path / "second.tar.gz")
    pack(bundle, first, lock, operation=operation)
    pack(bundle, second, lock, operation=operation)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as tar:
        for member in tar:
            assert member.uid == member.gid == member.mtime == 0
            assert member.uname == member.gname == ""
            assert member.name.startswith(f"{operation}_reference_bundle/")
    output = tmp_path / "staged"
    unpack(first, output, lock, operation=operation)
    unpack(first, output, lock, operation=operation)
    validate_bundle(output, lock, operation=operation)
    assert (output / "payload/kernel.hsaco").read_bytes() == b"host archive fixture"


@pytest.mark.parametrize("change", ["payload", "manifest", "extra"])
def test_corrupt_bundle_cannot_be_packed(tmp_path, change, operation):
    bundle, lock = _fixture(tmp_path, operation)
    path = {
        "payload": "payload/kernel.hsaco",
        "manifest": "manifest.json",
        "extra": "payload/extra",
    }[change]
    (bundle / path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload|manifest"):
        pack(bundle, tmp_path / "bad.tar.gz", lock, operation=operation)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("sdpa_reference_bundle/../../escaped", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("sdpa_reference_bundle/link", tarfile.SYMTYPE),
        ("sdpa_reference_bundle/hardlink", tarfile.LNKTYPE),
    ],
)
def test_unsafe_archive_is_rejected_without_exposing_output(
    tmp_path, name, kind, operation
):
    name = name.replace("sdpa_reference_bundle", f"{operation}_reference_bundle")
    _, lock = _fixture(tmp_path, operation)
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "../../escaped" if kind != tarfile.REGTYPE else ""
        tar.addfile(member, io.BytesIO())
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="invalid .* archive member"):
        unpack(archive, output, lock, operation=operation)
    assert not output.exists()
    assert not (tmp_path / "escaped").exists()


def test_unpacked_payload_is_checked_against_the_lock(tmp_path, operation):
    bundle, lock = _fixture(tmp_path, operation)
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(
            bundle / "manifest.json",
            arcname=f"{operation}_reference_bundle/manifest.json",
        )
        data = b"corrupt"
        member = tarfile.TarInfo(f"{operation}_reference_bundle/payload/kernel.hsaco")
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="payload"):
        unpack(archive, tmp_path / "output", lock, operation=operation)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("suffix", [".npz", ".npy"])
@pytest.mark.parametrize("location", ["payload", "."])
def test_generated_bundle_cannot_reintroduce_tensor_files(
    tmp_path, suffix, location, operation
):
    bundle, lock = _fixture(tmp_path, operation)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["schema"] = 2
    name = "inputs" + suffix
    (bundle / location / name).write_bytes(b"tensor")
    if location == "payload":
        manifest["files"][name] = hashlib.sha256(b"tensor").hexdigest()
    encoded = json.dumps(manifest).encode()
    (bundle / "manifest.json").write_bytes(encoded)
    expected = json.loads(lock.read_text())
    expected.update(schema=2, manifest_sha256=hashlib.sha256(encoded).hexdigest())
    lock.write_text(json.dumps(expected))
    with pytest.raises(ValueError, match="must not contain tensor files"):
        pack(bundle, tmp_path / "bad.tar.gz", lock, operation=operation)


@pytest.fixture(params=OPERATIONS)
def operation(request):
    return request.param


@pytest.mark.parametrize("schema", [1, 3])
def test_unsupported_storage_schema_is_rejected(tmp_path, schema):
    bundle, lock = _fixture(tmp_path, "conv")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["schema"] = schema
    encoded = json.dumps(manifest).encode()
    (bundle / "manifest.json").write_bytes(encoded)
    expected = json.loads(lock.read_text())
    expected.update(schema=schema, manifest_sha256=hashlib.sha256(encoded).hexdigest())
    lock.write_text(json.dumps(expected))
    with pytest.raises(ValueError, match="only tensor-free schema-2"):
        validate_bundle(bundle, lock, operation="conv")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("layout", ["source", "bin/hip_kernel_provider", "standalone"])
def test_bundle_lookup_stays_in_operation_and_architecture_domain(
    tmp_path, operation, layout
):
    from reference_common.paths import default_bundle_path

    if layout == "source":
        tests = tmp_path / "rocke/library/tests"
        expected = tests / "reference_bundles" / operation
    else:
        root = tmp_path / layout
        tests = root / "tests/library/tests"
        (tests / "reference_bundles" / operation / "gfx942").mkdir(parents=True)
        expected = root / "engines/test_arch_content/rocke" / operation
    assert (
        default_bundle_path(tests, "gfx942", operation=operation) == expected / "gfx942"
    )
    assert (
        default_bundle_path(tests, "gfx950", operation=operation) == expected / "gfx950"
    )


@pytest.mark.parametrize("persistent", [False, True])
def test_reference_workers_block_installed_torch(tmp_path, persistent):
    """Exercise both transports with an importable Torch package that must not run."""
    from contextlib import nullcontext
    from reference_common.runner import run_worker
    from reference_common.session import reuse_workers

    (tmp_path / "torch.py").write_text(
        "raise AssertionError('Torch package executed in a reference worker')\n"
    )
    (tmp_path / "probe_worker.py").write_text(
        """
import json
import sys
from pathlib import Path
import numpy as np

def run(request, work):
    try:
        import torch
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("Torch import was not blocked")
    assert "torch" not in sys.modules
    np.savez(work / "outputs.npz", out_0=np.zeros(1))
    (work / "report.json").write_text(
        json.dumps({"launches": 1, "torch_imported": False})
    )

if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
"""
    )
    context = reuse_workers("probe_worker") if persistent else nullcontext()
    with context:
        for index in range(2):
            _, report = run_worker(
                {"mode": "source", "repetitions": 1},
                runner=tmp_path,
                platform=tmp_path,
                library=None,
                work=tmp_path / str(index),
                module="probe_worker",
            )
            assert not report["torch_imported"]


def test_reference_pytest_selects_cohort_and_keeps_negative_checks(tmp_path):
    """Test collection with a fake HIP probe; this does not execute GPU kernels."""
    import os

    import rocke

    (tmp_path / "conftest.py").write_text(
        """
from reference_common.pytest_support import start_reference_session, select_reference_items
from rocke.runtime import hip_module
hip_module.get_device_target_id = lambda device=0: "gfx942"
def pytest_addoption(parser):
    parser.addoption("--rocke-reference-arch")
    parser.addoption("--rocke-reference-operation")
    parser.addoption("--rocke-reference-bundle")
    parser.addoption("--rocke-reference-lock")
def pytest_sessionstart(session):
    start_reference_session(session.config)
def pytest_collection_modifyitems(config, items):
    select_reference_items(config, items)
"""
    )
    (tmp_path / "test_probe.py").write_text(
        """
import pytest
@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
def test_numerical_case(architecture):
    assert architecture == "gfx942"
def test_negative_control():
    pass
"""
    )
    env = dict(os.environ, AMDGPU_FAMILIES="gfx94X-dcgpu", AMDGPU_TARGETS="gfx942")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(rocke.__file__).resolve().parent.parent),
            str(Path(__file__).parent),
        ]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path / "test_probe.py"),
            "-q",
            "--rocke-reference-operation",
            "conv",
            "--rocke-reference-arch",
            "gfx942",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed, 1 deselected" in result.stdout


@pytest.mark.parametrize(
    "architecture,device_target,targets,outcome",
    [
        ("gfx1200", "gfx1200", "gfx1200,gfx1201", "run"),
        ("gfx1200", "gfx1201", "gfx1200,gfx1201", "skip"),
        ("gfx1201", "gfx1200", "gfx1200,gfx1201", "skip"),
        ("gfx1201", "gfx1200-strict:xnack-", "gfx1200,gfx1201", "skip"),
        ("gfx1201", "gfx1201", "gfx1200,gfx1201", "run"),
        ("gfx1201", "gfx1201-strict:xnack-", "gfx1200,gfx1201", "run"),
        ("gfx1201", None, "gfx1200,gfx1201", "error"),
        ("gfx1201", "gfx950", "gfx1200,gfx1201", "error"),
        ("gfx1151", "gfx1100", "gfx1100,gfx1151", "skip"),
        ("gfx942", "gfx942:sramecc+:xnack-", "gfx942", "run"),
        ("gfx1201", "gfx1201", "gfx1200;gfx1201", "run"),
        ("gfx1201", "gfx1200", "gfx1200;gfx1201", "skip"),
    ],
)
def test_reference_session_uses_hardware_target_identity(
    monkeypatch, architecture, device_target, targets, outcome
):
    """Keep real catalog normalization while stubbing only the HIP target query."""
    from types import SimpleNamespace

    from reference_common.pytest_support import start_reference_session
    from rocke.runtime import hip_module

    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: device_target
    )
    monkeypatch.setenv("AMDGPU_FAMILIES", "")
    monkeypatch.setenv("AMDGPU_TARGETS", targets)
    options = {
        "--rocke-reference-operation": "sdpa",
        "--rocke-reference-arch": architecture,
    }
    config = SimpleNamespace(getoption=options.get)
    if outcome == "skip":
        with pytest.raises(pytest.exit.Exception, match="different GPU lane") as error:
            start_reference_session(config)
        assert error.value.returncode == 77
        assert f"AMDGPU_TARGETS={targets!r}" in str(error.value)
    elif outcome == "error":
        with pytest.raises(pytest.UsageError, match=f"required {architecture} GPU"):
            start_reference_session(config)
    else:
        try:
            start_reference_session(config)
        except pytest.exit.Exception:
            pytest.fail("matching hardware must not skip as a different GPU lane")


def test_reference_entry_skips_a_different_gpu_lane():
    from reference_common.pytest_support import architecture_available

    assert not architecture_available("gfx942", "gfx950")


@pytest.mark.parametrize("device", [None, "gfx950"], ids=["missing-gpu", "wrong-gpu"])
def test_reference_selected_target_requires_matching_hardware(device):
    from reference_common.pytest_support import architecture_available

    with pytest.raises(RuntimeError, match="required gfx942 GPU unavailable"):
        architecture_available("gfx942", device, "gfx942")


def test_reference_exact_targets_select_only_matching_cohorts():
    from reference_common.pytest_support import architecture_available

    # Same family, different artifact target: do not require the gfx1201 bundle.
    assert not architecture_available("gfx1201", "gfx1200", "gfx1200")
    # Downloading both artifacts does not mean both devices are present.
    assert not architecture_available("gfx1201", "gfx1200", "gfx1200,gfx1201")


@pytest.mark.parametrize(
    "targets",
    ["gfx942;gfx1201", " gfx1201 ; gfx942 ", "gfx1201,gfx942;", ";gfx942;; ,gfx1201"],
    ids=["semicolon", "spaced", "mixed", "empty-items"],
)
def test_reference_exact_targets_accept_cmake_lists(targets):
    from reference_common.pytest_support import architecture_available

    # A CMake-style list must not hide matching hardware as a different lane.
    assert architecture_available("gfx942", "gfx942", targets)
    assert not architecture_available("gfx942", "gfx1201", targets)
    assert not architecture_available("gfx950", "gfx942", targets)
    with pytest.raises(RuntimeError, match="required gfx942 GPU unavailable"):
        architecture_available("gfx942", "gfx950", targets)


def test_reference_empty_target_list_selects_by_processor():
    from reference_common.pytest_support import architecture_available

    assert architecture_available("gfx942", "gfx942", " ; , ")
    assert not architecture_available("gfx942", "gfx950", " ; , ")
    with pytest.raises(RuntimeError, match="required gfx942 GPU unavailable"):
        architecture_available("gfx942", None, " ; , ")


def test_reference_local_run_without_a_gpu_fails():
    from reference_common.pytest_support import architecture_available

    with pytest.raises(RuntimeError, match="required gfx942 GPU unavailable"):
        architecture_available("gfx942", None)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_reference_family_labels_do_not_require_another_processors_cohort(
    operation, monkeypatch
):
    """A multi-family job may run on a supported GPU without this reference."""
    from importlib import import_module
    from types import SimpleNamespace

    from reference_common.pytest_support import start_reference_session
    from rocke.runtime import hip_module

    # Request any enrolled processor while the job's GPU is another one.
    registry = import_module(f"{operation}_reference.architectures")
    architecture = registry.ARCHITECTURES[0]
    detected = "gfx942" if architecture == "gfx950" else "gfx950"
    monkeypatch.setenv("AMDGPU_FAMILIES", "gfx94X-dcgpu,gfx950-dcgpu")
    monkeypatch.delenv("AMDGPU_TARGETS", raising=False)
    monkeypatch.setattr(hip_module, "get_device_target_id", lambda device=0: detected)
    options = {
        "--rocke-reference-operation": operation,
        "--rocke-reference-arch": architecture,
    }
    with pytest.raises(pytest.exit.Exception, match="different GPU lane") as error:
        start_reference_session(SimpleNamespace(getoption=options.get))
    assert error.value.returncode == 77


def test_reference_family_label_does_not_hide_a_missing_gpu(monkeypatch):
    from types import SimpleNamespace

    from reference_common.pytest_support import start_reference_session
    from rocke.runtime import hip_module

    monkeypatch.setenv("AMDGPU_FAMILIES", "gfx120X-all")
    monkeypatch.delenv("AMDGPU_TARGETS", raising=False)
    monkeypatch.setattr(hip_module, "get_device_target_id", lambda device=0: None)
    options = {
        "--rocke-reference-operation": "sdpa",
        "--rocke-reference-arch": "gfx1201",
    }
    with pytest.raises(pytest.UsageError, match="required gfx1201 GPU unavailable"):
        try:
            start_reference_session(SimpleNamespace(getoption=options.get))
        except pytest.exit.Exception:
            pytest.fail("a selected reference must not skip a missing GPU")


@pytest.mark.parametrize(
    "operation,architecture",
    [("sdpa", "gfx1100"), ("conv", "gfx1151")],
)
def test_reference_fixture_skips_supported_gpu_without_cohort(
    operation, architecture, monkeypatch
):
    """Broader build families do not enroll reference coverage for this GPU."""
    from importlib import import_module
    from types import SimpleNamespace

    from rocke.runtime import hip_module

    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    monkeypatch.setenv("AMDGPU_FAMILIES", "gfx94X-dcgpu,gfx110X-all,gfx950-dcgpu")
    monkeypatch.setenv("AMDGPU_TARGETS", architecture)
    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: architecture
    )

    def unexpected_bundle_lookup(*args, **kwargs):
        pytest.fail("an unenrolled processor must skip before looking up a bundle")

    monkeypatch.setattr(gpu_tests, "default_bundle_path", unexpected_bundle_lookup)
    fixture = gpu_tests.reference_bundle.__wrapped__(
        SimpleNamespace(getoption=lambda option: None)
    )
    enrolled = ", ".join(
        import_module(f"{operation}_reference.architectures").ARCHITECTURES
    )
    with pytest.raises(pytest.skip.Exception, match="no .* reference cohort") as skip:
        next(fixture)
    assert f"enrolled for {architecture}; run on an enrolled" in str(skip.value)
    assert f"({enrolled})" in str(skip.value)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_reference_fixture_skip_says_no_gpu_was_detected(operation, monkeypatch):
    """Without a GPU, the skip must not read as an enrollment problem."""
    from importlib import import_module
    from types import SimpleNamespace

    from rocke.runtime import hip_module

    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    registry = import_module(f"{operation}_reference.architectures")
    monkeypatch.setattr(hip_module, "get_device_target_id", lambda device=0: None)
    fixture = gpu_tests.reference_bundle.__wrapped__(
        SimpleNamespace(getoption=lambda option: None)
    )
    with pytest.raises(pytest.skip.Exception, match="no GPU detected") as skip:
        next(fixture)
    reason = str(skip.value)
    assert f"({', '.join(registry.ARCHITECTURES)})" in reason
    assert "None" not in reason and "cohort" not in reason


@pytest.mark.parametrize(
    "operation,architecture",
    [
        ("sdpa", "gfx942"),
        ("sdpa", "gfx1200"),
        ("sdpa", "gfx1201"),
        ("conv", "gfx942"),
        ("conv", "gfx950"),
    ],
)
@pytest.mark.parametrize("candidate", [False, True])
def test_reference_fixture_selects_explicit_paths(
    operation, architecture, candidate, tmp_path, monkeypatch
):
    from contextlib import closing
    from importlib import import_module
    from types import SimpleNamespace

    from rocke.runtime import hip_module

    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    options = {
        "--rocke-reference-operation": operation if candidate else None,
        "--rocke-reference-bundle": tmp_path if candidate else None,
        "--rocke-reference-lock": (
            tmp_path / "candidate-lock.json" if candidate else None
        ),
    }
    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: architecture
    )
    monkeypatch.setattr(
        gpu_tests, "default_bundle_path", lambda *args, **kwargs: tmp_path
    )
    calls = []

    def load(bundle, lock, *, architecture):
        calls.append((bundle, lock, architecture))
        return {"candidate": candidate}

    monkeypatch.setattr(gpu_tests, "load_bundle", load)
    config = SimpleNamespace(getoption=options.get)
    with closing(gpu_tests.reference_bundle.__wrapped__(config)) as fixture:
        target, bundle, manifest = next(fixture)
        assert (target.NAME, bundle, manifest) == (
            architecture,
            tmp_path,
            {"candidate": candidate},
        )
    assert calls == [(tmp_path, options["--rocke-reference-lock"], architecture)]
    if candidate:
        options["--rocke-reference-operation"] = _another_operation(operation)
        calls.clear()
        with closing(gpu_tests.reference_bundle.__wrapped__(config)) as fixture:
            with pytest.raises(pytest.skip.Exception, match="another operation"):
                next(fixture)
        assert not calls


@pytest.mark.parametrize(
    "options, message",
    [
        (
            {"--rocke-reference-bundle": "candidate"},
            "explicit operation and architecture",
        ),
        (
            {
                "--rocke-reference-operation": "sdpa",
                "--rocke-reference-arch": "gfx942",
                "--rocke-reference-lock": "lock.json",
            },
            "requires a reference bundle",
        ),
    ],
)
def test_reference_path_options_require_explicit_scope(options, message):
    from types import SimpleNamespace
    from reference_common.pytest_support import start_reference_session

    with pytest.raises(pytest.UsageError, match=message):
        start_reference_session(SimpleNamespace(getoption=options.get))


@pytest.mark.parametrize(
    "operation,architecture",
    [("sdpa", "gfx942"), ("conv", "gfx942"), ("sdpa", "gfx1200")],
)
@pytest.mark.parametrize("selected", [False, True])
def test_reference_missing_bundle_fails_only_when_selected(
    operation, architecture, selected, tmp_path, monkeypatch
):
    from contextlib import closing
    from importlib import import_module
    from types import SimpleNamespace

    from rocke.runtime import hip_module

    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: architecture
    )
    monkeypatch.setattr(
        gpu_tests, "default_bundle_path", lambda *args, **kwargs: tmp_path / "missing"
    )
    options = {"--rocke-reference-operation": operation if selected else None}
    config = SimpleNamespace(getoption=options.get)
    with closing(gpu_tests.reference_bundle.__wrapped__(config)) as fixture:
        with pytest.raises(
            pytest.fail.Exception if selected else pytest.skip.Exception
        ) as outcome:
            next(fixture)
    message = str(outcome.value)
    name = operation.upper()
    assert (
        f"required {architecture} {name} reference bundle is missing"
        if selected
        else f"no extracted {architecture} {name} reference bundle at"
    ) in message
    # A source checkout tracks each published archive beside the default bundle.
    pointer = Path(__file__).parent / (
        f"reference_bundles/{operation}/{architecture}.tar.gz.dvc"
    )
    if pointer.is_file():
        assert "fetch and extract it with dvc pull and artifact.py unpack" in message
        assert f"reference_bundles/{operation}/{architecture}.tar.gz.dvc" in message
        assert f"artifact.py unpack --operation {operation} " in message
    else:
        assert f"no published {architecture} {name} archive is tracked" in message


@pytest.mark.parametrize("operation", OPERATIONS)
def test_reference_fixture_explains_an_archive_passed_as_the_bundle(
    operation, tmp_path, monkeypatch
):
    from contextlib import closing
    from importlib import import_module
    from types import SimpleNamespace

    from rocke.runtime import hip_module

    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    registry = import_module(f"{operation}_reference.architectures")
    architecture = registry.ARCHITECTURES[0]
    archive = tmp_path / f"{architecture}.tar.gz"
    archive.write_bytes(b"archive fixture")
    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: architecture
    )
    options = {
        "--rocke-reference-operation": operation,
        "--rocke-reference-bundle": str(archive),
    }
    fixture = gpu_tests.reference_bundle.__wrapped__(
        SimpleNamespace(getoption=options.get)
    )
    with closing(fixture), pytest.raises(pytest.fail.Exception) as failure:
        next(fixture)
    message = str(failure.value)
    assert "--rocke-reference-bundle needs an extracted bundle directory" in message
    assert f"artifact.py unpack --operation {operation} " in message
    assert f"then pass --rocke-reference-bundle {tmp_path / architecture}" in message


@pytest.mark.parametrize("operation", OPERATIONS)
def test_reference_worker_requires_explicit_architecture(operation, tmp_path):
    from importlib import import_module

    worker = import_module(f"{operation}_reference.worker")
    # Reject the malformed request before runtime imports, GPU probing, or files.
    with pytest.raises(KeyError, match="architecture"):
        worker.run({}, tmp_path)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "architecture,device_target,accepted",
    [
        ("gfx1200", "gfx1200", True),
        ("gfx1100", "gfx1100", True),
        ("gfx942", "gfx942:sramecc+:xnack-", True),
        ("gfx1200", "gfx1201", False),
        ("gfx1100", "gfx1101", False),
        ("gfx942", None, False),
    ],
)
def test_reference_worker_requires_the_physical_processor(
    operation, architecture, device_target, accepted, tmp_path, monkeypatch
):
    """A generic catalog row (gfx1200 -> gfx12-generic) must not hide the device."""
    from importlib import import_module
    from types import SimpleNamespace

    import rocke
    from rocke.runtime import hip_module

    worker = import_module(f"{operation}_reference.worker")
    # Only the enrolled name is read before the device check, so the processor
    # need not have an adapter for this operation.
    monkeypatch.setattr(
        worker, "get_architecture", lambda name: SimpleNamespace(NAME=name)
    )
    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: device_target
    )
    request = {
        "architecture": architecture,
        "platform_root": str(Path(rocke.__file__).resolve().parent),
        "mode": "source",
        "repetitions": 1,
    }
    # An accepted device proceeds to decode the case this request omits.
    with pytest.raises(
        KeyError if accepted else RuntimeError,
        match="case" if accepted else f"HIP-visible {architecture} device",
    ):
        worker.run(request, tmp_path)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_operation_packages_plug_into_the_shared_plumbing(operation):
    """Each package declares its own parts; qualification and tests are shared."""
    from importlib import import_module

    cli = import_module(f"{operation}_reference.cli")
    spec = cli.OPERATION
    assert spec.name == operation
    assert spec.contract is import_module(f"{operation}_reference.contract")
    assert spec.architectures is import_module(f"{operation}_reference.architectures")
    assert set(spec.reference) == {"implementation", "metric", "input_generator"}
    assert callable(import_module(spec.worker_module).run)
    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    assert gpu_tests.load_bundle == spec.load_bundle
    for name in (
        "correctness_against_qualified_rocke",
        "rejects_perturbed_gpu_results",
        "rejects_missing_gpu_launch",
    ):
        assert callable(getattr(gpu_tests, f"test_{operation}_{name}"))


def test_bf16_rounds_ties_to_even_and_preserves_storage_meaning():
    values = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], np.uint32)
    encoded = encode(values.view(np.float32), "bf16")
    np.testing.assert_array_equal(encoded, [0x3F80, 0x3F82, 0xBF80, 0xBF82])
    np.testing.assert_array_equal(
        decode(encoded, "bf16"), [1.0, 1.015625, -1.0, -1.015625]
    )
    with pytest.raises(ValueError, match="storage"):
        decode(encoded.view(np.float16), "bf16")


def test_absolute_max_promotes_before_subtraction():
    left = np.array([65504.0, 0.0], dtype=np.float16)
    right = np.array([-65504.0, 1.0], dtype=np.float16)
    assert max_abs_upper(left, right) == math.nextafter(131008.0, math.inf)
    assert max_abs_upper(left, left) == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_results_cannot_pass(bad):
    with pytest.raises(ValueError, match="non-finite"):
        max_abs_upper(np.array([bad]), np.zeros(1))
    with pytest.raises(ValueError, match="invalid"):
        ErrorBudget(0.02, 0.002, 0.001).check(bad)


def test_shapes_and_empty_outputs_cannot_broadcast_or_pass():
    for left, right in [(np.zeros(2), np.zeros(1)), (np.zeros(0), np.zeros(0))]:
        with pytest.raises(ValueError, match="outputs"):
            max_abs_upper(left, right)


@pytest.mark.parametrize("margin", [0.0, 0.001])
def test_budget_is_strict_even_at_the_float_boundary(margin):
    budget = ErrorBudget(0.02, 0.002, margin)
    limit = budget.comparison_limit
    assert Fraction(limit) + Fraction(budget.baseline_error_bound) + Fraction(
        budget.margin
    ) <= Fraction(budget.tolerance)
    assert Fraction(limit) + Fraction(budget.baseline_error_bound) < Fraction(0.02)
    budget.check(limit)
    with pytest.raises(AssertionError, match="remaining limit"):
        budget.check(math.nextafter(limit, math.inf))


@pytest.mark.parametrize(
    "tolerance,bound,margin",
    [
        (0.02, 0.019, 0.002),
        (0.02, -0.001, 0.001),
        (0.02, 0.02, 0.0),
        (0.02, 0.0, -0.001),
        (math.inf, 0.0, 0.001),
        (0.02, math.nan, 0.001),
    ],
)
def test_invalid_budgets_are_rejected(tolerance, bound, margin):
    with pytest.raises(ValueError):
        ErrorBudget(tolerance, bound, margin)


def test_tensor_digest_binds_shape_dtype_and_values():
    array = np.array([1, 2, 3, 4], dtype="<u2")
    assert array_digest(array) != array_digest(array.reshape(2, 2))
    assert array_digest(array) != array_digest(array.view("<f2"))
    assert array_digest(array) != array_digest(array + 1)
    assert array_digest(array) == array_digest(array.astype(">u2"))


def test_reused_workers_isolate_roles_and_import_roots(tmp_path):
    import json
    import os

    from reference_common.session import WorkerSession

    def environment(name):
        root = tmp_path / name
        package = root / "fixture_reference"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        (package / "worker.py").write_text(
            "import json, os\ncount = 0\n"
            "def run(request, work):\n"
            "    global count\n"
            "    count += 1\n"
            "    (work / 'result.json').write_text(json.dumps([os.getpid(), count]))\n"
        )
        return dict(os.environ, PYTHONPATH=str(root), PYTHONNOUSERSITE="1")

    first, second = environment("first"), environment("second")
    session = WorkerSession(module="fixture_reference.worker", timeout=10)
    processes = []
    try:
        results = []
        for i, (mode, env) in enumerate(
            [
                ("replay", first),
                ("replay", first),
                ("source", first),
                ("replay", second),
            ]
        ):
            work = tmp_path / str(i)
            work.mkdir()
            request = work / "request.json"
            request.write_text("{}")
            session.execute(mode, request, env)
            results.append(json.loads((work / "result.json").read_text()))
        assert results[0][0] == results[1][0]
        assert [row[1] for row in results] == [1, 2, 1, 1]
        assert len({results[i][0] for i in [0, 2, 3]}) == 3
        processes = [worker.process for worker in session.workers.values()]
    finally:
        session.close()
    assert all(process.poll() is not None for process in processes)


@pytest.mark.parametrize("behavior", ["raise", "exit", "timeout"])
def test_reused_worker_failures_are_not_silently_retried(tmp_path, behavior):
    import os

    from reference_common.session import WorkerSession

    package = tmp_path / "fixture_reference"
    package.mkdir()
    (package / "__init__.py").touch()
    actions = {
        "raise": "raise ValueError('deliberate worker failure')",
        "exit": "os._exit(17)",
        "timeout": "time.sleep(30)",
    }
    (package / "worker.py").write_text(
        "import os, time\ndef run(request, work):\n    " + actions[behavior] + "\n"
    )
    work = tmp_path / "request"
    work.mkdir()
    request = work / "request.json"
    request.write_text("{}")
    session = WorkerSession(
        module="fixture_reference.worker", timeout=0.5 if behavior == "timeout" else 10
    )
    # Each failure names the worker's role and keeps its log, which has the cause.
    expected = {
        "raise": "(?s)replay worker failed with exit code 1:.*deliberate worker failure",
        "exit": "replay worker failed with exit code 17",
        "timeout": "replay worker exceeded its 0.5 s request timeout",
    }[behavior]
    try:
        with pytest.raises(
            TimeoutError if behavior == "timeout" else RuntimeError, match=expected
        ):
            session.execute(
                "replay", request, dict(os.environ, PYTHONPATH=str(tmp_path))
            )
        assert not session.workers
    finally:
        session.close()


def test_removing_reserve_accepts_more_drift_without_widening_tolerance():
    budget = ErrorBudget(0.02, 0.002)
    budget.check(0.0175)
    assert Fraction(0.002) + Fraction(0.0175) < Fraction(0.02)
    with pytest.raises(AssertionError):
        ErrorBudget(0.02, 0.002, 0.001).check(0.0175)
    with pytest.raises(AssertionError):
        budget.check(0.0185)


# Only these operations published bundles before budget_policy existed.
@pytest.mark.parametrize("operation", ["sdpa", "conv"])
def test_legacy_budget_limits_are_preserved_and_policy_is_validated(
    tmp_path, operation
):
    from dataclasses import asdict
    from importlib import import_module

    cli = import_module(f"{operation}_reference.cli")
    fixtures = import_module(f"test_{operation}_reference_contract")
    manifest = (
        fixtures._bundle(tmp_path) if operation == "sdpa" else fixtures._manifest()
    )
    cli.OPERATION.validate_cases(manifest, "gfx942")
    manifest.pop("budget_policy")
    for entry in manifest["cases"].values():
        budget = ErrorBudget(
            entry["budget"]["tolerance"],
            entry["budget"]["baseline_error_bound"],
            entry["budget"]["tolerance"] / 20,
        )
        entry["budget"] = asdict(budget)
        entry["comparison_limit"] = budget.comparison_limit
    cli.OPERATION.validate_cases(manifest, "gfx942")
    manifest["budget_policy"] = "max-reference-bound-v1"
    with pytest.raises(ValueError, match="budget changed"):
        cli.OPERATION.validate_cases(manifest, "gfx942")
    manifest["budget_policy"] = "unknown"
    with pytest.raises(ValueError, match="unsupported reference budget policy"):
        cli.OPERATION.validate_cases(manifest, "gfx942")


def test_manifest_cases_match_by_contract_and_ids_cannot_collide():
    from dataclasses import dataclass

    from reference_common.cases import index_cases, match_manifest_cases

    @dataclass(frozen=True)
    class Case:
        name: str
        size: int = 1

        @property
        def id(self) -> str:
            return self.name

    cases = (Case("a"), Case("b", 2))
    entries = {
        "earlier-a": {"case": {"name": "a", "size": 1}},
        "earlier-b": {"case": {"name": "b", "size": 2}},
    }
    assert match_manifest_cases(entries, cases, label="test") == {
        "a": "earlier-a",
        "b": "earlier-b",
    }
    # An id that omits a contract field must not let one case shadow another.
    with pytest.raises(ValueError, match="duplicate test case id: a"):
        index_cases((*cases, Case("a", 3)), label="test")
    with pytest.raises(ValueError, match="duplicate test case id: a"):
        match_manifest_cases(entries, (*cases, Case("a", 3)), label="test")


def test_cohort_comparison_reports_each_difference_without_raising():
    from dataclasses import dataclass

    from reference_common.cases import compare_manifest_cases

    @dataclass(frozen=True)
    class Case:
        name: str
        size: int = 1

        @property
        def id(self) -> str:
            return self.name

    entries = {
        "a": {"case": {"name": "a", "size": 1}},
        "b": {"case": {"name": "b", "size": 1}},
        "b-copy": {"case": {"name": "b", "size": 1}},
        "retired": {"case": {"name": "c", "size": 1}},
        "unrecorded": {},
    }
    comparison = compare_manifest_cases(
        entries, (Case("a"), Case("b"), Case("d", 2)), label="test"
    )
    assert comparison.keys == {"a": "a"}
    assert comparison.missing == {"d": {"name": "d", "size": 2}}
    assert comparison.extra == {"retired": {"name": "c", "size": 1}, "unrecorded": None}
    assert comparison.repeated == {"b": ["b", "b-copy"]}


def _qualified_cohort(tmp_path, operation):
    """A locked synthetic bundle recording an operation's first enrolled cohort."""
    from dataclasses import asdict
    from importlib import import_module

    from reference_common.numeric import payload_digests

    spec = import_module(f"{operation}_reference.cli").OPERATION
    architecture = spec.architectures.ARCHITECTURES[0]
    target = spec.architectures.get_architecture(architecture)
    bundle = tmp_path / "bundle"
    (bundle / "payload").mkdir(parents=True)
    (bundle / "payload/kernel.hsaco").write_bytes(b"host cohort fixture")
    entries = {}
    for case in target.CASES:
        budget = ErrorBudget(case.tolerance, 0.001)
        entries[case.id] = {
            "case": asdict(case),
            "device_target": architecture,
            "kernel": {
                "compiler_target": getattr(target, "COMPILER_TARGET", architecture)
            },
            "budget": asdict(budget),
            "comparison_limit": budget.comparison_limit,
            **spec.metric.parameters(np.ones(1)),
        }
    manifest = {
        "schema": 2,
        "budget_policy": "max-reference-bound-v1",
        "operation": manifest_operation(operation),
        "baseline_revision": "a" * 40,
        "input_generation": spec.contract.INPUT_GENERATOR,
        "files": payload_digests(bundle / "payload"),
        "cases": entries,
    }
    lock = tmp_path / "lock.json"
    _relock(bundle, manifest, lock)
    return spec, architecture, bundle, lock


def _relock(bundle, manifest, lock):
    encoded = json.dumps(manifest).encode()
    (bundle / "manifest.json").write_bytes(encoded)
    lock.write_text(
        json.dumps(
            {
                "schema": 2,
                "baseline_revision": manifest["baseline_revision"],
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    )


@pytest.mark.parametrize("operation", OPERATIONS)
def test_cases_command_lists_each_cohort_without_gpu_or_torch(tmp_path, operation):
    import os
    from dataclasses import asdict
    from importlib import import_module

    # Any import of Torch, the rocKE runtime, or production kernels fails.
    for name in ("torch", "rocke", "kernels", "dispatch"):
        (tmp_path / f"{name}.py").write_text(
            f"raise AssertionError('imported {name}')\n"
        )
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONNOUSERSITE="1")
    script = Path(__file__).with_name(f"run_{operation}_reference.py")
    registry = import_module(f"{operation}_reference.architectures")
    for architecture in registry.ARCHITECTURES:
        cases = registry.get_architecture(architecture).CASES
        result = subprocess.run(
            [sys.executable, str(script), "cases", "--arch", architecture, "--json"],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        listing = json.loads(result.stdout)
        assert (listing["operation"], listing["architecture"]) == (
            operation,
            architecture,
        )
        assert [entry["id"] for entry in listing["cases"]] == [c.id for c in cases]
        assert [entry["case"] for entry in listing["cases"]] == [
            json.loads(json.dumps(asdict(case))) for case in cases
        ]
        assert [entry["tolerance"] for entry in listing["cases"]] == [
            case.tolerance for case in cases
        ]
    result = subprocess.run(
        [sys.executable, str(script), "cases", "--arch", architecture],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert [line.split()[0] for line in lines[:-1]] == [case.id for case in cases]
    assert lines[-1].endswith(f"{architecture}: {len(cases)} enrolled cases")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("form", ["directory", "archive"])
def test_diff_reports_every_difference_that_verification_rejects(
    tmp_path, capsys, monkeypatch, operation, form
):
    from dataclasses import asdict

    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    target = spec.architectures.get_architecture(architecture)
    cases = target.CASES
    source = bundle
    if form == "archive":
        source = tmp_path / f"{architecture}.tar.gz"
        pack(bundle, source, lock, operation=operation)
    command = ["diff", "--arch", architecture, "--bundle", str(source)]
    spec.main([*command, "--lock", str(lock)])
    output = capsys.readouterr().out
    assert f"all {len(cases)} enrolled cases match" in output
    # The author dropped one qualified case and enrolled another after
    # qualification; the bundle also changed one budget and repeats a case.
    monkeypatch.setattr(target, "CASES", cases[1:])
    manifest = json.loads((bundle / "manifest.json").read_text())
    entries = manifest["cases"]
    del entries[cases[-1].id]
    budget = ErrorBudget(cases[1].tolerance * 2, 0.001)
    entries[cases[1].id].update(
        budget=asdict(budget), comparison_limit=budget.comparison_limit
    )
    entries["copy"] = entries[cases[2].id]
    _relock(bundle, manifest, lock)
    if form == "archive":
        source.unlink()
        pack(bundle, source, lock, operation=operation)
    with pytest.raises(SystemExit) as exit:
        spec.main([*command, "--lock", str(lock)])
    assert exit.value.code == 1
    output = capsys.readouterr().out
    for expected in (
        f"enrolled but not in the bundle: {cases[-1].id} (",
        f"in the bundle but not enrolled: {cases[0].id} (",
        f"recorded more than once: {cases[2].id} (",
        (
            f"tolerance or budget changed: {cases[1].id} (enrolled tolerance "
            f"{cases[1].tolerance:g}; recorded tolerance {budget.tolerance:g}"
        ),
        "needs a complete requalification",
        "library/tests/reference_common/docs/gpu-reference-workflow.md",
    ):
        assert expected in output
    assert len(spec.diff_bundle(source, lock, architecture=architecture)) == 4
    # diff reports exactly what verification refuses to run.
    with pytest.raises(ValueError):
        spec.load_bundle(bundle, lock, architecture=architecture)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_diff_authenticates_an_archive_before_reading_it(tmp_path, operation):
    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    archive = tmp_path / "bundle.tar.gz"
    pack(bundle, archive, lock, operation=operation)
    other = tmp_path / "other-lock.json"
    other.write_text(
        json.dumps(dict(json.loads(lock.read_text()), manifest_sha256="0" * 64))
    )
    with pytest.raises(ValueError, match="does not match the pinned lock"):
        spec.diff_bundle(archive, other, architecture=architecture)
    unsafe = tmp_path / "unsafe.tar.gz"
    with tarfile.open(unsafe, "w:gz") as tar:
        tar.addfile(
            tarfile.TarInfo(f"{operation}_reference_bundle/../../escaped"),
            io.BytesIO(),
        )
    with pytest.raises(ValueError, match="invalid .* archive member"):
        spec.diff_bundle(unsafe, lock, architecture=architecture)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("fetched", ["archive", "directory", None])
def test_diff_defaults_to_the_published_bundle_and_committed_lock(
    tmp_path, capsys, monkeypatch, operation, fetched
):
    from reference_common import cli as common_cli

    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    published = tmp_path / "published" / architecture
    published.parent.mkdir()
    if fetched == "archive":
        pack(
            bundle,
            published.with_name(f"{architecture}.tar.gz"),
            lock,
            operation=operation,
        )
    elif fetched == "directory":
        bundle.rename(published)
    monkeypatch.setattr(
        common_cli, "default_bundle_path", lambda *args, **kwargs: published
    )
    monkeypatch.setattr(spec.architectures, "baseline_lock", lambda name: lock)
    if fetched is None:
        with pytest.raises(SystemExit) as exit:
            spec.main(["diff", "--arch", architecture])
        assert exit.value.code == 2
        assert "fetch the archive with DVC or pass --bundle" in capsys.readouterr().err
    else:
        spec.main(["diff", "--arch", architecture])
        output = capsys.readouterr().out
        assert "enrolled cases match" in output
        assert output.rstrip().endswith(
            f"{architecture}.tar.gz" if fetched == "archive" else architecture
        )
    lock.unlink()
    with pytest.raises(SystemExit) as exit:
        spec.main(["diff", "--arch", architecture])
    assert exit.value.code == 2
    assert f"{architecture} has no published" in capsys.readouterr().err


@pytest.mark.parametrize("operation", OPERATIONS)
def test_fetched_archives_record_the_enrolled_cohorts(operation):
    """Each fetched DVC archive matches its committed lock and current cohort."""
    import re
    from importlib import import_module

    spec = import_module(f"{operation}_reference.cli").OPERATION
    registry = spec.architectures
    checked = []
    for architecture in registry.ARCHITECTURES:
        archive = Path(__file__).parent / (
            f"reference_bundles/{operation}/{architecture}.tar.gz"
        )
        pointer = archive.with_name(archive.name + ".dvc")
        if not (archive.is_file() and pointer.is_file()):
            continue
        # Skip an archive left behind by an earlier pointer; dvc pull updates it.
        recorded = re.search(r"md5: ([0-9a-f]{32})", pointer.read_text())
        digest = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
        if not recorded or digest != recorded.group(1):
            continue
        differences = spec.diff_bundle(archive, architecture=architecture)
        assert not differences, (
            f"{operation} {architecture} cohort differs from its published "
            f"bundle: {differences}"
        )
        checked.append(architecture)
    if not checked:
        pytest.skip(f"no current {operation} reference archive has been fetched")


@pytest.mark.parametrize("operation", OPERATIONS)
def test_verify_explains_how_to_unpack_an_archive(tmp_path, capsys, operation):
    import shlex

    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    archive = tmp_path / f"{architecture}.tar.gz"
    pack(bundle, archive, lock, operation=operation)
    command = ["verify", "--arch", architecture, "--lock", str(lock), "--bundle"]
    with pytest.raises(SystemExit) as exit:
        spec.main([*command, str(archive)])
    assert exit.value.code == 2
    error = capsys.readouterr().err
    assert "Traceback" not in error
    assert f"artifact.py unpack --operation {operation} " in error
    # The suggested command extracts and authenticates a usable bundle.
    suggested = next(line for line in error.splitlines() if "artifact.py" in line)
    subprocess.run(
        [sys.executable, *shlex.split(suggested)[1:]], check=True, capture_output=True
    )
    assert spec.load_bundle(tmp_path / architecture, lock, architecture=architecture)
    with pytest.raises(SystemExit) as exit:
        spec.main([*command, str(tmp_path / "missing")])
    assert exit.value.code == 2
    assert "bundle directory not found" in capsys.readouterr().err


@pytest.mark.parametrize("operation", OPERATIONS)
def test_verify_subset_runs_selected_cases_and_says_it_is_partial(
    tmp_path, capsys, monkeypatch, operation
):
    from reference_common.cli import ReferenceOperation

    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    cases = spec.architectures.get_architecture(architecture).CASES
    ran = []

    def verify_case(self, case, **kwargs):
        ran.append(case.id)
        return {"case": case.id}

    monkeypatch.setattr(ReferenceOperation, "verify_case", verify_case)
    command = ["verify", "--arch", architecture, "--bundle", str(bundle)]
    command += ["--lock", str(lock)]
    spec.main([*command, "--case", cases[2].id, "--case", cases[0].id])
    assert ran == [cases[0].id, cases[2].id]
    output = capsys.readouterr().out
    assert output.count("not a full verification") == 2
    assert "2/2 selected GPU cases passed" in output
    ran.clear()
    spec.main(command)
    assert ran == [case.id for case in cases]
    output = capsys.readouterr().out
    assert "not a full verification" not in output
    assert f"{len(cases)}/{len(cases)} required GPU cases passed" in output
    with pytest.raises(SystemExit) as exit:
        spec.main([*command, "--case", "unenrolled"])
    assert exit.value.code == 2
    assert f"not enrolled for {architecture}: unenrolled" in capsys.readouterr().err


@pytest.mark.parametrize("operation", OPERATIONS)
def test_every_reference_command_and_option_has_help(operation):
    import argparse
    from importlib import import_module

    parser = import_module(f"{operation}_reference.cli").OPERATION._parser()
    assert parser.description and parser.epilog
    (commands,) = (
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(commands.choices) == {"snapshot", "qualify", "verify", "cases", "diff"}
    summaries = {action.dest: action.help for action in commands._choices_actions}
    for name, command in commands.choices.items():
        assert summaries[name] and command.description, name
        for action in command._actions:
            assert action.help, f"{name} {action.option_strings} has no help"


_WORKFLOW_DOC = "library/tests/reference_common/docs/gpu-reference-workflow.md"


def _verifiable_case(monkeypatch, operation):
    """An operation's first case enrolled alone, with a synthetic qualified entry."""
    from dataclasses import asdict
    from importlib import import_module

    spec = import_module(f"{operation}_reference.cli").OPERATION
    architecture = spec.architectures.ARCHITECTURES[0]
    target = spec.architectures.get_architecture(architecture)
    case = target.CASES[0]
    monkeypatch.setattr(target, "CASES", (case,))
    inputs = spec.contract.make_inputs(case)
    output = encode(np.ones(case.output_shape), case.dtype)
    budget = ErrorBudget(case.tolerance, 0.001)
    entry = {
        "case": asdict(case),
        "input_digests": {name: array_digest(a) for name, a in inputs.items()},
        "output_digest": array_digest(output),
        "device_target": f"{architecture}:sramecc+:xnack-",
        "budget": asdict(budget),
        "comparison_limit": budget.comparison_limit,
        "kernel": {},
        **spec.metric.parameters(np.ones(1)),
    }
    return spec, architecture, case, {"cases": {case.id: entry}}, output


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "failure", ["baseline-output", "current-output", "qualified-target", "features"]
)
def test_verification_failures_name_side_values_cause_and_next_step(
    tmp_path, monkeypatch, operation, failure
):
    from reference_common import cli as common_cli

    spec, architecture, case, manifest, output = _verifiable_case(
        monkeypatch, operation
    )
    entry = manifest["cases"][case.id]
    changed = decode(output, case.dtype).copy()
    changed.flat[0] += 100
    changed = encode(changed, case.dtype)
    other_target = f"{architecture}:sramecc+:xnack+"
    modes = []

    def worker(request, **kwargs):
        mode = request["mode"]
        modes.append(mode)
        outputs = [output, output]
        if (failure, mode) in (
            ("baseline-output", "replay"),
            ("current-output", "source"),
        ):
            outputs = [output, changed]
        device = entry["device_target"]
        if (failure, mode) in (("qualified-target", "replay"), ("features", "source")):
            device = other_target
        return outputs, {
            "device_target": device,
            "launches": 2,
            "torch_imported": False,
        }

    monkeypatch.setattr(common_cli, "run_worker", worker)
    with pytest.raises(AssertionError if "output" in failure else ValueError) as error:
        spec.verify_case(
            case,
            bundle=tmp_path,
            manifest=manifest,
            current_root=tmp_path,
            architecture=architecture,
        )
    message = str(error.value)
    subject = f"{spec.label} {architecture} case {case.id}"
    assert subject in message
    assert "\n  likely cause: " in message and "\n  next step: " in message
    assert message.endswith(_WORKFLOW_DOC)
    if failure == "baseline-output":
        # The environment, not the current kernel, changed the frozen output.
        assert f"reference qualification failure for {subject}" in message
        assert "frozen baseline (replay) output of repetition 2 of 2" in message
        assert f"has digest {array_digest(changed)[:12]}" in message
        assert f"qualification recorded {entry['output_digest'][:12]}" in message
        assert "the current kernel did not run" in message
        assert modes == ["replay"]
    elif failure == "current-output":
        budget = ErrorBudget(**entry["budget"])
        assert "current kernel (source) output of repetition 2 of 2" in message
        assert f"GPU reference {spec.metric.name}=" in message
        assert f"exceeds remaining limit {budget.comparison_limit:.9g}" in message
        assert f"baseline_bound={budget.baseline_error_bound:.9g}" in message
        assert f"original_tolerance={case.tolerance:.9g}" in message
        assert "frozen baseline reproduced its qualified output" in message
        assert (
            f"run_{operation}_reference.py verify --arch {architecture} --bundle "
            f"{tmp_path} --current-root {tmp_path} --case {case.id}`"
        ) in message
    elif failure == "qualified-target":
        assert (
            f"frozen baseline (replay) ran on device target {other_target}, but the "
            f"bundle was qualified on {entry['device_target']}"
        ) in message
        assert "differs from the qualified target" in message
    else:
        assert (
            f"current kernel (source) ran on device target {other_target}, but the "
            f"frozen baseline (replay) ran on {entry['device_target']}"
        ) in message
        assert "different target features" in message


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("mode", ["replay", "source"])
@pytest.mark.parametrize("kind", [RuntimeError, TimeoutError])
def test_worker_failures_say_which_side_failed_and_keep_the_log(
    tmp_path, monkeypatch, operation, mode, kind
):
    from reference_common import cli as common_cli

    spec, architecture, case, manifest, output = _verifiable_case(
        monkeypatch, operation
    )
    entry = manifest["cases"][case.id]
    log = "Traceback (most recent call last):\nAttributeError: renamed_dispatch_api"

    def worker(request, **kwargs):
        if request["mode"] == mode:
            raise kind(f"GPU reference {mode} worker failed with exit code 1:\n{log}")
        return [output, output], {
            "device_target": entry["device_target"],
            "launches": 2,
            "torch_imported": False,
        }

    monkeypatch.setattr(common_cli, "run_worker", worker)
    with pytest.raises(kind) as error:
        spec.verify_case(
            case,
            bundle=tmp_path,
            manifest=manifest,
            current_root=tmp_path,
            architecture=architecture,
        )
    message = str(error.value)
    side = "frozen baseline (replay)" if mode == "replay" else "current kernel (source)"
    assert message.startswith(
        f"{spec.label} {architecture} case {case.id}: the {side} worker failed\n"
    )
    assert f"\n  see {_WORKFLOW_DOC}\n" in message
    assert message.endswith(log)
    # Only the current side points at the adapter that calls production APIs.
    adapter = f"library/tests/{operation}_reference/architectures/{architecture}/"
    assert (adapter in message) == (mode == "source")


@pytest.mark.parametrize("persistent", [False, True])
def test_reference_workers_report_missing_launches(tmp_path, persistent):
    from contextlib import nullcontext

    from reference_common.runner import run_worker
    from reference_common.session import reuse_workers

    (tmp_path / "silent_worker.py").write_text(
        """
import json
import sys
from pathlib import Path
import numpy as np

def run(request, work):
    np.savez(work / "outputs.npz", out_0=np.zeros(1))
    (work / "report.json").write_text(json.dumps({"launches": 0}))

if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
"""
    )
    context = reuse_workers("silent_worker") if persistent else nullcontext()
    expected = "source worker reported 0 launches for 1 repetitions"
    with context, pytest.raises(RuntimeError, match=expected):
        run_worker(
            {"mode": "source", "repetitions": 1},
            runner=tmp_path,
            platform=tmp_path,
            library=None,
            work=tmp_path / "work",
            module="silent_worker",
        )


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "problem", ["stale", "no-manifest", "modified", "archive", "candidate"]
)
def test_rejected_bundles_say_how_to_replace_them(
    tmp_path, monkeypatch, operation, problem
):
    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    if problem != "candidate":
        # The committed lock authenticates the published bundle.
        monkeypatch.setattr(spec.architectures, "baseline_lock", lambda name: lock)
    if problem in ("stale", "candidate"):
        # The lock now pins a newer manifest than the one extracted earlier.
        extracted = (bundle / "manifest.json").read_bytes()
        manifest = json.loads(extracted)
        manifest["baseline_revision"] = "b" * 40
        _relock(bundle, manifest, lock)
        (bundle / "manifest.json").write_bytes(extracted)
        expected = (
            "manifest does not match the pinned lock: "
            f"{bundle.resolve() / 'manifest.json'} has sha256 "
            f"{hashlib.sha256(extracted).hexdigest()[:12]}, but {lock.resolve()} pins "
            f"{json.loads(lock.read_text())['manifest_sha256'][:12]}"
        )
    elif problem == "no-manifest":
        (bundle / "manifest.json").unlink()
        expected = f"bundle {bundle.resolve()} has no manifest.json"
    elif problem == "modified":
        (bundle / "payload/kernel.hsaco").write_bytes(b"modified")
        expected = "payload has missing, modified, or extra files: 1 differ, "
        expected += "including kernel.hsaco"
    else:
        archive = tmp_path / f"{architecture}.tar.gz"
        pack(bundle, archive, lock, operation=operation)
        bundle = archive
        expected = "is a file, but --bundle needs an extracted bundle directory"
    with pytest.raises(ValueError) as error:
        spec.load_bundle(
            bundle, lock if problem == "candidate" else None, architecture=architecture
        )
    message = str(error.value)
    assert expected in message
    if problem == "candidate":
        assert "qualification-lock.json that qualification wrote" in message
        assert "dvc pull" not in message
    elif problem == "archive":
        assert f"artifact.py unpack --operation {operation} " in message
    else:
        # Unpack does not replace a directory, so the stale one goes first.
        assert "stale (left from an earlier archive)" in message
        assert (
            f"delete {bundle.resolve()}, then re-run dvc pull and artifact.py unpack:"
        ) in message
        assert f"reference_bundles/{operation}/{architecture}.tar.gz.dvc" in message
        assert f"artifact.py unpack --operation {operation} --archive " in message
        assert f"--bundle {bundle.resolve()} --lock {lock.resolve()}" in message
        assert message.endswith(_WORKFLOW_DOC)


def test_unpack_explains_a_stale_archive_or_extracted_bundle(tmp_path, operation):
    import shutil

    bundle, lock = _fixture(tmp_path, operation)
    archive = tmp_path / "earlier.tar.gz"
    pack(bundle, archive, lock, operation=operation)
    output = tmp_path / "extracted"
    unpack(archive, output, lock, operation=operation)
    # A pulled lock pins a newer manifest; the archive and extraction are stale.
    old_digest = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["baseline_revision"] = "b" * 40
    encoded = json.dumps(manifest).encode()
    (bundle / "manifest.json").write_bytes(encoded)
    newer = tmp_path / "newer-lock.json"
    new_digest = hashlib.sha256(encoded).hexdigest()
    newer.write_text(
        json.dumps(
            {"schema": 2, "baseline_revision": "b" * 40, "manifest_sha256": new_digest}
        )
    )
    with pytest.raises(ValueError, match="does not match the pinned lock") as error:
        unpack(archive, output, newer, operation=operation)
    message = str(error.value)
    assert (
        f"sha256 {old_digest[:12]}, newer-lock.json pins {new_digest[:12]}" in message
    )
    assert f"fetch that archive with: dvc pull {archive}.dvc" in message
    current = tmp_path / "current.tar.gz"
    pack(bundle, current, newer, operation=operation)
    with pytest.raises(ValueError, match="does not match the pinned lock") as error:
        unpack(current, output, newer, operation=operation)
    assert (
        f"{output} is a stale or modified extracted bundle; delete it and re-run"
    ) in str(error.value)
    shutil.rmtree(output)
    unpack(current, output, newer, operation=operation)
    validate_bundle(output, newer, operation=operation)
