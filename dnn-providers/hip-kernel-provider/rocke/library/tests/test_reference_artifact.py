# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Common archive CLI, integrity checks, and operation isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from reference_common.artifact import pack, unpack, validate_bundle
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
    if operation == "conv":
        manifest["operation"] = "conv-fwd"
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


@pytest.mark.parametrize("operation", ["sdpa", "conv"])
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
    repeated = tmp_path / "repeated.tar.gz"
    result = _run(
        "pack",
        "--operation",
        operation,
        "--bundle",
        bundle,
        "--lock",
        lock,
        "--archive",
        repeated,
    )
    assert result.returncode == 0, result.stderr
    assert archive.read_bytes() == repeated.read_bytes()
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


@pytest.mark.parametrize("operation", ["sdpa", "conv"])
@pytest.mark.parametrize("command", ["pack", "validate"])
def test_common_cli_rejects_wrong_operation_even_with_matching_lock(
    tmp_path, operation, command
):
    bundle, lock = _fixture(tmp_path, operation)
    other = "conv" if operation == "sdpa" else "sdpa"
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


@pytest.fixture(params=["sdpa", "conv"])
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


@pytest.mark.parametrize("operation", ["sdpa", "conv"])
@pytest.mark.parametrize("layout", ["source", "bin/hip_kernel_provider", "standalone"])
def test_bundle_lookup_stays_in_operation_and_architecture_domain(
    tmp_path, operation, layout
):
    from importlib import import_module

    paths = import_module(f"{operation}_reference.paths")
    if layout == "source":
        tests = tmp_path / "rocke/library/tests"
        expected = tests / "reference_bundles" / operation
    else:
        root = tmp_path / layout
        tests = root / "tests/library/tests"
        (tests / "reference_bundles" / operation / "gfx942").mkdir(parents=True)
        expected = root / "engines/test_arch_content/rocke" / operation
    assert paths.default_bundle_path(tests, "gfx942") == expected / "gfx942"
    assert paths.default_bundle_path(tests, "gfx950") == expected / "gfx950"


def test_published_bundles_have_supported_cohorts_and_committed_locks():
    """Publication enrollment must not accidentally require unqualified data."""
    from reference_common.artifact import OPERATIONS

    tests = Path(__file__).parent
    published = json.loads(
        (tests / "reference_common/published_bundles.json").read_text()
    )
    assert isinstance(published, dict)
    assert set(published) == set(OPERATIONS)
    for operation, architectures in published.items():
        assert isinstance(architectures, list)
        assert len(architectures) == len(set(architectures))
        root = tests / f"{operation}_reference/architectures"
        supported = json.loads((root / "registry.json").read_text())
        assert set(architectures) <= set(supported)
        for architecture in architectures:
            lock = json.loads((root / architecture / "baseline_lock.json").read_text())
            assert lock["schema"] == 2
            assert len(bytes.fromhex(lock["baseline_revision"])) == 20
            assert len(bytes.fromhex(lock["manifest_sha256"])) == 32


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

    runtime = tmp_path / "rocke/runtime"
    runtime.mkdir(parents=True)
    (runtime.parent / "__init__.py").touch()
    (runtime / "__init__.py").touch()
    (runtime / "hip_module.py").write_text("def get_device_arch(): return 'gfx942'\n")
    (tmp_path / "conftest.py").write_text(
        """
from reference_common.pytest_support import start_reference_session, select_reference_items
def pytest_addoption(parser):
    parser.addoption("--rocke-reference-arch")
    parser.addoption("--rocke-reference-operation")
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
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).parent)])
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


def test_reference_entry_skips_a_different_gpu_lane():
    from reference_common.pytest_support import architecture_available

    assert not architecture_available(
        "gfx942", ("gfx942", "gfx94x"), "gfx950", "gfx950-dcgpu"
    )


@pytest.mark.parametrize("device", [None, "gfx950"], ids=["missing-gpu", "wrong-gpu"])
def test_reference_selected_target_requires_matching_hardware(device):
    from reference_common.pytest_support import architecture_available

    with pytest.raises(RuntimeError, match="required gfx942 GPU unavailable"):
        architecture_available(
            "gfx942", ("gfx942", "gfx94x"), device, "gfx94X-dcgpu", "gfx942"
        )


def test_reference_exact_targets_override_the_family():
    from reference_common.pytest_support import architecture_available

    # Same family, different artifact target: do not require the gfx942 bundle.
    assert not architecture_available(
        "gfx942", ("gfx942", "gfx94x"), "gfx941", "gfx94X-dcgpu", "gfx941"
    )
    # Downloading both artifacts does not mean both devices are present.
    assert not architecture_available(
        "gfx942", ("gfx942", "gfx94x"), "gfx941", "gfx94X-dcgpu", "gfx941,gfx942"
    )


def test_reference_local_run_without_a_gpu_fails():
    from reference_common.pytest_support import architecture_available

    with pytest.raises(RuntimeError, match="required gfx942 GPU unavailable"):
        architecture_available("gfx942", ("gfx942", "gfx94x"), None, "")


@pytest.mark.parametrize("operation", ["sdpa", "conv"])
def test_reference_worker_requires_explicit_architecture(operation, tmp_path):
    from importlib import import_module

    worker = import_module(f"{operation}_reference.worker")
    # Reject the malformed request before runtime imports, GPU probing, or files.
    with pytest.raises(KeyError, match="architecture"):
        worker.run({}, tmp_path)
