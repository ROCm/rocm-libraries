# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared reference numerics, archives, worker isolation, and test selection."""

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

from reference_common.artifact import pack, unpack, validate_bundle
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
@pytest.mark.parametrize("candidate", [False, True])
def test_reference_fixture_selects_explicit_paths(
    operation, candidate, tmp_path, monkeypatch
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
    monkeypatch.setattr(hip_module, "get_device_arch", lambda: "gfx942")
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
            "gfx942",
            tmp_path,
            {"candidate": candidate},
        )
    assert calls == [(tmp_path, options["--rocke-reference-lock"], "gfx942")]
    if candidate:
        options["--rocke-reference-operation"] = (
            "conv" if operation == "sdpa" else "sdpa"
        )
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


@pytest.mark.parametrize("operation", ["sdpa", "conv"])
@pytest.mark.parametrize("selected", [False, True])
def test_reference_missing_bundle_fails_only_when_selected(
    operation, selected, tmp_path, monkeypatch
):
    from contextlib import closing
    from importlib import import_module
    from types import SimpleNamespace

    from rocke.runtime import hip_module

    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    monkeypatch.setattr(hip_module, "get_device_arch", lambda: "gfx942")
    monkeypatch.setattr(
        gpu_tests, "default_bundle_path", lambda *args, **kwargs: tmp_path / "missing"
    )
    options = {"--rocke-reference-operation": operation if selected else None}
    config = SimpleNamespace(getoption=options.get)
    with closing(gpu_tests.reference_bundle.__wrapped__(config)) as fixture:
        with pytest.raises(
            pytest.fail.Exception if selected else pytest.skip.Exception
        ):
            next(fixture)


@pytest.mark.parametrize("operation", ["sdpa", "conv"])
def test_reference_worker_requires_explicit_architecture(operation, tmp_path):
    from importlib import import_module

    worker = import_module(f"{operation}_reference.worker")
    # Reject the malformed request before runtime imports, GPU probing, or files.
    with pytest.raises(KeyError, match="architecture"):
        worker.run({}, tmp_path)


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


def test_budget_is_strict_even_at_the_float_boundary():
    budget = ErrorBudget(0.02, 0.002, 0.001)
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
        (0.02, 0.0, 0.0),
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
    try:
        with pytest.raises(TimeoutError if behavior == "timeout" else RuntimeError):
            session.execute(
                "replay", request, dict(os.environ, PYTHONPATH=str(tmp_path))
            )
        assert not session.workers
    finally:
        session.close()
