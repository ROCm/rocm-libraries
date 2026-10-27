# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared reference numerics, archives, CLI, worker isolation, and test selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from fractions import Fraction
from itertools import pairwise
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from reference_common.artifact import (
    OPERATIONS,
    InconsistentLockError,
    cohort_digest,
    display_command,
    manifest_cohort_digest,
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


def _arguments(command):
    """Split a command that ``display_command`` showed back into its arguments.

    ``shlex.split`` would treat the backslashes of a Windows path as escapes.
    """
    found = re.findall(r'"([^"]*)"|(\S+)', command)
    return [quoted or plain for quoted, plain in found]


def _commands(message):
    """The commands that a hint shows, each split into its arguments.

    A command starts at ``dvc pull`` or at a Python script that a Python
    interpreter runs, and runs to the end of its line or to ``&&``, whatever
    prose comes before it; a full stop after it is prose. The interpreter is
    not part of the command. Prose that only names a command, such as "dvc
    pull and artifact.py unpack:", neither runs a script nor ends one argument
    after ``dvc pull``.
    """
    commands = []
    for line in message.splitlines():
        parts = [[]]
        for argument in _arguments(line):
            if argument == "&&":
                parts.append([])
            else:
                parts[-1].append(argument)
        for words in parts:
            # A full stop may follow a quoted argument or end the last one.
            if words[-1:] == ["."]:
                words.pop()
            elif words:
                words[-1] = words[-1].removesuffix(".")
            for at, word in enumerate(words):
                pull = words[at : at + 2] == ["dvc", "pull"] and len(words) == at + 3
                interpreter = Path(words[at - 1]).name if at else ""
                run = word.endswith(".py") and interpreter.lower().startswith("python")
                if pull or run:
                    commands.append(words[at:])
                    break
    return commands


def _unpacked(command):
    """Parse a shown artifact.py command with artifact.py's own parser."""
    from reference_common import artifact

    script, *arguments = command
    assert _same(script, artifact.__file__)
    return artifact._parser().parse_args(arguments)


def _passed(message, option):
    """The path that a hint's last step outside its commands passes to ``option``."""
    passed = []
    for line in message.splitlines():
        if not _commands(line):
            pairs = pairwise(_arguments(line))
            passed += [after for word, after in pairs if word == option]
    assert passed, f"no step passes {option}"
    return passed[-1].rstrip(".")


def _step(message):
    """The next step that a triaged failure gives, through the end of the message."""
    parts = re.split("next step:", message, maxsplit=1, flags=re.IGNORECASE)
    assert len(parts) == 2, "no next step"
    return parts[1]


def _deletes(step, path):
    """Whether a step says to delete ``path``, before any command it shows."""
    lead = step.lstrip().splitlines()[0]
    deletes = re.search(r"(?<!not )(?<!n't )\b(delete|remove)\b", lead, re.IGNORECASE)
    return bool(deletes) and _mentions(lead, path)


def _same(shown, path):
    """Whether a path that a hint shows, absolute or relative, is ``path``."""
    return Path(shown).resolve() == Path(path).resolve()


def _mentions(text, path):
    """Whether prose names ``path``, absolute or relative to the working directory.

    The name must end where the path does, so a directory is not found in the
    name of the archive beside it.
    """
    path = Path(path)
    forms = {str(path), str(path.resolve())}
    try:
        forms.add(os.path.relpath(path.resolve()))
    except ValueError:  # Windows has no path from one drive to another.
        pass
    return any(
        re.search(re.escape(form) + r"(?![\w\\/-]|\.\w)", text) for form in forms
    )


def _outcome(call):
    """Run ``call`` and return how it ended, with the message it ended with.

    ``pytest.raises(pytest.fail.Exception)`` lets a skip escape, so the test
    reports a skip, and accepts an xfail, which subclasses Failed. Either would
    let a required reference that compared nothing stay green, so this names
    the exact outcome: ``fail``, ``xfail``, ``skip``, or ``pass``. A fail or
    skip raised inside an exception handler would report that error again, so
    none may be.
    """
    try:
        call()
    except (pytest.fail.Exception, pytest.skip.Exception) as error:
        assert error.__context__ is None
        kinds = {
            pytest.fail.Exception: "fail",
            pytest.xfail.Exception: "xfail",
            pytest.skip.Exception: "skip",
        }
        return kinds[type(error)], str(error)
    return "pass", ""


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


def test_published_lists_each_committed_lock_without_site_packages():
    """Tools read the published variants and their paths with the standard library."""
    from importlib import import_module

    result = _run("published", "--json")
    assert result.returncode == 0, result.stderr
    entries = json.loads(result.stdout)
    slots = [(e["operation"], e["architecture"], e["variant"]) for e in entries]
    assert slots == _published_variants()
    tests = Path(__file__).resolve().parent
    for (operation, architecture, variant), entry in zip(slots, entries):
        registry = import_module(f"{operation}_reference.architectures")
        assert Path(entry["lock"]) == registry.baseline_lock(architecture, variant)
        archive = tests / "reference_bundles" / operation / architecture
        archive /= f"{variant}.tar.gz"
        assert Path(entry["archive"]) == archive
        assert Path(entry["pointer"]) == archive.with_name(f"{archive.name}.dvc")
    result = _run("published")
    assert result.stdout.splitlines() == [" ".join(slot) for slot in slots]
    for operation in OPERATIONS:
        result = _run("published", "--operation", operation)
        assert result.stdout.splitlines() == [
            " ".join(slot) for slot in slots if slot[0] == operation
        ]


def test_published_lists_locks_in_enrollment_order(tmp_path):
    """A committed lock publishes its variant; an adapter alone does not.

    A lock under an architecture that the registry does not enroll, which
    configuration rejects, is listed after the rest rather than hidden.
    """
    tests = tmp_path / "tests"
    script = tests / "reference_common/artifact.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(_SCRIPT.read_bytes())
    root = tests / "gemm_reference/architectures"
    root.mkdir(parents=True)
    (root / "registry.json").write_text('["gfx950", "gfx942"]\n')
    slots = (
        "gfx942/wide",
        "gfx942/narrow",
        "gfx950/wide",
        "gfx950/draft",
        "gfx90a/wide",
    )
    for slot in slots:
        (root / slot).mkdir(parents=True)
        (root / slot / "__init__.py").touch()
        if slot != "gfx950/draft":
            (root / slot / "baseline_lock.json").write_text("{}\n")
    result = _run("published", script=script)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "gemm gfx950 wide",
        "gemm gfx942 narrow",
        "gemm gfx942 wide",
        "gemm gfx90a wide",
    ]
    # The bundle commands still need a bundle and its lock.
    result = _run("validate", "--operation", "gemm", script=script)
    assert result.returncode == 2
    assert "required for validate: --bundle, --lock" in result.stderr


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


@pytest.mark.parametrize("declared", [None, "family", "other"])
def test_common_cli_accepts_only_the_selected_declared_variant(
    tmp_path, operation, declared
):
    """A manifest from before variants declares none; its lock identifies it."""
    bundle, lock = _fixture(tmp_path, operation)
    if declared is not None:
        manifest = json.loads((bundle / "manifest.json").read_text())
        _relock(bundle, dict(manifest, variant=declared), lock)
    archive = tmp_path / "bundle.tar.gz"
    # Without a selected variant, any declared one is accepted.
    pack(bundle, archive, lock, operation=operation)
    for command, paths in (
        ("validate", ["--bundle", bundle]),
        ("unpack", ["--archive", archive, "--bundle", tmp_path / "extracted"]),
    ):
        result = _run(
            command,
            "--operation",
            operation,
            "--variant",
            "family",
            *paths,
            "--lock",
            lock,
        )
        if declared == "other":
            assert result.returncode != 0
            assert "bundle variant 'other' does not match 'family'" in result.stderr
            # The lock pins this archive, so it is not called the wrong one.
            assert "is not the archive that" not in result.stderr
        else:
            assert result.returncode == 0, result.stderr
    assert (tmp_path / "extracted").exists() == (declared != "other")


def test_archive_cannot_be_unpacked_as_another_operation(tmp_path):
    bundle, lock = _fixture(tmp_path, "conv")
    archive = tmp_path / "conv.tar.gz"
    pack(bundle, archive, lock, operation="conv")
    with pytest.raises(ValueError, match="invalid SDPA archive member"):
        unpack(archive, tmp_path / "wrong", lock, operation="sdpa")
    assert not (tmp_path / "wrong").exists()


def test_python_api_rejects_invalid_operation_before_writing(tmp_path):
    bundle, lock = _fixture(tmp_path, "conv")
    archive = tmp_path / "missing/bad.tar.gz"
    with pytest.raises(ValueError, match="unsupported reference operation"):
        pack(bundle, archive, lock, operation="../escape")
    # Not even the archive's directory is created.
    assert not archive.parent.exists()


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


def test_pack_creates_the_archive_directory(tmp_path, operation):
    """A processor's first variant has no reference_bundles/<op>/<arch>/ yet."""
    bundle, lock = _fixture(tmp_path, operation)
    archive = tmp_path / f"reference_bundles/{operation}/gfx942/family.tar.gz"
    pack(bundle, archive, lock, operation=operation)
    output = tmp_path / "staged"
    unpack(archive, output, lock, operation=operation)
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


def test_cohort_digest_binds_contracts_inputs_and_tolerances():
    """Only what a lock certifies changes the digest; ids, order and outputs do not."""
    entries = {
        "a": {
            "case": {"name": "a", "shape": [1, 2]},
            "input_digests": {"q": "1" * 64},
            "budget": {"tolerance": 0.01, "baseline_error_bound": 0.001},
            "output_digest": "2" * 64,
        },
        "b": {
            "case": {"name": "b", "shape": [3, 4]},
            "input_digests": {"q": "3" * 64},
            "budget": {"tolerance": 0.02, "baseline_error_bound": 0.002},
            "output_digest": "4" * 64,
        },
    }
    digest = manifest_cohort_digest({"cases": entries})
    # The enrolled side digests the same rows built from each case.
    assert digest == cohort_digest(
        {
            "case": entry["case"],
            "input_digests": entry["input_digests"],
            "tolerance": entry["budget"]["tolerance"],
        }
        for entry in entries.values()
    )
    # Earlier id formats, case order, and another qualification's outputs and
    # bounds keep the cohort, so processors sharing cases share its digest.
    renamed = {f"old-{key}": entry for key, entry in reversed(entries.items())}
    assert manifest_cohort_digest({"cases": renamed}) == digest
    requalified = json.loads(json.dumps(entries))
    requalified["a"]["output_digest"] = "5" * 64
    requalified["a"]["budget"]["baseline_error_bound"] = 0.003
    assert manifest_cohort_digest({"cases": requalified}) == digest
    for (field, key), value in (
        (("case", "shape"), [1, 3]),
        (("input_digests", "q"), "6" * 64),
        (("budget", "tolerance"), 0.03),
    ):
        changed = json.loads(json.dumps(entries))
        changed["a"][field][key] = value
        assert manifest_cohort_digest({"cases": changed}) != digest, field
    del renamed["old-b"]
    assert manifest_cohort_digest({"cases": renamed}) != digest


def test_lock_pins_the_cohort_that_its_manifest_records(tmp_path, operation):
    bundle, lock = _fixture(tmp_path, operation)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["cases"] = {
        "a": {
            "case": {"name": "a"},
            "input_digests": {"q": "1" * 64},
            "budget": {"tolerance": 0.01},
        }
    }
    encoded = json.dumps(manifest).encode()
    (bundle / "manifest.json").write_bytes(encoded)
    pinned = {
        "schema": 2,
        "baseline_revision": manifest["baseline_revision"],
        "cohort_sha256": manifest_cohort_digest(manifest),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    lock.write_text(json.dumps(pinned))
    validate_bundle(bundle, lock, operation=operation)
    archive = tmp_path / "bundle.tar.gz"
    pack(bundle, archive, lock, operation=operation)
    # A candidate lock from before the digest still authenticates its bundle.
    lock.write_text(
        json.dumps({k: v for k, v in pinned.items() if k != "cohort_sha256"})
    )
    validate_bundle(bundle, lock, operation=operation)
    # A lock mixed from two qualifications, or a digest edited to follow an
    # unqualified cohort change, matches no archive.
    lock.write_text(json.dumps(dict(pinned, cohort_sha256="0" * 64)))
    with pytest.raises(
        ValueError, match="pins a cohort that its manifest does not record"
    ):
        validate_bundle(bundle, lock, operation=operation)
    output = tmp_path / "extracted"
    with pytest.raises(ValueError) as error:
        unpack(archive, output, lock, operation=operation)
    message = str(error.value)
    assert (
        f"cases digest {pinned['cohort_sha256'][:12]}, lock.json pins "
        f"cohort_sha256 {'0' * 12}"
    ) in message
    assert "a cohort change needs requalification" in message
    # The archive holds the pinned manifest, so no fetch is suggested.
    assert "is not the archive" not in message
    assert not output.exists()


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
    """Each variant has its own bundle below its processor's directory."""
    from reference_common.paths import default_bundle_path

    if layout == "source":
        tests = tmp_path / "rocke/library/tests"
        expected = tests / "reference_bundles" / operation
    else:
        root = tmp_path / layout
        tests = root / "tests/library/tests"
        (tests / "reference_bundles" / operation / "gfx942/first").mkdir(parents=True)
        expected = root / "engines/test_arch_content/rocke" / operation
    slots = [("gfx942", "first"), ("gfx942", "second"), ("gfx950", "first")]
    for architecture, variant in slots:
        path = default_bundle_path(
            tests, architecture, operation=operation, variant=variant
        )
        assert path == expected / architecture / variant


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
    parser.addoption("--rocke-reference-variant")
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


_VARIANT_SESSION_CONFTEST = """
from reference_common.pytest_support import (
    select_reference_items,
    start_reference_session,
)
from rocke.runtime import hip_module

hip_module.get_device_target_id = lambda device=0: {architecture!r}


def pytest_addoption(parser):
    for option in ("arch", "bundle", "lock", "operation", "variant"):
        parser.addoption(f"--rocke-reference-{{option}}")


def pytest_sessionstart(session):
    start_reference_session(session.config)


def pytest_collection_modifyitems(config, items):
    select_reference_items(config, items)
"""
# The shared pinned tests of a synthetic operation, whose bundles are missing.
_VARIANT_SESSION_TESTS = """
from dataclasses import replace
from pathlib import Path

import pytest

from fictional_reference import architectures
from reference_common.pinned_tests import pinned_reference_tests, select_bundle
from {operation}_reference.cli import OPERATION, load_bundle

OPERATION = replace(OPERATION, name="fictional", architectures=architectures)


@pytest.fixture(scope="module")
def reference_bundle(pytestconfig):
    missing = Path(__file__).with_name("missing")
    yield from select_bundle(
        pytestconfig, OPERATION, lambda *args, **kwargs: missing, load_bundle
    )


(
    test_correctness_against_qualified_rocke,
    test_rejects_perturbed_gpu_results,
    test_rejects_missing_gpu_launch,
) = pinned_reference_tests(OPERATION, perturbation=1.0)
"""


@pytest.mark.parametrize("operation", OPERATIONS[:1])
@pytest.mark.parametrize("variant", [None, "narrow"])
def test_reference_pytest_selects_variants_by_id(
    second_variant, tmp_path, operation, variant
):
    """A session of the shared pinned tests over a synthetic second variant.

    Ids name each slot's architecture and variant. Selecting the operation and
    architecture runs every variant there: the published one fails on its
    missing bundle and the unpublished one skips. Selecting a variant
    deselects the others, so its unpublished slot fails instead of passing
    with nothing run. This does not execute GPU kernels.
    """
    import os
    from xml.etree import ElementTree

    import rocke

    registry = second_variant.architectures
    architecture = registry.ARCHITECTURES[0]
    registry.baseline_lock(architecture, "wide").write_text("{}")
    (tmp_path / "conftest.py").write_text(
        _VARIANT_SESSION_CONFTEST.format(architecture=architecture)
    )
    tests = tmp_path / "test_fictional_pinned_reference.py"
    tests.write_text(_VARIANT_SESSION_TESTS.format(operation=operation))
    # The missing-launch control imports the operation's worker module.
    (tmp_path / "fictional_reference/worker.py").write_text(
        f"from {operation}_reference.worker import run\n"
    )
    env = dict(os.environ, AMDGPU_TARGETS=architecture)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(rocke.__file__).resolve().parent.parent),
            str(Path(__file__).parent),
            str(tmp_path),
        ]
    )
    selection = [] if variant is None else ["--rocke-reference-variant", variant]
    report = tmp_path / "report.xml"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests), "-q", "-p", "no:cacheprovider"]
        + [f"--junitxml={report}", "--rocke-reference-operation", "fictional"]
        + ["--rocke-reference-arch", architecture, *selection],
        check=False,
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    outcomes = {
        case.get("name"): next(
            (child.tag for child in case if child.tag in ("failure", "skipped")),
            "passed",
        )
        for case in ElementTree.parse(report).iter("testcase")
    }

    def slot(name):
        prefix = f"{architecture}-{name}"
        return [
            f"test_correctness_against_qualified_rocke[{prefix}-{case.id}]"
            for case in registry.get_adapter(architecture, name).CASES
        ] + [
            f"test_rejects_perturbed_gpu_results[{prefix}-source]",
            f"test_rejects_perturbed_gpu_results[{prefix}-replay]",
            f"test_rejects_missing_gpu_launch[{prefix}]",
        ]

    expected = dict.fromkeys(
        slot("narrow"), "skipped" if variant is None else "failure"
    )
    if variant is None:
        expected.update(dict.fromkeys(slot("wide"), "failure"))
    # The other processor's slots are deselected, not skipped.
    assert outcomes == expected


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
    variant = gpu_tests.OPERATION.architectures.resolve_variant(architecture)
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

    def load(bundle, lock, *, architecture, variant):
        calls.append((bundle, lock, architecture, variant))
        return {"candidate": candidate}

    monkeypatch.setattr(gpu_tests, "load_bundle", load)
    config = SimpleNamespace(getoption=options.get)
    registry = gpu_tests.OPERATION.architectures
    with closing(gpu_tests.reference_bundle.__wrapped__(config)) as fixture:
        open_slot = next(fixture)
        if not (candidate or registry.baseline_lock(architecture, variant).is_file()):
            # An unpublished variant has no lock for its default bundle.
            with pytest.raises(pytest.skip.Exception, match="^not published: "):
                open_slot(architecture, variant)
            assert not calls
            return
        target, bundle, manifest = open_slot(architecture, variant)
        assert target is gpu_tests.OPERATION.architectures.get_adapter(
            architecture, variant
        )
        assert (target.NAME, bundle, manifest) == (
            architecture,
            tmp_path,
            {"candidate": candidate},
        )
        # The open slot is reused, not loaded again.
        assert open_slot(architecture, variant)[1:] == (bundle, manifest)
    lock = options["--rocke-reference-lock"]
    assert calls == [(tmp_path, lock, architecture, variant)]
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
            {"--rocke-reference-operation": "sdpa", "--rocke-reference-variant": "x"},
            "variant selection requires explicit operation and architecture",
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


@pytest.fixture
def second_variant(tmp_path, monkeypatch, operation):
    """The operation with a synthetic second variant, enrolled under tmp_path.

    ``fictional_reference.architectures`` copies the operation's registry
    module and enrolls its first two processors. The first enrolls ``wide``,
    with the first processor's real cases, and ``narrow``, without the first
    of them; the second enrolls ``wide`` with the same cases, so the two
    ``wide`` slots share a cohort. A variant is published once a test writes
    its lock. Pytest options name the package's operation ``fictional``; the
    returned operation keeps the real name, so its manifests are the real ones.
    """
    import shutil
    from dataclasses import replace
    from importlib import import_module

    spec = import_module(f"{operation}_reference.cli").OPERATION
    real = Path(spec.architectures.__file__).resolve().parent
    first, second = spec.architectures.ARCHITECTURES[:2]
    cases = spec.architectures.get_adapter(first).__name__
    root = tmp_path / "fictional_reference/architectures"
    root.mkdir(parents=True)
    (root.parent / "__init__.py").touch()
    shutil.copy(real / "__init__.py", root)
    (root / "registry.json").write_text(json.dumps([first, second]))
    for architecture in (first, second):
        (root / architecture).mkdir()
        shutil.copy(real / architecture / "__init__.py", root / architecture)
    for slot, cohort in (
        (f"{first}/narrow", "CASES = CASES[1:]\n"),
        (f"{first}/wide", ""),
        (f"{second}/wide", ""),
    ):
        (root / slot).mkdir()
        (root / slot / "__init__.py").write_text(
            f"from {cases} import CASES\nfrom .. import NAME\n\n{cohort}"
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        architectures = import_module("fictional_reference.architectures")
        yield replace(spec, architectures=architectures)
    finally:
        for name in [name for name in sys.modules if name.startswith("fictional_")]:
            del sys.modules[name]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "processor,selection,problem",
    [
        (0, {"variant": "missing"}, "reference variant is not enrolled: missing"),
        (0, {"bundle": "candidate"}, "needs a selected reference variant"),
        (0, {"variant": "narrow", "bundle": "candidate", "lock": "lock.json"}, None),
        (0, {"variant": "wide"}, None),
        (1, {"bundle": "candidate"}, None),
    ],
    ids=["unenrolled", "ambiguous-override", "override", "variant", "sole-variant"],
)
def test_reference_variant_option_selects_one_enrolled_variant(
    second_variant, monkeypatch, processor, selection, problem
):
    """A selected variant must be enrolled; an override needs one variant.

    An architecture with one variant selects it by default, so its override
    needs no ``--rocke-reference-variant``. Each error lists the choices.
    """
    from types import SimpleNamespace

    from reference_common.pytest_support import start_reference_session
    from rocke.runtime import hip_module

    architecture = second_variant.architectures.ARCHITECTURES[processor]
    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: architecture
    )
    monkeypatch.delenv("AMDGPU_TARGETS", raising=False)
    options = {
        "--rocke-reference-operation": "fictional",
        "--rocke-reference-arch": architecture,
    }
    options.update(
        (f"--rocke-reference-{name}", value) for name, value in selection.items()
    )
    config = SimpleNamespace(getoption=options.get)
    if problem is None:
        start_reference_session(config)
        return
    with pytest.raises(pytest.UsageError) as error:
        start_reference_session(config)
    assert str(error.value) == (
        f"--rocke-reference-variant: {second_variant.label} {architecture} "
        f"{problem}; enrolled: narrow, wide"
    )


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "published,bundled,options,variant,outcome,expected",
    [
        pytest.param(("wide",), True, {}, "wide", "run", None, id="published-runs"),
        pytest.param(
            ("wide",),
            False,
            {"operation": True},
            "wide",
            "fail",
            "required {arch} {name} reference bundle is missing",
            id="published-missing-fails-when-selected",
        ),
        pytest.param(
            ("wide",),
            False,
            {},
            "wide",
            "skip",
            "no extracted {arch} {name} reference bundle at",
            id="published-missing-skips-locally",
        ),
        pytest.param(
            ("wide",),
            True,
            {},
            "narrow",
            "skip",
            "not published: the {arch} narrow {name} reference is unpublished",
            id="unpublished-skips-locally",
        ),
        pytest.param(
            ("wide",),
            True,
            {"operation": True},
            "narrow",
            "skip",
            "not published: the {arch} narrow {name} reference is unpublished",
            id="unpublished-skips-beside-a-published-variant",
        ),
        pytest.param(
            ("wide",),
            True,
            {"operation": True, "variant": "narrow"},
            "narrow",
            "fail",
            "--rocke-reference-variant selects narrow, but the {arch} narrow "
            "{name} reference is unpublished",
            id="unpublished-fails-when-selected",
        ),
        pytest.param(
            (),
            True,
            {"operation": True, "variant": "narrow", "bundle": True, "lock": True},
            "narrow",
            "run",
            None,
            id="unpublished-candidate-runs",
        ),
        pytest.param(
            (),
            True,
            {"operation": True, "variant": "narrow", "bundle": True},
            "narrow",
            "fail",
            "the {arch} narrow {name} reference is unpublished, so no committed "
            "lock authenticates the candidate bundle",
            id="unpublished-candidate-needs-its-lock",
        ),
        pytest.param(
            (),
            True,
            {"operation": True},
            "wide",
            "fail",
            "no {arch} {name} reference variant is published, so the selected "
            "{arch} suite would run nothing",
            id="nothing-published-fails-when-selected",
        ),
        pytest.param(
            ("narrow", "wide"),
            True,
            {"operation": True, "variant": "narrow"},
            "wide",
            "skip",
            "not selected: --rocke-reference-variant narrow selects another {arch} "
            "variant",
            id="another-variant-selected-skips",
        ),
    ],
)
def test_reference_fixture_runs_a_variant_by_its_publication(
    second_variant,
    tmp_path,
    monkeypatch,
    published,
    bundled,
    options,
    variant,
    outcome,
    expected,
):
    """Each variant of the GPU's processor runs, skips, or fails on its own state.

    ``published`` variants have a lock, and ``bundled`` says whether their
    shared default bundle exists. A published variant runs its default bundle,
    which an explicit selection requires. An unpublished one never reads it:
    it runs only a candidate bundle and lock, and a selection that would run no
    variant fails instead of passing empty.
    """
    from contextlib import closing
    from types import SimpleNamespace

    from reference_common.pinned_tests import select_bundle
    from rocke.runtime import hip_module

    spec = second_variant
    registry = spec.architectures
    architecture = registry.ARCHITECTURES[0]
    for name in published:
        registry.baseline_lock(architecture, name).write_text("{}")
    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: architecture
    )
    default = tmp_path / "default"
    if bundled:
        default.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    lock = tmp_path / "qualification-lock.json"
    selected = {
        "--rocke-reference-operation": spec.name if options.get("operation") else None,
        "--rocke-reference-variant": options.get("variant"),
        "--rocke-reference-bundle": candidate if options.get("bundle") else None,
        "--rocke-reference-lock": lock if options.get("lock") else None,
    }
    loads = []

    def load(bundle, lock, *, architecture, variant):
        loads.append((bundle, lock, architecture, variant))
        return {}

    config = SimpleNamespace(getoption=selected.get)
    fixture = select_bundle(config, spec, lambda *args, **kwargs: default, load)
    with closing(fixture):
        open_slot = next(fixture)
        if outcome == "run":
            target = open_slot(architecture, variant)[0]
            assert target is registry.get_adapter(architecture, variant)
            run = (candidate, lock) if options.get("bundle") else (default, None)
            assert loads == [(*run, architecture, variant)]
            return
        with pytest.raises(
            pytest.fail.Exception if outcome == "fail" else pytest.skip.Exception
        ) as result:
            open_slot(architecture, variant)
    assert not loads
    assert expected.format(arch=architecture, name=spec.name.upper()) in str(
        result.value
    )


def _relocate(monkeypatch, tmp_path, operation, layout):
    """Place an operation's package in a source checkout or an installed test tree.

    Recovery hints depend on the tree around the package, so a test that
    relocates it checks each layout's hint from either layout.
    """
    from importlib import import_module

    tests = tmp_path / (
        "bin/hip_kernel_provider/tests/library/tests"
        if layout == "installed"
        else "rocke/library/tests"
    )
    contract = import_module(f"{operation}_reference.contract")
    monkeypatch.setattr(
        contract, "__file__", str(tests / f"{operation}_reference/contract.py")
    )
    return tests.resolve()


def _track(tests, operation, architecture, variant):
    """Add the DVC pointer that a source checkout keeps for a published archive."""
    archives = tests / "reference_bundles" / operation / architecture
    pointer = archives / f"{variant}.tar.gz.dvc"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text("outs:\n")
    return pointer


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "index,published",
    [(0, True), (-1, True), (-1, False)],
    ids=["first", "last", "unpublished"],
)
@pytest.mark.parametrize("layout", ["source", "installed"])
@pytest.mark.parametrize("selected", [False, True])
def test_reference_missing_bundle_fails_only_when_selected(
    operation, index, published, layout, selected, tmp_path, monkeypatch
):
    from contextlib import closing
    from importlib import import_module
    from types import SimpleNamespace

    from rocke.runtime import hip_module

    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    spec = gpu_tests.OPERATION
    architecture = spec.architectures.ARCHITECTURES[index]
    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: architecture
    )
    missing = tmp_path / "missing"
    monkeypatch.setattr(
        gpu_tests, "default_bundle_path", lambda *args, **kwargs: missing
    )
    tests = _relocate(monkeypatch, tmp_path, operation, layout)
    variant = spec.architectures.resolve_variant(architecture)
    # A published pair has a committed lock and, in a source checkout, a DVC
    # pointer beside the default bundle. The test decides which pairs those
    # are, so publishing another architecture cannot remove a case.
    locks = tmp_path / "locks"
    monkeypatch.setattr(
        spec.architectures,
        "baseline_lock",
        lambda name, variant=None: locks / name / "baseline_lock.json",
    )
    lock = spec.architectures.baseline_lock(architecture, variant)
    archives = tests / "reference_bundles" / operation / architecture
    pointer = archives / f"{variant}.tar.gz.dvc"
    if published:
        lock.parent.mkdir(parents=True)
        lock.write_text("{}\n")
        if layout == "source":
            _track(tests, operation, architecture, variant)
    options = {"--rocke-reference-operation": operation if selected else None}
    config = SimpleNamespace(getoption=options.get)
    with closing(gpu_tests.reference_bundle.__wrapped__(config)) as fixture:
        open_slot = next(fixture)
        outcome, message = _outcome(lambda: open_slot(architecture, variant))
    # A selected suite without its bundle must fail, so that it cannot pass
    # in CI without running a comparison.
    assert outcome == ("fail" if selected else "skip")
    first = message.splitlines()[0]
    if not published:
        # Only a candidate can verify an unpublished variant, and selecting an
        # architecture that publishes no variant must not pass with nothing run.
        assert ("would run nothing" if selected else "unpublished") in first
        assert architecture in first and operation.upper() in first
        if not selected:
            assert variant in first and _mentions(first, lock)
        assert "--rocke-reference-bundle" in message
        assert "--rocke-reference-lock" in message
        assert _commands(message) == [] and "dvc pull" not in message
        return
    key = "reference bundle is missing" if selected else "no extracted"
    assert _documented(key) and key in first
    assert architecture in first and operation.upper() in first
    assert _mentions(first, missing)
    commands = _commands(message)
    installed = tmp_path / "bin/hip_kernel_provider/engines/test_arch_content/rocke"
    installed /= f"{operation}/{architecture}/{variant}"
    if layout == "installed":
        # Installation stages published bundles; there is nothing to fetch.
        assert commands == [] and "dvc pull" not in message
        assert _mentions(message, installed)
        assert "ROCKE_INSTALL_TEST_GPU_REFERENCES=ON" in message
    else:
        pull, extract = commands
        assert pull[:2] == ["dvc", "pull"] and _same(pull[2], pointer)
        args = _unpacked(extract)
        assert (args.command, args.operation) == ("unpack", operation)
        # The archive that the pointer tracks unpacks into the missing bundle.
        assert _same(args.archive, pointer.with_suffix(""))
        assert _same(args.bundle, missing) and _same(args.lock, lock)
        assert args.variant == variant
        # A checkout without the variant's DVC pointer has nothing to fetch.
        pointer.unlink()
        hint = spec.bundle_hint(architecture, missing, variant=variant)
        assert "no published" in hint and _mentions(hint, pointer)
        assert _commands(hint) == []


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("index", [0, -1])
def test_reference_fixture_explains_an_archive_passed_as_the_bundle(
    operation, index, tmp_path, monkeypatch
):
    from contextlib import closing
    from importlib import import_module
    from types import SimpleNamespace

    from rocke.runtime import hip_module

    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    registry = import_module(f"{operation}_reference.architectures")
    architecture = registry.ARCHITECTURES[index]
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
    variant = registry.resolve_variant(architecture)
    with closing(fixture):
        outcome, message = _outcome(lambda: next(fixture)(architecture, variant))
    assert outcome == "fail"
    assert "reference bundle is missing" in message.splitlines()[0]
    (extract,) = _commands(message)
    args = _unpacked(extract)
    assert (args.command, args.operation) == ("unpack", operation)
    assert args.variant == variant and _same(args.archive, archive)
    assert _same(args.lock, registry.baseline_lock(architecture, variant))
    # The pytest option, not the CLI's --bundle, then selects the directory
    # that the command extracts beside the archive.
    passed = _passed(message, "--rocke-reference-bundle")
    assert _same(passed, tmp_path / architecture) and _same(args.bundle, passed)


def test_displayed_commands_are_plain_text_for_every_shell():
    """Hints show arguments as typed and double-quote only whitespace.

    POSIX quoting would wrap a Windows path in single quotes, which cmd.exe
    keeps as part of the argument.
    """
    plain = ["dvc", "pull", "--operation", "conv", "gfx942.tar.gz.dvc"]
    windows = [
        str(PureWindowsPath("tests/library/tests/reference_common/artifact.py")),
        str(PureWindowsPath("D:/build/rocke$1/gfx942.tar.gz")),
    ]
    spaced = ["--bundle", str(PureWindowsPath("D:/build root/rocke $1/gfx942"))]
    assert display_command(plain) == "dvc pull --operation conv gfx942.tar.gz.dvc"
    assert display_command(windows) == (
        r"tests\library\tests\reference_common\artifact.py "
        r"D:\build\rocke$1\gfx942.tar.gz"
    )
    assert display_command(spaced) == r'--bundle "D:\build root\rocke $1\gfx942"'
    for arguments in (plain, windows, spaced):
        assert "'" not in display_command(arguments)
        assert _arguments(display_command(arguments)) == arguments


@pytest.mark.parametrize("operation", OPERATIONS)
def test_recovery_hint_shows_windows_paths_as_typed(tmp_path, monkeypatch, operation):
    """A recovery hint shows Windows paths as typed, without POSIX quoting.

    This host's paths have no backslashes, so the hint shows each path as the
    Windows text a Windows host would: the script and committed lock relative
    to the working directory above an installed test tree, and the temporary
    directory as an absolute path whose names contain a space and a $.
    """
    from importlib import import_module

    from reference_common import cli as common_cli

    spec = import_module(f"{operation}_reference.cli").OPERATION
    tests = _relocate(monkeypatch, tmp_path, operation, "source")
    variant = spec.architectures.resolve_variant("gfx942")
    _track(tests, operation, "gfx942", variant)
    here = Path(__file__).parent.resolve()
    root = PureWindowsPath("D:/build root/rocke $1")
    installed = PureWindowsPath("tests/library/tests")

    def shown(path):
        path = path.resolve()
        if path.is_relative_to(tmp_path.resolve()):
            return str(root.joinpath(*path.relative_to(tmp_path.resolve()).parts))
        return str(installed.joinpath(*path.relative_to(here).parts))

    monkeypatch.setattr(common_cli, "_shown", shown)
    hint = spec.bundle_hint("gfx942", tmp_path / "missing")
    archive = root / "rocke/library/tests/reference_bundles" / operation
    archive /= f"gfx942/{variant}.tar.gz"
    lock = installed / f"{operation}_reference/architectures/gfx942"
    lock /= f"{variant}/baseline_lock.json"
    assert hint == (
        "fetch and extract it with dvc pull and artifact.py unpack:\n"
        f'    dvc pull "{archive}.dvc"\n'
        r"    python tests\library\tests\reference_common\artifact.py unpack "
        f'--operation {operation} --variant {variant} --archive "{archive}" '
        rf'--bundle "D:\build root\rocke $1\missing" --lock {lock}'
    )
    assert "'" not in hint


@pytest.mark.parametrize("operation", OPERATIONS)
def test_reference_fixture_keeps_one_slot_open_at_a_time(
    second_variant, tmp_path, monkeypatch
):
    """Another processor's slot skips untouched; another variant replaces the open one.

    So the tests, which run slot by slot, never keep two slots' workers alive.
    """
    from contextlib import closing, contextmanager
    from types import SimpleNamespace

    from reference_common import pinned_tests
    from rocke.runtime import hip_module

    registry = second_variant.architectures
    architecture, other = registry.ARCHITECTURES
    for variant in registry.variants(architecture):
        registry.baseline_lock(architecture, variant).write_text("{}")
    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: architecture
    )
    loads, workers = [], []

    def load(bundle, lock, *, architecture, variant):
        loads.append(variant)
        return {"variant": variant}

    @contextmanager
    def reuse_workers(module):
        workers.append("open")
        try:
            yield
        finally:
            workers[workers.index("open")] = "closed"

    monkeypatch.setattr(pinned_tests, "reuse_workers", reuse_workers)
    config = SimpleNamespace(getoption=lambda option: None)
    fixture = pinned_tests.select_bundle(
        config, second_variant, lambda *args, **kwargs: tmp_path, load
    )
    with closing(fixture):
        open_slot = next(fixture)
        assert open_slot(architecture, "narrow")[2] == {"variant": "narrow"}
        with pytest.raises(pytest.skip.Exception) as skip:
            open_slot(other, "wide")
        assert str(skip.value) == (
            f"not selected: this {other} wide reference runs only on a {other} "
            f"GPU, and this GPU is {architecture}"
        )
        narrow = registry.get_adapter(architecture, "narrow")
        assert open_slot(architecture, "narrow")[0] is narrow
        assert (loads, workers) == (["narrow"], ["open"])
        wide = registry.get_adapter(architecture, "wide")
        assert open_slot(architecture, "wide")[0] is wide
        assert (loads, workers) == (["narrow", "wide"], ["closed", "open"])
    assert workers == ["closed", "closed"]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_reference_worker_requires_explicit_architecture(operation, tmp_path):
    from importlib import import_module

    worker = import_module(f"{operation}_reference.worker")
    # Reject the malformed request before runtime imports, GPU probing, or files.
    with pytest.raises(KeyError, match="architecture"):
        worker.run({}, tmp_path)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("mode", ["source", "replay"])
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
    operation, mode, architecture, device_target, accepted, tmp_path, monkeypatch
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
        worker, "get_adapter", lambda name, variant: SimpleNamespace(NAME=name)
    )
    monkeypatch.setattr(
        hip_module, "get_device_target_id", lambda device=0: device_target
    )
    request = {
        "architecture": architecture,
        "platform_root": str(Path(rocke.__file__).resolve().parent),
        "mode": mode,
        "repetitions": 1,
    }
    # An accepted device proceeds to decode the case this request omits.
    with pytest.raises(
        KeyError if accepted else RuntimeError,
        match="case" if accepted else f"HIP-visible {architecture} device",
    ):
        worker.run(request, tmp_path)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_reference_worker_imports_an_adapter_only_in_source_mode(
    operation, tmp_path, monkeypatch
):
    """Replay reads only the processor's name, so it never imports an adapter.

    Source mode runs the requested variant's adapter, and rejects a variant
    that is not enrolled before running anything.
    """
    from importlib import import_module

    worker = import_module(f"{operation}_reference.worker")
    registry = import_module(f"{operation}_reference.architectures")
    architecture = registry.ARCHITECTURES[0]
    variant = registry.resolve_variant(architecture)
    targets = []
    monkeypatch.setattr(
        worker, "run_case", lambda operation, target, *args: targets.append(target)
    )
    request = {"architecture": architecture, "variant": "unenrolled"}
    with monkeypatch.context() as patch:
        # A None entry makes any import of the adapter fail.
        adapter = f"{registry.__name__}.{architecture}.{variant}"
        patch.setitem(sys.modules, adapter, None)
        worker.run(dict(request, mode="replay"), tmp_path)
    assert targets == [registry.get_architecture(architecture)]
    with pytest.raises(ValueError, match="reference variant is not enrolled"):
        worker.run(dict(request, mode="source"), tmp_path)
    worker.run(dict(request, mode="source", variant=variant), tmp_path)
    assert targets[1:] == [registry.get_adapter(architecture, variant)]


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


@pytest.mark.parametrize("operation", OPERATIONS)
def test_enrollment_rejects_unlisted_names_before_importing(operation, monkeypatch):
    """Each package enrolls exactly its registry.json through the shared registry.

    Each enrolled processor has a variant adapter, and a variant's lock
    resolves beside it. Any other name is rejected with the operation's label
    before an adapter import is attempted.
    """
    import re
    from importlib import import_module

    spec = import_module(f"{operation}_reference.cli").OPERATION
    registry = spec.architectures
    root = Path(registry.__file__).resolve().parent
    listed = json.loads((root / "registry.json").read_text())
    assert registry.ARCHITECTURES == tuple(listed)
    for architecture in registry.ARCHITECTURES:
        assert registry.get_architecture(architecture).NAME == architecture
        assert registry.variants(architecture)
        for variant in registry.variants(architecture):
            assert registry.get_adapter(architecture, variant).NAME == architecture
            assert registry.baseline_lock(architecture, variant) == (
                root / architecture / variant / "baseline_lock.json"
            )
        # A None entry makes any import of a name fail with ImportError instead.
        for name in ("missing", "../" + registry.variants(architecture)[0]):
            monkeypatch.setitem(
                sys.modules, f"{registry.__name__}.{architecture}.{name}", None
            )
            for lookup in (registry.get_adapter, registry.baseline_lock):
                with pytest.raises(
                    ValueError,
                    match=f"^{spec.label} {architecture} reference variant is not "
                    f"enrolled: {re.escape(name)}; enrolled: ",
                ):
                    lookup(architecture, name)
    # common is the adapters' shared helper module, not an adapter.
    for name in ("gfx1100", "../gfx942", "common"):
        monkeypatch.setitem(sys.modules, f"{registry.__name__}.{name}", None)
        for lookup in (
            registry.get_architecture,
            registry.variants,
            registry.resolve_variant,
            registry.get_adapter,
            registry.baseline_lock,
        ):
            with pytest.raises(
                ValueError,
                match=f"^{spec.label} reference architecture is not enrolled: ",
            ):
                lookup(name)


def test_registry_finds_variants_and_defaults_only_a_sole_one(tmp_path, monkeypatch):
    """A variant is a processor subpackage; only a sole variant is a default."""
    from reference_common.registry import AdapterRegistry

    root = tmp_path / "fictional_reference/architectures"
    root.mkdir(parents=True)
    for package in (root.parent, root):
        (package / "__init__.py").touch()
    (root / "registry.json").write_text('["gfx942", "gfx950"]\n')
    for architecture in ("gfx942", "gfx950"):
        (root / architecture).mkdir()
        (root / architecture / "__init__.py").write_text(f"NAME = {architecture!r}\n")
    for package in ("gfx942/wide", "gfx942/narrow", "gfx950/wide"):
        (root / package).mkdir()
        (root / package / "__init__.py").write_text("from .. import NAME\n")
    # Neither a cache nor a directory without a package marker is a variant.
    for directory in ("gfx942/__pycache__", "gfx942/notes"):
        (root / directory).mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = AdapterRegistry(
        "fictional_reference.architectures", root, label="Fictional"
    )
    try:
        assert registry.variants("gfx942") == ("narrow", "wide")
        assert registry.resolve_variant("gfx950") == "wide"
        assert (
            registry.baseline_lock("gfx950") == root / "gfx950/wide/baseline_lock.json"
        )
        assert registry.baseline_lock("gfx942", "narrow") == (
            root / "gfx942/narrow/baseline_lock.json"
        )
        assert registry.get_architecture("gfx942").NAME == "gfx942"
        # Neither lookups nor the processor's package import its variants.
        assert {name for name in sys.modules if name.startswith("fictional_")} == {
            "fictional_reference",
            "fictional_reference.architectures",
            "fictional_reference.architectures.gfx942",
        }
        adapter = registry.get_adapter("gfx942", "wide")
        assert (adapter.__name__, adapter.NAME) == (
            "fictional_reference.architectures.gfx942.wide",
            "gfx942",
        )
        for lookup in (
            registry.resolve_variant,
            registry.get_adapter,
            registry.baseline_lock,
        ):
            with pytest.raises(
                ValueError,
                match="^Fictional gfx942 needs a selected reference variant; "
                "enrolled: narrow, wide$",
            ):
                lookup("gfx942")
    finally:
        for name in [name for name in sys.modules if name.startswith("fictional_")]:
            del sys.modules[name]


# A variant is named for the module of the kernel family that it adapts.
# TheRock splits installed bundles by their first gfx-named directory, which
# must stay the processor's, so a variant name cannot start with gfx.
_VARIANT_NAME = re.compile(r"(?!gfx)[a-z][a-z0-9_]*")
# Replay imports only a processor's package. This child imports each one in a
# fresh interpreter and reports its NAME and the modules that it loaded.
_IMPORT_PROCESSORS = """
import json
import sys
from importlib import import_module

sys.path[:0] = sys.argv[2:]
registry = import_module(sys.argv[1])
loaded = {}
for architecture in registry.ARCHITECTURES:
    before = set(sys.modules)
    package = registry.get_architecture(architecture)
    loaded[architecture] = [package.NAME, sorted(set(sys.modules) - before)]
print(json.dumps(loaded))
"""


def _check_variant_layout(registry):
    """Fail unless the enrolled variants of ``registry`` follow the layout rules.

    ``registry`` is an operation's architectures package. Each variant names a
    module of the operation's ``implementations`` package, and its adapter's
    ``NAME`` is its processor directory's. Each processor package names its
    own directory and imports no variant, so replay never imports an adapter.
    """
    root = Path(registry.__file__).resolve().parent
    implementations = root.parent / "implementations"
    # The child imports the operation package and reference_common from these.
    paths = (root.parents[1], Path(__file__).resolve().parent)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _IMPORT_PROCESSORS, registry.__name__, *paths],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode:
        pytest.fail(
            f"importing the processor packages of {registry.__name__} failed:\n"
            f"{completed.stderr[-12000:]}",
            pytrace=False,
        )
    processors = json.loads(completed.stdout)
    adapters = {
        f"{registry.__name__}.{architecture}.{variant}"
        for architecture in registry.ARCHITECTURES
        for variant in registry.variants(architecture)
    }
    problems = []
    for architecture in registry.ARCHITECTURES:
        package = root / architecture / "__init__.py"
        name, loaded = processors[architecture]
        if name != architecture:
            problems.append(f"{package} sets NAME to {name!r}, not {architecture!r}")
        imported = sorted(adapters.intersection(loaded))
        if imported:
            problems.append(
                f"{package} imports {', '.join(imported)}, but replay imports "
                "processor packages and must not import a variant adapter"
            )
        for variant in registry.variants(architecture):
            adapter = root / architecture / variant / "__init__.py"
            module = implementations / f"{variant}.py"
            if not _VARIANT_NAME.fullmatch(variant):
                problems.append(
                    f"{adapter.parent} has an invalid variant name: a variant "
                    "name matches ^[a-z][a-z0-9_]*$ and does not start with gfx"
                )
            if not module.is_file():
                problems.append(
                    f"{adapter.parent} names no kernel family module: {module} "
                    "does not exist"
                )
            target = registry.get_adapter(architecture, variant).NAME
            if target != architecture:
                problems.append(
                    f"{adapter} sets NAME to {target!r}, not {architecture!r}; "
                    "import it from the processor package"
                )
    if problems:
        pytest.fail(
            f"{registry.__name__} breaks the reference variant layout:\n"
            + "".join(f"  {problem}\n" for problem in problems),
            pytrace=False,
        )


@pytest.mark.parametrize("operation", OPERATIONS)
def test_enrolled_variants_follow_the_layout_rules(operation):
    """Tools address a variant by its name, and replay by its processor alone."""
    from importlib import import_module

    _check_variant_layout(import_module(f"{operation}_reference.architectures"))


@pytest.mark.parametrize(
    "problem",
    [None, "gfx-name", "pattern-name", "module", "processor", "adapter", "import"],
)
def test_variant_layout_check_names_each_violation(tmp_path, monkeypatch, problem):
    """Each broken rule names the file to fix; a conforming tree passes."""
    from importlib import import_module

    package = tmp_path / "fictional_reference"
    root = package / "architectures"
    root.mkdir(parents=True)
    (package / "implementations").mkdir()
    for directory in (package, package / "implementations"):
        (directory / "__init__.py").touch()
    (root / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from reference_common.registry import AdapterRegistry\n"
        "_REGISTRY = AdapterRegistry(\n"
        "    __name__, Path(__file__).resolve().parent, label='Fictional'\n"
        ")\n"
        "ARCHITECTURES = _REGISTRY.architectures\n"
        "get_architecture = _REGISTRY.get_architecture\n"
        "variants = _REGISTRY.variants\n"
        "get_adapter = _REGISTRY.get_adapter\n"
    )
    (root / "registry.json").write_text('["gfx942"]\n')
    variant, processor, adapter = "wide", 'NAME = "gfx942"\n', "from .. import NAME\n"
    if problem == "gfx-name":
        variant = "gfx942_wide"
    elif problem == "pattern-name":
        variant = "Wide"
    elif problem == "processor":
        processor = 'NAME = "gfx950"\n'
    elif problem == "adapter":
        adapter = 'NAME = "gfx950"\n'
    elif problem == "import":
        processor += "from . import wide\n"
    (root / "gfx942" / variant).mkdir(parents=True)
    (root / "gfx942/__init__.py").write_text(processor)
    (root / "gfx942" / variant / "__init__.py").write_text(adapter)
    if problem != "module":
        (package / f"implementations/{variant}.py").touch()
    expected = {
        "gfx-name": f"{root / 'gfx942/gfx942_wide'} has an invalid variant name",
        "pattern-name": f"{root / 'gfx942/Wide'} has an invalid variant name",
        "module": f"{root / 'gfx942/wide'} names no kernel family module: "
        f"{package / 'implementations/wide.py'} does not exist",
        "processor": f"{root / 'gfx942/__init__.py'} sets NAME to 'gfx950', not "
        "'gfx942'",
        "adapter": f"{root / 'gfx942/wide/__init__.py'} sets NAME to 'gfx950', not "
        "'gfx942'",
        "import": f"{root / 'gfx942/__init__.py'} imports "
        "fictional_reference.architectures.gfx942.wide, but replay",
    }
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        registry = import_module("fictional_reference.architectures")
        if problem is None:
            _check_variant_layout(registry)
            return
        with pytest.raises(pytest.fail.Exception) as failure:
            _check_variant_layout(registry)
    finally:
        for name in [name for name in sys.modules if name.startswith("fictional_")]:
            del sys.modules[name]
    assert expected[problem] in str(failure.value)


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


def _qualified_cohort(tmp_path, operation, index=0):
    """A locked synthetic bundle recording one of an operation's enrolled cohorts."""
    from dataclasses import asdict
    from importlib import import_module

    from reference_common.numeric import payload_digests

    spec = import_module(f"{operation}_reference.cli").OPERATION
    architecture = spec.architectures.ARCHITECTURES[index]
    target = spec.architectures.get_adapter(architecture)
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
    spec = import_module(f"{operation}_reference.cli").OPERATION
    registry = spec.architectures
    for architecture in registry.ARCHITECTURES:
        variant = registry.resolve_variant(architecture)
        cases = registry.get_adapter(architecture, variant).CASES
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
        assert (
            listing["operation"],
            listing["architecture"],
            listing["variant"],
        ) == (operation, architecture, variant)
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
    # The count names the other slots whose locks pin the same cohort digest.
    cohort = spec.enrolled_cohort_digest(architecture, variant)
    shared = [
        f"{name} {other}"
        for name in registry.ARCHITECTURES
        for other in registry.variants(name)
        if (name, other) != (architecture, variant)
        and spec.enrolled_cohort_digest(name, other) == cohort
    ]
    count = f"{spec.label} {architecture} {variant}: {len(cases)} enrolled cases"
    assert lines[-1] == (
        f"{count}; cohort shared with {', '.join(shared)}" if shared else count
    )


@pytest.mark.parametrize("operation", OPERATIONS)
def test_cases_command_groups_variants_and_names_shared_cohorts(second_variant, capsys):
    """Text lists every variant of a processor; JSON describes one variant."""
    spec = second_variant
    registry = spec.architectures
    first, second = registry.ARCHITECTURES
    cases = registry.get_adapter(first, "wide").CASES
    spec.main(["cases", "--arch", first])
    narrow, wide = (
        group.splitlines() for group in capsys.readouterr().out.split("\n\n")
    )
    assert [line.split()[0] for line in narrow[:-1]] == [c.id for c in cases[1:]]
    assert narrow[-1] == f"{spec.label} {first} narrow: {len(cases) - 1} enrolled cases"
    assert [line.split()[0] for line in wide[:-1]] == [c.id for c in cases]
    assert wide[-1] == (
        f"{spec.label} {first} wide: {len(cases)} enrolled cases; cohort shared "
        f"with {second} wide"
    )
    spec.main(["cases", "--arch", first, "--variant", "narrow"])
    assert capsys.readouterr().out.splitlines() == narrow
    spec.main(["cases", "--arch", first, "--variant", "narrow", "--json"])
    listing = json.loads(capsys.readouterr().out)
    assert (listing["architecture"], listing["variant"]) == (first, "narrow")
    assert [entry["id"] for entry in listing["cases"]] == [c.id for c in cases[1:]]
    # A processor's only variant needs no selection.
    spec.main(["cases", "--arch", second, "--json"])
    assert json.loads(capsys.readouterr().out)["variant"] == "wide"
    for selection, problem in (
        ([], "needs a selected reference variant"),
        (["--variant", "missing"], "reference variant is not enrolled: missing"),
    ):
        with pytest.raises(SystemExit) as exit:
            spec.main(["cases", "--arch", first, "--json", *selection])
        assert exit.value.code == 2
        assert capsys.readouterr().err.endswith(
            f"error: argument --variant: {spec.label} {first} {problem}; "
            "enrolled: narrow, wide\n"
        )


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("command", ["qualify", "verify", "diff-bundle", "diff-lock"])
def test_commands_need_a_variant_when_a_processor_has_several(
    second_variant, tmp_path, capsys, command
):
    """Qualification, verification, and a selected bundle's diff use one variant.

    Each refuses an ambiguous or unenrolled variant before reading any path,
    and the error lists the processor's variants.
    """
    spec = second_variant
    architecture = spec.architectures.ARCHITECTURES[0]
    bundle = tmp_path / "bundle"
    arguments = {
        "qualify": ["qualify", "--baseline", tmp_path / "baseline", "--output", bundle],
        "verify": ["verify", "--bundle", bundle],
        "diff-bundle": ["diff", "--bundle", bundle],
        "diff-lock": ["diff", "--lock", tmp_path / "lock.json"],
    }[command]
    for selection, problem in (
        ([], "needs a selected reference variant"),
        (["--variant", "missing"], "reference variant is not enrolled: missing"),
    ):
        with pytest.raises(SystemExit) as exit:
            spec.main([*map(str, arguments), "--arch", architecture, *selection])
        assert exit.value.code == 2
        assert capsys.readouterr().err.endswith(
            f"error: argument --variant: {spec.label} {architecture} {problem}; "
            "enrolled: narrow, wide\n"
        )
    assert not bundle.exists()


@pytest.mark.parametrize("operation", OPERATIONS)
def test_qualification_records_the_selected_variant(
    second_variant, tmp_path, monkeypatch, capsys, operation
):
    """A new manifest declares the variant that it was qualified for.

    Its lock then authenticates the bundle only as that variant, so
    verification and staging refuse it as another variant of the processor,
    even one that enrolls the same cases.
    """
    from reference_common import cli as common_cli
    from reference_common.numeric import file_digest, payload_digests, write_json

    spec = second_variant
    architecture = spec.architectures.ARCHITECTURES[0]
    baseline = tmp_path / "baseline"
    runtime = baseline / "source/platform/python/rocke"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text('"""Host qualification fixture."""\n')
    write_json(
        baseline / "snapshot.json",
        {"revision": "a" * 40, "files": payload_digests(baseline / "source")},
    )
    narrow, wide = (
        spec.architectures.get_adapter(architecture, variant)
        for variant in ("narrow", "wide")
    )
    case = min(narrow.CASES, key=lambda c: math.prod(c.output_shape))
    for target in (narrow, wide):
        monkeypatch.setattr(target, "CASES", (case,))

    def worker(request, **kwargs):
        assert request["variant"] == "narrow"
        Path(request["export"]).write_bytes(b"synthetic-code-object")
        with np.load(request["inputs"], allow_pickle=False) as arrays:
            reference = spec.contract.independent_reference(case, dict(arrays))
        return [encode(reference, case.dtype)] * request["repetitions"], {
            "kernel": {"sha256": file_digest(Path(request["export"]))},
            "device_target": f"{architecture}:fixture",
            "compiler": {"fixture": True},
        }

    monkeypatch.setattr(common_cli, "run_worker", worker)
    bundle = tmp_path / "bundle"
    spec.qualify(baseline, bundle, 2, architecture, variant="narrow")
    assert capsys.readouterr().out.startswith(
        f"{spec.label} {architecture} narrow: qualifying 1 enrolled cases\n"
    )
    lock = bundle / "qualification-lock.json"
    manifest = spec.load_bundle(
        bundle, lock, architecture=architecture, variant="narrow"
    )
    assert manifest["variant"] == "narrow"
    # SDPA manifests still leave their operation to the lock.
    assert ("operation" in manifest) == spec.declares_operation
    validate_bundle(bundle, lock, operation=operation, variant="narrow")
    with pytest.raises(ValueError) as error:
        spec.load_bundle(bundle, lock, architecture=architecture, variant="wide")
    assert str(error.value) == (
        "wrong reference variant: the manifest declares 'narrow', but the "
        f"selected {spec.label} {architecture} variant is 'wide'"
    )
    with pytest.raises(InconsistentLockError, match="'narrow' does not match 'wide'"):
        validate_bundle(bundle, lock, operation=operation, variant="wide")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("form", ["directory", "archive"])
def test_diff_reports_every_difference_that_verification_rejects(
    tmp_path, capsys, monkeypatch, operation, form
):
    from dataclasses import asdict

    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    target = spec.architectures.get_adapter(architecture)
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
    monkeypatch.setattr(
        spec.architectures, "baseline_lock", lambda name, variant=None: lock
    )
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
    variant = spec.architectures.resolve_variant(architecture)
    assert f"{architecture} {variant} has no published" in capsys.readouterr().err


@pytest.mark.parametrize("operation", OPERATIONS)
def test_diff_compares_every_published_variant(
    second_variant, tmp_path, capsys, monkeypatch, operation
):
    """Without a selection, diff reports each published variant of a processor.

    It exits with status 1 when any variant differs and 2 when any variant's
    data is unusable, after reporting the variants that it could compare.
    """
    import shutil

    from reference_common import cli as common_cli

    spec = second_variant
    registry = spec.architectures
    architecture = registry.ARCHITECTURES[0]
    cases = registry.get_adapter(architecture, "wide").CASES
    # The bundle records the wide cohort; narrow enrolls all but its first case.
    _, _, bundle, lock = _qualified_cohort(tmp_path, operation)
    published = tmp_path / "published" / architecture
    monkeypatch.setattr(
        common_cli,
        "default_bundle_path",
        lambda tests, name, *, operation, variant: published / variant,
    )
    for variant in ("narrow", "wide"):
        shutil.copytree(bundle, published / variant)
    matched = (
        f"{spec.label} {architecture} wide: all {len(cases)} enrolled cases match "
        f"{published / 'wide'}\n"
    )
    # An unpublished variant, without a lock, is not compared.
    shutil.copy(lock, registry.baseline_lock(architecture, "wide"))
    spec.main(["diff", "--arch", architecture])
    assert capsys.readouterr().out == matched

    shutil.copy(lock, registry.baseline_lock(architecture, "narrow"))
    with pytest.raises(SystemExit) as exit:
        spec.main(["diff", "--arch", architecture])
    assert exit.value.code == 1
    output = capsys.readouterr().out
    assert output.startswith(
        f"{spec.label} {architecture} narrow: the enrolled cases differ from "
        f"{published / 'narrow'}\n  in the bundle but not enrolled: {cases[0].id} ("
    )
    assert matched in output
    assert output.endswith(f"requalification; see {_WORKFLOW_DOC}.\n")
    # --variant compares only that variant.
    spec.main(["diff", "--arch", architecture, "--variant", "wide"])
    assert capsys.readouterr().out == matched

    shutil.rmtree(published / "narrow")
    with pytest.raises(SystemExit) as exit:
        spec.main(["diff", "--arch", architecture])
    assert exit.value.code == 2
    output = capsys.readouterr()
    assert output.out == matched
    assert (
        f"error: no {architecture} narrow {spec.label} bundle at "
        f"{published / 'narrow.tar.gz'} or {published / 'narrow'}; fetch the "
        "archive with DVC or pass --bundle\n"
    ) in output.err


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "problem", ["another-lock", "another-operation", "pointer", "truncated", "stale"]
)
def test_diff_reports_an_unusable_bundle_once_with_status_2(
    tmp_path, capsys, operation, problem
):
    """A bundle that diff cannot read or authenticate is an error, not a difference."""
    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    archive = tmp_path / f"{architecture}.tar.gz"
    pack(bundle, archive, lock, operation=operation)
    source, hint = archive, None
    if problem == "another-lock":
        # Such as another architecture's archive checked against this lock.
        lock.write_text(
            json.dumps(dict(json.loads(lock.read_text()), manifest_sha256="0" * 64))
        )
        expected = "GPU reference manifest does not match the pinned lock"
        hint = f"{archive} is not the archive that {lock} pins"
    elif problem == "another-operation":
        other = _another_operation(operation)
        other_bundle, other_lock = _fixture(tmp_path / other, other)
        source = tmp_path / f"{other}.tar.gz"
        pack(other_bundle, source, other_lock, operation=other)
        expected = (
            f"invalid {operation.upper()} archive member: {other}_reference_bundle/"
        )
    elif problem == "pointer":
        source = archive.with_name(f"{archive.name}.dvc")
        source.write_text("outs:\n- md5: 0\n")
        expected = f"archive {source.resolve()} cannot be read: not a gzip file"
        hint = f"pass the archive that this DVC pointer tracks, {archive.resolve()}"
    elif problem == "truncated":
        source = tmp_path / "truncated.tar.gz"
        source.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
        expected = f"archive {source.resolve()} cannot be read: "
        hint = "pass an extracted bundle directory or a .tar.gz archive"
    else:
        (bundle / "payload/kernel.hsaco").write_bytes(b"modified")
        source = bundle
        expected = "payload has missing, modified, or extra files"
    command = ["diff", "--arch", architecture, "--bundle", str(source)]
    with pytest.raises(SystemExit) as exit:
        spec.main([*command, "--lock", str(lock)])
    assert exit.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "Traceback" not in output.err
    assert output.err.count(expected) == 1
    assert hint is None or output.err.count(hint) == 1


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("declared", [None, "selected", "other"])
def test_bundle_commands_accept_only_the_selected_declared_variant(
    tmp_path, capsys, operation, declared
):
    """A manifest from before variants declares none; its lock identifies it.

    Verification and diff, of a directory or an archive, report a manifest
    that declares another variant as unusable, before any case runs.
    """
    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    variant = spec.architectures.resolve_variant(architecture)
    if declared is not None:
        recorded = variant if declared == "selected" else declared
        manifest = json.loads((bundle / "manifest.json").read_text())
        _relock(bundle, dict(manifest, variant=recorded), lock)
    archive = tmp_path / "bundle.tar.gz"
    pack(bundle, archive, lock, operation=operation)
    selected = {"architecture": architecture, "variant": variant}
    if declared != "other":
        manifest = spec.load_bundle(bundle, lock, **selected)
        assert manifest.get("variant") == (variant if declared else None)
        for source in (bundle, archive):
            assert spec.diff_bundle(source, lock, **selected) == []
        return
    expected = (
        "wrong reference variant: the manifest declares 'other', but the "
        f"selected {spec.label} {architecture} variant is {variant!r}"
    )
    for command, source in (("verify", bundle), ("diff", bundle), ("diff", archive)):
        with pytest.raises(SystemExit) as exit:
            spec.main(
                [command, "--arch", architecture, "--bundle", str(source)]
                + ["--lock", str(lock)]
            )
        assert exit.value.code == 2
        output = capsys.readouterr()
        assert output.out == ""
        assert output.err.count(expected) == 1


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("command", ["diff", "verify"])
def test_bundle_commands_say_how_to_provide_the_rocke_platform(
    tmp_path, operation, command
):
    """Matching recorded device targets needs rocKE; its absence is not a crash."""
    import os

    _, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    stub = tmp_path / "without-rocke"
    stub.mkdir()
    (stub / "rocke.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'rocke'\", name='rocke')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(stub), PYTHONNOUSERSITE="1")
    script = Path(__file__).with_name(f"run_{operation}_reference.py")
    result = subprocess.run(
        [sys.executable, str(script), command, "--arch", architecture]
        + ["--bundle", str(bundle), "--lock", str(lock)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert result.stderr.count(f"{command} reads recorded device target ids") == 1
    assert "install rocKE or set PYTHONPATH=platform/python" in result.stderr


@pytest.mark.parametrize("operation", OPERATIONS)
def test_status_reports_every_enrolled_variant_and_exits_on_a_difference(
    second_variant, tmp_path, capsys, monkeypatch, operation
):
    """One line per enrolled variant: publication, shared cohort, and bundle state.

    status exits with status 1 when a published bundle differs from its
    enrolled cases, and otherwise with status 2 when one is unusable.
    """
    import shutil

    from reference_common import cli as common_cli

    spec = second_variant
    registry = spec.architectures
    first, second = registry.ARCHITECTURES
    # The bundle records the wide cohort; narrow enrolls all but its first case.
    _, _, bundle, lock = _qualified_cohort(tmp_path, operation)
    published = tmp_path / "published"
    monkeypatch.setattr(
        common_cli,
        "default_bundle_path",
        lambda tests, name, *, operation, variant: published / name / variant,
    )
    for variant in ("narrow", "wide"):
        shutil.copytree(bundle, published / first / variant)
        shutil.copy(lock, registry.baseline_lock(first, variant))
    # Paths below the working directory are shown relative to it.
    monkeypatch.chdir(tmp_path)
    script = Path(__file__).resolve().with_name(f"run_{operation}_reference.py")

    def diff(variant):
        command = ["python", str(script), "diff", "--arch", first]
        return display_command([*command, "--variant", variant])

    head = f"published at baseline {'a' * 12}"
    narrow = (
        f"{spec.label} {first} narrow: {head}; bundle "
        f"{Path('published', first, 'narrow')} differs from the enrolled cases; "
        f"list the differences with: {diff('narrow')}"
    )
    wide = f"{spec.label} {first} wide: {head}; cohort shared with {second} wide"
    matched = (
        f"{wide}; bundle {Path('published', first, 'wide')} matches the enrolled cases"
    )
    unpublished = (
        f"{spec.label} {second} wide: unpublished; cohort shared with {first} wide"
    )
    with pytest.raises(SystemExit) as exit:
        spec.main(["status"])
    assert exit.value.code == 1
    assert capsys.readouterr().out.splitlines() == [
        narrow,
        matched,
        unpublished,
        (
            "A cohort or budget change needs a complete requalification; see "
            f"{_WORKFLOW_DOC}."
        ),
    ]

    registry.baseline_lock(first, "narrow").unlink()
    spec.main(["status"])
    assert capsys.readouterr().out.splitlines() == [
        f"{spec.label} {first} narrow: unpublished",
        matched,
        unpublished,
    ]

    payload = published / first / "wide/payload/kernel.hsaco"
    payload.write_bytes(b"modified")
    with pytest.raises(SystemExit) as exit:
        spec.main(["status"])
    assert exit.value.code == 2
    assert capsys.readouterr().out.splitlines()[1] == (
        f"{wide}; bundle {Path('published', first, 'wide')} is unusable: "
        f"{spec.label} bundle payload has missing, modified, or extra files: 1 "
        f"differ, including kernel.hsaco; see the full error with: {diff('wide')}"
    )
    # A difference decides the status even beside an unusable bundle.
    shutil.copy(lock, registry.baseline_lock(first, "narrow"))
    with pytest.raises(SystemExit) as exit:
        spec.main(["status"])
    assert exit.value.code == 1
    assert capsys.readouterr().out.splitlines()[0] == narrow

    # Without the rocKE platform package, no published bundle can be compared.
    payload.write_bytes((bundle / "payload/kernel.hsaco").read_bytes())

    def unavailable(target_id):
        raise ModuleNotFoundError("No module named 'rocke'", name="rocke")

    monkeypatch.setattr(common_cli, "processor_from_target_id", unavailable)
    with pytest.raises(SystemExit) as exit:
        spec.main(["status"])
    assert exit.value.code == 2
    lines = capsys.readouterr().out.splitlines()
    for line in lines[:2]:
        assert "is unusable: status reads recorded device target ids" in line
    assert lines[2] == unpublished


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("layout", ["source", "installed"])
def test_status_compares_locked_cohorts_without_bundle_data(
    second_variant, tmp_path, capsys, monkeypatch, operation, layout
):
    """A cohort edit differs from the digest that each sharing lock pins.

    A tree without bundle data still learns which published variants need
    requalification, and how it gets their bundles: a source checkout fetches
    the tracked archive, and an installed tree needs a build that installs the
    references. A lock that cannot be read, such as one left in a merge
    conflict, is unusable and does not end the listing.
    """
    from reference_common.paths import default_bundle_path

    spec = second_variant
    registry = spec.architectures
    first, second = registry.ARCHITECTURES
    tests = _relocate(monkeypatch, tmp_path, operation, layout)
    narrow = spec.enrolled_cohort_digest(first, "narrow")
    wide = spec.enrolled_cohort_digest(first, "wide")
    pinned = {"schema": 2, "baseline_revision": "a" * 40, "manifest_sha256": "0" * 64}
    # Both wide locks pin the cohort from before wide enrolled its first case.
    for architecture, variant in ((first, "narrow"), (first, "wide"), (second, "wide")):
        lock = registry.baseline_lock(architecture, variant)
        lock.write_text(json.dumps(dict(pinned, cohort_sha256=narrow)))
    # Only a source checkout tracks archives, and only the wide ones here.
    pointers = (
        {
            architecture: _track(tests, operation, architecture, "wide")
            for architecture in (first, second)
        }
        if layout == "source"
        else {}
    )
    monkeypatch.chdir(tmp_path)
    script = tests / f"run_{operation}_reference.py"

    def shown(path):
        return path.resolve().relative_to(tmp_path.resolve())

    def bundle(architecture, variant):
        return default_bundle_path(
            tests, architecture, operation=operation, variant=variant
        )

    def diff(architecture):
        command = ["python", str(shown(script)), "diff", "--arch", architecture]
        return display_command([*command, "--variant", "wide"])

    def missing(architecture, variant, differs=False):
        """How this tree gets a variant's absent bundle, as status says it."""
        if layout == "installed":
            return (
                f"bundle missing at {shown(bundle(architecture, variant))}; "
                "installed trees receive it only when built with "
                "ROCKE_INSTALL_TEST_GPU_REFERENCES=ON and assembled with its "
                "architecture's test artifact"
            )
        if architecture not in pointers or variant != "wide":
            archive = bundle(architecture, variant).with_name(f"{variant}.tar.gz")
            return (
                "bundle missing; no DVC pointer tracks its archive at "
                f"{shown(archive)}.dvc"
            )
        fetch = display_command(["dvc", "pull", str(shown(pointers[architecture]))])
        then = f", then list the differences with: {diff(architecture)}"
        return f"bundle missing; fetch it with: {fetch}" + (then if differs else "")

    head = f"published at baseline {'a' * 12}"
    expected = [f"{spec.label} {first} narrow: {head}; {missing(first, 'narrow')}"]
    for architecture, other in ((first, second), (second, first)):
        lock = shown(registry.baseline_lock(architecture, "wide"))
        expected.append(
            f"{spec.label} {architecture} wide: {head}; cohort shared with "
            f"{other} wide; the enrolled cases differ from the cohort that {lock} "
            f"pins; {missing(architecture, 'wide', differs=True)}"
        )
    expected.append(
        f"A cohort or budget change needs a complete requalification; see "
        f"{_WORKFLOW_DOC}."
    )
    with pytest.raises(SystemExit) as exit:
        spec.main(["status"])
    assert exit.value.code == 1
    assert capsys.readouterr().out.splitlines() == expected

    # The differing lock decides before any bundle data present is read.
    extracted = bundle(second, "wide")
    extracted.mkdir(parents=True)
    with pytest.raises(SystemExit) as exit:
        spec.main(["status"])
    assert exit.value.code == 1
    line = capsys.readouterr().out.splitlines()[2]
    assert line.endswith(f" pins; list the differences with: {diff(second)}")
    extracted.rmdir()

    for architecture in (first, second):
        lock = registry.baseline_lock(architecture, "wide")
        lock.write_text(json.dumps(dict(pinned, cohort_sha256=wide)))
    conflicted = registry.baseline_lock(first, "narrow")
    conflicted.write_text("<<<<<<< ours\n")
    with pytest.raises(SystemExit) as exit:
        spec.main(["status"])
    assert exit.value.code == 2
    assert capsys.readouterr().out.splitlines() == [
        (
            f"{spec.label} {first} narrow: published; lock {shown(conflicted)} is "
            "unusable: JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
        ),
        *(
            f"{spec.label} {architecture} wide: {head}; cohort shared with "
            f"{other} wide; {missing(architecture, 'wide')}"
            for architecture, other in ((first, second), (second, first))
        ),
    ]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("layout", ["tracked", "untracked", "installed"])
def test_status_says_how_to_provide_missing_bundle_data(
    tmp_path, capsys, monkeypatch, operation, layout
):
    """A missing published bundle is reported with what provides it; it is no failure.

    A source checkout fetches the archive that the variant's DVC pointer
    tracks; an installed tree receives the bundle only from installation.
    """
    from importlib import import_module

    from reference_common.paths import default_bundle_path

    tests = _relocate(
        monkeypatch,
        tmp_path,
        operation,
        "installed" if layout == "installed" else "source",
    )
    spec = import_module(f"{operation}_reference.cli").OPERATION
    root = tmp_path.resolve()
    monkeypatch.chdir(root)
    expected = {}
    for name, architecture, variant in _published_variants():
        if name != operation:
            continue
        bundle = default_bundle_path(
            tests, architecture, operation=operation, variant=variant
        )
        pointer = bundle.with_name(f"{variant}.tar.gz.dvc")
        if layout == "tracked":
            assert _track(tests, operation, architecture, variant) == pointer
            fetch = display_command(["dvc", "pull", str(pointer.relative_to(root))])
            detail = f"bundle missing; fetch it with: {fetch}"
        elif layout == "untracked":
            detail = (
                "bundle missing; no DVC pointer tracks its archive at "
                f"{pointer.relative_to(root)}"
            )
        else:
            detail = (
                f"bundle missing at {bundle.relative_to(root)}; installed trees "
                "receive it only when built with ROCKE_INSTALL_TEST_GPU_REFERENCES=ON "
                "and assembled with its architecture's test artifact"
            )
        expected[f"{spec.label} {architecture} {variant}: "] = detail
    assert expected
    spec.main(["status"])
    lines = capsys.readouterr().out.splitlines()
    for subject, detail in expected.items():
        (line,) = (line for line in lines if line.startswith(subject))
        assert line.endswith(f"; {detail}")


@pytest.mark.parametrize("operation", OPERATIONS)
def test_status_reports_this_tree_without_a_gpu_or_torch(tmp_path, operation):
    """status reads this tree's locks and published bundle data on a host.

    Host tests require each bundle in the tree to record its enrolled cohort,
    so a published variant's bundle matches unless the tree lacks its data.
    """
    import importlib.util
    import os
    from importlib import import_module

    # Any import of Torch or production kernels fails.
    for name in ("torch", "kernels", "dispatch"):
        (tmp_path / f"{name}.py").write_text(
            f"raise AssertionError('imported {name}')\n"
        )
    # Reading recorded device targets needs the rocKE platform package.
    platform = Path(importlib.util.find_spec("rocke").origin).parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(tmp_path), str(platform)]),
        PYTHONNOUSERSITE="1",
    )
    script = Path(__file__).with_name(f"run_{operation}_reference.py")
    result = subprocess.run(
        [sys.executable, str(script), "status"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    spec = import_module(f"{operation}_reference.cli").OPERATION
    registry = spec.architectures
    slots = [
        (architecture, variant)
        for architecture in registry.ARCHITECTURES
        for variant in registry.variants(architecture)
    ]
    lines = result.stdout.splitlines()
    assert len(lines) == len(slots), result.stdout
    for line, (architecture, variant) in zip(lines, slots):
        subject = f"{spec.label} {architecture} {variant}: "
        lock = registry.baseline_lock(architecture, variant)
        if not lock.is_file():
            assert line.startswith(f"{subject}unpublished"), line
            continue
        revision = json.loads(lock.read_text())["baseline_revision"]
        assert line.startswith(f"{subject}published at baseline {revision[:12]}")
        assert line.endswith(" matches the enrolled cases") or (
            "; bundle missing" in line
        ), line


_WORKFLOW_DOC = "library/tests/reference_common/docs/gpu-reference-workflow.md"
_TRIAGE_DOC = f"{_WORKFLOW_DOC}#triage"


def _documented(key):
    """Whether the workflow doc's triage table finds a failure by ``key``."""
    doc = Path(__file__).parent / _WORKFLOW_DOC.removeprefix("library/tests/")
    text = doc.read_text(encoding="utf-8")
    section = text.partition("\n## Triage\n")[2].partition("\n#")[0]
    rows = (row for row in section.splitlines() if row.startswith("| `"))
    return any(f"`{key}`" in row.split(" | ")[0] for row in rows)


def _published_variants():
    """Every operation, architecture, and variant with a committed lock.

    Configuration requires a lock for each published architecture, and
    installed trees carry the locks but not the CMake publication list, so
    these variants cover every published one in both layouts. Variants and
    their locks are found in the registry's directories without importing an
    adapter: a broken adapter must fail its own tests, not this module's
    collection.
    """
    from importlib import import_module

    published = []
    for operation in OPERATIONS:
        registry = import_module(f"{operation}_reference.architectures")
        published += [
            (operation, architecture, variant)
            for architecture in registry.ARCHITECTURES
            for variant in registry.variants(architecture)
            if registry.baseline_lock(architecture, variant).is_file()
        ]
    return published


def _published_bundle_data(spec, architecture, variant):
    """Return the bundle data that ``diff`` reads by default, or skip without it.

    The skip says what the tree lacks: a fetched archive or extracted bundle in
    a source checkout, or a staged bundle in an installed tree.
    """
    name = f"{spec.label} {architecture} {variant}"
    bundle = spec.published_bundle(architecture, variant)
    if not bundle.exists():
        archive = bundle.with_name(f"{bundle.name}.tar.gz")
        pointer = archive.with_name(f"{archive.name}.dvc")
        if pointer.is_file():
            pytest.skip(
                f"no {name} bundle data: {archive} has not been fetched and no "
                f"bundle is extracted at {bundle}; fetch the archive with: "
                + display_command(["dvc", "pull", str(pointer)])
            )
        pytest.skip(
            f"no {name} bundle is installed at {bundle}; installed trees have it "
            "only when built with ROCKE_INSTALL_TEST_GPU_REFERENCES=ON and "
            "assembled with its architecture's test artifact"
        )
    return bundle


def _check_published_cohort(operation, architecture, variant):
    """Fail when the published bundle does not record the enrolled cohort.

    Verification rejects every case of such a bundle, so otherwise only a
    matching GPU would notice an enrolled case that was never qualified. This
    reads the bundle data that ``diff`` reads by default and compares it the
    same way; it skips only when the tree has no bundle data for the pair.
    """
    from importlib import import_module

    spec = import_module(f"{operation}_reference.cli").OPERATION
    name = f"{spec.label} {architecture} {variant}"
    bundle = _published_bundle_data(spec, architecture, variant)
    unusable = None
    try:
        differences = spec.diff_bundle(
            bundle, architecture=architecture, variant=variant
        )
    except (ValueError, OSError) as error:
        # Fail outside the handler so pytest reports the message only once.
        unusable = str(error)
    if unusable is not None:
        pytest.fail(
            f"{name} bundle data at {bundle} is unreadable or is not the bundle "
            f"that its committed lock pins:\n{unusable}",
            pytrace=False,
        )
    if differences:
        script = Path(__file__).with_name(f"run_{operation}_reference.py")
        pytest.fail(
            f"{name}: the enrolled cases differ from the published bundle "
            f"{bundle}, so verification on a {architecture} GPU rejects every "
            "case:\n"
            + "".join(f"  {difference}\n" for difference in differences)
            + "A cohort or budget change needs a complete requalification and a "
            "new published archive, lock, and DVC pointer; see "
            f"{_WORKFLOW_DOC}. Repeat this comparison with: "
            + display_command(
                ["python", str(script), "diff", "--arch", architecture]
                + ["--variant", variant]
            ),
            pytrace=False,
        )


@pytest.mark.parametrize(
    ("operation", "architecture", "variant"), _published_variants()
)
def test_published_bundles_record_the_enrolled_cohorts(
    operation, architecture, variant
):
    """A cohort edit cannot pass host tests while its published bundle lags."""
    _check_published_cohort(operation, architecture, variant)


def _check_locked_cohort(operation, architecture, variant):
    """Fail when the committed lock does not pin the enrolled cohort's digest.

    Configuration checks the lock's ``cohort_sha256`` against the manifest
    that the lock pins, so the digest stands for the published bundle's
    cohort. Comparing it with the adapter's cases needs no bundle data, so a
    cohort edit without requalification fails on every host, including CI jobs
    for other processors. A cohort that several processors share therefore
    fails for each processor that was not requalified.
    """
    from importlib import import_module

    spec = import_module(f"{operation}_reference.cli").OPERATION
    name = f"{spec.label} {architecture} {variant}"
    path = spec.architectures.baseline_lock(architecture, variant)
    locked = json.loads(path.read_text()).get("cohort_sha256")
    if locked is None:
        pytest.fail(
            f"{name}: {path} records no cohort_sha256, which every committed "
            "lock needs. Qualification writes it to qualification-lock.json; a "
            "lock qualified before then needs the manifest_cohort_digest, from "
            "reference_common/artifact.py, of the manifest that it pins",
            pytrace=False,
        )
    enrolled = spec.enrolled_cohort_digest(architecture, variant)
    if enrolled != locked:
        script = Path(__file__).with_name(f"run_{operation}_reference.py")
        pytest.fail(
            f"{name}: the enrolled cases differ from the cohort that {path} pins "
            f"(enrolled cohort_sha256 {enrolled[:12]}, locked {locked[:12]}), so "
            f"the published bundle no longer certifies them on a {architecture} "
            "GPU. A change to the cases, their tolerances, or their generated "
            "inputs needs a complete requalification and a new published "
            f"archive, lock, and DVC pointer; see {_WORKFLOW_DOC}. List the "
            "differences with: "
            + display_command(
                ["python", str(script), "diff", "--arch", architecture]
                + ["--variant", variant]
            ),
            pytrace=False,
        )


@pytest.mark.parametrize(
    ("operation", "architecture", "variant"), _published_variants()
)
def test_committed_locks_pin_the_enrolled_cohorts(operation, architecture, variant):
    """A cohort edit fails without bundle data, so every host and CI job sees it."""
    _check_locked_cohort(operation, architecture, variant)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_locked_cohort_check_names_the_lock_and_the_requalification(
    tmp_path, monkeypatch, operation
):
    from importlib import import_module

    spec = import_module(f"{operation}_reference.cli").OPERATION
    architecture = spec.architectures.ARCHITECTURES[0]
    variant = spec.architectures.resolve_variant(architecture)
    target = spec.architectures.get_adapter(architecture, variant)
    lock = tmp_path / "baseline_lock.json"
    monkeypatch.setattr(
        spec.architectures, "baseline_lock", lambda name, variant=None: lock
    )
    pinned = {"schema": 2, "baseline_revision": "a" * 40, "manifest_sha256": "0" * 64}
    lock.write_text(json.dumps(pinned))

    def check():
        return _outcome(lambda: _check_locked_cohort(operation, architecture, variant))

    outcome, message = check()
    assert outcome == "fail"
    assert _mentions(message, lock) and "records no cohort_sha256" in message
    locked = spec.enrolled_cohort_digest(architecture, variant)
    lock.write_text(json.dumps(dict(pinned, cohort_sha256=locked)))
    assert check() == ("pass", "")

    # The author enrolled one more case, or edited a cohort that this processor
    # shares, without requalifying.
    monkeypatch.setattr(target, "CASES", target.CASES[:-1])
    outcome, message = check()
    assert outcome == "fail"
    assert _mentions(message, lock) and f"locked {locked[:12]}" in message
    assert "needs a complete requalification and a new published archive" in message
    assert _WORKFLOW_DOC in message
    (diff,) = _commands(message)
    assert _same(diff[0], Path(__file__).with_name(f"run_{operation}_reference.py"))
    args = spec._parser().parse_args(diff[1:])
    assert (args.command, args.arch, args.variant) == ("diff", architecture, variant)


def test_published_variants_are_listed_without_importing_adapters(monkeypatch):
    """A broken adapter cannot stop the shared tests from collecting."""
    from importlib import import_module

    expected = []
    for operation in OPERATIONS:
        registry = import_module(f"{operation}_reference.architectures")
        root = Path(registry.__file__).parent
        expected += [
            (operation, lock.parent.parent.name, lock.parent.name)
            for architecture in registry.ARCHITECTURES
            for lock in sorted((root / architecture).glob("*/baseline_lock.json"))
        ]
        # A None entry makes any import of a processor's package or adapter
        # fail, as a broken one would.
        for architecture in registry.ARCHITECTURES:
            package = f"{registry.__name__}.{architecture}"
            monkeypatch.setitem(sys.modules, package, None)
            for variant in registry.variants(architecture):
                monkeypatch.setitem(sys.modules, f"{package}.{variant}", None)
    assert _published_variants() == expected


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("layout", ["archive", "extracted", "installed"])
def test_published_cohort_check_names_unqualified_cases(
    tmp_path, monkeypatch, operation, layout
):
    """Fail on an unqualified case or unpinned data; skip only when data is absent."""
    import shutil
    from importlib import import_module

    from reference_common import cli as common_cli
    from reference_common.paths import default_bundle_path

    registry = import_module(f"{operation}_reference.architectures")
    variant = registry.resolve_variant(registry.ARCHITECTURES[0])
    target = registry.get_adapter(registry.ARCHITECTURES[0], variant)
    cases = target.CASES
    # The bundle was qualified before the author enrolled the last case.
    monkeypatch.setattr(target, "CASES", cases[:-1])
    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    monkeypatch.setattr(
        spec.architectures, "baseline_lock", lambda name, variant=None: lock
    )
    tests = tmp_path / (
        "bin/hip_kernel_provider/tests/library/tests"
        if layout == "installed"
        else "rocke/library/tests"
    )
    monkeypatch.setattr(
        common_cli,
        "default_bundle_path",
        lambda _, name, *, operation, variant: default_bundle_path(
            tests, name, operation=operation, variant=variant
        ),
    )
    # Where each layout keeps the bundle, written out rather than looked up,
    # so that a wrong default location cannot place and find it alike.
    published = tmp_path.resolve() / (
        "bin/hip_kernel_provider/engines/test_arch_content/rocke"
        if layout == "installed"
        else "rocke/library/tests/reference_bundles"
    )
    published /= f"{operation}/{architecture}/{variant}"
    published.parent.mkdir(parents=True)
    archive = published.with_name(f"{variant}.tar.gz")
    pointer = archive.with_name(f"{archive.name}.dvc")
    if layout != "installed":
        pointer.write_text("outs:\n")
    if layout == "archive":
        pack(bundle, archive, lock, operation=operation)
    else:
        bundle.rename(published)

    def check():
        return _outcome(
            lambda: _check_published_cohort(operation, architecture, variant)
        )

    assert check() == ("pass", "")
    monkeypatch.setattr(target, "CASES", cases)
    outcome, message = check()
    assert outcome == "fail"
    assert f"{cases[-1].id} (" in message and _WORKFLOW_DOC in message
    (diff,) = _commands(message)
    assert _same(diff[0], Path(__file__).with_name(f"run_{operation}_reference.py"))
    args = spec._parser().parse_args(diff[1:])
    assert (args.command, args.arch, args.variant) == ("diff", architecture, variant)
    # Data the lock does not pin, such as an archive from an earlier pointer,
    # fails instead of skipping, though it records the enrolled cohort.
    monkeypatch.setattr(target, "CASES", cases[:-1])
    lock.write_text(
        json.dumps(dict(json.loads(lock.read_text()), manifest_sha256="0" * 64))
    )
    outcome, message = check()
    assert outcome == "fail"
    key = "does not match the pinned lock"
    assert _documented(key) and key in message
    if layout == "archive":
        # A truncated fetch fails with the step that replaces it: delete the
        # archive, then fetch it again through its pointer.
        archive.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
        outcome, message = check()
        assert outcome == "fail"
        assert _documented("cannot be read") and "cannot be read" in message
        assert _deletes(_step(message), archive)
        (pull,) = _commands(message)
        assert pull[:2] == ["dvc", "pull"] and _same(pull[2], pointer)

    if layout == "archive":
        archive.unlink()
    else:
        shutil.rmtree(published)
    outcome, reason = check()
    assert outcome == "skip"
    if layout == "installed":
        assert _mentions(reason, published)
        assert "ROCKE_INSTALL_TEST_GPU_REFERENCES=ON" in reason
    else:
        assert _mentions(reason, archive)
        (pull,) = _commands(reason)
        assert pull[:2] == ["dvc", "pull"] and _same(pull[2], pointer)


# A host stand-in for the frozen rocKE runtime that a bundle's runner replays
# with, so replay needs no HIP. Device buffers are host memory, and the device
# reports the target id written beside hip_module. A launch requires every
# argument of the frozen kernel's signature, as the runtime's argument packer
# does, then zeroes each bound buffer that the worker poisoned with 0xff bytes,
# so an output that the replay leaves unbound still reads back as unwritten.
_STUB_RUNTIME = {
    "rocke/__init__.py": "",
    "rocke/core/__init__.py": "",
    # Recorded device targets carry no profile suffix such as -strict, which is
    # all that the runtime's conversion removes before features are split off.
    "rocke/core/arch.py": """
def compiler_target_from_target_id(target_id):
    return target_id
""",
    "rocke/runtime/hip_module.py": """
from pathlib import Path


def get_device_target_id(device=0):
    return Path(__file__).with_name("device_target").read_text()


def get_device_arch(device=0):
    return get_device_target_id(device).split(":", 1)[0]
""",
    "rocke/runtime/__init__.py": """
import ctypes
from dataclasses import dataclass

_MEMORY = {}


class Runtime:
    def memcpy_h2d(self, destination, source, nbytes):
        ctypes.memmove(destination, source, nbytes)

    def memcpy_d2h(self, destination, source, nbytes):
        ctypes.memmove(destination, source, nbytes)

    def memset(self, pointer, value, nbytes):
        ctypes.memset(pointer, value, nbytes)

    def sync(self):
        pass


class DeviceMem:
    def __init__(self, nbytes):
        self._pointer = 0
        self.realloc(nbytes)

    def ptr(self):
        return self._pointer

    def realloc(self, nbytes):
        _MEMORY.pop(self._pointer, None)
        self._pointer = 0
        if nbytes > 0:
            memory = ctypes.create_string_buffer(nbytes)
            self._pointer = ctypes.addressof(memory)
            _MEMORY[self._pointer] = memory


@dataclass
class LaunchConfig:
    grid: tuple = (1, 1, 1)
    block: tuple = (64, 1, 1)


@dataclass(frozen=True)
class LaunchSummary:
    launches: int


class KernelLauncher:
    def __init__(self, *, hsaco, kernel_name, signature, **options):
        self.kernel_name = kernel_name
        self.signature = list(signature)

    def __call__(self, values, *, config):
        poisoned = []
        for argument in self.signature:
            name, kind = argument["name"], argument["type"]
            if name not in values:
                raise KeyError(f"missing kernel arg {name!r}")
            value = values[name]
            if kind.startswith("ptr<"):
                pointer = value.ptr() if hasattr(value, "ptr") else value
                if pointer and pointer not in _MEMORY:
                    raise ValueError(f"kernel arg {name!r} is not a device buffer")
                memory = _MEMORY.get(pointer)
                if memory is not None and memory.raw.count(0xFF) == len(memory):
                    poisoned.append(memory)
            elif kind == "f32":
                float(value)
            elif kind in ("i32", "i64"):
                int(value)
            else:
                raise ValueError(f"unsupported kernel arg type {kind!r} for {name}")
        for memory in poisoned:
            ctypes.memset(memory, 0, len(memory))
        return LaunchSummary(launches=1)
""",
}


# A request can gain keys that published runners predate, such as the selected
# variant that verification now sends. An unenrolled name fails any runner that
# resolves the key, so the key passes only when the runner ignores it.
@pytest.mark.parametrize(
    "added", [{}, {"variant": "unenrolled"}], ids=["current-request", "variant-key"]
)
@pytest.mark.parametrize(
    ("operation", "architecture", "variant"), _published_variants()
)
def test_published_runners_replay_the_current_request(
    tmp_path, monkeypatch, operation, architecture, variant, added
):
    """Every published frozen runner accepts the request that verification sends.

    A published bundle's runner is frozen, but verification builds its request
    with the current harness, so a request change must keep every published
    runner replaying without requalification. This verifies each enrolled case
    of the published bundle with a host stand-in for the frozen runtime, so it
    needs no GPU, and ends each verification after its replay, since the
    stand-in computes no output to compare.
    """
    from importlib import import_module

    from reference_common import cli as common_cli
    from reference_common.runner import run_worker
    from reference_common.session import reuse_workers

    spec = import_module(f"{operation}_reference.cli").OPERATION
    bundle = _published_bundle_data(spec, architecture, variant)
    if bundle.is_file():
        lock = spec.architectures.baseline_lock(architecture, variant)
        unpack(bundle, tmp_path / "bundle", lock, operation=operation)
        bundle = tmp_path / "bundle"
    manifest = spec.load_bundle(bundle, architecture=architecture, variant=variant)
    runtime = tmp_path / "runtime"
    for name, source in _STUB_RUNTIME.items():
        (runtime / name).parent.mkdir(parents=True, exist_ok=True)
        (runtime / name).write_text(source)

    class Replayed(Exception):
        """Ends verification once the frozen worker has replayed its case."""

    def replay(request, *, platform, **paths):
        # Verification replays the frozen baseline before the current kernel.
        assert request["mode"] == "replay"
        request = dict(request, **added)
        try:
            result = run_worker(request, platform=runtime, **paths)
        except (RuntimeError, TimeoutError) as error:
            result = error
        raise Replayed(request, result)

    def check_replay(case):
        key = spec.manifest_key(
            manifest, case, architecture=architecture, variant=variant
        )
        (runtime / "rocke/runtime/device_target").write_text(
            manifest["cases"][key]["device_target"]
        )
        with pytest.raises(Replayed) as replayed:
            spec.verify_case(
                case,
                bundle=bundle,
                manifest=manifest,
                architecture=architecture,
                variant=variant,
            )
        request, result = replayed.value.args
        if isinstance(result, Exception):
            pytest.fail(
                f"{spec.label} {architecture} case {case.id}: the published frozen "
                "runner rejected the request that verification sends, with keys "
                f"{sorted(request)}, so verification fails on a {architecture} "
                "GPU. A published runner cannot change: keep sending the request "
                "keys and case fields that it reads, or requalify the bundle:\n"
                f"{result}",
                pytrace=False,
            )
        # Every repetition read back what the stand-in's launch wrote.
        assert not any(output.any() for output in result[0])

    monkeypatch.setattr(common_cli, "run_worker", replay)
    cases = spec.architectures.get_adapter(architecture, variant).CASES
    # Without a worker session, run_worker runs the worker module as a script.
    check_replay(cases[0])
    # Verification reuses one worker per bundle, which calls the module's run().
    with reuse_workers(spec.worker_module):
        for case in cases:
            check_replay(case)


# Qualification runs this tree's adapter with the baseline snapshot's rocKE and
# library ahead of it on the path. This child imports them the same way, but
# replaces compilation and code-object loading so that it runs on any host.
_PREPARE_ENROLLED_CASES = """
import json
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import rocke
import rocke.helpers
import rocke.runtime

operation, architecture, variant, source, output = sys.argv[1:]
source = Path(source).resolve()


def compile_kernel(kernel, **options):
    return SimpleNamespace(hsaco=b"", kernel_name=kernel.name)


class KernelLauncher:
    def __init__(self, *, hsaco, kernel_name, signature, **options):
        self._hsaco = hsaco
        self.kernel_name = kernel_name
        self.signature = list(signature)


rocke.compile_kernel = rocke.helpers.compile_kernel = compile_kernel
rocke.runtime.KernelLauncher = KernelLauncher
target = import_module(f"{operation}_reference.architectures").get_adapter(
    architecture, variant
)
failures = {}
for case in target.CASES:
    try:
        target.prepare(case, str(source / "library"))
    except Exception as error:
        failures[case.id] = f"{type(error).__name__}: {error}"
outside = {
    name: sys.modules[name].__file__
    for name in ("rocke", "kernels", "dispatch")
    if name in sys.modules
    and not Path(sys.modules[name].__file__ or "").resolve().is_relative_to(source)
}
Path(output).write_text(
    json.dumps(
        {
            "cases": [case.id for case in target.CASES],
            "failures": failures,
            "outside": outside,
        }
    )
)
"""


def _baseline_repository(tests, revision):
    """Return the git repository that holds ``revision``, or skip saying why not.

    Only a source checkout's history can export a locked baseline: installed
    trees carry none, and a shallow clone can lack the revision.
    """
    import shutil

    from reference_common.paths import installed_test_root

    if installed_test_root(tests) is not None:
        pytest.skip(
            "installed test trees have no git history to export locked baselines "
            "from; run this check from a rocKE source checkout"
        )
    if shutil.which("git") is None:
        pytest.skip("git is not installed; exporting a locked baseline needs it")
    found = subprocess.run(
        ["git", "-C", str(tests), "rev-parse", "--show-toplevel"],
        capture_output=True,
        check=False,
        text=True,
    )
    if found.returncode:
        pytest.skip(
            f"{tests} is not in a git checkout, so locked baselines cannot be "
            f"exported: {found.stderr.strip()}"
        )
    repository = Path(found.stdout.strip())
    present = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if present.returncode:
        pytest.skip(
            f"locked baseline {revision} is not in the git repository at "
            f"{repository}, as in a shallow clone; fetch it to run this check"
        )
    return repository


@pytest.fixture(scope="session")
def locked_baseline_sources():
    """Export each locked baseline revision at most once per session.

    Yields ``sources(revision)``, which returns the production sources that
    qualification at ``revision`` runs, or skips when this tree cannot export
    them.
    """
    import tempfile

    from reference_common.source import snapshot

    tests = Path(__file__).resolve().parent
    exported = {}
    with tempfile.TemporaryDirectory(prefix="rocke-reference-baselines-") as temporary:

        def sources(revision):
            if revision not in exported:
                output = Path(temporary) / revision
                snapshot(_baseline_repository(tests, revision), revision, output)
                exported[revision] = output / "source"
            return exported[revision]

        yield sources


def _check_baseline_preparation(operation, architecture, variant, source, revision):
    """Fail unless the adapter prepares every enrolled case with ``source``.

    ``source`` holds the ``platform/python`` and ``library`` trees of baseline
    ``revision``. As in the qualification worker, a child interpreter without
    Torch imports them ahead of this tree's tests, which provide the adapter.
    It prepares each case without compiling, loading, or launching a kernel.
    """
    import os
    import tempfile
    from importlib import import_module

    from reference_common.session import _NO_TORCH

    spec = import_module(f"{operation}_reference.cli").OPERATION
    name = f"{spec.label} {architecture} {variant}"
    paths = (source / "platform/python", source / "library", Path(__file__).parent)
    with tempfile.TemporaryDirectory(prefix=f"rocke-{operation}-prepare-") as work:
        result = Path(work) / "prepared.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-s",
                "-c",
                _NO_TORCH + _PREPARE_ENROLLED_CASES,
                operation,
                architecture,
                variant,
                str(source),
                str(result),
            ],
            cwd=work,
            env=dict(
                os.environ,
                PYTHONPATH=os.pathsep.join(str(path.resolve()) for path in paths),
                PYTHONNOUSERSITE="1",
                PYTHONDONTWRITEBYTECODE="1",
            ),
            capture_output=True,
            check=False,
            text=True,
            timeout=300,
        )
        prepared = None if completed.returncode else json.loads(result.read_text())
    if prepared is None:
        pytest.fail(
            f"{name}: preparing the enrolled cases with locked baseline {revision} "
            f"failed with exit code {completed.returncode}:\n"
            f"{completed.stderr[-12000:]}",
            pytrace=False,
        )
    if prepared["outside"]:
        escaped = ", ".join(
            f"{package} from {path}" for package, path in prepared["outside"].items()
        )
        pytest.fail(
            f"{name}: preparation imported {escaped}, outside the sources of "
            f"locked baseline {revision}",
            pytrace=False,
        )
    if prepared["failures"]:
        errors = {}
        for case_id, error in prepared["failures"].items():
            errors.setdefault(error, []).append(case_id)
        adapter = (
            f"library/tests/{operation}_reference/architectures/{architecture}/"
            f"{variant}/__init__.py"
        )
        pytest.fail(
            f"{name}: the current adapter cannot prepare "
            f"{len(prepared['failures'])} of {len(prepared['cases'])} enrolled "
            f"cases with the sources of its locked baseline {revision}, so this "
            "pair cannot be requalified at that baseline:\n"
            + "".join(
                f"  {', '.join(ids)}:\n    {error}\n" for error, ids in errors.items()
            )
            + f"Extend {adapter} for that baseline's API, or requalify the cohort "
            f"at a newer committed baseline; see {_WORKFLOW_DOC}.",
            pytrace=False,
        )


@pytest.mark.parametrize(
    ("operation", "architecture", "variant"), _published_variants()
)
def test_adapters_prepare_against_their_locked_baselines(
    operation, architecture, variant, locked_baseline_sources
):
    """An adapter port cannot silently drop its locked baseline's API.

    Verification runs the adapter only against the current sources, so
    otherwise the next GPU qualification at that baseline would be the first
    to fail.
    """
    from importlib import import_module

    registry = import_module(f"{operation}_reference.architectures")
    lock = json.loads(registry.baseline_lock(architecture, variant).read_text())
    revision = lock["baseline_revision"]
    _check_baseline_preparation(
        operation, architecture, variant, locked_baseline_sources(revision), revision
    )


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("broken", ["prepare", "isolation", "startup"])
def test_baseline_preparation_check_names_what_failed(tmp_path, operation, broken):
    """Fail with each case and its error, an escaped import, or the child's error."""
    from importlib import import_module

    registry = import_module(f"{operation}_reference.architectures")
    architecture = registry.ARCHITECTURES[0]
    variant = registry.resolve_variant(architecture)
    cases = registry.get_adapter(architecture, variant).CASES
    # A stand-in baseline whose production packages no longer match the adapter.
    source = tmp_path / "source"
    for package in ("rocke", "rocke/helpers", "rocke/runtime"):
        (source / "platform/python" / package).mkdir(parents=True)
        (source / "platform/python" / package / "__init__.py").touch()
    (source / "library/kernels").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere/kernels/__init__.py"
    kernels = {
        "prepare": "raise ImportError('kernels moved')\n",
        # As if an installed or current kernels package shadowed the baseline's.
        "isolation": f"__file__ = {str(elsewhere)!r}\n",
        "startup": "",
    }[broken]
    (source / "library/kernels/__init__.py").write_text(kernels)
    if broken == "startup":
        (source / "platform/python/rocke/runtime/__init__.py").write_text(
            "raise ImportError('runtime moved')\n"
        )
    revision = "a" * 40
    outcome, message = _outcome(
        lambda: _check_baseline_preparation(
            operation, architecture, variant, source, revision
        )
    )
    assert outcome == "fail"
    if broken == "prepare":
        assert (
            f"cannot prepare {len(cases)} of {len(cases)} enrolled cases with the "
            f"sources of its locked baseline {revision}"
        ) in message
        ids = ", ".join(case.id for case in cases)
        assert f"\n  {ids}:\n    ImportError: kernels moved\n" in message
        assert (
            f"Extend library/tests/{operation}_reference/architectures/"
            f"{architecture}/{variant}/__init__.py for that baseline's API"
        ) in message
    elif broken == "isolation":
        assert message.endswith(
            f"preparation imported kernels from {elsewhere}, outside the sources "
            f"of locked baseline {revision}"
        )
    else:
        assert f"with locked baseline {revision} failed with exit code 1:" in message
        assert message.rstrip().endswith("ImportError: runtime moved")


@pytest.mark.parametrize("missing", ["installed", "git", "checkout", "revision"])
def test_baseline_preparation_check_says_why_it_skips(tmp_path, monkeypatch, missing):
    """A tree that cannot export a locked baseline skips with what it lacks."""
    import shutil

    tests = tmp_path / "rocke/library/tests"
    tests.mkdir(parents=True)
    # Never find a repository that encloses the temporary directory.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    revision = "a" * 40
    if missing == "installed":
        tests = tmp_path / "bin/hip_kernel_provider/tests/library/tests"
    elif missing == "git":
        monkeypatch.setattr(shutil, "which", lambda name: None)
    elif shutil.which("git") is None:
        pytest.skip("git is not installed")
    elif missing == "revision":
        subprocess.run(["git", "init", "-q", str(tmp_path / "rocke")], check=True)
    with pytest.raises(pytest.skip.Exception) as skip:
        _baseline_repository(tests, revision)
    expected = {
        "installed": "installed test trees have no git history",
        "git": "git is not installed",
        "checkout": f"{tests} is not in a git checkout",
        "revision": (
            f"locked baseline {revision} is not in the git repository at "
            f"{tmp_path / 'rocke'}, as in a shallow clone"
        ),
    }[missing]
    assert expected in str(skip.value)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_verify_explains_how_to_unpack_an_archive(tmp_path, capsys, operation):
    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    archive = tmp_path / f"{architecture}.tar.gz"
    pack(bundle, archive, lock, operation=operation)
    command = ["verify", "--arch", architecture, "--lock", str(lock), "--bundle"]
    with pytest.raises(SystemExit) as exit:
        spec.main([*command, str(archive)])
    assert exit.value.code == 2
    error = capsys.readouterr().err
    assert "Traceback" not in error
    # The command unpacks the archive with the selected lock into the
    # directory beside it, which the message then says to pass.
    (extract,) = _commands(error)
    args = _unpacked(extract)
    assert (args.command, args.operation) == ("unpack", operation)
    assert args.variant == spec.architectures.resolve_variant(architecture)
    assert _same(args.archive, archive) and _same(args.lock, lock)
    passed = _passed(error, "--bundle")
    assert _same(passed, tmp_path / architecture) and _same(args.bundle, passed)
    # The suggested command extracts and authenticates a usable bundle.
    subprocess.run([sys.executable, *extract], check=True, capture_output=True)
    assert spec.load_bundle(Path(passed), lock, architecture=architecture)
    with pytest.raises(SystemExit) as exit:
        spec.main([*command, str(tmp_path / "missing")])
    assert exit.value.code == 2
    error = capsys.readouterr().err
    assert "not found" in error and "artifact.py" not in error


@pytest.mark.parametrize("operation", OPERATIONS)
def test_verify_reports_an_unusable_lock_or_bundle_once(
    tmp_path, capsys, monkeypatch, operation
):
    """Verification refuses a missing lock or rejected bundle with one message."""
    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    archive = tmp_path / f"{architecture}.tar.gz"
    pack(bundle, archive, lock, operation=operation)

    def refused(*args):
        with pytest.raises(SystemExit) as exit:
            spec.main(["verify", "--arch", architecture, "--bundle", *map(str, args)])
        assert exit.value.code == 2
        output = capsys.readouterr()
        assert "Traceback" not in output.err
        return output.err

    # An architecture with no published bundle has no committed lock.
    variant = spec.architectures.resolve_variant(architecture)
    unpublished = (tmp_path / "unpublished/baseline_lock.json").resolve()
    monkeypatch.setattr(
        spec.architectures, "baseline_lock", lambda name, variant=None: unpublished
    )
    for selected in (bundle, archive):
        error = refused(selected)
        assert (
            f"{architecture} {variant} has no published {spec.label} bundle: "
            f"{unpublished} does not exist; pass a candidate's --bundle and --lock"
        ) in error
        # No unpack command names the lock that does not exist.
        assert "artifact.py" not in error
    assert f"lock not found: {unpublished}" in refused(bundle, "--lock", unpublished)
    # load_bundle rejects a cohort the bundle does not record and stale data.
    target = spec.architectures.get_adapter(architecture)
    cases = target.CASES
    monkeypatch.setattr(target, "CASES", cases[1:])
    error = refused(bundle, "--lock", lock)
    assert error.count("does not cover the complete enrolled cohort") == 1
    monkeypatch.setattr(target, "CASES", cases)
    (bundle / "payload/kernel.hsaco").write_bytes(b"modified")
    error = refused(bundle, "--lock", lock)
    assert error.count("payload has missing, modified, or extra files") == 1
    assert error.count("likely cause:") == 1


@pytest.mark.parametrize("operation", OPERATIONS)
def test_verify_subset_runs_selected_cases_and_says_it_is_partial(
    tmp_path, capsys, monkeypatch, operation
):
    from reference_common.cli import ReferenceOperation

    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation)
    cases = spec.architectures.get_adapter(architecture).CASES
    ran = []

    def verify_case(self, case, **kwargs):
        # A failure's rerun command repeats the lock that authenticated the bundle.
        assert kwargs["lock"] == lock
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
    variant = spec.architectures.resolve_variant(architecture)
    error = capsys.readouterr().err
    assert f"not enrolled for {architecture} {variant}: unenrolled" in error
    assert error.endswith(f"cases --arch {architecture} --variant {variant}\n")


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
    assert set(commands.choices) == {
        "snapshot",
        "qualify",
        "verify",
        "cases",
        "diff",
        "status",
    }
    summaries = {action.dest: action.help for action in commands._choices_actions}
    for name, command in commands.choices.items():
        assert summaries[name] and command.description, name
        for action in command._actions:
            assert action.help, f"{name} {action.option_strings} has no help"
    # Scripts can tell a difference or failed comparison from unusable data.
    for name in ("diff", "verify", "status"):
        description = commands.choices[name].description
        assert "status 1" in description and " 2 when the lock or bundle" in description
    assert "PYTHONPATH=platform/python" in commands.choices["diff"].description
    assert "--case" in commands.choices["verify"].description


@pytest.mark.parametrize("operation", OPERATIONS)
def test_operation_without_torch_offers_no_torch_cross_check(tmp_path, operation):
    """A new operation need not declare a placeholder Torch reference."""
    from dataclasses import replace
    from importlib import import_module

    spec = replace(import_module(f"{operation}_reference.cli").OPERATION, torch=None)
    parser = spec._parser()
    qualify = ["qualify", "--baseline", "baseline", "--output", "bundle"]
    assert parser.parse_args(qualify).torch_reference is None
    with pytest.raises(SystemExit):
        parser.parse_args([*qualify, "--torch-reference"])
    with pytest.raises(ValueError, match="qualification has no Torch reference"):
        spec.qualify(
            tmp_path / "baseline",
            tmp_path / "bundle",
            2,
            spec.architectures.ARCHITECTURES[0],
            torch_reference="cpu",
        )
    assert not (tmp_path / "bundle").exists()


def _verifiable_case(monkeypatch, operation, index=0):
    """An architecture's first case enrolled alone, with a synthetic qualified entry."""
    from dataclasses import asdict
    from importlib import import_module

    spec = import_module(f"{operation}_reference.cli").OPERATION
    architecture = spec.architectures.ARCHITECTURES[index]
    target = spec.architectures.get_adapter(architecture)
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
@pytest.mark.parametrize("index", [0, -1])
@pytest.mark.parametrize(
    "failure", ["baseline-output", "current-output", "qualified-target", "features"]
)
def test_verification_failures_name_side_values_cause_and_next_step(
    tmp_path, monkeypatch, operation, index, failure
):
    from reference_common import cli as common_cli

    spec, architecture, case, manifest, output = _verifiable_case(
        monkeypatch, operation, index
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
    # Distinct paths, so that a rerun command that swaps them is wrong.
    bundle, current_root = tmp_path / "bundle", tmp_path / "current"
    with pytest.raises(AssertionError if "output" in failure else ValueError) as error:
        spec.verify_case(
            case,
            bundle=bundle,
            manifest=manifest,
            current_root=current_root,
            architecture=architecture,
        )
    message = str(error.value)
    problem = message.splitlines()[0]
    assert f"{spec.label} {architecture} case {case.id}" in problem
    assert re.search(r"^\s*likely cause:\s*\S", message, re.MULTILINE | re.IGNORECASE)
    step = _step(message)
    assert step.strip() and _TRIAGE_DOC in message
    # A step may rule out a wider tolerance, never suggest one.
    assert not re.search(
        r"(?<!rather than )(?<!instead of )(?<!not )\bwiden", step, re.IGNORECASE
    )
    key = {
        "baseline-output": "reference qualification failure",
        "current-output": "exceeds remaining limit",
        "qualified-target": "differs from the qualified target",
        "features": "used different target features",
    }[failure]
    assert _documented(key) and key in problem
    baseline, current = "frozen baseline (replay)", "current kernel (source)"
    if failure == "baseline-output":
        # The environment, not the current kernel, changed the frozen output,
        # so the current kernel did not run.
        assert modes == ["replay"]
        assert baseline in problem and current not in problem
        assert re.search(r"repetition\D+2\D+2\b", problem)
        measured, recorded = array_digest(changed)[:12], entry["output_digest"][:12]
        assert problem.index(measured) < problem.index(recorded)
    elif failure == "current-output":
        assert current in problem and baseline not in problem
        assert re.search(r"repetition\D+2\D+2\b", problem)

        def shown(label):
            value = re.search(rf"{re.escape(label)}\s*=?\s*([^\s,;]+)", problem)
            return float(value.group(1).rstrip("."))

        budget = ErrorBudget(**entry["budget"])
        distance = spec.metric.distance(
            decode(changed, case.dtype), decode(output, case.dtype), entry
        )
        for label, value in (
            (spec.metric.name, distance),
            (key, budget.comparison_limit),
            ("baseline_bound", budget.baseline_error_bound),
            ("original_tolerance", case.tolerance),
        ):
            assert math.isclose(shown(label), value, rel_tol=1e-6), label
        # The rerun repeats this case with the same bundle and current tree.
        (rerun,) = re.findall(r"`([^`]+)`", step)
        script, *arguments = _arguments(rerun)[1:]
        assert _same(script, Path(__file__).with_name(f"run_{operation}_reference.py"))
        args = spec._parser().parse_args(arguments)
        variant = spec.architectures.resolve_variant(architecture)
        assert (args.command, args.arch, args.variant, args.cases, args.lock) == (
            "verify",
            architecture,
            variant,
            [case.id],
            None,
        )
        assert _same(args.bundle, bundle) and _same(args.current_root, current_root)
    else:
        # The side that ran on the other target comes first, then the target
        # it was expected to match.
        side = baseline if failure == "qualified-target" else current
        assert side in problem
        assert problem.index(side) < problem.index(other_target)
        assert problem.index(other_target) < problem.index(entry["device_target"])


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
    assert f"\n  see {_TRIAGE_DOC}\n" in message
    assert message.endswith(log)
    # Only the current side points at the adapter that calls production APIs.
    variant = spec.architectures.resolve_variant(architecture)
    adapter = (
        f"library/tests/{operation}_reference/architectures/{architecture}/"
        f"{variant}/__init__.py"
    )
    assert (adapter in message) == (mode == "source")


def test_failures_point_at_the_workflow_triage_section():
    """The section that failures link to exists in source and installed trees."""
    from reference_common import cli as common_cli

    path, _, anchor = common_cli._TRIAGE.partition("#")
    # The path is relative to the rocKE root, or to an installed tree's tests
    # directory; both hold library/tests.
    doc = Path(common_cli.__file__).resolve().parents[3] / path
    headings = {
        re.sub(r"[^\w\- ]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
        for line in doc.read_text().splitlines()
        if re.match(r"#{1,6} ", line)
    }
    assert anchor in headings


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
@pytest.mark.parametrize("index", [0, -1])
@pytest.mark.parametrize("layout", ["source", "installed"])
@pytest.mark.parametrize(
    "problem",
    ["stale", "no-manifest", "modified", "archive", "candidate", "candidate-archive"],
)
def test_rejected_bundles_say_how_to_replace_them(
    tmp_path, monkeypatch, operation, index, layout, problem
):
    spec, architecture, bundle, lock = _qualified_cohort(tmp_path, operation, index)
    variant = spec.architectures.resolve_variant(architecture)
    tests = _relocate(monkeypatch, tmp_path, operation, layout)
    if layout == "source":
        pointer = _track(tests, operation, architecture, variant)
    candidate = problem.startswith("candidate")
    if not candidate:
        # The committed lock authenticates the published bundle; other
        # architectures' locks are elsewhere.
        monkeypatch.setattr(
            spec.architectures,
            "baseline_lock",
            lambda name, variant=None: (
                lock if name == architecture else tmp_path / f"{name}.json"
            ),
        )
    # The problem names these paths and values.
    paths, values = [], []
    if problem in ("stale", "candidate"):
        # The lock now pins a newer manifest than the one extracted earlier.
        extracted = (bundle / "manifest.json").read_bytes()
        manifest = json.loads(extracted)
        manifest["baseline_revision"] = "b" * 40
        _relock(bundle, manifest, lock)
        (bundle / "manifest.json").write_bytes(extracted)
        paths += [bundle / "manifest.json", lock]
        values += [hashlib.sha256(extracted).hexdigest()[:12]]
        values += [json.loads(lock.read_text())["manifest_sha256"][:12]]
        key = "does not match the pinned lock"
    elif problem == "no-manifest":
        (bundle / "manifest.json").unlink()
        paths += [bundle]
        key = "has no manifest.json"
    elif problem == "modified":
        (bundle / "payload/kernel.hsaco").write_bytes(b"modified")
        values += ["kernel.hsaco"]
        key = "missing, modified, or extra files"
    else:
        archive = tmp_path / f"{architecture}.tar.gz"
        pack(bundle, archive, lock, operation=operation)
        bundle = archive
        paths += [archive]
        key = None
    with pytest.raises(ValueError) as error:
        spec.load_bundle(bundle, lock if candidate else None, architecture=architecture)
    message = str(error.value)
    first = message.splitlines()[0]
    assert all(_mentions(first, path) for path in paths)
    assert all(value in first for value in values)
    if problem == "modified":
        # Only the one rewritten payload file differs.
        assert re.search(r"\b1 differ", first)
    if problem in ("stale", "candidate"):
        # Each digest follows the file that it belongs to, in either clause order.
        parts = re.split(f"({'|'.join(values)})", first)
        before = dict(zip(parts[1::2], parts[0::2]))
        assert all(_mentions(before[value], path) for path, value in zip(paths, values))
    if key:
        assert _documented(key) and key in first and _TRIAGE_DOC in message
    commands = _commands(message)
    if problem == "candidate":
        # A candidate comes from qualification, so there is nothing to fetch.
        assert commands == [] and "dvc pull" not in message
        assert "qualification-lock.json" in _step(message)
    elif bundle.is_file():
        # The archive unpacks, with the selected lock, into the directory
        # beside it that the message then says to pass.
        assert "dvc pull" not in message
        (extract,) = commands
        args = _unpacked(extract)
        assert (args.command, args.operation) == ("unpack", operation)
        assert args.variant == variant
        assert _same(args.archive, archive) and _same(args.lock, lock)
        passed = _passed(message, "--bundle")
        assert _same(passed, tmp_path / architecture) and _same(args.bundle, passed)
        unpack(args.archive, args.bundle, args.lock, operation=args.operation)
        selected = lock if candidate else None
        assert spec.load_bundle(Path(passed), selected, architecture=architecture)
    elif layout == "installed":
        # Installation does not replace a directory either, and needs no fetch.
        assert commands == [] and "dvc pull" not in message
        installed = tmp_path / "bin/hip_kernel_provider/engines/test_arch_content"
        installed /= f"rocke/{operation}/{architecture}/{variant}"
        step = _step(message)
        assert _deletes(step, bundle)
        assert "install" in step and _mentions(step, installed)
    else:
        # Unpack does not replace a directory, so the stale one goes first.
        assert _deletes(_step(message), bundle)
        pull, extract = commands
        assert pull[:2] == ["dvc", "pull"] and _same(pull[2], pointer)
        args = _unpacked(extract)
        assert (args.command, args.operation) == ("unpack", operation)
        assert args.variant == variant
        # The archive that the pointer tracks replaces the rejected bundle.
        assert _same(args.archive, pointer.with_suffix(""))
        assert _same(args.bundle, bundle) and _same(args.lock, lock)


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
    # An untracked archive cannot be fetched again; only the mismatch is stated.
    assert f"{archive} is not the archive that {newer} pins" in message
    assert "dvc pull" not in message
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


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "pairing",
    [
        "published",
        "another-variant",
        "another-architecture",
        "candidate-archive",
        "candidate-lock",
    ],
)
def test_unpack_suggests_a_fetch_only_for_a_published_pair(
    tmp_path, operation, pairing
):
    """Fetching again repairs only a DVC-tracked archive and its committed lock.

    The CLI runs from a tree whose archives the committed locks no longer pin,
    as configuration does after a lock update without a fetch.
    """
    tests = tmp_path / "tests"
    script = tests / "reference_common/artifact.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(_SCRIPT.read_bytes())
    package = tests / f"{operation}_reference"
    (package / "architectures").mkdir(parents=True)
    (package / "architectures/registry.json").write_text('["gfx942", "gfx950"]\n')
    (package / "operation.json").write_text(
        json.dumps({"manifest_operation": manifest_operation(operation)})
    )
    bundle, lock = _fixture(tmp_path, operation)
    newer = json.dumps(dict(json.loads(lock.read_text()), manifest_sha256="0" * 64))
    archives = tests / "reference_bundles" / operation
    # Each published variant has its own tracked archive and committed lock.
    for slot in ("gfx942/family", "gfx942/second", "gfx950/family"):
        archive = archives / f"{slot}.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        pack(bundle, archive, lock, operation=operation)
        archive.with_name(f"{archive.name}.dvc").write_text("outs:\n")
        committed = package / f"architectures/{slot}/baseline_lock.json"
        committed.parent.mkdir(parents=True)
        committed.write_text(newer)
    candidate = tmp_path / "candidate.tar.gz"
    pack(bundle, candidate, lock, operation=operation)
    candidate_lock = tmp_path / "qualification-lock.json"
    candidate_lock.write_text(newer)
    gfx942_lock = package / "architectures/gfx942/family/baseline_lock.json"
    archive, selected = {
        "published": (archives / "gfx942/family.tar.gz", gfx942_lock),
        "another-variant": (archives / "gfx942/second.tar.gz", gfx942_lock),
        "another-architecture": (archives / "gfx950/family.tar.gz", gfx942_lock),
        "candidate-archive": (candidate, candidate_lock),
        "candidate-lock": (archives / "gfx942/family.tar.gz", candidate_lock),
    }[pairing]
    output = tmp_path / "extracted"
    result = _run(
        "unpack",
        "--operation",
        operation,
        "--archive",
        archive,
        "--bundle",
        output,
        "--lock",
        selected,
        script=script,
    )
    assert result.returncode != 0
    assert "GPU reference manifest does not match the pinned lock" in result.stderr
    assert f"{archive} is not the archive that {selected} pins" in result.stderr
    pull = display_command(["dvc", "pull", f"{archive}.dvc"])
    fetch = f"; fetch that archive with: {pull}"
    assert (fetch in result.stderr) == (pairing == "published")
    assert ("dvc pull" in result.stderr) == (pairing == "published")
    assert not output.exists()


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("lock", [None, "committed", "candidate"])
def test_rerun_command_repeats_the_failing_bundle_and_lock(
    tmp_path, monkeypatch, operation, lock
):
    """A failing candidate's rerun must not check it against the committed lock."""
    from reference_common import cli as common_cli

    spec, architecture, case, manifest, output = _verifiable_case(
        monkeypatch, operation
    )
    entry = manifest["cases"][case.id]
    changed = decode(output, case.dtype).copy()
    changed.flat[0] += 100
    changed = encode(changed, case.dtype)

    def worker(request, **kwargs):
        current = request["mode"] == "source"
        return [output, changed if current else output], {
            "device_target": entry["device_target"],
            "launches": 2,
            "torch_imported": False,
        }

    monkeypatch.setattr(common_cli, "run_worker", worker)
    selected = {
        None: None,
        "committed": spec.architectures.baseline_lock(architecture),
        "candidate": tmp_path / "qualification-lock.json",
    }[lock]
    bundle = tmp_path / "bundle"
    with pytest.raises(AssertionError) as error:
        spec.verify_case(
            case,
            bundle=bundle,
            manifest=manifest,
            lock=selected,
            architecture=architecture,
        )
    (rerun,) = re.findall(r"rerun only this case with `([^`]+)`", str(error.value))
    python, script, *arguments = _arguments(rerun)
    # The script is installed beside the tests, so the command runs in both layouts.
    assert python == "python"
    assert Path(script).is_file()
    assert Path(script).resolve() == (
        Path(__file__).with_name(f"run_{operation}_reference.py").resolve()
    )
    args = spec._parser().parse_args(arguments)
    assert (args.command, args.arch, args.cases) == ("verify", architecture, [case.id])
    # The variant is named even as the default, so another variant cannot make
    # the command ambiguous.
    assert args.variant == spec.architectures.resolve_variant(architecture)
    assert args.bundle.resolve() == bundle.resolve()
    # The committed lock is the default, so only a candidate lock is repeated.
    if lock == "candidate":
        assert args.lock.resolve() == selected.resolve()
    else:
        assert args.lock is None


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("candidate", [False, True])
def test_pinned_correctness_verifies_with_the_selected_lock(
    tmp_path, monkeypatch, operation, candidate
):
    from importlib import import_module
    from types import SimpleNamespace

    from reference_common.cli import ReferenceOperation

    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    registry = gpu_tests.OPERATION.architectures
    architecture = registry.ARCHITECTURES[0]
    variant = registry.resolve_variant(architecture)
    target = registry.get_adapter(architecture, variant)
    lock = tmp_path / "qualification-lock.json" if candidate else None
    calls = []

    def verify_case(self, case, **kwargs):
        calls.append((kwargs["lock"], kwargs["architecture"], kwargs["variant"]))
        return {"torch_imported": False, "old_launches": 2, "current_launches": 2}

    def open_slot(*slot):
        assert slot == (architecture, variant)
        return target, tmp_path, {}

    monkeypatch.setattr(ReferenceOperation, "verify_case", verify_case)
    config = SimpleNamespace(getoption={"--rocke-reference-lock": lock}.get)
    test = getattr(gpu_tests, f"test_{operation}_correctness_against_qualified_rocke")
    test(architecture, variant, target.CASES[0], open_slot, config)
    assert calls == [(lock, architecture, variant)]


@pytest.mark.parametrize(
    "report,expected",
    [
        (
            {"torch_imported": True, "old_launches": 2, "current_launches": 2},
            "assert not True",
        ),
        (
            {"torch_imported": False, "old_launches": 2, "current_launches": 1},
            "assert 2 == 1",
        ),
    ],
)
def test_pinned_correctness_failures_show_the_reported_values(
    tmp_path, monkeypatch, report, expected
):
    """A failed shared pinned assert shows the reported values, as in a test module."""
    from importlib import import_module
    from types import SimpleNamespace

    from reference_common.cli import ReferenceOperation

    operation = OPERATIONS[0]
    gpu_tests = import_module(f"test_{operation}_pinned_reference")
    registry = gpu_tests.OPERATION.architectures
    architecture = registry.ARCHITECTURES[0]
    variant = registry.resolve_variant(architecture)
    target = registry.get_adapter(architecture, variant)
    monkeypatch.setattr(
        ReferenceOperation, "verify_case", lambda self, case, **kwargs: report
    )
    config = SimpleNamespace(getoption={"--rocke-reference-lock": None}.get)
    test = getattr(gpu_tests, f"test_{operation}_correctness_against_qualified_rocke")
    with pytest.raises(AssertionError) as error:
        test(
            architecture,
            variant,
            target.CASES[0],
            lambda *slot: (target, tmp_path, {}),
            config,
        )
    assert expected in str(error.value)
