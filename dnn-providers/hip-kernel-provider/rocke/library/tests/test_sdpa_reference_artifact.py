# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise DVC archive staging before any GPU or NumPy runtime is needed."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest

from sdpa_reference.artifact import pack, unpack, validate_bundle


def _bundle(tmp_path):
    bundle = tmp_path / "qualified"
    (bundle / "payload").mkdir(parents=True)
    (bundle / "payload/kernel.hsaco").write_bytes(b"fixture")
    manifest = {
        "schema": 1,
        "baseline_revision": "a" * 40,
        "files": {"kernel.hsaco": hashlib.sha256(b"fixture").hexdigest()},
    }
    encoded = json.dumps(manifest).encode()
    (bundle / "manifest.json").write_bytes(encoded)
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema": 1,
                "baseline_revision": "a" * 40,
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    )
    return bundle, lock


def test_archive_is_reproducible_and_has_no_host_metadata(tmp_path):
    bundle, lock = _bundle(tmp_path)
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    pack(bundle, first, lock)
    pack(bundle, second, lock)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as tar:
        for member in tar:
            assert member.uid == member.gid == member.mtime == 0
            assert member.uname == member.gname == ""
            assert member.name.startswith("sdpa_reference_bundle/")
    output = tmp_path / "staged"
    unpack(first, output, lock)
    unpack(first, output, lock)
    validate_bundle(output, lock)
    assert (output / "payload/kernel.hsaco").read_bytes() == b"fixture"


@pytest.mark.parametrize("change", ["payload", "manifest", "extra"])
def test_corrupt_bundle_cannot_be_packed(tmp_path, change):
    bundle, lock = _bundle(tmp_path)
    path = {
        "payload": "payload/kernel.hsaco",
        "manifest": "manifest.json",
        "extra": "payload/extra",
    }[change]
    (bundle / path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload|manifest"):
        pack(bundle, tmp_path / "bad.tar.gz", lock)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("sdpa_reference_bundle/../../escaped", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("sdpa_reference_bundle/link", tarfile.SYMTYPE),
        ("sdpa_reference_bundle/hardlink", tarfile.LNKTYPE),
    ],
)
def test_unsafe_archive_is_rejected_without_exposing_output(tmp_path, name, kind):
    _, lock = _bundle(tmp_path)
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "../../escaped" if kind != tarfile.REGTYPE else ""
        tar.addfile(member, io.BytesIO())
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="invalid SDPA archive member"):
        unpack(archive, output, lock)
    assert not output.exists()
    assert not (tmp_path / "escaped").exists()


def test_unpacked_payload_is_checked_against_the_lock(tmp_path):
    bundle, lock = _bundle(tmp_path)
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle / "manifest.json", arcname="sdpa_reference_bundle/manifest.json")
        data = b"corrupt"
        member = tarfile.TarInfo("sdpa_reference_bundle/payload/kernel.hsaco")
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="payload"):
        unpack(archive, tmp_path / "output", lock)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("suffix", [".npz", ".npy"])
@pytest.mark.parametrize("location", ["payload", "."])
def test_generated_bundle_cannot_reintroduce_tensor_files(tmp_path, suffix, location):
    bundle, lock = _bundle(tmp_path)
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
        pack(bundle, tmp_path / "bad.tar.gz", lock)


@pytest.mark.parametrize("layout", ["bin/hip_kernel_provider", "standalone"])
def test_installed_bundle_lookup_is_relocatable_and_arch_specific(tmp_path, layout):
    from sdpa_reference.paths import default_bundle_path

    root = tmp_path / layout
    tests = root / "tests/library/tests"
    # A stale generic bundle must not hide missing per-arch test content.
    (tests / "reference_bundles/gfx942").mkdir(parents=True)
    expected = root / "engines/test_arch_content/rocke/sdpa"
    assert default_bundle_path(tests, "gfx942") == expected / "gfx942"
    assert default_bundle_path(tests, "gfx950") == expected / "gfx950"


def test_source_bundle_lookup_preserves_local_layout(tmp_path):
    from sdpa_reference.paths import default_bundle_path

    tests = tmp_path / "rocke/library/tests"
    assert (
        default_bundle_path(tests, "gfx942") == tests / "reference_bundles/sdpa/gfx942"
    )
