# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Refusal tests for the two FlyDSL drift guards.

``gen_descriptors.py --check`` runs before every pack and ``tools/check_shards.py`` after
it; together they are what stops a code object whose bytes, target or descriptors no
longer agree with the checked-in tree from shipping. Each test here copies a real tree,
breaks one thing, and requires the guard to refuse it with the diagnostic for that thing.
Each guard also runs once over the untouched copy, so a refusal cannot come from the copy
itself.

The shard tests read the product pack this build produced, which CMake names through
``FLYDSL_PRODUCT_OUT_ROOT`` and ``FLYDSL_SHARD_ARCH``; run them through CTest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
PROVIDER = FLYDSL.parent
CONTENT = PROVIDER / "src/engines/kernel_ingestor_engine/descriptors/FlyDSL"
GEN_DESCRIPTORS = FLYDSL / "gen_descriptors.py"
CHECK_SHARDS = FLYDSL / "tools" / "check_shards.py"

# A generic target, so the copy also exercises the ELF generic-machine check.
CONTENT_ARCH = "gfx12-generic"
OP = "rmsnorm"


def _run(*args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *map(str, args)], capture_output=True, text=True, check=False
    )


# ---------------------------------------------------------------------------
# gen_descriptors.py --check
# ---------------------------------------------------------------------------


@pytest.fixture
def content(tmp_path: Path) -> Path:
    """A private copy of the checked-in FlyDSL content tree."""
    copy = tmp_path / "FlyDSL"
    shutil.copytree(CONTENT, copy)
    return copy


def _check(content_dir: Path) -> subprocess.CompletedProcess:
    args = [
        GEN_DESCRIPTORS,
        "--content-dir",
        content_dir,
        "--arch",
        CONTENT_ARCH,
        "--check",
    ]
    table = os.environ.get("HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON")
    if table:
        args += ["--generic-targets-json", table]
    return _run(*args)


def _manifest_path(content_dir: Path) -> Path:
    return content_dir / OP / CONTENT_ARCH / "manifest.json"


def _manifest(content_dir: Path) -> dict:
    return json.loads(_manifest_path(content_dir).read_text(encoding="utf-8"))


def _write_manifest(content_dir: Path, manifest: dict) -> None:
    _manifest_path(content_dir).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _first_object(content_dir: Path) -> Path:
    return (
        content_dir / OP / CONTENT_ARCH / _manifest(content_dir)["instances"][0]["file"]
    )


def _refused(result: subprocess.CompletedProcess, diagnostic: str) -> None:
    assert result.returncode == 1, result.stdout + result.stderr
    assert diagnostic in result.stderr, result.stderr


def test_check_accepts_the_checked_in_tree(content: Path) -> None:
    result = _check(content)
    assert result.returncode == 0, result.stdout + result.stderr


def test_check_refuses_an_object_whose_bytes_changed(content: Path) -> None:
    obj = _first_object(content)
    data = bytearray(obj.read_bytes())
    data[-1] ^= 0xFF
    obj.write_bytes(bytes(data))
    _refused(_check(content), "does not match the")


def test_check_refuses_an_object_built_for_another_target(content: Path) -> None:
    # Bytes from another arch's directory under this one's name, with the manifest
    # updated to match them, so only the target check can catch it.
    manifest = _manifest(content)
    record = manifest["instances"][0]
    foreign = content / OP / "gfx942" / record["file"]
    target = content / OP / CONTENT_ARCH / record["file"]
    shutil.copyfile(foreign, target)
    record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    _write_manifest(content, manifest)
    _refused(_check(content), "is being filed under")


def test_check_refuses_objects_generated_from_other_sources(content: Path) -> None:
    manifest = _manifest(content)
    manifest["toolchain"]["inputs_sha256"] = "0" * 64
    _write_manifest(content, manifest)
    _refused(_check(content), "were generated from kernel and generator sources")


def test_check_refuses_a_missing_object(content: Path) -> None:
    _first_object(content).unlink()
    _refused(_check(content), "does not exist")


def test_check_refuses_an_object_no_manifest_names(content: Path) -> None:
    obj = _first_object(content)
    shutil.copyfile(obj, obj.with_name("rmsnorm_bf16_n1_bt256.hsaco"))
    _refused(_check(content), "the manifest does not")


def test_check_refuses_a_descriptor_that_disagrees_with_its_object(
    content: Path,
) -> None:
    kdp_path = next((content / OP / CONTENT_ARCH).glob("*.kdp.json"))
    kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
    kdp["kernelDescriptors"][0]["priority"] += 1
    kdp_path.write_text(json.dumps(kdp, indent=2), encoding="utf-8")
    _refused(_check(content), "DESCRIPTOR FAIL")


# ---------------------------------------------------------------------------
# tools/check_shards.py
# ---------------------------------------------------------------------------


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is not set; run these tests through CTest")
    return value


@pytest.fixture
def shard(tmp_path: Path) -> tuple[Path, str]:
    """A private copy of one arch's packed FlyDSL descriptors: (out root, arch)."""
    out_root = Path(_required_env("FLYDSL_PRODUCT_OUT_ROOT"))
    arch = _required_env("FLYDSL_SHARD_ARCH")
    copy = tmp_path / "out"
    shutil.copytree(out_root / arch / CONTENT.name, copy / arch / CONTENT.name)
    return copy, arch


def _check_shards(
    content_dir: Path, out_root: Path, arch: str
) -> subprocess.CompletedProcess:
    return _run(
        CHECK_SHARDS,
        "--content-dir",
        content_dir,
        "--out-root",
        out_root,
        "--arches",
        arch,
    )


def _shipped_kdp(out_root: Path, arch: str) -> Path:
    return next((out_root / arch / CONTENT.name / OP).glob("*/*.kdp.json"))


def _shipped_target(out_root: Path, arch: str) -> str:
    """The content directory the shard's objects came from (a generic target or arch)."""
    return _shipped_kdp(out_root, arch).parent.name


def _edit_shipped(out_root: Path, arch: str, edit) -> None:
    path = _shipped_kdp(out_root, arch)
    kdp = json.loads(path.read_text(encoding="utf-8"))
    edit(kdp["kernelDescriptors"])
    path.write_text(json.dumps(kdp), encoding="utf-8")


def _edit_checked_in(content_dir: Path, target: str, edit) -> None:
    path = content_dir / OP / target / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    edit(manifest["instances"])
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _shard_refused(result: subprocess.CompletedProcess, diagnostic: str) -> None:
    assert result.returncode == 1, result.stdout + result.stderr
    assert "SHARD FAIL" in result.stderr and diagnostic in result.stderr, result.stderr


def test_shards_accept_what_the_pack_shipped(
    content: Path, shard: tuple[Path, str]
) -> None:
    result = _check_shards(content, *shard)
    assert result.returncode == 0, result.stdout + result.stderr


# The shard is what a pack produced and the tree is what was checked in; each case below
# makes the two disagree in one way. A packed UKD is bound to its provenance sidecar by
# digest, so a pack that shipped other bytes shows up as a manifest that records others.


def test_shards_refuse_a_payload_the_manifest_does_not_record(
    content: Path, shard: tuple[Path, str]
) -> None:
    def edit(instances: list) -> None:
        instances[0]["sha256"] = "0" * 64

    _edit_checked_in(content, _shipped_target(*shard), edit)
    _shard_refused(_check_shards(content, *shard), "ships payload")


def test_shards_refuse_a_shard_missing_an_object(
    content: Path, shard: tuple[Path, str]
) -> None:
    _edit_shipped(*shard, lambda documents: documents.pop(0))
    _shard_refused(_check_shards(content, *shard), "is not in the shard")


def test_shards_refuse_an_object_no_manifest_names(
    content: Path, shard: tuple[Path, str]
) -> None:
    _edit_checked_in(
        content, _shipped_target(*shard), lambda instances: instances.pop(0)
    )
    _shard_refused(_check_shards(content, *shard), "ships but no manifest names it")


def test_shards_refuse_a_descriptor_edited_after_the_pack(
    content: Path, shard: tuple[Path, str]
) -> None:
    # The sidecar binding: a shipped UKD changed after packing no longer matches the
    # provenance the pack recorded for it, whatever field changed.
    def edit(documents: list) -> None:
        documents[0]["arch"] = ["gfx000"]

    _edit_shipped(*shard, edit)
    result = _check_shards(content, *shard)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "come from different packs" in result.stderr, result.stderr
