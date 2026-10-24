# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Refusal tests for the two FlyDSL drift guards.

``gen_descriptors.py --check`` runs before every pack and ``tools/check_shards.py`` after
it; together they are what stops a code object whose bytes, target or descriptors no
longer agree with the checked-in tree from shipping. Each test here copies a real tree,
breaks one thing, and requires the guard to refuse it with the diagnostic for that thing.
Each guard also runs once over the untouched copy, so a refusal cannot come from the copy
itself. Where an earlier check would also refuse the break, the test repairs what that
check reads (a manifest SHA256, a sidecar binding), so only the check under test decides.

The tests pick their op and target from whatever the tree holds rather than naming one:
the content tree is ``FLYDSL_CONTENT_DIR`` if set, else the in-source tree. The shard
tests read the product pack this build produced, which CMake names through
``FLYDSL_PRODUCT_OUT_ROOT`` and ``FLYDSL_SHARD_ARCH``; run them through CTest.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
PROVIDER = FLYDSL.parent
CONTENT = Path(
    os.environ.get("FLYDSL_CONTENT_DIR")
    or PROVIDER / "src/engines/kernel_ingestor_engine/descriptors/FlyDSL"
)
GEN_DESCRIPTORS = FLYDSL / "gen_descriptors.py"
CHECK_SHARDS = FLYDSL / "tools" / "check_shards.py"

# The packer's reader and writer for the provenance a packed KDP keeps in its sidecar.
sys.path.insert(0, str(PROVIDER / "descriptor-packaging" / "python"))
from hkp_pack import provenance_sidecar  # noqa: E402

MANIFEST_NAME = "manifest.json"
REFERENCE_CSV = Path(__file__).resolve().parent / "data" / "sdpa_bwd_reference.csv"
# ELF64 header offset of e_flags, and its EF_AMDGPU_MACH field.
E_FLAGS_OFFSET = 48
EF_AMDGPU_MACH = 0x0FF


def _run(*args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *map(str, args)], capture_output=True, text=True, check=False
    )


def _first_record(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))["instances"][0]


@functools.cache
def _subject() -> tuple[str, str, str | None]:
    """(op, content target, foreign target) the --check cases work on.

    A generic target when the tree has one, so the copy reaches the ELF generic-machine
    check; the foreign target is another of the op's directories holding an object of
    the same name, built for a different target (None if the op has none).
    """
    manifests = sorted(CONTENT.glob(f"*/gfx*/{MANIFEST_NAME}"))
    if not manifests:
        pytest.fail(f"no */gfx*/{MANIFEST_NAME} under {CONTENT}; pull the content")
    generic = [path for path in manifests if path.parent.name.endswith("-generic")]
    candidates = generic or manifests
    for path in candidates:
        op, target = path.parent.parent.name, path.parent.name
        name = _first_record(path)["file"]
        for other in manifests:
            if other.parent.parent.name == op and other.parent.name != target:
                if (other.parent / name).is_file():
                    return op, target, other.parent.name
    path = candidates[0]
    return path.parent.parent.name, path.parent.name, None


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
        _subject()[1],
        "--check",
    ]
    table = os.environ.get("HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON")
    if table:
        args += ["--generic-targets-json", table]
    return _run(*args)


def _folder(content_dir: Path) -> Path:
    op, target, _ = _subject()
    return content_dir / op / target


def _manifest_path(content_dir: Path) -> Path:
    return _folder(content_dir) / MANIFEST_NAME


def _manifest(content_dir: Path) -> dict:
    return json.loads(_manifest_path(content_dir).read_text(encoding="utf-8"))


def _write_manifest(content_dir: Path, manifest: dict) -> None:
    _manifest_path(content_dir).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _first_object(content_dir: Path) -> Path:
    return _folder(content_dir) / _manifest(content_dir)["instances"][0]["file"]


def _replace_first_object(content_dir: Path, data: bytes) -> None:
    """Swap in other bytes for the first object, re-recording its SHA256 to match."""
    manifest = _manifest(content_dir)
    record = manifest["instances"][0]
    (_folder(content_dir) / record["file"]).write_bytes(data)
    record["sha256"] = hashlib.sha256(data).hexdigest()
    _write_manifest(content_dir, manifest)


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
    # Bytes from another target's directory under this one's name, with the manifest
    # updated to match them, so only the target check can catch it.
    op, _, foreign = _subject()
    if foreign is None:
        pytest.skip(f"no other target of {op} holds an object of the same name")
    record = _manifest(content)["instances"][0]
    _replace_first_object(
        content, (content / op / foreign / record["file"]).read_bytes()
    )
    _refused(_check(content), "is being filed under")


def test_check_refuses_a_generic_object_whose_elf_header_is_concrete(
    content: Path,
) -> None:
    # amdhsa.target still names the generic target, so the target check passes; only
    # the ELF e_flags check sees the header that the loader decides compatibility from.
    target = _subject()[1]
    if not target.endswith("-generic"):
        pytest.skip("the tree has no generic target, so there is no e_flags check")
    data = bytearray(_first_object(content).read_bytes())
    assert data[:4] == b"\x7fELF" and data[4] == 2, "expected a bare ELF64 object"
    flags = int.from_bytes(data[E_FLAGS_OFFSET : E_FLAGS_OFFSET + 4], "little")
    flags &= ~EF_AMDGPU_MACH  # no machine at all; the generic version is kept
    data[E_FLAGS_OFFSET : E_FLAGS_OFFSET + 4] = flags.to_bytes(4, "little")
    _replace_first_object(content, bytes(data))
    _refused(_check(content), "ELF e_flags carry machine")


def test_check_refuses_a_manifest_filed_under_another_target(content: Path) -> None:
    manifest = _manifest(content)
    manifest["arch"] = "gfx000"
    _write_manifest(content, manifest)
    _refused(_check(content), "manifest records arch 'gfx000'")


def test_check_refuses_a_manifest_filed_under_another_op(content: Path) -> None:
    manifest = _manifest(content)
    manifest["op"] = "not_" + manifest["op"]
    _write_manifest(content, manifest)
    _refused(_check(content), "manifest records op")


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
    shutil.copyfile(obj, obj.with_name("stray_" + obj.name))
    _refused(_check(content), "the manifest does not")


def test_check_refuses_a_descriptor_that_disagrees_with_its_object(
    content: Path,
) -> None:
    kdp_path = next(_folder(content).glob("*.kdp.json"))
    kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
    kdp["kernelDescriptors"][0]["priority"] += 1
    kdp_path.write_text(json.dumps(kdp, indent=2), encoding="utf-8")
    _refused(_check(content), "DESCRIPTOR FAIL")


def test_check_refuses_a_descriptor_no_manifest_derives(content: Path) -> None:
    kdp_path = next(_folder(content).glob("*.kdp.json"))
    shutil.copyfile(kdp_path, kdp_path.with_name("stale_" + kdp_path.name))
    _refused(_check(content), "names no object in the manifest")


def test_check_refuses_a_descriptor_the_authoring_validator_rejects(
    content: Path,
) -> None:
    # A type-tagged descriptor outside every folder the objects derive, so the
    # comparison above does not see it and only the packer's validator, which walks
    # the whole content tree, can refuse it.
    (content / "orphan.uhd.json").write_text("{}\n", encoding="utf-8")
    _refused(_check(content), "descriptor validation failed")


def test_merged_pins_are_the_reference_pins() -> None:
    # A re-pinned sha256 under an unchanged Spec does not change the op's inputs digest,
    # so --check cannot stale the shipped descriptors; this compares them with the reference.
    manifests = sorted(CONTENT.glob(f"sdpa_bwd_merged/gfx*/{MANIFEST_NAME}"))
    if not manifests:
        pytest.skip(f"no sdpa_bwd_merged content under {CONTENT}")
    with REFERENCE_CSV.open(newline="", encoding="utf-8") as handle:
        reference = {
            row["spec"]: row["sha256"]
            for row in csv.DictReader(handle)
            if row["spec"].startswith("merged,")
        }
    for path in manifests:
        shipped = {
            record["spec"]: record["pin_sha256"]
            for record in json.loads(path.read_text(encoding="utf-8"))["instances"]
        }
        stale = [
            f"{spec}: {path.parent.name} ships pin {sha}, reference {reference.get(spec)}"
            for spec, sha in shipped.items()
            if reference.get(spec) != sha
        ]
        missing = sorted(set(reference) - set(shipped))
        assert not stale, "pins differ from the reference:\n" + "\n".join(stale)
        assert not missing, f"{path.parent.name} ships no pin for: {missing}"


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
    content_dir: Path, out_root: Path, *arches: str
) -> subprocess.CompletedProcess:
    return _run(
        CHECK_SHARDS,
        "--content-dir",
        content_dir,
        "--out-root",
        out_root,
        "--arches",
        *arches,
    )


def _shipped_kdp(out_root: Path, arch: str) -> Path:
    """The first packed KDP in the shard, whichever op and content target it is."""
    found = sorted((out_root / arch / CONTENT.name).glob("*/*/*.kdp.json"))
    if not found:
        pytest.fail(f"the {arch} shard under {out_root} holds no FlyDSL KDP")
    return found[0]


def _edit_shipped(out_root: Path, arch: str, edit) -> None:
    path = _shipped_kdp(out_root, arch)
    kdp = json.loads(path.read_text(encoding="utf-8"))
    edit(kdp["kernelDescriptors"])
    path.write_text(json.dumps(kdp), encoding="utf-8")


def _repack_shipped(
    out_root: Path,
    arch: str,
    edit_document: Callable[[dict], None] = lambda document: None,
    edit_provenance: Callable[[dict, list], None] = lambda provenance, others: None,
) -> None:
    """Edit the first shipped UKD and/or its provenance as a pack would have written
    them, re-binding the sidecar so the binding check cannot be what refuses."""
    path = _shipped_kdp(out_root, arch)
    side = provenance_sidecar.sidecar_path(path)
    kdp = json.loads(path.read_text(encoding="utf-8"))
    sidecar = provenance_sidecar.load_sidecar(side)
    documents = kdp["kernelDescriptors"]
    entry = sidecar["entries"][documents[0]["id"]]
    edit_document(documents[0])
    others = [sidecar["entries"][d["id"]]["provenance"] for d in documents[1:]]
    edit_provenance(entry["provenance"], others)
    entry["ukd_sha256"] = provenance_sidecar.ukd_sha256(documents[0])
    path.write_text(json.dumps(kdp), encoding="utf-8")
    side.write_bytes(provenance_sidecar.encode(sidecar))


def _edit_checked_in(content_dir: Path, out_root: Path, arch: str, edit) -> None:
    """Edit the checked-in manifest of the object set the first shipped KDP came from."""
    kdp = _shipped_kdp(out_root, arch)
    path = content_dir / kdp.parent.parent.name / kdp.parent.name / MANIFEST_NAME
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


def test_shards_accept_a_rebound_but_unchanged_pack(
    content: Path, shard: tuple[Path, str]
) -> None:
    # The re-binding the cases below use leaves an untouched shard acceptable.
    _repack_shipped(*shard)
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

    _edit_checked_in(content, *shard, edit)
    _shard_refused(_check_shards(content, *shard), "ships payload")


def test_shards_refuse_a_shard_missing_an_object(
    content: Path, shard: tuple[Path, str]
) -> None:
    _edit_shipped(*shard, lambda documents: documents.pop(0))
    _shard_refused(_check_shards(content, *shard), "is not in the shard")


def test_shards_refuse_an_object_no_manifest_names(
    content: Path, shard: tuple[Path, str]
) -> None:
    _edit_checked_in(content, *shard, lambda instances: instances.pop(0))
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


def test_shards_refuse_a_descriptor_listing_another_arch(
    content: Path, shard: tuple[Path, str]
) -> None:
    # A packer that stamped a member's shard with more than that member, re-bound as
    # the pack would have, so only the per-UKD arch check can refuse it.
    def edit(document: dict) -> None:
        document["arch"] = [*document["arch"], "gfx000"]

    _repack_shipped(*shard, edit_document=edit)
    _shard_refused(_check_shards(content, *shard), "lists arch")


def test_shards_refuse_a_descriptor_not_packed_from_an_hsaco(
    content: Path, shard: tuple[Path, str]
) -> None:
    def edit(provenance: dict, others: list) -> None:
        provenance["origin_kind"] = "embedded_source"

    _repack_shipped(*shard, edit_provenance=edit)
    _shard_refused(_check_shards(content, *shard), "was not packed from an hsaco")


def test_shards_refuse_a_descriptor_packed_from_another_object(
    content: Path, shard: tuple[Path, str]
) -> None:
    def edit(provenance: dict, others: list) -> None:
        provenance["file"] = others[0]["file"] if others else "FlyDSL/other.hsaco"

    _repack_shipped(*shard, edit_provenance=edit)
    _shard_refused(_check_shards(content, *shard), "was packed from")


def test_shards_refuse_an_arch_no_object_set_runs_on(
    content: Path, shard: tuple[Path, str]
) -> None:
    out_root, arch = shard
    _shard_refused(
        _check_shards(content, out_root, arch, "gfx000"),
        "gfx000: no checked-in FlyDSL object set runs on it",
    )
