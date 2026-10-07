# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The provenance sidecar's own contract: what detach writes, and every way
attach refuses a descriptor and sidecar that do not belong together."""

import copy
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hkp_pack import descriptor_context, pipeline, provenance_sidecar
from hkp_pack.errors import HkpPackError
from pack_helpers import is_compact, write_shipped

SHA = "a" * 64
OTHER_SHA = "b" * 64

_DESK_CHECK = Path(__file__).resolve().parents[1] / "tools" / "hkp_desk_check.py"


def _ukd(ident, sha=SHA, kind="kpack"):
    return {
        "version": "1.0",
        "id": ident,
        "name": ident,
        "kernel_source": {"kind": kind, "sha256": sha},
        "provenance": {"origin_kind": "hip", "source_label": ident},
    }


def _embedded_ukd(ident):
    """A UKD of a kind with no sha256 of its own: only `ukd_sha256` binds it."""
    return {
        "version": "1.0",
        "id": ident,
        "name": ident,
        "kernel_source": {
            "kind": "embedded_source",
            "source_file": "kernels/PointwiseAdd.cpp",
            "entry_point": "PointwiseAdd",
        },
        "metadata": {"block_size": 64},
        "priority": 0,
        "provenance": {
            "origin_kind": "embedded_source",
            "source_label": "unit",
            "spec": {"block_size": 64},
        },
    }


def _kdp(*ukds):
    return {
        "version": "1.0",
        "id": "pack-id",
        "name": "pack",
        "kernelDescriptors": list(ukds or (_ukd("k0"), "a-standalone-id", _ukd("k1"))),
    }


def _sidecar(path):
    return json.loads(
        gzip.decompress(provenance_sidecar.sidecar_path(path).read_bytes())
    )


def _rewrite_sidecar(path, mutate):
    data = _sidecar(path)
    mutate(data)
    provenance_sidecar.sidecar_path(path).write_bytes(provenance_sidecar.encode(data))


def _attach_fresh(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    return provenance_sidecar.attach(path, doc)


# --- What detach writes ------------------------------------------------------


@pytest.mark.quick
def test_attach_restores_what_detach_moved(tmp_path):
    original = _kdp()
    path = tmp_path / "solo.kdp.json"
    shipped = write_shipped(path, original)

    inline = [e for e in shipped["kernelDescriptors"] if isinstance(e, dict)]
    assert all("provenance" not in ukd for ukd in inline)
    assert provenance_sidecar.attach(path, shipped) == original


@pytest.mark.quick
def test_the_sidecar_is_the_version_1_document(tmp_path):
    """The format, spelled here independently of the module."""
    ukd = _embedded_ukd("k0")
    ukd["name"] = "Pointwise \u00e9"
    path = tmp_path / "solo.kdp.json"
    write_shipped(path, _kdp(ukd))

    body = {k: v for k, v in ukd.items() if k != "provenance"}
    digest = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    assert _sidecar(path) == {
        "version": "1.0",
        "kdp_id": "pack-id",
        "entries": {"k0": {"ukd_sha256": digest, "provenance": ukd["provenance"]}},
    }


@pytest.mark.quick
def test_a_ukd_with_no_provenance_detaches_to_an_empty_record(tmp_path):
    ukd = _embedded_ukd("k0")
    del ukd["provenance"]
    path = tmp_path / "solo.ukd.json"
    write_shipped(path, ukd)

    assert _attach_fresh(path)["provenance"] == {}


@pytest.mark.quick
def test_detach_refuses_a_ukd_id_named_twice(tmp_path):
    doc = _kdp()
    doc["kernelDescriptors"].append(_ukd("k0", sha=OTHER_SHA))

    with pytest.raises(HkpPackError, match="appears twice"):
        provenance_sidecar.detach("solo.kdp.json", doc)


@pytest.mark.quick
def test_encode_is_reproducible():
    data = provenance_sidecar.encode({"b": 1, "a": 2})

    assert data == provenance_sidecar.encode({"a": 2, "b": 1})
    assert data[4:8] == b"\0\0\0\0"  # MTIME


# --- What the packer writes --------------------------------------------------


@pytest.mark.quick
def test_the_written_descriptor_binds_to_its_sidecar_as_parsed(tmp_path):
    """detach digests the dict the packer holds; the reader digests the dict it
    parses back. Non-ASCII text, floats and tuples must digest alike on both."""
    ukd = _embedded_ukd("k0")
    ukd["name"] = "Pointwise \u00e9 \u4e2d"
    ukd["metadata"] = {"scale": 0.1, "shape": (1, 2)}
    doc = _kdp(ukd)
    pipeline._write_packed_at(tmp_path, ".", "solo.kdp.json", doc)

    attached = _attach_fresh(tmp_path / "solo.kdp.json")

    assert (
        attached["kernelDescriptors"][0]["provenance"]
        == _embedded_ukd("k0")["provenance"]
    )


@pytest.mark.quick
def test_the_packed_kdp_is_compact_with_kernels_last(tmp_path):
    """The runtime loader reads a KDP in one pass only while `kernelDescriptors` is
    its last key; key-sorting would move `matchers`, `name` and `version` after it."""
    doc = {
        "version": "1.0",
        "id": "pack-id",
        "name": "pack",
        "arch": ["gfx942"],
        "matchers": [],
        "kernelDescriptors": [_ukd("k0")],
    }
    pipeline._write_packed_at(tmp_path, ".", "solo.kdp.json", doc)

    text = (tmp_path / "solo.kdp.json").read_text(encoding="utf-8")
    written = json.loads(text)
    assert list(written)[-1] == "kernelDescriptors"
    assert list(written) == list(doc)
    assert is_compact(text)


@pytest.mark.quick
def test_the_packer_marks_every_directory_it_writes_a_descriptor_into(tmp_path):
    pipeline._write_packed_at(tmp_path, ".", "solo.kdp.json", _kdp(_ukd("k0")))
    pipeline._write_packed_at(tmp_path, "nested/set", "solo.ukd.json", _ukd("k1"))
    pipeline._write_bytes_at(tmp_path, "generic", "engine.engine.json", b"{}")

    assert provenance_sidecar.is_packed(tmp_path)
    assert provenance_sidecar.is_packed(tmp_path / "nested" / "set")
    assert (tmp_path / "nested" / "set" / "hkp-packed.marker").read_bytes() == b""
    assert not provenance_sidecar.is_packed(tmp_path / "nested")
    assert not provenance_sidecar.is_packed(tmp_path / "generic")


# --- What attach refuses -----------------------------------------------------


@pytest.mark.quick
def test_a_packed_ukd_without_its_sidecar_is_refused(tmp_path):
    path = tmp_path / "solo.kdp.json"
    shipped = write_shipped(path, _kdp())
    provenance_sidecar.sidecar_path(path).unlink()

    with pytest.raises(
        HkpPackError,
        match=r"has no provenance sidecar; expected .*solo\.kdp\.provenance\.json\.gz\."
        r" For an installed tree, pass --provenance-root",
    ):
        provenance_sidecar.attach(path, shipped)


@pytest.mark.quick
def test_a_missing_sidecar_fails_an_embedded_source_ukd_only_when_marked(tmp_path):
    """No field of an embedded_source UKD says the packer wrote it; the marker
    beside it does."""
    path = tmp_path / "solo.ukd.json"
    shipped = write_shipped(path, _embedded_ukd("k0"))
    provenance_sidecar.sidecar_path(path).unlink()

    with pytest.raises(HkpPackError, match="has no provenance sidecar"):
        provenance_sidecar.attach(path, copy.deepcopy(shipped))
    (tmp_path / provenance_sidecar.PACKED_MARKER).unlink()
    assert provenance_sidecar.attach(path, copy.deepcopy(shipped)) == shipped


@pytest.mark.quick
def test_a_ukd_the_sidecar_has_no_entry_for_is_refused(tmp_path):
    path = tmp_path / "solo.kdp.json"
    shipped = write_shipped(path, _kdp())
    _rewrite_sidecar(path, lambda data: data["entries"].pop("k1"))

    with pytest.raises(HkpPackError, match="'k1' has no entry in the sidecar"):
        provenance_sidecar.attach(path, shipped)


@pytest.mark.quick
def test_an_entry_bound_to_another_kernel_source_is_refused(tmp_path):
    path = tmp_path / "solo.kdp.json"
    shipped = write_shipped(path, _kdp())
    shipped["kernelDescriptors"][0]["kernel_source"]["sha256"] = OTHER_SHA

    with pytest.raises(HkpPackError, match="come from different packs"):
        provenance_sidecar.attach(path, shipped)


@pytest.mark.quick
@pytest.mark.parametrize(
    "edit",
    [
        lambda ukd: ukd["kernel_source"].update({"entry_point": "PointwiseSub"}),
        lambda ukd: ukd["metadata"].update({"block_size": 128}),
        lambda ukd: ukd.update({"priority": 1}),
    ],
    ids=["entry_point", "metadata", "priority"],
)
def test_an_embedded_source_ukd_is_bound_by_its_whole_body(tmp_path, edit):
    path = tmp_path / "solo.ukd.json"
    shipped = write_shipped(path, _embedded_ukd("k0"))
    edit(shipped)

    with pytest.raises(HkpPackError, match="come from different packs"):
        provenance_sidecar.attach(path, shipped)


@pytest.mark.quick
def test_a_sidecar_naming_another_kdp_is_refused(tmp_path):
    path = tmp_path / "solo.kdp.json"
    shipped = write_shipped(path, _kdp())
    _rewrite_sidecar(path, lambda data: data.update({"kdp_id": "another-pack"}))

    with pytest.raises(HkpPackError, match="names KDP 'another-pack'"):
        provenance_sidecar.attach(path, shipped)


@pytest.mark.quick
def test_inline_provenance_beside_a_sidecar_is_refused_not_merged(tmp_path):
    path = tmp_path / "solo.kdp.json"
    shipped = write_shipped(path, _kdp())
    shipped["kernelDescriptors"][0]["provenance"] = {"origin_kind": "hip"}

    with pytest.raises(HkpPackError, match="carries inline provenance"):
        provenance_sidecar.attach(path, shipped)


def _version(version):
    return lambda data: data.update({"version": version})


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data.update({"entries": []}), "has no 'entries' object"),
        (lambda data: data["entries"].update({"k0": ["k0"]}), "is not an object"),
        (
            lambda data: data["entries"]["k0"].update({"provenance": "hip"}),
            "has no 'provenance' object",
        ),
        (
            lambda data: data["entries"]["k0"].update({"provenance": None}),
            "has no 'provenance' object",
        ),
        (lambda data: data["entries"]["k0"].pop("ukd_sha256"), "no 'ukd_sha256'"),
        (lambda data: data.update({"kdp_id": 7}), "non-string 'kdp_id'"),
        (lambda data: data.pop("version"), "carries no 'version'"),
        (_version(1), "carries no 'version'"),
        (_version("2.0"), "reads major version 1 only, at minor 0 or earlier"),
        (_version("0.9"), "reads major version 1 only, at minor 0 or earlier"),
        (_version("1.1"), "reads major version 1 only, at minor 0 or earlier"),
        (_version("1"), "not major.minor"),
        (_version("v1.0"), "not major.minor"),
    ],
    ids=[
        "entries-array",
        "entry-array",
        "provenance-string",
        "provenance-null",
        "no-digest",
        "kdp-id-int",
        "no-version",
        "version-int",
        "newer-major",
        "older-major",
        "newer-minor",
        "no-minor",
        "version-prefix",
    ],
)
def test_a_malformed_sidecar_is_refused(tmp_path, mutate, message):
    path = tmp_path / "solo.ukd.json"
    write_shipped(path, _ukd("k0"))
    _rewrite_sidecar(path, mutate)

    with pytest.raises(HkpPackError, match=message):
        _attach_fresh(path)


# --- Corrupt and oversized input --------------------------------------------


@pytest.mark.quick
def test_an_unreadable_sidecar_is_refused(tmp_path):
    path = tmp_path / "solo.ukd.json"
    write_shipped(path, _ukd("k0"))
    side = provenance_sidecar.sidecar_path(path)

    side.write_bytes(b"not gzip")
    with pytest.raises(HkpPackError, match="cannot read provenance sidecar"):
        _attach_fresh(path)

    side.write_bytes(gzip.compress(b"[" * 100000 + b"]" * 100000))
    with pytest.raises(HkpPackError, match="cannot read provenance sidecar"):
        _attach_fresh(path)


@pytest.mark.quick
@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda intact: intact[:-8], "is truncated"),
        (lambda intact: intact + b"\0", "has bytes after its gzip body"),
    ],
    ids=["no-trailer", "trailing-byte"],
)
def test_a_damaged_gzip_frame_is_refused(tmp_path, damage, message):
    # The deflate body is intact in both rows, so the JSON parses: only the frame
    # check can refuse it.
    path = tmp_path / "solo.ukd.json"
    write_shipped(path, _ukd("k0"))
    side = provenance_sidecar.sidecar_path(path)
    side.write_bytes(damage(side.read_bytes()))

    with pytest.raises(HkpPackError, match=message):
        _attach_fresh(path)


@pytest.mark.quick
def test_every_corrupted_byte_is_refused_or_harmless(tmp_path):
    """Never another exception: a harmless flip, in the mtime or OS byte, loads
    identically to the intact sidecar."""
    path = tmp_path / "solo.kdp.json"
    write_shipped(path, _kdp(_ukd("k0"), _embedded_ukd("k1")))
    side = provenance_sidecar.sidecar_path(path)
    intact = side.read_bytes()
    expected = _attach_fresh(path)

    refused = 0
    for index in range(len(intact)):
        for mask in (0x01, 0x80, 0xFF):
            corrupt = bytearray(intact)
            corrupt[index] ^= mask
            side.write_bytes(bytes(corrupt))
            try:
                attached = _attach_fresh(path)
            except HkpPackError:
                refused += 1
                continue
            assert attached == expected, (index, mask)
    # The deflate body and trailer dominate a sidecar, so most flips are refused.
    assert refused > len(intact)


@pytest.mark.quick
def test_decompression_is_capped(tmp_path, monkeypatch):
    path = tmp_path / "solo.ukd.json"
    write_shipped(path, _ukd("k0"))
    size = len(gzip.decompress(provenance_sidecar.sidecar_path(path).read_bytes()))

    monkeypatch.setattr(provenance_sidecar, "MAX_DECOMPRESSED_BYTES", size)
    assert _attach_fresh(path)["provenance"] == _ukd("k0")["provenance"]

    monkeypatch.setattr(provenance_sidecar, "MAX_DECOMPRESSED_BYTES", size - 1)
    with pytest.raises(HkpPackError, match="inflates past"):
        _attach_fresh(path)


# --- Malformed descriptors ---------------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize("source", ["kpack", ["kpack"], None, 7])
def test_a_non_object_kernel_source_is_no_attribute_error(tmp_path, source):
    ukd = _ukd("k0")
    ukd["kernel_source"] = source
    path = tmp_path / "solo.ukd.json"
    path.write_text(json.dumps(ukd), encoding="utf-8")

    assert provenance_sidecar.attach(path, copy.deepcopy(ukd)) == ukd
    (tmp_path / provenance_sidecar.PACKED_MARKER).write_bytes(b"")
    with pytest.raises(HkpPackError, match="has no provenance sidecar"):
        provenance_sidecar.attach(path, copy.deepcopy(ukd))


@pytest.mark.quick
@pytest.mark.parametrize("entries", [{"k0": {}}, "k0"])
def test_non_array_kernel_descriptors_is_a_named_error(tmp_path, entries):
    path = tmp_path / "solo.kdp.json"
    doc = {"id": "pack-id", "kernelDescriptors": entries}

    with pytest.raises(HkpPackError, match="'kernelDescriptors' is"):
        provenance_sidecar.attach(path, doc)
    with pytest.raises(HkpPackError, match="'kernelDescriptors' is"):
        provenance_sidecar.detach(path.name, doc)


@pytest.mark.quick
@pytest.mark.parametrize("ident", [["k0"], 7])
def test_a_non_string_ukd_id_is_a_named_error(tmp_path, ident):
    path = tmp_path / "solo.ukd.json"
    write_shipped(path, _ukd("k0"))
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["id"] = ident

    with pytest.raises(HkpPackError, match="has no entry in the sidecar"):
        provenance_sidecar.attach(path, doc)
    with pytest.raises(HkpPackError, match="is not a string"):
        provenance_sidecar.detach(path.name, _ukd(ident))


# --- Provenance root ---------------------------------------------------------


@pytest.mark.quick
def test_a_provenance_root_mirrors_the_descriptor_root(tmp_path):
    """The provenance tree holds sidecars only, as the test package installs it:
    the marker stays beside the descriptors."""
    descriptors = tmp_path / "arch_content" / "hip-kernel-provider"
    provenance = tmp_path / "test_arch_content" / "hip-kernel-provider" / "provenance"
    path = descriptors / "gfx950" / "attention" / "solo.kdp.json"
    path.parent.mkdir(parents=True)
    write_shipped(path, _kdp())
    expected = _attach_fresh(path)
    side = provenance_sidecar.sidecar_path(path)
    moved = provenance / "gfx950" / "attention" / side.name
    moved.parent.mkdir(parents=True)
    side.rename(moved)
    doc = json.loads(path.read_text(encoding="utf-8"))

    assert provenance_sidecar.sidecar_path(path, provenance, descriptors) == moved
    assert not provenance_sidecar.is_packed(moved.parent)
    assert (
        provenance_sidecar.attach(
            path,
            copy.deepcopy(doc),
            provenance_root=provenance,
            descriptor_root=descriptors,
        )
        == expected
    )
    with pytest.raises(HkpPackError, match="is not under the descriptor root"):
        provenance_sidecar.sidecar_path(path, provenance, tmp_path / "elsewhere")


@pytest.mark.quick
def test_a_refusal_under_a_provenance_root_drops_the_hint_to_name_one(tmp_path):
    path = tmp_path / "descriptors" / "solo.ukd.json"
    path.parent.mkdir()
    write_shipped(path, _embedded_ukd("k0"))
    provenance_sidecar.sidecar_path(path).unlink()

    with pytest.raises(HkpPackError, match="has no provenance sidecar") as refused:
        provenance_sidecar.attach(
            path,
            json.loads(path.read_text(encoding="utf-8")),
            provenance_root=tmp_path / "provenance",
            descriptor_root=tmp_path / "descriptors",
        )
    assert "--provenance-root" not in str(refused.value)


@pytest.mark.quick
def test_a_provenance_root_does_not_make_an_unmarked_descriptor_packed(tmp_path):
    """Its sidecar waits under the root, so only the missing marker refuses it."""
    descriptors = tmp_path / "descriptors"
    provenance = tmp_path / "provenance"
    path = descriptors / "solo.ukd.json"
    path.parent.mkdir()
    write_shipped(path, _embedded_ukd("k0"))
    provenance.mkdir()
    side = provenance_sidecar.sidecar_path(path)
    side.rename(provenance / side.name)
    (descriptors / provenance_sidecar.PACKED_MARKER).unlink()

    with pytest.raises(
        HkpPackError,
        match=r"usage error: --provenance-root .* holds no hkp-packed\.marker",
    ) as refused:
        provenance_sidecar.attach(
            path,
            json.loads(path.read_text(encoding="utf-8")),
            provenance_root=provenance,
            descriptor_root=descriptors,
        )
    assert str(descriptors) in str(refused.value)


# --- The marker --------------------------------------------------------------


@pytest.mark.quick
def test_an_unmarked_descriptor_is_authored_and_its_sidecar_is_not_read(tmp_path):
    """The sidecar beside it is corrupt, so reading it at all would raise."""
    path = tmp_path / "solo.ukd.json"
    shipped = write_shipped(path, _embedded_ukd("k0"))
    (tmp_path / provenance_sidecar.PACKED_MARKER).unlink()
    provenance_sidecar.sidecar_path(path).write_bytes(b"not gzip")

    assert provenance_sidecar.attach(path, copy.deepcopy(shipped)) == shipped


@pytest.mark.quick
def test_an_unmarked_kpack_ukd_is_refused(tmp_path):
    """A kpack UKD says on its own that the packer wrote it."""
    path = tmp_path / "solo.ukd.json"
    write_shipped(path, _ukd("k0"))
    (tmp_path / provenance_sidecar.PACKED_MARKER).unlink()
    doc = json.loads(path.read_text(encoding="utf-8"))

    with pytest.raises(
        HkpPackError,
        match=r"looks packed \(it holds kpack UKD\(s\) \['k0'\].*no hkp-packed\.marker",
    ):
        provenance_sidecar.attach(path, doc)


# --- The Index ---------------------------------------------------------------


def _embedded_shard(root):
    """Two packed embedded_source UKDs: no field of either says it is packed."""
    root.mkdir(parents=True, exist_ok=True)
    write_shipped(root / "a.ukd.json", _embedded_ukd("ka"))
    write_shipped(root / "b.ukd.json", _embedded_ukd("kb"))
    return root


@pytest.mark.quick
def test_an_index_refuses_a_tampered_sidecar(tmp_path):
    """The Index surfaces attach's refusal; a reader swallowing it would hand
    every gate a descriptor with no provenance."""
    root = _embedded_shard(tmp_path / "shard")
    _rewrite_sidecar(
        root / "a.ukd.json",
        lambda data: data["entries"]["ka"].update({"ukd_sha256": OTHER_SHA}),
    )

    with pytest.raises(
        descriptor_context.DescriptorContextError, match="different packs"
    ):
        descriptor_context.Index(str(root))


@pytest.mark.quick
def test_an_index_decides_packed_per_directory(tmp_path):
    """A marker counts only in its own directory: neither a parent's nor a
    sibling's reaches an authored descriptor."""
    root = tmp_path / "tree"
    _embedded_shard(root / "packed")
    (root / provenance_sidecar.PACKED_MARKER).write_bytes(b"")
    authored = _embedded_ukd("kc")
    (root / "authored").mkdir()
    (root / "authored" / "c.ukd.json").write_text(json.dumps(authored))

    index = descriptor_context.Index(str(root))

    by_id = {d.doc["id"]: d.doc for d in index.documents}
    assert by_id["ka"]["provenance"] == _embedded_ukd("ka")["provenance"]
    assert by_id["kc"] == authored
    provenance_sidecar.sidecar_path(root / "packed" / "a.ukd.json").unlink()
    with pytest.raises(
        descriptor_context.DescriptorContextError,
        match=r"a\.ukd\.json: packed descriptor has no provenance sidecar",
    ):
        descriptor_context.Index(str(root))


@pytest.mark.quick
def test_an_index_reads_sidecars_from_a_provenance_root(tmp_path):
    root = _embedded_shard(tmp_path / "shard")
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    for name in ("a.ukd.json", "b.ukd.json"):
        side = provenance_sidecar.sidecar_path(root / name)
        side.rename(provenance / side.name)

    index = descriptor_context.Index(str(root), provenance_root=str(provenance))

    assert [d.doc["provenance"] for d in index.documents] == [
        _embedded_ukd("ka")["provenance"],
        _embedded_ukd("kb")["provenance"],
    ]


@pytest.mark.quick
@pytest.mark.parametrize("mode", ["structural", "full"])
def test_desk_check_fails_a_kdp_whose_sidecar_does_not_bind(tmp_path, mode):
    """An Index that swallowed attach's refusal would let the structural run pass on
    a descriptor read without its provenance."""
    kdp = tmp_path / "shard.kdp.json"
    write_shipped(kdp, {"id": "pack-id", "kernelDescriptors": [_embedded_ukd("k0")]})
    control = subprocess.run(
        [sys.executable, str(_DESK_CHECK), "--mode", "structural", str(kdp)],
        capture_output=True,
        text=True,
    )
    _rewrite_sidecar(
        kdp, lambda data: data["entries"]["k0"].update({"ukd_sha256": OTHER_SHA})
    )

    result = subprocess.run(
        [sys.executable, str(_DESK_CHECK), "--mode", mode, str(kdp)],
        capture_output=True,
        text=True,
    )

    assert control.returncode == 0, control.stdout + control.stderr
    assert result.returncode == 1
    assert "different packs" in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.quick
def test_desk_check_reads_sidecars_from_a_provenance_root(tmp_path):
    descriptors = tmp_path / "arch_content"
    kdp = descriptors / "gfx942" / "shard.kdp.json"
    kdp.parent.mkdir(parents=True)
    write_shipped(kdp, {"id": "pack-id", "kernelDescriptors": [_embedded_ukd("k0")]})
    provenance = tmp_path / "provenance"
    side = provenance_sidecar.sidecar_path(kdp)
    (provenance / "gfx942").mkdir(parents=True)
    side.rename(provenance / "gfx942" / side.name)

    def run(*flags):
        return subprocess.run(
            [
                sys.executable,
                str(_DESK_CHECK),
                "--mode",
                "structural",
                *flags,
                str(kdp),
            ],
            capture_output=True,
            text=True,
        )

    found = run(
        "--provenance-root", str(provenance), "--descriptor-root", str(descriptors)
    )
    unnamed = run("--provenance-root", str(provenance))
    missing = run()

    assert found.returncode == 0, found.stdout + found.stderr
    # Without --descriptor-root the KDP's own directory is the mirrored root, so
    # the sidecar is looked for at <provenance>/shard.kdp.provenance.json.gz.
    assert unnamed.returncode == 1
    assert "has no provenance sidecar" in unnamed.stderr
    assert missing.returncode == 1
    assert "pass --provenance-root" in missing.stderr


@pytest.mark.quick
def test_desk_check_refuses_a_provenance_root_for_an_unmarked_kdp(tmp_path):
    kdp = tmp_path / "shard.kdp.json"
    kdp.write_text(json.dumps({"id": "pack-id", "kernelDescriptors": []}))

    result = subprocess.run(
        [
            sys.executable,
            str(_DESK_CHECK),
            "--mode",
            "structural",
            "--provenance-root",
            str(tmp_path / "provenance"),
            str(kdp),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "holds no hkp-packed.marker" in result.stderr
