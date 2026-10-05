# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The provenance sidecar's own contract: what detach writes, and every way
attach refuses a descriptor and sidecar that do not belong together."""

import copy
import gzip
import json

import pytest

from conftest import write_shipped
from hkp_pack import pipeline, provenance_sidecar
from hkp_pack.errors import HkpPackError

SHA = "a" * 64
OTHER_SHA = "b" * 64


def _ukd(ident, sha=SHA, kind="kpack"):
    return {
        "version": "1.0",
        "id": ident,
        "name": ident,
        "kernel_source": {"kind": kind, "sha256": sha},
        "provenance": {"origin_kind": "hip", "source_label": ident},
    }


def _kdp():
    return {
        "version": "1.0",
        "id": "pack-id",
        "name": "pack",
        "kernelDescriptors": [_ukd("k0"), "a-standalone-id", _ukd("k1")],
    }


def _rewrite_sidecar(path, mutate):
    side = provenance_sidecar.sidecar_path(path)
    data = json.loads(gzip.decompress(side.read_bytes()))
    mutate(data)
    side.write_bytes(provenance_sidecar.encode(data))


@pytest.mark.quick
def test_attach_restores_what_detach_moved(tmp_path):
    original = _kdp()
    path = tmp_path / "solo.kdp.json"
    shipped = write_shipped(path, original)

    inline = [e for e in shipped["kernelDescriptors"] if isinstance(e, dict)]
    assert all("provenance" not in ukd for ukd in inline)
    assert provenance_sidecar.attach(path, shipped) == original


@pytest.mark.quick
def test_a_packed_ukd_without_its_sidecar_is_refused(tmp_path):
    path = tmp_path / "solo.kdp.json"
    shipped = write_shipped(path, _kdp())
    provenance_sidecar.sidecar_path(path).unlink()

    with pytest.raises(HkpPackError, match="has no provenance sidecar"):
        provenance_sidecar.attach(path, shipped)


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


@pytest.mark.quick
def test_an_unreadable_sidecar_is_refused(tmp_path):
    path = tmp_path / "solo.ukd.json"
    shipped = write_shipped(path, _ukd("k0"))
    side = provenance_sidecar.sidecar_path(path)

    side.write_bytes(b"not gzip")
    with pytest.raises(HkpPackError, match="cannot read provenance sidecar"):
        provenance_sidecar.attach(path, copy.deepcopy(shipped))

    side.write_bytes(provenance_sidecar.encode({"kdp_id": None, "entries": []}))
    with pytest.raises(HkpPackError, match="has no 'entries' object"):
        provenance_sidecar.attach(path, copy.deepcopy(shipped))


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
    assert data[3] & 0x08 == 0  # FLG.FNAME clear
    assert data[4:8] == b"\0\0\0\0"  # MTIME


@pytest.mark.quick
def test_two_descriptors_sharing_a_stem_cannot_share_a_sidecar(tmp_path):
    sidecars = set()
    pipeline._write_packed_at(tmp_path, ".", "solo.kdp.json", _kdp(), sidecars)

    with pytest.raises(HkpPackError, match="sharing its stem"):
        pipeline._write_packed_at(tmp_path, ".", "solo.ukd.json", _ukd("k9"), sidecars)
