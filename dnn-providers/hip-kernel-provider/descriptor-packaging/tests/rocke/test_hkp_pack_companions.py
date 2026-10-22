"""A UKD's `companions`: further code objects the kernel launches after its own.

The packer compiles each companion as a rocke variant of its own, validates it against its
builder's support predicate, packs and signs it like the UKD's `kernel_source`, and ships it
as the packed kernel's `companions` -- not as a catalog entry.
"""

import copy
import hashlib
import json
import shutil

import pytest

from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError
from pack_helpers import GENERIC_TARGETS, read_shipped, run_pipeline

ARCH = "gfx950"

_REDUCE = {
    "kind": "rocke",
    "source": "kernels/gfx950/attention_tiled_3d.py",
    "builder": "build_unified_attention_reduce_tiled",
    "spec": {
        "head_size": 128,
        "num_query_heads": 8,
        "num_kv_heads": 8,
        "dtype": "bf16",
        "num_segments": 4,
    },
}


def _with_companions(tmp_path, rocke_fixture, companions):
    src = tmp_path / "src"
    shutil.copytree(rocke_fixture, src)
    path = src / "attention.kdp.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["kernelDescriptors"][0]["companions"] = companions
    path.write_text(json.dumps(doc), encoding="utf-8")
    return src


def _load(tmp_path, rocke_fixture, companions):
    return load_flat_input(
        _with_companions(tmp_path, rocke_fixture, companions), GENERIC_TARGETS
    )


@pytest.mark.quick
def test_validation_accepts_a_rocke_companion(tmp_path, rocke_fixture):
    flat = _load(tmp_path, rocke_fixture, [copy.deepcopy(_REDUCE)])
    kdp = next(k for k in flat.kdps() if k.id == "kdp-attention")
    assert kdp.doc["kernelDescriptors"][0]["companions"][0]["builder"] == (
        "build_unified_attention_reduce_tiled"
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    "companions, message",
    [
        ({"kind": "rocke"}, "must be a non-empty list"),
        ([], "must be a non-empty list"),
        (["not-an-object"], "not a kernel source"),
        ([{"source": "x"}], "not a kernel source"),
        ([{**_REDUCE, "kind": "hip"}], "unsupported kind 'hip'"),
        ([{k: v for k, v in _REDUCE.items() if k != "builder"}], "'builder'"),
        ([{**_REDUCE, "spec": "nope"}], "invalid spec"),
        ([{**_REDUCE, "target": "sideways"}], "invalid target"),
    ],
)
def test_validation_rejects_a_malformed_companions_list(
    tmp_path, rocke_fixture, companions, message
):
    with pytest.raises(HkpPackError, match=message):
        _load(tmp_path, rocke_fixture, companions)


@pytest.mark.quick
def test_validation_names_the_position_of_the_bad_companion(tmp_path, rocke_fixture):
    broken = {**_REDUCE, "kind": "hip"}
    with pytest.raises(HkpPackError, match=r"companions\[1\]"):
        _load(tmp_path, rocke_fixture, [copy.deepcopy(_REDUCE), broken])


@pytest.mark.quick
def test_validation_refuses_companions_on_a_kernel_that_runs_no_producer(
    tmp_path, rocke_fixture
):
    src = _with_companions(tmp_path, rocke_fixture, [copy.deepcopy(_REDUCE)])
    path = src / "attention.kdp.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["kernelDescriptors"][0]["kernel_source"] = {
        "kind": "embedded_source",
        "source_file": "x.cpp",
        "entry_point": "x",
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="runs no producer"):
        load_flat_input(src, GENERIC_TARGETS)


def _pack(tmp_path, src, hipcc, rocm_kpack_dir):
    run_pipeline(
        source_root=src,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )
    return read_shipped(tmp_path / "out" / ARCH / "attention.kdp.json")


def test_a_companion_is_compiled_signed_and_shipped_beside_the_kernel(
    tmp_path, rocke_fixture, hipcc, rocm_kpack_dir, rocke_available
):
    from hkp_pack.kpack_resolver import load_kpack

    src = _with_companions(tmp_path, rocke_fixture, [copy.deepcopy(_REDUCE)])
    shipped = _pack(tmp_path, src, hipcc, rocm_kpack_dir)

    ukd = shipped["kernelDescriptors"][0]
    primary = ukd["kernel_source"]
    (companion,) = ukd["companions"]

    assert companion["kind"] == "kpack"
    assert companion["library"] == primary["library"]
    assert companion["symbol"] != primary["symbol"]
    assert companion["toc_key"] != primary["toc_key"]
    # Signed from the object comgr produced, so the pack's launch is checked against
    # the real argument list rather than a copy of it.
    assert [a["name"] for a in companion["signature"]] == [
        "output_ptr",
        "segm_output_ptr",
        "segm_max_ptr",
        "segm_expsum_ptr",
        "seq_lens_ptr",
    ]

    kpack, _ = load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(
        tmp_path / "out" / ARCH / "kpack" / f"hip_kernel_provider_{ARCH}.kpack"
    )
    blob = archive.get_kernel(companion["toc_key"], ARCH)
    assert blob is not None
    assert hashlib.sha256(blob).hexdigest() == companion["sha256"]
    assert companion["symbol"].encode("ascii") in blob

    # The companion is provenance of the kernel, not a kernel: nothing else in the
    # packed KDP names it, and it carries no metadata of its own.
    assert len(shipped["kernelDescriptors"]) == 1
    assert set(companion) == {
        "kind",
        "library",
        "toc_key",
        "symbol",
        "sha256",
        "signature",
    }
    recorded = ukd["provenance"]["companions"][0]
    assert recorded["builder"] == _REDUCE["builder"]
    assert recorded["spec"] == _REDUCE["spec"]
    assert recorded["origin_kind"] == "rocke"


def test_companions_ship_in_launch_order(
    tmp_path, rocke_fixture, hipcc, rocm_kpack_dir, rocke_available
):
    wide = {**_REDUCE, "spec": {**_REDUCE["spec"], "num_segments": 8}}
    src = _with_companions(
        tmp_path, rocke_fixture, [copy.deepcopy(_REDUCE), copy.deepcopy(wide)]
    )

    shipped = _pack(tmp_path, src, hipcc, rocm_kpack_dir)

    first, second = shipped["kernelDescriptors"][0]["companions"]
    assert "seg4" in first["symbol"] and "seg8" in second["symbol"]


def test_a_companion_outside_its_builders_envelope_fails_the_pack(
    tmp_path, rocke_fixture, hipcc, rocm_kpack_dir, rocke_available
):
    """The reduce builder assumes head_size splits across 64 lanes and checks nothing;
    only the pack-time predicate refuses the spec, so the pack must consult it for a
    companion just as it does for a UKD's own source."""
    bad = {**_REDUCE, "spec": {**_REDUCE["spec"], "head_size": 96}}
    src = _with_companions(tmp_path, rocke_fixture, [bad])

    with pytest.raises(HkpPackError, match="supports_tiled_reduce_spec"):
        _pack(tmp_path, src, hipcc, rocm_kpack_dir)


def test_a_standalone_ukd_ships_its_companions_too(
    tmp_path, rocke_fixture, hipcc, rocm_kpack_dir, rocke_available
):
    src = _with_companions(tmp_path, rocke_fixture, [copy.deepcopy(_REDUCE)])
    kdp_path = src / "attention.kdp.json"
    doc = json.loads(kdp_path.read_text(encoding="utf-8"))
    inline = doc["kernelDescriptors"][0]
    doc["kernelDescriptors"] = [inline["id"]]
    kdp_path.write_text(json.dumps(doc), encoding="utf-8")
    (src / "attention.ukd.json").write_text(json.dumps(inline), encoding="utf-8")

    _pack(tmp_path, src, hipcc, rocm_kpack_dir)
    shipped = read_shipped(tmp_path / "out" / ARCH / "attention.ukd.json")

    (companion,) = shipped["companions"]
    assert companion["kind"] == "kpack"
    assert companion["symbol"] != shipped["kernel_source"]["symbol"]
