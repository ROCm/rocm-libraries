# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A slotted engine of authored `hsaco` UKDs, shaped like the SDPA backward split
engine (RFC 0021 §3): `prep`, `dkdv` and `dq`, each with its own KMD and dispatch
descriptor; `dq` packed for gfx950 and for `gfx11-generic`, `prep` and `dkdv` for
gfx950 only; `dq` is a two-launch kernel (`add`, `scale`) of one code object.

Compiler-free: the code objects are committed under `fixtures/slotted_hsaco/objects/`.
"""

import hashlib
import json
from pathlib import Path

import pytest

from hkp_pack import descriptor_context
from hkp_pack.kernel_signature import exported_kernels
from pack_helpers import read_shipped, run_pipeline

pytestmark = pytest.mark.quick

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "slotted_hsaco"
SLOTS = ("prep", "dkdv", "dq")
ARCH = "gfx950"
GENERIC = "gfx11-generic"
GENERIC_MEMBER = "gfx1100"
# The generic pack reuses the gfx942 object, as test_hkp_pack_generic.py does: packing
# does not read the object's arch.
GFX950_OBJECT = FIXTURE / "objects" / "gfx950" / "HsacoFixture.co"
GENERIC_OBJECT = FIXTURE / "objects" / "gfx942" / "HsacoFixture.co"
SYMBOLS = {"prep": "HsacoFixtureScale", "dkdv": "HsacoFixtureAdd"}
LAUNCH_SYMBOLS = {"add": "HsacoFixtureAdd", "scale": "HsacoFixtureScale"}


@pytest.fixture(scope="module")
def packed(tmp_path_factory, rocm_kpack_dir):
    tmp_path = tmp_path_factory.mktemp("slotted_hsaco")
    results = run_pipeline(
        source_root=FIXTURE,
        arches=[ARCH, GENERIC_MEMBER],
        out_root=tmp_path / "out",
        hipcc="hipcc-not-invoked",
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
        source_label="slotted-hsaco-test",
    )
    return tmp_path / "out", results


def _archive(rocm_kpack_dir, kpack_path):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    return kpack.PackedKernelArchive.read(kpack_path)


def _kernel(shard, kdp_name):
    (kernel,) = read_shipped(shard / kdp_name)["kernelDescriptors"]
    return kernel


def test_the_gfx950_shard_ships_every_slot(packed, rocm_kpack_dir):
    out, results = packed
    shard = out / ARCH
    object_bytes = GFX950_OBJECT.read_bytes()
    archive = _archive(rocm_kpack_dir, results[ARCH].kpack_path)

    # prep and dkdv have a gfx950 pack each, dq its gfx950 one; the generic pack of dq
    # belongs to the generic's members and not to gfx950.
    assert sorted(p.name for p in shard.glob("*.kdp.json")) == [
        "dkdv.kdp.json",
        "dq_gfx950.kdp.json",
        "prep.kdp.json",
    ]
    for stem, slot in (("prep", "prep"), ("dkdv", "dkdv"), ("dq_gfx950", "dq")):
        pack = read_shipped(shard / f"{stem}.kdp.json")
        assert pack["slot"] == slot
        (kernel,) = pack["kernelDescriptors"]
        if slot == "dq":
            launches = list(kernel["sources"].values())
            assert {k["symbol"] for k in launches} == set(LAUNCH_SYMBOLS.values())
        else:
            launches = [kernel["kernel_source"]]
            assert launches[0]["symbol"] == SYMBOLS[slot]
        for ks in launches:
            assert ks["kind"] == "kpack"
            assert ks["sha256"] == hashlib.sha256(object_bytes).hexdigest()
            assert bytes(archive.get_kernel(ks["toc_key"], ARCH)) == object_bytes

    # One object file serves the three slots: one archive entry.
    assert len(list(archive.toc)) == 1
    assert len({p.name for p in shard.glob("*.udd.json")}) == len(SLOTS)


def test_dq_ships_two_launches_of_one_object_and_prep_and_dkdv_one_each(
    packed, rocm_kpack_dir
):
    out, results = packed
    shard = out / ARCH
    bundles = {
        bundle.kdp_doc["slot"]: bundle
        for bundle in descriptor_context.resolve_bundles(
            descriptor_context.Index(str(shard))
        )
    }
    assert [f["name"] for f in bundles["dq"].kmd["fields"]] == ["dtype"]

    (kernel,) = bundles["dq"].entries
    assert "kernel_source" not in kernel.ukd
    assert kernel.ukd["version"] == "1.1"
    sources = kernel.ukd["sources"]
    assert {name: s["symbol"] for name, s in sources.items()} == LAUNCH_SYMBOLS
    assert len({s["toc_key"] for s in sources.values()}) == 1

    toc_key = sources["add"]["toc_key"]
    archive = _archive(rocm_kpack_dir, results[ARCH].kpack_path)
    exported = exported_kernels(bytes(archive.get_kernel(toc_key, ARCH)), "dq")
    assert set(LAUNCH_SYMBOLS.values()) <= set(exported)

    # prep and dkdv stay single-launch.
    for slot in ("prep", "dkdv"):
        (single,) = bundles[slot].entries
        assert "sources" not in single.ukd and "kernel_source" in single.ukd


def test_the_generic_member_shard_ships_dq_only_and_so_no_complete_engine(
    packed, rocm_kpack_dir
):
    out, results = packed
    shard = out / GENERIC_MEMBER
    assert [p.name for p in shard.glob("*.kdp.json")] == ["dq_generic.kdp.json"]

    kernel = _kernel(shard, "dq_generic.kdp.json")
    assert kernel["arch"] == [GENERIC]
    assert {n: k["symbol"] for n, k in kernel["sources"].items()} == LAUNCH_SYMBOLS
    for ks in kernel["sources"].values():
        assert ks["library"] == f"kpack/hip_kernel_provider_{GENERIC}.kpack"
        archive = _archive(rocm_kpack_dir, shard / ks["library"])
        assert bytes(archive.get_kernel(ks["toc_key"], GENERIC)) == (
            GENERIC_OBJECT.read_bytes()
        )

    # The engine's descriptors ship, but only dq of its three slots has a pack here.
    ued = json.loads((shard / "fixture.ued.json").read_text(encoding="utf-8"))
    assert [slot["name"] for slot in ued["slots"]] == list(SLOTS)
    served = {
        bundle.kdp_doc["slot"]
        for bundle in descriptor_context.resolve_bundles(
            descriptor_context.Index(str(shard))
        )
    }
    assert served == {"dq"}
