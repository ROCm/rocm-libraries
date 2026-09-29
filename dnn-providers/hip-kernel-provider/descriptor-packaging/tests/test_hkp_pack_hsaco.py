"""Authored `kind: hsaco` UKDs: a prebuilt code object packed as-is.

The invariant: a shipped UKD that came from an authored hsaco is a `kind:
kpack` UKD whose archive entry is byte-identical to the file the descriptor
named.

Every test here is compiler-free. The code objects are the committed fixtures
under `fixtures/hsaco/`, and packing goes through `run_pipeline` with a hipcc
path nothing may invoke.
"""

import copy
import hashlib
import json
import shutil

import pytest

from hkp_pack import pipeline, toolchain
from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError
from hkp_pack.hsaco_source import hsaco_file_relpath, hsaco_variant_key
from hkp_pack.kernel_signature import kernel_signature
from hkp_pack.pipeline import compile_intermediate, run_pipeline

ARCH = "gfx942"
OTHER_ARCH = "gfx950"
CO_NAME = "HsacoFixture.co"
NO_HIPCC = "hipcc-not-invoked"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, doc):
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _load_kpack(rocm_kpack_dir):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    return kpack


def _run(source_root, tmp_path, rocm_kpack_dir, arches=(ARCH,), **kwargs):
    return run_pipeline(
        source_root=source_root,
        arches=list(arches),
        out_root=tmp_path / "out",
        hipcc=NO_HIPCC,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
        **kwargs,
    )


def _nest(root, sub, fixture):
    """Copy a flat fixture into `root/sub`, returning the child folder."""
    dest = root / sub
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fixture, dest)
    return dest


def _rename_ids(folder, stem, new_stem):
    """Re-stem a fixture's files and ids so two copies can coexist in one root."""
    for src in sorted(folder.glob(f"{stem}.*")):
        dst = folder / src.name.replace(f"{stem}.", f"{new_stem}.", 1)
        dst.write_text(
            src.read_text(encoding="utf-8").replace(f"-{stem}", f"-{new_stem}"),
            encoding="utf-8",
        )
        src.unlink()


def _hsaco_source(file=CO_NAME, symbol="HsacoFixtureAdd"):
    return {"kind": "hsaco", "file": file, "symbol": symbol}


def _hsaco_ukd(template, uid, file=CO_NAME, symbol="HsacoFixtureAdd"):
    """A copy of the fixture's inline UKD, re-identified and pointed at `file`."""
    ukd = copy.deepcopy(template)
    ukd["id"] = uid
    ukd["name"] = f"{uid} ({symbol})"
    ukd["kernel_source"] = _hsaco_source(file, symbol)
    return ukd


def _hsaco_root(tmp_path, empty_arch_fixture, hsaco_fixture_dir, mutate=None):
    """`empty_arch` nested under `solo/`, its inline UKD rewritten to hsaco.

    The gfx942 fixture is copied beside the KDP. `mutate(kdp_doc, template_ukd)`
    may rewrite the KDP further before it is written back.
    """
    root = tmp_path / "root"
    child = _nest(root, "solo", empty_arch_fixture)
    shutil.copyfile(hsaco_fixture_dir / ARCH / CO_NAME, child / CO_NAME)
    kdp_path = child / "solo.kdp.json"
    doc = _read(kdp_path)
    template = doc["kernelDescriptors"][0]
    if mutate is None:
        template["kernel_source"] = _hsaco_source()
    else:
        mutate(doc, template)
    _write(kdp_path, doc)
    return root


def _shipped_ukds(kdp_path):
    return {u["id"]: u for u in _read(kdp_path)["kernelDescriptors"]}


@pytest.mark.quick
def test_inline_hsaco_round_trips_byte_identical(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir, monkeypatch
):
    """The archive entry is the authored file, and provenance names only it."""
    monkeypatch.setattr(
        toolchain, "hipcc_version", lambda _h: "HIP version: 0.0.0-test"
    )
    # Under the patch a hip UKD would carry `hipcc_version`, so its absence below
    # is a statement about hsaco and not about an unreachable hipcc.
    assert toolchain.hip_provenance("x")

    root = _hsaco_root(tmp_path, empty_arch_fixture, hsaco_fixture_dir)
    authored = _read(root / "solo" / "solo.kdp.json")["kernelDescriptors"][0]
    fixture = (hsaco_fixture_dir / ARCH / CO_NAME).read_bytes()
    digest = hashlib.sha256(fixture).hexdigest()

    results = _run(root, tmp_path, rocm_kpack_dir)

    out = tmp_path / "out" / ARCH
    shipped = _read(out / "solo" / "solo.kdp.json")["kernelDescriptors"][0]
    ks = shipped["kernel_source"]
    assert ks["kind"] == "kpack"
    assert ks["toc_key"] == hsaco_variant_key(hsaco_file_relpath("solo", CO_NAME))
    assert ks["sha256"] == digest
    assert ks["symbol"] == "HsacoFixtureAdd"
    assert ks["signature"] == kernel_signature(fixture, "HsacoFixtureAdd", "fixture")

    archive = _load_kpack(rocm_kpack_dir).PackedKernelArchive.read(
        results[ARCH].kpack_path
    )
    assert bytes(archive.get_kernel(ks["toc_key"], ARCH)) == fixture

    assert shipped["provenance"] == {
        **authored["provenance"],
        "origin_kind": "hsaco",
        "file": CO_NAME,
        "sha256": digest,
        "symbol": "HsacoFixtureAdd",
    }
    assert "hipcc_version" not in shipped["provenance"]

    # Packed from the authored file: no copy of it anywhere in either tree.
    assert not list((tmp_path / "out").rglob("*.co"))
    assert not list((tmp_path / "inter" / ARCH).rglob("*.co"))
    # The intermediate KDP keeps the authored form.
    inter = _read(tmp_path / "inter" / ARCH / "solo" / "solo.kdp.json")
    assert inter["kernelDescriptors"][0]["kernel_source"] == _hsaco_source()


@pytest.mark.quick
def test_one_file_serves_two_symbols_as_one_entry(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    """Two symbols of one file share a toc_key and one archive entry, and each
    keeps its own argument list."""

    def two_symbols(doc, template):
        doc["kernelDescriptors"] = [
            _hsaco_ukd(template, "ukd-add", symbol="HsacoFixtureAdd"),
            _hsaco_ukd(template, "ukd-scale", symbol="HsacoFixtureScale"),
        ]

    root = _hsaco_root(tmp_path, empty_arch_fixture, hsaco_fixture_dir, two_symbols)

    results = _run(root, tmp_path, rocm_kpack_dir)

    shipped = _shipped_ukds(tmp_path / "out" / ARCH / "solo" / "solo.kdp.json")
    add = shipped["ukd-add"]["kernel_source"]
    scale = shipped["ukd-scale"]["kernel_source"]
    assert add["toc_key"] == scale["toc_key"]
    archive = _load_kpack(rocm_kpack_dir).PackedKernelArchive.read(
        results[ARCH].kpack_path
    )
    assert list(archive.toc) == [add["toc_key"]]
    assert len(add["signature"]) == 4
    assert len(scale["signature"]) == 3


@pytest.mark.quick
def test_standalone_and_nested_resolution(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    """A standalone UKD resolves `../shared/...` from its own folder, and two
    descriptors naming a same-named sibling file get distinct keys."""
    fixture = (hsaco_fixture_dir / ARCH / CO_NAME).read_bytes()
    standalone_id = "ukd-standalone-hsaco"

    def reference_standalone(doc, template):
        standalone = _hsaco_ukd(template, standalone_id, file=f"../shared/{CO_NAME}")
        standalone_dir = tmp_path / "root" / "a"
        standalone_dir.mkdir(parents=True)
        _write(standalone_dir / "standalone.ukd.json", standalone)
        doc["kernelDescriptors"] = [standalone_id]

    root = _hsaco_root(
        tmp_path, empty_arch_fixture, hsaco_fixture_dir, reference_standalone
    )
    (root / "solo" / CO_NAME).unlink()
    (root / "shared").mkdir()
    (root / "shared" / CO_NAME).write_bytes(fixture)

    results = _run(root, tmp_path, rocm_kpack_dir)

    shipped = _read(tmp_path / "out" / ARCH / "a" / "standalone.ukd.json")
    ks = shipped["kernel_source"]
    assert ks["kind"] == "kpack"
    assert ks["toc_key"] == hsaco_variant_key(f"shared/{CO_NAME}")
    assert shipped["provenance"]["file"] == f"../shared/{CO_NAME}"
    archive = _load_kpack(rocm_kpack_dir).PackedKernelArchive.read(
        results[ARCH].kpack_path
    )
    assert bytes(archive.get_kernel(ks["toc_key"], ARCH)) == fixture

    nested = tmp_path / "nested"
    for sub in ("p", "q"):
        folder = _nest(nested / "root", sub, empty_arch_fixture)
        _rename_ids(folder, "solo", sub)
        (folder / CO_NAME).write_bytes(fixture)
        kdp_path = folder / f"{sub}.kdp.json"
        doc = _read(kdp_path)
        doc["kernelDescriptors"][0]["kernel_source"] = _hsaco_source()
        _write(kdp_path, doc)

    _run(nested / "root", nested, rocm_kpack_dir)

    key_p = _read(nested / "out" / ARCH / "p" / "p.kdp.json")["kernelDescriptors"][0]
    key_q = _read(nested / "out" / ARCH / "q" / "q.kdp.json")["kernelDescriptors"][0]
    assert key_p["kernel_source"]["toc_key"] == hsaco_variant_key(f"p/{CO_NAME}")
    assert key_q["kernel_source"]["toc_key"] == hsaco_variant_key(f"q/{CO_NAME}")
    assert key_p["kernel_source"]["toc_key"] != key_q["kernel_source"]["toc_key"]


@pytest.mark.quick
def test_missing_file_has_no_root_fallback(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir
):
    """A file present at the root but not beside the descriptor is not found,
    and a path leaving the root is refused."""
    root = _hsaco_root(tmp_path, empty_arch_fixture, hsaco_fixture_dir)
    shutil.move(root / "solo" / CO_NAME, root / CO_NAME)
    flat = load_flat_input(root)

    with pytest.raises(HkpPackError) as excinfo:
        compile_intermediate(flat, root, ARCH, NO_HIPCC, tmp_path / "inter")
    message = str(excinfo.value)
    assert "hsaco file not found" in message
    assert "ukd-solo-add-f32-b64" in message
    assert CO_NAME in message

    kdp_path = root / "solo" / "solo.kdp.json"
    doc = _read(kdp_path)
    doc["kernelDescriptors"][0]["kernel_source"]["file"] = f"../../{CO_NAME}"
    _write(kdp_path, doc)
    flat = load_flat_input(root)

    with pytest.raises(HkpPackError, match="hsaco file escapes the source root"):
        compile_intermediate(flat, root, ARCH, NO_HIPCC, tmp_path / "inter")


@pytest.mark.quick
def test_same_key_for_two_files_is_refused_in_the_walk(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, monkeypatch
):
    """Two different files forced onto one key fail at the second claim."""

    def two_files(doc, template):
        doc["kernelDescriptors"] = [
            _hsaco_ukd(template, "ukd-first", file=CO_NAME),
            _hsaco_ukd(template, "ukd-second", file=f"other/{CO_NAME}"),
        ]

    root = _hsaco_root(tmp_path, empty_arch_fixture, hsaco_fixture_dir, two_files)
    (root / "solo" / "other").mkdir()
    shutil.copyfile(
        hsaco_fixture_dir / ARCH / CO_NAME, root / "solo" / "other" / CO_NAME
    )
    flat = load_flat_input(root)
    monkeypatch.setattr(pipeline, "hsaco_variant_key", lambda *a, **k: "COLLIDE")

    with pytest.raises(HkpPackError) as excinfo:
        compile_intermediate(flat, root, ARCH, NO_HIPCC, tmp_path / "inter")
    message = str(excinfo.value)
    assert "toc_key collision: hsaco key 'COLLIDE'" in message
    assert str((root / "solo" / CO_NAME).resolve()) in message
    assert str((root / "solo" / "other" / CO_NAME).resolve()) in message


@pytest.mark.quick
def test_per_arch_objects_land_in_their_own_shards(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    """Per-arch objects in per-arch UKDs each land in their own shard."""
    per_arch = tmp_path / "per-arch"

    def one_ukd_per_arch(doc, template):
        doc["arch"] = [ARCH, OTHER_ARCH]
        ukds = []
        for arch in (ARCH, OTHER_ARCH):
            ukd = _hsaco_ukd(template, f"ukd-{arch}", file=f"{arch}/{CO_NAME}")
            ukd["arch"] = [arch]
            ukds.append(ukd)
        doc["kernelDescriptors"] = ukds

    root = _hsaco_root(
        per_arch, empty_arch_fixture, hsaco_fixture_dir, one_ukd_per_arch
    )
    for arch in (ARCH, OTHER_ARCH):
        shutil.copytree(hsaco_fixture_dir / arch, root / "solo" / arch)

    results = _run(root, per_arch, rocm_kpack_dir, arches=(ARCH, OTHER_ARCH))

    kpack = _load_kpack(rocm_kpack_dir)
    for arch in (ARCH, OTHER_ARCH):
        shipped = _shipped_ukds(per_arch / "out" / arch / "solo" / "solo.kdp.json")
        assert list(shipped) == [f"ukd-{arch}"]
        ks = shipped[f"ukd-{arch}"]["kernel_source"]
        archive = kpack.PackedKernelArchive.read(results[arch].kpack_path)
        blob = bytes(archive.get_kernel(ks["toc_key"], arch))
        assert blob == (hsaco_fixture_dir / arch / CO_NAME).read_bytes()


@pytest.mark.quick
def test_expected_sha256_applies_to_the_hsaco_key(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    root = _hsaco_root(tmp_path, empty_arch_fixture, hsaco_fixture_dir)
    toc_key = hsaco_variant_key(hsaco_file_relpath("solo", CO_NAME))

    with pytest.raises(HkpPackError, match="sha256 mismatch"):
        _run(root, tmp_path, rocm_kpack_dir, expected_sha256={toc_key: "0" * 64})


@pytest.mark.quick
def test_hsaco_and_hip_on_one_key_collide_at_pack(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir, monkeypatch
):
    """An hsaco key reused by a hip UKD is caught by pack_arch's signature check.

    The hsaco UKD is authored first, so the hip arm finds the key taken and
    skips its compile: no hipcc runs, and only the pack-time check can object.
    """

    def hsaco_then_hip(doc, template):
        hip = copy.deepcopy(template)
        doc["kernelDescriptors"] = [_hsaco_ukd(template, "ukd-hsaco"), hip]

    root = _hsaco_root(tmp_path, empty_arch_fixture, hsaco_fixture_dir, hsaco_then_hip)
    monkeypatch.setattr(pipeline, "hsaco_variant_key", lambda *a, **k: "COLLIDE")
    monkeypatch.setattr(pipeline, "hip_variant_key", lambda *a, **k: "COLLIDE")

    with pytest.raises(
        HkpPackError, match="toc_key collision: 'COLLIDE' maps to two distinct inputs"
    ):
        _run(root, tmp_path, rocm_kpack_dir)


@pytest.mark.quick
def test_hsaco_cannot_fulfil_compiled_bindings(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir
):
    """No compiler ran, so a contract naming compiled fields is refused."""

    def bound_contract(doc, template):
        template["kernel_source"] = _hsaco_source()
        (consumer,) = template["provenance"]["specialization_contract"]["consumers"]
        consumer["metadata_fields"] = ["block_size"]
        consumer["matcher_only_fields"] = ["dtype"]
        consumer["bindings"] = {"block_size": {"field": "block_size"}}

    root = _hsaco_root(tmp_path, empty_arch_fixture, hsaco_fixture_dir, bound_contract)
    flat = load_flat_input(root)

    with pytest.raises(HkpPackError, match="a 'hsaco' source cannot fulfil"):
        compile_intermediate(flat, root, ARCH, NO_HIPCC, tmp_path / "inter")
