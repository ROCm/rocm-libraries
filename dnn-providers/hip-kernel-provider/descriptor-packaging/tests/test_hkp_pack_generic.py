"""Generic GPU targets at pack time; see descriptor-packaging/README.md."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from hkp_pack import pipeline
from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError
from hkp_pack.generic_targets import GenericTargets
from hkp_pack.pipeline import _merge_generic_into_member, run_pipeline
from pack_helpers import GENERIC_TARGETS, GENERIC_TARGETS_JSON

GENERIC = "gfx11-generic"
MEMBER_A = "gfx1100"
MEMBER_B = "gfx1151"
UNSELECTED_MEMBER = "gfx1101"
CO_NAME = "HsacoFixture.co"

_SHARED_SUFFIXES = (".kmd.json", ".ued.json", ".udd.json", ".uhd.json", ".umd.json")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _load_kpack(rocm_kpack_dir):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    return kpack


def _embedded(uid, arch=None):
    ukd = {
        "version": "0.1",
        "id": uid,
        "name": uid,
        "kernel_source": {
            "kind": "embedded_source",
            "source_file": "kernels/PointwiseAdd.cpp",
            "entry_point": "PointwiseAdd",
        },
        "metadata": {"dtype": "FLOAT", "block_size": 64},
        "priority": 0,
    }
    if arch is not None:
        ukd["arch"] = arch
    return ukd


_CONTRACT = {
    "specialization_contract": {
        "schema_version": 1,
        "consumers": [
            {
                "engine_id": "ued-solo",
                "kmd_id": "kmd-solo",
                "metadata_fields": [],
                "matcher_only_fields": ["block_size", "dtype"],
                "bindings": {},
                "vocabulary": {},
            }
        ],
    }
}


def _kdp(kid, arch, ukds):
    return {
        "version": "0.1",
        "id": kid,
        "name": kid,
        "arch": arch,
        "matchers": ["umd-solo"],
        "engine": "ued-solo",
        "dispatch": "udd-solo",
        "kernelDescriptors": ukds,
    }


def _root(tmp_path, empty_arch_fixture, kdps, standalone=()):
    """A source root: the fixture's shared descriptors plus KDP and standalone-UKD docs by stem."""
    root = tmp_path / "src"
    root.mkdir()
    for path in empty_arch_fixture.iterdir():
        if path.name.endswith(_SHARED_SUFFIXES):
            shutil.copyfile(path, root / path.name)
    for stem, doc in dict(kdps).items():
        _write(root / f"{stem}.kdp.json", doc)
    for stem, doc in dict(standalone).items():
        _write(root / f"{stem}.ukd.json", doc)
    return root


def _pack(
    root,
    tmp_path,
    arches,
    rocm_kpack_dir,
    table=GENERIC_TARGETS_JSON,
    hipcc="hipcc-not-invoked",
):
    return run_pipeline(
        source_root=root,
        arches=list(arches),
        out_root=tmp_path / "out",
        hipcc=hipcc,
        generic_targets_json=table,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
        source_label="generic-test",
    )


def _files(folder):
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _load(root, table=GENERIC_TARGETS):
    return load_flat_input(root, table, log=lambda *_: None)


@pytest.mark.quick
def test_generic_entries_land_in_every_selected_member_folder_only(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {
            "c": _kdp("kdp-c", ["gfx942"], [_embedded("ukd-c")]),
            "explicit": _kdp("kdp-explicit", [MEMBER_B], [_embedded("ukd-e")]),
            "g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-g")]),
            "wild": _kdp("kdp-wild", [], [_embedded("ukd-wild")]),
        },
    )
    _pack(root, tmp_path, ["gfx942", MEMBER_A, MEMBER_B], rocm_kpack_dir)

    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [MEMBER_A, MEMBER_B, "gfx942"]
    assert not (out / UNSELECTED_MEMBER).exists()
    assert not (out / GENERIC).exists()

    for member in (MEMBER_A, MEMBER_B):
        kdp = _read(out / member / "g.kdp.json")
        assert kdp["arch"] == [GENERIC]
        assert [u["arch"] for u in kdp["kernelDescriptors"]] == [[GENERIC]]
        assert _read(out / member / "wild.kdp.json")["arch"] == [member]
    assert (out / MEMBER_A / "g.kdp.json").read_bytes() == (
        out / MEMBER_B / "g.kdp.json"
    ).read_bytes()
    assert not (out / "gfx942" / "g.kdp.json").exists()

    assert (out / "gfx942" / "c.kdp.json").is_file()
    assert not (out / MEMBER_A / "c.kdp.json").exists()
    assert (out / MEMBER_B / "explicit.kdp.json").is_file()
    assert not (out / MEMBER_A / "explicit.kdp.json").exists()

    shared = [p for p in root.iterdir() if p.name.endswith(_SHARED_SUFFIXES)]
    assert shared
    for arch in ("gfx942", MEMBER_A, MEMBER_B):
        files = _files(out / arch)
        for path in shared:
            assert files[path.name] == path.read_bytes()


@pytest.mark.quick
def test_generic_entry_is_not_materialized_when_no_member_is_selected(
    tmp_path, empty_arch_fixture, rocm_kpack_dir, monkeypatch
):
    seen = []
    for step in ("compile_intermediate", "pack_arch"):
        real = getattr(pipeline, step)

        def spy(*args, _real=real, _step=step, **kwargs):
            seen.append(args[2] if _step == "compile_intermediate" else args[1].arch)
            return _real(*args, **kwargs)

        monkeypatch.setattr(pipeline, step, spy)

    root = _root(
        tmp_path,
        empty_arch_fixture,
        {
            "c": _kdp("kdp-c", ["gfx942"], [_embedded("ukd-c")]),
            "g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-g")]),
        },
    )
    _pack(root, tmp_path, ["gfx942"], rocm_kpack_dir)

    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["gfx942"]
    assert not (out / "gfx942" / "g.kdp.json").exists()
    assert seen
    assert GENERIC not in seen


@pytest.mark.quick
def test_merge_collision_with_differing_bytes_is_refused_and_leaves_the_member_untouched(
    tmp_path,
):
    generic_dir = tmp_path / "inter" / ".generic-out" / GENERIC
    member_dir = tmp_path / "out" / MEMBER_A
    staging = tmp_path / "out" / f".{MEMBER_A}.merge.staging"
    _write(generic_dir / "only_generic.kdp.json", {"id": "kdp-only"})
    _write(generic_dir / "dup.ukd.json", {"id": "ukd-dup", "side": "generic"})
    _write(member_dir / "dup.ukd.json", {"id": "ukd-dup", "side": "concrete"})
    _write(member_dir / "concrete_only.kdp.json", {"id": "kdp-c"})
    before = _files(member_dir)

    with pytest.raises(HkpPackError):
        _merge_generic_into_member(generic_dir, member_dir, staging)

    assert _files(member_dir) == before
    assert not staging.exists()


@pytest.mark.quick
def test_merge_copy_failure_midway_is_refused_and_leaves_the_member_untouched(
    tmp_path, monkeypatch
):
    generic_dir = tmp_path / "inter" / ".generic-out" / GENERIC
    member_dir = tmp_path / "out" / MEMBER_A
    staging = tmp_path / "out" / f".{MEMBER_A}.merge.staging"
    _write(generic_dir / "first.kdp.json", {"id": "kdp-first"})
    _write(generic_dir / "second.kdp.json", {"id": "kdp-second"})
    _write(member_dir / "concrete_only.kdp.json", {"id": "kdp-c"})
    before = _files(member_dir)

    real_copyfile = shutil.copyfile
    generic_copies = []

    def failing_on_second_generic_copy(src, dst, *args, **kwargs):
        if Path(src).is_relative_to(generic_dir):
            generic_copies.append(src)
            if len(generic_copies) == 2:
                raise OSError("injected copy failure")
        return real_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", failing_on_second_generic_copy)

    with pytest.raises(HkpPackError):
        _merge_generic_into_member(generic_dir, member_dir, staging)

    assert len(generic_copies) == 2, "the merge no longer copies via shutil.copyfile"
    assert _files(member_dir) == before
    assert not staging.exists()


@pytest.mark.quick
@pytest.mark.parametrize(
    "step, dropped, with_generic",
    [
        ("compile_intermediate", (MEMBER_A,), (MEMBER_B,)),
        ("pack_arch", (), ()),
    ],
)
def test_a_failed_pass_leaves_no_partial_member_or_generic_copy(
    tmp_path,
    empty_arch_fixture,
    rocm_kpack_dir,
    monkeypatch,
    step,
    dropped,
    with_generic,
):
    """A failing concrete compile drops that member; a failing generic pack drops every generic copy."""
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {
            "c": _kdp("kdp-c", [MEMBER_A, MEMBER_B], [_embedded("ukd-c")]),
            "g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-g")]),
        },
    )
    real = getattr(pipeline, step)

    def failing(*args, **kwargs):
        arch = args[2] if step == "compile_intermediate" else args[1].arch
        if arch == (MEMBER_A if step == "compile_intermediate" else GENERIC):
            raise HkpPackError("synthetic failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, step, failing)
    with pytest.raises(HkpPackError):
        _pack(root, tmp_path, [MEMBER_A, MEMBER_B], rocm_kpack_dir)

    out = tmp_path / "out"
    for member in (MEMBER_A, MEMBER_B):
        if member in dropped:
            assert not (out / member).exists()
            continue
        assert (out / member / "c.kdp.json").is_file()
        assert (out / member / "g.kdp.json").is_file() == (member in with_generic)
    assert not [p for p in out.iterdir() if p.name.startswith(".")]


@pytest.mark.quick
def test_generic_copies_of_every_producer_are_equal_across_separate_builds(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    """Generic copies from separate per-arch builds are identical; the build's targets never leak in."""
    hsaco = _hsaco_ukd("ukd-hsaco", [GENERIC])
    folders = {}
    for member in (MEMBER_A, MEMBER_B):
        base = tmp_path / member
        base.mkdir()
        root = _root(
            base,
            empty_arch_fixture,
            {
                "e": _kdp("kdp-e", [GENERIC], [_embedded("ukd-e")]),
                "h": _kdp("kdp-h", [GENERIC], [copy.deepcopy(hsaco)]),
            },
        )
        shutil.copyfile(hsaco_fixture_dir / "gfx942" / CO_NAME, root / CO_NAME)
        _pack(root, base, [member], rocm_kpack_dir)
        folders[member] = _files(base / "out" / member)
    assert any(name.endswith(".kpack") for name in folders[MEMBER_A])
    assert folders[MEMBER_A] == folders[MEMBER_B]


@pytest.mark.quick
def test_hsaco_under_a_generic_packs_under_the_generic_key_into_every_member_folder(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    authored = (hsaco_fixture_dir / "gfx942" / CO_NAME).read_bytes()
    ukd = _hsaco_ukd("ukd-hsaco", [GENERIC])
    root = _root(tmp_path, empty_arch_fixture, {"g": _kdp("kdp-g", [GENERIC], [ukd])})
    shutil.copyfile(hsaco_fixture_dir / "gfx942" / CO_NAME, root / CO_NAME)
    results = _pack(root, tmp_path, [MEMBER_A, MEMBER_B], rocm_kpack_dir)

    out = tmp_path / "out"
    shipped = _read(out / MEMBER_A / "g.kdp.json")["kernelDescriptors"][0]
    ks = shipped["kernel_source"]
    assert ks["kind"] == "kpack"
    assert ks["library"] == f"kpack/hip_kernel_provider_{GENERIC}.kpack"
    assert shipped["arch"] == [GENERIC]
    for member in (MEMBER_A, MEMBER_B):
        archive_path = out / member / ks["library"]
        assert results[member].generic_kpack_paths == (archive_path,)
        archive = _load_kpack(rocm_kpack_dir).PackedKernelArchive.read(archive_path)
        assert bytes(archive.get_kernel(ks["toc_key"], GENERIC)) == authored
        assert archive.get_kernel(ks["toc_key"], member) is None
    assert _files(out / MEMBER_A) == _files(out / MEMBER_B)


def _rocke_ukd(uid, arch=None):
    ukd = _embedded(uid, arch)
    ukd["kernel_source"] = {
        "kind": "rocke",
        "source": "pkg/kernels/attention.py",
        "builder": "build_attention",
        "spec": {"tile": 64},
    }
    return ukd


def _hsaco_ukd(uid, arch):
    ukd = _embedded(uid, arch)
    ukd["kernel_source"] = {
        "kind": "hsaco",
        "file": CO_NAME,
        "symbol": "HsacoFixtureAdd",
    }
    ukd["provenance"] = copy.deepcopy(_CONTRACT)
    return ukd


_OVERLAPPING_TABLE = {
    "schemaVersion": 1,
    "generics": {
        "gfx11-generic": ["gfx1100", "gfx1101"],
        "gfx11-b-generic": ["gfx1101", "gfx1102"],
    },
}

# (kdp arch, UKD form: inline|standalone|rocke, UKD arch, table override)
_REJECTED = {
    "generic_beside_its_member": ([GENERIC, MEMBER_B], "inline", None, None),
    "two_generics_sharing_a_member": (
        ["gfx11-generic", "gfx11-b-generic"],
        "inline",
        None,
        _OVERLAPPING_TABLE,
    ),
    "unknown_generic": (["gfx99-generic"], "inline", None, None),
    "inline_narrower_than_generic_kdp": ([GENERIC], "inline", [MEMBER_B], None),
    "inline_unknown_generic": ([GENERIC], "inline", ["gfx99-generic"], None),
    "inline_generic_beside_member": ([GENERIC], "inline", [GENERIC, MEMBER_B], None),
    "mixed_list_ukd_missing_the_generic": (
        ["gfx942", GENERIC],
        "inline",
        ["gfx942"],
        None,
    ),
    "standalone_without_arch_under_concrete_kdp": (
        ["gfx942"],
        "standalone",
        None,
        None,
    ),
    "generic_ukd_under_empty_arch_kdp": ([], "inline", [GENERIC], None),
    "generic_ukd_under_kdp_not_listing_it": ([MEMBER_B], "inline", [GENERIC], None),
    "rocke_inheriting_a_generic": ([GENERIC], "rocke", None, None),
}

_ACCEPTED = {
    "standalone_listing_the_generic": ([GENERIC], "standalone", [GENERIC]),
    "standalone_without_arch_under_arch_less_kdp": ([], "standalone", None),
    "inline_inheriting_a_concrete_kdp": (["gfx942"], "inline", None),
    "inline_inheriting_a_generic_kdp": ([GENERIC], "inline", None),
    "mixed_list_ukd_listing_every_entry": (
        ["gfx942", GENERIC],
        "inline",
        ["gfx942", GENERIC],
    ),
}


def _arch_rule_root(tmp_path, empty_arch_fixture, kdp_arch, form, ukd_arch):
    if form == "standalone":
        return _root(
            tmp_path,
            empty_arch_fixture,
            {"g": _kdp("kdp-g", kdp_arch, ["ukd-s"])},
            {"s": _embedded("ukd-s", ukd_arch)},
        )
    ukd = (
        _rocke_ukd("ukd-i", ukd_arch)
        if form == "rocke"
        else _embedded("ukd-i", ukd_arch)
    )
    return _root(tmp_path, empty_arch_fixture, {"g": _kdp("kdp-g", kdp_arch, [ukd])})


@pytest.mark.quick
@pytest.mark.parametrize("case", sorted(_REJECTED))
def test_an_arch_rule_violation_is_an_error(tmp_path, empty_arch_fixture, case):
    kdp_arch, form, ukd_arch, table_doc = _REJECTED[case]
    root = _arch_rule_root(tmp_path, empty_arch_fixture, kdp_arch, form, ukd_arch)
    table = GENERIC_TARGETS
    if table_doc is not None:
        table_path = tmp_path / "table.json"
        _write(table_path, table_doc)
        table = GenericTargets.load(table_path)
    with pytest.raises(HkpPackError):
        _load(root, table)


@pytest.mark.quick
@pytest.mark.parametrize("case", sorted(_ACCEPTED))
def test_a_consistent_arch_declaration_loads(tmp_path, empty_arch_fixture, case):
    kdp_arch, form, ukd_arch = _ACCEPTED[case]
    _load(_arch_rule_root(tmp_path, empty_arch_fixture, kdp_arch, form, ukd_arch))


def _hip_generic_root(dest, empty_arch_fixture):
    shutil.copytree(empty_arch_fixture, dest)
    kdp_path = dest / "solo.kdp.json"
    doc = _read(kdp_path)
    doc["arch"] = [GENERIC]
    _write(kdp_path, doc)
    return dest


def test_compiled_agreement_holds_on_a_generic_copy(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    """A generic copy's literal arch names no device; `consumer_records` must still count it."""
    from hkp_pack.desk_check import compiled_agreement

    root = _hip_generic_root(tmp_path / "src", empty_arch_fixture)
    _pack(root, tmp_path, [MEMBER_A], rocm_kpack_dir, hipcc=hipcc)
    failures, _unclaimed, _verified = compiled_agreement(
        tmp_path / "out" / MEMBER_A / "solo.kdp.json", GENERIC_TARGETS, rocm_kpack_dir
    )
    assert failures == []


def test_generic_hip_archive_is_keyed_by_the_generic_spelling_and_byte_identical_across_builds(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir, monkeypatch
):
    """Separate per-arch builds (own roots, cwd, source copy) yield identical generic files."""
    builds = {}
    for label, arches in (
        ("only_a", [MEMBER_A]),
        ("only_b", [MEMBER_B]),
        ("both", [MEMBER_A, MEMBER_B]),
    ):
        base = tmp_path / label
        base.mkdir()
        root = _hip_generic_root(base / "deeper" / label / "src", empty_arch_fixture)
        (base / "cwd").mkdir()
        monkeypatch.chdir(base / "cwd")
        results = _pack(root, base, arches, rocm_kpack_dir, hipcc=hipcc)
        if label == "only_a":
            shipped = _read(base / "out" / MEMBER_A / "solo.kdp.json")
            ks = shipped["kernelDescriptors"][0]["kernel_source"]
            assert shipped["arch"] == [GENERIC]
            assert ks["library"] == f"kpack/hip_kernel_provider_{GENERIC}.kpack"
            archive = _load_kpack(rocm_kpack_dir).PackedKernelArchive.read(
                base / "out" / MEMBER_A / ks["library"]
            )
            assert archive.get_kernel(ks["toc_key"], GENERIC) is not None
            assert archive.get_kernel(ks["toc_key"], MEMBER_A) is None
            assert results[MEMBER_A].kpack_path is None
        for member in arches:
            builds[(label, member)] = _files(base / "out" / member)

    reference_key, reference = next(iter(builds.items()))
    assert any(name.endswith(".kpack") for name in reference)
    assert any(name.endswith(".kdp.json") for name in reference)
    for key, files in builds.items():
        assert files == reference, f"{key} differs from {reference_key}"
