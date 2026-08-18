import hashlib
import json
import shutil

import pytest

from hkp_pack.variant import variant_key
from hkp_pack.descriptors import load_flat_input, reachable_generic_ids
from hkp_pack.errors import HkpPackError
from hkp_pack.pipeline import run_pipeline

ARCHES = ["gfx942", "gfx950", "gfx90a"]


def _load_kpack(kpack_python_dir):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(kpack_python_dir)
    return kpack


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inline_ukds(out_dir, kdp_name):
    return _read(out_dir / kdp_name)["kernelDescriptors"]


@pytest.fixture(scope="session")
def built(tmp_path_factory, main_fixture, hipcc, kpack_python_dir):
    """Compile + prune + pack the main fixture once for the 3-arch matrix."""
    base = tmp_path_factory.mktemp("built")
    results = run_pipeline(
        source_root=main_fixture,
        arches=ARCHES,
        out_root=base / "out",
        hipcc=hipcc,
        kpack_python_dir=kpack_python_dir,
        inter_root=base / "inter",
    )
    return {
        "root": base,
        "out": base / "out",
        "inter": base / "inter",
        "results": results,
    }


def _copy_fixture(tmp_path, fixture):
    dst = tmp_path / "src"
    shutil.copytree(fixture, dst)
    return dst


# --- A. Intermediate (real compile) ----------------------------------------
def test_int1_compile_and_place(built):
    inter = built["inter"] / "gfx942"
    # PA-f32 (block64) and PA-f16 and PA-f32-256 and CP each land as <vk>.co.
    pa_f32 = variant_key(
        "PointwiseAdd.cpp",
        {
            "defines": {
                "HIP_PLUGIN_POINTWISE_ADD_TYPE": "float",
                "HIP_PLUGIN_POINTWISE_ADD_BLOCK_SIZE": 64,
            }
        },
    )
    assert (inter / f"{pa_f32}.co").is_file()
    # KDP rewritten hip -> hsaco with build carried top-level.
    kdp = _read(inter / "pointwise.kdp.json")
    ukd = kdp["kernelDescriptors"][0]
    assert ukd["kernel_source"]["kind"] == "hsaco"
    assert ukd["kernel_source"]["file"] == f"{pa_f32}.co"
    assert ukd["kernel_source"]["symbol"] == "PointwiseAdd"
    assert ukd["build"]["defines"]["HIP_PLUGIN_POINTWISE_ADD_TYPE"] == "float"
    # Generics copied through.
    assert (inter / "shared.uhd.json").is_file()


def test_int2_symbol_in_real_elf(built):
    # Read the intermediate .co compiled on disk (the pre-pack artifact),
    # not the packed/round-tripped blob: the symbols must be in the ELF hipcc
    # produced. The intermediate hsaco UKD names its .co via kernel_source.file.
    inter = built["inter"] / "gfx942"
    pa = _read(inter / "pointwise.kdp.json")["kernelDescriptors"][0]
    pa_co = (inter / pa["kernel_source"]["file"]).read_bytes()
    assert b"PointwiseAdd" in pa_co and b"PointwiseMul" in pa_co
    cp = _read(inter / "copy.kdp.json")["kernelDescriptors"][0]
    cp_co = (inter / cp["kernel_source"]["file"]).read_bytes()
    assert b"Copy" in cp_co


def test_int3_pre_prune_completeness(built):
    inter = built["inter"] / "gfx942"
    # gfx942 targets every KDP; all are present in the intermediate.
    for name in (
        "pointwise.kdp.json",
        "pointwise_half.kdp.json",
        "pointwise_wild.kdp.json",
        "copy.kdp.json",
    ):
        assert (inter / name).is_file()


# --- B. Pruning ------------------------------------------------------------
def _arch_files(out_dir):
    return {p.name for p in out_dir.glob("*.json")}


def test_prn1_mixed_prune_gfx950(built):
    files = _arch_files(built["out"] / "gfx950")
    # KDP-PH and KDP-C dropped; Copy chain generics pruned.
    for gone in (
        "pointwise_half.kdp.json",
        "copy.kdp.json",
        "copy.umd.json",
        "copy.ued.json",
        "copy.udd.json",
        "copy.kmd.json",
    ):
        assert gone not in files, gone
    # gfx942 retains all.
    files942 = _arch_files(built["out"] / "gfx942")
    for kept in ("pointwise_half.kdp.json", "copy.kdp.json", "copy.kmd.json"):
        assert kept in files942, kept


def test_prn2_no_over_prune_shared_uhd(built):
    files = _arch_files(built["out"] / "gfx950")
    # Pointwise chain survives; shared UHD-S survives (UED-P still refs it).
    for kept in (
        "pointwise.umd.json",
        "pointwise.ued.json",
        "pointwise.udd.json",
        "pointwise.kmd.json",
        "shared.uhd.json",
    ):
        assert kept in files, kept


def test_prn3_exact_post_prune_set(built):
    expected = {
        "gfx942": {
            "pointwise.kdp.json",
            "pointwise_half.kdp.json",
            "pointwise_wild.kdp.json",
            "copy.kdp.json",
            "pointwise.umd.json",
            "pointwise.ued.json",
            "pointwise.udd.json",
            "pointwise.kmd.json",
            "shared.uhd.json",
            "copy.umd.json",
            "copy.ued.json",
            "copy.udd.json",
            "copy.kmd.json",
        },
        "gfx950": {
            "pointwise.kdp.json",
            "pointwise_wild.kdp.json",
            "pointwise.umd.json",
            "pointwise.ued.json",
            "pointwise.udd.json",
            "pointwise.kmd.json",
            "shared.uhd.json",
        },
        "gfx90a": {
            "pointwise_wild.kdp.json",
            "pointwise.umd.json",
            "pointwise.ued.json",
            "pointwise.udd.json",
            "pointwise.kmd.json",
            "shared.uhd.json",
        },
    }
    for arch, want in expected.items():
        assert _arch_files(built["out"] / arch) == want, arch


def test_prn4_empty_arch_skip(tmp_path, empty_arch_fixture, hipcc, kpack_python_dir):
    logs = []
    results = run_pipeline(
        source_root=empty_arch_fixture,
        arches=["gfx942", "gfx950"],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        kpack_python_dir=kpack_python_dir,
        inter_root=tmp_path / "inter",
        log=logs.append,
    )
    assert results["gfx950"].skipped
    assert not (tmp_path / "out" / "gfx950").exists()
    assert "no kernels for gfx950, skipping" in logs
    assert (tmp_path / "out" / "gfx942").is_dir()


def test_prn5_wildcard_survives_gfx90a(built):
    files = _arch_files(built["out"] / "gfx90a")
    assert "pointwise_wild.kdp.json" in files
    # Only wildcard + pointwise chain; no explicit-arch KDP, no Copy chain.
    assert "pointwise.kdp.json" not in files
    assert "copy.kdp.json" not in files
    assert files  # non-empty


# --- C. Downstream kpack ---------------------------------------------------
def test_byte_round_trip(built, kpack_python_dir):
    kpack = _load_kpack(kpack_python_dir)
    for arch in ARCHES:
        archive = kpack.PackedKernelArchive.read(
            built["out"] / arch / "kpack" / f"hip_kernel_provider_{arch}.kpack"
        )
        for kdp in (built["out"] / arch).glob("*.kdp.json"):
            for ukd in _read(kdp)["kernelDescriptors"]:
                ks = ukd["kernel_source"]
                blob = archive.get_kernel(ks["toc_key"], arch)
                assert blob is not None
                assert hashlib.sha256(blob).hexdigest() == ks["sha256"]


def test_symbol_in_round_tripped_blob(built, kpack_python_dir):
    kpack = _load_kpack(kpack_python_dir)
    archive = kpack.PackedKernelArchive.read(
        built["out"] / "gfx942" / "kpack" / "hip_kernel_provider_gfx942.kpack"
    )
    for kdp in (built["out"] / "gfx942").glob("*.kdp.json"):
        for ukd in _read(kdp)["kernelDescriptors"]:
            ks = ukd["kernel_source"]
            blob = archive.get_kernel(ks["toc_key"], "gfx942")
            assert ks["symbol"].encode("ascii") in blob


def test_rewrite_kpack_form_and_provenance(built):
    kdp = _read(built["out"] / "gfx942" / "pointwise.kdp.json")
    ukd = kdp["kernelDescriptors"][0]
    ks = ukd["kernel_source"]
    assert ks["kind"] == "kpack"
    assert ks["library"] == "kpack/hip_kernel_provider_gfx942.kpack"
    assert "file" not in ks and "build" not in ks
    assert ks["toc_key"] and ks["sha256"] and ks["symbol"] == "PointwiseAdd"
    prov = ukd["provenance"]
    assert prov["origin_kind"] == "hip"
    assert prov["source"] == "PointwiseAdd.cpp"
    assert prov["entry"] == "PointwiseAdd"
    assert prov["build"]["defines"]["HIP_PLUGIN_POINTWISE_ADD_TYPE"] == "float"
    # metadata / priority preserved.
    assert ukd["metadata"] == {"dtype": "FLOAT", "block_size": 64}
    assert ukd["priority"] == 0
    # An authored top-level field the tool does not model (version) survives the
    # rewrite onto both the shipped KDP and its inline UKD.
    assert kdp["version"] == "0.1"
    assert ukd["version"] == "0.1"


def test_distinct_variant_storage(built):
    add = _inline_ukds(built["out"] / "gfx942", "pointwise.kdp.json")[0]
    half = _inline_ukds(built["out"] / "gfx942", "pointwise_half.kdp.json")[0]
    # Same symbol, different build -> distinct toc_key and distinct sha256.
    assert (
        add["kernel_source"]["symbol"]
        == half["kernel_source"]["symbol"]
        == "PointwiseAdd"
    )
    assert add["kernel_source"]["toc_key"] != half["kernel_source"]["toc_key"]
    assert add["kernel_source"]["sha256"] != half["kernel_source"]["sha256"]


def test_multi_kernel_stored_once(built, kpack_python_dir):
    ukds = _inline_ukds(built["out"] / "gfx942", "pointwise.kdp.json")
    add, mul = ukds[0], ukds[1]
    shared = add["kernel_source"]["toc_key"]
    assert shared == mul["kernel_source"]["toc_key"]
    assert add["kernel_source"]["sha256"] == mul["kernel_source"]["sha256"]
    assert add["kernel_source"]["symbol"] != mul["kernel_source"]["symbol"]
    kpack = _load_kpack(kpack_python_dir)
    archive = kpack.PackedKernelArchive.read(
        built["out"] / "gfx942" / "kpack" / "hip_kernel_provider_gfx942.kpack"
    )
    assert archive.get_kernel(shared, "gfx942") is not None
    # The two UKDs collapse to one stored blob: the shared toc_key owns exactly
    # one TOC entry (one gfx942 ordinal), not one per UKD.
    entries = archive.toc[shared]
    assert list(entries) == ["gfx942"]
    # And overall the archive stores one blob per distinct toc_key, not per UKD:
    # gfx942 carries five inline UKDs but only four distinct (source,build)
    # variants, so a per-UKD duplication regression would show up as > 4 entries.
    all_toc_keys = {
        u["kernel_source"]["toc_key"]
        for kdp in (built["out"] / "gfx942").glob("*.kdp.json")
        for u in _read(kdp)["kernelDescriptors"]
    }
    assert len(archive.toc) == len(all_toc_keys)


def test_self_describing_ukd(built, kpack_python_dir):
    kpack = _load_kpack(kpack_python_dir)
    archive = kpack.PackedKernelArchive.read(
        built["out"] / "gfx942" / "kpack" / "hip_kernel_provider_gfx942.kpack"
    )
    for ukd in _inline_ukds(built["out"] / "gfx942", "pointwise.kdp.json"):
        ks = ukd["kernel_source"]
        blob = archive.get_kernel(ks["toc_key"], "gfx942")
        assert hashlib.sha256(blob).hexdigest() == ks["sha256"]


# --- D. Negatives: compile-spec --------------------------------------------
def _run(source_root, tmp_path, hipcc, kpack_python_dir, arches=("gfx942",)):
    return run_pipeline(
        source_root=source_root,
        arches=list(arches),
        out_root=tmp_path / "out",
        hipcc=hipcc,
        kpack_python_dir=kpack_python_dir,
        inter_root=tmp_path / "inter",
    )


def test_neg_missing_source(tmp_path, main_fixture, hipcc, kpack_python_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    (src / "Copy.cpp").unlink()
    with pytest.raises(HkpPackError, match="source not found"):
        _run(src, tmp_path, hipcc, kpack_python_dir)


def test_neg_compile_failed(tmp_path, main_fixture, hipcc, kpack_python_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    (src / "Copy.cpp").write_text("this is not valid hip source\n", encoding="utf-8")
    with pytest.raises(HkpPackError, match="compile failed"):
        _run(src, tmp_path, hipcc, kpack_python_dir)


def test_neg_malformed_build(tmp_path, main_fixture, hipcc, kpack_python_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    doc["kernelDescriptors"][0]["kernel_source"]["build"] = {"defines": [1, 2, 3]}
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="invalid build"):
        _run(src, tmp_path, hipcc, kpack_python_dir)


# --- D. Negatives: descriptor ----------------------------------------------
def test_neg_malformed_json(tmp_path, main_fixture, hipcc, kpack_python_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    (src / "copy.kdp.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(HkpPackError, match="malformed descriptor JSON"):
        _run(src, tmp_path, hipcc, kpack_python_dir)


def test_neg_missing_field(tmp_path, main_fixture, hipcc, kpack_python_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    del doc["kernelDescriptors"][0]["priority"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="missing required field 'priority'"):
        _run(src, tmp_path, hipcc, kpack_python_dir)


def test_neg_dangling_id(tmp_path, main_fixture, hipcc, kpack_python_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    doc["engine"] = "ued-does-not-exist"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(
        HkpPackError, match="unknown descriptor Id 'ued-does-not-exist'"
    ):
        _run(src, tmp_path, hipcc, kpack_python_dir)


def test_neg_sha256_mismatch(tmp_path, main_fixture, hipcc, kpack_python_dir):
    build = {
        "defines": {
            "HIP_PLUGIN_COPY_TYPE": "float",
            "HIP_PLUGIN_COPY_BLOCK_SIZE": 64,
        }
    }
    key = variant_key("Copy.cpp", build)
    with pytest.raises(HkpPackError, match="sha256 mismatch"):
        # An expected digest that cannot match the freshly compiled blob.
        run_pipeline(
            source_root=main_fixture,
            arches=["gfx942"],
            out_root=tmp_path / "out",
            hipcc=hipcc,
            kpack_python_dir=kpack_python_dir,
            inter_root=tmp_path / "inter",
            expected_sha256={key: "0" * 64},
        )


def test_neg_toc_key_collision(
    tmp_path, main_fixture, hipcc, kpack_python_dir, monkeypatch
):
    src = _copy_fixture(tmp_path, main_fixture)
    from hkp_pack import pipeline

    # Force every variant to collapse to one toc_key while (source,build) stay
    # distinct -> the collision guard must hard-fail.
    monkeypatch.setattr(pipeline, "variant_key", lambda source, build: "COLLIDE")
    from hkp_pack import hip_compile as compile_mod

    monkeypatch.setattr(compile_mod, "variant_key", lambda source, build: "COLLIDE")
    with pytest.raises(HkpPackError, match="toc_key collision"):
        _run(src, tmp_path, hipcc, kpack_python_dir)


# --- D. CLI / arch-selection -----------------------------------------------
def test_cli1_single_arch(tmp_path, main_fixture, hipcc, kpack_python_dir):
    results = _run(main_fixture, tmp_path, hipcc, kpack_python_dir, arches=["gfx942"])
    assert (tmp_path / "out" / "gfx942").is_dir()
    assert not (tmp_path / "out" / "gfx950").exists()
    assert not (tmp_path / "out" / "gfx90a").exists()
    assert set(results) == {"gfx942"}


def test_cli2_empty_gpu_targets(tmp_path, main_fixture, hipcc, kpack_python_dir):
    results = run_pipeline(
        source_root=main_fixture,
        arches=[],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        kpack_python_dir=kpack_python_dir,
        inter_root=tmp_path / "inter",
    )
    assert results == {}
    assert not (tmp_path / "out").exists()


# --- Unit: pruning reachability + wildcard ---------------------------------
def test_wildcard_arch_matches(main_fixture):
    flat = load_flat_input(main_fixture)
    wild = next(k for k in flat.kdps() if k.id == "kdp-pointwise-wild")
    from hkp_pack.descriptors import arch_matches

    assert arch_matches(wild.doc, "gfx90a")
    assert arch_matches(wild.doc, "anything")
    explicit = next(k for k in flat.kdps() if k.id == "kdp-copy")
    assert arch_matches(explicit.doc, "gfx942")
    assert not arch_matches(explicit.doc, "gfx950")


# --- Per-shard arch narrowing (C-004 part 1) -------------------------------
def test_shard_kdp_arch_is_narrowed(built):
    # Every shipped KDP in a shard targets exactly that shard's arch, even when
    # the authored arch list spans several arches or is empty (wildcard).
    for arch in ("gfx942", "gfx950", "gfx90a"):
        shard = built["out"] / arch
        if not shard.is_dir():
            continue
        for kdp_path in shard.glob("*.kdp.json"):
            assert _read(kdp_path)["arch"] == [arch], kdp_path.name
    # The pointwise KDP authored [gfx942, gfx950] narrows in each shard.
    assert _read(built["out"] / "gfx942" / "pointwise.kdp.json")["arch"] == ["gfx942"]
    assert _read(built["out"] / "gfx950" / "pointwise.kdp.json")["arch"] == ["gfx950"]
    # The wildcard KDP (authored []) narrows to the shard arch wherever it lands.
    assert _read(built["out"] / "gfx90a" / "pointwise_wild.kdp.json")["arch"] == [
        "gfx90a"
    ]


# --- Cross-shard collision invariant (C-001) -------------------------------
def test_same_ukd_id_across_shards_distinct_content(built):
    # The pointwise KDP survives on both gfx942 and gfx950; the same UKD id ships
    # in both shards but with per-arch kpack details (different library + sha256).
    # This is the (id, arch) identity shape the runtime loader must accept without
    # treating the two as a global-id collision.
    add942 = _inline_ukds(built["out"] / "gfx942", "pointwise.kdp.json")[0]
    add950 = _inline_ukds(built["out"] / "gfx950", "pointwise.kdp.json")[0]
    assert add942["id"] == add950["id"]
    ks942, ks950 = add942["kernel_source"], add950["kernel_source"]
    assert ks942["library"] == "kpack/hip_kernel_provider_gfx942.kpack"
    assert ks950["library"] == "kpack/hip_kernel_provider_gfx950.kpack"
    assert ks942["sha256"] != ks950["sha256"]


# --- Compiler determinism (C-008) ------------------------------------------
def test_determinism_same_variant_twice(
    tmp_path, main_fixture, hipcc, kpack_python_dir
):
    # -fuse-cuid=none makes hipcc emit byte-identical .co for identical inputs, so
    # the sha256 stamped on each shipped UKD is stable across builds. Two full
    # runs of the same fixture must yield identical UKD sha256 values.
    def _shas(out):
        result = {}
        for kdp in sorted((out).glob("gfx942/*.kdp.json")):
            for ukd in _read(kdp)["kernelDescriptors"]:
                result[ukd["id"]] = ukd["kernel_source"]["sha256"]
        return result

    run_pipeline(
        source_root=main_fixture,
        arches=["gfx942"],
        out_root=tmp_path / "out1",
        hipcc=hipcc,
        kpack_python_dir=kpack_python_dir,
        inter_root=tmp_path / "inter1",
    )
    run_pipeline(
        source_root=main_fixture,
        arches=["gfx942"],
        out_root=tmp_path / "out2",
        hipcc=hipcc,
        kpack_python_dir=kpack_python_dir,
        inter_root=tmp_path / "inter2",
    )
    a, b = _shas(tmp_path / "out1"), _shas(tmp_path / "out2")
    assert a and a == b


# --- Non-descriptor .json warn/skip (C-007) --------------------------------
def test_non_descriptor_json_skipped(tmp_path, main_fixture, hipcc, kpack_python_dir):
    # A stray .json whose name carries no <type> token is skipped, not fatal.
    src = _copy_fixture(tmp_path, main_fixture)
    (src / "notes.json").write_text('{"arbitrary": true}\n', encoding="utf-8")
    results = _run(src, tmp_path, hipcc, kpack_python_dir, arches=["gfx942"])
    assert set(results) == {"gfx942"}
    assert not (tmp_path / "out" / "gfx942" / "notes.json").exists()


def test_unknown_type_token_still_errors(
    tmp_path, main_fixture, hipcc, kpack_python_dir
):
    # A file that IS type-tagged (<name>.<type>.json) but with an unrecognized
    # token still hard-errors; only token-less files are skipped.
    src = _copy_fixture(tmp_path, main_fixture)
    (src / "stray.bogus.json").write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(HkpPackError, match="unknown type token 'bogus'"):
        _run(src, tmp_path, hipcc, kpack_python_dir, arches=["gfx942"])
