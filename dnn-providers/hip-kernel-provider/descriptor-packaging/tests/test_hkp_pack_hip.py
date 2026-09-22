import hashlib
import json
import shutil

import pytest

from hkp_pack.hip_compile import hip_variant_key as variant_key
from hkp_pack.descriptors import load_flat_input, reachable_generic_ids
from hkp_pack.errors import HkpPackError
from hkp_pack.pipeline import run_pipeline

ARCHES = ["gfx942", "gfx950", "gfx90a"]


def _load_kpack(rocm_kpack_dir):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    return kpack


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inline_ukds(out_dir, kdp_name):
    # Only the inline (object) entries; standalone-UKD id refs are bare strings.
    return [
        u for u in _read(out_dir / kdp_name)["kernelDescriptors"] if isinstance(u, dict)
    ]


@pytest.fixture(scope="session")
def built(tmp_path_factory, main_fixture, hipcc, rocm_kpack_dir):
    """Compile + prune + pack the main fixture once for the 3-arch matrix."""
    base = tmp_path_factory.mktemp("built")
    results = run_pipeline(
        source_root=main_fixture,
        arches=ARCHES,
        out_root=base / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
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
            "pointwise_add_b128.ukd.json",
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
            "pointwise_add_b128.ukd.json",
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


def test_prn4_empty_arch_skip(tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir):
    logs = []
    results = run_pipeline(
        source_root=empty_arch_fixture,
        arches=["gfx942", "gfx950"],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
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
def test_byte_round_trip(built, rocm_kpack_dir):
    kpack = _load_kpack(rocm_kpack_dir)
    for arch in ARCHES:
        archive = kpack.PackedKernelArchive.read(
            built["out"] / arch / "kpack" / f"hip_kernel_provider_{arch}.kpack"
        )
        for kdp in (built["out"] / arch).glob("*.kdp.json"):
            for ukd in _read(kdp)["kernelDescriptors"]:
                if isinstance(ukd, str):
                    continue
                ks = ukd["kernel_source"]
                blob = archive.get_kernel(ks["toc_key"], arch)
                assert blob is not None
                assert hashlib.sha256(blob).hexdigest() == ks["sha256"]


def test_symbol_in_round_tripped_blob(built, rocm_kpack_dir):
    kpack = _load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(
        built["out"] / "gfx942" / "kpack" / "hip_kernel_provider_gfx942.kpack"
    )
    for kdp in (built["out"] / "gfx942").glob("*.kdp.json"):
        for ukd in _read(kdp)["kernelDescriptors"]:
            if isinstance(ukd, str):
                continue
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
    # The KDP file and each inline UKD carry their own version, both surviving
    # the rewrite.
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


def test_multi_kernel_stored_once(built, rocm_kpack_dir):
    ukds = _inline_ukds(built["out"] / "gfx942", "pointwise.kdp.json")
    add, mul = ukds[0], ukds[1]
    shared = add["kernel_source"]["toc_key"]
    assert shared == mul["kernel_source"]["toc_key"]
    assert add["kernel_source"]["sha256"] == mul["kernel_source"]["sha256"]
    assert add["kernel_source"]["symbol"] != mul["kernel_source"]["symbol"]
    kpack = _load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(
        built["out"] / "gfx942" / "kpack" / "hip_kernel_provider_gfx942.kpack"
    )
    assert archive.get_kernel(shared, "gfx942") is not None
    # The two UKDs collapse to one stored blob: the shared toc_key owns exactly
    # one TOC entry (one gfx942 ordinal), not one per UKD.
    entries = archive.toc[shared]
    assert list(entries) == ["gfx942"]
    # And overall the archive stores one blob per distinct toc_key, not per UKD.
    # gfx942 carries five UKDs (four distinct inline variants) plus one
    # standalone UKD (a fifth distinct variant), so a per-UKD duplication
    # regression would show up as more TOC entries than distinct toc_keys.
    all_toc_keys = set()
    for jp in (built["out"] / "gfx942").glob("*.json"):
        if jp.name == "kpack" or not (
            jp.name.endswith(".kdp.json") or jp.name.endswith(".ukd.json")
        ):
            continue
        doc = _read(jp)
        entries_doc = (
            doc["kernelDescriptors"] if jp.name.endswith(".kdp.json") else [doc]
        )
        for u in entries_doc:
            if isinstance(u, dict):
                all_toc_keys.add(u["kernel_source"]["toc_key"])
    assert len(archive.toc) == len(all_toc_keys)


def test_self_describing_ukd(built, rocm_kpack_dir):
    kpack = _load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(
        built["out"] / "gfx942" / "kpack" / "hip_kernel_provider_gfx942.kpack"
    )
    for ukd in _inline_ukds(built["out"] / "gfx942", "pointwise.kdp.json"):
        ks = ukd["kernel_source"]
        blob = archive.get_kernel(ks["toc_key"], "gfx942")
        assert hashlib.sha256(blob).hexdigest() == ks["sha256"]


# --- D. Negatives: compile-spec --------------------------------------------
def _run(source_root, tmp_path, hipcc, rocm_kpack_dir, arches=("gfx942",)):
    return run_pipeline(
        source_root=source_root,
        arches=list(arches),
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )


def test_neg_missing_source(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    (src / "Copy.cpp").unlink()
    with pytest.raises(HkpPackError, match="source not found"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir)


def test_neg_compile_failed(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    (src / "Copy.cpp").write_text("this is not valid hip source\n", encoding="utf-8")
    with pytest.raises(HkpPackError, match="compile failed"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir)


@pytest.mark.quick
def test_neg_malformed_build(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    doc["kernelDescriptors"][0]["kernel_source"]["build"] = {"defines": [1, 2, 3]}
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="invalid build"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir)


# --- D. Negatives: descriptor ----------------------------------------------
@pytest.mark.quick
def test_neg_malformed_json(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    (src / "copy.kdp.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(HkpPackError, match="malformed descriptor JSON"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir)


@pytest.mark.quick
def test_neg_missing_field(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    del doc["kernelDescriptors"][0]["priority"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="missing required field 'priority'"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir)


@pytest.mark.quick
def test_neg_dangling_id(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    doc["engine"] = "ued-does-not-exist"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(
        HkpPackError, match="unknown descriptor Id 'ued-does-not-exist'"
    ):
        _run(src, tmp_path, hipcc, rocm_kpack_dir)


def test_neg_sha256_mismatch(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
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
            rocm_kpack_dir=rocm_kpack_dir,
            inter_root=tmp_path / "inter",
            expected_sha256={key: "0" * 64},
        )


def test_neg_toc_key_collision(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir, monkeypatch
):
    src = _copy_fixture(tmp_path, main_fixture)
    from hkp_pack import pipeline

    # Force every variant to collapse to one toc_key while (source,build) stay
    # distinct -> the collision guard must hard-fail.
    monkeypatch.setattr(pipeline, "hip_variant_key", lambda source, build: "COLLIDE")
    from hkp_pack import hip_compile as compile_mod

    monkeypatch.setattr(
        compile_mod,
        "hip_variant_key",
        lambda source, build: "COLLIDE",
    )
    with pytest.raises(HkpPackError, match="toc_key collision"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir)


# --- D. CLI / arch-selection -----------------------------------------------
def test_cli1_single_arch(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    results = _run(main_fixture, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942"])
    assert (tmp_path / "out" / "gfx942").is_dir()
    assert not (tmp_path / "out" / "gfx950").exists()
    assert not (tmp_path / "out" / "gfx90a").exists()
    assert set(results) == {"gfx942"}


@pytest.mark.quick
def test_cli2_empty_gpu_targets(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    results = run_pipeline(
        source_root=main_fixture,
        arches=[],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )
    assert results == {}
    assert not (tmp_path / "out").exists()


# --- Unit: pruning reachability + wildcard ---------------------------------
@pytest.mark.quick
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
def test_determinism_same_variant_twice(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    # -fuse-cuid=none makes hipcc emit byte-identical .co for identical inputs, so
    # the sha256 stamped on each shipped UKD is stable across builds. Two full
    # runs of the same fixture must yield identical UKD sha256 values.
    def _shas(out):
        result = {}
        for kdp in sorted((out).glob("gfx942/*.kdp.json")):
            for ukd in _read(kdp)["kernelDescriptors"]:
                if isinstance(ukd, str):
                    continue
                result[ukd["id"]] = ukd["kernel_source"]["sha256"]
        return result

    run_pipeline(
        source_root=main_fixture,
        arches=["gfx942"],
        out_root=tmp_path / "out1",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter1",
    )
    run_pipeline(
        source_root=main_fixture,
        arches=["gfx942"],
        out_root=tmp_path / "out2",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter2",
    )
    a, b = _shas(tmp_path / "out1"), _shas(tmp_path / "out2")
    assert a and a == b


@pytest.mark.quick
def test_fuse_cuid_pinned_after_authored_flags():
    # The pinned -fuse-cuid=none is appended after the authored build flags, so
    # clang's last-flag-wins keeps it regardless of what the author wrote.
    from pathlib import Path

    from hkp_pack.hip_compile import _hipcc_command

    build = {"flags": ["-fuse-cuid=random", "-O3"]}
    cmd = _hipcc_command("hipcc", Path("src.cpp"), "gfx942", build, Path("out.co"))
    assert cmd.count("-fuse-cuid=none") == 1
    assert cmd.index("-fuse-cuid=none") > cmd.index("-fuse-cuid=random")


@pytest.mark.quick
def test_neg_authored_fuse_cuid_flag(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    # An authored -fuse-cuid flag is reserved and rejected at load.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    doc["kernelDescriptors"][0]["kernel_source"]["build"]["flags"] = [
        "-fuse-cuid=random"
    ]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="invalid build"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir)


# --- Non-descriptor .json warn/skip (C-007) --------------------------------
def test_non_descriptor_json_skipped(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    # A stray .json whose name carries no <type> token is skipped, not fatal.
    src = _copy_fixture(tmp_path, main_fixture)
    (src / "notes.json").write_text('{"arbitrary": true}\n', encoding="utf-8")
    results = _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942"])
    assert set(results) == {"gfx942"}
    assert not (tmp_path / "out" / "gfx942" / "notes.json").exists()


@pytest.mark.quick
def test_unknown_type_token_still_errors(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    # A file that IS type-tagged (<name>.<type>.json) but with an unrecognized
    # token still hard-errors; only token-less files are skipped.
    src = _copy_fixture(tmp_path, main_fixture)
    (src / "stray.bogus.json").write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(HkpPackError, match="unknown type token 'bogus'"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942"])


# --- E. Standalone UKD -----------------------------------------------------
_STANDALONE_UKD_FILE = "pointwise_add_b128.ukd.json"
_STANDALONE_UKD_ID = "ukd-pointwise-add-f32-b128"


def test_standalone_ukd_copied_to_both_arch_shards(built):
    # LOAD-BEARING: the wildcard standalone UKD referenced by the multi-arch
    # pointwise KDP ([gfx942, gfx950]) ships as its own <name>.ukd.json in BOTH
    # shards, same id, kpack-form, with per-arch details (different library +
    # sha256), each stamped with the single shard arch.
    u942 = _read(built["out"] / "gfx942" / _STANDALONE_UKD_FILE)
    u950 = _read(built["out"] / "gfx950" / _STANDALONE_UKD_FILE)
    assert u942["id"] == u950["id"] == _STANDALONE_UKD_ID
    # A shipped standalone UKD carries the single shard arch it was emitted for.
    assert u942["arch"] == ["gfx942"] and u950["arch"] == ["gfx950"]
    ks942, ks950 = u942["kernel_source"], u950["kernel_source"]
    assert ks942["kind"] == ks950["kind"] == "kpack"
    assert ks942["library"] == "kpack/hip_kernel_provider_gfx942.kpack"
    assert ks950["library"] == "kpack/hip_kernel_provider_gfx950.kpack"
    assert ks942["sha256"] != ks950["sha256"]


def test_standalone_ukd_matches_inline_kpack_shape(built):
    # A standalone UKD's shipped kernel_source is kpack-form with the same
    # structure as an inline UKD's (library/toc_key/symbol/sha256 + provenance).
    ukd = _read(built["out"] / "gfx942" / _STANDALONE_UKD_FILE)
    ks = ukd["kernel_source"]
    assert set(["kind", "library", "toc_key", "symbol", "sha256"]).issubset(ks)
    assert "file" not in ks and "build" not in ks
    assert ks["symbol"] == "PointwiseAdd"
    prov = ukd["provenance"]
    assert prov["origin_kind"] == "hip"
    assert prov["source"] == "PointwiseAdd.cpp"
    assert prov["entry"] == "PointwiseAdd"
    assert prov["build"]["defines"]["HIP_PLUGIN_POINTWISE_ADD_BLOCK_SIZE"] == 128
    # Authored top-level fields the tool does not model survive the rewrite.
    assert ukd["version"] == "0.1"


def test_standalone_ukd_blob_round_trips(built, rocm_kpack_dir):
    # The standalone UKD's kpack blob is really in each shard's archive and its
    # stamped sha256/symbol match the stored bytes.
    kpack = _load_kpack(rocm_kpack_dir)
    for arch in ("gfx942", "gfx950"):
        archive = kpack.PackedKernelArchive.read(
            built["out"] / arch / "kpack" / f"hip_kernel_provider_{arch}.kpack"
        )
        ks = _read(built["out"] / arch / _STANDALONE_UKD_FILE)["kernel_source"]
        blob = archive.get_kernel(ks["toc_key"], arch)
        assert blob is not None
        assert hashlib.sha256(blob).hexdigest() == ks["sha256"]
        assert ks["symbol"].encode("ascii") in blob


def test_heterogeneous_kdp_keeps_string_ref(built):
    # The KDP that references both inline UKD objects and the standalone id string
    # packs both: its shipped kernelDescriptors keeps the two inline objects (now
    # kpack-form) AND the bare string ref verbatim, in authored order.
    kds = _read(built["out"] / "gfx942" / "pointwise.kdp.json")["kernelDescriptors"]
    objs = [e for e in kds if isinstance(e, dict)]
    strs = [e for e in kds if isinstance(e, str)]
    assert len(objs) == 2
    assert strs == [_STANDALONE_UKD_ID]
    assert kds[-1] == _STANDALONE_UKD_ID  # authored order preserved (ref last)
    for o in objs:
        assert o["kernel_source"]["kind"] == "kpack"


def test_standalone_ukd_pruned_from_nonreferencing_shard(built):
    # gfx90a keeps only the wildcard KDP, which does not reference the standalone
    # UKD, so the standalone file is absent from that shard.
    assert not (built["out"] / "gfx90a" / _STANDALONE_UKD_FILE).exists()


@pytest.mark.quick
def test_standalone_ukd_dangling_id(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    # A KDP referencing a UKD id string with no matching .ukd.json hard-errors.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "pointwise.kdp.json"
    doc = _read(p)
    doc["kernelDescriptors"].append("ukd-does-not-exist")
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="unknown UKD Id 'ukd-does-not-exist'"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942"])


def test_standalone_ukd_shared_by_two_kdps_stored_once(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    # Two KDPs on the same arch both reference the one standalone UKD -> its blob
    # is stored once (deduped by variant_key), not once per referencing KDP.
    src = _copy_fixture(tmp_path, main_fixture)
    # copy.kdp.json is gfx942-only; add the standalone ref to it too, so both it
    # and pointwise.kdp.json reference the same standalone id on gfx942.
    p = src / "copy.kdp.json"
    doc = _read(p)
    doc["kernelDescriptors"].append(_STANDALONE_UKD_ID)
    p.write_text(json.dumps(doc), encoding="utf-8")
    _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942"])
    kpack = _load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(
        tmp_path / "out" / "gfx942" / "kpack" / "hip_kernel_provider_gfx942.kpack"
    )
    ukd = _read(tmp_path / "out" / "gfx942" / _STANDALONE_UKD_FILE)
    toc_key = ukd["kernel_source"]["toc_key"]
    # The shared standalone toc_key owns exactly one arch entry (one blob).
    assert list(archive.toc[toc_key]) == ["gfx942"]


def test_standalone_ukd_referenced_by_wildcard_kdp(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    # A standalone UKD referenced by a wildcard KDP (arch []) ships in every shard.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "pointwise_wild.kdp.json"
    doc = _read(p)
    doc["kernelDescriptors"].append(_STANDALONE_UKD_ID)
    p.write_text(json.dumps(doc), encoding="utf-8")
    _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942", "gfx90a"])
    for arch in ("gfx942", "gfx90a"):
        assert (tmp_path / "out" / arch / _STANDALONE_UKD_FILE).exists(), arch


@pytest.mark.quick
def test_standalone_ukd_id_collides_with_inline(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    # An inline UKD sharing an id with a standalone UKD is a hard error.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "pointwise.kdp.json"
    doc = _read(p)
    # Reuse the standalone id on one of the inline entries.
    doc["kernelDescriptors"][0]["id"] = _STANDALONE_UKD_ID
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="collides with a standalone UKD"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942"])


# --- F. Per-UKD arch ------------------------------------------------------
def test_per_ukd_arch_filters_standalone(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    # A standalone UKD narrowed to gfx942 ships only in the gfx942 shard; the
    # pointwise KDP still ships in gfx950 on its wildcard inline UKDs alone.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / _STANDALONE_UKD_FILE
    doc = _read(p)
    doc["arch"] = ["gfx942"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942", "gfx950"])
    assert (tmp_path / "out" / "gfx942" / _STANDALONE_UKD_FILE).exists()
    assert not (tmp_path / "out" / "gfx950" / _STANDALONE_UKD_FILE).exists()
    # The narrowed standalone drops out of gfx950, but its inline UKDs are
    # wildcard so the KDP itself still ships there.
    assert (tmp_path / "out" / "gfx950" / "pointwise.kdp.json").exists()
    kds = _read(tmp_path / "out" / "gfx950" / "pointwise.kdp.json")["kernelDescriptors"]
    assert _STANDALONE_UKD_ID not in kds
    # The gfx942 shard keeps the ref, and the shipped file is stamped gfx942.
    kds942 = _read(tmp_path / "out" / "gfx942" / "pointwise.kdp.json")[
        "kernelDescriptors"
    ]
    assert _STANDALONE_UKD_ID in kds942
    assert _read(tmp_path / "out" / "gfx942" / _STANDALONE_UKD_FILE)["arch"] == [
        "gfx942"
    ]


def test_every_shipped_ukd_stamped_with_shard_arch(built):
    # Every shipped UKD -- each inline UKD object in each *.kdp.json AND every
    # standalone *.ukd.json -- carries arch == [shard-arch] for its shard.
    for arch in ARCHES:
        shard = built["out"] / arch
        if not shard.is_dir():
            continue
        for kdp_path in shard.glob("*.kdp.json"):
            for ukd in _read(kdp_path)["kernelDescriptors"]:
                if isinstance(ukd, dict):
                    assert ukd["arch"] == [arch], (kdp_path.name, ukd.get("id"))
        for ukd_path in shard.glob("*.ukd.json"):
            assert _read(ukd_path)["arch"] == [arch], ukd_path.name


@pytest.mark.quick
def test_ukd_arch_not_subset_of_kdp_hard_errors(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    # A standalone UKD arch outside the referencing KDP's arch is a hard error
    # at load, independent of the requested GPU targets.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / _STANDALONE_UKD_FILE
    doc = _read(p)
    doc["arch"] = ["gfx90a"]  # pointwise KDP is [gfx942, gfx950]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="is not a subset of KDP"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942"])


def test_ukd_arch_subset_wildcard_combinations_accepted(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    # (a) UKD explicit subset of KDP explicit; (b) UKD wildcard + KDP explicit;
    # (c) KDP wildcard + UKD explicit. Each loads without error and emits the
    # standalone into the expected shard(s).
    # (a) explicit subset: b128 -> [gfx942] under pointwise KDP [gfx942, gfx950].
    src_a = _copy_fixture(tmp_path / "a", main_fixture)
    p = src_a / _STANDALONE_UKD_FILE
    doc = _read(p)
    doc["arch"] = ["gfx942"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    _run(src_a, tmp_path / "a", hipcc, rocm_kpack_dir, arches=["gfx942", "gfx950"])
    assert (tmp_path / "a" / "out" / "gfx942" / _STANDALONE_UKD_FILE).exists()
    assert not (tmp_path / "a" / "out" / "gfx950" / _STANDALONE_UKD_FILE).exists()

    # (b) UKD wildcard (unchanged fixture) under explicit KDP -> ships in both.
    src_b = _copy_fixture(tmp_path / "b", main_fixture)
    _run(src_b, tmp_path / "b", hipcc, rocm_kpack_dir, arches=["gfx942", "gfx950"])
    assert (tmp_path / "b" / "out" / "gfx942" / _STANDALONE_UKD_FILE).exists()
    assert (tmp_path / "b" / "out" / "gfx950" / _STANDALONE_UKD_FILE).exists()

    # (c) KDP wildcard + UKD explicit: reference b128 (arch [gfx942]) from the
    # wildcard KDP; it ships only in gfx942.
    src_c = _copy_fixture(tmp_path / "c", main_fixture)
    up = src_c / _STANDALONE_UKD_FILE
    udoc = _read(up)
    udoc["arch"] = ["gfx942"]
    up.write_text(json.dumps(udoc), encoding="utf-8")
    wp = src_c / "pointwise_wild.kdp.json"
    wdoc = _read(wp)
    wdoc["kernelDescriptors"].append(_STANDALONE_UKD_ID)
    wp.write_text(json.dumps(wdoc), encoding="utf-8")
    _run(src_c, tmp_path / "c", hipcc, rocm_kpack_dir, arches=["gfx942", "gfx90a"])
    assert (tmp_path / "c" / "out" / "gfx942" / _STANDALONE_UKD_FILE).exists()
    assert not (tmp_path / "c" / "out" / "gfx90a" / _STANDALONE_UKD_FILE).exists()


def test_empty_kdp_dropped_and_generics_pruned(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    # Widen copy.kdp to [gfx942, gfx950] but narrow its inline Copy UKD to
    # [gfx942]. On gfx950 the KDP empties out: it is dropped and its exclusive
    # generics prune away. On gfx942 the KDP and its whole chain survive.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    doc["arch"] = ["gfx942", "gfx950"]
    doc["kernelDescriptors"][0]["arch"] = ["gfx942"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942", "gfx950"])
    gone = (
        "copy.kdp.json",
        "copy.umd.json",
        "copy.ued.json",
        "copy.udd.json",
        "copy.kmd.json",
    )
    files950 = _arch_files(tmp_path / "out" / "gfx950")
    for name in gone:
        assert name not in files950, name
    files942 = _arch_files(tmp_path / "out" / "gfx942")
    for name in gone:
        assert name in files942, name


def test_wildcard_kdp_narrowed_per_kernel(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    # A wildcard KDP survives every arch, but a UKD it references narrowed to
    # gfx942 ships only there -- the KDP survives gfx90a on its wildcard inline
    # UKD while the narrowed standalone drops out.
    src = _copy_fixture(tmp_path, main_fixture)
    up = src / _STANDALONE_UKD_FILE
    udoc = _read(up)
    udoc["arch"] = ["gfx942"]
    up.write_text(json.dumps(udoc), encoding="utf-8")
    wp = src / "pointwise_wild.kdp.json"
    wdoc = _read(wp)
    wdoc["kernelDescriptors"].append(_STANDALONE_UKD_ID)
    wp.write_text(json.dumps(wdoc), encoding="utf-8")
    _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942", "gfx90a"])
    assert (tmp_path / "out" / "gfx942" / _STANDALONE_UKD_FILE).exists()
    assert not (tmp_path / "out" / "gfx90a" / _STANDALONE_UKD_FILE).exists()
    # The wildcard KDP itself survives both shards.
    assert (tmp_path / "out" / "gfx942" / "pointwise_wild.kdp.json").exists()
    assert (tmp_path / "out" / "gfx90a" / "pointwise_wild.kdp.json").exists()


@pytest.mark.quick
def test_multi_kdp_subset_violation_hard_errors(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    # Two KDPs reference the same standalone with incompatible arches: the UKD
    # arch is admissible under one referencing KDP but not the other, so the
    # per-referencing-KDP subset check fires.
    src = _copy_fixture(tmp_path, main_fixture)
    # Narrow the standalone to gfx950 (subset of pointwise KDP [gfx942, gfx950]).
    up = src / _STANDALONE_UKD_FILE
    udoc = _read(up)
    udoc["arch"] = ["gfx950"]
    up.write_text(json.dumps(udoc), encoding="utf-8")
    # copy.kdp is gfx942-only; make it reference the same standalone. gfx950 is
    # not a subset of [gfx942], so the check fails against copy.kdp.
    cp = src / "copy.kdp.json"
    cdoc = _read(cp)
    cdoc["kernelDescriptors"].append(_STANDALONE_UKD_ID)
    cp.write_text(json.dumps(cdoc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="is not a subset of KDP"):
        _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942"])


# --- G. Global id uniqueness -------------------------
@pytest.mark.quick
def test_neg_duplicate_standalone_ukd_ids(tmp_path, main_fixture):
    # Two standalone .ukd.json files sharing an id are rejected globally.
    src = _copy_fixture(tmp_path, main_fixture)
    dup = _read(src / _STANDALONE_UKD_FILE)
    (src / "dup.ukd.json").write_text(json.dumps(dup), encoding="utf-8")
    with pytest.raises(HkpPackError, match="duplicate"):
        load_flat_input(src)


@pytest.mark.quick
def test_neg_duplicate_kdp_ids(tmp_path, main_fixture):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    doc["id"] = "kdp-pointwise"  # already the id of pointwise.kdp.json
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="duplicate"):
        load_flat_input(src)


@pytest.mark.quick
def test_neg_duplicate_generic_ids(tmp_path, main_fixture):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kmd.json"
    doc = _read(p)
    doc["id"] = "kmd-pointwise"  # already the id of pointwise.kmd.json
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="duplicate"):
        load_flat_input(src)


@pytest.mark.quick
def test_neg_cross_type_id_reuse_rejected(tmp_path, main_fixture):
    # An id reused across two descriptor # types (here a UED taking a KMD's id)
    # is rejected at pack time.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "pointwise.ued.json"
    doc = _read(p)
    doc["id"] = "kmd-pointwise"  # a KMD id, different type
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="duplicate"):
        load_flat_input(src)


# --- H. Version enforcement --------------------------
@pytest.mark.quick
def test_neg_file_backed_missing_version(tmp_path, main_fixture):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kmd.json"
    doc = _read(p)
    del doc["version"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="missing required field 'version'"):
        load_flat_input(src)


@pytest.mark.quick
@pytest.mark.parametrize("bad", ["1", "1.x", "1.2.3"])
def test_neg_malformed_version(tmp_path, main_fixture, bad):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kmd.json"
    doc = _read(p)
    doc["version"] = bad
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="invalid version"):
        load_flat_input(src)


@pytest.mark.quick
def test_neg_inline_ukd_missing_version(tmp_path, main_fixture):
    # An inline UKD carries its own version; omitting it is an error.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    del doc["kernelDescriptors"][0]["version"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="missing required field 'version'"):
        load_flat_input(src)


# --- I. UED scoped name ------------------------------
@pytest.mark.quick
def test_neg_ued_name_not_scoped(tmp_path, main_fixture):
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "pointwise.ued.json"
    doc = _read(p)
    doc["name"] = "Pointwise engine"  # unscoped
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="scoped"):
        load_flat_input(src)


@pytest.mark.quick
def test_scoped_ued_name_loads_clean(main_fixture):
    # The renamed fixture UED (test_fixture:pointwise) validates without error.
    flat = load_flat_input(main_fixture)
    ued = next(d for d in flat.by_type("ued") if d.id == "ued-pointwise")
    assert ued.doc["name"] == "test_fixture:pointwise"


# --- J. Provenance protection ------------------------
def test_authored_provenance_cannot_hijack(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    # An authored top-level 'provenance' is dropped; the shipped block is the
    # generated traceability record, not the authored value.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / _STANDALONE_UKD_FILE
    doc = _read(p)
    doc["provenance"] = "HIJACKED"
    p.write_text(json.dumps(doc), encoding="utf-8")
    _run(src, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942"])
    prov = _read(tmp_path / "out" / "gfx942" / _STANDALONE_UKD_FILE)["provenance"]
    assert prov != "HIJACKED"
    assert prov["origin_kind"] == "hip"
    assert prov["source"] == "PointwiseAdd.cpp"
    assert prov["entry"] == "PointwiseAdd"


# --- K. Drop diagnostics + arch warning --------------
@pytest.mark.quick
def test_orphan_standalone_ukd_warns(tmp_path, main_fixture):
    # A standalone .ukd.json no KDP references is a non-fatal warning at load.
    src = _copy_fixture(tmp_path, main_fixture)
    orphan = _read(src / _STANDALONE_UKD_FILE)
    orphan["id"] = "ukd-orphan"
    (src / "orphan.ukd.json").write_text(json.dumps(orphan), encoding="utf-8")
    logs = []
    load_flat_input(src, log=logs.append)
    assert any("orphan.ukd.json" in m and "not referenced" in m for m in logs)


def test_empty_pruned_kdp_is_logged(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    # A KDP whose only UKD filters out for an arch is dropped with a log line.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    doc["arch"] = ["gfx942", "gfx950"]
    doc["kernelDescriptors"][0]["arch"] = ["gfx942"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    logs = []
    run_pipeline(
        source_root=src,
        arches=["gfx950"],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
        log=logs.append,
    )
    assert any("copy.kdp.json" in m and "filtered out for gfx950" in m for m in logs)


@pytest.mark.quick
def test_bare_arch_boundaries():
    # Mirrors the loader's isPlausibleArchBaseId. Bare gfx ids pass; a feature
    # suffix, an uppercase spelling, or a missing/non-alnum body is rejected.
    # 'gfx9-4-generic' is an LLVM generic target and must stay legal.
    from hkp_pack.descriptors import _reject_nonbare_arch

    for good in ("gfx90a", "gfx942", "gfx1100", "gfx1201", "gfx9-4-generic"):
        _reject_nonbare_arch([good], "where")
    for bad in ("gfx942:xnack-", "GFX942", "gfx", "gfx942 ", "gfx942+"):
        with pytest.raises(HkpPackError, match="is not usable"):
            _reject_nonbare_arch([bad], "where")


def test_nonbare_arch_is_rejected(tmp_path, main_fixture, hipcc, rocm_kpack_dir):
    # A feature-suffixed arch matches no shard, so the KDP prunes everywhere and
    # the pack would otherwise exit 0 having installed nothing.
    src = _copy_fixture(tmp_path, main_fixture)
    p = src / "copy.kdp.json"
    doc = _read(p)
    doc["arch"] = ["gfx942:xnack-"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="feature suffix"):
        run_pipeline(
            source_root=src,
            arches=["gfx942"],
            out_root=tmp_path / "out",
            hipcc=hipcc,
            rocm_kpack_dir=rocm_kpack_dir,
            inter_root=tmp_path / "inter",
        )
