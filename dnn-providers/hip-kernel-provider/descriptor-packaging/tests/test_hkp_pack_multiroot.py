import hashlib
import json
import re
import shutil

import pytest

from hkp_pack.descriptors import load_flat_inputs
from hkp_pack.errors import HkpPackError
from hkp_pack.hip_compile import hip_variant_key
from hkp_pack.pipeline import run_pipeline

ARCH = "gfx942"
ROCKE_ARCH = "gfx950"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_kpack(rocm_kpack_dir):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    return kpack


def _run(source_roots, tmp_path, hipcc, rocm_kpack_dir, arches):
    return run_pipeline(
        source_roots=list(source_roots),
        arches=list(arches),
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )


# --- A. Merge / origin (quick, comgr-free) ----------------------------------
@pytest.mark.quick
def test_merge_tags_per_descriptor_origin(main_fixture, empty_arch_fixture):
    flat = load_flat_inputs([main_fixture, empty_arch_fixture])
    roots = {d.origin_root for d in flat.descriptors}
    assert roots == {main_fixture, empty_arch_fixture}
    # Each descriptor's origin is exactly the folder it was globbed from.
    for d in flat.descriptors:
        assert d.path.parent == d.origin_root


@pytest.mark.quick
def test_origin_index_positional(main_fixture, empty_arch_fixture):
    # A single root leaves every descriptor at ordinal 0; merging two roots tags
    # each descriptor with its root's position in the passed order (0 then 1).
    single = load_flat_inputs([main_fixture])
    assert {d.origin_index for d in single.descriptors} == {0}

    merged = load_flat_inputs([main_fixture, empty_arch_fixture])
    by_index = {}
    for d in merged.descriptors:
        by_index.setdefault(d.origin_index, set()).add(d.origin_root)
    assert set(by_index) == {0, 1}
    assert by_index[0] == {main_fixture}
    assert by_index[1] == {empty_arch_fixture}


@pytest.mark.quick
def test_merge_runs_union_validation_once(main_fixture):
    # Loading the SAME root twice makes every id a cross-root duplicate, which the
    # union validation must reject (proving it runs over the merged set, not
    # per-root).
    with pytest.raises(HkpPackError, match="duplicate descriptor id"):
        load_flat_inputs([main_fixture, main_fixture])


@pytest.mark.quick
def test_cross_root_duplicate_id_rejected(tmp_path, main_fixture, empty_arch_fixture):
    # Two independently-valid folders that share a UKD id are rejected by the
    # whole-set validation: give the copied root's UKD an id already used by main.
    root2 = tmp_path / "dup_root"
    shutil.copytree(empty_arch_fixture, root2)
    kdp2 = root2 / "solo.kdp.json"
    doc = _read(kdp2)
    doc["kernelDescriptors"][0]["id"] = "ukd-copy-f32-b64"
    kdp2.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(
        HkpPackError, match="duplicate descriptor id 'ukd-copy-f32-b64'"
    ):
        load_flat_inputs([main_fixture, root2])


# --- B. Repeatable --source-root / per-origin hip resolution (real compile) --
def test_repeatable_source_root_merges_both(
    tmp_path, main_fixture, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    # Two hip roots, each with its own .cpp, pack into ONE kpack per arch.
    _run([main_fixture, empty_arch_fixture], tmp_path, hipcc, rocm_kpack_dir, [ARCH])
    out = tmp_path / "out" / ARCH
    # A KDP from each root (distinct filenames) ships in the single shard.
    km = _read(out / "copy.kdp.json")
    ks = _read(out / "solo.kdp.json")
    assert km["kernelDescriptors"][0]["kernel_source"]["kind"] == "kpack"
    assert ks["kernelDescriptors"][0]["kernel_source"]["kind"] == "kpack"
    # One kpack per arch holds both roots' kernels.
    assert (out / "kpack" / f"hip_kernel_provider_{ARCH}.kpack").exists()


def test_per_origin_cpp_resolves_against_own_root(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    """Two roots sharing source relpath + build + entry but differing in .cpp bytes
    ship two distinct blobs, proving the positional origin ordinal folds into the
    hip variant key (root 0 and root 1 key apart)."""
    # Seed two roots from one single-kernel fixture, then make rootB id- and
    # filename-distinct (so both survive the merged whole-set validation) while
    # keeping source/build/entry identical -- that shared relpath is the collision
    # surface. rootB's .cpp is then mutated so the two roots differ ONLY in bytes.
    rootA = tmp_path / "rootA"
    shutil.copytree(empty_arch_fixture, rootA)
    rootB = tmp_path / "rootB"
    shutil.copytree(empty_arch_fixture, rootB)
    for src in sorted(rootB.glob("solo.*")):
        dst = rootB / src.name.replace("solo.", "solob.", 1)
        dst.write_text(
            src.read_text(encoding="utf-8").replace("-solo", "-solob"),
            encoding="utf-8",
        )
        src.unlink()

    cpp_b = rootB / "PointwiseAdd.cpp"
    mutated, n = re.subn(
        r"a\[i\] \+ b\[i\]", "a[i] - b[i]", cpp_b.read_text(encoding="utf-8")
    )
    if n == 0:
        pytest.fail("seed kernel line changed; collision regex no longer applies")
    cpp_b.write_text(mutated, encoding="utf-8")
    # The mutation must actually change the bytes, else the origin-collision proof
    # would pass vacuously.
    assert (rootA / "PointwiseAdd.cpp").read_bytes() != cpp_b.read_bytes()

    ks_seed = _read(rootA / "solo.kdp.json")["kernelDescriptors"][0]["kernel_source"]
    source = ks_seed["source"]
    build = ks_seed["build"]

    # With the positional origin ordinal in the hip key each compiles to its OWN
    # blob; without it the second UKD would silently reuse the first's bytes.
    # rootA is passed first (ordinal 0), rootB second (ordinal 1).
    _run([rootA, rootB], tmp_path, hipcc, rocm_kpack_dir, [ARCH])
    out = tmp_path / "out" / ARCH
    ukd_a = _read(out / "solo.kdp.json")["kernelDescriptors"][0]["kernel_source"]
    ukd_b = _read(out / "solob.kdp.json")["kernelDescriptors"][0]["kernel_source"]
    tk_a, tk_b = ukd_a["toc_key"], ukd_b["toc_key"]
    # Each shipped UKD's toc_key is the ordinal-folded key for its own root, so the
    # two roots' identically-authored variants key apart.
    assert tk_a != tk_b
    assert tk_a == hip_variant_key(source, build, 0)
    assert tk_b == hip_variant_key(source, build, 1)

    kpack = _load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(
        out / "kpack" / f"hip_kernel_provider_{ARCH}.kpack"
    )
    # Two distinct toc_keys, two distinct stored blobs.
    assert len(archive.toc) == 2
    blob_a = bytes(archive.get_kernel(tk_a, ARCH))
    blob_b = bytes(archive.get_kernel(tk_b, ARCH))
    assert blob_a != blob_b, "cross-root same-name .cpp must ship two blobs"
    # Each shipped UKD maps to its OWN root's compiled bytes.
    assert hashlib.sha256(blob_a).hexdigest() == ukd_a["sha256"]
    assert hashlib.sha256(blob_b).hexdigest() == ukd_b["sha256"]


def test_single_root_key_path_independent(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    # The single-root key folds the positional ordinal ("root0"), NOT the
    # absolute path, so packing the SAME fixture from two differently-named parent
    # directories yields the SAME toc_key -- reproducible across build locations.
    ks_seed = _read(empty_arch_fixture / "solo.kdp.json")["kernelDescriptors"][0][
        "kernel_source"
    ]
    source = ks_seed["source"]
    build = ks_seed["build"]

    def _pack_from(parent_name):
        parent = tmp_path / parent_name
        parent.mkdir()
        root = parent / "src"
        shutil.copytree(empty_arch_fixture, root)
        run_pipeline(
            source_roots=[root],
            arches=[ARCH],
            out_root=parent / "out",
            hipcc=hipcc,
            rocm_kpack_dir=rocm_kpack_dir,
            inter_root=parent / "inter",
        )
        ukd = _read(parent / "out" / ARCH / "solo.kdp.json")["kernelDescriptors"][0]
        return ukd["kernel_source"]["toc_key"]

    tk_first = _pack_from("alpha_parent")
    tk_second = _pack_from("beta_parent_differently_named")
    # Same key from two different absolute locations: the ordinal, not the path,
    # is hashed.
    assert tk_first == tk_second
    # And it is exactly the single-root ordinal-0 key.
    assert tk_first == hip_variant_key(source, build, 0)


# --- C. Mixed hip+rocke integration (comgr-gated) ---------------------------
def test_mixed_hip_rocke_one_kpack_per_arch(
    tmp_path, main_fixture, rocke_fixture, hipcc, rocm_kpack_dir, rocke_available
):
    # Two real folders -> one kpack per arch holding BOTH kinds. On gfx950 the
    # main fixture contributes hip UKDs and the rocke fixture a rocke UKD.
    _run(
        [main_fixture, rocke_fixture],
        tmp_path,
        hipcc,
        rocm_kpack_dir,
        [ROCKE_ARCH],
    )
    out = tmp_path / "out" / ROCKE_ARCH
    kpack_path = out / "kpack" / f"hip_kernel_provider_{ROCKE_ARCH}.kpack"
    assert kpack_path.exists()
    kpack = _load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(kpack_path)

    # Gather every shipped UKD across both producers' descriptors.
    kinds = {}
    for kdp in out.glob("*.kdp.json"):
        for ukd in _read(kdp)["kernelDescriptors"]:
            if isinstance(ukd, str):
                continue
            ks = ukd["kernel_source"]
            if ks["kind"] != "kpack":
                continue
            prov = ukd["provenance"]
            kinds.setdefault(prov["origin_kind"], []).append((ks, prov))

    # Kind is asserted via provenance.origin_kind, NOT the filename (the kpack
    # name is a fixed group constant regardless of content).
    assert "hip" in kinds and "rocke" in kinds

    # Per-kind provenance isolation: hip carries {source,entry,build}; rocke
    # carries {source,builder,spec} side by side.
    for ks, prov in kinds["hip"]:
        assert set(("source", "entry", "build")).issubset(prov)
        assert "builder" not in prov and "spec" not in prov
    for ks, prov in kinds["rocke"]:
        assert set(("source", "builder", "spec")).issubset(prov)
        assert "entry" not in prov and "build" not in prov

    # Symbol-in-bytes + sha256 for each shipped UKD's own blob.
    for kind_ukds in kinds.values():
        for ks, _prov in kind_ukds:
            blob = archive.get_kernel(ks["toc_key"], ROCKE_ARCH)
            assert blob is not None
            assert hashlib.sha256(blob).hexdigest() == ks["sha256"]
            assert ks["symbol"].encode("ascii") in blob


# --- D. comgr self-diagnosing error (quick, comgr-free) ---------------------
@pytest.mark.quick
def test_comgr_error_names_loaded_lib(tmp_path, monkeypatch):
    import sys
    import textwrap

    from hkp_pack import rocke_compile

    base = tmp_path / "diagpkg"
    (base / "sub").mkdir(parents=True)
    (base / "__init__.py").write_text("", encoding="utf-8")
    (base / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (base / "sub" / "mod.py").write_text(
        textwrap.dedent(
            """
            import dataclasses

            @dataclasses.dataclass
            class StubSpec:
                n: int

            def build_stub(spec: StubSpec, *, arch="gfx950"):
                return ("kernel", spec, arch)
            """
        ),
        encoding="utf-8",
    )
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))

    class _FakeComgrError(Exception):
        pass

    def _boom(kernel, *, arch, capture_ir_text=False):
        raise _FakeComgrError("codegen exploded")

    monkeypatch.setattr(
        rocke_compile, "_load_compiler", lambda: (_boom, _FakeComgrError)
    )
    monkeypatch.setattr(
        rocke_compile, "_resolved_comgr_path", lambda: "/fake/path/amd_comgr.dll"
    )
    with pytest.raises(HkpPackError, match="comgr loaded from /fake/path") as excinfo:
        rocke_compile.compile_rocke_variant(
            tmp_path,
            "diagpkg/sub/mod.py",
            "build_stub",
            {"n": 1},
            "gfx950",
            tmp_path / "co",
        )
    assert "ROCKE_COMGR_LIB" in str(excinfo.value)
