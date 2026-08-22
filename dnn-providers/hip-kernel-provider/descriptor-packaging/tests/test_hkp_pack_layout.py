"""Nested-layout behaviour for the single authored source root.

There is exactly ONE source root. Child folders under it scope the content
(a `hip/` tree, a `rocKE/` tree, per-integration folders beneath those), and
each descriptor's authored subpath is preserved verbatim into the staged and
installed trees. Producer selection is per-UKD on `kernel_source.kind`, never
per-folder.

This file replaces the multi-root suite. The invariants that survived the
collapse are kept and re-expressed against one root: whole-set id validation,
descriptor-relative hip source resolution, hip+rocKE coexistence in one kpack,
and the comgr diagnostic.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from hkp_pack.descriptors import load_flat_input
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


def _run(source_root, tmp_path, hipcc, rocm_kpack_dir, arches):
    return run_pipeline(
        source_root=source_root,
        arches=list(arches),
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
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


# --- A. Recursive discovery and rel_dir (quick, compile-free) ---------------
@pytest.mark.quick
def test_discovery_is_recursive(tmp_path, main_fixture):
    # A flat glob finds nothing under a nested tree; the loader must descend.
    root = tmp_path / "root"
    _nest(root, "hip/pointwise", main_fixture)

    flat = load_flat_input(root)
    assert flat.descriptors, "nested descriptors must be discovered"
    for d in flat.descriptors:
        assert d.rel_dir.as_posix() == "hip/pointwise"


@pytest.mark.quick
def test_rel_dir_is_root_relative_parent(tmp_path, main_fixture, empty_arch_fixture):
    root = tmp_path / "root"
    _nest(root, "hip/a", main_fixture)
    _nest(root, "rocKE/b", empty_arch_fixture)

    flat = load_flat_input(root)
    by_rel = {}
    for d in flat.descriptors:
        by_rel.setdefault(d.rel_dir.as_posix(), set()).add(d.path.name)
    assert set(by_rel) == {"hip/a", "rocKE/b"}
    # rel_dir is exactly the parent, relative to the root.
    for d in flat.descriptors:
        assert (root / d.rel_dir / d.path.name) == d.path


@pytest.mark.quick
def test_same_filename_in_two_folders_both_survive(
    tmp_path, main_fixture, empty_arch_fixture
):
    """The collision the flat tool dropped silently.

    Two child folders may carry the same filename; distinct rel_dirs keep them
    apart. The in-tree ingestor corpus does exactly this with
    kernel_dtype_matches_graph.umd.json.
    """
    root = tmp_path / "root"
    a = _nest(root, "hip/a", empty_arch_fixture)
    b = _nest(root, "hip/b", empty_arch_fixture)
    # Same filenames in both folders, but distinct ids so the whole-set
    # validation is satisfied — filename reuse is the point of the test.
    _rename_ids(b, "solo", "solob")
    for src in sorted(b.glob("solob.*")):
        src.rename(b / src.name.replace("solob.", "solo.", 1))
    assert {p.name for p in a.glob("solo.*")} == {p.name for p in b.glob("solo.*")}

    flat = load_flat_input(root)
    kdp_paths = {(d.rel_dir.as_posix(), d.path.name) for d in flat.kdps()}
    assert ("hip/a", "solo.kdp.json") in kdp_paths
    assert ("hip/b", "solo.kdp.json") in kdp_paths


@pytest.mark.quick
def test_duplicate_id_across_folders_rejected(tmp_path, empty_arch_fixture):
    # Whole-set validation runs over the union of the tree, not per folder:
    # the same id in two child folders is still a duplicate.
    root = tmp_path / "root"
    _nest(root, "hip/a", empty_arch_fixture)
    _nest(root, "hip/b", empty_arch_fixture)

    with pytest.raises(HkpPackError, match="duplicate"):
        load_flat_input(root)


# --- B. Path-preserving output (real compile) -------------------------------
def test_output_mirrors_authored_subpath(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    root = tmp_path / "root"
    _nest(root, "hip/solo_add", empty_arch_fixture)

    _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])
    out = tmp_path / "out" / ARCH

    # The authored subpath is carried through verbatim...
    assert (out / "hip" / "solo_add" / "solo.kdp.json").is_file()
    # ...and the kpack stays arch-root-relative, one per arch, not per folder.
    assert (out / "kpack" / f"hip_kernel_provider_{ARCH}.kpack").is_file()
    assert not (out / "hip" / "solo_add" / "kpack").exists()


def test_hip_source_resolves_relative_to_its_descriptor(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    """Two folders, same source relpath and build, different .cpp bytes.

    Resolution is descriptor-relative, so each compiles its OWN neighbour file
    and the two ship distinct blobs. Under root-relative resolution both would
    bind to the same file and one kernel would silently ship the other's bytes.
    """
    root = tmp_path / "root"
    a = _nest(root, "hip/a", empty_arch_fixture)
    b = _nest(root, "hip/b", empty_arch_fixture)
    _rename_ids(b, "solo", "solob")

    cpp_b = b / "PointwiseAdd.cpp"
    mutated, n = re.subn(
        r"a\[i\] \+ b\[i\]", "a[i] - b[i]", cpp_b.read_text(encoding="utf-8")
    )
    if n == 0:
        pytest.fail(
            "seed kernel line changed; the byte-difference proof no longer applies"
        )
    cpp_b.write_text(mutated, encoding="utf-8")
    assert (a / "PointwiseAdd.cpp").read_bytes() != cpp_b.read_bytes()

    _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])
    out = tmp_path / "out" / ARCH
    ukd_a = _read(out / "hip" / "a" / "solo.kdp.json")["kernelDescriptors"][0]
    ukd_b = _read(out / "hip" / "b" / "solob.kdp.json")["kernelDescriptors"][0]
    ks_a, ks_b = ukd_a["kernel_source"], ukd_b["kernel_source"]

    kpack = _load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(
        out / "kpack" / f"hip_kernel_provider_{ARCH}.kpack"
    )
    blob_a = bytes(archive.get_kernel(ks_a["toc_key"], ARCH))
    blob_b = bytes(archive.get_kernel(ks_b["toc_key"], ARCH))
    assert blob_a != blob_b, "each descriptor must compile its own neighbour .cpp"
    assert hashlib.sha256(blob_a).hexdigest() == ks_a["sha256"]
    assert hashlib.sha256(blob_b).hexdigest() == ks_b["sha256"]


@pytest.mark.quick
def test_missing_descriptor_local_source_is_an_error(tmp_path, empty_arch_fixture):
    """No root-relative fallback.

    A descriptor naming a source it does not have beside it is an error, even
    when a same-named file exists at the root. Falling back would turn a typo
    into a silent bind to the wrong kernel.
    """
    from hkp_pack.hip_compile import compile_hip_variant

    root = tmp_path / "root"
    child = _nest(root, "hip/a", empty_arch_fixture)
    # Move the .cpp up to the root: root-relative resolution would find it.
    shutil.move(str(child / "PointwiseAdd.cpp"), str(root / "PointwiseAdd.cpp"))

    with pytest.raises(HkpPackError, match="source not found"):
        compile_hip_variant(
            "hipcc-not-invoked",
            root,
            "hip/a",
            "PointwiseAdd.cpp",
            {},
            ARCH,
            tmp_path / "co",
        )


@pytest.mark.quick
def test_source_escaping_the_root_is_rejected(tmp_path, empty_arch_fixture):
    from hkp_pack.hip_compile import compile_hip_variant

    root = tmp_path / "root"
    _nest(root, "hip/a", empty_arch_fixture)
    (tmp_path / "outside.cpp").write_text("// not ours\n", encoding="utf-8")

    with pytest.raises(HkpPackError, match="escapes the source root"):
        compile_hip_variant(
            "hipcc-not-invoked",
            root,
            "hip/a",
            "../../../outside.cpp",
            {},
            ARCH,
            tmp_path / "co",
        )


def test_variant_key_is_location_independent(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    # The key hashes the ROOT-RELATIVE source path and the build — never the
    # absolute path and never a root ordinal — so the same tree packs to the
    # same toc_key from any build location.
    ks_seed = _read(empty_arch_fixture / "solo.kdp.json")["kernelDescriptors"][0][
        "kernel_source"
    ]
    source = ks_seed["source"]
    build = ks_seed["build"]

    def _pack_from(parent_name):
        parent = tmp_path / parent_name
        parent.mkdir()
        root = parent / "src"
        _nest(root, "hip/solo_add", empty_arch_fixture)
        run_pipeline(
            source_root=root,
            arches=[ARCH],
            out_root=parent / "out",
            hipcc=hipcc,
            rocm_kpack_dir=rocm_kpack_dir,
            inter_root=parent / "inter",
        )
        ukd = _read(parent / "out" / ARCH / "hip" / "solo_add" / "solo.kdp.json")[
            "kernelDescriptors"
        ][0]
        return ukd["kernel_source"]["toc_key"]

    tk_first = _pack_from("alpha_parent")
    tk_second = _pack_from("beta_parent_differently_named")
    assert tk_first == tk_second
    # It is the key for the nested path, not the bare filename.
    assert tk_first == hip_variant_key(f"hip/solo_add/{source}", build)
    assert tk_first != hip_variant_key(source, build)


@pytest.mark.quick
def test_flat_layout_keys_match_pre_nesting(empty_arch_fixture):
    """A flat root keys exactly as it did before nesting existed.

    rel_dir is "." at the root, so hip_source_relpath is the identity on
    `source` and the payload is the original {source, build}. This is what makes
    the "hip single-root path preserved byte-for-byte" claim true for artifact
    keys, not just kernel bytes.
    """
    from hkp_pack.hip_compile import hip_source_relpath

    ks = _read(empty_arch_fixture / "solo.kdp.json")["kernelDescriptors"][0][
        "kernel_source"
    ]
    source, build = ks["source"], ks["build"]

    assert hip_source_relpath(".", source) == source
    assert hip_variant_key(hip_source_relpath(".", source), build) == hip_variant_key(
        source, build
    )


# --- C. Mixed hip+rocke integration (comgr-gated) ---------------------------
def test_mixed_hip_rocke_one_kpack_per_arch(
    tmp_path, main_fixture, rocke_fixture, hipcc, rocm_kpack_dir, rocke_available
):
    # Two child folders under ONE root -> one kpack per arch holding BOTH kinds.
    # This is the concrete demonstration that multi-root was never needed for
    # producer selection: the dispatch is per-UKD on kernel_source.kind.
    root = tmp_path / "root"
    _nest(root, "hip/pointwise", main_fixture)
    _nest(root, "rocKE/attention", rocke_fixture)

    _run(root, tmp_path, hipcc, rocm_kpack_dir, [ROCKE_ARCH])
    out = tmp_path / "out" / ROCKE_ARCH
    kpack_path = out / "kpack" / f"hip_kernel_provider_{ROCKE_ARCH}.kpack"
    assert kpack_path.exists()
    kpack = _load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(kpack_path)

    # Gather every shipped UKD across both producers' descriptors, anywhere in
    # the nested output tree.
    kinds = {}
    for kdp in out.rglob("*.kdp.json"):
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


# --- D. Per-arch atomicity and isolation ------------------------------------
def test_failed_arch_leaves_no_partial_tree(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir, monkeypatch
):
    """A failing arch must leave NO directory behind, not an empty one.

    pack_arch creates <out>/kpack/ before it validates anything, so an in-place
    write leaves a present-but-empty arch dir on failure. install(DIRECTORY ...
    OPTIONAL) skips only a MISSING directory, so that partial tree would install
    -- shipping an arch with no kernels in it.
    """
    from hkp_pack import pipeline

    root = tmp_path / "root"
    _nest(root, "hip/pointwise", main_fixture)

    real_pack = pipeline.pack_arch
    calls = {"n": 0}

    def flaky_pack(flat, inter, out_arch_dir, *a, **kw):
        calls["n"] += 1
        if inter.arch == "gfx950":
            # Fail AFTER pack_arch has created its output dir, which is what
            # makes the partial tree possible in the first place.
            Path(out_arch_dir / "kpack").mkdir(parents=True, exist_ok=True)
            raise HkpPackError("induced failure on gfx950")
        return real_pack(flat, inter, out_arch_dir, *a, **kw)

    monkeypatch.setattr(pipeline, "pack_arch", flaky_pack)

    out_root = tmp_path / "out"
    with pytest.raises(HkpPackError, match="gfx950"):
        pipeline.run_pipeline(
            source_root=root,
            arches=["gfx942", "gfx950"],
            out_root=out_root,
            hipcc=hipcc,
            rocm_kpack_dir=rocm_kpack_dir,
            inter_root=tmp_path / "inter",
        )

    # The good arch survived...
    assert (out_root / "gfx942" / "kpack").is_dir()
    assert any((out_root / "gfx942" / "kpack").iterdir())
    # ...and the failed one left nothing at all, not an empty shell.
    assert not (out_root / "gfx950").exists()
    # No staging residue either.
    assert not list(out_root.glob(".*staging"))


def test_failure_names_every_failed_arch(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir, monkeypatch
):
    """Exit non-zero listing which arches failed -- a silent 0 would hide it."""
    from hkp_pack import pipeline

    root = tmp_path / "root"
    _nest(root, "hip/pointwise", main_fixture)

    def always_fail(flat, inter, out_arch_dir, *a, **kw):
        raise HkpPackError(f"induced {inter.arch}")

    monkeypatch.setattr(pipeline, "pack_arch", always_fail)

    with pytest.raises(HkpPackError) as exc:
        pipeline.run_pipeline(
            source_root=root,
            arches=["gfx942", "gfx950"],
            out_root=tmp_path / "out",
            hipcc=hipcc,
            rocm_kpack_dir=rocm_kpack_dir,
            inter_root=tmp_path / "inter",
        )

    message = str(exc.value)
    assert "gfx942" in message and "gfx950" in message
    assert "2 of 2" in message


# --- E. The shipped example tree -------------------------------------------
EXAMPLE_ROOT = Path(__file__).resolve().parent.parent / "examples" / "descriptors"


def test_example_tree_packs_both_producers(
    tmp_path, hipcc, rocm_kpack_dir, rocke_available
):
    """The in-repo example tree must actually drive both producers end to end.

    This is the only thing in the repository that exercises the production path,
    which is how a silent-empty install and a silent descriptor drop both
    survived unnoticed. Packing the real committed tree -- not a fixture -- is
    what keeps it honest: if the example rots, this fails.
    """
    results = run_pipeline(
        source_root=EXAMPLE_ROOT,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )
    assert not results[ARCH].skipped

    out = tmp_path / "out" / ARCH
    kpack = out / "kpack" / f"hip_kernel_provider_{ARCH}.kpack"
    assert kpack.is_file()
    assert kpack.stat().st_size > 0, "an empty kpack is finding 3.9 reappearing"

    # Authored subpaths are preserved verbatim into the shipped tree.
    assert (out / "hip" / "pointwise_add" / "pointwise_add.kdp.json").is_file()
    assert (
        out / "rocKE" / "gfx942_tiled_attention" / "tiled_attention.kdp.json"
    ).is_file()

    # Both producers contributed, asserted via provenance rather than filename.
    kinds = set()
    for kdp in out.rglob("*.kdp.json"):
        for ukd in _read(kdp)["kernelDescriptors"]:
            if isinstance(ukd, str):
                continue
            kinds.add(ukd["provenance"]["origin_kind"])
    assert kinds == {"hip", "rocke"}


def test_example_tree_keeps_both_shared_filenames(
    tmp_path, hipcc, rocm_kpack_dir, rocke_available
):
    """Standing regression test for review 2.1.

    The example tree deliberately reuses `shared.umd.json` across its two child
    folders. A flat packer drops one silently; path preservation keeps both.
    """
    run_pipeline(
        source_root=EXAMPLE_ROOT,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )

    shared = sorted((tmp_path / "out" / ARCH).rglob("shared.umd.json"))
    assert len(shared) == 2, "both same-named descriptors must survive"
    assert len({_read(p)["id"] for p in shared}) == 2


@pytest.mark.quick
def test_example_tree_is_self_consistent():
    """Load-time validation of the committed tree, no toolchain required.

    Catches a broken example on any box, including one with neither hipcc nor
    comgr, so the tree cannot rot silently between full runs.
    """
    flat = load_flat_input(EXAMPLE_ROOT)

    ids = [d.id for d in flat.descriptors]
    assert len(ids) == len(set(ids)), "duplicate descriptor ids in the example"
    rel_dirs = {d.rel_dir.as_posix() for d in flat.kdps()}
    assert rel_dirs == {"hip/pointwise_add", "rocKE/gfx942_tiled_attention"}
