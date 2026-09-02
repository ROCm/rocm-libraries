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
import os
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


def _run(source_root, tmp_path, hipcc, rocm_kpack_dir, arches, source_label=None):
    """Pack one root. A root holding an `embedded_source` descriptor needs a label."""
    return run_pipeline(
        source_root=source_root,
        arches=list(arches),
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
        source_label=source_label,
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


def test_non_hkp_failure_still_leaves_no_partial_tree(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir, monkeypatch
):
    """Staging must protect the output even when no cleanup handler runs.

    run_pipeline's `except HkpPackError` tidies up after an expected failure, so
    it alone makes an in-place write look safe. It does not run for a
    MemoryError, a TypeError from a bug, or a SIGKILL -- and pack_arch creates
    <out>/kpack/ before it validates anything. Only staging-then-rename makes
    the output directory safe against a failure nobody caught, which is the
    actual reason to do it.
    """
    from hkp_pack import pipeline

    root = tmp_path / "root"
    _nest(root, "hip/pointwise", main_fixture)

    def crash(flat, inter, out_arch_dir, *a, **kw):
        # Create the output dir the way pack_arch does, then die in a way
        # run_pipeline does not catch.
        Path(out_arch_dir / "kpack").mkdir(parents=True, exist_ok=True)
        raise RuntimeError("uncaught failure mid-pack")

    monkeypatch.setattr(pipeline, "pack_arch", crash)

    out_root = tmp_path / "out"
    with pytest.raises(RuntimeError, match="uncaught failure"):
        pipeline.run_pipeline(
            source_root=root,
            arches=[ARCH],
            out_root=out_root,
            hipcc=hipcc,
            rocm_kpack_dir=rocm_kpack_dir,
            inter_root=tmp_path / "inter",
        )

    # The shipped path must not exist. A staging directory may survive -- it is
    # never installed, and leaving it aids debugging -- but <out>/<arch> must be
    # absent so install(DIRECTORY ... OPTIONAL) skips the arch entirely.
    assert not (
        out_root / ARCH
    ).exists(), "an uncaught failure left a partial arch tree that install() would ship"


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


# --- F. Toolchain provenance ------------------------------------------------
def test_provenance_records_the_toolchain_that_built_each_kernel(
    tmp_path, hipcc, rocm_kpack_dir, rocke_available
):
    """Authored fields say what was asked for; these say what answered.

    Without them two builds of byte-identical descriptors are indistinguishable
    after the fact, even though a hipcc, comgr, or rocKE wheel change may be the
    whole difference between them.
    """
    stamp = tmp_path / "wheels.sha256"
    stamp.write_text("deadbeefcafe\n", encoding="utf-8")

    run_pipeline(
        source_root=EXAMPLE_ROOT,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
        rocke_wheel_stamp=stamp,
    )

    by_kind = {}
    for kdp in (tmp_path / "out" / ARCH).rglob("*.kdp.json"):
        for ukd in _read(kdp)["kernelDescriptors"]:
            if isinstance(ukd, str):
                continue
            by_kind[ukd["provenance"]["origin_kind"]] = ukd["provenance"]

    # hip records the compiler that ran.
    assert "hipcc_version" in by_kind["hip"]
    # rocke records the comgr that was actually LOADED (not merely requested --
    # rocke falls through an unloadable override silently) and the wheel digest
    # the build keyed its staleness on.
    assert by_kind["rocke"]["rocke_wheel_sha256"] == "deadbeefcafe"
    assert by_kind["rocke"]["comgr_path"]

    # Producer-specific fields must not bleed across.
    assert "hipcc_version" not in by_kind["rocke"]
    assert "comgr_path" not in by_kind["hip"]


@pytest.mark.quick
def test_wheel_digest_absent_stamp_is_not_fatal(tmp_path):
    """Provenance is a record, not a gate.

    A hip-only build has no wheel stamp at all; that must degrade to omitting
    the field rather than failing the pack.
    """
    from hkp_pack import toolchain

    toolchain.wheel_digest.cache_clear()
    assert toolchain.wheel_digest(None) is None
    assert toolchain.wheel_digest(tmp_path / "does-not-exist") is None


@pytest.mark.quick
def test_hipcc_version_probe_is_best_effort():
    from hkp_pack import toolchain

    toolchain.hipcc_version.cache_clear()
    assert toolchain.hipcc_version(None) is None
    assert toolchain.hipcc_version("/nonexistent/hipcc") is None


# --- G. The library field, resolved the way the runtime resolves it ---------
def _resolve_library_like_runtime(descriptor_path, library):
    """Mirror IngestorKernelCode.hpp's `originDirectory / library` join.

    originDirectory is the parent of the descriptor FILE (DescriptorLoader.hpp
    sets it from `path.parent_path()`), and the C++ applies weakly_canonical to
    the join. os.path.normpath is the equivalent for a path that need not exist.
    """
    return Path(os.path.normpath(Path(descriptor_path).parent / library))


def _assert_runtime_would_load(descriptor_path, library, tree_root):
    """Both halves of what the runtime does with `library`, not just one.

    Resolution and CONTAINMENT are separate rules and only the first was checked
    here. That gap is exactly how the packer and the guard shipped mutually
    incompatible behaviour with this suite green: the packer emitted `../..` for
    a nested descriptor, the guard refused anything leaving the descriptor's own
    directory, and a test that only asked "does the file exist" saw nothing wrong.

    So assert what the runtime asserts (IngestorKernelCode.hpp, KPACK branch):
    the join resolves to a real archive, AND it stays inside the descriptor TREE
    -- which is the boundary, not the descriptor's own folder.

    This is still a reimplementation; the authoritative check is the C++
    `TestPackedDescriptorLoad.PackedKernelsSatisfyTheRuntimeContainmentGuard`,
    which reads the loader's own fields. Keeping a Python copy is worth it only
    because it fails at pack time, where the packer's author is looking.
    """
    resolved = _resolve_library_like_runtime(descriptor_path, library)
    assert resolved.is_file(), (
        f"{descriptor_path} declares library={library!r}, which resolves to "
        f"{resolved} -- the runtime would fail to open it"
    )

    root = Path(os.path.normpath(tree_root))
    assert root == resolved or root in resolved.parents, (
        f"{descriptor_path} declares library={library!r}, which resolves to "
        f"{resolved} -- OUTSIDE the descriptor tree {root}. The runtime's "
        f"containment guard refuses this and the kernel never loads."
    )
    return resolved


def test_library_resolves_from_a_nested_descriptor(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    """A packed UKD's `library` must resolve against ITS OWN directory.

    The runtime joins originDirectory (the descriptor's parent) with `library`.
    Writing it arch-root-relative works only for a flat layout and silently
    breaks the moment a descriptor nests -- which path preservation made the
    normal case. The archive is written once per arch at the ARCH ROOT, so a
    nested descriptor has to climb back out to reach it.
    """
    root = tmp_path / "root"
    _nest(root, "hip/deep/deeper", main_fixture)

    run_pipeline(
        source_root=root,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )

    out = tmp_path / "out" / ARCH
    checked = 0
    for kdp in out.rglob("*.kdp.json"):
        for ukd in _read(kdp)["kernelDescriptors"]:
            if isinstance(ukd, str):
                continue
            ks = ukd["kernel_source"]
            if ks.get("kind") != "kpack":
                continue
            _assert_runtime_would_load(kdp, ks["library"], out)
            checked += 1
    assert checked, "no kpack UKD was produced, so nothing was actually asserted"


def test_library_resolves_for_a_flat_descriptor(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    # The flat case must keep working: it is the shape every pre-nesting
    # descriptor has, and the one the original implementation got right.
    root = tmp_path / "root"
    shutil.copytree(empty_arch_fixture, root)

    run_pipeline(
        source_root=root,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )

    out = tmp_path / "out" / ARCH
    kdp = out / "solo.kdp.json"
    ks = _read(kdp)["kernelDescriptors"][0]["kernel_source"]
    assert ks["library"] == f"kpack/hip_kernel_provider_{ARCH}.kpack"
    _assert_runtime_would_load(kdp, ks["library"], out)


@pytest.mark.quick
def test_authored_kpack_folder_is_rejected(tmp_path, empty_arch_fixture):
    """`kpack/` is where the archive lands; an authored folder cannot claim it.

    Descriptors placed there would be written into the reserved directory
    alongside the archive. Nothing corrupts today only because the archive is
    written last -- a write-order accident, not a guarantee.
    """
    root = tmp_path / "root"
    _nest(root, "kpack", empty_arch_fixture)

    with pytest.raises(HkpPackError, match="reserved"):
        load_flat_input(root)


@pytest.mark.quick
def test_kpack_folder_rejected_only_at_the_arch_root(tmp_path, empty_arch_fixture):
    # Only the FIRST path segment is reserved: the archive lives at
    # <arch>/kpack/, so hip/kpack/ is a different path and must stay legal.
    root = tmp_path / "root"
    _nest(root, "hip/kpack", empty_arch_fixture)

    flat = load_flat_input(root)
    assert {d.rel_dir.as_posix() for d in flat.kdps()} == {"hip/kpack"}


# --- H. The example tree must satisfy the RUNTIME's schema ------------------
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
RUNTIME_FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "src/engines/kernel_ingestor_engine/test_descriptors/integration/archive_fixture"
)


def _descriptor_files(root):
    return sorted(p for p in Path(root).rglob("*.json"))


@pytest.mark.quick
def test_example_tree_uses_the_runtime_descriptor_version():
    """Every descriptor must be major version 1, which is what the loader reads.

    `DescriptorLoader.hpp` gates each type on a major/minor and this build reads
    major 1 (UKD_VERSION_MAJOR). An earlier version of this tree was authored at
    "0.1" -- copied from tests/fixtures/, which is packer-only test data and
    never passes through the C++ loader -- so the whole tree was unloadable
    while every packer test still passed.
    """
    for path in _descriptor_files(EXAMPLE_ROOT):
        version = _read(path).get("version")
        assert (
            isinstance(version, str) and "." in version
        ), f"{path.name}: missing or malformed version {version!r}"
        assert version.split(".")[0] == "1", (
            f"{path.name}: version {version!r} is not major 1; the loader "
            "rejects it outright"
        )


@pytest.mark.quick
def test_example_tree_ids_are_uuids():
    """Descriptor ids are UUIDs; the loader cross-references packs by them."""
    for path in _descriptor_files(EXAMPLE_ROOT):
        did = _read(path).get("id")
        assert isinstance(did, str) and _UUID_RE.match(
            did
        ), f"{path.name}: id {did!r} is not a UUID"


@pytest.mark.quick
def test_example_tree_field_shape_matches_the_runtime_fixture():
    """Per descriptor type, carry the fields the runtime fixture carries.

    `integration/archive_fixture/` is the tree the C++ integration test actually loads
    and dispatches, so it is the authority on shape. Comparing against it catches an
    invented field set -- the failure that shipped here once already, where UDD
    had `grid`/`block`/`args` instead of `dispatch_symbol` and UMD had
    `criteria`/`nodes` instead of `match_symbol`.
    """
    if not RUNTIME_FIXTURE.is_dir():
        pytest.skip(f"runtime fixture not present at {RUNTIME_FIXTURE}")

    def shapes(root):
        out = {}
        for path in _descriptor_files(root):
            kind = path.name.split(".")[-2]
            out.setdefault(kind, set()).update(_read(path).keys())
        return out

    fixture = shapes(RUNTIME_FIXTURE)
    example = shapes(EXAMPLE_ROOT)

    for kind, required in fixture.items():
        if kind not in example:
            # The example need not exercise every descriptor type the fixture
            # does (it has no standalone UKD, for instance).
            continue
        missing = required - example[kind]
        assert not missing, (
            f"example {kind} descriptors omit {sorted(missing)}, which the "
            f"runtime fixture carries -- the loader will not resolve them"
        )


@pytest.mark.quick
def test_example_tree_native_symbols_are_registered():
    """Symbols the descriptors name must exist in a compiled native pack.

    A descriptor can only resolve to something the C++ side registered. Naming
    an unregistered symbol produces a tree that packs cleanly and then fails to
    dispatch -- the packer has no way to know the difference.
    """
    packs_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src/engines/kernel_ingestor_engine/packs"
    )
    if not packs_dir.is_dir():
        pytest.skip(f"native packs not present at {packs_dir}")

    registered = set()
    for cpp in packs_dir.glob("*.cpp"):
        registered.update(re.findall(r'"(hipkernel\.[\w.]+)"', cpp.read_text()))
    assert registered, "no native symbols found; the scan is broken, not the tree"

    named = set()
    for path in _descriptor_files(EXAMPLE_ROOT):
        doc = json.dumps(_read(path))
        named.update(re.findall(r'"(hipkernel\.[\w.]+)"', doc))

    unknown = named - registered
    assert not unknown, (
        f"example descriptors name unregistered native symbols {sorted(unknown)}; "
        f"registered: {sorted(registered)}"
    )


def test_library_resolves_for_a_nested_standalone_ukd(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    """The standalone-UKD branch of the library rule.

    A standalone UKD ships as its own file and anchors on its own directory --
    a different code path from an inline UKD, which ships inside its KDP and
    anchors on the KDP's.
    """
    root = tmp_path / "root"
    _nest(root, "hip/deep", main_fixture)

    run_pipeline(
        source_root=root,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )

    out = tmp_path / "out" / ARCH
    checked = 0
    for ukd_file in out.rglob("*.ukd.json"):
        doc = _read(ukd_file)
        ks = doc.get("kernel_source", {})
        if ks.get("kind") != "kpack":
            continue
        _assert_runtime_would_load(ukd_file, ks["library"], out)
        checked += 1
    assert checked, "no standalone kpack UKD shipped; the test asserted nothing"


def test_standalone_ukd_anchors_on_its_own_dir_not_the_kdps(
    tmp_path, main_fixture, hipcc, rocm_kpack_dir
):
    """A standalone UKD in a different folder from the KDP that references it.

    Standalone UKDs resolve by global id, not co-location, so the two may live
    apart. Pins both consequences of the rel_dir the UKD is packed with: the
    shipped file keeps its authored subpath, and its `library` resolves from
    that subpath.

    The path assertion is the load-bearing one. rel_dir drives placement and
    depth together, so anchoring on the KDP moves the file and recomputes the
    climb-out to match -- the library still resolves, consistently wrong, and
    only the path reveals it.
    """
    root = tmp_path / "root"
    _nest(root, "hip/packs", main_fixture)

    # Relocate the standalone UKD (and only it) into a sibling subtree.
    ukd_src = root / "hip/packs/pointwise_add_b128.ukd.json"
    assert ukd_src.is_file(), "fixture no longer ships a standalone UKD"
    ukd_dest = root / "hip/kernels/deep/pointwise_add_b128.ukd.json"
    ukd_dest.parent.mkdir(parents=True, exist_ok=True)
    ukd_src.rename(ukd_dest)
    # Its hip source is resolved relative to the descriptor that names it.
    shutil.copy2(
        root / "hip/packs" / _read(ukd_dest)["kernel_source"]["source"],
        ukd_dest.parent,
    )

    run_pipeline(
        source_root=root,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )

    out = tmp_path / "out" / ARCH
    shipped = out / "hip/kernels/deep/pointwise_add_b128.ukd.json"
    assert shipped.is_file(), (
        "the standalone UKD did not keep its authored subpath; shipped tree holds "
        f"{sorted(p.relative_to(out).as_posix() for p in out.rglob('*.ukd.json'))}"
    )

    ks = _read(shipped)["kernel_source"]
    assert ks["kind"] == "kpack"
    _assert_runtime_would_load(shipped, ks["library"], out)


@pytest.mark.quick
@pytest.mark.parametrize(
    "filename,mutate,expected",
    [
        # The loader's enums. Without the packer-side check each of these packs
        # cleanly and is rejected at load, dropping the matcher, then the pack
        # naming it, then the engine -- at a log level that is off by default.
        ("pointwise.umd.json", {"scope": "Kernel"}, "invalid scope"),
        ("shared.uhd.json", {"kind": "Native"}, "invalid kind"),
        (
            "pointwise.kmd.json",
            {"fields": [{"name": "block_size", "type": "integer"}]},
            "invalid type",
        ),
        # Required keys the loader demands.
        ("pointwise.udd.json", {"dispatch_symbol": None}, "dispatch_symbol"),
        ("pointwise.umd.json", {"match_symbol": None}, "match_symbol"),
    ],
)
def test_generic_descriptors_are_validated_against_the_loader_schema(
    tmp_path, main_fixture, filename, mutate, expected
):
    """KMD/UMD/UDD/UHD are checked at pack time, not just by the runtime.

    A `None` value in `mutate` means "delete this key".
    """
    root = tmp_path / "root"
    _nest(root, "hip", main_fixture)
    target = root / "hip" / filename
    doc = _read(target)
    for key, value in mutate.items():
        if value is None:
            doc.pop(key, None)
        else:
            doc[key] = value
    target.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    with pytest.raises(HkpPackError, match=expected):
        load_flat_input(root)


@pytest.mark.quick
@pytest.mark.parametrize("spelling", ["kpack", "KPACK", "Kpack"])
def test_reserved_kpack_folder_is_case_insensitive(
    tmp_path, empty_arch_fixture, spelling
):
    """Every spelling is reserved, because Windows cannot tell them apart.

    On Linux `KPACK/` and `kpack/` are distinct directories and coexist without
    colliding -- verified -- so a case-sensitive check would be correct here.
    But this tree is authored and consumed on Windows too, where they are the
    same directory and the collision returns. Rejecting all spellings keeps the
    rule identical on every platform and costs an author nothing.
    """
    root = tmp_path / "root"
    _nest(root, spelling, empty_arch_fixture)

    with pytest.raises(HkpPackError, match="reserved"):
        load_flat_input(root)


@pytest.mark.quick
def test_example_tree_cross_references_resolve_to_the_right_types():
    """Every id reference must exist AND name the correct descriptor kind.

    A dangling or mistyped reference is worse than a parse error: the tree loads
    and then fails to match, with a diagnostic that points at the runtime rather
    than at the descriptor that lied. Field-shape parity does not catch it.
    """
    by_id = {}
    for path in _descriptor_files(EXAMPLE_ROOT):
        doc = _read(path)
        by_id[doc["id"]] = (path.name.split(".")[-2], path.name)

    def expect(ref, kind, where):
        assert ref in by_id, f"{where}: reference {ref} resolves to nothing"
        actual = by_id[ref][0]
        assert (
            actual == kind
        ), f"{where}: reference {ref} is a .{actual}, expected a .{kind}"

    kdps = 0
    for path in EXAMPLE_ROOT.rglob("*.kdp.json"):
        doc = _read(path)
        for matcher in doc.get("matchers", []):
            expect(matcher, "umd", f"{path.name} matchers")
        expect(doc["engine"], "ued", f"{path.name} engine")
        expect(doc["dispatch"], "udd", f"{path.name} dispatch")
        kdps += 1

    ueds = 0
    for path in EXAMPLE_ROOT.rglob("*.ued.json"):
        doc = _read(path)
        expect(doc["heuristic"], "uhd", f"{path.name} heuristic")
        expect(doc["metadata"], "kmd", f"{path.name} metadata")
        ueds += 1

    assert kdps and ueds, "no references were checked; the walk is broken"


@pytest.mark.quick
def test_example_tree_metadata_matches_its_kmd_schema():
    """A UKD's metadata keys and types must match what its KMD declares.

    Another failure that loads cleanly and breaks at match time: the loader does
    not reconcile the two, so an undeclared key or a wrong type is silent until
    something tries to select on it.
    """
    schemas = {
        _read(p)["id"]: {f["name"]: f["type"] for f in _read(p).get("fields", [])}
        for p in EXAMPLE_ROOT.rglob("*.kmd.json")
    }
    engines = {
        _read(p)["id"]: _read(p)["metadata"] for p in EXAMPLE_ROOT.rglob("*.ued.json")
    }
    py_type = {"int": int, "string": str, "bool": bool, "float": float}

    checked = 0
    for path in EXAMPLE_ROOT.rglob("*.kdp.json"):
        doc = _read(path)
        schema = schemas[engines[doc["engine"]]]
        for ukd in doc["kernelDescriptors"]:
            if isinstance(ukd, str):
                continue
            for key, value in ukd.get("metadata", {}).items():
                assert (
                    key in schema
                ), f"{path.name}: metadata '{key}' is not declared by its KMD"
                expected = py_type.get(schema[key])
                assert expected is None or isinstance(
                    value, expected
                ), f"{path.name}: metadata '{key}'={value!r} is not {schema[key]}"
                checked += 1
    assert checked, "no metadata was checked; the walk is broken"


@pytest.mark.quick
def test_example_tree_ids_do_not_collide_with_other_shipped_trees():
    """Ids must be unique against every tree that could share a catalog.

    The example tree and any in-tree ingestor set can be loaded into one
    process. A duplicate id across them is a load-time rejection that would look
    like a bug in whichever tree loaded second.

    Each ingestor set is compared against the example only. The two pointwise
    sets share ids with each other by design: one engine, two dialects, two
    discovery roots that never merge.
    """

    def ids(root):
        root = Path(root)
        if not root.is_dir():
            return set()
        out = set()
        for path in root.rglob("*.json"):
            try:
                out.add(_read(path)["id"])
            except (KeyError, json.JSONDecodeError):
                continue
        return out

    provider = EXAMPLE_ROOT.parent.parent.parent
    example = ids(EXAMPLE_ROOT)
    assert len(example) == len(
        list(_descriptor_files(EXAMPLE_ROOT))
    ), "the example tree has duplicate ids within itself"
    descriptors = provider / "src/engines/kernel_ingestor_engine/test_descriptors"
    others = [
        descriptors / "shared/conv_fwd",
        descriptors / "unit/pointwise",
        descriptors / "integration/pointwise",
        descriptors / "integration/archive_fixture",
    ]
    assert any(
        ids(other) for other in others
    ), f"no ingestor descriptor ids found under {descriptors}"
    for other in others:
        clash = example & ids(other)
        assert not clash, (
            f"example ids collide with "
            f"{other.relative_to(descriptors).as_posix()}: {sorted(clash)}"
        )


# --- I. The embedded_source kind (quick, compile-free) ----------------------
_EMBEDDED_SOURCE = {
    "kind": "embedded_source",
    "source_file": "kernels/PointwiseAdd.cpp",
    "entry_point": "PointwiseAdd",
}


def _embedded_source_root(tmp_path, fixture, kernel_source):
    """Nest `fixture` under one child folder and set its inline UKD's source.

    The fixture carries exactly one inline UKD, so replacing its kernel_source
    puts the whole root on the kind under test.
    """
    root = tmp_path / "root"
    _nest(root, "pointwise", fixture)
    kdp = root / "pointwise" / "solo.kdp.json"
    doc = _read(kdp)
    doc["kernelDescriptors"][0]["kernel_source"] = kernel_source
    kdp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return root


@pytest.mark.quick
def test_embedded_source_root_loads(tmp_path, empty_arch_fixture):
    """The walk accepts the kind and leaves the block unmodified."""
    root = _embedded_source_root(tmp_path, empty_arch_fixture, dict(_EMBEDDED_SOURCE))

    flat = load_flat_input(root)
    kdps = list(flat.kdps())
    assert len(kdps) == 1
    ukd = kdps[0].doc["kernelDescriptors"][0]
    assert ukd["kernel_source"] == _EMBEDDED_SOURCE


@pytest.mark.quick
@pytest.mark.parametrize("missing", ["source_file", "entry_point"])
def test_embedded_source_requires_source_file_and_entry_point(
    tmp_path, empty_arch_fixture, missing
):
    kernel_source = dict(_EMBEDDED_SOURCE)
    kernel_source.pop(missing)
    root = _embedded_source_root(tmp_path, empty_arch_fixture, kernel_source)

    with pytest.raises(HkpPackError, match=missing):
        load_flat_input(root)


@pytest.mark.quick
def test_unhandled_kind_aborts_the_walk_and_lists_the_accepted_kinds(
    tmp_path, empty_arch_fixture
):
    """A kind no producer handles is an error, and the message says what is.

    A misspelling is the common case, so the diagnostic must let an author see
    the intended spelling next to theirs.
    """
    root = _embedded_source_root(
        tmp_path, empty_arch_fixture, dict(_EMBEDDED_SOURCE, kind="embedded_sources")
    )

    with pytest.raises(HkpPackError) as excinfo:
        load_flat_input(root)

    message = str(excinfo.value)
    assert "unsupported kind 'embedded_sources'" in message
    for kind in ("hip", "rocke", "hsaco", "kpack", "embedded_source"):
        assert f"'{kind}'" in message, f"the accepted-kind list omits {kind}"


@pytest.mark.quick
@pytest.mark.parametrize(
    "source_file",
    ["../shared/PointwiseAdd.cpp", "kernels/../kernels/PointwiseAdd.cpp", ".."],
)
def test_embedded_source_rejects_a_parent_segment(
    tmp_path, empty_arch_fixture, source_file
):
    """source_file is the embedded source's identity and is never normalised.

    Two spellings of one file would take two keys, so the file would be
    embedded twice.
    """
    root = _embedded_source_root(
        tmp_path, empty_arch_fixture, dict(_EMBEDDED_SOURCE, source_file=source_file)
    )

    with pytest.raises(HkpPackError, match=re.escape(source_file)):
        load_flat_input(root)


@pytest.mark.quick
@pytest.mark.parametrize(
    "source_file", ["/etc/PointwiseAdd.cpp", "C:/kernels/PointwiseAdd.cpp"]
)
def test_embedded_source_rejects_an_absolute_path(
    tmp_path, empty_arch_fixture, source_file
):
    """The emitted key must be the same string on every machine.

    An absolute path passes through the key computation unchanged, so it would
    name one machine's filesystem in a shipped descriptor.
    """
    root = _embedded_source_root(
        tmp_path, empty_arch_fixture, dict(_EMBEDDED_SOURCE, source_file=source_file)
    )

    with pytest.raises(HkpPackError, match=re.escape(source_file)):
        load_flat_input(root)


# --- J. Emitting embedded_source through pass-through and pruning -----------
_STANDALONE_ID = "ukd-solo-mul-f32-b64"
_STANDALONE_FILE = "solo_mul.ukd.json"
_STANDALONE_SOURCE = {
    "kind": "embedded_source",
    "source_file": "kernels/PointwiseMul.cpp",
    "entry_point": "PointwiseMul",
}
_GENERICS = (
    "solo.umd.json",
    "solo.ued.json",
    "solo.udd.json",
    "solo.kmd.json",
    "solo.uhd.json",
)
OTHER_ARCH = "gfx90a"
_LABEL = "solo_label"


def _expected_provenance(
    rel_dir,
    kernel_source,
    authored_arch=(),
    label=_LABEL,
    rewritten=("arch",),
):
    provenance = {
        "origin_kind": kernel_source["kind"],
        "source_label": label,
    }
    provenance.update(
        {
            "rel_dir": rel_dir,
            "source_file": kernel_source["source_file"],
            "authored_arch": list(authored_arch),
            "rewritten": list(rewritten),
        }
    )
    return provenance


def _make_embedded(folder, arch=None):
    """Put every kernel in a copied `empty_arch` folder on the embedded kind.

    The KDP keeps one inline UKD and gains a reference to a standalone UKD, so
    both authoring forms travel the pass-through path. `arch` is the authored
    KDP arch list; None authors the wildcard. The kernel sources move into a
    `kernels/` child, which the packer must not carry into a shard.
    """
    kernels = folder / "kernels"
    kernels.mkdir()
    (folder / "PointwiseAdd.cpp").rename(kernels / "PointwiseAdd.cpp")
    (kernels / "PointwiseMul.cpp").write_text("// PointwiseMul\n", encoding="utf-8")

    kdp_path = folder / "solo.kdp.json"
    kdp = _read(kdp_path)
    kdp["arch"] = [] if arch is None else list(arch)
    inline = kdp["kernelDescriptors"][0]
    inline["kernel_source"] = dict(_EMBEDDED_SOURCE)
    kdp["kernelDescriptors"] = [inline, _STANDALONE_ID]
    kdp_path.write_text(json.dumps(kdp, indent=2) + "\n", encoding="utf-8")

    (folder / _STANDALONE_FILE).write_text(
        json.dumps(
            {
                "version": "0.1",
                "id": _STANDALONE_ID,
                "name": "PointwiseMul f32 block64 (solo)",
                "kernel_source": dict(_STANDALONE_SOURCE),
                "metadata": {"dtype": "FLOAT", "block_size": 64},
                "priority": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return folder


def _embedded_root(tmp_path, fixture, arch=None):
    root = tmp_path / "root"
    _make_embedded(_nest(root, "pointwise", fixture), arch=arch)
    return root


def _add_wildcard_embedded_kdp(folder):
    """A second KDP on the same generics, wildcard arch, one inline UKD."""
    doc = _read(folder / "solo.kdp.json")
    doc["id"] = "kdp-solo-wild"
    doc["name"] = "Solo wildcard pack"
    doc["arch"] = []
    doc["kernelDescriptors"] = [
        {
            "version": "0.1",
            "id": "ukd-solo-wild-add-f32-b64",
            "name": "PointwiseAdd f32 block64 (wild)",
            "kernel_source": dict(_EMBEDDED_SOURCE),
            "metadata": {"dtype": "FLOAT", "block_size": 64},
            "priority": 0,
        }
    ]
    (folder / "solo_wild.kdp.json").write_text(
        json.dumps(doc, indent=2) + "\n", encoding="utf-8"
    )


def _pack_embedded(root, tmp_path, rocm_kpack_dir, arches, log=print, out="out"):
    """Pack a root that compiles nothing, so hipcc must never be invoked."""
    return run_pipeline(
        source_root=root,
        arches=list(arches),
        out_root=tmp_path / out,
        hipcc="hipcc-not-invoked",
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / f"inter-{out}",
        source_label=_LABEL,
        log=log,
    )


def test_embedded_source_shard_holds_the_authored_descriptors(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    """The shard carries the authored documents plus this shard's arch.

    The KDP and the standalone UKD are arch-stamped, they keep their authored
    kernel_source, and their provenance records what was authored. The generics
    are byte-identical to their files and carry no provenance.
    """
    root = _embedded_root(tmp_path, empty_arch_fixture)
    authored = root / "pointwise"

    _pack_embedded(root, tmp_path, rocm_kpack_dir, [ARCH])

    shard = tmp_path / "out" / ARCH / "pointwise"
    kdp = _read(shard / "solo.kdp.json")
    assert kdp["arch"] == [ARCH]
    inline = kdp["kernelDescriptors"][0]
    assert inline["arch"] == [ARCH]
    assert inline["kernel_source"] == _EMBEDDED_SOURCE
    assert inline["provenance"] == _expected_provenance("pointwise", _EMBEDDED_SOURCE)
    assert "provenance" not in inline["kernel_source"]
    assert kdp["kernelDescriptors"][1] == _STANDALONE_ID

    standalone = _read(shard / _STANDALONE_FILE)
    assert standalone["arch"] == [ARCH]
    assert standalone["kernel_source"] == _STANDALONE_SOURCE
    assert standalone["provenance"] == _expected_provenance(
        "pointwise", _STANDALONE_SOURCE
    )
    assert "provenance" not in standalone["kernel_source"]

    for name in _GENERICS:
        assert (shard / name).read_bytes() == (authored / name).read_bytes(), name
        assert "provenance" not in _read(shard / name), name

    # The authored KDP keeps the wildcard; only the emitted copy names an arch.
    assert _read(authored / "solo.kdp.json")["arch"] == []


def test_inline_embedded_ukd_is_narrowed_to_the_shard_arch(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    """An inline UKD must not reach past the arch of the KDP that holds it.

    A wildcard KDP admits a wider inline arch list, and the KDP narrows to the
    shard on emission. An inline list left wider makes the loader reject the
    whole KDP, so the emitted inline UKD names this shard alone.
    """
    root = _embedded_root(tmp_path, empty_arch_fixture)
    kdp_path = root / "pointwise" / "solo.kdp.json"
    kdp = _read(kdp_path)
    kdp["kernelDescriptors"][0]["arch"] = [ARCH, OTHER_ARCH]
    kdp_path.write_text(json.dumps(kdp, indent=2) + "\n", encoding="utf-8")

    _pack_embedded(root, tmp_path, rocm_kpack_dir, [ARCH])

    emitted = _read(tmp_path / "out" / ARCH / "pointwise" / "solo.kdp.json")
    assert emitted["arch"] == [ARCH]
    inline = emitted["kernelDescriptors"][0]
    assert inline["arch"] == [ARCH]
    assert inline["kernel_source"] == _EMBEDDED_SOURCE
    assert inline["provenance"]["authored_arch"] == [ARCH, OTHER_ARCH]
    # The authored list is untouched; only the emitted copy is narrowed.
    assert _read(kdp_path)["kernelDescriptors"][0]["arch"] == [ARCH, OTHER_ARCH]


def test_embedded_source_shard_writes_no_archive_and_no_sources(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    root = _embedded_root(tmp_path, empty_arch_fixture)

    results = _pack_embedded(root, tmp_path, rocm_kpack_dir, [ARCH])

    shard = tmp_path / "out" / ARCH
    assert not (shard / "kpack").exists()
    assert not list(shard.rglob("*.kpack"))
    assert results[ARCH].kpack_path is None
    assert not results[ARCH].skipped
    assert not (shard / "pointwise" / "kernels").exists()
    assert not list(shard.rglob("*.cpp"))


def test_embedded_source_generics_are_identical_across_shards(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    """Two copies of a generic that differ poison the catalogue entry.

    The loader deduplicates untagged descriptors by content equality, so every
    shard's copy must be byte-identical to every other shard's.
    """
    root = _embedded_root(tmp_path, empty_arch_fixture)
    authored = root / "pointwise"

    _pack_embedded(root, tmp_path, rocm_kpack_dir, [ARCH, OTHER_ARCH])

    first = tmp_path / "out" / ARCH / "pointwise"
    second = tmp_path / "out" / OTHER_ARCH / "pointwise"
    for name in _GENERICS:
        data = (authored / name).read_bytes()
        assert (first / name).read_bytes() == data, name
        assert (second / name).read_bytes() == data, name


def test_arch_narrowed_embedded_kdp_is_pruned_from_the_other_shard(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    """An authored arch list narrows which shards an embedded KDP reaches.

    The standalone UKD prunes with the only KDP that references it.
    """
    root = _embedded_root(tmp_path, empty_arch_fixture, arch=[ARCH])
    _add_wildcard_embedded_kdp(root / "pointwise")

    _pack_embedded(root, tmp_path, rocm_kpack_dir, [ARCH, OTHER_ARCH])

    narrowed = tmp_path / "out" / ARCH / "pointwise"
    assert (narrowed / "solo.kdp.json").is_file()
    assert (narrowed / "solo_wild.kdp.json").is_file()
    assert (narrowed / _STANDALONE_FILE).is_file()

    other = tmp_path / "out" / OTHER_ARCH / "pointwise"
    assert (other / "solo_wild.kdp.json").is_file()
    assert not (other / "solo.kdp.json").exists()
    assert not (other / _STANDALONE_FILE).exists()


def test_embedded_only_shard_is_emitted_and_logged(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    """A shard whose surviving KDPs compile nothing is still written."""
    root = _embedded_root(tmp_path, empty_arch_fixture)
    logs = []

    results = _pack_embedded(root, tmp_path, rocm_kpack_dir, [ARCH], log=logs.append)

    assert not results[ARCH].skipped
    assert f"no kernels for {ARCH}, skipping" not in logs
    assert (tmp_path / "out" / ARCH / "pointwise" / "solo.kdp.json").is_file()
    passed_through = [m for m in logs if "emitting kind 'embedded_source'" in m]
    assert len(passed_through) == 2, logs


def test_mixed_hip_and_embedded_source_root_packs_in_one_invocation(
    tmp_path, empty_arch_fixture, main_fixture, hipcc, rocm_kpack_dir
):
    """One invocation over a root holding both dialects.

    The hip half produces an archive and kpack descriptors; the embedded half
    keeps its authored kernel_source.
    """
    root = tmp_path / "root"
    _nest(root, "hip/pointwise", main_fixture)
    _make_embedded(_nest(root, "embedded/pointwise", empty_arch_fixture))

    _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH], source_label=_LABEL)

    out = tmp_path / "out" / ARCH
    assert (out / "kpack" / f"hip_kernel_provider_{ARCH}.kpack").is_file()
    hip_kdp = _read(out / "hip" / "pointwise" / "pointwise.kdp.json")
    assert hip_kdp["kernelDescriptors"][0]["kernel_source"]["kind"] == "kpack"

    embedded = out / "embedded" / "pointwise"
    emb_kdp = _read(embedded / "solo.kdp.json")
    assert emb_kdp["kernelDescriptors"][0]["kernel_source"] == _EMBEDDED_SOURCE
    assert emb_kdp["kernelDescriptors"][0]["provenance"]["source_label"] == _LABEL
    assert _read(embedded / _STANDALONE_FILE)["kernel_source"] == _STANDALONE_SOURCE
    assert not (embedded / "kpack").exists()


def _embedded_copy(root, sub, fixture, suffix=""):
    """Copy the fixture to `root/sub`, put it on the embedded kind, re-stem it.

    `suffix` re-stems every file name and every id, so two copies of one fixture
    coexist under one root.
    """
    dest = root / sub if sub else root
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fixture, dest)
    _make_embedded(dest)
    if suffix:
        for path in sorted(dest.glob("*.json")):
            text = path.read_text(encoding="utf-8").replace("solo", f"solo{suffix}")
            renamed = path.with_name(path.name.replace("solo", f"solo{suffix}", 1))
            renamed.write_text(text, encoding="utf-8")
            if renamed != path:
                path.unlink()
    return dest


def test_a_field_the_packer_left_alone_is_not_reported_as_rewritten(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    """A descriptor that needed no change says so.

    The block tells a reader what happened on the way here, so it must not
    claim a rewrite the packer did not make.
    """
    root = tmp_path / "root"
    folder = _embedded_copy(root, "", empty_arch_fixture)
    kdp_path = folder / "solo.kdp.json"
    kdp = _read(kdp_path)
    kdp["kernelDescriptors"][0]["arch"] = [ARCH]
    kdp_path.write_text(json.dumps(kdp, indent=2) + "\n", encoding="utf-8")

    _pack_embedded(root, tmp_path, rocm_kpack_dir, [ARCH])

    inline = _read(tmp_path / "out" / ARCH / "solo.kdp.json")["kernelDescriptors"][0]
    assert inline["provenance"]["rewritten"] == []


def test_embedded_source_shards_are_reproducible(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    """Two runs over one source tree write the same bytes.

    Provenance records nothing that varies by machine, time or invocation.
    """
    root = tmp_path / "root"
    _embedded_copy(root, "", empty_arch_fixture)
    _embedded_copy(root, "deep/child", empty_arch_fixture, suffix="2")

    _pack_embedded(root, tmp_path, rocm_kpack_dir, [ARCH], out="out-first")
    _pack_embedded(root, tmp_path, rocm_kpack_dir, [ARCH], out="out-second")

    first = sorted((tmp_path / "out-first").rglob("*"))
    second = sorted((tmp_path / "out-second").rglob("*"))
    assert [p.relative_to(tmp_path / "out-first") for p in first] == [
        p.relative_to(tmp_path / "out-second") for p in second
    ]
    for left, right in zip(first, second):
        if left.is_file():
            assert left.read_bytes() == right.read_bytes(), left


def test_packing_without_a_source_label_is_refused(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    """Every pass-through descriptor records the build rule that packs it.

    The message names the descriptor, so an author of a hand-run sees which
    document the packer stopped on.
    """
    root = _embedded_root(tmp_path, empty_arch_fixture)

    with pytest.raises(HkpPackError) as excinfo:
        run_pipeline(
            source_root=root,
            arches=[ARCH],
            out_root=tmp_path / "out",
            hipcc="hipcc-not-invoked",
            rocm_kpack_dir=rocm_kpack_dir,
            inter_root=tmp_path / "inter",
        )

    message = str(excinfo.value)
    assert "source_label is required" in message
    assert "--source-label" in message
    assert "pointwise/kernels/PointwiseAdd.cpp" in message
