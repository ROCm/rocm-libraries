"""The rocKE producer's half of the nested-layout behaviour held in
`tests/test_hkp_pack_layout.py`: hip and rocKE descriptors coexisting under one
root and in one kpack per arch, and the committed example tree packed with both
producers.
"""

import hashlib

from hkp_pack.pipeline import run_pipeline
from test_hkp_pack_layout import (
    ARCH,
    EXAMPLE_ROOT,
    ROCKE_ARCH,
    _load_kpack,
    _nest,
    _read,
    _run,
)


# --- Mixed hip+rocke integration (comgr-gated) -----------------------------
def test_mixed_hip_rocke_one_kpack_per_arch(
    tmp_path, main_fixture, rocke_fixture, hipcc, rocm_kpack_dir, rocke_available
):
    # Two child folders under ONE root -> one kpack per arch holding BOTH kinds.
    # Producer selection is per-UKD on kernel_source.kind.
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


# --- The shipped example tree, both producers ------------------------------
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
    """The example tree deliberately reuses `shared.umd.json` across its two
    child folders. A flat packer drops one silently; path preservation keeps
    both.
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


# --- Toolchain provenance, both producers ----------------------------------
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
