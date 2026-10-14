"""derive -> real pack -> assert, for hip-only and mixed rocke+hip roots.

The roots are built from the packer's own fixtures. Real hipcc and rocke compiles:
this is the proof the probe tools work for producers other than rocke and that the
KDP paths derive records are the ones the packed output carries.
"""

import json
import shutil

import pytest

from probe_support import (  # noqa: F401 (pytest fixtures)
    ARCH,
    EMPTY_ARCH_FIXTURE,
    HSACO_FIXTURES,
    ROCKE_FIXTURE,
    comgr_lib,
    derive_and_pack,
    hip_source,
    hipcc,
    rocm_kpack_dir,
    run_assert,
)

# What derive keeps from the main fixture for gfx950: pointwise.kdp.json (gfx942 and
# gfx950) holds two hip UKDs in one compile group and the wildcard KDP one more in
# another; the gfx942-only KDPs ship nothing.
_HIP_EXPECT = [
    {"kdp": "pointwise.kdp.json", "name": "PointwiseAdd f32 block64", "kind": "hip"},
    {
        "kdp": "pointwise_wild.kdp.json",
        "name": "PointwiseAdd f32 block256",
        "kind": "hip",
    },
]


def _hsaco_source(dst, arch=ARCH):
    """empty_arch's single UKD rewritten to an authored hsaco for `arch`."""
    shutil.copytree(EMPTY_ARCH_FIXTURE, dst)
    shutil.copyfile(HSACO_FIXTURES / arch / "HsacoFixture.co", dst / "HsacoFixture.co")
    kdp = json.loads((dst / "solo.kdp.json").read_text())
    kdp["arch"] = [arch]
    ukd = kdp["kernelDescriptors"][0]
    ukd["arch"] = [arch]
    ukd["kernel_source"] = {
        "kind": "hsaco",
        "file": "HsacoFixture.co",
        "symbol": "HsacoFixtureAdd",
    }
    (dst / "solo.kdp.json").write_text(json.dumps(kdp, indent=2))
    return dst


@pytest.fixture(scope="module")
def hip_probe(tmp_path_factory, rocm_kpack_dir, hipcc, comgr_lib):
    work = tmp_path_factory.mktemp("hip_probe")
    return derive_and_pack(
        hip_source(work / "src"), work, rocm_kpack_dir, hipcc, comgr_lib
    )


def test_hip_root_passes(hip_probe, rocm_kpack_dir):
    out, expect = hip_probe
    assert json.loads(expect.read_text()) == _HIP_EXPECT
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr


def test_rocke_off_probe_expects_and_packs_only_the_other_kinds(
    tmp_path, rocm_kpack_dir, hipcc, comgr_lib
):
    # A root holding hip UKDs beside a rocke KDP, probed as a build with rocKE OFF
    # does: the rocKE/ family folder excluded and the rocke kind disabled, with no
    # rocKE toolchain. Derive must expect exactly what that pack ships, or the probe
    # would fail on UKDs the build never packs.
    src = tmp_path / "src"
    hip_source(src / "hip")
    shutil.copytree(ROCKE_FIXTURE, src / "rocKE")
    shutil.copytree(ROCKE_FIXTURE, src / "other_rocke")
    filters = (("rocKE",), ("rocke",))
    out, expect = derive_and_pack(
        src, tmp_path, rocm_kpack_dir, hipcc, comgr_lib, filters=filters
    )
    assert {(e["kdp"], e["kind"]) for e in json.loads(expect.read_text())} == {
        ("hip/pointwise.kdp.json", "hip"),
        ("hip/pointwise_wild.kdp.json", "hip"),
    }
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr


def test_hsaco_root_passes(tmp_path, rocm_kpack_dir, hipcc, comgr_lib):
    src = _hsaco_source(tmp_path / "src")
    out, expect = derive_and_pack(src, tmp_path, rocm_kpack_dir, hipcc, comgr_lib)
    entries = json.loads(expect.read_text())
    assert [(e["kdp"], e["kind"]) for e in entries] == [("solo.kdp.json", "hsaco")]
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr


def test_hip_and_hsaco_mixed_root_passes(tmp_path, rocm_kpack_dir, hipcc, comgr_lib):
    src = tmp_path / "src"
    hip_source(src / "hip")
    _hsaco_source(src / "hsaco")
    out, expect = derive_and_pack(src, tmp_path, rocm_kpack_dir, hipcc, comgr_lib)
    kinds = {e["kdp"]: e["kind"] for e in json.loads(expect.read_text())}
    assert kinds == {
        "hip/pointwise.kdp.json": "hip",
        "hip/pointwise_wild.kdp.json": "hip",
        "hsaco/solo.kdp.json": "hsaco",
    }
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr


def test_hsaco_ukd_for_other_arch_is_not_packed_or_expected(
    tmp_path, rocm_kpack_dir, hipcc, comgr_lib
):
    # gfx942-only hsaco beside a gfx950 hip root: derive ignores it, the pack ships
    # nothing for it, and the assertion still passes on the hip UKDs alone.
    src = tmp_path / "src"
    hip_source(src / "hip")
    _hsaco_source(src / "hsaco", arch="gfx942")
    out, expect = derive_and_pack(src, tmp_path, rocm_kpack_dir, hipcc, comgr_lib)
    assert {e["kind"] for e in json.loads(expect.read_text())} == {"hip"}
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr


# --- UKDS: the probe packs exactly the listed UKDs ----------------------------
# Not the default pick of its group (that is "PointwiseAdd f32 block64"), and both in
# pointwise.kdp.json, so pointwise_wild.kdp.json keeps nothing and is left out.
_LISTED = ["PointwiseMul f32 block64", "PointwiseAdd f32 block64"]


def test_listed_ukds_pack_and_pass(tmp_path, rocm_kpack_dir, hipcc, comgr_lib):
    src = hip_source(tmp_path / "src")
    out, expect = derive_and_pack(
        src, tmp_path, rocm_kpack_dir, hipcc, comgr_lib, ukds=_LISTED
    )
    entries = json.loads(expect.read_text())
    assert sorted((e["kdp"], e["name"]) for e in entries) == sorted(
        ("pointwise.kdp.json", n) for n in _LISTED
    )
    assert not (out / ARCH / "pointwise_wild.kdp.json").exists()
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr
