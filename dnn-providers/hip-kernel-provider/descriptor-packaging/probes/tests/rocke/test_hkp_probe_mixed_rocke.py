"""derive -> real pack -> assert, for a root mixing rocke and hip UKDs, packed with the
rocKE toolchain. A build with HIPKERNELPROVIDER_ENABLE_ROCKE=OFF never collects this
directory; the rocKE-free equivalent is in test_hkp_probe_end_to_end.py."""

import json
import shutil

import pytest

from probe_support import (  # noqa: F401 (pytest fixtures)
    ROCKE_FIXTURE,
    comgr_lib,
    derive_and_pack,
    hip_source,
    hipcc,
    rocm_kpack_dir,
    run_assert,
)


@pytest.fixture(scope="module")
def mixed_probe(tmp_path_factory, rocm_kpack_dir, hipcc, comgr_lib):
    work = tmp_path_factory.mktemp("mixed_probe")
    src = work / "src"
    hip_source(src / "hip")
    shutil.copytree(ROCKE_FIXTURE, src / "rocke")
    return derive_and_pack(src, work, rocm_kpack_dir, hipcc, comgr_lib, rocke=True)


def test_mixed_root_passes_with_per_ukd_kinds(mixed_probe, rocm_kpack_dir, comgr_lib):
    out, expect = mixed_probe
    entries = json.loads(expect.read_text())
    assert {(e["kdp"], e["kind"]) for e in entries} == {
        ("hip/pointwise.kdp.json", "hip"),
        ("hip/pointwise_wild.kdp.json", "hip"),
        ("rocke/attention.kdp.json", "rocke"),
    }
    result = run_assert(out, expect, rocm_kpack_dir, expect_comgr=comgr_lib)
    assert result.returncode == 0, result.stderr


def test_mixed_root_origin_is_checked_per_ukd(mixed_probe, rocm_kpack_dir, tmp_path):
    # Expecting every UKD to be hip (a single global kind) must fail on the rocke one.
    out, expect = mixed_probe
    entries = [dict(e, kind="hip") for e in json.loads(expect.read_text())]
    wrong = tmp_path / "expect.json"
    wrong.write_text(json.dumps(entries))
    result = run_assert(out, wrong, rocm_kpack_dir)
    assert result.returncode == 1
    assert "FAIL provenance-origin:" in result.stderr
    assert "rocke/attention.kdp.json" in result.stderr
    assert "hip/pointwise" not in result.stderr
