"""derive -> real pack -> assert, for a multi-launch (`sources`) hip UKD.

Two launches name two kernels of one translation unit built once; the packed UKD carries
no compiler evidence, so the assertion applies the structural per-launch checks alone.
Real HIP compiler compile.
"""

import json
import shutil

import pytest

from probe_support import (  # noqa: F401 (pytest fixtures)
    ARCH,
    comgr_lib,
    derive_and_pack,
    device_lib_path,
    hip_compiler,
    hip_path,
    hip_source,
    read_packed,
    rocm_kpack_dir,
    run_assert,
    write_packed,
)

KDP = "pointwise.kdp.json"
_BUILD = {
    "defines": {
        "HIP_PLUGIN_POINTWISE_ADD_TYPE": "float",
        "HIP_PLUGIN_POINTWISE_ADD_BLOCK_SIZE": 64,
    }
}


def _hip_sources_root(dst):
    """The hip fixture with pointwise.kdp.json's two UKDs fused into one `sources` UKD.

    The fused KDP gets a UDD of its own: the fixture's other pointwise KDPs stay
    single-launch, and one dispatch handler declares one launch set."""
    hip_source(dst)
    kdp = json.loads((dst / KDP).read_text())
    first = kdp["kernelDescriptors"][0]
    ukd = {k: first[k] for k in ("id", "name", "metadata", "priority", "provenance")}
    ukd["version"] = "1.1"
    ukd["sources"] = {
        launch: {
            "kind": "hip",
            "source": "PointwiseAdd.cpp",
            "entry": entry,
            "build": _BUILD,
        }
        for launch, entry in (("add", "PointwiseAdd"), ("mul", "PointwiseMul"))
    }
    kdp["kernelDescriptors"] = [ukd]
    udd = json.loads((dst / "pointwise.udd.json").read_text())
    udd.update(
        id="udd-pointwise-sources",
        name="Pointwise multi-launch dispatch",
        dispatch_symbol="hipkernel.pointwise_sources.dispatch",
    )
    (dst / "pointwise_sources.udd.json").write_text(json.dumps(udd, indent=2))
    kdp["dispatch"] = udd["id"]
    (dst / KDP).write_text(json.dumps(kdp, indent=2))
    return dst


@pytest.fixture(scope="module")
def hip_sources_probe(
    tmp_path_factory, rocm_kpack_dir, hip_compiler, hip_path, device_lib_path, comgr_lib
):
    work = tmp_path_factory.mktemp("hip_sources_probe")
    return derive_and_pack(
        _hip_sources_root(work / "src"),
        work,
        rocm_kpack_dir,
        hip_compiler,
        comgr_lib,
        hip_path=hip_path,
        device_lib_path=device_lib_path,
    )


def test_a_packed_hip_sources_ukd_passes_launch_by_launch(
    hip_sources_probe, rocm_kpack_dir
):
    out, expect = hip_sources_probe
    (entry,) = [e for e in json.loads(expect.read_text()) if e["kdp"] == KDP]
    assert entry["kind"] == "hip"
    assert entry["launches"] == {"add": None, "mul": None}
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr


def test_a_tampered_launch_signature_is_reported_by_name(
    hip_sources_probe, rocm_kpack_dir, tmp_path
):
    out, expect = hip_sources_probe
    tree = tmp_path / "out"
    shutil.copytree(out, tree)
    path = tree / ARCH / KDP
    doc = read_packed(path)
    (ukd,) = doc["kernelDescriptors"]
    assert ukd["sources"]["add"]["toc_key"] == ukd["sources"]["mul"]["toc_key"]
    ukd["sources"]["mul"]["signature"] = ukd["sources"]["add"]["signature"][:-1] + [
        {"kind": "by_value", "size": 4, "offset": 1024}
    ]
    write_packed(path, doc)
    result = run_assert(tree, expect, rocm_kpack_dir)
    assert result.returncode == 1, result.stderr
    assert any(
        "FAIL launch-signature:" in ln and "launch 'mul'" in ln
        for ln in result.stderr.splitlines()
    ), result.stderr
