"""A loader-valid multi-launch rocKE bundle, packed with the real compiler.

The fixtures under tests/fixtures are authored the way a shipped engine's are:
UUID ids, version 1.0 documents, a `hipkernel:` engine with `graph_match`, and a UKD
at 1.1 carrying `sources`. Packing them needs rocke and libamd_comgr; a machine
without them fails here (conftest `rocke_available`) rather than skipping.

`HKP_PACK_OUTPUT_DIR`, when set, receives each packed tree under its fixture's name,
for a later run of `hipdnn_validate_descriptors --expect-engine hipkernel:MultiLaunch`.
"""

import os
import shutil
from pathlib import Path

import pytest

from pack_helpers import _run_cli, read_shipped, run_pipeline

ARCH = "gfx1151"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _pack(fixture, tmp_path, hipcc, rocm_kpack_dir):
    override = os.environ.get("HKP_PACK_OUTPUT_DIR")
    out = Path(override) / fixture if override else tmp_path / "out"
    if out.exists():
        shutil.rmtree(out)
    run_pipeline(
        source_root=FIXTURES / fixture,
        arches=[ARCH],
        out_root=out,
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )
    return out


@pytest.mark.parametrize(
    "fixture, launches, distinct_symbols",
    [
        ("multi_launch_decode_merge", {"decode", "merge"}, 2),
        ("multi_launch_same_kernel_twice", {"first", "second"}, 1),
    ],
)
def test_a_multi_launch_bundle_packs_and_passes_the_full_desk_check(
    tmp_path,
    hipcc,
    rocm_kpack_dir,
    rocke_available,
    fixture,
    launches,
    distinct_symbols,
):
    out = _pack(fixture, tmp_path, hipcc, rocm_kpack_dir)
    kdp = out / ARCH / "multi_launch.kdp.json"
    ukd = read_shipped(kdp)["kernelDescriptors"][0]
    assert set(ukd["sources"]) == launches
    assert all(ks["kind"] == "kpack" for ks in ukd["sources"].values())
    assert len({ks["symbol"] for ks in ukd["sources"].values()}) == distinct_symbols

    result = _run_cli(str(kdp), "--kpack-python-dir", rocm_kpack_dir, mode="full")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "compiled specialization agreement: OK for 1 kernel(s)" in result.stdout
