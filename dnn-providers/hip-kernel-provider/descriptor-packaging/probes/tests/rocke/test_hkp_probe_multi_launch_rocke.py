"""derive -> real pack -> assert, for the loader-valid multi-launch rocke fixtures.

One fixture's launches name two kernels of one module; the other's two launches name
one kernel. Both are packed with the rocKE toolchain at gfx1151 and asserted launch by
launch. The slotted multi-launch fixture is derived and packed the same way: the derive
walk has no engine context, so it reads each slot's pack as any other. A build with
HIPKERNELPROVIDER_ENABLE_ROCKE=OFF never collects this directory.
"""

import json

import pytest

from probe_support import (  # noqa: F401 (pytest fixtures)
    MULTI_LAUNCH_FIXTURES,
    SLOTTED_MULTI_LAUNCH_FIXTURE,
    comgr_lib,
    derive_and_pack,
    hipcc,
    rocm_kpack_dir,
    run_assert,
)

ARCH = "gfx1151"


@pytest.fixture(
    params=MULTI_LAUNCH_FIXTURES, ids=[f.name for f in MULTI_LAUNCH_FIXTURES]
)
def probe(request, tmp_path, rocm_kpack_dir, hipcc, comgr_lib):
    out, expect = derive_and_pack(
        request.param, tmp_path, rocm_kpack_dir, hipcc, comgr_lib, rocke=True, arch=ARCH
    )
    return request.param.name, out, expect


def test_the_probe_expects_every_launch_and_the_pack_meets_it(
    probe, rocm_kpack_dir, comgr_lib
):
    fixture, out, expect = probe
    (entry,) = json.loads(expect.read_text())
    expected = {
        "multi_launch_decode_merge": {"decode": None, "merge": "merge"},
        "multi_launch_same_kernel_twice": {"first": None, "second": None},
    }[fixture]
    assert entry["launches"] == expected
    result = run_assert(out, expect, rocm_kpack_dir, expect_comgr=comgr_lib, arch=ARCH)
    assert result.returncode == 0, result.stderr


def test_the_probe_derives_every_slot_pack_and_the_pack_meets_it(
    tmp_path, rocm_kpack_dir, hipcc, comgr_lib
):
    out, expect = derive_and_pack(
        SLOTTED_MULTI_LAUNCH_FIXTURE,
        tmp_path,
        rocm_kpack_dir,
        hipcc,
        comgr_lib,
        rocke=True,
        arch=ARCH,
    )
    entries = {entry["kdp"]: entry for entry in json.loads(expect.read_text())}
    assert set(entries) == {"prep.kdp.json", "dkdv.kdp.json", "dq.kdp.json"}
    assert entries["dkdv.kdp.json"]["launches"] == {"decode": None, "merge": "merge"}
    assert "launches" not in entries["prep.kdp.json"]
    assert "launches" not in entries["dq.kdp.json"]
    result = run_assert(out, expect, rocm_kpack_dir, expect_comgr=comgr_lib, arch=ARCH)
    assert result.returncode == 0, result.stderr
