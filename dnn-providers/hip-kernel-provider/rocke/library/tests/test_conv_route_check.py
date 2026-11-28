# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# benchmarks/common/conv_route_check.py: the accuracy gate both routes of an
# A/B go through, and the provenance every result row carries. Host-only.

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from benchmarks.common.conv_route_check import _check, _provenance  # noqa: E402


def _ref():
    return torch.linspace(-4.0, 4.0, 4096).reshape(1, 4, 4, 4, 64)


def test_check_passes_a_bf16_rounded_output():
    ref = _ref()
    r = _check(ref.to(torch.bfloat16), ref, torch)
    assert r["verify"] == "PASS" and r["nan"] == 0
    assert r["rel_err"] < 5e-2


def test_check_fails_an_unwritten_element():
    ref = _ref()
    out = ref.to(torch.bfloat16)
    out.view(-1)[17] = float("nan")
    r = _check(out, ref, torch)
    assert r["verify"] == "FAIL" and r["nan"] == 1


def test_check_fails_a_wrong_output():
    ref = _ref()
    r = _check((ref * 1.2).to(torch.bfloat16), ref, torch)
    assert r["verify"] == "FAIL" and r["rel_err"] > 5e-2


_KEYS = {
    "gfx_target",
    "device_target_id",
    "cache_arch",
    "llvm_flavor",
    "table_llvm_flavor",
    "comgr",
    "harness",
    "harness_rev",
    "argv",
    "timing",
}


@pytest.mark.parametrize("rounds", [0, 3])
def test_provenance_names_targets_toolchain_harness_and_method(rounds):
    a = SimpleNamespace(arch="gfx942", rounds=rounds, warmup=3, iters=20)
    argv = ["--routes", "t.json#deterministic", "--cache", "c", "--verify"]
    routes = SimpleNamespace(llvm_flavor="llvm22")
    prov = _provenance(a, argv, "gfx942:sramecc+:xnack-", routes)
    assert _KEYS <= set(prov)
    assert prov["gfx_target"] == prov["cache_arch"] == "gfx942"
    assert prov["device_target_id"] == "gfx942:sramecc+:xnack-"
    assert prov["table_llvm_flavor"] == "llvm22" and prov["llvm_flavor"]
    assert prov["harness"].endswith("conv_route_check.py")
    assert prov["argv"] == argv
    if rounds:
        t = prov["timing"]
        assert t["method"] == "staged" and t["rounds"] == 3 and t["iters"] == 20
    else:
        assert prov["timing"] is None
