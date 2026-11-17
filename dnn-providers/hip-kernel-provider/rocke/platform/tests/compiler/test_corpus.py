# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Contracts for the bounded COMGR behavior corpus."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rocke.compiler import corpus as C
from rocke.compiler import model as M
from rocke.core import llvm_profile

_PLATFORM = Path(__file__).resolve().parents[2]


def test_flavor_consumer_inventory_is_mechanically_closed() -> None:
    platform = _PLATFORM
    roots = {
        "python": platform / "python" / "rocke" / "core",
        "cpp": platform / "cpp" / "core",
    }
    if not all(root.is_dir() for root in roots.values()):
        pytest.skip("source Python/C++ flavor consumers are not installed test assets")
    tokens = {
        "python": (
            "_emission_profile",
            "_supports_llvm23_target_ops",
            "_flavor_at_least",
            "_is_modern_flavor",
            "_datalayout_for_flavor",
            "_flavor_for_llvm",
            "minimum_llvm_flavor",
            "required_llvm_flavor",
        ),
        "cpp": (
            "rocke_ll_emission_profile",
            "rocke_ll_profile_has",
            "rocke_ll_flavor_at_least",
            "rocke_ll_datalayout_for_flavor",
            "rocke_ll_flavor_for_llvm_major",
            "minimum_llvm_flavor",
            "required_llvm_flavor",
        ),
    }
    observed = {}
    for language, root in roots.items():
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".cpp", ".h"}:
                continue
            source = path.read_text(encoding="utf-8")
            for token in tokens[language]:
                count = len(
                    re.findall(
                        rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
                        source,
                    )
                )
                if count:
                    observed[(str(path.relative_to(platform)), token)] = count
    expected = {
        ("cpp/core/ir/ir_tile.cpp", "minimum_llvm_flavor"): 1,
        ("cpp/core/isa/backend.cpp", "rocke_ll_datalayout_for_flavor"): 1,
        ("cpp/core/llvm_profile.cpp", "rocke_ll_datalayout_for_flavor"): 1,
        ("cpp/core/llvm_profile.cpp", "rocke_ll_emission_profile"): 4,
        ("cpp/core/llvm_profile.cpp", "rocke_ll_flavor_at_least"): 1,
        ("cpp/core/llvm_profile.cpp", "rocke_ll_flavor_for_llvm_major"): 1,
        ("cpp/core/llvm_profile.cpp", "rocke_ll_profile_has"): 1,
        ("cpp/core/lower_llvm/control.cpp", "rocke_ll_profile_has"): 1,
        ("cpp/core/lower_llvm/core.cpp", "rocke_ll_datalayout_for_flavor"): 1,
        ("cpp/core/lower_llvm/core.cpp", "rocke_ll_flavor_for_llvm_major"): 1,
        ("cpp/core/lower_llvm/core.cpp", "rocke_ll_profile_has"): 1,
        ("cpp/core/lower_llvm/crosslane.cpp", "minimum_llvm_flavor"): 2,
        ("cpp/core/lower_llvm/crosslane.cpp", "required_llvm_flavor"): 1,
        ("cpp/core/lower_llvm/crosslane.cpp", "rocke_ll_flavor_at_least"): 1,
        ("cpp/core/lower_llvm/data.cpp", "rocke_ll_emission_profile"): 1,
        ("cpp/core/lower_llvm/mma.cpp", "rocke_ll_profile_has"): 1,
        ("python/rocke/core/ir.py", "minimum_llvm_flavor"): 1,
        ("python/rocke/core/isa/backend.py", "_datalayout_for_flavor"): 3,
        ("python/rocke/core/isa/backend.py", "_supports_llvm23_target_ops"): 2,
        ("python/rocke/core/llvm_profile.py", "_datalayout_for_flavor"): 2,
        ("python/rocke/core/llvm_profile.py", "_emission_profile"): 3,
        ("python/rocke/core/llvm_profile.py", "_flavor_at_least"): 1,
        ("python/rocke/core/llvm_profile.py", "_flavor_for_llvm"): 1,
        ("python/rocke/core/llvm_profile.py", "_is_modern_flavor"): 1,
        ("python/rocke/core/llvm_profile.py", "_supports_llvm23_target_ops"): 1,
        ("python/rocke/core/lower_llvm.py", "_datalayout_for_flavor"): 1,
        ("python/rocke/core/lower_llvm.py", "_emission_profile"): 4,
        ("python/rocke/core/lower_llvm.py", "_flavor_at_least"): 2,
        ("python/rocke/core/lower_llvm.py", "_flavor_for_llvm"): 2,
        ("python/rocke/core/lower_llvm.py", "_is_modern_flavor"): 3,
        ("python/rocke/core/lower_llvm.py", "_supports_llvm23_target_ops"): 2,
        ("python/rocke/core/lower_llvm.py", "minimum_llvm_flavor"): 2,
        ("python/rocke/core/lower_llvm.py", "required_llvm_flavor"): 1,
    }
    assert observed == expected


def test_protocol_digest_is_deterministic_and_target_specific() -> None:
    gfx950 = C.protocol_sha256("gfx950")
    assert gfx950 == C.protocol_sha256("gfx950")
    assert len(gfx950) == 64
    assert gfx950 != C.protocol_sha256("gfx1250")


def test_current_profile_layouts_remain_in_probe_owned_archaeology() -> None:
    assert C.LAYOUT_CANDIDATES["L0"] == llvm_profile._DATALAYOUT_LLVM20
    assert C.LAYOUT_CANDIDATES["L1"] == llvm_profile._DATALAYOUT_LLVM22
    assert C.LAYOUT_CANDIDATES["L3"] == llvm_profile._DATALAYOUT_LLVM23


def test_intrinsic_candidate_corpora_are_exhaustive() -> None:
    assert len(C._buffer_resource_candidates()) == 4
    assert len(C._fp8_packing_candidates()) == 16
    assert len(C._named_barrier_candidates()) == 8
    assert len(C._scaled_mfma_candidates()) == 4
    assert {
        (candidate.family, candidate.candidate)
        for candidate in C._fp8_packing_candidates()
    } == {
        (f"{shape}_{lhs}_{rhs}", packed)
        for shape in ("16x16x32", "32x32x16")
        for lhs, rhs in (
            ("fp8", "fp8"),
            ("bf8", "bf8"),
            ("fp8", "bf8"),
            ("bf8", "fp8"),
        )
        for packed in ("v2i32", "i64")
    }


def test_generic_processor_names_are_explicit_not_digit_derived() -> None:
    assert C.GENERIC_PROCESSORS["gfx950"] == "gfx9-4-generic"
    assert C.GENERIC_PROCESSORS["gfx942"] == "gfx9-4-generic"
    assert C.GENERIC_PROCESSORS["gfx906"] == "gfx9-generic"
    assert "gfx1250" not in C.GENERIC_PROCESSORS
    assert "gfx999" not in C.GENERIC_PROCESSORS


def test_requests_use_comgr_isa_options_and_complete_modules() -> None:
    requests, metadata = C.requests_for_processor("gfx950")
    assert len(requests) == sum(
        len(entries) for entries in C.expected_observations("gfx950").values()
    )
    assert len(requests) == len(metadata) == len({request.key for request in requests})
    assert all(request.isa.startswith("amdgcn-amd-amdhsa--") for request in requests)
    assert all(
        'target triple = "amdgcn-amd-amdhsa"' in request.ir_text for request in requests
    )
    assert len(requests) == 16
    composite = {
        metadata[request.key][0]: request.ir_text
        for request in requests
        if metadata[request.key][2] in {"v2i32", "as3", "current_9"}
    }
    assert composite[M.CAPABILITY_FP8_PACKING].count("define amdgpu_kernel") == 8
    assert composite[M.CAPABILITY_NAMED_BARRIER].count("define amdgpu_kernel") == 4
    assert composite[M.CAPABILITY_SCALED_MFMA].count("define amdgpu_kernel") == 2
