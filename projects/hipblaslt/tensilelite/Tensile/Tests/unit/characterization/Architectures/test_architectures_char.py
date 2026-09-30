################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Common.Architectures``: the pure
gfx<->ISA helpers, codename/variant lookups, CLI-arch parsing, and the
detection helper (subprocess monkeypatched)."""

import pytest

import Tensile.Common.Architectures as A

pytestmark = pytest.mark.unit


def test_supports_chip_id_predicate():
    assert A.supportsChipIdPredicate("gfx950") is True
    assert A.supportsChipIdPredicate("gfx942") is False


def test_isa_to_gfx_and_back(snapshot):
    gfx = A.isaToGfx((9, 4, 2))
    assert {"gfx": gfx, "roundtrip": tuple(A.gfxToIsa(gfx))} == snapshot


def test_isa_to_gfx_hex_step(snapshot):
    # Step digit is hex-encoded (e.g. (9,4,10) -> gfx94a).
    assert A.isaToGfx((9, 4, 10)) == snapshot


def test_gfx_to_isa_invalid_returns_none():
    assert A.gfxToIsa("not-a-gfx") is None


def test_gfx_to_sw_codename(snapshot):
    assert {
        "gfx942": A.gfxToSwCodename("gfx942"),
        "unknown": A.gfxToSwCodename("gfxZZZZ"),
    } == snapshot


def test_gfx_to_variants(snapshot):
    # Unknown gfx falls back to [gfx].
    assert A.gfxToVariants("gfxNope") == snapshot


def test_cli_archs_to_isa_separators(snapshot):
    assert {
        "semicolon": [tuple(i) for i in A.cliArchsToIsa("gfx942;gfx90a")],
        "underscore": [tuple(i) for i in A.cliArchsToIsa("gfx942_gfx90a")],
    } == snapshot


def test_cli_archs_to_isa_all():
    assert A.cliArchsToIsa("all") == A.SUPPORTED_ISA


def test_detect_global_current_isa_success(monkeypatch, snapshot):
    # Monkeypatch the subprocess `run` to return canned gfx output + rc 0.
    class _Proc:
        returncode = 0
        stdout = b"gfx942\ngfx90a\n"

    monkeypatch.setattr(A, "run", lambda *a, **k: _Proc())
    rv = A._detectGlobalCurrentISA("amdgpu-arch", 0)
    assert tuple(rv) == snapshot


def test_detect_global_current_isa_failure(monkeypatch):
    class _Proc:
        returncode = 3
        stdout = b""

    monkeypatch.setattr(A, "run", lambda *a, **k: _Proc())
    assert A._detectGlobalCurrentISA("amdgpu-arch", 0) == 3


def test_detect_global_current_isa_public_success(monkeypatch, snapshot):
    class _Proc:
        returncode = 0
        stdout = b"gfx942\n"

    monkeypatch.setattr(A, "run", lambda *a, **k: _Proc())
    assert tuple(A.detectGlobalCurrentISA(0, "amdgpu-arch")) == snapshot


def test_detect_global_current_isa_public_failure(monkeypatch):
    class _Proc:
        returncode = 5
        stdout = b""

    monkeypatch.setattr(A, "run", lambda *a, **k: _Proc())
    with pytest.raises(Exception):
        A.detectGlobalCurrentISA(0, "amdgpu-arch")


def test_split_archs_no_predicates(snapshot):
    archs, preds = A.splitArchsFromPredicates(["gfx942"])
    assert {"archs": archs, "preds": preds} == snapshot


def test_split_archs_unsupported_raises():
    with pytest.raises(ValueError):
        A.splitArchsFromPredicates(["not-a-real-arch"])


def test_split_archs_invalid_predicate_raises():
    # gfx942 is valid, but 'bogus=1' is neither an id= nor cu= predicate.
    with pytest.raises(ValueError):
        A.splitArchsFromPredicates(["gfx942[bogus=1]"])
