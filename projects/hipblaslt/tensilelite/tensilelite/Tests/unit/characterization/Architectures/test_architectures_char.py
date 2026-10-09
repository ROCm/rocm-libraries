################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Common.Architectures``: the pure
gfx<->ISA helpers, codename/variant lookups, CLI-arch parsing, and the
detection helper (subprocess monkeypatched)."""

import pytest

import Tensile.Common.Architectures as A
from Tensile.Common.Types import IsaVersion

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


def _shellOutReturning(monkeypatch, stdout, returncode=0):
    """Wire the enumerator shell-out to canned output.

    amdgpu-arch and rocminfo back it up, so they are silenced too -- left alone
    they answer from the real device once the canned output is empty.
    """

    class _Proc:
        pass

    _Proc.returncode = returncode
    _Proc.stdout = stdout
    monkeypatch.setattr(A, "detect_gpu_archs", lambda: [])
    monkeypatch.setattr(A, "run", lambda *a, **k: _Proc())


def test_detect_global_current_isa_success(monkeypatch, snapshot):
    _shellOutReturning(monkeypatch, b"gfx942\ngfx90a\n")
    rv = A._detectGlobalCurrentISA("amdgpu-arch", 0)
    assert tuple(rv) == snapshot


def test_detect_global_current_isa_failure(monkeypatch):
    # Not the tool's exit code any more, just "this is not an ISA". Detection
    # has several sources now, so there is no single returncode to hand back,
    # and no caller ever read the number: both public wrappers only check the
    # type before raising.
    _shellOutReturning(monkeypatch, b"", returncode=3)
    assert not isinstance(A._detectGlobalCurrentISA("amdgpu-arch", 0), IsaVersion)


def test_detect_global_current_isa_public_success(monkeypatch, snapshot):
    _shellOutReturning(monkeypatch, b"gfx942\n")
    assert tuple(A.detectGlobalCurrentISA(0, "amdgpu-arch")) == snapshot


def test_detect_global_current_isa_public_failure(monkeypatch):
    _shellOutReturning(monkeypatch, b"", returncode=5)
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
