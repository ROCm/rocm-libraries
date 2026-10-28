# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts for the shared toolchain helpers."""

from types import SimpleNamespace

import pytest

import toolchain_test_helpers as helpers

pytestmark = pytest.mark.unit


def _map_reporting_unsupported(monkeypatch):
    def fake(arch):
        isa = helpers._isa_of(arch)
        return {isa: SimpleNamespace(asmCaps={"SupportedISA": False})}

    monkeypatch.setattr(helpers, "shared_isa_info_map", fake)


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950"])
def test_unsupported_stable_arch_fails_instead_of_skipping(arch, monkeypatch):
    _map_reporting_unsupported(monkeypatch)
    with pytest.raises(pytest.fail.Exception, match=f"does not support {arch}"):
        helpers.shared_isa_info_map_or_skip(arch)


def test_unsupported_newer_arch_skips(monkeypatch):
    _map_reporting_unsupported(monkeypatch)
    with pytest.raises(pytest.skip.Exception, match="does not support gfx1250"):
        helpers.shared_isa_info_map_or_skip("gfx1250")


@pytest.mark.parametrize("arch", ["gfx942", "gfx1250"])
def test_supported_arch_returns_the_shared_map(arch):
    assert helpers.shared_isa_info_map_or_skip(arch) is helpers.shared_isa_info_map(arch)
