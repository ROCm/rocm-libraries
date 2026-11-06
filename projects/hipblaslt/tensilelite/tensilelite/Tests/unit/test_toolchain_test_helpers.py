# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Contracts for the shared toolchain helpers."""

import copy
import pickle
from types import SimpleNamespace

import pytest

import toolchain_test_helpers as helpers
from tensilelite.Common.Types import IsaInfo

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


# The tests below use a hand-built map, so they need neither amdclang++ nor a
# real capability probe.

_KEY = (9, 4, 2)


def _shared_map():
    info = IsaInfo(
        asmCaps={"SupportedISA": True},
        archCaps={"HasWMMA": False},
        regCaps={"MaxVgpr": 256},
        asmBugs={"ExplicitCO": False},
    )
    return helpers._ReadOnlyDict({_KEY: helpers._read_only_isa_info(info)})


def test_shared_map_refuses_changes_and_names_the_copy_helper():
    shared = _shared_map()
    with pytest.raises(TypeError, match="copy_isa_info_map"):
        shared[(1, 2, 3)] = None
    with pytest.raises(TypeError, match="copy_isa_info_map"):
        del shared[_KEY]
    with pytest.raises(TypeError, match="copy_isa_info_map"):
        shared.update({})


def test_shared_isa_info_refuses_attribute_and_capability_changes():
    info = _shared_map()[_KEY]
    with pytest.raises(TypeError, match="copy_isa_info_map"):
        info.asmCaps = {}
    with pytest.raises(TypeError, match="copy_isa_info_map"):
        del info.asmCaps
    with pytest.raises(TypeError, match="copy_isa_info_map"):
        info.asmCaps["SupportedISA"] = False
    with pytest.raises(TypeError, match="copy_isa_info_map"):
        info.archCaps.pop("HasWMMA")


def test_copy_isa_info_map_is_a_plain_mutable_copy():
    shared = _shared_map()
    private = helpers.copy_isa_info_map(shared)

    assert type(private) is dict
    assert type(private[_KEY]) is IsaInfo
    assert type(private[_KEY].asmCaps) is dict

    private[_KEY].asmCaps["SupportedISA"] = False
    private[_KEY].archCaps["HasWMMA"] = True
    private[_KEY] = IsaInfo({}, {}, {}, {})
    assert shared[_KEY].asmCaps == {"SupportedISA": True}
    assert shared[_KEY].archCaps == {"HasWMMA": False}


def test_deepcopy_of_the_shared_map_is_plain_and_mutable():
    shared = _shared_map()
    private = copy.deepcopy(shared)

    assert type(private) is dict
    assert type(private[_KEY]) is IsaInfo
    private[_KEY].regCaps["MaxVgpr"] = 512
    assert shared[_KEY].regCaps == {"MaxVgpr": 256}


def test_shared_map_pickles_to_plain_objects_with_equal_contents():
    shared = _shared_map()
    restored = pickle.loads(pickle.dumps(shared))

    assert type(restored) is dict
    assert type(restored[_KEY]) is IsaInfo
    assert restored[_KEY] == shared[_KEY]
    restored[_KEY].asmCaps["SupportedISA"] = False  # mutable after the round trip


def test_build_toolchain_for_returns_a_private_map(monkeypatch):
    shared = _shared_map()
    monkeypatch.setattr(helpers, "shared_assembler", lambda *a, **k: "assembler")
    monkeypatch.setattr(helpers, "shared_isa_info_map", lambda arch: shared)

    assembler, private = helpers.build_toolchain_for("gfx942")

    assert assembler == "assembler"
    assert private is not shared
    assert type(private[_KEY]) is IsaInfo
    private[_KEY].asmCaps["SupportedISA"] = False
    assert shared[_KEY].asmCaps == {"SupportedISA": True}
