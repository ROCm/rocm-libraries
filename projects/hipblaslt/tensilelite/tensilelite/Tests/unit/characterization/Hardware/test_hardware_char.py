################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Hardware``: device-name parsing, chip-id
extraction, ``HardwarePredicate`` construction (FromISA / FromHardware incl. the
chip-id supported/unsupported/mixed branches), and the ``__lt__`` ordering."""

import importlib

import pytest

H = importlib.import_module("Tensile.Hardware")
HardwarePredicate = H.HardwarePredicate

pytestmark = pytest.mark.unit

_GFX950_ISA = (9, 5, 0)


def test_parse_device_name():
    assert H.parseDeviceNameToHex("Device 75a0") == "75a0"
    assert H.parseDeviceNameToHex(None) is None
    with pytest.raises(ValueError):
        H.parseDeviceNameToHex("not a device")


def test_extract_pci_chip_ids():
    assert H._extractPciChipIds(None) == frozenset()
    assert H._extractPciChipIds(HardwarePredicate("PciChipId", value=7)) == frozenset([7])
    orp = HardwarePredicate.Or([HardwarePredicate("PciChipId", value=1),
                                HardwarePredicate("PciChipId", value=2)])
    assert H._extractPciChipIds(orp) == frozenset([1, 2])
    assert H._extractPciChipIds(HardwarePredicate("Processor", value="gfx950")) == frozenset()


def test_from_isa(snapshot):
    assert HardwarePredicate.FromISA((9, 4, 2)).state() == snapshot


def test_from_hardware_processor_only(snapshot):
    assert HardwarePredicate.FromHardware((9, 4, 2)).state() == snapshot


def test_from_hardware_with_cucount(snapshot):
    assert HardwarePredicate.FromHardware((9, 4, 2), cuCount=64).state() == snapshot


def test_from_hardware_single_chipid(snapshot):
    p = HardwarePredicate.FromHardware(_GFX950_ISA, deviceNames=["Device 75a0"])
    assert p.state() == snapshot


def test_from_hardware_multi_chipid(snapshot):
    p = HardwarePredicate.FromHardware(_GFX950_ISA, deviceNames=["Device 75a0", "Device 75a2"])
    assert p.state() == snapshot


def test_from_hardware_unsupported_chipid_warns(snapshot, capsys):
    # All requested IDs unsupported -> warning + no chip-id predicate (Processor only).
    p = HardwarePredicate.FromHardware(_GFX950_ISA, deviceNames=["Device ffff"])
    out = capsys.readouterr().out
    assert {"state": p.state(), "warned": "WARNING" in out} == snapshot


def test_from_hardware_mixed_chipid_warns(snapshot, capsys):
    # Some supported, some not -> warning + chip-id predicate for the supported one.
    p = HardwarePredicate.FromHardware(_GFX950_ISA, deviceNames=["Device 75a0", "Device ffff"])
    out = capsys.readouterr().out
    assert {"state": p.state(), "warned": "WARNING" in out} == snapshot


def test_from_hardware_empty_devicenames(snapshot):
    assert HardwarePredicate.FromHardware(_GFX950_ISA, deviceNames=[]).state() == snapshot


# --- __lt__ ordering --------------------------------------------------------

def test_lt_truepred():
    p = HardwarePredicate.FromHardware((9, 4, 2))
    tp = HardwarePredicate("TruePred")
    assert p < tp
    assert not (tp < p)


def test_lt_chipid_more_specific():
    withId = HardwarePredicate.FromHardware(_GFX950_ISA, deviceNames=["Device 75a0"])
    without = HardwarePredicate.FromHardware(_GFX950_ISA)
    assert withId < without          # chip-id set is more specific -> sorts first
    assert not (without < withId)


def test_lt_cucount_priority():
    hi = HardwarePredicate.FromHardware((9, 4, 2), cuCount=128)
    lo = HardwarePredicate.FromHardware((9, 4, 2), cuCount=64)
    assert hi < lo                   # higher CU count first
    assert not (lo < hi)


def test_lt_processor_compare():
    a = HardwarePredicate.FromHardware((9, 4, 2))
    b = HardwarePredicate.FromHardware((9, 0, 10))
    # Falls to the Processor predicate comparison; deterministic ordering.
    assert (a < b) != (b < a)


def test_lt_different_chipid_sets(snapshot):
    one = HardwarePredicate.FromHardware(_GFX950_ISA, deviceNames=["Device 75a0"])
    two = HardwarePredicate.FromHardware(_GFX950_ISA, deviceNames=["Device 75a0", "Device 75a2"])
    # Pin the actual ordering of two differing chip-id sets (rank/size/key path).
    assert {"one_lt_two": one < two, "two_lt_one": two < one} == snapshot


def test_from_hardware_single_string_devicename(snapshot):
    # deviceNames as a bare string (not a list) -> the str-wrap branch.
    p = HardwarePredicate.FromHardware(_GFX950_ISA, deviceNames="Device 75a0")
    assert p.state() == snapshot
