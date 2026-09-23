# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Behavioral tests for KernelWriterAssembly.disableWmmaArbStall.

DISABLE_XDL_ARB_STALL lives in SCHED_MODE (hwreg 26) at a bit position each arch
declares for itself. The declaration is a one-row-per-arch table (stinkytofu
HwReg::schedModeDisableXdlArbStall, mirrored into rocisa's
archCaps["WmmaArbStallBitOffset"]) and the emitter holds no constant of its own
-- it used to hardcode 4, which is wrong for gfx1250.

Nothing below is arch-specific: the archs under test and their expected bits are
DECLARED_BITS, and an arch whose toolchain is unavailable skips. A new arch adds
a row there and needs no new test.

Companion to test_gfx1250_asic_revision.py's
test_wmma_arb_stall_bit_offset_is_declared_per_arch, which pins the declared
value in the capability map. This file pins what the emitter does with it.
"""

import re
from types import SimpleNamespace

import pytest

from Tensile.Common.Architectures import gfxToIsa
from Tensile.Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
from Tensile.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit

_SCHED_MODE_HWREG = 26

# One row per arch that declares the bit, expected value pinned as a literal so a
# silent change to the declaration fails here instead of passing as a tautology.
DECLARED_BITS = {
    "gfx1250": 2,
}

_ARCH_CAPS = {}


def _caps_for(arch):
    """The real archCaps for ``arch``, as an ``--architecture <arch>`` build
    produces them, or a clean skip where the probe cannot run. Cached per arch."""
    if arch not in _ARCH_CAPS:
        from Tensile.Toolchain.Validators import validateToolchain

        try:
            cxx = validateToolchain("amdclang++")
        except (ValueError, FileNotFoundError) as e:
            pytest.skip(f"amdclang++ is unavailable: {e}")

        isa = gfxToIsa(arch)
        iim = makeIsaInfoMap([isa], cxx)
        if not iim[isa].asmCaps["SupportedISA"]:
            pytest.skip(f"amdclang++ in this environment does not support {arch}")
        applyArchCapOverrides(iim, [arch])
        _ARCH_CAPS[arch] = iim[isa].archCaps
    return _ARCH_CAPS[arch]


def _emit(archCaps, disableXdlArbStall=True):
    """Run disableWmmaArbStall against a stub holding only the arch caps.

    The method reads exactly two things -- self.states.archCaps
    ["WmmaArbStallBitOffset"] and kernel["DisableXdlArbStall"] -- so an unbound
    call on a SimpleNamespace is a complete stand-in for a KernelWriterAssembly,
    without the kernel and toolchain setup a real instance needs.
    """
    stub = SimpleNamespace(states=SimpleNamespace(archCaps=archCaps))
    kernel = {"DisableXdlArbStall": disableXdlArbStall}
    return KernelWriterAssembly.disableWmmaArbStall(stub, kernel)


def _emitted_bits(src):
    """Every SCHED_MODE bit offset the emitted source writes."""
    return [int(m) for m in re.findall(rf"hwreg\({_SCHED_MODE_HWREG},(\d+),1\)", src)]


ARCHS = sorted(DECLARED_BITS.items())


@pytest.mark.parametrize("arch, bit", ARCHS)
def test_declares_the_expected_bit(arch, bit):
    """The premise the emission tests rest on."""
    assert _caps_for(arch)["WmmaArbStallBitOffset"] == bit


@pytest.mark.parametrize("arch, bit", ARCHS)
def test_emits_at_its_declared_bit(arch, bit):
    """Driven by the real capability map, so this fails if either the emitter
    stops following the declaration or the arch's declaration moves."""
    src = str(_emit(_caps_for(arch))).strip()
    assert src.startswith(f"s_setreg_IMM32_b32 hwreg({_SCHED_MODE_HWREG},{bit},1), 1")
    assert "Disable WMMA arb stall" in src


@pytest.mark.parametrize("arch, bit", ARCHS)
def test_emits_no_other_bit(arch, bit):
    """The emitter used to write bit 4 unconditionally. Stated as "only the
    declared bit" so an arch that really declares 4 still passes."""
    assert _emitted_bits(str(_emit(_caps_for(arch)))) == [bit]


@pytest.mark.parametrize("arch, bit", ARCHS)
def test_one_instruction_is_emitted(arch, bit):
    """The s_setreg is the whole module -- no wait or restore around it."""
    assert len(_emit(_caps_for(arch)).items()) == 1


@pytest.mark.parametrize("arch, bit", ARCHS)
def test_emits_nothing_when_the_kernel_opts_out(arch, bit):
    """The gate is on the kernel, not the arch, so an arch that declares the bit
    still emits nothing here. This is the sparse path."""
    assert len(_emit(_caps_for(arch), disableXdlArbStall=False).items()) == 0


def test_emits_nothing_where_the_field_is_absent():
    """The other half of the guard. An arch that does not declare the field gets
    -1, not 0, because 0 would alias DEP_MODE's LSB and write a real bit."""
    assert len(_emit({"WmmaArbStallBitOffset": -1}).items()) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
