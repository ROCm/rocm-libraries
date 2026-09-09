# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDMFuse=2's stagger gates must dispatch per wave, not by two-way parity.

``TDMFuse=2`` (``tdmFuseAMx``) aliases ``{A,MXSA,MXSB}`` onto one descriptor set
and splits them 1/1/2 over four waves: waves 0-1 carry A, wave 2 carries MXSA,
wave 3 carries MXSB. ``B`` owns its own set and every wave carries a component
of it.

A two-way parity gate cannot express that split. Gating on ``WaveIdx`` bit 0 and
picking the arm with ``"A" in tc`` -- which is true for ``MXSA`` too -- puts A's
and MXSA's offsets on the same wave and leaves B unstaggered on the even waves,
and ``removeStagger`` repeats the shape rather than cancelling it.

These tests pin the gate each tensor gets. They run the emitters unbound against
a stub, because the wave sets are a pure function of the kernel and reaching them
through a real solution costs a full toolchain build. The companion evidence that
the emitted kernel agrees lives in the codegen characterization snapshots.
"""
import pytest

from Tensile.Components.TDMFuse import tdmWavePartition
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from rocisa.code import Module

pytestmark = pytest.mark.unit

NUM_WAVES = 4


class _Tmp:
    def __init__(self, idx, size):
        self.idx = idx
        self.size = size


class _TmpCtx:
    """Stand-in for allocTmpSgpr's context manager, from a fixed high base."""

    def __init__(self, idx, size):
        self._t = _Tmp(idx, size)

    def __enter__(self):
        return self._t

    def __exit__(self, *exc):
        return False


class _Labels:
    def __init__(self):
        self._n = {}

    def getNameInc(self, name):
        self._n[name] = self._n.get(name, -1) + 1
        return name if self._n[name] == 0 else f"{name}_{self._n[name]}"


class _States:
    def __init__(self, waveIdxLive, packedBits):
        self.staggerUCode = True
        self.tdmParityPackedInArgType = packedBits > 0
        self.tdmWaveIdBitsInArgType = packedBits
        self.waveIdxReleasedAfterStagger = not waveIdxLive
        self.unrollIdx = 0


class _Writer:
    """Enough of KernelWriterAssembly for the stagger gate emitters."""

    def __init__(self, kernel, waveIdxLive=True, packedBits=0):
        self._kernel = kernel
        self._waveIdxLive = waveIdxLive
        self.states = _States(waveIdxLive, packedBits)
        self.labels = _Labels()
        self.sgprs = {"WaveIdx": 4, "ArgType": 5}

    # -- predicates come from the real class so the tests cannot drift from it --
    def isTdmWaveSeparated(self, kernel):
        return KernelWriterAssembly.isTdmWaveSeparated(self, kernel)

    def tdmFuseAMx(self, kernel):
        return KernelWriterAssembly.tdmFuseAMx(self, kernel)

    def tdmFusePaired(self, kernel):
        return KernelWriterAssembly.tdmFusePaired(self, kernel)

    def tdmArgTypeWaveIdBits(self, kernel):
        return KernelWriterAssembly.tdmArgTypeWaveIdBits(self, kernel)

    def isTdmWaveIdxLive(self, kernel):
        return self._waveIdxLive

    def _emitTdmWaveSetSkipSCC(self, module, kernel, waves, tc):
        return KernelWriterAssembly._emitTdmWaveSetSkipSCC(self, module, kernel, waves, tc)

    def _emitTdmWaveIdIntoSgpr(self, kernel, dstIdx, comment="waveId"):
        return KernelWriterAssembly._emitTdmWaveIdIntoSgpr(self, kernel, dstIdx, comment)

    def allocTmpSgpr(self, num, align=None, tag=None):
        return _TmpCtx(90, num)

    # -- emitters under test --
    def applyStagger(self, kernel, tc, group0, offsetSgpr=80,
                     labelName="SkipStagger", commentTag="stagger"):
        mod = Module("t")
        KernelWriterAssembly._applyStaggerTdmFuseAMx(
            self, mod, kernel, tc, group0, offsetSgpr, labelName, commentTag)
        return str(mod)

    def waveSetSkip(self, kernel, waves, tc):
        mod = Module("t")
        KernelWriterAssembly._emitTdmWaveSetSkipSCC(self, mod, kernel, waves, tc)
        return str(mod)

    def waveId(self, kernel, dst=90):
        return str(KernelWriterAssembly._emitTdmWaveIdIntoSgpr(self, kernel, dst))

    def hoist(self, kernel, tcA="A"):
        return str(KernelWriterAssembly._hoistTdmFuseAMxWrapUSel(self, kernel, tcA))


def _kernel(tdmFuse=2, numWaves=NUM_WAVES):
    return {
        "TDMFuse": tdmFuse,
        "NumWaves": numWaves,
        "WavefrontSize": 32,
        "enableTDMA": True,
        "enableTDMB": True,
        "TDMInst": 0x03,
        "TDMSplit": False,
        "UseSubtileImpl": False,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    }


# --------------------------------------------------------------- wave sets ---
# The layout every gate below has to agree with. Spelled out rather than derived
# so a change to tdmWavePartition cannot quietly redefine "correct".
EXPECTED_WAVES = {"A": (0, 1), "MXSA": (2,), "MXSB": (3,), "B": (0, 1, 2, 3)}


@pytest.mark.parametrize("tc, waves", sorted(EXPECTED_WAVES.items()))
def test_partition_is_the_1_1_2_remainder_split(tc, waves):
    assert tdmWavePartition(_kernel(), tc)[1] == waves


# --------------------------------------------------- which waves are admitted --
def _admitted(text, numWaves=NUM_WAVES):
    """Waves whose stagger add runs, by evaluating the emitted guard.

    SCC1 means skip, matching _tdmFuseAMxDispatch, so an admitted wave is one
    where the compare is false.
    """
    import re

    if "s_cbranch_scc1" not in text:
        return tuple(range(numWaves))
    for rx, fn in (
        (r"s_cmp_lg_u32 s\[sgprWaveIdx\], (\d+)", lambda w, n: w != n),
        (r"s_cmp_ge_u32 s\[sgprWaveIdx\], (\d+)", lambda w, n: w >= n),
        (r"s_cmp_eq_u32 s\[sgprWaveIdx\], (\d+)", lambda w, n: w == n),
        (r"s_bitcmp1_b32 s\[sgprWaveIdx\], (\d+)", lambda w, n: bool((w >> n) & 1)),
    ):
        m = re.search(rx, text)
        if m:
            n = int(m.group(1))
            return tuple(w for w in range(numWaves) if not fn(w, n))
    raise AssertionError("no recognized guard in:\n" + text)


@pytest.mark.parametrize("tc, waves", sorted(EXPECTED_WAVES.items()))
def test_stagger_add_runs_on_exactly_the_waves_that_carry_the_tensor(tc, waves):
    kernel = _kernel()
    text = _Writer(kernel).applyStagger(kernel, tc, f"tdm{tc}Group0")
    assert _admitted(text) == waves


def test_the_shared_descriptor_gets_one_tensors_offset_per_wave():
    """The bug in one assertion: {A,MXSA,MXSB} share a descriptor, so across the
    three calls each wave must be admitted by exactly one of them."""
    kernel = _kernel()
    admits = {tc: _admitted(_Writer(kernel).applyStagger(kernel, tc, "tdmAGroup0"))
              for tc in ("A", "MXSA", "MXSB")}
    for wave in range(NUM_WAVES):
        owners = [tc for tc, waves in admits.items() if wave in waves]
        assert len(owners) == 1, f"wave {wave} staggered by {owners}, want exactly one"


def test_b_is_not_gated_because_every_wave_carries_a_component():
    kernel = _kernel()
    text = _Writer(kernel).applyStagger(kernel, "B", "tdmBGroup0")
    assert "s_cbranch" not in text
    assert text.count("s_add_u32") == 1 and text.count("s_addc_u32") == 1


@pytest.mark.parametrize("tc", sorted(EXPECTED_WAVES))
def test_gate_never_uses_two_way_parity(tc):
    # s_bitcmp1 WaveIdx,0 is exactly the two-way test this fix removes.
    kernel = _kernel()
    text = _Writer(kernel).applyStagger(kernel, tc, f"tdm{tc}Group0")
    assert "s_bitcmp1_b32 s[sgprWaveIdx], 0" not in text


def test_single_wave_tensors_compare_for_equality_not_parity():
    kernel = _kernel()
    for tc, wave in (("MXSA", 2), ("MXSB", 3)):
        text = _Writer(kernel).applyStagger(kernel, tc, f"tdm{tc}Group0")
        assert f"s_cmp_lg_u32 s[sgprWaveIdx], {wave}" in text


def test_the_a_run_is_a_low_contiguous_bound():
    kernel = _kernel()
    text = _Writer(kernel).applyStagger(kernel, "A", "tdmAGroup0")
    assert "s_cmp_ge_u32 s[sgprWaveIdx], 2" in text


# --------------------------------------------------------- wave-id sourcing ---
def test_live_waveidx_is_read_directly_with_no_temporary():
    kernel = _kernel()
    text = _Writer(kernel, waveIdxLive=True).waveSetSkip(kernel, (2,), "MXSA")
    assert "s[sgprWaveIdx]" in text
    assert "v_readfirstlane" not in text and "sgprArgType" not in text


def test_dead_waveidx_recovers_the_index_from_the_argtype_pack():
    kernel = _kernel()
    text = _Writer(kernel, waveIdxLive=False, packedBits=2).waveId(kernel)
    # Two bits from bit 8 up: the whole index, not just parity.
    assert "s_lshr_b32" in text and "s[sgprArgType]" in text
    assert "0x3" in text


def test_dead_waveidx_without_a_pack_rematerializes_from_serial():
    kernel = _kernel()
    text = _Writer(kernel, waveIdxLive=False, packedBits=0).waveId(kernel)
    assert "v_readfirstlane_b32" in text
    assert "s_lshr_b32" in text


def test_argtype_pack_is_one_bit_for_two_way_paths_and_wide_for_fuse_amx():
    # Widening the pack for everyone would move TDMFuse=0/1 codegen; it must not.
    for fuse in (0, 1):
        kernel = _kernel(tdmFuse=fuse)
        assert _Writer(kernel).tdmArgTypeWaveIdBits(kernel) == 1
    kernel = _kernel(tdmFuse=2)
    assert _Writer(kernel).tdmArgTypeWaveIdBits(kernel) == 2


def test_argtype_wave_id_bits_stay_inside_the_masked_side_channel():
    # cmpNamedArgTypeEq masks 0xFF, so the pack has bits 8..31 to live in.
    kernel = _kernel(numWaves=4)
    assert 8 + _Writer(kernel).tdmArgTypeWaveIdBits(kernel) <= 32


# ---------------------------------------------------------------- the hoist ---
def test_hoist_leaves_the_a_waves_on_their_own_wrapu():
    """Waves 0-1 carry A, so a two-way fold would corrupt wave 1's WrapUA."""
    kernel = _kernel()
    text = _Writer(kernel).hoist(kernel)
    assert "s_cmp_eq_u32 s[sgprWaveIdx], 2" in text
    assert "s_cmp_eq_u32 s[sgprWaveIdx], 3" in text
    # Nothing selects WrapUB into WrapUA: B has its own descriptor here.
    assert "sgprWrapUB" not in text
    # Both halves of both scale wraps are selected over.
    assert text.count("s_cselect_b32") == 4


def test_hoist_does_not_disturb_the_scale_wrapu_registers():
    # removeStagger MXSA/MXSB still read WrapUMXSA/WrapUMXSB, so the fold may
    # only ever write WrapUA.
    kernel = _kernel()
    for line in _Writer(kernel).hoist(kernel).splitlines():
        if "s_cselect_b32" in line:
            assert line.split(",")[0].endswith("s[sgprWrapUA+0]") or \
                   line.split(",")[0].endswith("s[sgprWrapUA+1]")
