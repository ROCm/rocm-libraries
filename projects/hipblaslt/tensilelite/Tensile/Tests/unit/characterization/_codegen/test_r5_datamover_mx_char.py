################################################################################
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""R5 — gfx1250 single-wave MX scale TDM descriptor characterization.

Target: the MXS arms of ``KernelWriterAssembly.initTDMDescriptor`` and the scale
descriptor build in ``KernelWriter.setupNewTile``.

The sibling R3 suite drives the non-wave-separated TDM path, but its config
carries no MX scales (DataType=F4 alone leaves MXBlockA/MXBlockB at 0), so the
scale descriptors were never built there. At one wave the scale loads issue from
``tdmMXS{A,B}Group*`` that nothing had written: the geometry those registers
carry is what this suite pins.

The kernel declares 117 SGPRs against a 106 max, so it only exists once the
scalar allocator has settled the pool -- hence ``StinkyTofuRegisterAllocation``
3 on every emit here. That also means operands arrive renumbered, so assertions
match against ``with_symbol_names``.
"""

import os
import re

import pytest

from config_harness import emit_kernels_from_config, with_symbol_names

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "datamover_mx.yaml",
)

# Forced apply: checkResources judges the pool before allocation runs, so this
# shape is only emittable when the verdict is deferred to the allocated count.
_FORCE_ALLOC = {"StinkyTofuRegisterAllocation": 3}

# MXBlock=32 scales under MatrixInstK=128 -> mxUnit = 4 scales per K step, and
# DepthU=128 -> one scale K group. So the scale tile is MacroTile(16) * 4 wide
# and one group deep, while the F6 data tile stays 128 * 3/4 = 96 by 16.
_MXS_TILE0, _MXS_TILE1 = 64, 1
_DATA_TILE0, _DATA_TILE1 = 96, 16


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    # The allocation pipeline dumps kernel_*.stir and ssa_live_out.txt to
    # relative paths with no way to turn them off, so emit from a scratch
    # directory and leave the source tree alone.
    scratch = tmp_path_factory.mktemp("datamover_mx")
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        return emit_kernels_from_config(_CONFIG, limit=2, arch=_ARCH,
                                        global_params=_FORCE_ALLOC)
    finally:
        os.chdir(cwd)


def _prologue(src):
    """Text before the first tensor load, with allocator renaming undone."""
    named = with_symbol_names(src)
    return named[:named.index("tensor_load_to_lds")]


def _writes_to(prologue, group):
    """Instructions whose destination is one of ``group``'s registers."""
    return [line for line in prologue.splitlines()
            if re.match(r"\s*\w+\s+s\[?sgpr%s" % group, line)]


def test_r5_datamover_mx_emits_assembly(emitted):
    """The single-wave MX TDM shape emits under forced scalar allocation."""
    assert len(emitted) == 1, f"expected 1 kernel, got {len(emitted)}"
    base, src, err = emitted[0]
    assert err == 0, f"kernel {base!r} failed to emit (err={err})"
    assert "MXAE8B32_MXBE8B32" in base, f"not an MX-scaled kernel: {base!r}"


@pytest.mark.parametrize("tc", ["MXSA", "MXSB"])
def test_r5_datamover_mx_scale_descriptor_is_built(emitted, tc):
    """Each scale rides a descriptor of its own, written before the load.

    Without the build in setupNewTile the loads still issue, from a descriptor
    group holding whatever the pool left there.
    """
    base, src, _err = emitted[0]
    prologue = _prologue(src)
    assert _writes_to(prologue, f"tdm{tc}Group1"), (
        f"kernel {base!r}: nothing writes tdm{tc}Group1 before the first load, "
        f"so {tc} loads from an uninitialized descriptor"
    )
    assert f"s[sgprAddress{tc}:sgprAddress{tc}+1]" in prologue, (
        f"kernel {base!r}: {tc}'s descriptor must take the scale tensor's base "
        f"address, not the data tensor's"
    )


@pytest.mark.parametrize("tc", ["MXSA", "MXSB"])
def test_r5_datamover_mx_scale_tile_geometry(emitted, tc):
    """The scale tile is counted in scale elements, one K group per fetch."""
    base, src, _err = emitted[0]
    writes = "\n".join(_writes_to(_prologue(src), f"tdm{tc}Group1"))
    assert f"set tile0 to {_MXS_TILE0}" in writes, (
        f"kernel {base!r}: {tc} tile0 should span MacroTile * mxUnit "
        f"(= {_MXS_TILE0} scales); got:\n{writes}"
    )
    assert f"set tile1 to {_MXS_TILE1}" in writes, (
        f"kernel {base!r}: {tc} tile1 should be the {_MXS_TILE1} scale K group "
        f"the unroll holds; got:\n{writes}"
    )


@pytest.mark.parametrize("tc", ["A", "B"])
def test_r5_datamover_mx_data_tile_geometry_unchanged(emitted, tc):
    """The data descriptors keep their own geometry next to the scale ones."""
    base, src, _err = emitted[0]
    writes = "\n".join(_writes_to(_prologue(src), f"tdm{tc}Group1"))
    assert f"set tile0 to {_DATA_TILE0}" in writes, (
        f"kernel {base!r}: {tc} tile0 should stay the F6 byte width; got:\n{writes}"
    )
    assert f"set tile1 to {_DATA_TILE1}" in writes, (
        f"kernel {base!r}: {tc} tile1 should stay MacroTile; got:\n{writes}"
    )


def _lines_with(src, text):
    return [line for line in with_symbol_names(src).splitlines() if text in line]


def test_r5_datamover_mx_start_addr_uses_free_axis(emitted):
    """The scale offset follows the free axis, not isA/isB.

    A scale tensor is neither operand, so ``0 if isA else 1`` sends MXSA down
    B's axis: WorkGroup1, SizeJ, and StrideMXSAJ. MXSB already lands on 1, so
    the pair of GSU comments is what distinguishes the two: one SizeI and one
    SizeJ. The standalone ``TDM calc start addr`` comment does not survive
    emission, so this matches the instruction comments that do. The tile-stride
    multiply does not: the free axis is the unit stride, the multiply folds,
    and every stride name is allocated whether or not the offset uses it.
    """
    base, src, _err = emitted[0]
    gsu = _lines_with(src, "scale GSU offset by tile size")
    assert sum("SizeI" in line for line in gsu) == 1, (
        f"kernel {base!r}: MXSA's GSU offset should be the only SizeI; got:\n"
        + "\n".join(gsu)
    )
    assert sum("SizeJ" in line for line in gsu) == 1, (
        f"kernel {base!r}: MXSB's GSU offset should be the only SizeJ; got:\n"
        + "\n".join(gsu)
    )
    # s_mul_u64_u32 splits into a hi and a lo multiply, so each tensor
    # contributes two lines. A and MXSA are axis 0; B and MXSB are axis 1.
    # Sending MXSA down isA/isB puts it on axis 1 and the two counts diverge.
    workgroups = _lines_with(src, "*= wgId")
    axis0 = sum("WorkGroup0" in line for line in workgroups)
    axis1 = sum("WorkGroup1" in line for line in workgroups)
    assert axis0 == axis1 and axis0 > 0, (
        f"kernel {base!r}: workgroup offsets should split evenly across the two "
        f"free axes (got WorkGroup0={axis0}, WorkGroup1={axis1}):\n"
        + "\n".join(workgroups)
    )


def test_r5_datamover_mx_loads_all_four_tensors(emitted):
    """A, B and both scales each issue their own descriptor-driven load."""
    base, src, _err = emitted[0]
    named = with_symbol_names(src)
    loaded = {m for m in re.findall(r"tensor_load_to_lds s\[?sgprtdm(\w+?)Group0",
                                    named)}
    assert loaded == {"A", "B", "MXSA", "MXSB"}, (
        f"kernel {base!r}: expected loads for A, B, MXSA and MXSB; got {sorted(loaded)}"
    )
