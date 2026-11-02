# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A rocRoller custom kernel must declare the sizes its capture covers.

These kernels are pre-assembled objects captured at one tile. They carry no
generated edge handling in M and no partial-k-tile drain, so they are correct
only where M is a whole number of MacroTile0 and K a whole number of DepthU.
The generated logic declared neither, so a GridBased lookup handed them
neighbouring sizes and they returned wrong, run-to-run nondeterministic
answers.

N is deliberately unconstrained -- measured clean at N = 160, 192 and 320
against a 128-wide tile -- because a predicate that rejects working sizes
makes the kernels unreachable, which is the opposite failure.

The family is the kernels whose custom.config declares Source.Origin
rocroller, so these tests name kernels that are checked in.
"""

import pytest

from tensilelite.Contractions import ProblemPredicate
from tensilelite.CustomKernels import customKernelOrigin, getAllCustomKernelNames

_RR_KERNEL = "RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_"
# A rocRoller capture that predates the RR_GEMM_ naming.
_LEGACY_RR_KERNEL = (
    "RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_"
    "LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_"
    "SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e"
)
_AITER_KERNEL = "_ZN5aiter24bf16gemm_bf16_tn_256x256E"


M_TAG = "Free0SizeMultiple"
K_TAG = "BoundSizeMultiple"
N_TAG = "Free1SizeMultiple"


class _DataTypeStub:
    def isInt8x4(self):
        return False


class _ProblemTypeStub:
    def __init__(self):
        self.mxBlockA = 32
        self.mxBlockB = 32
        self.aType = _DataTypeStub()


def _state(name=_RR_KERNEL, macrotile=(256, 128, 128), customKernel=True,
           generated=False, **overrides):
    state = {
        "ProblemType": {
            "SwizzleTensorA": False,
            "SwizzleTensorB": False,
            "FusedGemmA2A": False,
            "TLUA": False,
            "TLUB": False,
        },
        "InternalSupportParams": {"KernArgsVersion": 0},
        "MacroTile0": 256,
        "MacroTile1": 128,
        "DepthU": 128,
        "GlobalReadVectorWidthA": 1,
        "GlobalReadVectorWidthB": 1,
        "PackedC0IndicesX": [],
        "WorkGroupMappingXCC": 1,
        "WorkGroupMappingXCCGroup": -1,
        "MXScaleFormat": "NoSwizzle",
    }
    if customKernel:
        state["CustomKernel"] = {"name": name, "macrotile": list(macrotile)}
        if generated:
            state["CustomKernel"]["generated"] = True
    state.update(overrides)
    return state


def _by_tag(state):
    preds = ProblemPredicate.CompoundPredicates(state, _ProblemTypeStub())
    out = {}
    for p in preds:
        out.setdefault(p.tag, []).append(p)
    return out


# -- the constraint is declared -------------------------------------------


def test_m_and_k_multiples_are_declared():
    tags = _by_tag(_state())
    assert len(tags.get(M_TAG, [])) == 1
    assert len(tags.get(K_TAG, [])) == 1
    assert tags[M_TAG][0].value == 256   # MacroTile0
    assert tags[M_TAG][0].index == 0     # free index 0 is M
    assert tags[K_TAG][0].value == 128   # DepthU
    assert tags[K_TAG][0].index == -1    # last bound index is K


def test_n_is_not_constrained():
    """Measured clean at N = 160, 192, 320 against a 128-wide tile. A
    Free1SizeMultiple here would reject sizes the kernels handle."""
    assert N_TAG not in _by_tag(_state())


@pytest.mark.parametrize("macrotile,expectM,expectK", [
    ((256, 128, 128), 256, 128),
    ((32, 32, 64), 32, 64),
    ((16, 16, 128), 16, 128),
    ((256, 256, 128), 256, 128),
])
def test_the_value_tracks_each_kernels_own_tile(macrotile, expectM, expectK):
    tags = _by_tag(_state(macrotile=macrotile))
    assert tags[M_TAG][0].value == expectM
    assert tags[K_TAG][0].value == expectK


# -- and nothing else is touched ------------------------------------------


def test_non_custom_solutions_are_untouched():
    tags = _by_tag(_state(customKernel=False))
    assert M_TAG not in tags and K_TAG not in tags


@pytest.mark.parametrize("name", [
    "Cijk_Alik_Bljk_F8SS_MXAE8B32_MXBE8B32_BH_UserArgs_MT64x64x256",
    "wvSpltK_bf16_tn_m1",
    "Custom_Cijk_Alik_Bljk_BBS_BH_UserArgs",
    _AITER_KERNEL,
])
def test_other_custom_kernel_families_are_untouched(name):
    """Scoped to the rocRoller captures because that is the family that was
    measured. The AITER and generated custom kernels keep their existing
    behaviour."""
    tags = _by_tag(_state(name=name))
    assert M_TAG not in tags and K_TAG not in tags


def test_the_scope_is_the_origin_not_the_name():
    # Named like the family but not a checked-in capture: no origin to read.
    tags = _by_tag(_state(name="RR_GEMM_TN_FP8_FP8_Float_Float_Float_not_checked_in"))
    assert M_TAG not in tags and BATCH_TAG not in tags
    # A capture that does not follow the RR_GEMM_ naming is still one.
    tags = _by_tag(_state(name=_LEGACY_RR_KERNEL, macrotile=(64, 64, 64)))
    assert tags[M_TAG][0].value == 64 and tags[K_TAG][0].value == 64
    assert BATCH_TAG in tags and BUFFER_TAG in tags


def test_a_generated_kernel_is_untouched_whatever_it_is_called():
    tags = _by_tag(_state(generated=True))
    assert M_TAG not in tags and BATCH_TAG not in tags


def test_origin_is_read_from_custom_config():
    assert customKernelOrigin(_RR_KERNEL) == "rocroller"
    assert customKernelOrigin(_LEGACY_RR_KERNEL) == "rocroller"
    assert customKernelOrigin(_AITER_KERNEL) == "aiter"
    assert customKernelOrigin("") is None
    assert customKernelOrigin("no-such-kernel-anywhere") is None


def test_every_checked_in_rr_gemm_kernel_is_a_rocroller_capture():
    """The name and the origin describe the same family on the shipped corpus,
    so keying on the origin dropped no RR_GEMM_ kernel from the predicates."""
    names = [n for n in getAllCustomKernelNames() if n.startswith("RR_GEMM_")]
    assert names
    assert [n for n in names if customKernelOrigin(n) != "rocroller"] == []


def test_a_unit_tile_emits_nothing():
    """A multiple of 1 constrains nothing, and Tensile drops such assertions
    rather than emitting a predicate that is always true."""
    tags = _by_tag(_state(macrotile=(1, 1, 1)))
    assert M_TAG not in tags and K_TAG not in tags


# -- batch and buffer-addressing limits -----------------------------------

BATCH_TAG = "BatchSizeEqual"
BUFFER_TAG = "BufferAddressingLimitCheck"


def test_batch_is_pinned_to_one():
    """The kernels are 2-D: no batch strides, workgroup_id_z unread, but a
    grid of [TilesX, TilesY, Batch]. Every z-slice recomputes batch 0, so
    batches 1..N-1 come back wrong."""
    tags = _by_tag(_state())
    assert len(tags.get(BATCH_TAG, [])) == 1
    assert tags[BATCH_TAG][0].value == 1
    assert tags[BATCH_TAG][0].index == 0


def test_buffer_addressing_limit_is_declared():
    """One buffer descriptor per tensor, sized over the whole tensor with a
    32-bit num_records: at 4 GiB it wraps and the kernel zeroes its output."""
    assert len(_by_tag(_state()).get(BUFFER_TAG, [])) == 1


def test_batch_and_buffer_limits_survive_a_unit_tile():
    """Neither depends on the tile shape, so a kernel whose macrotile emits no
    size multiples still declares both."""
    tags = _by_tag(_state(macrotile=(1, 1, 1)))
    assert BATCH_TAG in tags and BUFFER_TAG in tags


@pytest.mark.parametrize("name", [
    "Cijk_Alik_Bljk_F8SS_MXAE8B32_MXBE8B32_BH_UserArgs_MT64x64x256",
    "wvSpltK_bf16_tn_m1",
])
def test_batch_and_buffer_limits_are_rocroller_only(name):
    """Other kernel families handle batching and large tensors themselves."""
    tags = _by_tag(_state(name=name))
    assert BATCH_TAG not in tags and BUFFER_TAG not in tags
