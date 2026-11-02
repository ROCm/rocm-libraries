# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A and B K are paired through the instruction's canonical coupling.

The hardware multiplies the A position and the B position that hold the same hardware K. For a
same-format instruction that is slot s with slot s. A mixed-width ``f8f6f4`` instruction differs: its
8-bit operand holds K in two runs of 16 per lane, its 6/4-bit operand in one run of 32, so comparing A
and B slot for slot rejects the correct canonical pair and accepts a wrong one. The K check reads the
coupling off the two canonical layouts, which every caller must pass.

Lowering tests pin the Python engine (``_lower_kernel_to_llvm_python``); the C++ engine is checked by
the byte-identity gate. Where the host clang targets the arch, the module is also compiled and linked.
Where the current GPU is a gfx950, the driven fp8 x fp4 tile also runs on random K-dependent operands
and must equal an independent numpy product exactly; on any other GPU, or none, it skips. The gfx1250
tile with two M blocks per lane is skipped everywhere until it has run on gfx1250 hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocke.core.ir import F32, FP4E2M1, FP8E4M3, I32, IRBuilder, PtrType, VectorType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.examples.common.scaled_gemm_reference import decode_fp4, pack_fp4_codes
from rocke.helpers.mma_io import storage_ir_type
from rocke.helpers.tiling import (
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling import tiling_recorder as tr
from rocke.helpers.tiling.analysis.soundness import verify_mma_soundness
from rocke.helpers.tiling.layout_optimizer import evaluate_transform, optimize_layout
from rocke.helpers.tiling.mma import TileMma
from rocke.helpers.tiling.mma.warp_encoding import canonical_layouts
from rocke.helpers.tiling.traits import load_mma_traits
from rocke.helpers.tiling.transforms import (
    as_forward_map,
    mma_pair_compatible,
    mma_pair_k_aligned,
)
from rocke.helpers.tiling.visualization import layout_render as lr
from rocke.tests.gpu import run_on_device

from .test_mma_issue import _calls, _compile_or_skip

_CATALOG = load_mma_traits()
_FP8_FP4 = "mfma_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1"
_FP8_FP8 = "mfma_f32_16x16x128_f8f6f4_fp8e4m3_fp8e4m3"
_FP4_FP4 = "mfma_f32_16x16x128_f8f6f4_fp4e2m1_fp4e2m1"


def _canonicals(op_id: str):
    """The single-atom canonical A and B layouts of a gfx950 16x16x128 instruction."""
    a_canon, b_canon, _ = canonical_layouts(_CATALOG.get(op_id), (1, 1, 1))
    return a_canon, b_canon


def _reverse_k(layout):
    """The same layout with every K label k replaced by 127 - k: a K relabelling."""
    return {
        slot: (coord[0], 127 - coord[1])
        for slot, coord in as_forward_map(layout).items()
    }


def _pairs():
    """The gfx950 16x16x128 pairs of the mixed-width table: (A, B, A canonical, B canonical)."""
    a8, b4 = _canonicals(_FP8_FP4)
    _, b8 = _canonicals(_FP8_FP8)
    a44, b44 = _canonicals(_FP4_FP4)
    return {
        "canonical-fp8-A-x-fp4-B": (a8, b4, a8, b4),
        "same-K-reversal-on-both": (_reverse_k(a8), _reverse_k(b4), a8, b4),
        "K-reversal-on-A-only": (_reverse_k(a8), b4, a8, b4),
        "fp4-B-loaded-with-fp8-map": (a8, b8, a8, b4),
        "canonical-fp4-x-fp4": (a44, b44, a44, b44),
    }


# Each case and its verdict. The canonical fp8 x fp4 pair and the jointly reversed one multiply the
# same K, so they pass (both were refused slot for slot); the fp4 B loaded with fp8's map multiplies
# A's K 64 with B's K 32 at the coupled positions, so it fails (it passed slot for slot).
_TABLE = [
    ("canonical-fp8-A-x-fp4-B", "ok"),
    ("same-K-reversal-on-both", "ok"),
    ("K-reversal-on-A-only", "error"),
    ("fp4-B-loaded-with-fp8-map", "error"),
    ("canonical-fp4-x-fp4", "ok"),
]


@pytest.mark.parametrize("case,severity", _TABLE, ids=[c for c, _ in _TABLE])
@pytest.mark.parametrize(
    "check",
    [
        pytest.param(mma_pair_k_aligned, id="k-aligned"),
        pytest.param(mma_pair_compatible, id="compatible"),
    ],
)
def test_mixed_width_pairs_are_judged_through_the_coupling(
    check, case, severity
) -> None:
    a, b, a_canon, b_canon = _pairs()[case]
    d = check(a, b, a_canon=a_canon, b_canon=b_canon)
    assert d.severity == severity, d.message


def test_the_coupling_pairs_a_second_run_with_another_lane() -> None:
    # fp8 A lane 0 holds K 64-79 in slots 16-31; fp4 B holds K 64-95 in lane 32. Moving one B label
    # at the coupled position (lane 32, slot 0) breaks the pair, though B's lane 0 is untouched.
    a8, b4 = _canonicals(_FP8_FP4)
    a_map, b_map = as_forward_map(a8), as_forward_map(b4)
    assert (a_map[(0, 16)][1], b_map[(32, 0)][1]) == (64, 64)
    moved = dict(b_map)
    moved[(32, 0)], moved[(32, 1)] = b_map[(32, 1)], b_map[(32, 0)]
    d = mma_pair_k_aligned(a8, moved, a_canon=a8, b_canon=b4)
    assert d.severity == "warning", d.message


def test_canonicals_of_two_instructions_are_refused() -> None:
    # An fp8 A canonical with an fp4 B canonical of another shape describes no instruction: no B slot
    # in A's column holds K 64 of the 16x16x128 A.
    a8, _ = _canonicals(_FP8_FP4)
    _, b_small = canonical_layouts(
        _CATALOG.get("mfma_f32_32x32x64_f8f6f4_fp4e2m1_fp4e2m1"), (1, 1, 1)
    )[:2]
    d = mma_pair_k_aligned(a8, b_small, a_canon=a8, b_canon=b_small)
    assert d.severity == "error"


# ---- an instruction whose lane holds two M blocks of A -----------------------------------------

# Each lane of this instruction holds two M rows of A (M 0 and M 16 on lane 0), each with the same 64 K,
# against one N row of B with those 64 K: the hardware multiplies both A rows with the same B registers.
_F4_TWO_M_BLOCKS = "wmma_f32_32x16x128_f4"


def _second_m_block_k_reversed(layout):
    """The layout with K labels k -> 127 - k on the second M block of every lane only."""
    fwd = as_forward_map(layout)
    half = max(reg for _, reg in fwd) // 2 + 1
    return {
        (lane, reg): (coord[0], 127 - coord[1]) if reg >= half else coord
        for (lane, reg), coord in fwd.items()
    }


@pytest.mark.parametrize(
    "second_block_reversed,severity",
    [
        pytest.param(False, "ok", id="canonical"),
        pytest.param(True, "error", id="block-2-reversed"),
    ],
)
@pytest.mark.parametrize(
    "check",
    [
        pytest.param(mma_pair_k_aligned, id="k-aligned"),
        pytest.param(mma_pair_compatible, id="compatible"),
    ],
)
def test_each_m_block_of_a_is_paired_with_the_same_b(
    check, second_block_reversed, severity
) -> None:
    # The canonical pair passes; relabelling K on A's second M block alone breaks that block's pairing
    # with B while the first block still matches, so the check must compare every block.
    a_canon, b_canon, _ = canonical_layouts(_CATALOG.get(_F4_TWO_M_BLOCKS), (1, 1, 1))
    a = _second_m_block_k_reversed(a_canon) if second_block_reversed else a_canon
    d = check(a, b_canon, a_canon=a_canon, b_canon=b_canon)
    assert d.severity == severity, d.message


# ---- the callers: driver, soundness gate, MmaTee, layout optimizer -----------------------------


def _wave_tile(mma: TileMma):
    """One wave computes C = A * B^T for the MMA's wave tile, A (M, K) and B (N, K) loaded from global
    tensors behind pointers of each dtype's storage type, the f32 result stored."""
    wave_m, wave_n, wave_k = mma.shape
    b = IRBuilder("k_coupling")
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size
    a_ptr = b.param(
        "A",
        PtrType(storage_ir_type(mma.a_dtype.name), "global"),
        noalias=True,
        readonly=True,
    )
    b_ptr = b.param(
        "B",
        PtrType(storage_ir_type(mma.b_dtype.name), "global"),
        noalias=True,
        readonly=True,
    )
    c_ptr = b.param("C", PtrType(F32, "global"), noalias=True, writeonly=True)
    lane = b.thread_id_x()
    zero = b.const_i32(0)
    a_td = make_tensor_desc((wave_m, wave_k), (wave_k, 1), mma.a_dtype)
    b_td = make_tensor_desc((wave_n, wave_k), (wave_k, 1), mma.b_dtype)
    c_td = make_tensor_desc((wave_m, wave_n), (wave_n, 1), F32)
    acc = make_fragment(mma.c_desc, F32)
    fill_fragment(b, acc, 0)
    fa = load_fragment(b, a_ptr, make_window(a_td, (zero, zero)), mma.a_desc, lane)
    fb = load_fragment(b, b_ptr, make_window(b_td, (zero, zero)), mma.b_desc, lane)
    result = mma(b, fa, fb, acc)
    store_fragment(b, c_ptr, make_window(c_td, (zero, zero)), result, lane)
    b.ret()
    return b.kernel


def _fp8_fp4_mma() -> TileMma:
    return TileMma((16, 16, 128), a=FP8E4M3, b=FP4E2M1, c=F32, target="gfx950")


def test_driven_fp8_by_fp4_tile_issues_and_compiles() -> None:
    # The canonical fragments pass the K check. A reaches the instruction as 32 fp8 elements, B as
    # its 4 packed words; core widens each to the intrinsic's 8-word operand.
    mma = _fp8_fp4_mma()
    assert mma.op_id == _FP8_FP4
    kernel = _wave_tile(mma)
    (op,) = [op for op in kernel.body.ops if op.name == "tile.mma"]
    assert op.operands[0].type == VectorType(FP8E4M3, 32)
    assert op.operands[1].type == VectorType(I32, 4)
    ir = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
    assert len(_calls(ir, mma.traits.llvm_intrinsic)) == 1
    _compile_or_skip(kernel, "gfx950")


def test_driven_tile_of_an_instruction_with_two_m_blocks_per_lane_issues_every_atom() -> (
    None
):
    # Two atoms along each of M, N and K: every A atom is one contiguous register block holding both
    # of a lane's M rows, so the driver slices and issues all eight.
    mma = TileMma((64, 32, 256), target="gfx1250", atom_override=_F4_TWO_M_BLOCKS)
    assert mma.plan.subtiles == (2, 2, 2)
    kernel = _wave_tile(mma)
    ir = _lower_kernel_to_llvm_python(kernel, arch="gfx1250")
    assert len(_calls(ir, mma.traits.llvm_intrinsic)) == 8
    _compile_or_skip(kernel, "gfx1250")


def _fp8e4m3_codes(rng, shape) -> np.ndarray:
    """Random fp8 e4m3 codes of magnitude 0.125 to 15 (biased exponent 4 to 10, any mantissa, either
    sign): no NaN, no subnormal."""
    exponent = rng.integers(4, 11, size=shape)
    mantissa = rng.integers(0, 8, size=shape)
    sign = rng.integers(0, 2, size=shape)
    return ((sign << 7) | (exponent << 3) | mantissa).astype(np.uint8)


def _decode_fp8e4m3(codes: np.ndarray) -> np.ndarray:
    """The value of normal fp8 e4m3 codes: (-1)^s * 2^(e - 7) * (1 + m / 8)."""
    exponent = ((codes >> 3) & 15).astype(np.int32)
    magnitude = np.ldexp(1.0 + (codes & 7) / 8.0, exponent - 7)
    return np.where(codes & 0x80, -magnitude, magnitude)


def _run_k_coupled_tile(mma: TileMma, a_fp8: bool, target: str) -> None:
    """Run `_wave_tile` for `mma` on random K-dependent operands where the current GPU is `target`,
    and require C = A * B^T exactly. A is fp8 e4m3 when `a_fp8`, else fp4 like B. Every product is a
    multiple of 2^-7 below 2^7 and K is at most 256, so every partial sum is exact in f32 and the
    order the hardware adds them in cannot change the result."""
    wave_m, wave_n, wave_k = mma.shape
    rng = np.random.default_rng(wave_k + wave_m)
    if a_fp8:
        a_codes = _fp8e4m3_codes(rng, (wave_m, wave_k))
        a_bytes, a_values = a_codes, _decode_fp8e4m3(a_codes)
    else:
        a_bytes = pack_fp4_codes(
            rng.integers(0, 16, size=(wave_m, wave_k), dtype=np.uint8)
        )
        a_values = decode_fp4(a_bytes)
    b_bytes = pack_fp4_codes(rng.integers(0, 16, size=(wave_n, wave_k), dtype=np.uint8))
    expected = (a_values @ decode_fp4(b_bytes).T).astype(np.float32)
    c = np.full((wave_m, wave_n), np.nan, dtype=np.float32)
    run_on_device(
        _wave_tile(mma), {"A": a_bytes, "B": b_bytes, "C": c}, mma.wave_size, {target}
    )
    np.testing.assert_array_equal(c, expected)


def test_driven_fp8_by_fp4_tile_computes_the_product() -> None:
    # A wrong K pairing multiplies A's K with another K of B, which random K-dependent operands turn
    # into a wrong C. Runs where the current GPU is a gfx950.
    _run_k_coupled_tile(_fp8_fp4_mma(), a_fp8=True, target="gfx950")


@pytest.mark.skip(reason="has not run on gfx1250 hardware yet; enable once it has")
def test_driven_tile_with_two_m_blocks_per_lane_computes_the_product() -> None:
    # Both of a lane's M blocks of A must meet the same B registers, on every one of the eight atoms.
    # Skipped until it has run on gfx1250 hardware; then it runs where the current GPU is a gfx1250.
    mma = TileMma((64, 32, 256), target="gfx1250", atom_override=_F4_TWO_M_BLOCKS)
    _run_k_coupled_tile(mma, a_fp8=False, target="gfx1250")


def test_soundness_gate_accepts_a_recorded_fp8_by_fp4_pipeline() -> None:
    _kernel, pipe = tr.record_build(lambda: _wave_tile(_fp8_fp4_mma()))
    assert verify_mma_soundness(pipe) == 1


def test_mma_tee_of_a_canonical_fp8_by_fp4_plan_shows_no_error() -> None:
    tee = lr.MmaTee.from_mma(TileMma(target="gfx950", atom_override=_FP8_FP4))
    d = tee.pairing_diagnostic()
    assert d.severity == "ok", d.message


def test_mma_tee_without_canonicals_judges_no_pairing() -> None:
    a8, b4 = _canonicals(_FP8_FP4)
    mma = TileMma(target="gfx950", atom_override=_FP8_FP4)
    tee = lr.MmaTee(
        a_enc=a8, b_enc=b4, c_enc=mma.c_desc.layout, atom_shape=mma.atom_shape
    )
    assert tee.pairing_diagnostic() is None


def test_layout_optimizer_pairs_k_through_both_canonicals() -> None:
    a8, b4 = _canonicals(_FP8_FP4)
    _, b8 = _canonicals(_FP8_FP8)
    ok = evaluate_transform(a8, a8, canon=a8, k_partner=b4, k_partner_canon=b4)
    assert ok.k_match == "ok" and ok.works
    wrong = evaluate_transform(a8, a8, canon=a8, k_partner=b8, k_partner_canon=b4)
    assert wrong.k_match == "error" and not wrong.works


_EK = (
    "k_partner needs canon= and k_partner_canon= -- A and B K positions pair through the "
    "instruction's canonical layouts, which define the coupling"
)


@pytest.mark.parametrize(
    "canons",
    [
        pytest.param({"canon": "a"}, id="no-k_partner_canon"),
        pytest.param({"k_partner_canon": "b"}, id="no-canon"),
    ],
)
def test_layout_optimizer_refuses_a_k_partner_without_both_canonicals(canons) -> None:
    a8, b4 = _canonicals(_FP8_FP4)
    given = {name: {"a": a8, "b": b4}[which] for name, which in canons.items()}
    with pytest.raises(ValueError) as excinfo:
        evaluate_transform(a8, a8, k_partner=b4, **given)
    assert str(excinfo.value) == _EK
    with pytest.raises(ValueError) as excinfo:
        optimize_layout(a8, {"self": a8}, k_partner=b4, **given)
    assert str(excinfo.value) == _EK
