# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""How TileMma decides what to issue: the measured backend coverage record, and the driver's checks.

``TileMmaPlan.emit_op()`` returns the backend op the record (``traits/data/backend_coverage.json``)
names for the plan's (instruction, target) pair, fetched by that op id; a pair the record does not
class ``issuable`` is refused with its class. The driver then refuses, before any IR, an instruction
family it cannot feed (scaled, sparse) and a result dtype other than the accumulator's that would be
fed back as the next K step's accumulator, and types its result by the instruction's result dtype.

Lowering tests pin the Python engine (``_lower_kernel_to_llvm_python``); the C++ engine is checked by
the byte-identity gate. Where the host clang targets the arch, the module is also compiled and linked
(compile-verified, never run).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from rocke.assets import platform_root
from rocke.core.arch import ArchTarget
from rocke.core.ir import (
    BF8E5M2,
    BF16,
    F16,
    F32,
    FP8E4M3,
    I8,
    I32,
    U8,
    IRBuilder,
    PtrType,
    dtype_to_ir_type,
)
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python
from rocke.helpers.tiling import (
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.mma import TileMma, TileMmaPlan, Tiling
from rocke.helpers.tiling.traits import load_backend_coverage, load_mma_traits
from rocke.helpers.tiling.traits.backend_coverage import committed_backend_coverage

_CATALOG = load_mma_traits()
_COVERAGE = load_backend_coverage()


def _wave_tile_kernel(mma: TileMma, name: str):
    """One wave computes C = A * B^T for the MMA's wave tile: A (M, K) and B (N, K) loaded from
    global memory, the result stored to a tensor of the result dtype. Returns the kernel and the
    result Fragment the driver gave back."""
    wave_m, wave_n, wave_k = mma.shape
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size
    a_ptr = b.param("A", PtrType(mma.a_dtype, "global"), noalias=True, readonly=True)
    b_ptr = b.param("B", PtrType(mma.b_dtype, "global"), noalias=True, readonly=True)
    c_ptr = b.param("C", PtrType(mma.d_dtype, "global"), noalias=True, writeonly=True)
    lane = b.thread_id_x()
    zero = b.const_i32(0)
    a_td = make_tensor_desc((wave_m, wave_k), (wave_k, 1), mma.a_dtype)
    b_td = make_tensor_desc((wave_n, wave_k), (wave_k, 1), mma.b_dtype)
    c_td = make_tensor_desc((wave_m, wave_n), (wave_n, 1), mma.d_dtype)
    acc = make_fragment(mma.c_desc, mma.c_dtype)
    fill_fragment(b, acc, 0)
    fa = load_fragment(b, a_ptr, make_window(a_td, (zero, zero)), mma.a_desc, lane)
    fb = load_fragment(b, b_ptr, make_window(b_td, (zero, zero)), mma.b_desc, lane)
    result = mma(b, fa, fb, acc)
    store_fragment(b, c_ptr, make_window(c_td, (zero, zero)), result, lane)
    b.ret()
    return b.kernel, result


def _calls(ir: str, intrinsic: str) -> list[str]:
    return [line for line in ir.splitlines() if "call" in line and intrinsic in line]


def _compile_or_skip(kernel, target: str) -> None:
    """Lower ``kernel`` at the flavor of the host clang and compile and link it for ``target`` with
    the arbiter of ``tools/check_ir_validity.py``; skip when there is no such clang, or it cannot
    target ``target`` (source tree only)."""
    tools = platform_root() / "tools"
    if not (tools / "check_ir_validity.py").is_file():
        pytest.skip("platform/tools is not installed beside rocke (source tree only)")
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import check_ir_validity
    from _llvm_identity import flavor_of_clang, tool_identity

    clang = check_ir_validity._llvm_tool("clang")
    if clang is None:
        pytest.skip("no clang to compile lowered IR")
    flavor = flavor_of_clang(tool_identity(clang))
    if flavor not in LLVM_FLAVORS:
        pytest.skip(f"clang {clang} reports no LLVM flavor rocke lowers for")
    ir = _lower_kernel_to_llvm_python(kernel, arch=target, llvm_flavor=flavor)
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "module.ll"
        source.write_text(ir)
        ok, diagnostic = check_ir_validity._compile(
            clang, source, target, Path(work) / "module.hsaco"
        )
    if not ok and check_ir_validity._target_unsupported(diagnostic):
        pytest.skip(f"host clang cannot target {target}")
    assert ok, diagnostic


# ---- emit_op reads the record ------------------------------------------------------------------


def _issuable_variants():
    """Every issuable (row, target) pair of the record, once per recorded variant."""
    for (op_id, target), entries in sorted(_COVERAGE.entries.items()):
        for entry in entries:
            if entry.status != "issuable":
                continue
            variant = entry.operand_dtypes or entry.scale_format
            label = f"{op_id}-{target}" + (f"-{'-'.join(variant)}" if variant else "")
            yield pytest.param(op_id, target, entry, id=label)


@pytest.mark.parametrize("op_id,target,entry", list(_issuable_variants()))
def test_emit_op_returns_the_recorded_backend_op(op_id, target, entry) -> None:
    # Catches any drift between the record and core's ops. A selectable integer row is pinned with
    # the operand dtypes of the variant (u8 picks the unsigned op); a scaled row names its format.
    dtypes = {}
    if entry.operand_dtypes is not None:
        dtypes = dict(zip("ab", map(dtype_to_ir_type, entry.operand_dtypes)))
    plan = TileMmaPlan(target=target, atom_override=op_id, catalog=_CATALOG, **dtypes)
    op = plan.emit_op(scale_format=entry.scale_format)
    assert op.op_id == entry.backend_op


def test_emit_op_loads_the_committed_record_once() -> None:
    # emit_op runs for every MMA a kernel issues, so it reads the committed record from the
    # once-per-process cache: repeated calls neither read nor validate the file again.
    plan = TileMma((16, 16, 16), a=F16, b=F16, c=F32, target="gfx90a").plan
    plan.emit_op()
    before = committed_backend_coverage.cache_info()
    for _ in range(3):
        plan.emit_op()
    after = committed_backend_coverage.cache_info()
    assert (after.misses, after.hits) == (before.misses, before.hits + 3)


def test_emit_op_refuses_a_pair_rocke_does_not_issue_naming_its_class() -> None:
    # The gfx950 sparse instruction's operand key is shared by a dense op, which an operand-key
    # search used to return; the record classes the pair other_instruction.
    mma = TileMma(target="gfx950", atom_override="smfmac_f32_16x16x32_f16")
    with pytest.raises(NotImplementedError) as excinfo:
        mma.emit_op()
    assert str(excinfo.value) == (
        "'smfmac_f32_16x16x32_f16' on 'gfx950' is not issued by rocke yet (backend coverage: "
        "other_instruction) -- the plan can be visualised but not issued; the classes are "
        "described in helpers/tiling/traits/README.md -- TODO: not implemented yet"
    )


def test_emit_op_takes_the_dense_op_where_scaled_ops_share_its_operands() -> None:
    # gfx1250 has a dense fp8 x fp8 16x16x128 op and two scaled ops with the same operand dtypes
    # and shape; the dense instruction issues the dense op.
    mma = TileMma((16, 16, 128), a=FP8E4M3, b=FP8E4M3, c=F32, target="gfx1250")
    assert mma.op_id == "wmma_f32_16x16x128_fp8_fp8"
    op = mma.emit_op()
    same_key = [
        other.op_id
        for other in ArchTarget.from_gfx("gfx1250").mma.ops
        if (other.a_dtype, other.b_dtype, other.c_dtype, other.m, other.n, other.k)
        == (op.a_dtype, op.b_dtype, op.c_dtype, op.m, op.n, op.k)
    ]
    assert sorted(same_key) == [
        "wmma_gfx1250_f32_16x16x128_fp8_fp8",
        "wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k16",
        "wmma_gfx1250_f32_16x16x128_fp8_fp8_scale_e8m0_e8m0_k32",
    ]
    assert op.op_id == "wmma_gfx1250_f32_16x16x128_fp8_fp8"


def test_emit_op_issues_the_unsigned_op_for_unsigned_operands() -> None:
    # One integer instruction takes signed or unsigned 8-bit operands; each pair has its own op.
    signed = TileMma((16, 16, 64), a=I8, b=I8, c=I32, target="gfx1250")
    unsigned_a = TileMma((16, 16, 64), a=U8, b=I8, c=I32, target="gfx1250")
    assert signed.op_id == unsigned_a.op_id == "wmma_i32_16x16x64_iu8"
    assert signed.emit_op().op_id == "wmma_gfx1250_i32_16x16x64_iu8"
    assert unsigned_a.emit_op().op_id == "wmma_gfx1250_i32_16x16x64_iu8_u8_i8"


_SCALED = "wmma_scale16_f32_16x16x128_f8f6f4_bf8e5m2_fp4e2m1"


def test_emit_op_of_a_scaled_instruction_takes_its_scale_format() -> None:
    mma = TileMma(target="gfx1250", atom_override=_SCALED)
    op = mma.emit_op(scale_format=("e8m0", "e5m3"))
    assert op.op_id == "wmma_gfx1250_f32_16x16x128_bf8_fp4_scale_e8m0_e5m3_k16"


def test_emit_op_of_a_scaled_instruction_needs_the_format_when_it_has_several() -> None:
    mma = TileMma(target="gfx1250", atom_override=_SCALED)
    with pytest.raises(ValueError) as excinfo:
        mma.emit_op()
    assert str(excinfo.value) == (
        f"({_SCALED!r}, 'gfx1250') is recorded per scale_format ([('e8m0', 'e8m0'), "
        "('e8m0', 'e4m3'), ('e8m0', 'e5m3')]) -- pass scale_format="
    )


def test_emit_op_refuses_a_scale_format_for_an_unscaled_instruction() -> None:
    mma = TileMma((16, 16, 16), a=F16, b=F16, c=F32, target="gfx90a")
    with pytest.raises(
        ValueError, match="recorded without variants -- pass no scale_format"
    ):
        mma.emit_op(scale_format=("e8m0", "e8m0"))


# ---- the driver's checks -----------------------------------------------------------------------


def _operand_fragments(b: IRBuilder, mma: TileMma):
    """Zero A, B and C fragments of the MMA's own descriptors and dtypes."""
    fragments = []
    for desc, dtype in (
        (mma.a_desc, mma.a_dtype),
        (mma.b_desc, mma.b_dtype),
        (mma.c_desc, mma.c_dtype),
    ):
        fragment = make_fragment(desc, dtype)
        fill_fragment(b, fragment, 0)
        fragments.append(fragment)
    return fragments


def test_driver_refuses_a_scaled_instruction_before_validating_layouts() -> None:
    # The record issues this scaled instruction, but the driver feeds no scale operands. The refusal
    # comes first, so the author is never sent to fix a layout of a tile that cannot be issued.
    mma = TileMma(target="gfx1250", atom_override=_SCALED)
    b = IRBuilder("driver_scaled")
    a, bf, acc = _operand_fragments(b, mma)
    with pytest.raises(NotImplementedError) as excinfo:
        mma(b, a, bf, acc)
    assert str(excinfo.value) == (
        f"{_SCALED!r} is a scaled instruction; the TileMma driver supplies only A, B and C "
        "operands -- the plan can be visualised, and emit_op() returns the backend op for a "
        "hand-written b.mma call wherever rocke issues it -- TODO: not implemented yet"
    )


def test_driver_refuses_a_sparse_instruction() -> None:
    # Swapped operands would fail the K-alignment check; the family refusal runs before it.
    mma = TileMma(target="gfx950", atom_override="smfmac_f32_16x16x32_f16")
    b = IRBuilder("driver_sparse")
    a, bf, acc = _operand_fragments(b, mma)
    with pytest.raises(
        NotImplementedError, match="is a sparse instruction; the TileMma"
    ):
        mma(b, bf, a, acc)


def test_driver_refuses_an_instruction_rocke_does_not_issue_before_validating_layouts() -> (
    None
):
    # A dense pair the record does not class issuable: the tied gfx11 instruction's backend op lowers
    # to the untied intrinsic. The A fragment passed as the accumulator would fail the accumulator
    # check; issue is decided first.
    pair = ("wmma_f16_16x16x16_f16_tied_w32", "gfx1100")
    assert _COVERAGE.status(*pair) == "other_instruction"
    mma = TileMma(target=pair[1], atom_override=pair[0])
    b = IRBuilder("driver_not_issued")
    a, bf, _ = _operand_fragments(b, mma)
    with pytest.raises(
        NotImplementedError, match=r"backend coverage: other_instruction"
    ):
        mma(b, a, bf, a)


# ---- a result dtype other than the accumulator's ----------------------------------------------


_BF16_RESULT = "wmma_bf16f32_16x16x32_bf16"


def test_result_dtype_selects_the_instruction() -> None:
    f32_result = TileMma((16, 16, 32), a=BF16, b=BF16, c=F32, target="gfx1250")
    bf16_result = TileMma((16, 16, 32), a=BF16, b=BF16, c=F32, d=BF16, target="gfx1250")
    assert (f32_result.op_id, f32_result.d_dtype) == ("wmma_f32_16x16x32_bf16", F32)
    assert (bf16_result.op_id, bf16_result.d_dtype) == (_BF16_RESULT, BF16)
    assert bf16_result.c_dtype == F32


def test_atom_override_takes_the_result_dtype_from_the_instruction() -> None:
    mma = TileMma(target="gfx1250", atom_override=_BF16_RESULT)
    assert (mma.a_dtype, mma.b_dtype, mma.c_dtype, mma.d_dtype) == (
        BF16,
        BF16,
        F32,
        BF16,
    )


def test_atom_override_refuses_another_result_dtype() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma(target="gfx1250", atom_override=_BF16_RESULT, d=F32)
    assert str(excinfo.value) == (
        f"MMA instruction dtype mismatch for {_BF16_RESULT!r} -- d=f32 but it returns D=bf16 "
        "-- pass d=rocke.core.ir.BF16, or omit a/b/c/d to take them from the instruction"
    )


@pytest.mark.parametrize("shape,calls", [((16, 16, 32), 1), ((32, 32, 32), 4)])
def test_bf16_result_tile_returns_a_bf16_fragment(shape, calls) -> None:
    # One C subtile, then four, each taking one instruction along K: the f32 accumulator goes in,
    # a bf16 result comes out, and it is never fed back as an accumulator.
    mma = TileMma(
        shape,
        a=BF16,
        b=BF16,
        c=F32,
        d=BF16,
        target="gfx1250",
        tiling=Tiling(atom_shape=(16, 16, 32)),
    )
    kernel, result = _wave_tile_kernel(mma, "bf16_result")
    assert result.dtype == BF16
    assert result.value.type == result.storage_type
    ir = _lower_kernel_to_llvm_python(kernel, arch="gfx1250")
    assert len(_calls(ir, "@llvm.amdgcn.wmma.bf16f32.16x16x32.bf16")) == calls


def test_bf16_result_tile_with_two_instructions_along_k_is_refused() -> None:
    mma = TileMma(
        (16, 16, 64),
        a=BF16,
        b=BF16,
        c=F32,
        d=BF16,
        target="gfx1250",
        tiling=Tiling(atom_shape=(16, 16, 32)),
    )
    assert mma.subtiles == (1, 1, 2)
    b = IRBuilder("bf16_result_k_walk")
    a, bf, acc = _operand_fragments(b, mma)
    with pytest.raises(ValueError) as excinfo:
        mma(b, a, bf, acc)
    assert str(excinfo.value) == (
        f"{_BF16_RESULT!r} returns bf16 from an f32 accumulator, so its result cannot be the "
        "next K step's accumulator -- this tile issues 2 instructions along K per C subtile; use "
        "a tile whose K is the instruction's K (32), or the f32-result instruction (omit d) and "
        "convert the finished tile"
    )


# ---- WMMA and MFMA tiles issued from the record -----------------------------------------------


@pytest.mark.parametrize(
    "target,atom,intrinsic",
    [
        ("gfx1250", (16, 16, 32), "@llvm.amdgcn.wmma.f32.16x16x32.f16"),
        ("gfx1151", (16, 16, 16), "@llvm.amdgcn.wmma.f32.16x16x16.f16"),
    ],
)
def test_f16_wmma_tile_lowers_to_wmma(target, atom, intrinsic) -> None:
    # A 32x32x32 wave tile: a grid of WMMA atoms, one call each.
    mma = TileMma(
        (32, 32, 32),
        a=F16,
        b=F16,
        c=F32,
        target=target,
        tiling=Tiling(atom_shape=atom),
    )
    kernel, _ = _wave_tile_kernel(mma, f"wmma_f16_{target}")
    ir = _lower_kernel_to_llvm_python(kernel, arch=target)
    assert len(_calls(ir, intrinsic)) == mma.plan.mfma_count
    _compile_or_skip(kernel, target)


def test_gfx11_f16_result_wmma_tile_lowers_and_compiles() -> None:
    # The 16-bit-result instruction keeps an F16 accumulator in its 8 logical slots per lane.
    mma = TileMma((16, 16, 16), a=F16, b=F16, c=F16, target="gfx1100")
    assert mma.op_id == "wmma_f16_16x16x16_f16_w32"
    assert mma.c_desc.register_count == 8
    kernel, result = _wave_tile_kernel(mma, "wmma_f16_result")
    assert result.dtype == F16
    ir = _lower_kernel_to_llvm_python(kernel, arch="gfx1100")
    assert len(_calls(ir, "@llvm.amdgcn.wmma.f16.16x16x16.f16")) == 1
    _compile_or_skip(kernel, "gfx1100")


def test_pinned_fp8_by_bf8_wmma_lowers_to_the_mixed_intrinsic() -> None:
    # The pinned mixed instruction issues fp8 x bf8, never the fp8 x fp8 one of the same shape.
    mma = TileMma(target="gfx1250", atom_override="wmma_f32_16x16x64_fp8_bf8")
    assert (mma.a_dtype, mma.b_dtype) == (FP8E4M3, BF8E5M2)
    kernel, _ = _wave_tile_kernel(mma, "wmma_fp8_bf8")
    ir = _lower_kernel_to_llvm_python(kernel, arch="gfx1250")
    assert len(_calls(ir, "@llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8")) == 1
    assert not _calls(ir, "@llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8")


def test_f32_mfma_16x16x4_issues_and_compiles() -> None:
    mma = TileMma((16, 16, 4), a=F32, b=F32, c=F32, target="gfx942")
    kernel, _ = _wave_tile_kernel(mma, "mfma_f32_16x16x4")
    ir = _lower_kernel_to_llvm_python(kernel, arch="gfx942")
    assert len(_calls(ir, "@llvm.amdgcn.mfma.f32.16x16x4f32")) == 1
    _compile_or_skip(kernel, "gfx942")
