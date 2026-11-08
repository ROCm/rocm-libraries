# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx950 tiled-2D QK paths read K from LDS in the type K was stored in.

With an fp8 KV cache and ``use_fp8_mfma_qk`` the kernel keeps K in LDS as raw
fp8 bytes and dequantizes it (``* k_scale``) in registers before the bf16 QK
MFMA. Every QK path has to read those bytes as fp8: a bf16 read of the fp8
buffer hands the MFMA pairs of fp8 bytes reinterpreted as bf16, with no
k_scale, and the scores come out wrong. The IR verifier does not catch such a
read, so this test walks the built kernel and checks every LDS load against
the element type of the buffer it reads, and that every fp8 K read reaches the
QK MFMA only through a multiply by the ``k_scale`` kernel argument. It also
pins the K slab pad: a laid-out pad is refused when Q would be staged in the
single K slot, and otherwise every K read goes through the padded slab layout;
a zero-width pad is not laid out. And it pins where the kernel stages Q to the
spec's rule, and that every fp8 LDS load claims only its byte width.

Pure Python: it builds the kernel IR and needs no GPU. The C++ engine tests
need the ``rocke_engine`` binding (``ROCKE_BUILD_PYBIND=ON``), which CI does
not build, so they skip there and run only in developer builds; the C++
evidence for these kernels in CI is the developer byte-identity run recorded
with the change.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict

import pytest
from kernels.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from rocke.tests.engines import lower_with_engine

from ._attn_tiled_2d_gfx950_qk_paths import FP8_K_IN_LDS, QK_PATHS

_BASE = dict(
    head_size=64,
    block_size=32,
    num_query_heads=32,
    num_kv_heads=32,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    tile_size=64,
)


def _walk(ops):
    for op in ops:
        yield op
        for region in op.regions:
            yield from _walk(region.ops)


def _lds_loads(kernel):
    """``(buffer element type, loaded element type)`` of every LDS vector load
    that reads a buffer allocated in the kernel."""
    loads = []
    for op in _walk(kernel.body.ops):
        if op.name != "tile.smem_load_vN":
            continue
        buffer = op.operands[0]
        if buffer.op is None or buffer.op.name != "tile.smem_alloc":
            continue
        loads.append((buffer, buffer.type.elem.name, op.attrs["elem_type"]))
    return loads


def _build(**knobs):
    spec = UnifiedAttention2DTiledSpec(**_BASE, **knobs)
    return build_unified_attention_2d_tiled(spec, arch="gfx950")


@pytest.mark.parametrize("path", sorted(QK_PATHS))
@pytest.mark.parametrize("fp8_k_in_lds", [False, True], ids=["bf16_k", "fp8_k"])
def test_lds_loads_read_the_buffer_element_type(path, fp8_k_in_lds):
    knobs = dict(QK_PATHS[path], **(FP8_K_IN_LDS if fp8_k_in_lds else {}))
    loads = _lds_loads(_build(**knobs))
    assert loads, "expected LDS loads in the kernel"
    wrong = sorted(
        {
            f"{buffer.name}: {stored} buffer read as {read}"
            for buffer, stored, read in loads
            if read != stored
        }
    )
    assert not wrong, f"{path}: LDS loads that reinterpret the buffer: {wrong}"


@pytest.mark.parametrize("path", sorted(QK_PATHS))
def test_every_qk_path_reads_fp8_k_as_fp8(path):
    # The 32x32 non-transposed path used to read the fp8 K buffer as bf16.
    kernel = _build(**QK_PATHS[path], **FP8_K_IN_LDS)
    k_reads = [
        read
        for buffer, stored, read in _lds_loads(kernel)
        if stored == "fp8e4m3" and "Klds" in buffer.name
    ]
    assert k_reads, f"{path}: no QK read of the fp8 K buffer"
    assert set(k_reads) == {"fp8e4m3"}


@pytest.mark.parametrize("path", sorted(QK_PATHS))
def test_fp8_k_lds_loads_claim_their_byte_width(path):
    # Lanes 32-63 of a 32x32 K read start 8 bytes into a 16-byte line, so an
    # <8 x i8> load may claim only 8 bytes of alignment; a 16-byte claim is
    # undefined behaviour in LLVM.
    spec = UnifiedAttention2DTiledSpec(**_BASE, **QK_PATHS[path], **FP8_K_IN_LDS)
    text = lower_with_engine(
        build_unified_attention_2d_tiled(spec, arch="gfx950"), "gfx950", engine="python"
    )
    loads = re.findall(
        r"= load (?:volatile )?<(\d+) x i8>, ptr addrspace\(3\) [^,]+, align (\d+)",
        text,
    )
    assert loads, f"{path}: no fp8 LDS vector load"
    over_claimed = sorted({(int(n), int(a)) for n, a in loads if int(a) != int(n)})
    assert not over_claimed, f"{path}: (elements, align) over-claimed: {over_claimed}"


def _users(kernel):
    """Map ``id(value)`` to the ops that take ``value`` as an operand."""
    users = defaultdict(list)
    for op in _walk(kernel.body.ops):
        for value in op.operands:
            users[id(value)].append(op)
    return users


def _forward(starts, users, stop):
    """The ops reached from the results of ``starts`` through operand uses,
    not continuing past an op for which ``stop(op)`` is true."""
    seen, reached, todo = set(), [], list(starts)
    while todo:
        op = todo.pop()
        for result in op.results:
            for user in users[id(result)]:
                if id(user) in seen:
                    continue
                seen.add(id(user))
                reached.append(user)
                if not stop(user):
                    todo.append(user)
    return reached


@pytest.mark.parametrize("path", sorted(QK_PATHS))
def test_fp8_k_is_scaled_by_k_scale_before_the_mfma(path):
    # Reading the fp8 bytes as fp8 is not enough: each value has to be
    # multiplied by k_scale before it reaches the MFMA, or the scores are off
    # by that factor.
    kernel = _build(**QK_PATHS[path], **FP8_K_IN_LDS)
    assert "k_scale" in [param.name for param in kernel.params]
    users = _users(kernel)

    def is_k_scale(value):
        # A kernel parameter is a value with no defining op, named after it.
        return value.op is None and value.name == "%k_scale"

    def scales(op):
        return op.name == "arith.fmul" and any(map(is_k_scale, op.operands))

    k_reads = [
        op
        for op in _walk(kernel.body.ops)
        if op.name == "tile.smem_load_vN"
        and op.attrs["elem_type"] == "fp8e4m3"
        and "Klds" in op.operands[0].name
    ]
    assert k_reads, f"{path}: no QK read of the fp8 K buffer"
    for read in k_reads:
        reached = _forward([read], users, stop=scales)
        unscaled = [op.name for op in reached if op.name == "tile.mma"]
        assert not unscaled, f"{path}: fp8 K reaches the MFMA without * k_scale"
        multiplies = [op for op in reached if scales(op)]
        assert multiplies, f"{path}: fp8 K is never multiplied by k_scale"
        mfmas = _forward(multiplies, users, stop=lambda op: op.name == "tile.mma")
        assert any(
            op.name == "tile.mma" for op in mfmas
        ), f"{path}: scaled K does not reach an MFMA"


_LAID_OUT_K_PAD = dict(
    use_k_single_buffer=True, use_kq_lds_pad=True, kq_lds_pad_halves=8
)


@pytest.mark.parametrize("path", sorted(QK_PATHS))
def test_laid_out_k_pad_refuses_q_staged_in_the_k_slot(path):
    # Q staged in the single K slot is written and read unpadded, which does
    # not match a laid-out K slab pad.
    message = (
        "KQ LDS pad failed - use_q_reread=False, use_q_direct_reg=False, "
        "expected one of them True (transposed 32x32 path) or "
        "use_kq_lds_pad=False (Q staged in the padded single K slot is read "
        "unpadded)"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        UnifiedAttention2DTiledSpec(**_BASE, **QK_PATHS[path], **_LAID_OUT_K_PAD)


def _defining(value, name):
    """The op that defines ``value``, asserted to be a ``name`` op."""
    assert value.op is not None and value.op.name == name, (
        f"expected {value.name} to come from {name}, "
        f"got {value.op.name if value.op else 'a kernel parameter'}"
    )
    return value.op


def _constant(value):
    return _defining(value, "arith.constant").attrs["value"]


@pytest.mark.parametrize("q_source", ["use_q_reread", "use_q_direct_reg"])
def test_laid_out_k_pad_reads_k_through_the_padded_slabs(q_source):
    head, tile, pad = _BASE["head_size"], _BASE["tile_size"], 8
    slab_rows = 64 * 16 // 2 // head  # one wave's 1024-B async-DMA slab of bf16
    spec = UnifiedAttention2DTiledSpec(
        **_BASE, **QK_PATHS["32x32 transposed"], **_LAID_OUT_K_PAD, **{q_source: True}
    )
    assert spec.kq_lds_pad_slab_rows == slab_rows
    kernel = build_unified_attention_2d_tiled(spec, arch="gfx950")

    k_lds = [
        op.results[0]
        for op in _walk(kernel.body.ops)
        if op.name == "tile.smem_alloc" and "Klds" in op.results[0].name
    ]
    assert [buffer.type.shape for buffer in k_lds] == [
        (1, tile // slab_rows, slab_rows * head + pad)
    ]
    k_reads = [
        op
        for op in _walk(kernel.body.ops)
        if op.name == "tile.smem_load_vN" and op.operands[0] is k_lds[0]
    ]
    assert k_reads, "no QK read of the padded K buffer"
    for read in k_reads:
        # (buf, row, col) -> (buf, row // SLAB_ROWS, (row % SLAB_ROWS) * HD + col)
        slab = _defining(read.operands[2], "arith.div")
        k_row = slab.operands[0]
        assert _constant(slab.operands[1]) == slab_rows
        col_in_slab = _defining(read.operands[3], "arith.add")
        row_offset = _defining(col_in_slab.operands[0], "arith.mul")
        row_in_slab = _defining(row_offset.operands[0], "arith.mod")
        assert row_in_slab.operands[0] is k_row
        assert _constant(row_in_slab.operands[1]) == slab_rows
        assert _constant(row_offset.operands[1]) == head


def test_k_pad_the_kernel_cannot_lay_out_is_inert():
    # Without a single K slot the pad is not laid out, so Q may alias K.
    spec = UnifiedAttention2DTiledSpec(
        **_BASE, use_kq_lds_pad=True, kq_lds_pad_halves=8
    )
    assert spec.kq_lds_pad_slab_rows == 0
    assert spec.kq_lds_pad_off_reason == (
        "the KQ LDS pad is only laid out for a single-K schedule"
    )


def test_zero_width_k_pad_builds_the_unpadded_kernel():
    # A zero-width pad lays out no pad, so Q may still be staged in the single
    # K slot and the kernel is the one built without the knob.
    padded = UnifiedAttention2DTiledSpec(
        **_BASE, **dict(_LAID_OUT_K_PAD, kq_lds_pad_halves=0)
    )
    assert padded.kq_lds_pad_slab_rows == 0
    assert padded.kq_lds_pad_off_reason == "the KQ LDS pad width is 0"
    assert padded.q_staged_in_k_lds
    unpadded = UnifiedAttention2DTiledSpec(**_BASE, use_k_single_buffer=True)

    def lowered(spec):
        kernel = build_unified_attention_2d_tiled(spec, arch="gfx950")
        return lower_with_engine(kernel, "gfx950", engine="python")

    assert lowered(padded) == lowered(unpadded)


def test_negative_k_pad_width_is_refused():
    with pytest.raises(
        ValueError,
        match=(
            r"^KQ LDS pad width failed - kq_lds_pad_halves=-8, expected >= 0 "
            r"\(0 lays out no pad\)$"
        ),
    ):
        UnifiedAttention2DTiledSpec(**_BASE, use_kq_lds_pad=True, kq_lds_pad_halves=-8)


_TRANSPOSED = QK_PATHS["32x32 transposed"]

# (knobs, whether the spec stages Q in K_lds)
_Q_STAGING = {
    "16x16 bf16 K": ({}, True),
    "16x16 fp8 K": (FP8_K_IN_LDS, False),
    "Q larger than both K slots": (
        {"num_warps": 8, "tile_size": 32, "head_size": 128},
        False,
    ),
    "transposed, Q re-read": (dict(_TRANSPOSED, use_q_reread=True), False),
    "transposed, Q in registers": (dict(_TRANSPOSED, use_q_direct_reg=True), False),
}


@pytest.mark.parametrize("case", sorted(_Q_STAGING))
def test_kernel_stages_q_where_the_spec_says(case):
    knobs, staged = _Q_STAGING[case]
    spec = UnifiedAttention2DTiledSpec(**dict(_BASE, **knobs))
    assert spec.q_staged_in_k_lds is staged
    kernel = build_unified_attention_2d_tiled(spec, arch="gfx950")
    has_q_slab = any(
        op.name == "tile.smem_alloc" and "Qlds" in op.results[0].name
        for op in _walk(kernel.body.ops)
    )
    # Q gets its own slab unless it is staged in K_lds or gathered to registers.
    assert has_q_slab is not (staged or spec.use_q_direct_reg)


@pytest.mark.parametrize("path", sorted(QK_PATHS))
def test_cpp_engine_emits_the_same_llvm_ir(path):
    # The C++ builder serves these kernels without Python; it must read and
    # scale fp8 K exactly as the Python builder does. Developer builds only:
    # CI does not build the rocke_engine binding.
    engine = pytest.importorskip("rocke_engine")
    spec = UnifiedAttention2DTiledSpec(**_BASE, **QK_PATHS[path], **FP8_K_IN_LDS)
    expected = lower_with_engine(
        build_unified_attention_2d_tiled(spec, arch="gfx950"), "gfx950", engine="python"
    )
    assert (
        engine.gfx950_attention_tiled_2d_lower_llvm(asdict(spec), "gfx950") == expected
    )


@pytest.mark.parametrize("knob", ["use_kq_lds_pad", "use_kq_xor_swizzle"])
def test_cpp_engine_refuses_the_python_only_k_knobs(knob):
    # The C++ builder has neither the KQ slab pad nor the KQ XOR swizzle (a
    # probe knob the Python builder reads off a spec subclass); it refuses each
    # rather than lower an unpadded, unswizzled K read. Developer builds only
    # (see the module note).
    engine = pytest.importorskip("rocke_engine")
    spec = UnifiedAttention2DTiledSpec(
        **_BASE, **_TRANSPOSED, **_LAID_OUT_K_PAD, use_q_reread=True
    )
    knobs = asdict(spec) | {
        "use_kq_lds_pad": knob == "use_kq_lds_pad",
        "use_kq_xor_swizzle": knob == "use_kq_xor_swizzle",
    }
    message = (
        f"rocke_engine tiled-2d: {knob}=True is not implemented in the C++ "
        "engine; build this spec with the Python engine or set it to False"
    )
    with pytest.raises(NotImplementedError, match=f"^{re.escape(message)}$"):
        engine.gfx950_attention_tiled_2d_lower_llvm(knobs, "gfx950")
