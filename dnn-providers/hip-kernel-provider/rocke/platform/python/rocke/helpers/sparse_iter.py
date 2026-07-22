# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sparse-attention K-iterators (Jenga + VSA).

CK Tile's ``50_sparse_attn`` example ships two distinct sparse
attention patterns:

* **Jenga (block-sparse)** -- a one-hot bitmap
 ``Mask[q_block, k_block] in {0, 1}`` selects which K-blocks each
 Q-block attends to. Reads dense K / V; skips the entire inner
 K-tile loop body when the bit is 0.

* **VSA (variable-size attention)** -- each Q-block has its own
 variable-length list of K-block indices stored in an indirect
 LUT (``BlockLut[q_block, slot]``, length ``BlockCount[q_block]``).
 Walks only the listed K-blocks.

The two share the same general shape but differ in how the
K-iteration is decoded. This module exposes two iterator helpers
keyed by the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..core.ir import IRBuilder, PtrType, Value


__all__ = [
    "BlockSparseSpec",
    "VsaSparseSpec",
    "block_sparse_iter",
    "load_block_count",
    "vsa_lut_iter",
]


@dataclass(frozen=True)
class BlockSparseSpec:
    """One block-sparse mask configuration.

    Attributes
    ----------
    num_k_blocks
    Compile-time upper bound on the number of K-blocks; drives
    the ``scf.for`` trip count.
    stride_q_block
    Element stride to advance one q_block in the mask tensor.
    """

    num_k_blocks: int
    stride_q_block: int

    def __post_init__(self):
        if self.num_k_blocks <= 0:
            raise ValueError(
                f"BlockSparseSpec.num_k_blocks must be > 0 (got {self.num_k_blocks})"
            )
        if self.stride_q_block <= 0:
            raise ValueError(
                f"BlockSparseSpec.stride_q_block must be > 0 "
                f"(got {self.stride_q_block})"
            )


@dataclass(frozen=True)
class VsaSparseSpec:
    """One variable-size attention configuration.

    Attributes
    ----------
    max_blocks_per_q
    Compile-time upper bound on the per-q_block block count;
    drives the ``scf.for`` trip count.
    stride_q_block
    Element stride into the LUT to advance one q_block.
    """

    max_blocks_per_q: int
    stride_q_block: int

    def __post_init__(self):
        if self.max_blocks_per_q <= 0:
            raise ValueError(
                f"VsaSparseSpec.max_blocks_per_q must be > 0 "
                f"(got {self.max_blocks_per_q})"
            )
        if self.stride_q_block <= 0:
            raise ValueError(
                f"VsaSparseSpec.stride_q_block must be > 0 (got {self.stride_q_block})"
            )


def _validate_i8_ptr(ptr: Value, name: str) -> None:
    if not isinstance(ptr.type, PtrType):
        raise ValueError(f"{name} must be a typed pointer")
    if ptr.type.pointee.name != "i8":
        raise ValueError(f"{name} must be ptr<i8>, got ptr<{ptr.type.pointee.name}>")


def _validate_i32_ptr(ptr: Value, name: str) -> None:
    if not isinstance(ptr.type, PtrType):
        raise ValueError(f"{name} must be a typed pointer")
    if ptr.type.pointee.name != "i32":
        raise ValueError(f"{name} must be ptr<i32>, got ptr<{ptr.type.pointee.name}>")


def block_sparse_iter(
    b: IRBuilder,
    *,
    mask_ptr: Value,
    q_block: Value,
    spec: BlockSparseSpec,
    body: Callable[[Value], None],
) -> None:
    """Iterate K-blocks driven by ``MaskBitmap[q_block, :]``.

    For each ``k_block in [0, num_k_blocks)``:

    mask_byte = MaskBitmap[q_block * stride_q_block + k_block]
    if mask_byte != 0:
    body(k_block)

    The runtime ``if`` skips zeroed K-blocks; AMDGPU's branch
    predictor handles the dynamic check well because the mask byte
    is L1-resident.
    """
    _validate_i8_ptr(mask_ptr, "mask_ptr")
    from ..core.ir import I8

    base = b.mul(q_block, b.const_i32(spec.stride_q_block))

    loop = b.scf_for(
        b.const_i32(0),
        b.const_i32(spec.num_k_blocks),
        b.const_i32(1),
        iv_name="kb",
    )
    with loop as k_block:
        mask_off = b.add(base, k_block)
        mask_byte = b.global_load(mask_ptr, mask_off, I8)
        zero_i8 = b._op(  # noqa: SLF001 - i8 constant factory
            "arith.constant",
            result_types=[mask_byte.type],
            attrs={"value": 0, "ity": "i8"},
            result_name_hint="c",
        ).result
        keep = b.cmp_ne(mask_byte, zero_i8)
        with b.scf_if(keep):
            body(k_block)


def load_block_count(b: IRBuilder, block_count_ptr: Value, q_block: Value) -> Value:
    """``BlockCount[q_block]`` -- one i32 global load.

    Drives the VSA loop trip count.
    """
    _validate_i32_ptr(block_count_ptr, "block_count_ptr")
    return b.global_load_i32(block_count_ptr, q_block)


def vsa_lut_iter(
    b: IRBuilder,
    *,
    lut_ptr: Value,
    block_count_ptr: Value,
    q_block: Value,
    spec: VsaSparseSpec,
    body: Callable[[Value, Value], None],
) -> None:
    """Iterate K-blocks driven by ``BlockLut[q_block, :]``.

    For each ``slot in [0, BlockCount[q_block])``:

    physical_k_block = BlockLut[q_block * stride_q_block + slot]
    body(slot, physical_k_block)

    Body receives both the slot index (for any per-slot bookkeeping)
    and the decoded physical k_block (for the K/V loads). The upper
    bound is dynamic but the kernel author bakes in
    ``max_blocks_per_q`` for the compile-time loop trip count.
    """
    _validate_i32_ptr(lut_ptr, "lut_ptr")
    _validate_i32_ptr(block_count_ptr, "block_count_ptr")
    base = b.mul(q_block, b.const_i32(spec.stride_q_block))
    block_count = load_block_count(b, block_count_ptr, q_block)

    loop = b.scf_for(
        b.const_i32(0),
        b.const_i32(spec.max_blocks_per_q),
        b.const_i32(1),
        iv_name="slot",
    )
    with loop as slot:
        in_range = b.cmp_lt(slot, block_count)
        with b.scf_if(in_range):
            slot_off = b.add(base, slot)
            physical_k_block = b.global_load_i32(lut_ptr, slot_off)
            body(slot, physical_k_block)
