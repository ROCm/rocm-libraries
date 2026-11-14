# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for ``CoalescedTileLoader.col_xor``: the XOR-swizzled
LDS stores (``lds_col_xor_params`` / ``swizzle_lds_col``).

The swizzle is otherwise only reached through the kernels that opt into it,
which drive a narrow slice of the loader. Each case here builds the smallest
kernel that takes one store path of the loader with a swizzle: the ``"col"``
vector store (two row widths, so both a zero and a non-zero ``row_shift``),
the ``"row"`` transpose-on-store scatter, the scalar store, the split
``load_global`` / ``store_lds`` pair and the address-prefetch split. One case
must be rejected by both engines (``load_vec`` wider than the swizzle chunk),
and one leaves ``col_xor`` unset as the unswizzled control. The cases after it
set an explicit tuple both engines must refuse: an i32 shift by 32 (row or
flip), rows that do not hold whole XOR groups and a row that is not a
whole number of chunks.

Config index = case * len(ARCHES) + arch_index. Every builder call is its own
statement in the order the C emitter issues it: each call consumes an SSA id,
so a reordering emits the same IR under other names.
"""

import dataclasses

from rocke.core.ir import F16, I32, IRBuilder, PtrType
from rocke.helpers.loads import CoalescedTileLoader, lds_col_xor_params

from _emit_common import run_emit

ARCHES = ("gfx1201", "gfx950")

# (name, tile_rows, tile_cols, block_size, load_vec or 0 for from_tile,
#  vector_axis, swizzle chunk or 0 for no col_xor, path, explicit col_xor)
CASES = (
    ("col64", 64, 64, 256, 0, "col", 8, "load", None),
    ("col16", 32, 16, 64, 0, "col", 8, "load", None),
    ("row32", 64, 32, 256, 8, "row", 8, "load", None),
    ("scalar16", 16, 16, 256, 0, "col", 8, "load", None),
    ("split64", 64, 64, 256, 0, "col", 8, "split", None),
    ("slots64", 64, 64, 256, 0, "col", 8, "slots", None),
    # rejected: a 4-element chunk under an 8-wide vector
    ("narrow_chunk", 64, 64, 256, 0, "col", 4, "load", None),
    # control: no col_xor
    ("plain64", 64, 64, 256, 0, "col", 0, "load", None),
    # rejected: i32 shifts by 32 (row_shift, then flip_shift)
    ("shift32", 64, 64, 256, 8, "col", 0, "load", (32, 7, 3, 3)),
    ("flip32", 64, 64, 256, 8, "col", 0, "load", (0, 7, 3, 32)),
    # rejected: 6 chunks per row under the 32-column swizzle's groups of 4
    ("partial48", 16, 48, 64, 8, "col", 0, "load", (1, 3, 3, 3)),
    # rejected: one 64-column chunk per row; the flip bit needs two
    ("one_chunk64", 64, 64, 256, 8, "col", 0, "load", (0, 0, 6, 3)),
    # rejected: a ragged 36-column row (4 whole chunks of 8 plus 4 columns)
    ("ragged36", 64, 36, 64, 4, "col", 0, "load", (1, 3, 3, 3)),
)


def _loader(case) -> CoalescedTileLoader:
    _, rows, cols, block, vec, axis, chunk, _, col_xor = case
    if vec:
        ldr = CoalescedTileLoader(
            tile_rows=rows,
            tile_cols=cols,
            block_size=block,
            load_vec=vec,
            vector_axis=axis,
        )
    else:
        ldr = CoalescedTileLoader.from_tile(
            tile_rows=rows, tile_cols=cols, block_size=block
        )
    if chunk:
        ldr = dataclasses.replace(ldr, col_xor=lds_col_xor_params(cols, chunk))
    if col_xor is not None:
        ldr = dataclasses.replace(ldr, col_xor=col_xor)
    return ldr


def build(b: IRBuilder, case) -> None:
    _, rows, cols, _, _, _, _, path, _ = case
    ldr = _loader(case)
    x = b.param("X", PtrType(F16, "global"))
    n_bytes = b.param("N_bytes", I32)
    rsrc = b.buffer_rsrc(x, n_bytes)
    tid = b.thread_id_x()
    tile = b.smem_alloc(F16, [rows, cols], name_hint="tile")
    c_cols = b.const_i32(cols)

    def descriptor(b_, row, col):
        return b_.add(b_.mul(row, c_cols), col), None

    if path == "load":
        ldr.load(b, tid=tid, smem_dst=tile, descriptor=descriptor, rsrc=rsrc)
    elif path == "split":
        staged = ldr.load_global(b, tid=tid, descriptor=descriptor, rsrc=rsrc)
        ldr.store_lds(b, smem_dst=tile, staged=staged)
    else:  # slots
        slots = ldr.plan_slots(b, tid=tid)
        offsets = ldr.slot_offsets(b, slots=slots, descriptor=descriptor)
        staged = ldr.issue_slots(b, rsrc=rsrc, slots=slots, offsets=offsets)
        ldr.store_lds(b, smem_dst=tile, staged=staged)
    b.ret()


def _spec(idx):
    if not 0 <= idx < len(CASES) * len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    return CASES[idx // len(ARCHES)], ARCHES[idx % len(ARCHES)]


def _build(case, *, arch):
    b = IRBuilder("coalesced_loader_col_xor")
    b.kernel.attrs["max_workgroup_size"] = case[3]
    build(b, case)
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
