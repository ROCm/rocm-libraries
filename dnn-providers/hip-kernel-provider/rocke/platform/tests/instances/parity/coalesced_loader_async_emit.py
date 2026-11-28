# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for ``CoalescedTileLoader.load_async_to_lds``: the
gfx1250 per-lane async global->LDS copy with its unsigned range check and the
zero-fill of out-of-range chunks.

One case per element type the loader admits (f16, bf16, f32, i32) at the
16-byte copy, plus the 4- and 8-byte copies, and an f16 loader filling a bf16
tile (same width, so admitted). The cases after them must be refused by both
engines: an element type the loader has no width for, a 32-byte chunk, and a
tile or a source pointer whose element width differs from the loader's.

Config index = case. Every builder call is its own statement in the order the
C emitter issues it: each call consumes an SSA id, so a reordering emits the
same IR under other names.
"""

from rocke.core.ir import BF16, F16, F32, I8, I32, IRBuilder, PtrType
from rocke.helpers.loads import CoalescedTileLoader

from _emit_common import run_emit

ARCH = "gfx1250"
TILE_ROWS = 16
TILE_COLS = 32
BLOCK_SIZE = 64

# (name, loader elem_dtype, load_vec, LDS tile dtype, source pointer dtype)
CASES = (
    ("f16_b128", F16, 8, F16, F16),
    ("bf16_b128", BF16, 8, BF16, BF16),
    ("f32_b128", F32, 4, F32, F32),
    ("i32_b128", I32, 4, I32, I32),
    ("f16_b32", F16, 2, F16, F16),
    ("f32_b64", F32, 2, F32, F32),
    # an f16 loader filling a bf16 tile: the widths match
    ("f16_into_bf16", F16, 8, BF16, BF16),
    # rejected: no loader width for i8
    ("i8", I8, 8, I8, I8),
    # rejected: 8 x 4 B is a 32-byte chunk
    ("f32_b256", F32, 8, F32, F32),
    # rejected: an f16 loader filling an f32 tile
    ("f16_into_f32_tile", F16, 8, F32, F16),
    # rejected: an f32 loader reading an f16 source
    ("f32_from_f16_ptr", F32, 4, F32, F16),
)


def build(b: IRBuilder, case) -> None:
    _, elem_dtype, load_vec, tile_dtype, src_dtype = case
    loader = CoalescedTileLoader(
        tile_rows=TILE_ROWS,
        tile_cols=TILE_COLS,
        block_size=BLOCK_SIZE,
        load_vec=load_vec,
        elem_dtype=elem_dtype,
    )
    src = b.param("src", PtrType(src_dtype, "global"), readonly=True, align=16)
    num_bytes = b.param("num_bytes", I32)
    tid = b.thread_id_x()
    tile = b.smem_alloc(tile_dtype, [TILE_ROWS, TILE_COLS], name_hint="tile")
    c_cols = b.const_i32(TILE_COLS)

    def descriptor(b_, row, col):
        return b_.add(b_.mul(row, c_cols), col), None

    loader.load_async_to_lds(
        b,
        tid=tid,
        smem_dst=tile,
        descriptor=descriptor,
        ptr=src,
        num_bytes=num_bytes,
    )
    b.ret()


def _spec(idx):
    if not 0 <= idx < len(CASES):
        raise SystemExit(f"unknown config index {idx}")
    return CASES[idx], ARCH


def _build(case, *, arch):
    b = IRBuilder("coalesced_loader_async")
    b.kernel.attrs["max_workgroup_size"] = BLOCK_SIZE
    build(b, case)
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
