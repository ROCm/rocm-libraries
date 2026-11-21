# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for ``RegTransposeTileLoader``.

Each case builds the smallest kernel that drives one path of the loader: the
whole-tile ``load`` at every width (8, 4, 2 and the scalar 1), bf16, a
descriptor with a validity predicate, the split ``load_global`` /
``store_lds`` pair, partial participation (a tile with fewer micro-tiles than
threads) with and without the guarded reads (and with unmasked reads of a full
tile), a run descriptor and a column swizzle. Six cases must be rejected by both
engines: a width outside ``REGT_VECS``, a 32-bit element type, the guard where
it cannot apply (a full tile, a width of 1, unmasked reads) and the
``store_lds`` of an unmasked partial tile.

Config index = case * len(ARCHES) + arch_index. Every builder call is its own
statement in the order the C emitter issues it: each call consumes an SSA id,
so a reordering emits the same IR under other names.
"""

from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, PtrType
from rocke.helpers.loads import REGT_K_RUN, RegTransposeTileLoader

from _emit_common import run_emit

ARCHES = ("gfx1201", "gfx950")

# (name, tile_rows, tile_cols, block_size, wave_size, vec, dtype, path,
#  valid, run, swizzle). path: "load", "split", "guard" (split, guard=True),
#  "split_unmasked" (split, masked=False) or "guard_unmasked" (split,
#  masked=False, guard=True).
CASES = (
    ("full8", 64, 64, 64, 32, 8, "f16", "load", False, False, False),
    ("full4_bf16", 64, 32, 64, 32, 4, "bf16", "load", False, False, False),
    ("multi2", 64, 64, 64, 32, 2, "f16", "load", True, False, False),
    ("scalar1", 16, 32, 64, 32, 1, "f16", "load", True, False, False),
    ("split8", 64, 64, 64, 32, 8, "f16", "split", True, False, False),
    ("partial8", 64, 32, 128, 32, 8, "f16", "load", True, False, False),
    ("partial8_split", 64, 32, 128, 32, 8, "bf16", "split", True, False, False),
    ("partial8_guard", 64, 32, 128, 32, 8, "f16", "guard", True, False, False),
    ("run4", 64, 64, 64, 32, 4, "f16", "load", True, True, False),
    ("run8_guard", 64, 32, 128, 32, 8, "f16", "guard", True, True, False),
    ("swz8", 64, 64, 64, 32, 8, "f16", "split", False, False, True),
    ("swz1", 16, 32, 64, 32, 1, "f16", "load", False, False, True),
    ("wave64_2", 64, 64, 256, 64, 2, "f16", "load", False, False, False),
    # rejected: width 3
    ("vec3", 48, 32, 64, 32, 3, "f16", "load", False, False, False),
    # rejected: 32-bit elements
    ("f32", 64, 64, 64, 32, 8, "f32", "load", False, False, False),
    # rejected: guard on a full-participation tile
    ("guard_full", 64, 64, 64, 32, 8, "f16", "guard", True, False, False),
    # rejected: guard at width 1
    ("guard_vec1", 16, 32, 128, 32, 1, "f16", "guard", True, False, False),
    # rejected: store_lds of an unmasked partial tile
    (
        "partial8_unmasked",
        64,
        32,
        128,
        32,
        8,
        "f16",
        "split_unmasked",
        True,
        False,
        False,
    ),
    # rejected: guard with unmasked reads
    ("guard_unmasked", 64, 32, 128, 32, 8, "f16", "guard_unmasked", True, False, False),
    # unmasked reads of a full-participation tile (masked is moot there)
    ("full8_unmasked", 64, 64, 64, 32, 8, "f16", "split_unmasked", True, False, False),
)

_DTYPES = {"f16": F16, "bf16": BF16, "f32": F32}


def _spec(idx):
    if not 0 <= idx < len(CASES) * len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    return CASES[idx // len(ARCHES)], ARCHES[idx % len(ARCHES)]


def build(b: IRBuilder, case) -> None:
    _, rows, cols, block, wave, vec, dt, path, valid, run, swizzle = case
    dtype = _DTYPES[dt]
    x = b.param("X", PtrType(dtype, "global"))
    n_bytes = b.param("N_bytes", I32)
    rsrc = b.buffer_rsrc(x, n_bytes)
    tid = b.thread_id_x()
    tile = b.smem_alloc(dtype, [rows, cols], name_hint="tile")
    # The free axis (rows) is contiguous; K columns are ld elements apart.
    c_ld = b.const_i32(rows + 8)
    c_lim = b.const_i32(cols - 3)
    kc = cols // REGT_K_RUN

    def descriptor(b_, row, col):
        m = b_.mul(col, c_ld)
        off = b_.add(row, m)
        if not valid:
            return off, None
        ok = b_.cmp_lt(col, c_lim)
        return off, ok

    def run_descriptor(b_, row, col):
        m = b_.mul(col, c_ld)
        base = b_.add(row, m)
        out = []
        for j in range(REGT_K_RUN):
            if j:
                step = b_.const_i32(j * (rows + 8))
                off = b_.add(base, step)
            else:
                off = base
            if j % 2:
                c_j = b_.const_i32(j)
                cj = b_.add(col, c_j)
                ok = b_.cmp_lt(cj, c_lim)
            else:
                ok = None
            out.append((off, ok))
        return out

    def col_swizzle(b_, row, col):
        c_mask = b_.const_i32(kc - 1)
        f = b_.land(row, c_mask)
        c8 = b_.const_i32(8)
        sh = b_.mul(f, c8)
        return b_.xor(col, sh)

    kw = {"col_swizzle": col_swizzle} if swizzle else {}
    ldr = RegTransposeTileLoader(
        tile_rows=rows,
        tile_cols=cols,
        block_size=block,
        wave_size=wave,
        vec=vec,
        elem_dtype=dtype,
        **kw,
    )
    run_kw = {"run_descriptor": run_descriptor} if run else {}
    if path == "load":
        ldr.load(b, tid=tid, smem_dst=tile, descriptor=descriptor, rsrc=rsrc, **run_kw)
    else:
        staged = ldr.load_global(
            b,
            tid=tid,
            descriptor=descriptor,
            rsrc=rsrc,
            masked=not path.endswith("_unmasked"),
            guard=path.startswith("guard"),
            **run_kw,
        )
        ldr.store_lds(b, smem_dst=tile, staged=staged)
    b.ret()


def _build(case, *, arch):
    b = IRBuilder("reg_transpose_loader")
    b.kernel.attrs["max_workgroup_size"] = case[3]
    build(b, case)
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
