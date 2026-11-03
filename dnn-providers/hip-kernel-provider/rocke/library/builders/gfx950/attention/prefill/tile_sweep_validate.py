# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side build check for the gfx950 dense tile sweep.

The sweep that produces the tile-selection heuristic times one kernel per
(shape, tile). Building those kernels needs no GPU, so this runs the build and
comgr compile for every pair first and reports which ones fail. A config that
cannot compile is then found at a desk rather than on the device, and the GPU
window spends its time measuring instead of triaging.

The tiles are the ones the shipped hipDNN catalog carries, seven at D64 and six
at D128; ``catalog_tiles`` is the single source for them here. Specs are built
through the same ``build_attention_dense`` the sweep and the engine use, so a
build that passes here is the build that will be timed.

Shapes come from a JSON list of objects with ``batch``, ``seqlen_q``,
``seqlen_kv``, ``num_query_heads``, ``num_kv_heads``, ``head_size``, ``dtype``
and ``causal``. ``--emit-grid`` writes a provisional grid in that format for use
until the canonical suite is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The tiles the shipped catalog carries, per head size.
CATALOG_TILES = {
    64: (
        (128, 32),
        (128, 64),
        (128, 128),
        (256, 32),
        (256, 64),
        (256, 128),
        (256, 256),
    ),
    128: ((128, 32), (128, 64), (128, 128), (256, 32), (256, 64), (256, 128)),
}


def catalog_tiles(head_size: int):
    """The legal (block_m, block_n) pairs for ``head_size``, or () if unknown."""
    return CATALOG_TILES.get(int(head_size), ())


def spec_for(shape, block_m: int, block_n: int):
    """A ``Gfx950AttentionDenseSpec`` for one (shape, tile), non-persistent grid
    body: the catalog ships no persistent, windowed or split-KV kernel."""
    from kernels.gfx950.attention_dense import Gfx950AttentionDenseSpec

    sq, sk = int(shape["seqlen_q"]), int(shape["seqlen_kv"])
    return Gfx950AttentionDenseSpec(
        batch=int(shape["batch"]),
        seqlen_q=sq,
        seqlen_kv=sk,
        num_query_heads=int(shape["num_query_heads"]),
        num_kv_heads=int(shape["num_kv_heads"]),
        head_size=int(shape["head_size"]),
        causal=bool(shape["causal"]),
        dtype=str(shape["dtype"]),
        block_m=block_m,
        block_n=block_n,
        persistent=False,
        ragged=(sq % block_m != 0 or sk % block_n != 0),
    )


def shape_id(shape) -> str:
    return (
        f"b{shape['batch']}_sq{shape['seqlen_q']}_sk{shape['seqlen_kv']}"
        f"_hq{shape['num_query_heads']}_kv{shape['num_kv_heads']}"
        f"_d{shape['head_size']}_{shape['dtype']}"
        f"_{'causal' if shape['causal'] else 'full'}"
    )


def divides(shape, block_m: int, block_n: int) -> bool:
    """Whether the catalog can serve ``shape`` at this tile.

    A tile that does not divide both lengths builds a ragged kernel, and ragged
    specs bake the shape, so they are one kernel per shape rather than one per
    tile and none are in the 52-kernel catalog. The engine enforces the same
    rule (``tileDivides``), so a non-dividing tile is not a slow choice, it is
    not a choice at all.
    """
    return (
        int(shape["seqlen_q"]) % block_m == 0 and int(shape["seqlen_kv"]) % block_n == 0
    )


def validate(
    shapes, *, compile_kernels: bool, arch: str = "gfx950", ragged: bool = False
):
    """Build (and optionally compile) every servable (shape, tile).

    Yields one record per pair: ``ok``, ``skipped`` for a tile that cannot serve
    the shape, or ``error`` with the failure. Compiles are memoised by kernel
    name: under runtime shape the 52 catalog kernels cover every aligned config,
    so compiling per config would repeat the same work hundreds of times.
    """
    from kernels.gfx950.attention_dense import build_attention_dense

    compile_fn = None
    if compile_kernels:
        from rocke.helpers.compile import compile_kernel

        compile_fn = compile_kernel
    compiled: dict = {}

    for shape in shapes:
        tiles = catalog_tiles(shape["head_size"])
        if not tiles:
            yield {
                "shape": shape_id(shape),
                "tile": None,
                "status": "error",
                "detail": f"no catalog tiles for head_size={shape['head_size']}",
            }
            continue
        for block_m, block_n in tiles:
            record = {"shape": shape_id(shape), "tile": f"{block_m}x{block_n}"}
            if not ragged and not divides(shape, block_m, block_n):
                record["status"] = "skipped"
                record["detail"] = "tile does not divide the shape (ragged)"
                yield record
                continue
            try:
                kernel = build_attention_dense(
                    spec_for(shape, block_m, block_n), arch=arch
                )
                record["kernel"] = kernel.name
                if compile_fn is not None:
                    if kernel.name not in compiled:
                        art = compile_fn(kernel, arch=arch, capture_ir_text=False)
                        compiled[kernel.name] = art.hsaco_bytes
                    record["hsaco_bytes"] = compiled[kernel.name]
                record["status"] = "ok"
            except Exception as exc:  # a build failure is the result, not a crash
                record["status"] = "error"
                record["detail"] = f"{type(exc).__name__}: {exc}"
            yield record


def provisional_grid():
    """A stratified grid to work with until the canonical suite is available.

    Spans both head sizes, both dtypes, both masks, GQA ratios from 1 to 8, and
    sequence lengths that straddle the tile multiples (so the ragged path is
    covered) and the grid-fill boundary the heuristic is expected to turn on.
    """
    shapes = []
    for head_size in (64, 128):
        for dtype in ("bf16", "fp16"):
            for causal in (True, False):
                for batch, hq, hkv in ((1, 8, 1), (1, 32, 8), (2, 64, 8), (4, 16, 16)):
                    for sq, sk in (
                        (512, 512),
                        (1024, 1024),
                        (2048, 2048),
                        (4096, 4096),
                        (8192, 8192),
                        (1000, 1000),
                        (512, 4096),
                    ):
                        shapes.append(
                            {
                                "batch": batch,
                                "seqlen_q": sq,
                                "seqlen_kv": sk,
                                "num_query_heads": hq,
                                "num_kv_heads": hkv,
                                "head_size": head_size,
                                "dtype": dtype,
                                "causal": causal,
                            }
                        )
    return shapes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", help="JSON file of shape objects")
    ap.add_argument(
        "--emit-grid",
        metavar="PATH",
        help="write the provisional shape grid to PATH and exit",
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="build the IR only; skip the comgr compile (much faster, weaker check)",
    )
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--output-jsonl", default="", help="write one record per line")
    ap.add_argument("--limit", type=int, default=0, help="first N shapes only")
    ap.add_argument(
        "--ragged",
        action="store_true",
        help="also build tiles that do not divide the shape; the catalog ships none",
    )
    args = ap.parse_args(argv)

    if args.emit_grid:
        Path(args.emit_grid).write_text(json.dumps(provisional_grid(), indent=1))
        print(f"wrote {len(provisional_grid())} shapes -> {args.emit_grid}")
        return 0

    shapes = (
        json.loads(Path(args.shapes).read_text()) if args.shapes else provisional_grid()
    )
    if args.limit:
        shapes = shapes[: args.limit]

    out = open(args.output_jsonl, "w") if args.output_jsonl else None
    ok = failed = skipped = 0
    failures = []
    kernels = set()
    servable = set()
    try:
        for record in validate(
            shapes,
            compile_kernels=not args.no_compile,
            arch=args.arch,
            ragged=args.ragged,
        ):
            if out is not None:
                out.write(json.dumps(record) + "\n")
                out.flush()
            if record["status"] == "ok":
                ok += 1
                kernels.add(record["kernel"])
                servable.add(record["shape"])
            elif record["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(record)
    finally:
        if out is not None:
            out.close()

    print(
        f"shapes: {len(shapes)} ({len(servable)} servable)   "
        f"configs ok: {ok}   skipped: {skipped}   failed: {failed}   "
        f"distinct kernels: {len(kernels)}"
    )
    unservable = len(shapes) - len(servable)
    if unservable:
        print(f"  {unservable} shape(s) no tile divides: the engine declines them")
    for record in failures[:20]:
        print(f"  FAIL {record['shape']} {record['tile']}: {record['detail']}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
