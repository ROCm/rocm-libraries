"""Benchmark the gfx950 chunkwise KDA builder compositions.

Run from ``rocke/library``:

    python -m benchmarks.gfx950.kda.benchmark_chunkwise --path both
    python -m benchmarks.gfx950.kda.benchmark_chunkwise \
        --path split --shapes 8x8x1024 8x16x2048

The builder modules own input construction, launch configuration, and timing.
This driver provides a stable benchmark scenario without wiring the family into
dispatch.
"""

from __future__ import annotations

import argparse


def _shapes(raw: list[str]) -> list[tuple[int, int, int]]:
    result = []
    for value in raw:
        try:
            shape = tuple(int(part) for part in value.split("x"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"shape must be BxHxT, got {value!r}"
            ) from exc
        if len(shape) != 3 or any(dim <= 0 for dim in shape):
            raise argparse.ArgumentTypeError(
                f"shape must be positive BxHxT, got {value!r}"
            )
        if shape[2] % 32:
            raise argparse.ArgumentTypeError(
                f"T must be divisible by the default chunk size 32, got {value!r}"
            )
        result.append(shape)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--path",
        choices=("split", "fused", "both"),
        default="both",
        help="builder composition to benchmark",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["8x8x1024", "8x16x2048", "32x16x2048"],
        metavar="BxHxT",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    shapes = _shapes(args.shapes)

    if args.path in ("split", "both"):
        from builders.gfx950.kda import kda_chunk_split as split
        from kernels.gfx950.kda_chunkwise import tuned_kda_chunk_scan_spec

        print("== split ==")
        for batch, heads, tseq in shapes:
            spec = tuned_kda_chunk_scan_spec(batch * heads)
            print(
                f"  scan geometry: value_splits={spec.value_splits} "
                f"block={spec.tile.block_size} atom={spec.scan_atom.m}"
            )
            split.bench(
                spec,
                batch,
                heads,
                tseq,
                warmup=args.warmup,
                iters=args.iters,
            )

    if args.path in ("fused", "both"):
        from builders.gfx950.kda import kda_chunk_fused as fused
        from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec

        print("== fused ==")
        spec = KdaChunkFusedSpec()
        for batch, heads, tseq in shapes:
            fused.bench(
                spec,
                batch,
                heads,
                tseq,
                warmup=args.warmup,
                iters=args.iters,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
