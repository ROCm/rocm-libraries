# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure dense-prefill JIT recipe coverage and raw artifact size.

The production kernel still specializes a concrete
``Gfx950AttentionDenseSpec`` when JIT replay occurs.  This experiment moves
selected problem dimensions out of the *shipped* artifact's baked fields and
into the portable-recipe ``spec``:

* 2-axis recipe: ``seqlen_q`` x ``seqlen_kv``;
* 3-axis recipe: ``seqlen_q`` x ``seqlen_kv`` x ``num_query_heads``.

For every run, ``roll_nd`` records only the traces needed to infer the recipe,
then validates its expansion against fresh concrete recordings over the full
sample cross product and a held-out shape.  Sizes are measured from the exact
raw CBOR bytes written for the rolled recipe and for every unrolled concrete
recipe.

Run from ``rocke/platform`` with ``platform/python`` and ``library`` on
``PYTHONPATH``:

    python -m benchmarks.gfx950.attention.prefill.dense_prefill_jit_artifacts
    python -m benchmarks.gfx950.attention.prefill.dense_prefill_jit_artifacts \
        --artifact-dir ~/rocke-jit-dense-results --json-out results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    build_attention_dense,
)
from rocke.portable_ir.src.recipe_bundle import cbor_encode
from rocke.portable_ir.src.roll_nd import roll_nd

ARCH = "gfx950"

# Head size, dtype, algorithm, and tile geometry remain baked implementation
# choices.  The fields named by AXIS_SETS below become free recipe-spec inputs.
_BAKED_SPEC: dict[str, Any] = {
    "batch": 1,
    "seqlen_q": 256,
    "seqlen_kv": 64,
    "num_query_heads": 16,
    "num_kv_heads": 8,
    "head_size": 128,
    "causal": True,
    "dtype": "bf16",
    "block_n": 64,
    "waves_per_eu": 2,
}

AXIS_SETS: dict[int, tuple[str, ...]] = {
    2: ("seqlen_q", "seqlen_kv"),
    3: ("seqlen_q", "seqlen_kv", "num_query_heads"),
}

# Prefixes supply two or three inference samples.  Every value satisfies the
# dense kernel's alignment and GQA constraints.
_SAMPLES: dict[str, tuple[int, int, int]] = {
    "seqlen_q": (256, 512, 768),
    "seqlen_kv": (64, 128, 192),
    "num_query_heads": (16, 32, 48),
}
_HOLDOUT: dict[str, int] = {
    "seqlen_q": 1024,
    "seqlen_kv": 256,
    "num_query_heads": 64,
}


def _build_at(**problem: int):
    fields = {**_BAKED_SPEC, **problem}
    return build_attention_dense(Gfx950AttentionDenseSpec(**fields), arch=ARCH)


def _human_bytes(size: float) -> str:
    for unit, scale in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size:.0f} B"


def _point_name(point: dict[str, int], axes: tuple[str, ...]) -> str:
    return "__".join(f"{axis}-{point[axis]}" for axis in axes)


def run_case(
    axis_count: int,
    samples_per_axis: int,
    *,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Roll and verify one 2-axis or 3-axis dense-prefill recipe."""
    if axis_count not in AXIS_SETS:
        raise ValueError(f"axis_count must be one of {sorted(AXIS_SETS)}")
    if samples_per_axis not in (2, 3):
        raise ValueError("samples_per_axis must be 2 or 3")

    axis_names = AXIS_SETS[axis_count]
    axes = {axis: list(_SAMPLES[axis][:samples_per_axis]) for axis in axis_names}
    holdout = {axis: _HOLDOUT[axis] for axis in axis_names}
    result = roll_nd(_build_at, axes=axes, holdout_points=[holdout])
    if not result.ok:
        raise RuntimeError(f"{axis_count}-axis roll declined: {result.reason}")

    recipe = result.recipe
    assert recipe is not None
    recipe_axes = tuple(entry["name"] for entry in recipe["spec"])
    if recipe_axes != axis_names:
        raise RuntimeError(
            f"recipe free axes {recipe_axes!r} do not match requested {axis_names!r}"
        )

    rolled_blob = cbor_encode(recipe)
    concrete: list[tuple[dict[str, int], bytes]] = []
    for point in result.points:
        key = tuple(point[axis] for axis in axis_names)
        concrete.append((point, cbor_encode(result.traces[key])))

    if artifact_dir is not None:
        case_dir = artifact_dir / f"{axis_count}axis_{samples_per_axis}samples"
        concrete_dir = case_dir / "concrete"
        concrete_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "rolled.recipe.cbor").write_bytes(rolled_blob)
        for point, blob in concrete:
            (
                concrete_dir / f"{_point_name(point, axis_names)}.recipe.cbor"
            ).write_bytes(blob)

    concrete_sizes = [len(blob) for _, blob in concrete]
    concrete_total = sum(concrete_sizes)
    concrete_mean = concrete_total / len(concrete_sizes)
    return {
        "axis_count": axis_count,
        "samples_per_axis": samples_per_axis,
        "free_problem_axes": list(axis_names),
        "sample_values": axes,
        "held_out_shape": holdout,
        "verified_points": len(result.points),
        "inference_traces": result.n_recorded,
        "rolled_cbor_bytes": len(rolled_blob),
        "rolled_cbor_sha256": hashlib.sha256(rolled_blob).hexdigest(),
        "one_concrete_mean_bytes": concrete_mean,
        "one_concrete_min_bytes": min(concrete_sizes),
        "one_concrete_max_bytes": max(concrete_sizes),
        "all_concrete_bytes": concrete_total,
        "size_reduction": concrete_total / len(rolled_blob),
        "parametric_overhead_percent": 100.0
        * (len(rolled_blob) - concrete_mean)
        / concrete_mean,
        "validation": "all recipe expansions equivalent to fresh concrete recordings",
    }


def run_matrix(
    *,
    axis_counts: tuple[int, ...] = (2, 3),
    sample_counts: tuple[int, ...] = (2, 3),
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the requested axis/sample matrix and compare inferred recipes."""
    rows = [
        run_case(axis_count, samples, artifact_dir=artifact_dir)
        for axis_count in axis_counts
        for samples in sample_counts
    ]
    same_recipe: dict[str, bool] = {}
    for axis_count in axis_counts:
        pair = [row for row in rows if row["axis_count"] == axis_count]
        if len(pair) > 1:
            same_recipe[f"{axis_count}-axis"] = (
                len({row["rolled_cbor_sha256"] for row in pair}) == 1
            )
    return {
        "arch": ARCH,
        "baked_spec": _BAKED_SPEC,
        "rows": rows,
        "same_recipe_across_sample_counts": same_recipe,
    }


def _print_table(report: dict[str, Any]) -> None:
    print("dense-prefill portable-IR artifact experiment")
    print("validation: rolled expansion == fresh concrete recording at every point")
    print()
    header = (
        f"{'axes':>4} {'samples/axis':>12} {'rolled CBOR':>13} "
        f"{'1 concrete':>13} {'all concrete':>14} {'points':>7} "
        f"{'traces':>7} {'saved':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in report["rows"]:
        print(
            f"{row['axis_count']:>4} {row['samples_per_axis']:>12} "
            f"{_human_bytes(row['rolled_cbor_bytes']):>13} "
            f"{_human_bytes(row['one_concrete_mean_bytes']):>13} "
            f"{_human_bytes(row['all_concrete_bytes']):>14} "
            f"{row['verified_points']:>7} {row['inference_traces']:>7} "
            f"{row['size_reduction']:>7.1f}x"
        )
    print()
    for label, same in report["same_recipe_across_sample_counts"].items():
        print(f"{label}: 2-sample and 3-sample rolled CBOR byte-identical = {same}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis-count",
        action="append",
        type=int,
        choices=sorted(AXIS_SETS),
        help="axis count to run; repeatable (default: 2 and 3)",
    )
    parser.add_argument(
        "--samples",
        action="append",
        type=int,
        choices=(2, 3),
        help="samples per axis; repeatable (default: 2 and 3)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="write rolled and concrete raw-CBOR files under this directory",
    )
    parser.add_argument(
        "--json-out", type=Path, help="write the complete report as JSON"
    )
    args = parser.parse_args()

    report = run_matrix(
        axis_counts=tuple(args.axis_count or (2, 3)),
        sample_counts=tuple(args.samples or (2, 3)),
        artifact_dir=args.artifact_dir,
    )
    _print_table(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"JSON: {args.json_out}")
    if args.artifact_dir is not None:
        print(f"artifacts: {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
