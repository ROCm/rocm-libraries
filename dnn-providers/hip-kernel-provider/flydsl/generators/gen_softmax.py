# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in softmax code objects for one architecture.

    python -m generators.gen_softmax --arch gfx942

Writes ``<content>/softmax/<arch>/<instance>.hsaco`` for every row of
``_instances.softmax_instances()``, plus the ``manifest.json`` beside them and
``<content>/softmax/<arch>.SOURCE.md`` next to the folder.

The kernel (``kernels_src/kernels/softmax/softmax_kernel.py``) is traced from torch
CPU tensors under ``COMPILE_ONLY``: both extents of p and s lower to dynamic layout
fields, so the traced shape is not baked into the object -- the runtime tier reads N
from p's descriptor, and a baked object's N comes from the builder, not the tensor.
Built for the CDNA targets (gfx90a, gfx942, gfx950): the 16- and 64-thread rows
assume a wave of 64 lanes, and only CDNA hardware is here to verify on.

As for every FlyDSL generator, the full ``(kind, size, offset)`` kernarg sequence is
asserted against the layout below before an object is written, because FlyDSL 0.3.4
emits no argument names and the loader's signature check cannot see two pointers
swapped.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from . import _flydsl_env as env
from ._codeobject import (
    GeneratorError,
    arg_records,
    describe,
    verify_arch,
    verify_generic,
)
from ._extract_hsaco import hsaco_from_dump
from ._instances import SOFTMAX_TARGETS, Instance, instances_for
from ._manifest import refresh_source_md, sha256, write_op_manifest
from ._parallel import compile_all

OP = "softmax"

# The traced extents. Both lower to `?` in the layouts, so they exist only to give the
# tracer a dtype and a rank; a baked object traces at its own N so an IR dump reads true.
_TRACE_M = 4
_GENERIC_TRACE_N = 4096

# `softmax_kernel(P, S, Stats, MIn, StoreStats, StatsScale)`: p and s as (pointer,
# 2-D descriptor) pairs, stats as a (pointer, 1-D descriptor) pair, then the row
# count, the stats switch (i32) and the stats scale (f32).
_KERNARG_SEGMENT_SIZE = 72
_EXPECTED_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("by_value", 16, 8),
    ("global_buffer", 8, 24),
    ("by_value", 16, 32),
    ("global_buffer", 8, 48),
    ("by_value", 4, 56),
    ("by_value", 4, 60),
    ("by_value", 4, 64),
    ("by_value", 4, 68),
)
_SEMANTIC_ARG_NAMES = (
    "P.ptr",
    "P.desc",
    "S.ptr",
    "S.desc",
    "Stats.ptr",
    "Stats.desc",
    "MIn",
    "StoreStats",
    "StatsScale",
)

# `block_threads / threads_per_row` rows per block, so the grid is
# ceil(rows / rows_per_block). Recorded per instance for the dispatch handler.
_GRID_RULE = "rows_per_block"


def _verify(described: dict, instance: Instance, arch: str, where: str) -> None:
    """Fail before writing, rather than shipping a mislabelled object."""
    verify_arch(described, arch, where)
    if described["signature"] != _EXPECTED_SIGNATURE:
        raise GeneratorError(
            f"{where}: kernarg layout for {instance.name!r} is\n"
            f"  {described['signature']}\nbut this family is declared as\n"
            f"  {_EXPECTED_SIGNATURE}"
        )
    if described["kernarg_segment_size"] != _KERNARG_SEGMENT_SIZE:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, "
            f"expected {_KERNARG_SEGMENT_SIZE}"
        )
    if described.get("vgpr_spill_count"):
        raise GeneratorError(
            f"{where}: {instance.name!r} spills {described['vgpr_spill_count']} "
            "VGPRs; every softmax object is meant to fit its registers"
        )


def _build_one(kernel_module, torch, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    knobs = instance.knobs
    baked_n = knobs["N"]
    trace_n = _GENERIC_TRACE_N if baked_n is None else baked_n
    torch_dtype = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
    }

    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_softmax_module(
        baked_n,
        instance.dtype,
        knobs["out_dtype"],
        BLOCK_THREADS=knobs["block_threads"],
        THREADS_PER_ROW=knobs["threads_per_row"],
        VEC=knobs["n_align"],
        UNROLL=knobs["unroll"],
    )

    # CPU tensors on purpose: COMPILE_ONLY never dispatches.
    p = torch.zeros((_TRACE_M, trace_n), dtype=torch_dtype[instance.dtype])
    s = torch.zeros((_TRACE_M, trace_n), dtype=torch_dtype[knobs["out_dtype"]])
    stats = torch.zeros((_TRACE_M,), dtype=torch.float32)
    launch(p, s, stats, _TRACE_M, 1, 1.0)

    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    import torch  # noqa: PLC0415  (after prepare())

    from kernels.softmax import softmax_kernel  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(softmax_kernel, torch, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    if arch not in SOFTMAX_TARGETS:
        raise GeneratorError(
            f"softmax is built for {', '.join(SOFTMAX_TARGETS)}; not for {arch!r}"
        )
    env.assert_flydsl_version()
    env.prepare(arch)

    op_dir = out_root / OP / arch
    op_dir.mkdir(parents=True, exist_ok=True)

    instances = instances_for(OP)
    records: list[dict] = []

    blobs = compile_all(
        _compile,
        [(instance, keep_ir) for instance in instances],
        arch,
        lambda t: t[0].name,
    )

    for instance, blob in zip(instances, blobs):
        filename = f"{instance.name}.hsaco"
        where = f"{arch}/{OP}/{filename}"
        described = describe(blob, where)
        _verify(described, instance, arch, where)
        verify_generic(blob, arch, where)

        (op_dir / filename).write_bytes(blob)

        record = instance.to_record()
        record.update(
            {
                "file": filename,
                "symbol": described["symbol"],
                "bytes": len(blob),
                "sha256": sha256(blob),
                "kernarg_segment_size": described["kernarg_segment_size"],
                "lds_bytes": described["group_segment_fixed_size"],
                "vgprs": described["vgpr_count"],
                "block": [instance.knobs["block_threads"], 1, 1],
                "grid_rule": _GRID_RULE,
                "named_args": described["named_args"],
                "args": arg_records(_SEMANTIC_ARG_NAMES, described["signature"]),
            }
        )
        records.append(record)

    write_op_manifest(op_dir, arch, OP, env.provenance(arch), records)
    source_md = refresh_source_md(op_dir)
    print(f"wrote {len(records)} object(s) to {op_dir}")
    print(f"refreshed {source_md}")
    return op_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arch",
        required=True,
        help="GPU target: gfx90a, gfx942 or gfx950. One arch per invocation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=env.CONTENT_DIR,
        help="FlyDSL content root to write <op>/<arch>/ under "
        f"(default: {env.CONTENT_DIR})",
    )
    parser.add_argument(
        "--keep-ir",
        type=Path,
        default=None,
        help="keep each instance's IR dump under this directory (debugging)",
    )
    args = parser.parse_args(argv)

    try:
        generate(args.arch, args.out_dir, args.keep_ir)
    except (GeneratorError, env.FlydslEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
