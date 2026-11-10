# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in RoPE (rotate-half) code objects for one architecture.

    python -m generators.gen_rope --arch gfx942

Writes ``<content>/rope/<arch>/<instance>.hsaco`` for every row of
``_instances.rope_instances()``, plus the ``manifest.json`` beside them and
``<content>/rope/<arch>.SOURCE.md`` next to the folder. See gen_rmsnorm.py for
the pipeline this follows: compile under ``COMPILE_ONLY``, verify each object's
kernarg layout against the one declared here, record semantic argument names.

The kernel takes nothing baked but the element type of x and y: head_dim,
rope_dim, rope_offset, output_scale and every stride are kernel arguments, so
the trace below passes placeholders and one object serves every shape.
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
from ._instances import Instance, instances_for
from ._manifest import refresh_source_md, sha256, write_op_manifest
from ._parallel import compile_all

OP = "rope"

# The kernel's parameter list, in order: three pointers (x, f32 freqs, y), the
# four element strides of x, freqs and y outermost first, the three Int32 shape
# arguments and the f32 output scale.
_SEMANTIC_ARG_NAMES = (
    "X",
    "Freqs",
    "Y",
    "x_stride_0",
    "x_stride_1",
    "x_stride_2",
    "x_stride_3",
    "freqs_stride_0",
    "freqs_stride_1",
    "freqs_stride_2",
    "freqs_stride_3",
    "y_stride_0",
    "y_stride_1",
    "y_stride_2",
    "y_stride_3",
    "head_dim",
    "rope_dim",
    "rope_offset",
    "output_scale",
)

_KERNARG_SEGMENT_SIZE = 136
_EXPECTED_SIGNATURE = (
    *(("global_buffer", 8, 8 * i) for i in range(3)),
    *(("by_value", 8, 24 + 8 * i) for i in range(12)),
    *(("by_value", 4, 120 + 4 * i) for i in range(4)),
)

# Launch geometry the native dispatch handler reproduces: x padded to rank 4 as
# [d0, d1, d2, D], one workgroup per row on grid (d2, d1, d0).
_GRID_RULE = "one_block_per_row_d2_d1_d0"


def _verify(described: dict, instance: Instance, arch: str, where: str) -> None:
    """Fail before writing, rather than shipping a mislabelled object."""
    verify_arch(described, arch, where)

    if described["signature"] != _EXPECTED_SIGNATURE:
        raise GeneratorError(
            f"{where}: kernarg layout for {instance.name!r} is\n"
            f"  {described['signature']}\nbut this family is declared as\n"
            f"  {_EXPECTED_SIGNATURE}\n"
            "The kernel's parameter list changed; update the layout, the semantic "
            "names, and the native dispatch together."
        )
    if described["kernarg_segment_size"] != _KERNARG_SEGMENT_SIZE:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, "
            f"expected {_KERNARG_SEGMENT_SIZE}"
        )
    if described.get("vgpr_spill_count"):
        raise GeneratorError(f"{where}: {described['vgpr_spill_count']} VGPRs spilled")


def _build_one(kernel_module, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    import flydsl.compiler as flyc  # noqa: PLC0415  (after prepare())
    import flydsl.expr as fx  # noqa: PLC0415

    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_rope_module(
        instance.dtype, BLOCK_THREADS=instance.knobs["block_threads"]
    )
    if launch.block_threads != instance.knobs["block_threads"]:
        raise GeneratorError(
            f"{instance.name}: kernel built with {launch.block_threads} threads, "
            f"the instance declares {instance.knobs['block_threads']}"
        )

    # Placeholders only: every value below is a runtime kernel argument, and
    # COMPILE_ONLY never dispatches through the null pointers.
    null = flyc.from_c_void_p(fx.Uint8, 0)
    args = [null] * 3 + [1] * 12
    # head_dim, rope_dim, rope_offset, output_scale; then the grid d2, d1, d0.
    args += [2, 2, 0, 1.0, 1, 1, 1, fx.Stream(None)]
    flyc.compile(launch, *args)

    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    from kernels.rope import rope_kernel  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(rope_kernel, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
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
        help="GPU target, e.g. gfx942. One arch per invocation: FlyDSL reads ARCH "
        "from the environment at each compile.",
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
