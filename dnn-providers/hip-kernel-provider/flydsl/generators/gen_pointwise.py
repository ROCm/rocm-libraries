# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in pointwise code objects for one architecture.

    python -m generators.gen_pointwise --arch gfx942

Writes ``<content>/pointwise/<arch>/<instance>.hsaco`` for every row of
``_instances.pointwise_instances()``, plus the ``manifest.json`` beside them and
``<content>/pointwise/<arch>.SOURCE.md`` next to the folder. The folder is a DVC
output (``<arch>.dvc``); SOURCE.md is committed.

Every object is one elementwise operation over runtime extents and strides
(``kernels_src/kernels/pointwise/pointwise_kernel.py``): nothing about a shape is
baked, so the values traced below only give the tracer its argument types. As in
``gen_rmsnorm``, each object's kernarg layout is checked against the one declared
here before it is written, and the slots' meanings are carried into the manifest,
since FlyDSL 0.3.4 emits no argument names.
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

OP = "pointwise"

# `pointwise_kernel(Out, A, B, n, vec, d1, d2, d3, so0..so3, sa0..sa3, sb0..sb3,
# param)`: three raw pointers (a unary object never reads B), five Int32, twelve Int64
# element strides (padded to 8 after the Int32s), then the f32 parameter.
_KERNARG_SEGMENT_SIZE = 148
_EXPECTED_SIGNATURE = (
    (
        ("global_buffer", 8, 0),
        ("global_buffer", 8, 8),
        ("global_buffer", 8, 16),
        ("by_value", 4, 24),
        ("by_value", 4, 28),
        ("by_value", 4, 32),
        ("by_value", 4, 36),
        ("by_value", 4, 40),
    )
    + tuple(("by_value", 8, 48 + 8 * slot) for slot in range(12))
    + (("by_value", 4, 144),)
)
_SEMANTIC_ARG_NAMES = (
    "Out.ptr",
    "A.ptr",
    "B.ptr",
    "n",
    "vec",
    "d1",
    "d2",
    "d3",
    "so0",
    "so1",
    "so2",
    "so3",
    "sa0",
    "sa1",
    "sa2",
    "sa3",
    "sb0",
    "sb1",
    "sb2",
    "sb3",
    "param",
)

# One output element per thread, or with `vec` set one run of `vec_elems` elements per
# thread: the grid is ceil(n / (block_threads * (vec ? vec_elems : 1))) workgroups.
_GRID_RULE = "one_thread_per_element_or_vector"


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


def _build_one(kernel_module, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    from flydsl.compiler.jit_argument import from_c_void_p  # noqa: PLC0415

    from kernels.common.kernels_common import dtype_to_elem_type  # noqa: PLC0415

    knobs = instance.knobs
    in1 = None if knobs["in1_dtype"] == "none" else knobs["in1_dtype"]
    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_pointwise_module(
        knobs["operation"],
        instance.dtype,
        in1,
        knobs["out_dtype"],
        BLOCK_THREADS=knobs["block_threads"],
    )
    # The table mirrors the kernel's vector width; a drift would launch the dense path
    # with the wrong grid.
    if launch.vector_width != knobs["vec_elems"]:
        raise GeneratorError(
            f"{instance.name}: the kernel's vector width is {launch.vector_width} but "
            f"_instances.py records vec_elems {knobs['vec_elems']}"
        )

    # Null pointers on purpose: COMPILE_ONLY traces and lowers but never dispatches, so
    # no device (and no matching one) is needed to build an object.
    def pointer(dtype_str: str):
        return from_c_void_p(dtype_to_elem_type(dtype_str), 0)

    launch(
        pointer(knobs["out_dtype"]),
        pointer(instance.dtype),
        pointer(in1 or instance.dtype),
        *(16, 1, 1, 1, 16),
        *(0, 0, 0, 1) * 3,
        1.0,
        1,
    )
    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    from kernels.pointwise import pointwise_kernel  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(pointwise_kernel, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    env.assert_flydsl_version()
    env.prepare(arch)

    op_dir = out_root / OP / arch
    op_dir.mkdir(parents=True, exist_ok=True)

    instances = instances_for(OP)
    blobs = compile_all(
        _compile,
        [(instance, keep_ir) for instance in instances],
        arch,
        lambda t: t[0].name,
    )

    records: list[dict] = []
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
        help="GPU target, e.g. gfx942. One arch per invocation: FlyDSL reads ARCH from "
        "the environment at each compile.",
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
