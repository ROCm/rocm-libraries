# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in paged KV-cache store code objects for one architecture.

    python -m generators.gen_paged_cache_store --arch gfx942

Writes ``<content>/paged_cache_store/<arch>/<instance>.hsaco`` for every row of
``_instances.paged_cache_store_instances(arch)``, plus the ``manifest.json`` beside
them and ``<content>/paged_cache_store/<arch>.SOURCE.md`` next to the folder.

Every shape, stride, the slot dtype and the scale are kernel arguments, so the
trace below passes placeholder values: nothing about them is baked into the
object. What is baked is x's element type and the container's, which name the
instance. Each object's kernarg layout is checked against ``_EXPECTED_SIGNATURE``
before it is written, for the reason ``gen_rmsnorm`` gives: FlyDSL 0.3.4 emits no
argument names, so a permutation is only catchable here.
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

OP = "paged_cache_store"

# `paged_cache_store_kernel(...)`: four raw global pointers, then the six 64-bit
# strides, then eleven 32-bit integers and the f32 scale. Ordered widest first so
# the segment has no padding, which keeps the native handler's packed struct a
# plain mirror of it.
_SEMANTIC_ARG_NAMES = (
    "X",
    "SlotMapping",
    "Cache",
    "Scale",
    "XStrideB",
    "XStrideS",
    "SlotStrideB",
    "SlotStrideS",
    "CStrideBlock",
    "CStrideSlot",
    "Heads",
    "HeadDim",
    "XStrideH",
    "XStrideD",
    "CStrideH",
    "CStrideD",
    "BlockSize",
    "NumSlots",
    "SlotIs64",
    "ScaleFromPtr",
    "VecOk",
    "ScaleValue",
)
_EXPECTED_SIGNATURE = (
    tuple(("global_buffer", 8, 8 * i) for i in range(4))
    + tuple(("by_value", 8, 32 + 8 * i) for i in range(6))
    + tuple(("by_value", 4, 80 + 4 * i) for i in range(12))
)
_KERNARG_SEGMENT_SIZE = 128

# One block per token: grid (S, B, 1), `block_threads` wide.
_GRID_RULE = "one_block_per_token"


def _verify(described: dict, instance: Instance, arch: str, where: str) -> None:
    """Fail before writing, rather than shipping a mislabelled object."""
    verify_arch(described, arch, where)
    if described["signature"] != _EXPECTED_SIGNATURE:
        raise GeneratorError(
            f"{where}: kernarg layout for {instance.name!r} is\n"
            f"  {described['signature']}\nbut this family is declared as\n"
            f"  {_EXPECTED_SIGNATURE}\n"
            "The kernel's parameter list changed: update _SEMANTIC_ARG_NAMES, this "
            "layout and the native handler's argument struct together."
        )
    if described["kernarg_segment_size"] != _KERNARG_SEGMENT_SIZE:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, "
            f"expected {_KERNARG_SEGMENT_SIZE}"
        )


def _build_one(kernel_module, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    import flydsl.expr as fx  # noqa: PLC0415  (after prepare())
    from flydsl.compiler.jit_argument import from_c_void_p  # noqa: PLC0415

    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_paged_cache_store_module(
        instance.dtype,
        instance.knobs["cache_dtype"],
        BLOCK_THREADS=instance.knobs["block_threads"],
    )

    # Null pointers and unit extents: COMPILE_ONLY traces and lowers without
    # dispatching, and every value here is a runtime argument of the object.
    def null():
        return from_c_void_p(fx.Int8, 0)

    i64 = fx.Int64
    launch(
        *(null() for _ in range(4)),
        *(i64(0) for _ in range(6)),
        *(1 for _ in range(8)),
        0,
        0,
        0,
        1.0,
        1,
        1,
    )

    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    from kernels.paged_cache import paged_cache_store_kernel  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(paged_cache_store_kernel, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    env.assert_flydsl_version()
    env.prepare(arch)

    op_dir = out_root / OP / arch
    op_dir.mkdir(parents=True, exist_ok=True)

    instances = instances_for(OP, arch)
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
        help="GPU target: gfx90a, gfx942 or gfx950. One arch per invocation: FlyDSL "
        "reads ARCH from the environment at each compile.",
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
