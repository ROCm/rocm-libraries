# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in gather (embedding lookup) code objects for one architecture.

    python -m generators.gen_gather --arch gfx942

Writes ``<content>/gather/<arch>/<instance>.hsaco`` for every row of
``_instances.gather_instances()``, plus the ``manifest.json`` beside them and
``<content>/gather/<arch>.SOURCE.md`` next to the folder.

The kernel (``kernels_src/kernels/gather/gather_kernel.py``) takes three raw pointers
and four int32s, so it is traced from null pointers and placeholder extents: every
shape is a runtime argument and nothing the trace sees is baked into the object.
Built for the CDNA targets (gfx90a, gfx942, gfx950); it uses only buffer loads and
stores and scalar loads, which every one of them has.

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
from ._instances import GATHER_ELEM_BYTES, GATHER_VECTOR_BYTES, Instance, instances_for
from ._manifest import refresh_source_md, sha256, write_op_manifest
from ._parallel import compile_all

OP = "gather"

# The arches this op is built for: CDNA, whose buffer-descriptor flags the kernel's
# helpers emit. RDNA is not built (no hardware to verify it on).
TARGETS = ("gfx90a", "gfx942", "gfx950")

# `gather_kernel(X, Indices, Y, num_indices, axis_extent, inner)`: three raw pointers,
# then three int32s.
_KERNARG_SEGMENT_SIZE = 36
_EXPECTED_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("global_buffer", 8, 8),
    ("global_buffer", 8, 16),
    ("by_value", 4, 24),
    ("by_value", 4, 28),
    ("by_value", 4, 32),
)
_SEMANTIC_ARG_NAMES = (
    "X.ptr",
    "Indices.ptr",
    "Y.ptr",
    "num_indices",
    "axis_extent",
    "inner",
)

# Grid (num_indices, ceil(inner / (block_threads * vec_elems)), outer); block
# (block_threads, 1, 1), vec_elems = vec_bytes / the element size.
_GRID_RULE = "gather_rows"


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
    import flydsl.compiler as flyc  # noqa: PLC0415  (after prepare())
    import flydsl.expr as fx  # noqa: PLC0415

    knobs = instance.knobs
    if knobs["block_threads"] != kernel_module.BLOCK_THREADS:
        raise GeneratorError(
            f"{instance.name}: the kernel is built for {kernel_module.BLOCK_THREADS} "
            f"threads, the row declares {knobs['block_threads']}"
        )
    vector = knobs["vec_bytes"] == GATHER_VECTOR_BYTES
    if not vector and knobs["vec_bytes"] != GATHER_ELEM_BYTES[instance.dtype]:
        raise GeneratorError(
            f"{instance.name}: vec_bytes {knobs['vec_bytes']} is neither "
            f"{GATHER_VECTOR_BYTES} nor the element size"
        )
    if kernel_module.VECTOR_BYTES != GATHER_VECTOR_BYTES:
        raise GeneratorError(
            f"the kernel's vector is {kernel_module.VECTOR_BYTES} bytes, the instance "
            f"table's {GATHER_VECTOR_BYTES}"
        )

    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_gather_module(
        instance.dtype, knobs["index_dtype"], vector
    )
    # Placeholders only: COMPILE_ONLY never dispatches through the null pointers, and
    # every extent is a runtime argument.
    null = flyc.from_c_void_p(fx.Uint8, 0)
    flyc.compile(launch, null, null, null, 1, 1, 1, 1, fx.Stream(None))

    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    from kernels.gather import gather_kernel  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(gather_kernel, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    if arch not in TARGETS:
        raise GeneratorError(
            f"gather is built for {', '.join(TARGETS)}; not for {arch!r}"
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
        help=f"GPU target, one of {', '.join(TARGETS)}. One arch per invocation.",
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
