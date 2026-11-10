# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in top-k code objects for one architecture.

    python -m generators.gen_topk --arch gfx942

Writes ``<content>/topk/<arch>/<instance>.hsaco`` for every row of
``_instances.topk_instances()``, plus the ``manifest.json`` beside them and
``<content>/topk/<arch>.SOURCE.md`` next to the folder. ``<content>`` is FlyDSL's bundle
folder under the provider's production descriptor root (``_flydsl_env.CONTENT_DIR``).

Every object is one kernel, ``topk_kernel(X, Indices, Values, K, Largest, WriteValues,
VecOk)``, launched one workgroup per row; ``k_max`` picks its body (the k = 1 reduction,
or the radix select for every k up to the bound). Each object's ``(kind, size, offset)``
kernarg sequence is asserted against the layout below before it is written, and the
semantic names are carried into the manifest (FlyDSL 0.3.4 emits none; see
``gen_rmsnorm.py``).
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
from ._instances import (
    TOPK_GENERAL_BLOCK_THREADS,
    TOPK_GENERAL_K_MAX,
    Instance,
    instances_for,
)
from ._manifest import refresh_source_md, sha256, write_op_manifest
from ._parallel import compile_all

OP = "topk"

# Traced extents. Rows, row width and k lower to runtime layout fields
# (`(?,?):(?{i64},1)`), so these only give the tracer a dtype and a rank.
_TRACE_ROWS = 4
_TRACE_N = 4096
_TRACE_K = 8

# Three (pointer, 2-D descriptor) pairs -- X, Indices, Values -- then K, Largest,
# WriteValues and VecOk as i32 by value. Every object of the family has this layout.
_KERNARG_SEGMENT_SIZE = 88
_EXPECTED_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("by_value", 16, 8),
    ("global_buffer", 8, 24),
    ("by_value", 16, 32),
    ("global_buffer", 8, 48),
    ("by_value", 16, 56),
    ("by_value", 4, 72),
    ("by_value", 4, 76),
    ("by_value", 4, 80),
    ("by_value", 4, 84),
)
# Without a values output, the dispatch fills the Values slots with X's and passes
# WriteValues 0: the kernel never stores through them.
_SEMANTIC_ARG_NAMES = (
    "X.ptr",
    "X.desc",
    "Indices.ptr",
    "Indices.desc",
    "Values.ptr",
    "Values.desc",
    "K",
    "Largest",
    "WriteValues",
    "VecOk",
)

_GRID_RULE = "one_block_per_row"


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


def _assert_mirrors(kernel_module) -> None:
    """`_instances.py` mirrors the kernel's bounds (it must import without FlyDSL)."""
    for name, mirror in (
        ("GENERAL_K_MAX", TOPK_GENERAL_K_MAX),
        ("GENERAL_BLOCK_THREADS", TOPK_GENERAL_BLOCK_THREADS),
    ):
        upstream = getattr(kernel_module, name, None)
        if upstream != mirror:
            raise GeneratorError(
                f"kernels_src topk {name} is {upstream!r} but _instances.py mirrors "
                f"{mirror!r}; update the mirror"
            )


def _build_one(kernel_module, torch, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    torch_dtype = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
    }[instance.dtype]
    index_dtype = {"i32": torch.int32, "i64": torch.int64}[
        instance.knobs["index_dtype"]
    ]

    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_topk_module(
        instance.dtype,
        instance.knobs["index_dtype"],
        instance.knobs["k_max"],
        instance.knobs["block_threads"],
    )
    # CPU tensors: COMPILE_ONLY traces and lowers but never dispatches.
    x = torch.zeros((_TRACE_ROWS, _TRACE_N), dtype=torch_dtype)
    indices = torch.zeros((_TRACE_ROWS, _TRACE_K), dtype=index_dtype)
    values = torch.zeros((_TRACE_ROWS, _TRACE_K), dtype=torch_dtype)
    launch(x, indices, values, _TRACE_K, 1, 1, 1, _TRACE_ROWS)

    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    import torch  # noqa: PLC0415  (after prepare())

    from kernels.topk import topk_kernel  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(topk_kernel, torch, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    env.assert_flydsl_version()
    env.prepare(arch)

    from kernels.topk import topk_kernel  # noqa: PLC0415

    _assert_mirrors(topk_kernel)

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
