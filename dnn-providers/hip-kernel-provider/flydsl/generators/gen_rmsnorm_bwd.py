# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in RMSNorm backward code objects for one architecture.

    python -m generators.gen_rmsnorm_bwd --arch gfx942

Writes ``<content>/rmsnorm_bwd/<arch>/<instance>.hsaco`` for every row of
``_instances.rmsnorm_bwd_instances()``, plus the ``manifest.json`` beside them and
``<content>/rmsnorm_bwd/<arch>.SOURCE.md`` next to the folder, all committed.

Each object is the vendored two-stage builder (``rmsnorm_bwd_kernel.py``) and holds
two kernels: the partial kernel, which writes dx and one f32 dgamma row per program
into a workspace, and the reduce kernel, which sums those rows into dgamma. Both are
verified against the layouts recorded below before anything is written, and the
reduce kernel's symbol must be the one the instance table names, since the dispatch
loads it by that name from the same archive entry.
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
    kernel_names,
    verify_arch,
    verify_generic,
)
from ._extract_hsaco import hsaco_from_dump
from ._instances import Instance, instances_for
from ._manifest import refresh_source_md, sha256, write_op_manifest
from ._parallel import compile_all

OP = "rmsnorm_bwd"

# Traced extents. The row count and, for a generic object, the width are read from
# the tensor descriptors at run time; these only give the tracer a dtype and a rank.
_TRACE_M = 4
_GENERIC_TRACE_N = 1024
# Sizes only the convenience launcher's grid: the objects read the program count from
# the workspace descriptor (vendored modification 2).
_TRACE_PROGRAMS = 4

# The partial kernel: Source, Gamma, DY, DResidualOut (DY again in the plain
# variant), Rstd, DX, DWeightPartial -- each a pointer and its layout -- then the row
# count. 2-D tensors carry a 16-byte layout, 1-D ones (Gamma, Rstd) a 4-byte one.
_PARTIAL_KERNARG_SEGMENT_SIZE = 156
_PARTIAL_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("by_value", 16, 8),
    ("global_buffer", 8, 24),
    ("by_value", 4, 32),
    ("global_buffer", 8, 40),
    ("by_value", 16, 48),
    ("global_buffer", 8, 64),
    ("by_value", 16, 72),
    ("global_buffer", 8, 88),
    ("by_value", 4, 96),
    ("global_buffer", 8, 104),
    ("by_value", 16, 112),
    ("global_buffer", 8, 128),
    ("by_value", 16, 136),
    ("by_value", 4, 152),
)
_PARTIAL_ARG_NAMES = (
    "Source.ptr",
    "Source.desc",
    "Gamma.ptr",
    "Gamma.desc",
    "DY.ptr",
    "DY.desc",
    "DResidualOut.ptr",
    "DResidualOut.desc",
    "Rstd.ptr",
    "Rstd.desc",
    "DX.ptr",
    "DX.desc",
    "DWeightPartial.ptr",
    "DWeightPartial.desc",
    "MIn",
)
# The reduce kernel: DWeightPartial (2-D) and DWeight (1-D).
_REDUCE_KERNARG_SEGMENT_SIZE = 36
_REDUCE_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("by_value", 16, 8),
    ("global_buffer", 8, 24),
    ("by_value", 4, 32),
)
_REDUCE_ARG_NAMES = (
    "DWeightPartial.ptr",
    "DWeightPartial.desc",
    "DWeight.ptr",
    "DWeight.desc",
)

# Launch geometry the native dispatch reproduces: partial blocks of 512 threads, one
# per program (the workspace row count, chosen at prepare); reduce blocks of 256
# threads, one per 64 dgamma columns.
_GRID_RULE = "partial_one_block_per_program_reduce_one_block_per_64_columns"


def _verify_layout(
    described: dict, signature: tuple, size: int, where: str, what: str
) -> None:
    if described["signature"] != signature:
        raise GeneratorError(
            f"{where}: the {what} kernel's kernarg layout is\n  {described['signature']}\n"
            f"but this family is declared as\n  {signature}"
        )
    if described["kernarg_segment_size"] != size:
        raise GeneratorError(
            f"{where}: the {what} kernel's kernarg segment is "
            f"{described['kernarg_segment_size']} bytes, expected {size}"
        )
    if described["vgpr_spill_count"]:
        raise GeneratorError(
            f"{where}: the {what} kernel spills {described['vgpr_spill_count']} VGPRs; "
            "lower the instance's n_max"
        )


def _assert_geometry_mirror(kernel_module) -> None:
    """The dispatch hardcodes the builder's block sizes; confirm they still hold."""
    expected = {
        "TWO_STAGE_PARTIAL_THREADS": 512,
        "DWEIGHT_REDUCE_THREADS": 256,
        "DWEIGHT_REDUCE_COLS": 64,
    }
    for name, value in expected.items():
        actual = getattr(kernel_module, name, None)
        if actual != value:
            raise GeneratorError(
                f"kernels_src {name} is {actual!r}; the dispatch (FlydslRmsNormBwdNative.cpp) "
                f"and this generator assume {value}"
            )


def _build_one(kernel_module, torch, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    import_n = instance.knobs["N"]
    trace_n = _GENERIC_TRACE_N if import_n is None else import_n
    torch_dtype = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
    }[instance.dtype]
    weight_dtype = "f32" if instance.knobs["scale_f32"] else instance.dtype
    weight_torch = torch.float32 if instance.knobs["scale_f32"] else torch_dtype

    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_rmsnorm_bwd_two_stage_module(
        import_n,
        instance.dtype,
        _TRACE_PROGRAMS,
        weight_dtype_str=weight_dtype,
        n_max=None if import_n is not None else instance.knobs["n_max"],
    )

    # CPU tensors: COMPILE_ONLY never dispatches (see gen_rmsnorm.py).
    x = torch.zeros((_TRACE_M, trace_n), dtype=torch_dtype)
    gamma = torch.zeros((trace_n,), dtype=weight_torch)
    dy = torch.zeros((_TRACE_M, trace_n), dtype=torch_dtype)
    rstd = torch.zeros((_TRACE_M,), dtype=torch.float32)
    dx = torch.zeros((_TRACE_M, trace_n), dtype=torch_dtype)
    dgamma = torch.zeros((trace_n,), dtype=weight_torch)
    partial = torch.zeros((_TRACE_PROGRAMS, trace_n), dtype=torch.float32)
    launch(x, gamma, dy, rstd, dx, dgamma, partial, _TRACE_M)

    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    import torch  # noqa: PLC0415  (after prepare())

    from kernels.norm import rmsnorm_bwd_kernel  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(rmsnorm_bwd_kernel, torch, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    env.assert_flydsl_version()
    env.prepare(arch)

    from kernels.norm import rmsnorm_bwd_kernel  # noqa: PLC0415

    _assert_geometry_mirror(rmsnorm_bwd_kernel)

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
        reduce_symbol = instance.knobs["reduce_symbol"]
        names = kernel_names(blob, where)
        if len(names) != 2 or reduce_symbol not in names:
            raise GeneratorError(
                f"{where}: expected a partial kernel and {reduce_symbol!r}, found {names}"
            )
        partial_symbol = next(n for n in names if n != reduce_symbol)
        partial = describe(blob, where, partial_symbol)
        reduce = describe(blob, where, reduce_symbol)
        verify_arch(partial, arch, where)
        _verify_layout(
            partial, _PARTIAL_SIGNATURE, _PARTIAL_KERNARG_SEGMENT_SIZE, where, "partial"
        )
        _verify_layout(
            reduce, _REDUCE_SIGNATURE, _REDUCE_KERNARG_SEGMENT_SIZE, where, "reduce"
        )
        verify_generic(blob, arch, where)

        (op_dir / filename).write_bytes(blob)

        record = instance.to_record()
        record.update(
            {
                "file": filename,
                "symbol": partial["symbol"],
                "bytes": len(blob),
                "sha256": sha256(blob),
                "kernarg_segment_size": partial["kernarg_segment_size"],
                "lds_bytes": partial["group_segment_fixed_size"],
                "vgprs": partial["vgpr_count"],
                "block": [512, 1, 1],
                "grid_rule": _GRID_RULE,
                "named_args": partial["named_args"],
                "args": arg_records(_PARTIAL_ARG_NAMES, partial["signature"]),
                "reduce_args": arg_records(_REDUCE_ARG_NAMES, reduce["signature"]),
                "reduce_vgprs": reduce["vgpr_count"],
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
    parser.add_argument("--arch", required=True, help="GPU target, e.g. gfx942")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=env.CONTENT_DIR,
        help=f"FlyDSL content root to write <op>/<arch>/ under (default: {env.CONTENT_DIR})",
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
