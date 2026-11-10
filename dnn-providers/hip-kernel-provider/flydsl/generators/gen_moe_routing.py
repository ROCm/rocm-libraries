# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in MoE routing code objects for one architecture.

    python -m generators.gen_moe_routing --arch gfx942

Writes ``<content>/moe_routing/<arch>/<instance>.hsaco`` for every row of
``_instances.moe_routing_instances()``, plus the ``manifest.json`` beside them and
``<content>/moe_routing/<arch>.SOURCE.md`` next to the folder.

Each object is ``kernels_src/kernels/moe_routing/moe_routing_kernel.py``'s counting
sort and holds three kernels, launched in order on one stream: the histogram kernel
(the UKD's symbol), which writes each block's per-expert counts to the workspace;
the prefix kernel, which scans them into each block's first rows and writes
first_token_offset; and the scatter kernel, which writes token_index and token_ks.
The second and third are named in the instance's metadata (``prefix_symbol``,
``scatter_symbol``), and all three are verified against the layouts below before
anything is written, since the dispatch loads them by those names from the same
archive entry and marshals their arguments by position.
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

OP = "moe_routing"

# CDNA: the kernels group lanes with 64-bit ballots, so they need wave64, and their
# buffer-resource helpers emit CDNA descriptor flags.
TARGETS = ("gfx90a", "gfx942", "gfx950")

# histogram(TopkIds, Counts, n_pairs, num_experts, span, num_blocks)
_HISTOGRAM_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("by_value", 4, 8),
    ("global_buffer", 8, 16),
    ("by_value", 4, 24),
    ("by_value", 4, 28),
    ("by_value", 4, 32),
    ("by_value", 4, 36),
    ("by_value", 4, 40),
)
_HISTOGRAM_KERNARG_SEGMENT_SIZE = 44
_HISTOGRAM_ARG_NAMES = (
    "TopkIds.ptr",
    "TopkIds.desc",
    "Counts.ptr",
    "Counts.desc",
    "n_pairs",
    "num_experts",
    "span",
    "num_blocks",
)
# prefix(Counts, FirstTokenOffset, num_experts, num_blocks, offset_is_i64)
_PREFIX_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("by_value", 4, 8),
    ("global_buffer", 8, 16),
    ("by_value", 4, 24),
    ("by_value", 4, 28),
    ("by_value", 4, 32),
    ("by_value", 4, 36),
)
_PREFIX_KERNARG_SEGMENT_SIZE = 40
_PREFIX_ARG_NAMES = (
    "Counts.ptr",
    "Counts.desc",
    "FirstTokenOffset.ptr",
    "FirstTokenOffset.desc",
    "num_experts",
    "num_blocks",
    "offset_is_i64",
)
# scatter(TopkIds, Counts, TokenIndex, TokenKs, n_pairs, num_experts, span,
#         num_blocks, top_k)
_SCATTER_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("by_value", 4, 8),
    ("global_buffer", 8, 16),
    ("by_value", 4, 24),
    ("global_buffer", 8, 32),
    ("by_value", 4, 40),
    ("global_buffer", 8, 48),
    ("by_value", 4, 56),
    ("by_value", 4, 60),
    ("by_value", 4, 64),
    ("by_value", 4, 68),
    ("by_value", 4, 72),
    ("by_value", 4, 76),
)
_SCATTER_KERNARG_SEGMENT_SIZE = 80
_SCATTER_ARG_NAMES = (
    "TopkIds.ptr",
    "TopkIds.desc",
    "Counts.ptr",
    "Counts.desc",
    "TokenIndex.ptr",
    "TokenIndex.desc",
    "TokenKs.ptr",
    "TokenKs.desc",
    "n_pairs",
    "num_experts",
    "span",
    "num_blocks",
    "top_k",
)

# Launch geometry the native dispatch reproduces: histogram and scatter launch one
# 256-thread block per span of pairs, the prefix kernel one 1024-thread block.
_GRID_RULE = "histogram_and_scatter_one_block_per_span_prefix_one_block"

# Traced extents: every shape is a kernel argument, so these only give the tracer
# int32 tensors of rank 1.
_TRACE_PAIRS = 8


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
            f"{where}: the {what} kernel spills {described['vgpr_spill_count']} VGPRs"
        )


def _assert_geometry_mirror(kernel_module, instance: Instance) -> None:
    """The dispatch hardcodes the builder's block sizes; confirm they still hold."""
    expected = {
        "BLOCK_THREADS": 256,
        "PREFIX_THREADS": 1024,
        "PREFIX_ITEMS": 4,
        "E_MAX": instance.knobs["e_max"],
    }
    for name, value in expected.items():
        actual = getattr(kernel_module, name, None)
        if actual != value:
            raise GeneratorError(
                f"kernels_src moe_routing {name} is {actual!r}; the dispatch "
                f"(FlydslMoeRoutingNative.cpp) and this generator assume {value}"
            )


def _build_one(kernel_module, torch, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_moe_routing_module(instance.knobs["e_max"])

    # CPU tensors: COMPILE_ONLY never dispatches.
    def i32(extent: int):
        return torch.zeros((extent,), dtype=torch.int32)

    launch(
        i32(_TRACE_PAIRS),  # TopkIds
        i32(4),  # Counts
        i32(4),  # FirstTokenOffset
        i32(_TRACE_PAIRS),  # TokenIndex
        i32(_TRACE_PAIRS),  # TokenKs
        _TRACE_PAIRS,  # n_pairs
        2,  # num_experts
        2,  # top_k
        256,  # span
        1,  # num_blocks
        0,  # offset_is_i64
    )
    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    import torch  # noqa: PLC0415  (after prepare())

    from kernels.moe_routing import moe_routing_kernel  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(moe_routing_kernel, torch, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    if arch not in TARGETS:
        raise GeneratorError(
            f"moe_routing is built for {', '.join(TARGETS)}; not for {arch!r}"
        )
    env.assert_flydsl_version()
    env.prepare(arch)

    from kernels.moe_routing import moe_routing_kernel  # noqa: PLC0415

    instances = instances_for(OP)
    for instance in instances:
        _assert_geometry_mirror(moe_routing_kernel, instance)

    op_dir = out_root / OP / arch
    op_dir.mkdir(parents=True, exist_ok=True)
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
        prefix_symbol = instance.knobs["prefix_symbol"]
        scatter_symbol = instance.knobs["scatter_symbol"]
        names = kernel_names(blob, where)
        if len(names) != 3 or prefix_symbol not in names or scatter_symbol not in names:
            raise GeneratorError(
                f"{where}: expected a histogram kernel, {prefix_symbol!r} and "
                f"{scatter_symbol!r}, found {names}"
            )
        histogram_symbol = next(
            n for n in names if n not in (prefix_symbol, scatter_symbol)
        )
        histogram = describe(blob, where, histogram_symbol)
        prefix = describe(blob, where, prefix_symbol)
        scatter = describe(blob, where, scatter_symbol)
        verify_arch(histogram, arch, where)
        _verify_layout(
            histogram,
            _HISTOGRAM_SIGNATURE,
            _HISTOGRAM_KERNARG_SEGMENT_SIZE,
            where,
            "histogram",
        )
        _verify_layout(
            prefix, _PREFIX_SIGNATURE, _PREFIX_KERNARG_SEGMENT_SIZE, where, "prefix"
        )
        _verify_layout(
            scatter, _SCATTER_SIGNATURE, _SCATTER_KERNARG_SEGMENT_SIZE, where, "scatter"
        )
        verify_generic(blob, arch, where)

        (op_dir / filename).write_bytes(blob)

        record = instance.to_record()
        record.update(
            {
                "file": filename,
                "symbol": histogram["symbol"],
                "bytes": len(blob),
                "sha256": sha256(blob),
                "kernarg_segment_size": histogram["kernarg_segment_size"],
                "lds_bytes": histogram["group_segment_fixed_size"],
                "vgprs": histogram["vgpr_count"],
                "block": [256, 1, 1],
                "grid_rule": _GRID_RULE,
                "named_args": histogram["named_args"],
                "args": arg_records(_HISTOGRAM_ARG_NAMES, histogram["signature"]),
                "prefix_args": arg_records(_PREFIX_ARG_NAMES, prefix["signature"]),
                "prefix_vgprs": prefix["vgpr_count"],
                "scatter_args": arg_records(_SCATTER_ARG_NAMES, scatter["signature"]),
                "scatter_vgprs": scatter["vgpr_count"],
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
