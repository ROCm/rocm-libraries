# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in RDNA SDPA decode code objects for one architecture.

    python -m generators.gen_sdpa_rdna_decode --arch gfx1151

Writes ``<content>/sdpa_rdna_decode/<arch>/<instance>.hsaco`` for every row of
``_instances.sdpa_rdna_decode_instances()``, plus the ``manifest.json`` beside them and
``<content>/sdpa_rdna_decode/<arch>.SOURCE.md`` next to the folder.

The kernel is the vendored split-KV decode (``kernels_src/kernels/attention_rdna/
flash_attn_decode_rdna.py``): a GQA group's query heads packed into 16-row tiles, the
keys split across workgroups, and a second kernel in the same object that merges the
splits. It is its own engine (``hipkernel:flydsl_sdpa_rdna_decode``), separate from the
prefill kernel's (``gen_sdpa_rdna``): a different algorithm with its own argument lists
and launch. Both kernels' layouts are asserted before an object is written, and their
semantic argument names are carried into the manifest. No GPU is needed to build one.
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
from ._sdpa_rdna_build import LLVM_OPTIONS, WAVE_SIZE, verify_layout

OP = "sdpa_rdna_decode"

# The two kernels of a decode object. The main kernel: Q, K, V, O, LSE, the split
# workspace's O and (max, sum) pairs (pointers); the problem's i32 scalars and the
# split count; the scale; the (batch, sequence, head) strides of Q, K, V, O and LSE;
# the bias pointer and its (batch, head, query, key) strides; the runtime head_dim
# (read only by a generic object); the sink pointer and its head stride (read only by
# a sink object). Each appended, so every decode object has the same list. The merge
# kernel: O, LSE and the two workspace pointers; seq_len_q, num_heads, lse_on, num_splits; O's and LSE's
# strides; the runtime head_dim; the sink pointer and its head stride.
_DECODE_ARG_NAMES = (
    "Q",
    "K",
    "V",
    "O",
    "LSE",
    "WS_O",
    "WS_MAX_SUM",
    "seq_len_q",
    "seq_len_kv",
    "num_heads",
    "kv_group",
    "right_bound",
    "left_bound",
    "align_bottom_right",
    "lse_on",
    "num_splits",
    "scale",
    *(f"{t}_stride_{a}" for t in ("q", "k", "v", "o", "lse") for a in ("b", "s", "h")),
    "BIAS",
    "bias_stride_b",
    "bias_stride_h",
    "bias_stride_q",
    "bias_stride_k",
    "head_dim_rt",
    "SINK",
    "sink_stride",
)
_DECODE_SIGNATURE = (
    *(("global_buffer", 8, 8 * i) for i in range(7)),
    *(("by_value", 4, 56 + 4 * i) for i in range(10)),
    *(("by_value", 8, 96 + 8 * i) for i in range(15)),
    ("global_buffer", 8, 216),
    *(("by_value", 8, 224 + 8 * i) for i in range(4)),
    ("by_value", 4, 256),
    ("global_buffer", 8, 264),
    ("by_value", 8, 272),
)
_DECODE_KERNARG_SEGMENT_SIZE = 280
_MERGE_ARG_NAMES = (
    "O",
    "LSE",
    "WS_O",
    "WS_MAX_SUM",
    "seq_len_q",
    "num_heads",
    "lse_on",
    "num_splits",
    *(f"{t}_stride_{a}" for t in ("o", "lse") for a in ("b", "s", "h")),
    "head_dim_rt",
    "SINK",
    "sink_stride",
)
_MERGE_SIGNATURE = (
    *(("global_buffer", 8, 8 * i) for i in range(4)),
    *(("by_value", 4, 32 + 4 * i) for i in range(4)),
    *(("by_value", 8, 48 + 8 * i) for i in range(6)),
    ("by_value", 4, 96),
    ("global_buffer", 8, 104),
    ("by_value", 8, 112),
)
_MERGE_KERNARG_SEGMENT_SIZE = 120
# One workgroup per (batch, kv head, split, output-column tile of head_dim /
# dv_split) of DECODE_WAVES waves, by 16-row tile of the GQA group's rows along the
# grid's second dimension; the merge, one workgroup of DECODE_MERGE_THREADS per
# (batch, query head, query position).
_DECODE_GRID_RULE = "batch_x_kvheads_x_splits_x_dv_split_by_row_tiles"
_DECODE_WAVES = 4


def _build_decode(
    kernel_module, instance: Instance, dump_dir: Path, llvm_options: dict
) -> tuple[bytes, dict]:
    """Compile one decode instance (main and merge kernels); return its object."""
    import flydsl.compiler as flyc  # noqa: PLC0415  (after prepare())
    import flydsl.expr as fx  # noqa: PLC0415
    from flydsl.compiler.kernel_function import CompilationContext  # noqa: PLC0415

    knobs = instance.knobs
    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_flash_attn_decode_module(
        knobs["head_dim_max"],
        causal=bool(knobs["causal"]),
        dtype_str=instance.dtype,
        num_waves=_DECODE_WAVES,
        dv_split=knobs["dv_split"],
        has_bias=bool(knobs["has_bias"]),
        bias_query_dtype=bool(knobs["bias_q"]),
        generic_head_dim=knobs["head_dim"] is None,
        head_dim_v=None if knobs["head_dim"] is None else knobs["head_dim_v"],
        has_sink=bool(knobs["has_sink"]),
    )
    if launch.dv_split != knobs["dv_split"]:
        raise GeneratorError(
            f"{instance.name}: kernel built with dv_split {launch.dv_split}, "
            f"instance declares {knobs['dv_split']}"
        )
    null = flyc.from_c_void_p(fx.Uint8, 0)
    # Q, K, V, O, LSE, WS_O, WS_MAX_SUM; batch, seq_len_q, seq_len_kv, num_heads,
    # kv_group, right_bound, left_bound, align_bottom_right, lse_on, num_splits;
    # the scale; fifteen strides; the bias pointer and its four strides; the
    # runtime head_dim; the sink pointer and its head stride.
    args = [null] * 7 + [1, 1, 1, 1, 1, 0, -1, 0, 0, 1, 1.0] + [1] * 15
    args += [null] + [1] * 4 + [1]
    args += [null, 0]
    args.append(fx.Stream(None))
    effective = dict(getattr(launch, "compile_hints", {}).get("llvm_options") or {})
    effective.update(llvm_options)
    with CompilationContext.compile_hints({"llvm_options": effective}):
        flyc.compile(launch, *args)
    blob, _ = hsaco_from_dump(dump_dir)
    return blob, effective


def _compile(task: tuple):
    """One instance's code object and LLVM options, in a ``_parallel`` worker or here."""
    from kernels.attention_rdna import flash_attn_decode_rdna  # noqa: PLC0415

    instance, keep_ir, llvm_options = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_decode(
                flash_attn_decode_rdna, instance, dump_dir, llvm_options
            )
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    env.assert_flydsl_version()
    env.prepare(arch)

    op_dir = out_root / OP / arch
    op_dir.mkdir(parents=True, exist_ok=True)

    instances = instances_for(OP)
    llvm_options = LLVM_OPTIONS.get(arch, {})
    records: list[dict] = []

    built = compile_all(
        _compile,
        [(instance, keep_ir, llvm_options) for instance in instances],
        arch,
        lambda t: t[0].name,
    )

    for instance, (blob, effective_options) in zip(instances, built):

        filename = f"{instance.name}.hsaco"
        where = f"{arch}/{OP}/{filename}"
        record = instance.to_record()
        merge_symbol = instance.knobs["merge_symbol"]
        names = kernel_names(blob, where)
        if len(names) != 2 or merge_symbol not in names:
            raise GeneratorError(
                f"{where}: expected a main kernel and {merge_symbol!r}, found {names}"
            )
        main_symbol = next(n for n in names if n != merge_symbol)
        described = describe(blob, where, main_symbol)
        merge = describe(blob, where, merge_symbol)
        verify_arch(described, arch, where)
        verify_layout(
            described, _DECODE_SIGNATURE, _DECODE_KERNARG_SEGMENT_SIZE, where, "decode"
        )
        verify_layout(
            merge, _MERGE_SIGNATURE, _MERGE_KERNARG_SEGMENT_SIZE, where, "merge"
        )
        verify_generic(blob, arch, where)

        (op_dir / filename).write_bytes(blob)

        record.update(
            {
                "block": [_DECODE_WAVES * WAVE_SIZE, 1, 1],
                "grid_rule": _DECODE_GRID_RULE,
                "args": arg_records(_DECODE_ARG_NAMES, described["signature"]),
                "merge_args": arg_records(_MERGE_ARG_NAMES, merge["signature"]),
                "merge_vgprs": merge["vgpr_count"],
                "file": filename,
                "symbol": described["symbol"],
                "bytes": len(blob),
                "sha256": sha256(blob),
                "kernarg_segment_size": described["kernarg_segment_size"],
                "lds_bytes": described["group_segment_fixed_size"],
                "vgprs": described["vgpr_count"],
                "llvm_options": effective_options,
                "named_args": described["named_args"],
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
        help="GPU target, e.g. gfx11-generic (every RDNA3/RDNA3.5 part), "
        "gfx12-generic (every RDNA4 part) or a concrete gfx11/gfx120x arch. The "
        "kernels carry the gfx11 and gfx12 WMMA ABIs; one arch per invocation.",
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

    # Refused before compiling: built for another ABI, the kernel aborts the
    # process inside LLVM rather than raising.
    if not (
        args.arch.startswith("gfx11")
        or args.arch.startswith("gfx120")
        or args.arch == "gfx12-generic"
    ):
        print(
            f"error: the SDPA kernels target the gfx11 and gfx12 WMMA ABIs; "
            f"{args.arch!r} is neither",
            file=sys.stderr,
        )
        return 1

    try:
        generate(args.arch, args.out_dir, args.keep_ir)
    except (GeneratorError, env.FlydslEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
