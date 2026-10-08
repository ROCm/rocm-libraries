# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in CDNA SDPA-forward code objects for one architecture.

    python -m generators.gen_sdpa_cdna --arch gfx950

Writes ``<content>/sdpa_cdna/<arch>/<instance>.hsaco`` for every row of
``_instances.sdpa_cdna_instances(arch)``, plus the ``manifest.json`` beside them
and ``<content>/sdpa_cdna/<arch>.SOURCE.md`` next to the folder.

The kernels are FlyDSL's own CDNA attention forwards, vendored under
``kernels_src/kernels/attention_cdna/``: the dual-wave software-pipelined kernel
on gfx950 and the generic MFMA kernel on gfx942. Their launchers take
``fx.Tensor`` arguments, so each object is traced from torch ``meta`` tensors --
shapes and strides with no storage -- under ``COMPILE_ONLY=1``. No GPU is needed,
and the object is byte-identical to one traced from device tensors: every tensor
is lowered layout-dynamic, so the tracing shape is not baked.

Each tensor argument lowers to two kernel arguments: its 8-byte global pointer,
then a 40-byte by-value layout -- FlyDSL's dynamic-layout contract for a rank-4
tensor, the four i32 extents followed by the three i64 strides of the
non-unit-stride dimensions in ascending order (``flydsl/compiler/jit_argument.py``,
``_LayoutPlan``). Every object's layout is asserted against its family's below
before it is written, and the semantic argument names are carried into the
manifest: FlyDSL emits none, and the dispatcher marshals by position.
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
    hkp_pack_module,
    verify_arch,
)
from ._extract_hsaco import hsaco_from_dump
from ._instances import SDPA_CDNA_KERNEL, Instance, instances_for
from ._manifest import refresh_source_md, sha256, write_op_manifest

OP = "sdpa_cdna"

# One tensor argument: its pointer, then its dynamic layout.
_TENSOR_LAYOUT_BYTES = 40


def _signature(params: tuple[tuple[str, str], ...]) -> tuple[tuple, tuple]:
    """The (kind, size, offset) layout and the semantic names of a parameter list."""
    layout, names, offset = [], [], 0
    for name, kind in params:
        if kind == "tensor":
            layout += [
                ("global_buffer", 8, offset),
                ("by_value", _TENSOR_LAYOUT_BYTES, offset + 8),
            ]
            names += [name, f"{name}.layout"]
            offset += 8 + _TENSOR_LAYOUT_BYTES
        else:
            layout.append(("by_value", 4, offset))
            names.append(name)
            offset += 4
    return tuple(layout), tuple(names)


# The kernels' own parameter lists, in order (the @flyc.kernel signatures in
# flash_attn_gfx950.py and flash_attn_generic.py). A tensor the launch does not
# use is passed as O, as upstream's launchers do.
_PARAMS = {
    "dualwave": (
        ("Q", "tensor"),
        ("K", "tensor"),
        ("V", "tensor"),
        ("O", "tensor"),
        ("LSE", "tensor"),
        ("DebugCounts", "tensor"),
        ("CuSeqQ", "tensor"),
        ("CuSeqKv", "tensor"),
        ("BlockTable", "tensor"),
        ("Bias", "tensor"),
        ("AlibiSlopes", "tensor"),
        ("Sink", "tensor"),
        ("seq_len", "i32"),
        ("seq_len_kv", "i32"),
        ("stride_q_n", "i32"),
        ("stride_kv_n", "i32"),
        ("head_dim_runtime", "i32"),
        ("block_table_stride", "i32"),
        ("bias_stride0", "i32"),
        ("alibi_stride_b", "i32"),
    ),
    "generic": (
        ("Q", "tensor"),
        ("K", "tensor"),
        ("V", "tensor"),
        ("O", "tensor"),
        ("LSE", "tensor"),
        ("seq_len", "i32"),
        ("seq_len_kv", "i32"),
        ("CuSeqQ", "tensor"),
        ("CuSeqKv", "tensor"),
        ("BlockTable", "tensor"),
        ("block_table_stride", "i32"),
    ),
}

# Launch geometry the native dispatch handler reproduces (the launchers'
# `.launch(grid=...)`): the dual-wave kernel one workgroup per (query head, query
# block, batch); the generic one per (batch, query tile, query head), flattened.
_GRID_RULE = {
    "dualwave": "heads_by_qblocks_by_batch",
    "generic": "batch_x_qtiles_x_heads",
}
# Threads per workgroup: the dual-wave kernel's is fixed at 512 (two waves of 64
# per 32 query rows); the generic kernel runs block_m / 64 * 128 threads.
_BLOCK_THREADS = {
    "dualwave": lambda block_m: 512,
    "generic": lambda block_m: block_m * 2,
}

_TRACE_BATCH = 1
_TRACE_SEQ = 1024


def _build_one(instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance; return its code object."""
    import torch  # noqa: PLC0415  (after prepare(); meta tensors only)

    knobs = instance.knobs
    env.set_dump_dir(dump_dir)
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[instance.dtype]
    heads, kv_heads, head_dim = (
        knobs["num_heads"],
        knobs["num_kv_heads"],
        knobs["head_dim"],
    )
    q = torch.empty(
        _TRACE_BATCH, _TRACE_SEQ, heads, head_dim, dtype=dtype, device="meta"
    )
    k = torch.empty(
        _TRACE_BATCH, _TRACE_SEQ, kv_heads, head_dim, dtype=dtype, device="meta"
    )
    v = torch.empty_like(k)
    o = torch.empty_like(q)

    if knobs["kernel"] == "dualwave":
        from kernels.attention_cdna.flash_attn_gfx950 import (  # noqa: PLC0415
            build_flash_attn_dualwave_swp_module,
        )

        launch = build_flash_attn_dualwave_swp_module(
            heads,
            head_dim,
            causal=bool(knobs["causal"]),
            dtype_str=instance.dtype,
            num_kv_heads=kv_heads,
        )
    else:
        from kernels.attention_cdna.flash_attn_generic import (  # noqa: PLC0415
            build_flash_attn_func_module,
        )

        launch = build_flash_attn_func_module(
            heads,
            head_dim,
            causal=bool(knobs["causal"]),
            dtype_str=instance.dtype,
            num_kv_heads=kv_heads,
            block_m=knobs["block_m"],
            flat_work_group_size=_BLOCK_THREADS["generic"](knobs["block_m"]),
        )
    launch(q, k, v, o, _TRACE_BATCH, _TRACE_SEQ)
    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _max_flat_workgroup_size(blob: bytes, where: str) -> int:
    hkp = hkp_pack_module()
    document = hkp._metadata_document(hkp.amdgcn_object(blob, where), where)
    return int(document["amdhsa.kernels"][0][".max_flat_workgroup_size"])


def _verify(
    described: dict, blob: bytes, instance: Instance, arch: str, where: str
) -> None:
    """Fail before writing, rather than shipping a mislabelled object."""
    verify_arch(described, arch, where)
    kernel = instance.knobs["kernel"]
    expected, _ = _signature(_PARAMS[kernel])
    if described["signature"] != expected:
        raise GeneratorError(
            f"{where}: kernarg layout for {instance.name!r} is\n"
            f"  {described['signature']}\nbut the {kernel} family is declared as\n"
            f"  {expected}\n"
            "The kernel's parameter list changed; update _PARAMS, the native "
            "dispatch and its signature together."
        )
    size = expected[-1][2] + expected[-1][1]
    if described["kernarg_segment_size"] != size:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, expected {size}"
        )
    threads = _BLOCK_THREADS[kernel](instance.knobs["block_m"])
    if _max_flat_workgroup_size(blob, where) != threads:
        raise GeneratorError(
            f"{where}: built for {_max_flat_workgroup_size(blob, where)} threads per "
            f"workgroup, but the dispatcher launches {threads}"
        )
    if described.get("vgpr_spill_count"):
        raise GeneratorError(
            f"{where}: {described['vgpr_spill_count']} VGPRs spilled; the tile does "
            "not fit this arch's register file at these knobs"
        )


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    env.assert_flydsl_version()
    env.prepare(arch)

    op_dir = out_root / OP / arch
    op_dir.mkdir(parents=True, exist_ok=True)

    instances = instances_for(OP, arch)
    records: list[dict] = []

    for index, instance in enumerate(instances, start=1):
        print(f"[{index}/{len(instances)}] {instance.name}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
            dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
            try:
                blob = _build_one(instance, dump_dir)
            except Exception as exc:
                raise GeneratorError(f"{instance.name}: {exc}") from exc

        filename = f"{instance.name}.hsaco"
        where = f"{arch}/{OP}/{filename}"
        described = describe(blob, where)
        _verify(described, blob, instance, arch, where)
        kernel = instance.knobs["kernel"]
        _, names = _signature(_PARAMS[kernel])

        (op_dir / filename).write_bytes(blob)

        record = instance.to_record()
        record.update(
            {
                "block": [_BLOCK_THREADS[kernel](instance.knobs["block_m"]), 1, 1],
                "grid_rule": _GRID_RULE[kernel],
                "args": arg_records(names, described["signature"]),
                "file": filename,
                "symbol": described["symbol"],
                "bytes": len(blob),
                "sha256": sha256(blob),
                "kernarg_segment_size": described["kernarg_segment_size"],
                "lds_bytes": described["group_segment_fixed_size"],
                "vgprs": described["vgpr_count"],
                # The kernels carry their own compile hints; none are added here.
                "llvm_options": {},
                "named_args": described["named_args"],
            }
        )
        records.append(record)

    # Every CDNA kernel is FlyDSL's own; none comes from AITER, so its pin is not
    # part of how these bytes came to be.
    provenance = env.provenance(arch)
    provenance.pop("aiter_kernels_commit", None)
    write_op_manifest(op_dir, arch, OP, provenance, records)
    source_md = refresh_source_md(op_dir)
    print(f"wrote {len(records)} object(s) to {op_dir}")
    print(f"refreshed {source_md}")
    return op_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arch",
        required=True,
        help=f"GPU target: one of {', '.join(sorted(SDPA_CDNA_KERNEL))}; one arch per invocation.",
    )
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

    # Refused before compiling: these kernels are CDNA MFMA code.
    if args.arch not in SDPA_CDNA_KERNEL:
        print(
            f"error: the CDNA SDPA kernels target {', '.join(sorted(SDPA_CDNA_KERNEL))}; "
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
