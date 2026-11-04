# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in code objects of AOTriton's FlyDSL gfx1201 attention kernel.

    python -m generators.gen_sdpa_aiw --arch gfx12-generic

Writes ``<content>/sdpa_aiw/<arch>/<instance>.hsaco`` for every row of
``_instances.sdpa_aiw_instances(arch)``, plus the ``manifest.json`` beside them and
``<content>/sdpa_aiw/<arch>.SOURCE.md`` next to the folder.

The kernel is vendored under ``kernels_src/kernels/attention_rdna4/`` (its README lists the
modifications). It is one engine's (``hipkernel:flydsl_sdpa_aiw``): a kernel family of its
own, beside the RDNA prefill and decode kernels'. Each row is built through the kernel's
own tuning table (``fmha_tuning_gfx1201.plan``), which rounds the head_dim to a compiled
tile and picks the schedule; the schedule it picked is recorded with the object as knobs, so
a ranker can read what each object was built with. The launcher's argument list is traced
from placeholder CPU tensors under ``COMPILE_ONLY=1``; no GPU is needed.

Every object's argument layout is asserted against the 50 parameters declared below before
it is written, and their names are carried into the manifest.
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
    verify_generic,
)
from ._extract_hsaco import hsaco_from_dump
from ._instances import SDPA_AIW_TARGETS, Instance, instances_for
from ._manifest import refresh_source_md, sha256, write_op_manifest
from ._parallel import compile_all

OP = "sdpa_aiw"

# The kernel's parameters, in order (the @flyc.kernel signature in
# flash_attn_func_gfx1201_aiw.py, with modifications 3 and 5-7 appended).
_PARAMS = (
    ("Q", "ptr"),
    ("K", "ptr"),
    ("V", "ptr"),
    ("B", "ptr"),
    ("O", "ptr"),
    ("LSE", "ptr"),
    ("seqinfo_q0", "ptr"),
    ("seqinfo_q1", "ptr"),
    ("seqinfo_k0", "ptr"),
    ("seqinfo_k1", "ptr"),
    ("varlen_bits", "i32"),
    ("num_seqlens", "i32"),
    ("max_seqlen_q", "i32"),
    ("max_seqlen_k", "i32"),
    ("window_left", "i32"),
    ("window_right", "i32"),
    ("philox_seed_ptr", "ptr"),
    ("philox_offset1", "ptr"),
    ("philox_offset2", "i64"),
    ("philox_seed_output", "ptr"),
    ("philox_offset_output", "ptr"),
    ("idropout_p", "i32"),
    ("dropout_scale", "f32"),
    ("num_head_q", "i32"),
    ("num_head_k", "i32"),
    ("hdim_qk", "i32"),
    ("hdim_vo", "i32"),
    ("sm_scale", "f32"),
    *((f"stride_{t}_{a}", "i64") for t in "qkvo" for a in ("batch", "head", "seq")),
    ("stride_b_batch", "i64"),
    ("stride_b_head", "i64"),
    ("stride_b_seq_q", "i64"),
    ("SINK", "ptr"),
    ("sink_stride", "i64"),
    ("stride_b_seq_k", "i64"),
    ("num_head_v", "i32"),
    ("lse_stride_b", "i64"),
    ("lse_stride_h", "i64"),
    ("lse_stride_s", "i64"),
)
_SIZES = {"ptr": 8, "i64": 8, "i32": 4, "f32": 4}


def _signature() -> tuple[tuple, tuple]:
    """The (kind, size, offset) layout and the names of the parameter list: each argument
    at the next offset its own size aligns to."""
    layout, names, offset = [], [], 0
    for name, kind in _PARAMS:
        size = _SIZES[kind]
        offset = (offset + size - 1) // size * size
        layout.append(("global_buffer" if kind == "ptr" else "by_value", size, offset))
        names.append(name)
        offset += size
    return tuple(layout), tuple(names)


# The launch the native dispatch handler reproduces: one workgroup per (query head, query
# tile of block_m rows, batch), of the kernel's own workgroup size.
_GRID_RULE = "heads_by_qtiles_by_batch"

# The tuning-table fields recorded with each object as knobs: what a ranker can tell the
# objects of one class apart by. The rest of the plan is fixed across the family.
_SCHEDULE_KNOBS = (
    "block_m",
    "block_n",
    "row_subtiles",
    "waves_per_eu",
    "k_prefetch_dist",
    "v_prefetch_dist",
    "kv_addr_hoist",
)

# VGPR spills accepted, by (dtype-independent) class: (head_dim_max, causal, has_bias,
# bias_f32, padded). A spill is refused anywhere else, and above the count measured here;
# each object records its own (`vgpr_spills`). Each is the tuning table's own schedule:
# - head_dim 256, the one AOTriton ships, which spills 21-50 and is still 1.36-1.71x
#   faster than the RDNA kernel's head_dim 256 objects on medium and long sequences;
# - a mask at head_dim 128 (the f32 one's v8 load holds twice the registers of a 16-bit
#   one, README modification 5; the non-causal f32 one spills one more since the
#   natural-log scores of modification 9);
# - the padded tiers, which carry the column masks on top of their tile's schedule. Those
#   at 224 and 256 spill the most, and whether they beat the RDNA generic tiers anywhere
#   is for autotune (then UHD) to measure.
_SPILL_ALLOWANCE: dict[tuple[int, int, int, int, int], int] = {
    (128, 0, 1, 0, 0): 3,
    (128, 0, 1, 1, 0): 4,
    (128, 1, 1, 1, 0): 4,
    (256, 0, 0, 0, 0): 21,
    (256, 0, 1, 0, 0): 30,
    (256, 0, 1, 1, 0): 30,
    (256, 1, 0, 0, 0): 32,
    (256, 1, 1, 0, 0): 50,
    (256, 1, 1, 1, 0): 50,
    (128, 0, 0, 0, 1): 6,
    (160, 0, 0, 0, 1): 6,
    (224, 0, 0, 0, 1): 23,
    (256, 0, 0, 0, 1): 61,
    (128, 1, 0, 0, 1): 10,
    (224, 1, 0, 0, 1): 20,
    (256, 1, 0, 0, 1): 63,
}


def _plan(instance: Instance):
    from kernels.attention_rdna4 import fmha_tuning_gfx1201 as tuning  # noqa: PLC0415

    knobs = instance.knobs
    # A padded tier is built for a head_dim below its tile, which is what makes the tuning
    # table round it up and mark the build padded; the real head_dim is a runtime argument.
    head_dim = (
        knobs["head_dim"]
        if knobs["head_dim"] is not None
        else knobs["head_dim_max"] - 8
    )
    # A padded tier's V is its Q/K head_dim: left to the tuning table, which pads both alike.
    head_dim_v = knobs["head_dim_v"] if knobs["head_dim"] is not None else None
    meta = tuning.FmhaInputMetadata(
        num_heads=8,
        head_dim=head_dim,
        causal=bool(knobs["causal"]),
        dtype_str=instance.dtype,
        head_dim_v=head_dim_v,
        bias=bool(knobs["has_bias"]),
        bias_f32=bool(knobs["bias_f32"]),
        bias_key_strided=bool(knobs["bias_kstrided"]),
        sink=bool(knobs["has_sink"]),
    )
    plan = tuning.plan(meta)
    if plan.knobs.block_dmodel != knobs["head_dim_max"]:
        raise GeneratorError(
            f"{instance.name}: the tuning table builds a {plan.knobs.block_dmodel}-wide tile, "
            f"the instance declares head_dim_max {knobs['head_dim_max']}"
        )
    if bool(plan.knobs.padded_head) != (knobs["head_dim"] is None):
        raise GeneratorError(
            f"{instance.name}: padded_head {plan.knobs.padded_head} disagrees with the row"
        )
    return plan, head_dim, head_dim if head_dim_v is None else head_dim_v


def _build_one(instance: Instance, dump_dir: Path):
    """Compile one instance; return its code object and the plan it was built with."""
    import torch  # noqa: PLC0415  (after prepare(); placeholders only)

    from kernels.attention_rdna4 import (
        flash_attn_func_gfx1201_aiw as aiw,
    )  # noqa: PLC0415

    env.set_dump_dir(dump_dir)
    plan, head_dim, head_dim_v = _plan(instance)
    launch = aiw.build_flash_attn_func_aiw_module_primary(plan.meta, plan.knobs)
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[instance.dtype]
    knobs = instance.knobs
    # Placeholders: their shapes and strides only shape the argument list, every value of
    # which is a runtime kernel argument; nothing is dispatched.
    q = torch.zeros(1, 8, 128, head_dim, dtype=dtype)
    k = torch.zeros(1, 8, 128, head_dim, dtype=dtype)
    v = torch.zeros(1, 8, 128, head_dim_v, dtype=dtype)
    o = torch.zeros(1, 8, 128, head_dim_v, dtype=dtype)
    lse = torch.zeros(8, 128, dtype=torch.float32)
    sink = torch.zeros(8, dtype=dtype) if knobs["has_sink"] else None
    bias = None
    if knobs["has_bias"]:
        bias = torch.zeros(
            1, 8, 128, 128, dtype=torch.float32 if knobs["bias_f32"] else dtype
        )
    launch.compile(q, k, v, o, 1, 128, 128, lse=lse, sink=sink, bias=bias)
    blob, _ = hsaco_from_dump(dump_dir)
    return blob, plan


def _workgroup_size(blob: bytes, where: str, symbol: str) -> int:
    hkp = hkp_pack_module()
    document = hkp._metadata_document(hkp.amdgcn_object(blob, where), where)
    kernel = next(
        k
        for k in document["amdhsa.kernels"]
        if k[".symbol"].removesuffix(".kd") == symbol
    )
    return int(kernel[".max_flat_workgroup_size"])


def _compile(task: tuple):
    """One instance's code object and plan, in a ``_parallel`` worker or this process."""
    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def _kernel_block_m(knobs) -> int:
    """The query rows one workgroup covers, as the kernel resolves them: the block_m knob
    with the K prefetch off, else derived the way `flash_attn_func_gfx1201_aiw` derives it,
    through the tuning module's own policy functions."""
    from kernels.attention_rdna4 import fmha_tuning_gfx1201 as tuning  # noqa: PLC0415

    if knobs.k_prefetch_dist == 0:
        return knobs.block_m
    if knobs.shards is not None:
        shards = knobs.shards
    elif knobs.row_subtiles > 1:
        shards = 1
    else:
        shards = tuning.resolve_shards(
            knobs.block_dmodel, knobs.block_dmodel_v, knobs.block_n
        )
    q_tiles = max(
        1, tuning.q_tiles_per_block(knobs.block_dmodel, shards) // knobs.row_subtiles
    )
    return 16 * knobs.row_subtiles * q_tiles


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path | None:
    instances = instances_for(OP, arch)
    if not instances:
        print(f"{OP} has no instances for {arch}; nothing to build")
        return None
    env.assert_flydsl_version()
    env.prepare(arch)

    op_dir = out_root / OP / arch
    op_dir.mkdir(parents=True, exist_ok=True)
    expected, names = _signature()
    records: list[dict] = []

    built = compile_all(
        _compile,
        [(instance, keep_ir) for instance in instances],
        arch,
        lambda t: t[0].name,
    )

    for instance, (blob, plan) in zip(instances, built):

        filename = f"{instance.name}.hsaco"
        where = f"{arch}/{OP}/{filename}"
        described = describe(blob, where)
        verify_arch(described, arch, where)
        if described["signature"] != expected:
            raise GeneratorError(
                f"{where}: kernarg layout is\n  {described['signature']}\nbut is declared as\n"
                f"  {expected}\nUpdate _PARAMS, the native dispatch and its signature together."
            )
        threads = _workgroup_size(blob, where, described["symbol"])
        knobs = instance.knobs
        allowed = _SPILL_ALLOWANCE.get(
            (
                knobs["head_dim_max"],
                knobs["causal"],
                knobs["has_bias"],
                knobs["bias_f32"],
                int(knobs["head_dim"] is None),
            ),
            0,
        )
        if described.get("vgpr_spill_count", 0) > allowed:
            raise GeneratorError(
                f"{where}: {described['vgpr_spill_count']} VGPRs spilled; allowed {allowed}"
            )
        verify_generic(blob, arch, where)

        (op_dir / filename).write_bytes(blob)

        record = instance.to_record()
        # The dispatcher's grid tiles the queries by the recorded block_m. With the K
        # prefetch on, the kernel derives its row tile instead (from the head dim, the QK
        # shards and the row sub-tiles): refuse an object whose two disagree, which would
        # launch the wrong number of query tiles.
        derived = _kernel_block_m(plan.knobs)
        if derived != plan.knobs.block_m:
            raise GeneratorError(
                f"{where}: records block_m {plan.knobs.block_m} but the kernel tiles "
                f"{derived} query rows per workgroup"
            )
        # The schedule the tuning table chose, and the workgroup size it implies: knobs a
        # ranker can read, and the block the dispatcher launches.
        schedule = {name: getattr(plan.knobs, name) for name in _SCHEDULE_KNOBS}
        schedule = {
            k: int(v) if isinstance(v, bool) else v for k, v in schedule.items()
        }
        record["knobs"].update(schedule)
        record["knobs"]["block_threads"] = threads
        record.update(
            {
                "block": [threads, 1, 1],
                "grid_rule": _GRID_RULE,
                "args": arg_records(names, described["signature"]),
                "file": filename,
                "symbol": described["symbol"],
                "bytes": len(blob),
                "sha256": sha256(blob),
                "kernarg_segment_size": described["kernarg_segment_size"],
                "lds_bytes": described["group_segment_fixed_size"],
                "vgprs": described["vgpr_count"],
                "vgpr_spills": described.get("vgpr_spill_count", 0),
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
        help=f"GPU target: {', '.join(SDPA_AIW_TARGETS)}; other targets build nothing.",
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
    try:
        generate(args.arch, args.out_dir, args.keep_ir)
    except (GeneratorError, env.FlydslEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
