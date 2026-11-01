# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Build the checked-in CDNA SDPA-forward code objects: what the three CDNA SDPA
generators share.

The CDNA kernels form three engines, one kernel family each (``_instances``,
``SDPA_CDNA_FAMILY_CONSTANTS``): ``gen_sdpa_cdna`` (the generic kernel's prefill
objects), ``gen_sdpa_cdna_dualwave`` (gfx950's dual-wave kernel) and
``gen_sdpa_cdna_decode`` (the generic kernel's split-KV objects). Each writes
``<content>/<op>/<arch>/<instance>.hsaco`` for every row of its table, plus the
``manifest.json`` beside them and ``<content>/<op>/<arch>.SOURCE.md`` next to the folder.
An instance carries only its own family's knobs; the knobs that told the families apart
are restored from the family's constants before it is built, so every object builds
exactly as it did when the three were one table.

The kernels are FlyDSL's own CDNA attention forwards, vendored under
``kernels_src/kernels/attention_cdna/``: the dual-wave software-pipelined kernel
on gfx950 and the generic MFMA kernel on gfx942 and gfx90a. Their launchers take
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
import os
import sys
import tempfile
from pathlib import Path

from . import _flydsl_env as env
from ._codeobject import (
    GeneratorError,
    arg_records,
    describe,
    hkp_pack_module,
    kernel_names,
    verify_arch,
)
from ._extract_hsaco import hsaco_from_dump
from ._instances import (
    SDPA_CDNA_FAMILY_CONSTANTS,
    SDPA_CDNA_TIERS,
    Instance,
    instances_for,
)
from ._manifest import refresh_source_md, sha256, write_op_manifest
from ._parallel import compile_all

# One tensor argument: its pointer, then its dynamic layout.
_TENSOR_LAYOUT_BYTES = 40


def _signature(params: tuple[tuple[str, str], ...]) -> tuple[tuple, tuple]:
    """The (kind, size, offset) layout and the semantic names of a parameter list."""
    layout, names, offset = [], [], 0
    for name, kind in params:
        if kind == "tensor":
            # A pointer is 8-byte aligned in the kernarg segment.
            offset = (offset + 7) // 8 * 8
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
        ("sm_scale", "f32"),
        ("num_heads_q", "i32"),
        ("num_heads_kv", "i32"),
        ("lse_on", "i32"),
        ("diag_offset", "i32"),
        ("stride_q_b", "i32"),
        ("stride_q_h", "i32"),
        ("stride_o_b", "i32"),
        ("stride_o_n", "i32"),
        ("stride_o_h", "i32"),
        ("stride_kv_b", "i32"),
        ("stride_kv_h", "i32"),
        ("stride_bias_b", "i32"),
        ("stride_bias_h", "i32"),
    ),
    "generic": (
        ("Q", "tensor"),
        ("K", "tensor"),
        ("V", "tensor"),
        ("O", "tensor"),
        ("LSE", "tensor"),
        ("seq_len", "i32"),
        ("seq_len_kv", "i32"),
        # Upstream's cu_seqlens and page-table stride slots, which the dense objects
        # read as a padded launch's per-batch lengths and its varlen flags
        # (vendored-kernel modification 32).
        ("SeqLenQ", "tensor"),
        ("SeqLenKv", "tensor"),
        ("BlockTable", "tensor"),
        ("varlen_flags", "i32"),
        ("sm_scale", "f32"),
        ("num_heads_q", "i32"),
        ("num_heads_kv", "i32"),
        ("stride_q_n", "i32"),
        ("stride_kv_n", "i32"),
        ("lse_on", "i32"),
        ("diag_offset", "i32"),
        ("stride_q_b", "i32"),
        ("stride_q_h", "i32"),
        ("stride_o_b", "i32"),
        ("stride_o_n", "i32"),
        ("stride_o_h", "i32"),
        ("stride_kv_b", "i32"),
        ("stride_kv_h", "i32"),
        ("window_lo", "i32"),
        ("Bias", "tensor"),
        ("bias_stride0", "i32"),
        ("stride_bias_b", "i32"),
        ("stride_bias_h", "i32"),
        ("stride_v_n", "i32"),
        ("stride_v_b", "i32"),
        ("stride_v_h", "i32"),
        ("head_dim_qk", "i32"),
        ("WsO", "tensor"),
        ("WsLse", "tensor"),
        ("num_splits", "i32"),
        ("Sink", "tensor"),
        ("sink_stride", "i32"),
        ("sink_on", "i32"),
        ("stride_lse_b", "i32"),
        ("stride_lse_h", "i32"),
        ("stride_lse_n", "i32"),
        # The bias's key stride and element type, read by a strided bias build only
        # (vendored-kernel modification 31).
        ("stride_bias_k", "i32"),
        ("bias_f32", "i32"),
        # Packed sequences' row offsets and the dropout arguments: the Philox seed and
        # offset, the offset's by-value part (two words), the keep threshold and the
        # scale (vendored-kernel modification 32).
        ("RowOffQ", "tensor"),
        ("RowOffKv", "tensor"),
        ("DropoutSeed", "tensor"),
        ("DropoutOffset", "tensor"),
        ("dropout_offset2_lo", "i32"),
        ("dropout_offset2_hi", "i32"),
        ("idropout_p", "i32"),
        ("dropout_scale", "f32"),
        # V's own group size, Hq / Hv (vendored-kernel modification 42).
        ("v_group", "i32"),
    ),
}

# The split objects' merge kernel (vendored-kernel modification 14).
_MERGE_PARAMS = (
    ("O", "tensor"),
    ("LSE", "tensor"),
    ("WsO", "tensor"),
    ("WsLse", "tensor"),
    ("seq_len", "i32"),
    ("num_heads_q", "i32"),
    ("lse_on", "i32"),
    ("num_splits", "i32"),
    ("stride_o_b", "i32"),
    ("stride_o_n", "i32"),
    ("stride_o_h", "i32"),
    ("Sink", "tensor"),
    ("sink_stride", "i32"),
    ("sink_on", "i32"),
    ("stride_lse_b", "i32"),
    ("stride_lse_h", "i32"),
    ("stride_lse_n", "i32"),
    # A padded launch's per-batch query lengths, packed sequences' row offsets, and the
    # varlen flags (vendored-kernel modification 32).
    ("SeqLenQ", "tensor"),
    ("RowOffQ", "tensor"),
    ("varlen_flags", "i32"),
)

# Launch geometry the native dispatch handler reproduces (the launchers'
# `.launch(grid=...)`): the dual-wave kernel one workgroup per (query head, query
# block, batch); the generic one per (batch, query tile, query head), flattened.
_GRID_RULE = {
    "dualwave": "heads_by_qblocks_by_batch",
    "generic": "batch_x_qtiles_x_heads",
}
# Threads per workgroup: the dual-wave kernel runs a wave of 64 per 32 query rows (512 on
# its 256-row tile, 256 on head_dim 256's 128-row one); the generic kernel runs
# block_m / 64 * 128 threads.
_BLOCK_THREADS = {
    "dualwave": lambda block_m: block_m * 2,
    "generic": lambda block_m: block_m * 2,
}

# LLVM options added to the generic kernel's own, per (arch, head_dim), recorded per
# object as `llvm_options`. On gfx942 at head_dim 64 the register-pressure trackers
# keep the objects under the occupancy step (three waves per SIMD) the runtime
# strides otherwise push them past; at 128 they cut registers but schedule the loop
# measurably worse, so they apply only where they were measured to help.
_GENERIC_LLVM_OPTIONS = {("gfx942", 64): {"amdgpu-use-amdgpu-trackers": True}}

# The generic kernel's switch for its KV global prefetch (the `kv_gpfetch` knob).
_KV_GPFETCH_ENV = "FLYDSL_FLASH_ATTN_FUNC_ENABLE_KV_GPFETCH"

_TRACE_BATCH = 1
_TRACE_SEQ = 1024
# The head counts the objects are traced with. They are kernel arguments, so these
# only satisfy the builders' own checks; nothing about them is baked.
_TRACE_HEADS = 32
_TRACE_KV_HEADS = 8


# LDS row padding for the generic kernel, per (arch, head_dim), where upstream's
# default does not fit: at head_dim 256 its padded K and transposed-V tiles need
# 67,072 bytes, past gfx942's 64 KiB, and unpadded they need exactly 64 KiB. K keeps
# its XOR swizzle against bank conflicts. Recorded per object as `lds_pads`.
_GENERIC_LDS_PADS = {
    ("gfx942", 256): {"k_pad": 0, "vt_pad": 0},
    # gfx90a's LDS is gfx942's 64 KiB.
    ("gfx90a", 256): {"k_pad": 0, "vt_pad": 0},
}


def _lds_pads(instance: Instance, arch: str) -> dict:
    """The generic kernel's LDS padding for @p instance, or {} for upstream's."""
    # The f32 objects keep the padded tiles: their 256 splits O's columns (modification 35).
    if instance.knobs["kernel"] != "generic" or instance.dtype == "f32":
        return {}
    # A tile that stages K in column chunks or V in column slices holds upstream's padded
    # rows within the 64 KiB: it keeps them.
    if instance.knobs.get("k_chunk") or int(instance.knobs.get("dv_split", 1)) > 1:
        return {}
    return dict(_GENERIC_LDS_PADS.get((arch, instance.knobs["head_dim"]), {}))


def _llvm_options(instance: Instance, arch: str) -> dict:
    """The LLVM options this pack adds to the kernel's own for @p instance."""
    # Measured on the 16-bit objects only; f32's tiles were chosen without them.
    if instance.knobs["kernel"] != "generic" or instance.dtype == "f32":
        return {}
    return dict(_GENERIC_LLVM_OPTIONS.get((arch, instance.knobs["head_dim"]), {}))


def _build_one(instance: Instance, dump_dir: Path, arch: str) -> bytes:
    """Compile one instance; return its code object."""
    import torch  # noqa: PLC0415  (after prepare(); meta tensors only)

    knobs = instance.knobs
    env.set_dump_dir(dump_dir)
    dtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[
        instance.dtype
    ]
    heads, kv_heads, head_dim = _TRACE_HEADS, _TRACE_KV_HEADS, knobs["head_dim"]
    q = torch.empty(
        _TRACE_BATCH, _TRACE_SEQ, heads, head_dim, dtype=dtype, device="meta"
    )
    k = torch.empty(
        _TRACE_BATCH, _TRACE_SEQ, kv_heads, head_dim, dtype=dtype, device="meta"
    )
    v = torch.empty_like(k)
    o = torch.empty_like(q)
    # The LSE output is compiled in and switched at launch by `lse_on`. The kernels
    # read only its base pointer, so this stand-in exists to give the argument the
    # same rank-4, one-unit-stride layout every other tensor argument has.
    lse = torch.empty(
        _TRACE_BATCH, heads, _TRACE_SEQ, 2, dtype=torch.float32, device="meta"
    )

    if knobs["kernel"] == "dualwave":
        from kernels.attention_cdna.flash_attn_gfx950 import (  # noqa: PLC0415
            build_flash_attn_dualwave_swp_module,
        )

        has_bias = bool(knobs["has_bias"])
        launch = build_flash_attn_dualwave_swp_module(
            heads,
            head_dim,
            causal=bool(knobs["causal"]),
            dtype_str=instance.dtype,
            num_kv_heads=kv_heads,
            return_lse=True,
            has_bias=has_bias,
            # Built to handle any diagonal -- q-blocks no key reaches, and a diagonal
            # tile the pipeline would otherwise skip masking -- unless the object is
            # the offset-0 one (`any_diagonal` 0), which the matcher holds to that.
            cross_seqlen=bool(knobs["causal"]) and bool(knobs["any_diagonal"]),
            dualwave_swp_setprio=bool(knobs["setprio"]),
            # 32 query rows a wave: head_dim 256's 128-row tile runs 4 (vendored-kernel
            # modification 43).
            **(
                {"num_waves": knobs["block_m"] // 32} if knobs["block_m"] != 256 else {}
            ),
            **(
                {"waves_per_eu": knobs["waves_per_eu"]} if knobs["waves_per_eu"] else {}
            ),
        )
    else:
        from kernels.attention_cdna.flash_attn_generic import (  # noqa: PLC0415
            build_flash_attn_func_module,
        )

        # Read by the generic kernel's traits at build time, and cleared by generate().
        if not knobs["kv_gpfetch"]:
            os.environ[_KV_GPFETCH_ENV] = "0"
        else:
            os.environ.pop(_KV_GPFETCH_ENV, None)
        launch = build_flash_attn_func_module(
            heads,
            head_dim,
            causal=bool(knobs["causal"]),
            dtype_str=instance.dtype,
            num_kv_heads=kv_heads,
            block_m=knobs["block_m"],
            flat_work_group_size=_BLOCK_THREADS["generic"](knobs["block_m"]),
            # Always explicit: left unset, the launcher picks the variant by the
            # tracing length, and one that skips the KV tail mask is wrong at any
            # length that is not a multiple of the KV tile. `kv_align` says which
            # this object is, and the kernel matcher holds the graph to it.
            skip_kv_pad_mask=bool(knobs["kv_align"]),
            return_lse=True,
            extra_llvm_options=_llvm_options(instance, arch),
            has_bias=bool(knobs["has_bias"]),
            head_dim_v=knobs.get("head_dim_v") or head_dim,
            qk_mask=bool(knobs.get("qk_mask", 0)),
            split_kv=bool(knobs.get("split_kv", 0)),
            pack_gqa=bool(knobs.get("pack_gqa", 0)),
            dv_split=int(knobs.get("dv_split", 1)),
            k_chunk=knobs.get("k_chunk") or None,
            has_sink=bool(knobs.get("has_sink", 0)),
            bias_kstrided=bool(knobs.get("bias_kstrided", 0)),
            has_dropout=bool(knobs.get("has_dropout", 0)),
            v_mask=bool(knobs.get("v_mask", 0)),
            **_lds_pads(instance, arch),
            **(
                {"waves_per_eu": knobs["waves_per_eu"]} if knobs["waves_per_eu"] else {}
            ),
        )
    if knobs["has_bias"]:
        # A rank-4 stand-in, like every other tensor argument; the kernel reads
        # only its pointer and takes its strides as arguments.
        bias = torch.empty(
            _TRACE_BATCH, heads, _TRACE_SEQ, _TRACE_SEQ, dtype=dtype, device="meta"
        )
        launch(q, k, v, o, _TRACE_BATCH, _TRACE_SEQ, lse=lse, bias=bias)
    else:
        launch(q, k, v, o, _TRACE_BATCH, _TRACE_SEQ, lse=lse)
    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _max_flat_workgroup_size(blob: bytes, where: str, symbol: str) -> int:
    hkp = hkp_pack_module()
    document = hkp._metadata_document(hkp.amdgcn_object(blob, where), where)
    kernel = next(
        k
        for k in document["amdhsa.kernels"]
        if k[".symbol"].removesuffix(".kd") == symbol
    )
    return int(kernel[".max_flat_workgroup_size"])


# VGPR spills accepted for a class, keyed by (arch, kernel, head_dim, block_m,
# has_bias). A spill is refused anywhere else; an accepted one is measured and
# recorded with the object (`vgpr_spills` in its manifest record), so it cannot
# creep in unseen. Each was timed against the candidates that do not spill, and won
# on the shapes its tier is for (COVERAGE.md section 7.6):
# - the dual-wave kernel at head_dim 128 with a bias. With the softmax scale on the
#   f32 scores (kernel modification 21) it spills up to 13 VGPRs and pays for it on
#   causal shapes, yet still beats the generic kernel's bias objects everywhere;
# - the generic kernel's 64-row tile at head_dim 128 on gfx950 (causal);
# - the generic kernel's 256-row tile at head_dim 128 on gfx942.
_SPILL_ALLOWANCE: dict[tuple[str, str, int, int, int], int] = {
    ("gfx950", "dualwave", 128, 256, 1): 13,
    ("gfx950", "generic", 128, 64, 0): 6,
    ("gfx942", "generic", 128, 256, 0): 2,
}


# The f32 objects' (vendored-kernel modification 35), keyed by (arch, head_dim, block_m,
# causal): each was timed against the tile that does not spill.
_F32_SPILL_ALLOWANCE: dict[tuple[str, int, int, int], int] = {}


def _spill_allowance(instance: Instance, arch: str) -> int:
    knobs = instance.knobs
    if instance.dtype == "f32":
        return _F32_SPILL_ALLOWANCE.get(
            (arch, knobs["head_dim"], knobs["block_m"], knobs["causal"]), 0
        )
    return _SPILL_ALLOWANCE.get(
        (arch, knobs["kernel"], knobs["head_dim"], knobs["block_m"], knobs["has_bias"]),
        0,
    )


# LDS one workgroup may allocate. The compiler reports an object over it as an
# error, yet still emits the object, which then fails to launch.
_LDS_LIMIT_BYTES = {"gfx90a": 65536, "gfx942": 65536, "gfx950": 163840}


def _verify(
    described: dict, blob: bytes, instance: Instance, arch: str, where: str
) -> None:
    """Fail before writing, rather than shipping a mislabelled object."""
    verify_arch(described, arch, where)
    if described["group_segment_fixed_size"] > _LDS_LIMIT_BYTES[arch]:
        raise GeneratorError(
            f"{where}: needs {described['group_segment_fixed_size']} bytes of LDS; "
            f"{arch} allows {_LDS_LIMIT_BYTES[arch]} per workgroup"
        )
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
    if _max_flat_workgroup_size(blob, where, described["symbol"]) != threads:
        raise GeneratorError(
            f"{where}: built for {_max_flat_workgroup_size(blob, where, described["symbol"])} threads per "
            f"workgroup, but the dispatcher launches {threads}"
        )
    allowed = _spill_allowance(instance, arch)
    if described.get("vgpr_spill_count", 0) > allowed:
        raise GeneratorError(
            f"{where}: {described['vgpr_spill_count']} VGPRs spilled; the tile does "
            "not fit this arch's register file at these knobs"
        )


def _as_built(op: str, instance: Instance) -> Instance:
    """@p instance with its family's constant knobs restored: what the builder reads."""
    return Instance(
        op=instance.op,
        dtype=instance.dtype,
        priority=instance.priority,
        knobs={**SDPA_CDNA_FAMILY_CONSTANTS[op], **instance.knobs},
    )


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    op, arch, shipped, keep_ir = task
    instance = _as_built(op, shipped)
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{shipped.name}-") as tmp:
        dump_dir = Path(keep_ir) / shipped.name if keep_ir else Path(tmp)
        try:
            return _build_one(instance, dump_dir, arch)
        except Exception as exc:
            raise GeneratorError(f"{shipped.name}: {exc}") from exc


def generate(
    op: str, arch: str, out_root: Path, keep_ir: Path | None = None
) -> Path | None:
    """Build every instance of @p op for @p arch; None when the op has none there."""
    instances = instances_for(op, arch)
    if not instances:
        print(f"{op} has no instances for {arch}; nothing to build")
        return None
    env.assert_flydsl_version()
    env.prepare(arch)
    # The generic kernel reads these at build time; pinned so no shell changes the
    # bytes. Its FLYDSL_FLASH_ATTN_FUNC_* switches are left at upstream's defaults.
    os.environ["FLYDSL_LLVM_ENABLE_POST_MISCHED"] = "0"
    for name in [n for n in os.environ if n.startswith("FLYDSL_FLASH_ATTN_FUNC_")]:
        del os.environ[name]

    op_dir = out_root / op / arch
    op_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    blobs = compile_all(
        _compile,
        [(op, arch, shipped, keep_ir) for shipped in instances],
        arch,
        lambda task: task[2].name,
    )

    for shipped, blob in zip(instances, blobs):
        instance = _as_built(op, shipped)

        filename = f"{shipped.name}.hsaco"
        where = f"{arch}/{op}/{filename}"
        merge_symbol = instance.knobs["merge_symbol"]
        merge = None
        if merge_symbol:
            found = kernel_names(blob, where)
            if len(found) != 2 or merge_symbol not in found:
                raise GeneratorError(
                    f"{where}: expected a main kernel and {merge_symbol!r}, found {found}"
                )
            described = describe(
                blob, where, next(n for n in found if n != merge_symbol)
            )
            merge = describe(blob, where, merge_symbol)
            expected_merge, _ = _signature(_MERGE_PARAMS)
            if merge["signature"] != expected_merge:
                raise GeneratorError(
                    f"{where}: the merge kernel's kernarg layout is {merge['signature']}, "
                    f"declared {expected_merge}"
                )
        else:
            described = describe(blob, where)
        _verify(described, blob, instance, arch, where)
        kernel = instance.knobs["kernel"]
        _, names = _signature(_PARAMS[kernel])

        (op_dir / filename).write_bytes(blob)

        record = shipped.to_record()
        if merge is not None:
            _, merge_names = _signature(_MERGE_PARAMS)
            record.update(
                {
                    "merge_args": arg_records(merge_names, merge["signature"]),
                    "merge_vgprs": merge["vgpr_count"],
                }
            )
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
                "vgpr_spills": described.get("vgpr_spill_count", 0),
                # Added to the kernel's own compile hints, which are not repeated here.
                "llvm_options": _llvm_options(instance, arch),
                "lds_pads": _lds_pads(instance, arch),
                "named_args": described["named_args"],
            }
        )
        records.append(record)

    # Every CDNA kernel is FlyDSL's own; none comes from AITER, so its pin is not
    # part of how these bytes came to be.
    provenance = env.provenance(arch)
    provenance.pop("aiter_kernels_commit", None)
    write_op_manifest(op_dir, arch, op, provenance, records)
    source_md = refresh_source_md(op_dir)
    print(f"wrote {len(records)} object(s) to {op_dir}")
    print(f"refreshed {source_md}")
    return op_dir


def run(op: str, description: str, argv: list[str] | None = None) -> int:
    """A CDNA SDPA generator's command line: build @p op for one arch."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--arch",
        required=True,
        help=f"GPU target: one of {', '.join(sorted(SDPA_CDNA_TIERS))}; one arch per invocation.",
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
    if args.arch not in SDPA_CDNA_TIERS:
        print(
            f"error: the CDNA SDPA kernels target {', '.join(sorted(SDPA_CDNA_TIERS))}; "
            f"{args.arch!r} is neither",
            file=sys.stderr,
        )
        return 1

    try:
        generate(op, args.arch, args.out_dir, args.keep_ir)
    except (GeneratorError, env.FlydslEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
