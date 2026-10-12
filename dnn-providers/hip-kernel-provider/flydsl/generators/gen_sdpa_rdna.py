# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in RDNA SDPA prefill code objects for one architecture.

    python -m generators.gen_sdpa_rdna --arch gfx1151

Writes ``<content>/sdpa_rdna/<arch>/<instance>.hsaco`` for every row of
``_instances.sdpa_instances()``, plus the ``manifest.json`` beside them and
``<content>/sdpa_rdna/<arch>.SOURCE.md`` next to the folder (``<arch>/`` is kept in
DVC, the summary in git). ``<content>`` is FlyDSL's bundle folder under the provider's
production descriptor root (``_flydsl_env.CONTENT_DIR``). The decode kernel is another
engine's, built by ``gen_sdpa_rdna_decode``.

The kernel is the vendored gfx11 flash-attention forward
(``kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py``). Its launcher
takes raw pointers, so nothing is traced from a tensor: every argument is a
placeholder, and ``COMPILE_ONLY=1`` lowers without dispatching. No GPU is
needed to build an object.

As with RMSNorm, every object's kernarg layout is asserted against the layout
recorded below before it is written, and the semantic argument names are
carried into the manifest. FlyDSL emits no argument names, so the loader's
signature check can see a change of kind, size or offset but not a permutation
of two same-sized arguments -- and this ABI has many of those (six pointers,
ten i32, nineteen i64 strides). The names here are what make a permutation
answerable by reading the manifest.
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
from ._sdpa_rdna_build import LLVM_OPTIONS, WAVE_SIZE

OP = "sdpa_rdna"

# The kernel's parameter list, in order. Pointers, then the i32 problem scalars,
# the f32 scale, and the (batch, seq, head) element strides of Q, K, V, O and LSE.
_SEMANTIC_ARG_NAMES = (
    "Q",
    "K",
    "V",
    "O",
    "LSE",
    "seq_len_q",
    "seq_len_kv",
    "num_heads",
    "kv_group",
    "right_bound",
    "left_bound",
    "align_bottom_right",
    "lse_on",
    "scale",
    "q_stride_b",
    "q_stride_s",
    "q_stride_h",
    "k_stride_b",
    "k_stride_s",
    "k_stride_h",
    "v_stride_b",
    "v_stride_s",
    "v_stride_h",
    "o_stride_b",
    "o_stride_s",
    "o_stride_h",
    "lse_stride_b",
    "lse_stride_s",
    "lse_stride_h",
    "BIAS",
    "bias_stride_b",
    "bias_stride_h",
    "bias_stride_q",
    "bias_stride_k",
    "head_dim_rt",
    "v_group",
    "SINK",
    "sink_stride",
)

_KERNARG_SEGMENT_SIZE = 264
_EXPECTED_SIGNATURE = (
    *(("global_buffer", 8, 8 * i) for i in range(5)),
    *(("by_value", 4, 40 + 4 * i) for i in range(9)),
    *(("by_value", 8, 80 + 8 * i) for i in range(15)),
    ("global_buffer", 8, 200),
    *(("by_value", 8, 208 + 8 * i) for i in range(4)),
    ("by_value", 4, 240),
    ("by_value", 4, 244),
    ("global_buffer", 8, 248),
    ("by_value", 8, 256),
)

# Launch geometry the native dispatch handler reproduces: one workgroup per
# (batch, query tile of block_m rows, query head, output-column tile of
# head_dim / dv_split), `block_m / 16` waves of 32 -- twice that for an hd_pair
# object, whose waves share each 16 rows in pairs.
_GRID_RULE = "batch_x_qtiles_x_heads_x_dv_split"
_WAVE_ROWS = 16


def _prefill_block_threads(knobs: dict) -> int:
    return knobs["block_m"] // _WAVE_ROWS * WAVE_SIZE * (2 if knobs["hd_pair"] else 1)


def _verify(described: dict, instance: Instance, arch: str, where: str) -> None:
    """Fail before writing, rather than shipping a mislabelled object."""
    verify_arch(described, arch, where)

    if described["signature"] != _EXPECTED_SIGNATURE:
        raise GeneratorError(
            f"{where}: kernarg layout for {instance.name!r} is\n"
            f"  {described['signature']}\nbut this family is declared as\n"
            f"  {_EXPECTED_SIGNATURE}\n"
            "The kernel's parameter list changed; update the layout, the semantic "
            "names, and the native dispatch together."
        )
    if described["kernarg_segment_size"] != _KERNARG_SEGMENT_SIZE:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, "
            f"expected {_KERNARG_SEGMENT_SIZE}"
        )
    # Register spills would not be wrong, but they would be a silent performance
    # cliff on exactly the shapes this object exists to serve.
    if described.get("vgpr_spill_count"):
        raise GeneratorError(
            f"{where}: {described['vgpr_spill_count']} VGPRs spilled; the tile "
            "does not fit this arch's register file at these knobs"
        )


def _build_one(
    kernel_module, instance: Instance, dump_dir: Path, llvm_options: dict
) -> tuple[bytes, dict]:
    """Compile one instance; return its code object and the LLVM options used."""
    import flydsl.compiler as flyc  # noqa: PLC0415  (after prepare())
    import flydsl.expr as fx  # noqa: PLC0415
    from flydsl.compiler.kernel_function import CompilationContext  # noqa: PLC0415

    knobs = instance.knobs
    env.set_dump_dir(dump_dir)
    generic = knobs["head_dim"] is None
    launch = kernel_module.build_flash_attn_func_module(
        knobs["head_dim_max"],
        causal=bool(knobs["causal"]),
        dtype_str=instance.dtype,
        block_m=knobs["block_m"],
        block_n=knobs["block_n"],
        dv_split=knobs["dv_split"],
        has_bias=bool(knobs["has_bias"]),
        bias_query_dtype=bool(knobs["bias_q"]),
        generic_head_dim=generic,
        hd_pair=bool(knobs["hd_pair"]),
        head_dim_v=None if generic else knobs["head_dim_v"],
        has_sink=bool(knobs["has_sink"]),
    )
    if launch.dv_split != knobs["dv_split"]:
        raise GeneratorError(
            f"{instance.name}: kernel built with dv_split {launch.dv_split}, "
            f"instance declares {knobs['dv_split']}"
        )
    if launch.block_size != _prefill_block_threads(knobs):
        raise GeneratorError(
            f"{instance.name}: kernel built with {launch.block_size} threads, "
            f"the dispatcher launches {_prefill_block_threads(knobs)}"
        )

    # Placeholders only: every value below is a runtime kernel argument, so none
    # is baked, and COMPILE_ONLY never dispatches through the null pointers.
    null = flyc.from_c_void_p(fx.Uint8, 0)
    # batch, seq_len_q, seq_len_kv, num_heads, kv_group, right_bound, left_bound,
    # align_bottom_right, lse_on; then the scale.
    args = [null] * 5 + [1, 1, 1, 1, 1, 0, -1, 0, 0, 1.0]
    args += [1] * 15
    # The bias pointer and its (batch, head, query, key) strides.
    args += [null] + [1] * 4
    # The runtime head_dim (read only by a generic object), then the V group.
    args += [1, 1]
    # The sink pointer and its head stride (read only by a sink object).
    args += [null, 0]
    args.append(fx.Stream(None))
    # FlyDSL merges hint layers shallowly: an `llvm_options` passed here would
    # *replace* the kernel's own, not extend it. Merge onto them explicitly.
    effective = dict(getattr(launch, "compile_hints", {}).get("llvm_options") or {})
    effective.update(llvm_options)
    with CompilationContext.compile_hints({"llvm_options": effective}):
        flyc.compile(launch, *args)

    blob, _ = hsaco_from_dump(dump_dir)
    return blob, effective


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    env.assert_flydsl_version()
    env.prepare(arch)

    from kernels.attention_rdna import flash_attn_func_rdna  # noqa: PLC0415

    op_dir = out_root / OP / arch
    op_dir.mkdir(parents=True, exist_ok=True)

    instances = instances_for(OP)
    llvm_options = LLVM_OPTIONS.get(arch, {})
    records: list[dict] = []

    for index, instance in enumerate(instances, start=1):
        print(f"[{index}/{len(instances)}] {instance.name}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
            dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
            try:
                blob, effective_options = _build_one(
                    flash_attn_func_rdna, instance, dump_dir, llvm_options
                )
            except Exception as exc:
                raise GeneratorError(f"{instance.name}: {exc}") from exc

        filename = f"{instance.name}.hsaco"
        where = f"{arch}/{OP}/{filename}"
        record = instance.to_record()
        described = describe(blob, where)
        _verify(described, instance, arch, where)
        record.update(
            {
                "block": [_prefill_block_threads(instance.knobs), 1, 1],
                "grid_rule": _GRID_RULE,
                "args": arg_records(_SEMANTIC_ARG_NAMES, described["signature"]),
            }
        )
        verify_generic(blob, arch, where)

        (op_dir / filename).write_bytes(blob)

        record.update(
            {
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
