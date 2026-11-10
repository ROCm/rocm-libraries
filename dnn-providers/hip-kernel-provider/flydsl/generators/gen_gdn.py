# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in Gated DeltaNet code objects for one architecture.

    python -m generators.gen_gdn --arch gfx942

Writes ``<content>/gdn/<arch>/<instance>.hsaco`` for every row of
``_instances.gdn_instances()``, plus the ``manifest.json`` beside them and
``<content>/gdn/<arch>.SOURCE.md`` next to the folder.

The kernel (``kernels_src/kernels/gdn/gdn_recurrent_kernel.py``, ported from AITER's
``gdr_decode.py``) takes twelve raw pointers, then int32 counts, strides, flags and
operand dtypes, then the f32 scale, so it is traced from null pointers and placeholder
values: every shape, stride and flag is a runtime argument and nothing the trace sees is
baked into the object. Built for the CDNA targets (gfx90a, gfx942, gfx950): its K
reductions are wave64 shuffles, and it uses only buffer loads and stores, which every
one of them has.

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
from ._instances import GDN_BLOCK_THREADS, GDN_V_TILE, Instance, instances_for
from ._manifest import refresh_source_md, sha256, write_op_manifest
from ._parallel import compile_all

OP = "gdn"

# The arches this op is built for: CDNA, whose wave64 the kernel's K reductions assume.
TARGETS = ("gfx90a", "gfx942", "gfx950")

# `gdn_recurrent_kernel(Q, K, V, G, Beta, CuSeqlens, ALog, DtBias, StateIndices,
# InitialState, FinalState, O, <29 int32s>, scale)`: twelve raw pointers, the int32s
# the kernel module lists in INT_PARAMS, then the f32 scale.
_POINTERS = (
    "Q",
    "K",
    "V",
    "G",
    "Beta",
    "CuSeqlens",
    "ALog",
    "DtBias",
    "StateIndices",
    "InitialState",
    "FinalState",
    "O",
)
_INTS = (
    "num_seqs",
    "num_heads_out",
    "q_group",
    "k_group",
    "v_group",
    "stride_q_t",
    "stride_q_h",
    "stride_k_t",
    "stride_k_h",
    "stride_v_t",
    "stride_v_h",
    "stride_o_t",
    "stride_o_h",
    "stride_g_t",
    "stride_g_h",
    "stride_beta_t",
    "stride_beta_h",
    "stride_a_log",
    "stride_dt_bias",
    "stride_h0_n",
    "stride_h0_h",
    "stride_h0_v",
    "stride_h0_k",
    "stride_ht_n",
    "stride_ht_h",
    "stride_ht_v",
    "stride_ht_k",
    "flags",
    "aux_dtypes",
)
_EXPECTED_SIGNATURE = tuple(
    ("global_buffer", 8, 8 * i) for i in range(len(_POINTERS))
) + tuple(("by_value", 4, 8 * len(_POINTERS) + 4 * i) for i in range(len(_INTS) + 1))
_KERNARG_SEGMENT_SIZE = 8 * len(_POINTERS) + 4 * (len(_INTS) + 1)
_SEMANTIC_ARG_NAMES = tuple(f"{name}.ptr" for name in _POINTERS) + _INTS + ("scale",)

# Grid (num_seqs * dv_split, num_heads_out, 1); block (block_threads, 1, 1).
_GRID_RULE = "gdn_sequence_head_vslice"


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
    if described["vgpr_spill_count"]:
        raise GeneratorError(
            f"{where}: {instance.name!r} spills {described['vgpr_spill_count']} VGPRs"
        )


def _assert_mirrors(kernel_module) -> None:
    """The generator's argument names and the table's constants mirror the kernel's."""
    if kernel_module.POINTER_PARAMS != _POINTERS or kernel_module.INT_PARAMS != _INTS:
        raise GeneratorError(
            "gdn_recurrent_kernel's parameter list no longer matches gen_gdn's "
            "_POINTERS / _INTS; update both, and the native dispatch with them"
        )
    if kernel_module.BLOCK_THREADS != GDN_BLOCK_THREADS:
        raise GeneratorError(
            f"the kernel is built for {kernel_module.BLOCK_THREADS} threads, the "
            f"instance table's GDN_BLOCK_THREADS is {GDN_BLOCK_THREADS}"
        )
    tile_v = kernel_module.NUM_WARPS * kernel_module.WARP_THREADS_V
    if tile_v != GDN_V_TILE:
        raise GeneratorError(
            f"the kernel's V slice is {tile_v} rows, the instance table's "
            f"GDN_V_TILE is {GDN_V_TILE}"
        )


def _build_one(kernel_module, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    import flydsl.compiler as flyc  # noqa: PLC0415  (after prepare())
    import flydsl.expr as fx  # noqa: PLC0415

    knobs = instance.knobs
    head_v = knobs["head_dim_v"]
    if knobs["dv_split"] * GDN_V_TILE != head_v:
        raise GeneratorError(
            f"{instance.name}: dv_split {knobs['dv_split']} does not give "
            f"{GDN_V_TILE}-row slices of V {head_v}"
        )

    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_gdn_recurrent_module(
        instance.dtype,
        knobs["head_dim"],
        head_v,
        knobs["dv_split"],
        state_k_contiguous=not knobs["state_strided"],
    )
    # Placeholders only: COMPILE_ONLY never dispatches through the null pointers, and
    # every count, stride and flag is a runtime argument.
    null = flyc.from_c_void_p(fx.Uint8, 0)
    flyc.compile(
        launch,
        *([null] * len(_POINTERS)),
        *([1] * len(_INTS)),
        1.0,
        fx.Stream(None),
    )

    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    from kernels.gdn import gdn_recurrent_kernel  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(gdn_recurrent_kernel, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    if arch not in TARGETS:
        raise GeneratorError(f"gdn is built for {', '.join(TARGETS)}; not for {arch!r}")
    env.assert_flydsl_version()
    env.prepare(arch)

    from kernels.gdn import gdn_recurrent_kernel  # noqa: PLC0415

    _assert_mirrors(gdn_recurrent_kernel)

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
