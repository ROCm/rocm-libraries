# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in causal conv1d code objects for one architecture.

    python -m generators.gen_causal_conv1d --arch gfx942

Writes ``<content>/causal_conv1d/<arch>/<instance>.hsaco`` for every row of
``_instances.causal_conv1d_instances()``, the ``manifest.json`` beside them and
``<content>/causal_conv1d/<arch>.SOURCE.md`` next to the folder, as
``gen_rmsnorm.py`` does for its op.

The kernel takes raw 64-bit addresses and scalars only (no tensor descriptor
structs), so the kernarg layout below is the same for every object and the
native dispatch fills it directly. Every object is checked against it before it
is written: FlyDSL emits no argument names, so the ``(kind, size, offset)``
sequence is the only check that the object takes what the dispatch passes.
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
from ._parallel import compile_all

OP = "causal_conv1d"

# The arches this op is built for. The kernel uses no gfx94x-only instruction (global
# loads and stores, one scalar buffer load per sequence offset), so gfx90a is built
# beside gfx942 and gfx950. RDNA is not built: no hardware to verify it on.
TARGETS = ("gfx90a", "gfx942", "gfx950")

# `causal_conv1d_kernel`: eight raw addresses and eleven int64 element strides, then
# seven int32 scalars.
_POINTERS = (
    "x",
    "weight",
    "bias",
    "initial_state",
    "final_state",
    "cu_seqlens",
    "state_indices",
    "y",
)
_STRIDES = (
    "x_stride",
    "y_stride",
    "weight_stride_c",
    "weight_stride_k",
    "bias_stride",
    "initial_stride_n",
    "initial_stride_c",
    "initial_stride_k",
    "final_stride_n",
    "final_stride_c",
    "final_stride_k",
)
_SCALARS = (
    "tokens",
    "seqs",
    "channels",
    "state_rows",
    "cu_seqlens_step",
    "state_indices_step",
    "flags",
)
_SEMANTIC_ARG_NAMES = tuple(f"{p}.ptr" for p in _POINTERS) + _STRIDES + _SCALARS
_EXPECTED_SIGNATURE = tuple(
    ("by_value", 8, 8 * i) for i in range(len(_POINTERS) + len(_STRIDES))
) + tuple(
    ("by_value", 4, 8 * (len(_POINTERS) + len(_STRIDES)) + 4 * i)
    for i in range(len(_SCALARS))
)
_KERNARG_SEGMENT_SIZE = 8 * (len(_POINTERS) + len(_STRIDES)) + 4 * len(_SCALARS)

# Grid (N + ceil(T / tile_tokens), ceil(C / (block_threads * lanes)), 1); block
# (block_threads, 1, 1). Grid x is the sequence blocks, then the token tiles.
_GRID_RULE = "sequences_plus_token_tiles"


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
            f"{where}: {described['vgpr_spill_count']} VGPR spill(s); the schedule "
            "does not fit the register file"
        )


def _assert_kernel_mirror(kernel_module) -> None:
    """The flag bits are restated in the native dispatch; the kernel must still say them."""
    flags = (
        kernel_module.FLAG_BIAS,
        kernel_module.FLAG_INITIAL_STATE,
        kernel_module.FLAG_FINAL_STATE,
        kernel_module.FLAG_STATE_INDICES,
    )
    if flags != (1, 2, 4, 8):
        raise GeneratorError(
            f"kernels_src flag bits are {flags}, but FlydslCausalConv1dNative.cpp "
            "passes bias=1, initial=2, final=4, indices=8; update both together"
        )


def _build_one(kernel_module, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    knobs = instance.knobs
    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_causal_conv1d_module(
        knobs["width"],
        instance.dtype,
        knobs["activation"] == "silu",
        knobs["lanes"],
        knobs["block_threads"],
        knobs["tile_tokens"],
    )
    # COMPILE_ONLY traces and compiles without dispatching: every argument is a
    # scalar, so placeholders stand in for the addresses.
    launch(
        *([0] * len(_POINTERS)),
        *([1] * len(_STRIDES)),
        *([1] * len(_SCALARS)),
        1,
        1,
    )
    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    from kernels.causal_conv1d import causal_conv1d_kernel  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(causal_conv1d_kernel, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    if arch not in TARGETS:
        raise GeneratorError(f"{OP} is built for {', '.join(TARGETS)}, not {arch}")
    env.assert_flydsl_version()
    env.prepare(arch)

    from kernels.causal_conv1d import causal_conv1d_kernel  # noqa: PLC0415

    _assert_kernel_mirror(causal_conv1d_kernel)

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
        help="GPU target: one of " + ", ".join(TARGETS) + ". One arch per invocation.",
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
