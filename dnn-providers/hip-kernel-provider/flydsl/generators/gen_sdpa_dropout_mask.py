# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the SDPA dropout keep-mask writer (the graph's RNG dump) for one target.

    python -m generators.gen_sdpa_dropout_mask --arch gfx942

Writes ``<content>/sdpa_dropout_mask/<arch>/<instance>.hsaco`` for the one row of
``_instances.sdpa_dropout_mask_instances()``, its ``manifest.json`` and
``<content>/sdpa_dropout_mask/<arch>.SOURCE.md``. One object per target: gfx11-generic (the RDNA
prefill engine drops there), gfx12-generic (aiw), gfx90a, gfx942 and gfx950 (CDNA).

The kernel (``kernels_src/kernels/dropout_mask/sdpa_dropout_mask.py``) draws the attention
kernels' counters through the same ``attention_rdna4/philox.py`` and writes the keep mask as
FLOAT. It is not an attention axis: the dropping engines' natives launch it after the attention
launch, and only for a graph with an RNG dump, so no attention object changes. Its engine
(``hipkernel:flydsl_sdpa_dropout_mask``) serves no graph; the descriptors exist so the packer
packs the object and the natives can find it.
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

OP = "sdpa_dropout_mask"

# The targets whose attention objects drop, and so the ones a dump is written for.
TARGETS = ("gfx11-generic", "gfx12-generic", "gfx90a", "gfx942", "gfx950")

_SEMANTIC_ARG_NAMES = (
    "MASK",
    "SEED",
    "OFFSET1",
    "offset2",
    "threshold",
    "num_planes",
    "seq_len_q",
    "seq_len_kv",
)
_KERNARG_SEGMENT_SIZE = 48
_EXPECTED_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("global_buffer", 8, 8),
    ("global_buffer", 8, 16),
    ("by_value", 8, 24),
    ("by_value", 4, 32),
    ("by_value", 4, 36),
    ("by_value", 4, 40),
    ("by_value", 4, 44),
)

# One thread per Philox counter (four keys), `block_threads` to a workgroup:
# cdiv(num_planes * seq_len_q * cdiv(seq_len_kv, 4), block_threads) workgroups.
_GRID_RULE = "counters_over_block_threads"


def _verify(described: dict, instance: Instance, arch: str, where: str) -> None:
    """Fail before writing an object whose layout is not the one the natives launch."""
    verify_arch(described, arch, where)
    if described["signature"] != _EXPECTED_SIGNATURE:
        raise GeneratorError(
            f"{where}: kernarg layout for {instance.name!r} is\n"
            f"  {described['signature']}\nbut is declared as\n  {_EXPECTED_SIGNATURE}\n"
            "Update the layout, the semantic names and the natives' mask launch together."
        )
    if described["kernarg_segment_size"] != _KERNARG_SEGMENT_SIZE:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, "
            f"expected {_KERNARG_SEGMENT_SIZE}"
        )
    if described.get("vgpr_spill_count"):
        raise GeneratorError(f"{where}: spills {described['vgpr_spill_count']} VGPRs")


def _build_one(kernel_module, instance: Instance, dump_dir: Path) -> bytes:
    import flydsl.compiler as flyc  # noqa: PLC0415  (after prepare())
    import flydsl.expr as fx  # noqa: PLC0415

    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_sdpa_dropout_mask_module(
        block_threads=instance.knobs["block_threads"]
    )
    if launch.block_size != instance.knobs["block_threads"]:
        raise GeneratorError(
            f"{instance.name}: kernel built with {launch.block_size} threads, "
            f"the instance declares {instance.knobs['block_threads']}"
        )
    # Placeholders only: every value is a runtime argument, and COMPILE_ONLY never
    # dispatches through the null pointers.
    null = flyc.from_c_void_p(fx.Uint8, 0)
    flyc.compile(launch, null, null, null, 0, 0, 1, 1, 1, fx.Stream(None))
    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def _compile(task: tuple) -> bytes:
    """One instance's code object, in a ``_parallel`` worker or this process."""
    from kernels.dropout_mask import sdpa_dropout_mask  # noqa: PLC0415

    instance, keep_ir = task
    with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
        dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
        try:
            return _build_one(sdpa_dropout_mask, instance, dump_dir)
        except Exception as exc:
            raise GeneratorError(f"{instance.name}: {exc}") from exc


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    env.assert_flydsl_version()
    env.prepare(arch)

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
        choices=TARGETS,
        help="one target per invocation: FlyDSL reads ARCH from the environment at each "
        "compile, so a multi-target run could file an object under the wrong name",
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
        help="keep the IR dump under this directory (debugging)",
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
