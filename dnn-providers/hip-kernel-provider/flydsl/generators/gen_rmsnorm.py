# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in RMSNorm code objects for one architecture.

    python -m generators.gen_rmsnorm --arch gfx1151

Writes ``<content>/rmsnorm/<arch>/<instance>.hsaco`` for every row of
``_instances.rmsnorm_instances()``, plus the ``manifest.json`` beside them and
``<content>/rmsnorm/<arch>.SOURCE.md`` next to the folder (``<arch>/`` is kept in
DVC, the summary in git). ``<content>`` is FlyDSL's bundle folder under the
provider's production descriptor root (``_flydsl_env.CONTENT_DIR``).

Compilation needs no matching GPU. Under ``COMPILE_ONLY=1`` the launcher traces
and lowers but never dispatches, so a gfx942 object builds on an RDNA machine --
porting an op to another architecture is then a validation step on that
hardware, not a generation one.

**Every object is verified against its own AMDGPU metadata before it is
written.** The increment-1 gate established that FlyDSL emits no argument
*names* in that note (HIP ``extern "C"`` kernels do not; the AITER ASM objects
do), so ``requireSignatureMatch`` at load time can catch a kind/size/offset
change but cannot catch an operand *permutation* -- two pointers of the same
size swapped look identical to it. The compensation is here, at the point where
we still know what we asked for: assert the full ``(kind, size, offset)``
sequence against the layout recorded below, and carry the semantic names
forward into the descriptor metadata so the permutation question is answerable
by reading the manifest. Emitting ``.name`` from FlyDSL is the real fix and is
upstream's to make; this keeps it off the critical path.
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
from ._instances import RMSNORM_SMALL_N_THRESHOLD, Instance, instances_for
from ._manifest import refresh_source_md, sha256, write_op_manifest

OP = "rmsnorm"

# The traced tensor extents. Both dimensions lower to `?` in the kernel's layout
# (`!fly.layout<(?,?):(?{i64},1)>`), so these shapes are not baked into the
# object -- they exist only to give the tracer something with a dtype and a rank.
# `N` still has to be plausible for the instance: the *builder* reads the `N`
# argument, not the tensor, but a mismatch would make a debug IR dump misleading.
_TRACE_M = 4
_GENERIC_TRACE_N = 4096

# The kernarg layout every instance of this family must have. Four (pointer,
# descriptor) pairs for `rmsnorm_kernel(Input, Gamma, Rstd, Output)`; the 16/4
# descriptor sizes are 2D tensors vs 1D ones. Recorded from the gate run and
# byte-identical across both objects it examined.
_KERNARG_SEGMENT_SIZE = 80
_EXPECTED_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("by_value", 16, 8),
    ("global_buffer", 8, 24),
    ("by_value", 4, 32),
    ("global_buffer", 8, 40),
    ("by_value", 4, 48),
    ("global_buffer", 8, 56),
    ("by_value", 16, 64),
)
# What the slots above mean, in order. `Rstd` aliases `Gamma` whenever
# `store_rstd=False` -- upstream passes Gamma twice to fill the unused slot -- so
# slots 4/5 are live kernargs pointing at the weight tensor, not at an output.
_SEMANTIC_ARG_NAMES = (
    "Input.ptr",
    "Input.desc",
    "Gamma.ptr",
    "Gamma.desc",
    "Rstd.ptr",
    "Rstd.desc",
    "Output.ptr",
    "Output.desc",
)

# The launch geometry the native dispatch handler has to reproduce. One block per
# row, `BLOCK_THREADS` wide. Recorded per instance so the handler reads it rather
# than hardcoding a formula that silently stops matching when a second family
# (e.g. upstream's large-M/small-N builder) is added.
_GRID_RULE = "one_block_per_row"


def _verify(described: dict, instance: Instance, arch: str, where: str) -> None:
    """Fail before writing, rather than shipping a mislabelled object."""
    verify_arch(described, arch, where)

    if described["signature"] != _EXPECTED_SIGNATURE:
        raise GeneratorError(
            f"{where}: kernarg layout for {instance.name!r} is\n"
            f"  {described['signature']}\nbut this family is declared as\n"
            f"  {_EXPECTED_SIGNATURE}\n"
            "Either the kernel's parameter list changed, or this instance builds a "
            "different kernel than the rest of the family. The second is the one to "
            "check first: build_rmsnorm_module dispatches N <= "
            f"{RMSNORM_SMALL_N_THRESHOLD} to a separate builder with its own ABI."
        )
    if described["kernarg_segment_size"] != _KERNARG_SEGMENT_SIZE:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, "
            f"expected {_KERNARG_SEGMENT_SIZE}"
        )


def _assert_threshold_mirror(kernel_module) -> None:
    """The instance table mirrors SMALL_N_THRESHOLD; confirm the mirror is true.

    `_instances.py` must stay importable without FlyDSL installed, so it cannot
    read the constant from the kernel source directly. This is where the two are
    compared -- a drift here silently changes *which kernel* a specialized N
    builds, which the signature check would catch, but with a far less useful
    message than naming the constant.
    """
    upstream = getattr(kernel_module, "SMALL_N_THRESHOLD", None)
    if upstream != RMSNORM_SMALL_N_THRESHOLD:
        raise GeneratorError(
            f"kernels_src SMALL_N_THRESHOLD is {upstream!r} but _instances.py "
            f"mirrors {RMSNORM_SMALL_N_THRESHOLD!r}. Update the mirror and re-check "
            "RMSNORM_SPECIALIZED_N: values at or below the threshold build the "
            "large-M/small-N kernel, which has a different ABI and grid."
        )


def _build_one(kernel_module, torch, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    import_n = instance.knobs["N"]
    trace_n = _GENERIC_TRACE_N if import_n is None else import_n
    # `f32` has no instance in the table today, but the kernel supports it and
    # this map was the only thing that raised on it (NOTES §18.6). Kept so that
    # widening `DTYPES` is a one-line change here rather than two in two files.
    torch_dtype = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
    }[instance.dtype]

    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_rmsnorm_module(
        import_n,
        instance.dtype,
        BLOCK_THREADS=instance.knobs["block_threads"],
    )

    # CPU tensors on purpose: COMPILE_ONLY never dispatches, and requiring a live
    # HIP context here would mean a machine with a working GPU is needed to build
    # objects for a machine we do not have.
    x = torch.zeros((_TRACE_M, trace_n), dtype=torch_dtype)
    gamma = torch.zeros((trace_n,), dtype=torch_dtype)
    y = torch.zeros((_TRACE_M, trace_n), dtype=torch_dtype)
    launch(x, gamma, y, _TRACE_M)

    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    env.assert_flydsl_version()
    env.prepare(arch)

    import torch  # noqa: PLC0415  (after prepare(): the env must be set first)

    from kernels.norm import rmsnorm_kernel  # noqa: PLC0415

    _assert_threshold_mirror(rmsnorm_kernel)

    op_dir = out_root / OP / arch
    op_dir.mkdir(parents=True, exist_ok=True)

    instances = instances_for(OP)
    records: list[dict] = []

    for index, instance in enumerate(instances, start=1):
        print(f"[{index}/{len(instances)}] {instance.name}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
            dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
            try:
                blob = _build_one(rmsnorm_kernel, torch, instance, dump_dir)
            except Exception as exc:
                raise GeneratorError(f"{instance.name}: {exc}") from exc

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
                # The gate's finding, recorded per object rather than asserted once:
                # if a future FlyDSL starts emitting names, this stops being 0 and
                # the packer can hand real names to requireSignatureMatch.
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
        help="GPU target, e.g. gfx1151. One arch per invocation: FlyDSL reads ARCH "
        "from the environment at each compile, so a multi-arch run could file an "
        "object under the wrong name.",
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
