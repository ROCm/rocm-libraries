# Adapted from dnn-providers/hip-kernel-provider/flydsl/generators/gen_rmsnorm.py at commit 03911535bd0 (FlyDSL provider)
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compile the checked-in Triton code objects of one op and emit their descriptors.

For each arch, in one command: compiles every instance of the op contract
(``ops/<op>.py``), verifies it (see ``_codeobject``), writes the objects,
``manifest.json`` and ``<arch>.SOURCE.md`` into a staged content root, then
``gen_descriptors.emit`` adds the arch's descriptors, validates the op tree and swaps
the arch into ``<out>/<op>/``. A failure leaves the failing arch and every later one
untouched; earlier arches are complete. Usage: README.md.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import _family as F
from . import _triton_env as env
from ._codeobject import (
    arg_records,
    check_erratum_object,
    check_scratch,
    describe,
    expected_abi,
    llvm_commit,
    verify_abi,
    verify_arch,
)
from ._family import GeneratorError
from ._manifest import (
    inputs_digest,
    new_stage,
    read_manifest,
    sha256,
    write_manifest,
    write_source_md,
)
from .ops import OPS


def compile_one(op, fn, arch: str, instance) -> env.Compiled:
    return env.compile_instance(
        fn,
        op.triton_signature(instance),
        op.triton_constexprs(instance),
        op.triton_attrs(instance),
        arch,
        instance.num_warps,
    )


def verify_compiled(op, compiled: env.Compiled, instance, arch: str) -> dict:
    """All pre-write checks on one compiled instance; returns ``describe()``'s facts."""
    where = f"{op.OP}/{arch}/{F.hsaco_file(instance.name)}"
    check_scratch(
        instance.name, compiled.global_scratch_size, compiled.profile_scratch_size
    )
    described = describe(compiled.hsaco, where)
    verify_arch(described, arch, where)
    verify_abi(described, expected_abi(op, instance), instance.name, where)
    block_threads = instance.knobs["block_threads"]
    launched = compiled.num_warps * compiled.warp_size
    object_max = described["max_flat_workgroup_size"]
    if not block_threads == launched == object_max:
        raise GeneratorError(
            f"{F.ERR_ABI}: {where}: {instance.name} block_threads="
            f"{block_threads}, num_warps*warp_size={launched}, "
            f".max_flat_workgroup_size={object_max}"
        )
    check_erratum_object(compiled.hsaco, arch, where)
    return described


def _record(op, instance, compiled, described, filename) -> dict:
    return {
        "name": instance.name,
        "dtype": instance.dtype,
        "priority": instance.priority,
        "knobs": instance.knobs,
        "file": filename,
        "symbol": described["symbol"],
        "bytes": len(compiled.hsaco),
        "sha256": sha256(compiled.hsaco),
        "dynamic_lds_bytes": compiled.dynamic_lds_bytes,
        "global_scratch_size": compiled.global_scratch_size,
        "profile_scratch_size": compiled.profile_scratch_size,
        "kernarg_segment_size": described["kernarg_segment_size"],
        "args": arg_records(expected_abi(op, instance)),
    }


def generate(op, arches: list[str], out_root: Path) -> None:
    """Build ``arches`` of ``op`` and their descriptors into ``<out_root>/<op>/``, one
    arch at a time."""
    env.prepare()
    env.import_triton()
    env.assert_options_match_wheel_defaults()
    toolchain = {
        "triton_version": F.TRITON_VERSION,
        "triton_wheel_sha256": env.wheel_sha256(),
        "libtriton_sha256": env.libtriton_sha256(),
    }
    fn = env.load_kernel(op)

    for arch in arches:
        stage = new_stage(out_root)
        try:
            _build_arch(op, fn, arch, toolchain, stage)
            _emit_descriptors(stage, out_root, op, arch)
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def _emit_descriptors(stage_root: Path, out_root: Path, op, arch: str) -> None:
    """Emit ``arch``'s descriptors from the staged content root and swap it into ``out_root``."""
    if str(F.TRITON_DIR) not in sys.path:
        sys.path.insert(0, str(F.TRITON_DIR))
    import gen_descriptors  # noqa: PLC0415

    gen_descriptors.emit(op, stage_root, out_root, [arch])


def _build_arch(op, fn, arch: str, toolchain: dict, stage_root: Path) -> None:
    """Compile and verify every instance into the content root ``stage_root``."""
    op_dir = stage_root / op.OP
    arch_dir = op_dir / arch
    arch_dir.mkdir(parents=True)
    records: list[dict] = []
    commits: set[str] = set()
    for instance in op.instances():
        compiled = compile_one(op, fn, arch, instance)
        described = verify_compiled(op, compiled, instance, arch)

        filename = F.hsaco_file(instance.name)
        where = f"{op.OP}/{arch}/{filename}"
        if compiled.name != op.KERNEL_SYMBOL:
            raise GeneratorError(
                f"{F.ERR_ABI}: {where}: compiled symbol {compiled.name!r}"
            )
        commits.add(llvm_commit(compiled.hsaco, where))
        (arch_dir / filename).write_bytes(compiled.hsaco)
        record = _record(op, instance, compiled, described, filename)
        records.append(record)
        print(
            f"TRITONGEN: {arch} {instance.name} bytes={record['bytes']} "
            f"sha256={record['sha256'][:12]} "
            f"dynamic_lds_bytes={record['dynamic_lds_bytes']} "
            f"num_warps={compiled.num_warps}",
            flush=True,
        )
    if len(commits) != 1:
        raise GeneratorError(
            f"{F.ERR_ENV}: {arch}: objects carry LLVM commits {sorted(commits)}"
        )
    full_toolchain = {
        **toolchain,
        "llvm_commit": commits.pop(),
        "hip_options": dict(F.HIP_OPTIONS_FIXED),
        "env": env.env_record(),
        "inputs_sha256": inputs_digest(op),
    }
    write_manifest(arch_dir, op.OP, arch, full_toolchain, records)
    write_source_md(op_dir, read_manifest(arch_dir / F.MANIFEST_FILE))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--op", choices=sorted(OPS), required=True, help="op to build")
    parser.add_argument(
        "--arch",
        action="append",
        choices=sorted({arch for op in OPS.values() for arch in op.ARCHES}),
        required=True,
        help="GPU target of the op; repeatable",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=F.CONTENT_DIR,
        help=(
            "content root; writes the objects and descriptors under <out>/<op>/ "
            f"(default: {F.CONTENT_DIR})"
        ),
    )
    args = parser.parse_args(argv)
    op = OPS[args.op]
    arches = list(dict.fromkeys(args.arch))
    unknown = [arch for arch in arches if arch not in op.ARCHES]
    if unknown:
        parser.error(
            f"--op {op.OP} builds {', '.join(op.ARCHES)}, not {', '.join(unknown)}"
        )
    try:
        generate(op, arches, args.out)
    except GeneratorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: {F.ERR_INTERNAL}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
