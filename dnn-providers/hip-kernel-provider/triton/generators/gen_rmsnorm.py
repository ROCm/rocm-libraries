# Adapted from dnn-providers/hip-kernel-provider/flydsl/generators/gen_rmsnorm.py at commit 03911535bd0 (FlyDSL provider)
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compile the checked-in Triton RMSNorm code objects and emit their descriptors.

For each arch, in one command: compiles every ``_contract`` instance, verifies it (see
``_codeobject``), writes the objects, ``manifest.json`` and ``<arch>.SOURCE.md`` into a
staged content root, then ``gen_descriptors.emit`` adds the arch's descriptors,
validates the op tree and swaps the arch into ``<out>/rmsnorm/``. A failure leaves the
failing arch and every later one untouched; earlier arches are complete. Usage:
README.md.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import _contract as C
from . import _triton_env as env
from ._codeobject import (
    GeneratorError,
    arg_records,
    check_erratum_object,
    check_scratch,
    describe,
    llvm_commit,
    verify_abi,
    verify_arch,
)
from ._manifest import (
    new_stage,
    read_manifest,
    sha256,
    write_manifest,
    write_source_md,
)


def compile_rmsnorm(fn, arch: str, instance: C.Instance) -> env.Compiled:
    return env.compile_instance(
        fn,
        C.triton_signature(instance.dtype),
        C.triton_constexprs(instance.block_n),
        C.triton_attrs(),
        arch,
        instance.num_warps,
    )


def verify_compiled(compiled: env.Compiled, instance: C.Instance, arch: str) -> dict:
    """All pre-write checks on one compiled instance; returns ``describe()``'s facts."""
    where = f"{C.OP}/{arch}/{C.hsaco_file(instance.name)}"
    check_scratch(
        instance.name, compiled.global_scratch_size, compiled.profile_scratch_size
    )
    described = describe(compiled.hsaco, where)
    verify_arch(described, arch, where)
    verify_abi(described, instance.name, where)
    launched = compiled.num_warps * compiled.warp_size
    object_max = described["max_flat_workgroup_size"]
    if not instance.block_threads == launched == object_max:
        raise GeneratorError(
            f"{C.ERR_ABI}: {where}: {instance.name} block_threads="
            f"{instance.block_threads}, num_warps*warp_size={launched}, "
            f".max_flat_workgroup_size={object_max}"
        )
    check_erratum_object(compiled.hsaco, arch, where)
    return described


def _record(instance, compiled, described, filename) -> dict:
    return {
        "name": instance.name,
        "dtype": instance.dtype,
        "block_n": instance.block_n,
        "priority": instance.priority,
        "file": filename,
        "symbol": described["symbol"],
        "bytes": len(compiled.hsaco),
        "sha256": sha256(compiled.hsaco),
        "num_warps": compiled.num_warps,
        "warp_size": compiled.warp_size,
        "block": [instance.block_threads, 1, 1],
        "grid_rule": C.MANIFEST_GRID_RULE,
        "shared": compiled.shared,
        "global_scratch_size": compiled.global_scratch_size,
        "profile_scratch_size": compiled.profile_scratch_size,
        "kernarg_segment_size": described["kernarg_segment_size"],
        "args": arg_records(described["signature"]),
    }


def generate(arches: list[str], out_root: Path) -> None:
    """Build ``arches`` and their descriptors into ``<out_root>/rmsnorm/``, one arch at a time."""
    env.prepare()
    env.import_triton()
    env.assert_options_match_wheel_defaults()
    wheel = env.wheel_sha256()
    fn = env.load_kernel()

    for arch in arches:
        stage = new_stage(out_root)
        try:
            _build_arch(fn, arch, wheel, stage)
            _emit_descriptors(stage, out_root, arch)
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def _emit_descriptors(stage_root: Path, out_root: Path, arch: str) -> None:
    """Emit ``arch``'s descriptors from the staged content root and swap it into ``out_root``."""
    if str(env.TRITON_DIR) not in sys.path:
        sys.path.insert(0, str(env.TRITON_DIR))
    import gen_descriptors  # noqa: PLC0415

    gen_descriptors.emit(stage_root, out_root, [arch])


def _build_arch(fn, arch: str, wheel: str, stage_root: Path) -> None:
    """Compile and verify every instance into the content root ``stage_root``."""
    op_dir = stage_root / C.OP
    arch_dir = op_dir / arch
    arch_dir.mkdir(parents=True)
    records: list[dict] = []
    commits: set[str] = set()
    for instance in C.instances():
        compiled = compile_rmsnorm(fn, arch, instance)
        described = verify_compiled(compiled, instance, arch)

        filename = C.hsaco_file(instance.name)
        where = f"{C.OP}/{arch}/{filename}"
        if compiled.name != C.KERNEL_SYMBOL:
            raise GeneratorError(
                f"{C.ERR_ABI}: {where}: compiled symbol {compiled.name!r}"
            )
        commits.add(llvm_commit(compiled.hsaco, where))
        (arch_dir / filename).write_bytes(compiled.hsaco)
        record = _record(instance, compiled, described, filename)
        records.append(record)
        print(
            f"TRITONGEN: {arch} {instance.name} bytes={record['bytes']} "
            f"sha256={record['sha256'][:12]} shared={record['shared']} "
            f"num_warps={record['num_warps']}",
            flush=True,
        )
    if len(commits) != 1:
        raise GeneratorError(
            f"{C.ERR_ENV}: {arch}: objects carry LLVM commits {sorted(commits)}"
        )
    toolchain = {
        "triton_version": C.TRITON_VERSION,
        "triton_wheel_sha256": wheel,
        "llvm_commit": commits.pop(),
        "env": env.env_record(),
        "hip_options": dict(C.HIP_OPTIONS_FIXED),
    }
    write_manifest(arch_dir, arch, toolchain, records)
    write_source_md(op_dir, read_manifest(arch_dir / C.MANIFEST_FILE))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arch",
        action="append",
        choices=C.ARCHES,
        required=True,
        help="GPU target; repeatable",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=env.CONTENT_DIR,
        help=(
            f"content root; writes the objects and descriptors under <out>/{C.OP}/ "
            f"(default: {env.CONTENT_DIR})"
        ),
    )
    args = parser.parse_args(argv)
    try:
        generate(list(dict.fromkeys(args.arch)), args.out)
    except GeneratorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: {C.ERR_INTERNAL}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
