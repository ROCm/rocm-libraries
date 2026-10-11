# Adapted from dnn-providers/hip-kernel-provider/flydsl/generators/gen_rmsnorm.py @ 03911535bd0 (origin/users/brpepers/hipdnn-flydsl-provider)
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compile the checked-in Triton RMSNorm code objects.

Run from ``triton/`` with the generator venv (triton + msgpack; see README.md)::

    cd <hip-kernel-provider>/triton
    <venv>/bin/python -m generators.gen_rmsnorm --arch gfx942 --arch gfx950

Writes ``<out>/rmsnorm/<arch>/<instance>.hsaco`` for every instance of ``_contract``,
``<out>/rmsnorm/<arch>/manifest.json`` and ``<out>/rmsnorm/<arch>.SOURCE.md``.
``<out>`` is a content root (the same meaning as ``gen_descriptors --in/--out``) and
defaults to the committed ``descriptors/Triton/``. Each arch is built in a temporary
sibling and swapped in only after every instance verified, so a failed run leaves the
previous arch folder untouched; the swap replaces the whole ``<arch>/`` folder, so
``gen_descriptors`` must run afterwards to re-emit its descriptors. Compilation needs
no GPU.

Every object is verified before it is written: ELF arch (TRITONGEN-ARCH), the
contract kernarg tuple including both scratch slots (TRITONGEN-ABI), zero scratch
(TRITONGEN-SCRATCH) and the erratum ISA pattern (TRITONGEN-ERRATUM).
"""

from __future__ import annotations

import argparse
import os
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
    erratum_record,
    new_stage,
    read_manifest,
    replace_dir,
    sha256,
    source_md_path,
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
    where = f"{arch}/{C.OP}/{C.hsaco_file(instance.name)}"
    check_scratch(
        instance.name, compiled.global_scratch_size, compiled.profile_scratch_size
    )
    described = describe(compiled.hsaco, where)
    verify_arch(described, arch, where)
    verify_abi(described, instance.name, where)
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
        "erratum_guard": erratum_record(compiled.guard_applied),
        "args": arg_records(described["signature"]),
    }


def generate(arches: list[str], out_root: Path) -> list[Path]:
    """Build ``arches`` into ``<out_root>/rmsnorm/<arch>``; returns the manifests."""
    out_dir = out_root / C.OP
    env.prepare()
    env.import_triton()
    wheel = env.wheel_sha256()
    fn = env.load_kernel()

    manifests = []
    for arch in arches:
        stage = new_stage(out_dir)
        try:
            manifests.append(_build_arch(fn, arch, wheel, out_dir, stage))
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return manifests


def _build_arch(fn, arch: str, wheel: str, out_dir: Path, stage: Path) -> Path:
    """Compile and verify every instance into ``stage``, then swap it into place."""
    arch_dir = stage / arch
    arch_dir.mkdir()
    records: list[dict] = []
    commits: set[str] = set()
    for instance in C.instances():
        compiled = compile_rmsnorm(fn, arch, instance)
        described = verify_compiled(compiled, instance, arch)

        filename = C.hsaco_file(instance.name)
        where = f"{arch}/{C.OP}/{filename}"
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
            f"num_warps={record['num_warps']} "
            f"guard={'applied' if compiled.guard_applied else 'not-applicable'}",
            flush=True,
        )
    if len(commits) != 1:
        raise GeneratorError(f"{arch}: objects carry LLVM commits {sorted(commits)}")
    toolchain = {
        "triton_version": C.TRITON_VERSION,
        "triton_wheel_sha256": wheel,
        "llvm_commit": commits.pop(),
        "env": env.env_record(),
        "hip_options": dict(C.HIP_OPTIONS_FIXED),
    }
    write_manifest(arch_dir, arch, toolchain, records)
    write_source_md(stage, read_manifest(arch_dir / C.MANIFEST_FILE))
    replace_dir(arch_dir, out_dir / arch)
    os.replace(source_md_path(stage, arch), source_md_path(out_dir, arch))
    return out_dir / arch / C.MANIFEST_FILE


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
        help=f"content root; writes <out>/{C.OP}/<arch>/ (default: {env.CONTENT_DIR})",
    )
    args = parser.parse_args(argv)
    try:
        generate(list(dict.fromkeys(args.arch)), args.out)
    except GeneratorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
