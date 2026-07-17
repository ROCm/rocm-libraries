# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CLI: build ROCKE examples locally and run them on remote slurm compute nodes.

Examples:
  python -m rocke.benchmark.remote_test.cli probe
  python -m rocke.benchmark.remote_test.cli build --arch gfx942,gfx1151
  python -m rocke.benchmark.remote_test.cli run   --arch gfx1151
  python -m rocke.benchmark.remote_test.cli all   --arch gfx942,gfx1151
"""

from __future__ import annotations

import argparse
import shlex
from typing import List

from . import build, config, persistent, slurm, transport


def _arches(spec: str) -> List[str]:
    items = [a.strip() for a in spec.split(",") if a.strip()]
    bad = [a for a in items if a not in config.ARCHES]
    if bad:
        raise SystemExit(f"unknown arch(es) {bad}; have {list(config.ARCHES)}")
    return items


def _cmd_probe(args: argparse.Namespace) -> int:
    hostname = transport.check_connectivity()
    print(f"[probe] ssh OK -> {hostname}")
    for arch in _arches(args.arch):
        p = config.ARCHES[arch]
        r = transport.ssh_run(
            f"sinfo -h -o '%N %t %f' -p {p.slurm_partition} "
            f"| awk -v c={p.slurm_constraint} '$3 ~ c {{print $1, $2}}' | head -10",
            capture=True,
        )
        print(f"[probe:{arch}] candidate nodes:\n{r.stdout.rstrip() or '  (none)'}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    for arch in _arches(args.arch):
        build.build_arch(arch, clean=not args.no_clean)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    rc = 0
    for arch in _arches(args.arch):
        rc |= slurm.run_arch(arch)
    return rc


def _cmd_all(args: argparse.Namespace) -> int:
    transport.check_connectivity()
    for arch in _arches(args.arch):
        build.build_arch(arch, clean=not args.no_clean)
    rc = 0
    for arch in _arches(args.arch):
        rc |= slurm.run_arch(arch)
    return rc


def _cmd_persist(args: argparse.Namespace) -> int:
    """Persistent mode: reserve node, run multiple tests, release."""
    transport.check_connectivity()
    arches = _arches(args.arch)
    if len(arches) > 1:
        raise SystemExit("persist mode only supports one arch at a time")
    arch = arches[0]

    # Build first
    if not args.no_build:
        build.build_arch(arch, clean=not args.no_clean)

    # Allocate and hold the node
    with persistent.PersistentAllocation(arch) as alloc:
        # Run initial test
        print(f"\n[persist:{arch}] Running initial test...")
        rc = alloc.run_test()
        if rc != 0:
            print(
                f"[persist:{arch}] Initial test FAILED (rc={rc}) - continuing anyway..."
            )

        # Enter interactive loop
        print(f"\n[persist:{arch}] Node reserved. Commands:")
        print("  r         - re-run test (uses current local build)")
        print("  b         - rebuild locally then run")
        print("  q         - release node and quit")
        print()

        while True:
            try:
                cmd = input(f"[{arch} on {alloc.nodename}]> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[persist] Releasing...")
                break

            if cmd == "q":
                break
            elif cmd == "r":
                print(f"[persist:{arch}] Re-running test...")
                rc = alloc.run_test()
                print(f"[persist:{arch}] Test exit code: {rc}")
            elif cmd == "b":
                print(f"[persist:{arch}] Rebuilding locally...")
                build.build_arch(arch, clean=False)
                print(f"[persist:{arch}] Running test...")
                rc = alloc.run_test()
                print(f"[persist:{arch}] Test exit code: {rc}")
            else:
                print(f"unknown command: {cmd!r}")

    print(f"[persist:{arch}] Released.")
    return 0


def _cmd_hold(args: argparse.Namespace) -> int:
    """Reserve a node persistently and detach (leave it running)."""
    transport.check_connectivity()
    arches = _arches(args.arch)
    if len(arches) > 1:
        raise SystemExit("hold mode only supports one arch at a time")
    arch = arches[0]
    if not args.no_build:
        extra = shlex.split(args.build_args) if args.build_args else None
        build.build_arch(arch, clean=not args.no_clean, extra_args=extra)
    alloc = persistent.PersistentAllocation(arch)
    alloc.allocate()
    alloc.save_state()
    print(
        f"[hold:{arch}] node {alloc.nodename} held by job {alloc.jobid}; "
        f"use `exec --arch {arch}` to run tests, `free --arch {arch}` to release"
    )
    return 0


def _cmd_exec(args: argparse.Namespace) -> int:
    """Run a test against an already-held node (rsync + srun --jobid)."""
    transport.check_connectivity()
    arches = _arches(args.arch)
    if len(arches) > 1:
        raise SystemExit("exec mode only supports one arch at a time")
    arch = arches[0]
    if args.rebuild:
        extra = shlex.split(args.build_args) if args.build_args else None
        build.build_arch(arch, clean=not args.no_clean, extra_args=extra)
    alloc = persistent.PersistentAllocation.reattach(arch)
    # Refresh the python tree so local code edits (e.g. diff output in
    # run_manifest) propagate to the held node.
    alloc._remote_pkg = slurm.push_rocke_tree()
    env = {"ROCKE_NBITS_DEBUG": "1"} if args.debug else None
    rc = alloc.run_test(env=env)
    print(f"[exec:{arch}] test exit code: {rc}")
    return rc


def _cmd_free(args: argparse.Namespace) -> int:
    """Release a previously held node."""
    transport.check_connectivity()
    for arch in _arches(args.arch):
        try:
            alloc = persistent.PersistentAllocation.reattach(arch)
        except RuntimeError as e:
            print(f"[free:{arch}] {e}")
            persistent.PersistentAllocation(arch).clear_state()
            continue
        alloc.release()
        alloc.clear_state()
        print(f"[free:{arch}] released and cleared session")
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rocke.benchmark.remote_test")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("probe", _cmd_probe),
        ("build", _cmd_build),
        ("run", _cmd_run),
        ("all", _cmd_all),
        ("persist", _cmd_persist),
        ("hold", _cmd_hold),
        ("exec", _cmd_exec),
        ("free", _cmd_free),
    ):
        sp = sub.add_parser(name)
        sp.add_argument(
            "--arch",
            default=",".join(config.ARCHES),
            help="comma-separated archs (default: all configured)",
        )
        sp.add_argument(
            "--no-clean", action="store_true", help="keep prior stage dir contents"
        )
        if name in ("persist", "hold"):
            sp.add_argument(
                "--no-build",
                action="store_true",
                help="skip initial local build (reuse prior artifacts)",
            )
        if name in ("hold", "exec"):
            sp.add_argument(
                "--build-args",
                default="",
                help="extra args forwarded to the example build "
                "(shlex-split), e.g. '--m 64 --n 128 --k 128'",
            )
        if name == "exec":
            sp.add_argument(
                "--rebuild", action="store_true", help="rebuild locally before running"
            )
            sp.add_argument(
                "--debug",
                action="store_true",
                help="set ROCKE_NBITS_DEBUG=1 on the remote run "
                "for matmul_nbits mismatch diagnostics",
            )
        sp.set_defaults(_fn=fn)
    args = p.parse_args(argv)
    return args._fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
