#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# hsaco_parity.py -- carry the concrete parity matrix through to object code.
#
# parity_matrix asserts byte-identical .ll between the Python lowerer and the
# two C++ backend paths. Agreement on IR is not the same as agreement on the
# artifact that ships, and it is not even evidence that the IR is compilable.
# This driver compiles every kernel to HSACO and compares bytes.
#
# Three checks per kernel:
#   det     python .ll -> comgr twice          is comgr deterministic? (control)
#   eng     ir_export -> C import -> C lower   HSACO vs python
#   recipe  record -> CBOR -> C recipe VM      HSACO vs python
#
# The `det` control matters: without it, "eng == py" and "recipe == py" could
# both hold trivially if comgr were collapsing distinct inputs.
#
# Two hazards force the process model here, both observed on the current kernel
# set (see dsl_docs/architecture/portable_ir_production_readiness.md):
#
#   * LLVM reports fatal errors via abort(). An unsupported intrinsic -- e.g. a
#     gfx950-only MFMA asked to codegen for gfx942 -- kills the process, so a
#     single-process sweep loses every result after the first bad kernel.
#   * `moe_fused_mega_fp8` grows without bound in comgr (~1.5 TB observed before
#     capping) from a modest 97 KiB of .ll.
#
# So each kernel is compiled in a forked child under an RLIMIT_AS cap: one bad
# kernel is reported and stepped over instead of ending the run or the host.
#
#   python3 -m rocke.portable_ir.drivers.hsaco_parity [--arch gfx950] [--cap-gb 48]
#
# Needs a shared librocke (ROCKE_ONLINE_LIB) and comgr. No device required.

from __future__ import annotations

import argparse
import hashlib
import os
import resource
import sys
import time
from typing import Callable, Dict, List, Tuple


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _in_child(fn: Callable[[], str], cap_gb: int) -> str:
    """Run fn() in a forked child under a memory cap; return its report line.

    Returns the child's message, or a CRASH/MEMCAP description if it died. The
    cap is applied in the child so the parent survives to report."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (cap_gb * 1024**3, cap_gb * 1024**3))
            msg = fn()
        except MemoryError:
            msg = f"MEMCAP exceeded {cap_gb}G"
        except Exception as e:  # noqa: BLE001 - the reason is the result
            msg = f"EXC {type(e).__name__}: {e}"[:200]
        try:
            os.write(w, msg.encode()[:4000])
        finally:
            os._exit(0)
    os.close(w)
    buf = b""
    while True:
        chunk = os.read(r, 4096)
        if not chunk:
            break
        buf += chunk
    os.close(r)
    _, status = os.waitpid(pid, 0)
    if buf:
        return buf.decode()
    if os.WIFSIGNALED(status):
        return f"CRASH signal {os.WTERMSIG(status)} (LLVM fatal error / OOM)"
    return f"CRASH exit {os.WEXITSTATUS(status)}"


def run(arch: str, flavor: str, cap_gb: int, verbose: bool) -> Tuple[Dict, List]:
    from rocke.core import ir_export
    from rocke.core.arch import ArchTarget
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.portable_ir.drivers import parity_matrix as pm
    from rocke.portable_ir.src import online, recipe_bundle
    from rocke.portable_ir.src.recording_builder import record_kernel
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    isa = ArchTarget.from_gfx(arch).isa_triple

    def hsaco(ll: str) -> bytes:
        h, _ = build_hsaco_from_llvm_ir(ll, isa=isa, options=["-O3"])
        return h

    n = dict(cmp=0, det=0, eng=0, rec=0, uncompilable=0, refused=0)
    bad: List[Tuple[str, str, str]] = []
    notes: List[Tuple[str, str]] = []

    for label, thunk, _why in pm._kernels():
        if thunk is None:
            continue
        t0 = time.perf_counter()

        def work(thunk=thunk) -> str:
            k = thunk()
            py_ll = lower_kernel_to_llvm(k, llvm_flavor=flavor, arch=arch)
            py_h = hsaco(py_ll)
            det = hsaco(py_ll) == py_h
            eng_ll, _ = online.ir_json_to_llvm(
                ir_export.export_kernel_ir_json(k), arch=arch
            )
            eng = hsaco(eng_ll) == py_h
            _, recipe = record_kernel(thunk)
            vm_ll, _ = online.recipe_cbor_to_llvm(
                recipe_bundle.cbor_encode(recipe), arch=arch
            )
            rec = hsaco(vm_ll) == py_h
            return f"OK {int(det)}{int(eng)}{int(rec)} {len(py_h)} {_sha(py_h)}"

        out = _in_child(work, cap_gb)
        dt = time.perf_counter() - t0

        if not out.startswith("OK "):
            # A kernel the Python lowerer declines (wrong arch family) is
            # correct behavior; one that crashes LLVM is a defect in the kernel.
            if out.startswith("EXC NotImplementedError"):
                n["refused"] += 1
                kind = "refused"
            else:
                n["uncompilable"] += 1
                kind = "UNCOMPILABLE"
            notes.append((label, out[:110]))
            if verbose or kind == "UNCOMPILABLE":
                print(f"  [{kind:^12}] {label:<38} {out[:96]}")
                sys.stdout.flush()
            continue

        _, flags, size, digest = out.split()
        n["cmp"] += 1
        res = {k: flags[i] == "1" for i, k in enumerate(("det", "eng", "rec"))}
        for key in ("det", "eng", "rec"):
            if res[key]:
                n[key] += 1
            else:
                bad.append((label, key, "HSACO differs"))
        if not all(res.values()):
            print(
                f"  [    FAIL    ] {label:<38} det={res['det']} "
                f"eng={res['eng']} rec={res['rec']}"
            )
        elif verbose:
            print(
                f"  [     ok     ] {label:<38} {int(size)/1024.0:6.1f}KiB "
                f"{dt:5.1f}s sha={digest}"
            )
        sys.stdout.flush()
    return n, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx950,gfx942")
    ap.add_argument("--flavor", default="auto")
    ap.add_argument(
        "--cap-gb",
        type=int,
        default=48,
        help="per-compile address-space cap; keeps a pathological kernel from "
        "taking down the host",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.flavor == "auto":
        from rocke.core.lower_llvm import _flavor_for_rocm
        from rocke.runtime.comgr import resolved_lib_rocm_version

        ver = resolved_lib_rocm_version()
        flavor = _flavor_for_rocm(*ver) if ver else "llvm20"
    else:
        flavor = args.flavor
    os.environ["ROCKE_LLVM_FLAVOR"] = flavor
    os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

    from rocke.portable_ir.src import online

    online.load()

    rc = 0
    for arch in args.arch.split(","):
        print(f"\n== HSACO parity, concrete paths ({arch}, flavor={flavor}) ==")
        n, bad = run(arch, flavor, args.cap_gb, args.verbose)
        print(f"  compared at HSACO      : {n['cmp']}")
        print(f"  comgr deterministic    : {n['det']}/{n['cmp']}")
        print(f"  engine HSACO identical : {n['eng']}/{n['cmp']}")
        print(f"  recipe HSACO identical : {n['rec']}/{n['cmp']}")
        print(
            f"  not compilable         : {n['uncompilable']} "
            f"(LLVM fatal error -- a defect in the kernel, not in parity)"
        )
        print(
            f"  declined by lowerer    : {n['refused']} "
            f"(correct: wrong arch family)"
        )
        for lab, key, why in bad:
            print(f"    FAIL {lab:<34} {key}: {why}")
        rc |= 1 if bad else 0
    print("\n" + ("PASS" if not rc else "FAIL"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
