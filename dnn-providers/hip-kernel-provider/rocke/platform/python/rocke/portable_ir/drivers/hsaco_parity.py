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
#   python3 -m rocke.portable_ir.drivers.hsaco_parity [--arch gfx950] [--cap-gb N]
#
# Needs a shared librocke (ROCKE_ONLINE_LIB) and comgr. No device required.
#
# AS A CI GATE. Two different things can go wrong and they need different
# verdicts. A kernel whose HSACO *differs* between paths is a parity failure and
# always fails the run. A kernel that does not compile at all is usually a
# pre-existing defect in the kernel set -- 14 of them on gfx942 use gfx950-only
# MFMA intrinsics -- so failing on those would make the gate red on day one and
# it would be turned off. Instead the uncompilable and lowerer-declined sets are
# pinned in `hsaco_baseline.json` by NAME, and the gate fails when the sets move
# in the direction that matters:
#
#   new kernel cannot compile      -> FAIL (a regression this gate exists to catch)
#   fewer kernels compared         -> FAIL (coverage silently shrank)
#   baselined kernel now compiles  -> pass, but say so loudly; run --update-baseline
#
# Pinning names rather than counts is deliberate: one kernel fixed and another
# broken leaves the count unchanged, and that is exactly the swap worth catching.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from typing import Callable, Dict, List, Tuple

_BASELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hsaco_baseline.json"
)


def _auto_cap_gb() -> int:
    """Half of physical memory, clamped to [4, 48] GiB.

    The cap has to fit the machine, not the author's machine. A fixed 48 GiB is
    fine on a 3 TB developer node and useless on a 16 GiB CI runner, where a
    kernel that wants unbounded memory would be killed by the OOM killer --
    possibly taking a different process with it -- long before it reached the
    limit. Half of physical leaves room for the parent and the toolchain; the
    floor keeps the cap workable on small machines, and 45 of the 46 kernels
    compile in well under 4 GiB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    gb = int(line.split()[1]) // (1024 * 1024)
                    return max(4, min(48, gb // 2))
    except (OSError, IndexError, ValueError):
        pass
    return 8


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _llvm_reason(stderr: str) -> str:
    """The line that explains a crash, out of whatever the child printed.

    LLVM prints its diagnosis and then calls abort(), so the reason never comes
    back through the child's own message -- the child does not survive to send
    one. Recovering it from stderr is what lets the baseline record *why* a
    kernel is unshippable ("Cannot select: intrinsic ...mfma.f32.32x32x16.f16")
    instead of an uninformative signal number."""
    for line in reversed(stderr.strip().splitlines()):
        if "LLVM ERROR" in line or "error:" in line or "Allocation failed" in line:
            return line.strip()[:160]
    return ""


def _in_child(fn: Callable[[], str], cap_gb: int) -> Tuple[str, str]:
    """Run fn() in a forked child under a memory cap.

    Returns (message, stderr). The message is the child's own report, or a
    CRASH/MEMCAP description if it died. The cap is applied in the child so the
    parent survives to report. The child's stderr is captured rather than
    inherited so a crash reason can be attributed to the kernel that caused it;
    it is echoed by the caller, so nothing is hidden."""
    r, w = os.pipe()
    er, ew = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        os.close(er)
        os.dup2(ew, 2)  # LLVM writes its fatal errors straight to fd 2
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
    os.close(ew)

    def drain(fd: int) -> bytes:
        out = b""
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            out += chunk
        os.close(fd)
        return out

    buf = drain(r)
    err = drain(er).decode(errors="replace")
    _, status = os.waitpid(pid, 0)
    if buf:
        return buf.decode(), err
    if os.WIFSIGNALED(status):
        why = _llvm_reason(err) or "LLVM fatal error / OOM"
        return f"CRASH signal {os.WTERMSIG(status)}: {why}", err
    return f"CRASH exit {os.WEXITSTATUS(status)}: {_llvm_reason(err)}", err


def _load_baseline(path: str) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def check_baseline(arch: str, n: Dict, seen: Dict[str, str], base: Dict) -> List[str]:
    """Compare this run against the pinned expectations. Returns failures.

    Improvements are reported by the caller rather than failed on, so a kernel
    getting fixed never blocks a merge -- it just asks for a baseline update."""
    want = base.get(arch)
    if not want:
        return [f"{arch}: no baseline entry; run with --update-baseline to create one"]
    fails: List[str] = []
    got_bad = {k for k, v in seen.items() if v == "uncompilable"}
    got_ref = {k for k, v in seen.items() if v == "refused"}
    new_bad = sorted(got_bad - set(want.get("uncompilable", {})))
    new_ref = sorted(got_ref - set(want.get("refused", [])))
    if new_bad:
        fails.append(
            f"{arch}: {len(new_bad)} kernel(s) newly fail to compile: "
            + ", ".join(new_bad)
        )
    if new_ref:
        fails.append(
            f"{arch}: {len(new_ref)} kernel(s) newly declined by the lowerer: "
            + ", ".join(new_ref)
        )
    # Every newly broken or declined kernel already accounts for one fewer
    # comparison and is reported above, so only a shortfall those names do NOT
    # explain is worth a second line. That residue is the case this check is
    # really for: a kernel that vanished from the sweep altogether, which leaves
    # no name behind and so is invisible to the set differences.
    if n["cmp"] + len(new_bad) + len(new_ref) < want.get("compared", 0):
        fails.append(
            f"{arch}: compared {n['cmp']} kernels at HSACO, baseline expects "
            f"{want['compared']} -- coverage shrank for reasons not named above"
        )
    return fails


def baseline_improvements(arch: str, seen: Dict[str, str], base: Dict) -> List[str]:
    """Baselined-as-broken kernels that now work. Informational, never fatal."""
    want = base.get(arch) or {}
    fixed = sorted(set(want.get("uncompilable", {})) - set(seen))
    return [f"{arch}: {k} now compiles (drop it from the baseline)" for k in fixed]


def run(
    arch: str, flavor: str, cap_gb: int, verbose: bool
) -> Tuple[Dict, List, Dict[str, str], Dict[str, str]]:
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
    # label -> "uncompilable" | "refused", for the baseline comparison. Kernels
    # that compiled are absent, so `seen` is exactly the set of exceptions.
    # `reasons` carries the LLVM message for the uncompilable ones, so a written
    # baseline says why each kernel is on the list instead of just naming it.
    seen: Dict[str, str] = {}
    reasons: Dict[str, str] = {}

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

        out, child_err = _in_child(work, cap_gb)
        dt = time.perf_counter() - t0

        if not out.startswith("OK "):
            # Echo what the child printed; it was captured, not suppressed.
            if child_err.strip() and verbose:
                print(
                    "".join(
                        f"      | {ln}\n" for ln in child_err.strip().splitlines()[-4:]
                    ),
                    end="",
                )
            # A kernel the Python lowerer declines (wrong arch family) is
            # correct behavior; one that crashes LLVM is a defect in the kernel.
            if out.startswith("EXC NotImplementedError"):
                n["refused"] += 1
                kind = "refused"
                seen[label] = "refused"
            else:
                n["uncompilable"] += 1
                kind = "UNCOMPILABLE"
                seen[label] = "uncompilable"
                # Room for the whole diagnosis: clipping mid-intrinsic would
                # make the baseline entry unreadable, and it is the only record
                # of why the kernel is excused.
                reasons[label] = out[:200]
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
    return n, bad, seen, reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx950,gfx942")
    ap.add_argument("--flavor", default="auto")
    ap.add_argument(
        "--cap-gb",
        type=int,
        default=0,
        help="per-compile address-space cap in GiB; keeps a pathological kernel "
        "from taking down the host. 0 (default) sizes it to the machine",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--baseline",
        default=_BASELINE,
        help="pinned uncompilable/declined kernel sets to compare against",
    )
    ap.add_argument(
        "--no-baseline",
        action="store_true",
        help="report only; do not fail on compilability regressions",
    )
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the baseline from this run. Review the diff: it is the "
        "record of which kernels are known not to compile",
    )
    args = ap.parse_args()
    cap_gb = args.cap_gb or _auto_cap_gb()

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

    base = {} if args.update_baseline else _load_baseline(args.baseline)
    fresh: Dict[str, Dict] = {}
    fails: List[str] = []
    notes: List[str] = []

    rc = 0
    for arch in args.arch.split(","):
        print(
            f"\n== HSACO parity, concrete paths ({arch}, flavor={flavor}, "
            f"cap={cap_gb}G) =="
        )
        n, bad, seen, reasons = run(arch, flavor, cap_gb, args.verbose)
        fresh[arch] = {
            "compared": n["cmp"],
            "uncompilable": {
                k: reasons.get(k, "") for k, v in seen.items() if v == "uncompilable"
            },
            "refused": sorted(k for k, v in seen.items() if v == "refused"),
        }
        if not (args.update_baseline or args.no_baseline):
            fails += check_baseline(arch, n, seen, base)
            notes += baseline_improvements(arch, seen, base)
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

    if args.update_baseline:
        with open(args.baseline, "w") as f:
            json.dump(
                {
                    "_comment": (
                        "Kernels that do not reach HSACO, pinned by name so a NEW "
                        "one fails CI while these known-broken ones do not. "
                        "'uncompilable' is an LLVM fatal error -- a defect in the "
                        "kernel, not in portable IR. 'refused' is the Python "
                        "lowerer correctly declining a wrong-arch-family kernel. "
                        "Regenerate with: hsaco_parity --update-baseline"
                    ),
                    **fresh,
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
        print(f"\nbaseline written to {args.baseline}")
    for msg in notes:
        print(f"  NOTE {msg}")
    for msg in fails:
        print(f"  REGRESSION {msg}")
    rc |= 1 if fails else 0
    print("\n" + ("PASS" if not rc else "FAIL"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
