# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify, determinism-check and time routed convolutions THROUGH dispatch.

Every case of a route table (``dispatch/conv_routes.py``) is resolved by
:func:`dispatch.conv_routes.dispatch_conv_route` and bound by
:class:`benchmarks.common.conv_route_exec.RouteBinding` -- the launches a runtime
gets from dispatch -- then:

* ``--verify``: one call on training-like data (activations / gradients N(0, 1),
  weights Kaiming normal) into a poisoned (all-ones bytes) output, against an f32
  torch reference of the padded problem. PASS: max|err| / max(max|ref|, 1) < 5e-2
  (the Winograd benchmark's rule); the scale-free mean error and the bf16 rounding
  floor are reported beside it.
* ``--det-runs N``: N more calls, each into a re-poisoned output, compared bitwise
  with the first; the output's sha256 goes into the result row, so runs on two GPUs
  can be compared (``--against <jsonl>``).
* The two-stage Stage 2 kernels are the canonical ones
  (:func:`benchmark_implicit_gemm_conv.stage2_policy`); a run with a
  ``ROCKE_WGRAD_S2_*`` / ``ROCKE_PBD_S2_*`` override set is refused unless
  ``--allow-stage2-env``.
* ``--ref-routes``: the reference table's route of the case is bound on the same
  operands and verified against the same f32 reference (``ref_verify``; implies
  ``--verify``). ``--ref-torchws`` / ``--ref-dropin`` bindings must reproduce the
  routed output bitwise. A case whose routed or reference call fails is not timed.
* ``--rounds R``: staged per-call timing (``time_launches_staged``, median of
  ``--iters``) of the whole routed call, R interleaved rounds over the cases. With
  ``--parts`` the route's steps are also timed alone (pad copy, pinned kernel,
  Winograd launches).
* Every result row carries its provenance (:func:`_provenance`): the device's
  gfx target, the cache's target (they must match, or the run is refused), the
  LLVM flavor and COMGR the process compiles with, the table's and the pinned
  kernel's flavor, the harness and its arguments, and the timing method.

Usage::

    python -m benchmarks.common.conv_route_check --routes routes.json#deterministic \\
        --cache <aot cache> --cases fwd_s1_pad1,fwd_s3_pad1 --verify --det-runs 5 --rounds 3 \\
        --results out.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time


def _ref(direction, problem, A, Bt, torch):
    """f32 reference of the padded problem: fwd (X, W) -> Y, dgrad (dY, W) -> dX,
    wgrad (dY, X) -> dW; NDHWC tensors, W [K, Z, Y, X, C]."""
    F = torch.nn.functional
    pad = (problem.pD, problem.pH, problem.pW)
    if direction == "fwd":
        x = A.float().permute(0, 4, 1, 2, 3)
        w = Bt.float().permute(0, 4, 1, 2, 3)
        return F.conv3d(x, w, padding=pad).permute(0, 2, 3, 4, 1)
    if direction == "dgrad":
        dy = A.float().permute(0, 4, 1, 2, 3)
        w = Bt.float().permute(0, 4, 1, 2, 3)
        return F.conv_transpose3d(dy, w, padding=pad).permute(0, 2, 3, 4, 1)
    dy = A.float().permute(0, 4, 1, 2, 3)
    x = Bt.float().permute(0, 4, 1, 2, 3)
    wshape = (problem.K, problem.C, problem.Z, problem.Y, problem.X)
    return torch.nn.grad.conv3d_weight(x, wshape, dy, padding=pad).permute(
        0, 2, 3, 4, 1
    )


def _check(out, ref, torch) -> dict:
    """Accuracy of one call's output ``out`` against the f32 reference ``ref``."""
    out = out.float()
    err = (out - ref).abs()
    nan = int(torch.isnan(out).sum().item())
    rel = err.max().item() / max(ref.abs().max().item(), 1.0)
    rnd = (ref.to(torch.bfloat16).float() - ref).abs()
    return dict(
        verify="PASS" if (rel < 5e-2 and nan == 0) else "FAIL",
        rel_err=rel,
        nan=nan,
        mean_rel=err.mean().item() / ref.abs().mean().item(),
        max_rel_scale_free=err.max().item() / ref.abs().max().item(),
        bf16_floor_mean_rel=rnd.mean().item() / ref.abs().mean().item(),
    )


def _sha(D, torch) -> str:
    return hashlib.sha256(D.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


def _kernel_flavor(plan, KernelIdentity):
    """LLVM flavor the plan's pinned kernel was built for (``None``: no pinned kernel)."""
    if plan.kernel is None:
        return None
    return KernelIdentity.from_label(plan.kernel.spec.name).llvm_flavor


def _provenance(a, argv, device_target, routes) -> dict:
    """What a result row was measured with: run target (the device), compile
    target (the cache), toolchain, harness and timing method."""
    import os
    import subprocess

    from benchmarks.common.kernel_cache import current_comgr_id, current_llvm_flavor

    try:
        rev = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__))]
            + ["rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        rev = ""
    timing = None
    if a.rounds > 0:
        timing = {
            "method": "staged",
            "timer": "rocke.runtime.stall_gate.time_launches_staged",
            "stat": "median of iters, median of rounds",
            "warmup": a.warmup,
            "iters": a.iters,
            "rounds": a.rounds,
            "interleaved": True,
        }
    return {
        "gfx_target": device_target.split(":")[0],
        "device_target_id": device_target,
        "cache_arch": a.arch,
        "llvm_flavor": current_llvm_flavor(),
        "table_llvm_flavor": routes.llvm_flavor,
        "comgr": current_comgr_id(),
        "harness": "benchmarks/common/conv_route_check.py",
        "harness_rev": rev or None,
        "argv": list(argv),
        "timing": timing,
    }


def _operands(direction, p, torch, seed):
    """(A, B, D) torch tensors of one call (training-like data, fixed seed)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs = (p.N, p.Di, p.Hi, p.Wi, p.C)
    ys = (p.N, p.Do, p.Ho, p.Wo, p.K)
    ws = (p.K, p.Z, p.Y, p.X, p.C)

    def rn(shape, std=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * std).to(
            torch.bfloat16
        )

    wstd = (2.0 / (27 * p.C)) ** 0.5
    if direction == "fwd":
        return (
            rn(xs),
            rn(ws, wstd),
            torch.empty(ys, device="cuda", dtype=torch.bfloat16),
        )
    if direction == "dgrad":
        return (
            rn(ys),
            rn(ws, wstd),
            torch.empty(xs, device="cuda", dtype=torch.bfloat16),
        )
    return rn(ys), rn(xs), torch.empty(ws, device="cuda", dtype=torch.bfloat16)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--routes", required=True, help="ROUTE_TABLE[#MODE]")
    ap.add_argument(
        "--pins", default=None, help="PIN_TABLE#MODE for rows without a kernel"
    )
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arch", default="gfx942")
    ap.add_argument(
        "--cases", default=None, help="comma list (default: every case of the mode)"
    )
    ap.add_argument(
        "--route", default=None, help="only cases of these routes (comma list)"
    )
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--det-runs", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--parts", action="store_true", help="also time each step of the route alone"
    )
    ap.add_argument(
        "--ref-routes",
        default=None,
        metavar="ROUTE_TABLE[#MODE]",
        help="also verify (implies --verify) and time this table's route of every case on the same "
        "buffers, interleaved with the routed call (same-session A/B; e.g. a table of direct pins)",
    )
    ap.add_argument(
        "--ref-dropin",
        action="store_true",
        help="winograd routes: also run kernels.conv.conv_winograd3d_dropin.Winograd3dDropin on the "
        "same plan and buffers -- its output must equal the route's bitwise -- and time it "
        "interleaved (ms_dropin)",
    )
    ap.add_argument(
        "--ref-torchws",
        action="store_true",
        help="also bind the route with its workspaces from torch's allocator (timed interleaved, "
        "ms_torchws): isolates the workspace placement",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--against",
        default=None,
        help="results jsonl of another run: compare output sha256",
    )
    ap.add_argument("--results", default=None)
    ap.add_argument(
        "--allow-stage2-env",
        action="store_true",
        help="run although ROCKE_WGRAD_S2_* / ROCKE_PBD_S2_* Stage-2 overrides are set "
        "(the routes are then checked with a Stage 2 other than the canonical one)",
    )
    argv = sys.argv[1:] if argv is None else list(argv)
    a = ap.parse_args(argv)
    # An A/B against a reference route gates both routes on the f32 reference.
    a.verify = a.verify or bool(a.ref_routes)

    import os

    import torch

    from benchmarks.common import benchmark_implicit_gemm_conv as B
    from benchmarks.common.conv_route_exec import RouteBinding
    from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
    from dispatch.conv_pins import load_conv_pins
    from dispatch.conv_routes import (
        dispatch_conv_route,
        load_conv_routes,
        request_for_route,
    )
    from rocke.runtime.hip_module import Runtime, get_device_target_id
    from rocke.runtime.stall_gate import time_launches_staged
    from pathlib import Path

    s2_env = B.stage2_env_overrides()
    if s2_env and not a.allow_stage2_env:
        print(
            f"error: Stage-2 overrides set {s2_env}: the routes would be checked with a "
            "Stage 2 other than the one they are pinned with (unset them, or "
            "--allow-stage2-env)",
            file=sys.stderr,
        )
        return 2

    file, _, mode = a.routes.partition("#")
    routes = load_conv_routes(file, mode or None)
    pins = None
    if a.pins:
        pf, _, pm = a.pins.partition("#")
        pins = load_conv_pins(pf, pm or None)
    by_case = routes.by_case()
    ref_routes = None
    if a.ref_routes:
        rf, _, rm = a.ref_routes.partition("#")
        ref_routes = load_conv_routes(rf, rm or None)
    names = a.cases.split(",") if a.cases else sorted(by_case)
    if a.route:
        keep = set(a.route.split(","))
        names = [n for n in names if by_case[n].route in keep]
    against = {}
    if a.against:
        for line in open(a.against):
            r = json.loads(line)
            if r.get("sha256"):
                against[r["case"]] = r["sha256"]
    rt = Runtime()
    cache = KernelCache(Path(a.cache), a.arch)
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "?")
    # Run target vs compile target: binaries built for another gfx target
    # would run (or fail) on the device, not give a result for --arch.
    device_target = get_device_target_id(torch.cuda.current_device()) or "?"
    if device_target.split(":")[0] != a.arch:
        print(
            f"error: the device is {device_target}, the cache and routes are for "
            f"{a.arch} (--arch)",
            file=sys.stderr,
        )
        return 2
    prov = _provenance(a, argv, device_target, routes)
    stage2 = {}
    rows, bound = {}, {}
    rc = 0
    for case in names:
        r = by_case[case]
        req = request_for_route(r, a.arch)
        plan = dispatch_conv_route(req, routes, pins)
        p = plan.problem
        t0 = time.time()
        A, Bt, D = _operands(plan.direction, p, torch, a.seed)
        rb = RouteBinding(
            plan,
            cache,
            rt,
            a.arch,
            ptrs=(A.data_ptr(), Bt.data_ptr(), D.data_ptr()),
            sizes=tuple(t.numel() * 2 for t in (A, Bt, D)),
            stage2_kernels=stage2,
            B=B,
        )
        torch.cuda.synchronize()
        row = {
            "case": case,
            "route": plan.route,
            "kernel": plan.kernel_label,
            "winograd": dict(plan.winograd.spec) if plan.winograd is not None else None,
            "wtv": plan.winograd.wtv if plan.winograd is not None else None,
            "steps": list(plan.launches),
            "workspace_bytes": plan.workspace_bytes,
            "gpu": gpu,
            "bind_s": round(time.time() - t0, 2),
            "routes": a.routes,
            "cache": a.cache,
            **prov,
            "kernel_llvm_flavor": _kernel_flavor(plan, KernelIdentity),
        }
        failed = False
        rbr = None
        if ref_routes is not None and ref_routes.lookup(req) is not None:
            rplan = dispatch_conv_route(req, ref_routes, pins)
            rbr = RouteBinding(
                rplan,
                cache,
                rt,
                a.arch,
                ptrs=rb.ptrs,
                sizes=rb.sizes,
                stage2_kernels=stage2,
                B=B,
            )
            row["ref_route"], row["ref_kernel"] = rplan.route, rplan.kernel_label
            row["ref_routes"] = a.ref_routes
            row["ref_table_llvm_flavor"] = ref_routes.llvm_flavor
            row["ref_kernel_llvm_flavor"] = _kernel_flavor(rplan, KernelIdentity)
        if a.verify or a.det_runs:
            D.view(torch.int16).fill_(
                -1
            )  # poison (bf16 NaN): every output element must be written
            rb.run()
            torch.cuda.synchronize()
            first = D.clone()
            row["sha256"] = _sha(first, torch)
            if a.verify:
                ref = _ref(plan.direction, p, A, Bt, torch)
                row.update(_check(first, ref, torch))
                failed |= row["verify"] != "PASS"
                if rbr is not None:
                    # The reference route on the same operands, against the
                    # same f32 reference: an A/B is reported only when both pass.
                    D.view(torch.int16).fill_(-1)
                    rbr.run()
                    torch.cuda.synchronize()
                    row["ref_sha256"] = _sha(D, torch)
                    row.update(
                        {f"ref_{k}": v for k, v in _check(D, ref, torch).items()}
                    )
                    failed |= row["ref_verify"] != "PASS"
                del ref
            if a.det_runs:
                diff = 0
                for _ in range(a.det_runs):
                    D.view(torch.int16).fill_(-1)
                    rb.run()
                    torch.cuda.synchronize()
                    diff += int(
                        not torch.equal(D.view(torch.int16), first.view(torch.int16))
                    )
                row["det_runs"] = a.det_runs
                row["det_identical"] = diff == 0
                failed |= diff != 0
            if case in against:
                row["against_identical"] = against[case] == row["sha256"]
                failed |= not row["against_identical"]
            del first
        drop = None
        if a.ref_dropin and plan.route == "winograd":
            from types import SimpleNamespace

            from kernels.conv.conv_winograd3d_dropin import Winograd3dDropin

            dobj = Winograd3dDropin(p, plan.direction, plan.winograd, arch=a.arch)
            Dd = D.view(dobj.out)
            drop = SimpleNamespace(
                run=lambda dobj=dobj, A=A, Bt=Bt, Dd=Dd: dobj(A, Bt, Dd), obj=dobj
            )
            if "sha256" in row:
                D.view(torch.int16).fill_(-1)
                drop.run()
                torch.cuda.synchronize()
                row["dropin_sha256"] = _sha(D, torch)
                row["dropin_identical"] = row["dropin_sha256"] == row["sha256"]
                failed |= not row["dropin_identical"]
        rtw = None
        if a.ref_torchws and plan.workspaces:

            def _talloc(n):
                t = torch.empty(n, device="cuda", dtype=torch.uint8)
                return t.data_ptr(), t

            rtw = RouteBinding(
                plan,
                cache,
                rt,
                a.arch,
                ptrs=rb.ptrs,
                sizes=rb.sizes,
                stage2_kernels=stage2,
                B=B,
                alloc=_talloc,
            )
            if "sha256" in row:
                D.view(torch.int16).fill_(-1)
                rtw.run()
                torch.cuda.synchronize()
                row["torchws_identical"] = _sha(D, torch) == row["sha256"]
                failed |= not row["torchws_identical"]
        rows[case] = row
        if failed:
            rc = 1
            if a.rounds > 0:
                row["timing_skipped"] = "a routed or reference call failed its check"
        bound[case] = (rb, A, Bt, D, rbr, drop, rtw)
        print(
            f"[route] {case}: {plan.route} {row.get('verify', '')} det={row.get('det_identical', '')} "
            + (
                f"ref={row['ref_route']} {row.get('ref_verify', '')} "
                if "ref_route" in row
                else ""
            )
            + f"sha={row.get('sha256', '')} ws={plan.workspace_bytes / 2**20:.1f}MB"
            + (
                f" dropin_identical={row['dropin_identical']}"
                if "dropin_identical" in row
                else ""
            )
            + (" NOT TIMED (check failed)" if "timing_skipped" in row else ""),
            flush=True,
        )
        if a.rounds <= 0 or failed:
            torch.cuda.synchronize()
            for b in (rb, rbr, rtw):
                if b is not None:
                    b.free(B)
            del bound[case]
            torch.cuda.empty_cache()
    if a.rounds > 0:
        ms = {c: [] for c in bound}
        ms_ref = {c: [] for c in bound}
        ms_drop = {c: [] for c in bound}
        ms_tw = {c: [] for c in bound}
        parts = {c: {} for c in bound}
        for rnd in range(a.rounds):
            for c, (rb, _A, _B, _D, rbr, drop, rtw) in bound.items():
                order = (
                    [(ms[c], rb)]
                    + ([(ms_ref[c], rbr)] if rbr is not None else [])
                    + ([(ms_drop[c], drop)] if drop is not None else [])
                    + ([(ms_tw[c], rtw)] if rtw is not None else [])
                )
                if rnd % 2:
                    order.reverse()
                for acc, b in order:
                    acc.append(
                        statistics.median(
                            time_launches_staged(b.run, warmup=a.warmup, iters=a.iters)
                        )
                    )
                if a.parts and len(rb.runs) > 1:
                    for i, (run, nm) in enumerate(zip(rb.runs, rb.plan.launches)):
                        parts[c].setdefault(f"{i}:{nm[:40]}", []).append(
                            statistics.median(
                                time_launches_staged(
                                    run, warmup=a.warmup, iters=a.iters
                                )
                            )
                        )
        for c in bound:
            rows[c]["ms_rounds"] = [round(x, 4) for x in ms[c]]
            rows[c]["ms"] = round(statistics.median(ms[c]), 4)
            if ms_ref[c]:
                rows[c]["ms_ref_rounds"] = [round(x, 4) for x in ms_ref[c]]
                rows[c]["ms_ref"] = round(statistics.median(ms_ref[c]), 4)
                rows[c]["ratio_vs_ref"] = round(rows[c]["ms"] / rows[c]["ms_ref"], 4)
            if ms_tw[c]:
                rows[c]["ms_torchws"] = round(statistics.median(ms_tw[c]), 4)
            if ms_drop[c]:
                rows[c]["ms_dropin_rounds"] = [round(x, 4) for x in ms_drop[c]]
                rows[c]["ms_dropin"] = round(statistics.median(ms_drop[c]), 4)
            if parts[c]:
                rows[c]["ms_parts"] = {
                    k: round(statistics.median(v), 4) for k, v in parts[c].items()
                }
            print(
                f"[route] {c}: staged {rows[c]['ms']:.4f} ms  rounds {rows[c]['ms_rounds']}"
                + (
                    f"  ref {rows[c]['ms_ref']:.4f} ({rows[c]['ratio_vs_ref']:.3f}x)"
                    if ms_ref[c]
                    else ""
                )
                + (f"  dropin {rows[c]['ms_dropin']:.4f}" if ms_drop[c] else "")
                + (f"  torchws {rows[c]['ms_torchws']:.4f}" if ms_tw[c] else "")
                + (f"  parts {rows[c]['ms_parts']}" if parts[c] else ""),
                flush=True,
            )
    if a.results:
        with open(a.results, "a") as f:
            for c in names:
                f.write(json.dumps(rows[c]) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
