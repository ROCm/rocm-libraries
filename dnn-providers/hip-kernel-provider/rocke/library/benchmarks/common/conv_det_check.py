# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bitwise determinism and accuracy check of one cached conv kernel launch.

Picks one kernel out of an AOT cache by its label (as the benchmark tables
print it, ``<label> [@split_k=<d>[/full<t>]]``), binds it exactly as
``benchmark_implicit_gemm_conv.py --run-from-cache`` does, and

* **determinism** (default): launches it ``--runs`` times (>= 20) on the same
  inputs, each launch into a freshly allocated, zeroed output buffer with a
  freshly allocated launch scratch (``--batch`` launches queued back to back
  before they are read back), and compares every output bitwise with the
  first. Reports identical yes/no, the number of differing launches and the
  max ulp difference. ``--save FILE`` writes the first output (and its
  sha256) so a run on another GPU can be compared with ``--against FILE``.
* **accuracy** (``--acc``): max / mean / L2 relative error of the output vs an
  fp64 (``--ref fp64``, default) or fp32 reference computed on the GPU from the
  same bf16 inputs, plus the same numbers for the reference rounded to the
  output dtype (the floor no kernel can beat). Scale-free: every error is
  divided by the reference's own max / mean / norm.

Data: ``--data train`` (default) is training-like -- x ~ N(0,1), W Kaiming
normal (std sqrt(2 / (C*Z*Y*X))), dY ~ N(0,1); ``--data uniform`` is the
benchmark's U(-1, 1).

Prints one ``RESULT {json}`` line. Exit status 0 when deterministic (or
``--no-det``), 1 when outputs differ, 2 on usage errors.

Example::

  python benchmarks/common/conv_det_check.py --cache CACHE --arch gfx942 \\
      --miopen-cmd "MIOpenDriver convbfp16 ... -F 4 ..." \\
      --kernel "wgrad-implicit_gemm-...-llvm22 @split_k=304" --runs 20 --acc
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path

_DIR_OF_F = {1: "fwd", 2: "dgrad", 4: "wgrad"}


def _parse_launch(label: str):
    from benchmarks.common.benchmark_implicit_gemm_conv import TailSplit

    want = " ".join(label.split())
    m = re.search(r"@split_k=(\d+)(?:/full(\d+))?", want)
    sk = None
    if m:
        sk = int(m.group(1))
        if m.group(2) is not None:
            sk = TailSplit(sk, int(m.group(2)))
    return want.split()[0], sk


def _shapes(problem, direction):
    nhwc = (problem.N, problem.Di, problem.Hi, problem.Wi, problem.C)
    nhwk = (problem.N, problem.Do, problem.Ho, problem.Wo, problem.K)
    krsc = (problem.K, problem.Z, problem.Y, problem.X, problem.cpg)
    if direction == "wgrad":
        return nhwk, nhwc, krsc
    if direction == "dgrad":
        return nhwk, krsc, nhwc
    return nhwc, krsc, nhwk


def make_inputs(problem, direction, data: str, seed: int, torch_dt):
    """(A, B) host tensors bound to the A / B kernarg slots."""
    import torch

    a_shape, b_shape, _ = _shapes(problem, direction)
    g = torch.Generator().manual_seed(seed)
    fan_in = problem.cpg * problem.Z * problem.Y * problem.X
    w_std = (2.0 / fan_in) ** 0.5

    def act(shape):
        if data == "uniform":
            return torch.empty(*shape).uniform_(-1, 1, generator=g).to(torch_dt)
        return torch.randn(*shape, generator=g).to(torch_dt)

    def wt(shape):
        if data == "uniform":
            return torch.empty(*shape).uniform_(-1, 1, generator=g).to(torch_dt)
        return (torch.randn(*shape, generator=g) * w_std).to(torch_dt)

    if direction == "wgrad":  # dY, X
        return act(a_shape), act(b_shape)
    if direction == "dgrad":  # dY, W
        return act(a_shape), wt(b_shape)
    return act(a_shape), wt(b_shape)  # X, W


def reference(problem, direction, A, B, ref: str):
    """fp64 / fp32 reference in the kernel's output layout, on the GPU, as a
    tap-decomposed GEMM sum (stride 1, dilation 1, ungrouped, 3-D)."""
    import torch
    import torch.nn.functional as F

    p = problem
    if (p.sD, p.sH, p.sW) != (1, 1, 1) or (p.dD, p.dH, p.dW) != (1, 1, 1) or p.groups != 1:
        raise ValueError("reference: stride 1, dilation 1, groups 1 only")
    dt = torch.float64 if ref == "fp64" else torch.float32
    dev = "cuda"
    Z, Y, X = p.Z, p.Y, p.X
    pads = (p.pD, p.pH, p.pW)
    Do, Ho, Wo = p.Do, p.Ho, p.Wo
    # chunk the outer spatial depth so one tap operand stays ~64M elements
    if direction == "wgrad":
        dY = A.to(dev)
        Xp = F.pad(B.to(dev), (0, 0, pads[2], pads[2], pads[1], pads[1], pads[0], pads[0]))
        out = torch.zeros(p.K, Z, Y, X, p.C, dtype=dt, device=dev)
        plane = p.N * Ho * Wo * max(p.C, p.K)
        dc = max(1, (64 << 20) // plane)
        for d0 in range(0, Do, dc):
            d1 = min(Do, d0 + dc)
            a = dY[:, d0:d1].to(dt).reshape(-1, p.K).t()
            for z in range(Z):
                for y in range(Y):
                    for x in range(X):
                        xs = Xp[:, d0 + z:d1 + z, y:y + Ho, x:x + Wo, :].to(dt).reshape(-1, p.C)
                        out[:, z, y, x, :] += a @ xs
        return out
    if direction == "fwd":
        Xp = F.pad(A.to(dev), (0, 0, pads[2], pads[2], pads[1], pads[1], pads[0], pads[0]))
        W = B.to(dev).to(dt)  # K Z Y X C
        out = torch.zeros(p.N, Do, Ho, Wo, p.K, dtype=dt, device=dev)
        plane = p.N * Ho * Wo * max(p.C, p.K)
        dc = max(1, (64 << 20) // plane)
        for d0 in range(0, Do, dc):
            d1 = min(Do, d0 + dc)
            acc = torch.zeros(p.N * (d1 - d0) * Ho * Wo, p.K, dtype=dt, device=dev)
            for z in range(Z):
                for y in range(Y):
                    for x in range(X):
                        xs = Xp[:, d0 + z:d1 + z, y:y + Ho, x:x + Wo, :].to(dt).reshape(-1, p.C)
                        acc += xs @ W[:, z, y, x, :].t()
            out[:, d0:d1] = acc.reshape(p.N, d1 - d0, Ho, Wo, p.K)
        return out
    # dgrad: dX[i] = sum_t dYp[i + (F-1-t)] @ W[t], dY padded by F-1-p
    Di, Hi, Wi = p.Di, p.Hi, p.Wi
    L = (Z - 1 - pads[0], Y - 1 - pads[1], X - 1 - pads[2])
    if min(L) < 0:
        raise ValueError("reference: dgrad needs pad <= filter - 1")
    dYp = F.pad(A.to(dev), (0, 0, L[2], L[2], L[1], L[1], L[0], L[0]))
    W = B.to(dev).to(dt)  # K Z Y X C
    out = torch.zeros(p.N, Di, Hi, Wi, p.C, dtype=dt, device=dev)
    plane = p.N * Hi * Wi * max(p.C, p.K)
    dc = max(1, (64 << 20) // plane)
    for d0 in range(0, Di, dc):
        d1 = min(Di, d0 + dc)
        acc = torch.zeros(p.N * (d1 - d0) * Hi * Wi, p.C, dtype=dt, device=dev)
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    oz, oy, ox = Z - 1 - z, Y - 1 - y, X - 1 - x
                    ys = dYp[:, d0 + oz:d1 + oz, oy:oy + Hi, ox:ox + Wi, :].to(dt).reshape(-1, p.K)
                    acc += ys @ W[:, z, y, x, :]
        out[:, d0:d1] = acc.reshape(p.N, d1 - d0, Hi, Wi, p.C)
    return out


def accuracy(out_t, ref_t, torch_dt):
    import torch

    def stats(o):
        d = (o - ref_t).abs()
        r = ref_t.abs()
        return {
            "max_rel": float(d.max() / r.max()),
            "mean_rel": float(d.mean() / r.mean()),
            "l2_rel": float(torch.linalg.vector_norm(o - ref_t) / torch.linalg.vector_norm(ref_t)),
        }

    o = out_t.to("cuda").to(ref_t.dtype)
    s = stats(o)
    fl = stats(ref_t.to(torch_dt).to(ref_t.dtype))
    s.update({f"floor_{k}": v for k, v in fl.items()})
    # finer: distance in output ulps from the correctly rounded reference
    rr = ref_t.to(torch_dt).cpu()
    oc = out_t.to(torch_dt).cpu()
    s["frac_ne_rounded_ref"] = float((_ordered(oc) != _ordered(rr)).double().mean())
    s["max_ulp_vs_rounded_ref"] = max_ulp(oc, rr)
    return s


def _ordered(t):
    """Sign-magnitude float bits -> monotonic integers (ulp distance = difference)."""
    import torch

    if t.element_size() == 2:
        i = t.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    i = t.view(torch.int32).to(torch.int64)
    return torch.where(i < 0, -(i & 0x7FFFFFFF), i)


def max_ulp(a, b) -> int:
    return int((_ordered(a) - _ordered(b)).abs().max())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="AOT kernel cache directory")
    ap.add_argument("--arch", default="gfx942")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--miopen-cmd")
    src.add_argument("--miopen-file", help="case file; --line picks the line (1-based, default 1)")
    ap.add_argument("--line", type=int, default=1)
    ap.add_argument("--kernel", required=True, help="kernel label [@split_k=<d>[/full<t>]]")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4, help="launches queued back to back per readback")
    ap.add_argument("--no-det", action="store_true", help="skip the determinism runs")
    ap.add_argument("--acc", action="store_true", help="accuracy vs the fp64/fp32 reference")
    ap.add_argument("--ref", default="fp64", choices=["fp64", "fp32"])
    ap.add_argument("--data", default="train", choices=["train", "uniform"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save", help="write the first output + its sha256 here (torch.save)")
    ap.add_argument("--against", help="compare the first output with a --save file (another GPU)")
    args = ap.parse_args(argv)

    import torch

    from benchmarks.common import benchmark_implicit_gemm_conv as big
    try:
        from benchmarks.common.det_classify import classify
    except ImportError:  # run from another worktree that lacks it: use the copy beside this file
        import importlib.util

        _spec = importlib.util.spec_from_file_location(
            "_det_classify", Path(__file__).with_name("det_classify.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        classify = _mod.classify
    from benchmarks.common.kernel_cache import KernelCache
    from benchmarks.common.module_cache import ModuleCache
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if args.miopen_file:
        lines = [l for l in Path(args.miopen_file).read_text().splitlines() if l.strip().startswith("MIOpenDriver") or "MIOpenDriver" in l]
        cmd = lines[args.line - 1]
    else:
        cmd = args.miopen_cmd
    problem, dtype, forw = big.parse_miopen_cmd(cmd)
    direction = _DIR_OF_F.get(int(forw))
    if direction is None:
        print(f"error: -F {forw}: one direction per check (1/2/4)", file=sys.stderr)
        return 2
    torch_dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    label, split_k = _parse_launch(args.kernel)

    cache = KernelCache(Path(args.cache), args.arch)
    run_problem = big._launch_problem(direction, problem)
    batch_chunks = big._batch_chunks(direction, run_problem, dtype)
    fit_problem = dc_replace(run_problem, N=batch_chunks[0][1])
    hits = [c for c in cache.compatible(fit_problem, direction=direction) if c[0].label() == label]
    if not hits:
        print(f"error: {label} not in {args.cache} (or cannot run this case)", file=sys.stderr)
        return 2
    ident, hsaco_path, meta = hits[0]
    det_cls, det_why = classify(direction, ident, split_k)

    A_t, B_t = make_inputs(problem, direction, args.data, args.seed, torch_dt)
    _, _, d_shape = _shapes(problem, direction)
    D_t = torch.empty(*d_shape, dtype=torch_dt)

    def u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

    rt = Runtime()
    A_dev, B_dev = rt.alloc(A_t.nbytes), rt.alloc(B_t.nbytes)
    rt.memcpy_h2d(A_dev, u8(A_t), A_t.nbytes)
    rt.memcpy_h2d(B_dev, u8(B_t), B_t.nbytes)
    modules = ModuleCache(0)
    launcher = modules.launcher(
        hsaco_path=hsaco_path,
        kernel_name=meta["kernel_name"],
        signature=big._signature_for_identity(ident, dtype),
    )
    stage2: dict = {}
    sizes = (A_t.nbytes, B_t.nbytes, D_t.nbytes)

    def bind(d_dev, live):
        bound = big._cached_bind_batch(
            batch_chunks, direction, run_problem, ident, meta, rt, launcher, dtype,
            split_k, (A_dev, B_dev, d_dev), sizes, args.arch, stage2, KernelLauncher,
            LaunchConfig, live,
        )
        if bound is None:
            raise SystemExit(2)
        return bound

    result = {
        "kernel": args.kernel.strip(), "case": problem.short(), "direction": direction,
        "classified_deterministic": det_cls, "classify_reason": det_why,
        "gpu": __import__("os").environ.get("HIP_VISIBLE_DEVICES", ""),
        "data": args.data,
    }

    outs_first = None
    n_diff, worst_ulp, n_done, worst_rel = 0, 0, 0, 0.0
    t0 = time.time()
    runs = 1 if args.no_det else max(1, args.runs)
    while n_done < runs:
        nb = min(args.batch, runs - n_done)
        bufs, live = [], []
        runs_fn = []
        for _ in range(nb):
            d = rt.alloc(D_t.nbytes)
            rt.memset(d, 0, D_t.nbytes)
            run, _lbl = bind(d, live)
            bufs.append(d)
            runs_fn.append(run)
        for run in runs_fn:  # queued back to back, one drain
            run()
        synchronize_and_release()
        for d in bufs:
            o = torch.empty_like(D_t)
            rt.memcpy_d2h(u8(o), d, D_t.nbytes)
            if outs_first is None:
                outs_first = o
            elif not torch.equal(o.view(torch.uint8), outs_first.view(torch.uint8)):
                n_diff += 1
                worst_ulp = max(worst_ulp, max_ulp(o, outs_first))
                f0 = outs_first.float()
                worst_rel = max(worst_rel, float((o.float() - f0).abs().max() / f0.abs().max().clamp(min=1e-30)))
            rt.free(d)
            n_done += 1
        big._free_extras(rt, live)
    sha = hashlib.sha256(outs_first.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    result.update({
        "runs": n_done, "identical": n_diff == 0, "n_differing_runs": n_diff,
        "max_ulp": worst_ulp, "max_rel_diff": worst_rel, "sha256": sha, "det_seconds": round(time.time() - t0, 2),
    })
    if args.save:
        torch.save({"out": outs_first, "sha256": sha, "kernel": args.kernel, "case": problem.short()}, args.save)
    if args.against:
        other = torch.load(args.against)
        same = torch.equal(other["out"].view(torch.uint8), outs_first.view(torch.uint8))
        result.update({
            "against": args.against, "against_identical": same,
            "against_max_ulp": 0 if same else max_ulp(other["out"], outs_first),
        })
    if args.acc:
        ref_t = reference(problem, direction, A_t, B_t, args.ref)
        result["acc"] = accuracy(outs_first, ref_t, torch_dt)
        result["acc"]["ref"] = args.ref
        del ref_t
    rt.free(A_dev)
    rt.free(B_dev)
    print("RESULT " + json.dumps(result), flush=True)
    ok = result["identical"] and result.get("against_identical", True)
    return 0 if (ok or args.no_det) else 1


if __name__ == "__main__":
    raise SystemExit(main())
