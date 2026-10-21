# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build and benchmark the grouped-convolution dispatcher's registered instances.

The dispatcher (``dispatch.grouped_convolution``) registers a fixed list of
forward and backward-data (dgrad) AOT instances -- implicit-GEMM and
direct-conv kernels, listed in ``dispatch/grouped_convolution_instances/``
(edit with instance_list.py) and named by their lossless labels. This benchmark
compiles those instances ahead of time into a kernel cache and times them on
concrete shapes, the dispatcher deciding which instances can run each shape
and how (:func:`dispatch.grouped_convolution.launch_plan_for`: the MFMA dgrad
runs its weight transforms first, and they are timed with it). The instances
are shape-generic binaries (problem extents are kernargs), so a cache built
once serves every later shape.

--direction picks the pass the cases run (default fwd); MIOpen commands keep
the cases whose -F includes it.

Modes, as in benchmark_implicit_gemm_conv.py and benchmark_direct_conv.py:

  --compile-all         build every registered instance into --cache-dir. No
                        shape needed, no GPU used. Incremental: only instances
                        whose code changed are recompiled.
  --run-from-cache DIR  run the cases on the binaries in DIR. Nothing is
                        compiled; an admitted instance missing from DIR is
                        reported, not built.
  (neither)             build just the instances the cases need into
                        --cache-dir (default: a temporary directory), then run.
  --list                print the instances the dispatcher admits per case;
                        nothing is compiled or launched.

Filtering, per case: the dispatcher admits the instances whose binary can run
the shape (vector widths, baked filter geometry, grid limits, the direct
kernels' knob rules); --algorithm, --instance and --sample narrow that further.
The instances run in the dispatcher's priority order, so the table also shows
where its default pick (marked '*') ranks.

The instances record the LLVM flavor they are lowered for (``llvm22``); run
with ``ROCKE_LLVM_FLAVOR`` set to it.

Examples:
  # Build every registered instance (both dtypes), all cores:
  python benchmark_grouped_conv.py --compile-all --cache-dir ./kernel_cache

  # Run one shape from that cache, verifying every kernel:
  python benchmark_grouped_conv.py --run-from-cache ./kernel_cache \\
      --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --Y 3 --X 3 --verify

  # The same shape's backward-data pass:
  python benchmark_grouped_conv.py --run-from-cache ./kernel_cache \\
      --direction dgrad --N 8 --Hi 56 --Wi 56 --C 64 --K 64 --Y 3 --X 3 --verify

  # Build what a case list needs (reusing ./kernel_cache) and run it:
  python benchmark_grouped_conv.py --cache-dir ./kernel_cache \\
      --json-file bench_cases_conv.json --algorithm implicit_gemm
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Suppress the "fell back to Python lowerer" warning — expected in environments
# where the C++ engine extension is not built.
os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

from benchmarks.common.early_stop import EarlyStop, add_early_stop_arg  # noqa: E402

_DTYPES = ("fp16", "bf16")
_DIRECTIONS = ("fwd", "dgrad")
# MIOpen -F bit (and ckProfiler direction) of each pass.
_FORW_BIT = {"fwd": 1, "dgrad": 2}


# ---------------------------------------------------------------------------
# Case input: shape flags, MIOpenDriver commands, JSON case files
# ---------------------------------------------------------------------------

_MIOPEN_DTYPE_MAP = {
    "conv": "fp32",
    "convfp32": "fp32",
    "convfp16": "fp16",
    "convbfp16": "bf16",
    "convint8": "int8",
}


def parse_json_case(entry: dict):
    """Parse a single JSON benchmark-case entry into a ``(ConvProblem, dtype)`` tuple.

    Supports entries from bench_cases_conv.json. Fields: N, Cin, Cout, H, W,
    Kh, Kw, stride (int or [sH,sW]), pad (int or [pH,pW]), dilation (int or
    [dH,dW]), groups, dtype, layout, Di/D/Kd for 3-D.

    Raises ``ValueError`` for unsupported layouts (NCHW) or unsupported op types.
    """
    op = entry.get("op", "conv2d")
    layout = entry.get("layout", "NHWC").upper()

    if op not in ("conv2d", "conv3d", "conv1d"):
        raise ValueError(f"op={op!r} is not supported")

    dtype = entry.get("dtype", "fp16")
    if dtype not in ("fp16", "bf16", "fp32"):
        raise ValueError(f"dtype={dtype!r} is not supported (only fp16, bf16, fp32)")

    from kernels.common.conv_implicit_gemm import ConvProblem

    def _scalar_or_pair(val):
        if isinstance(val, (list, tuple)):
            return int(val[0]), int(val[1])
        return int(val), int(val)

    def _triple(val):
        if isinstance(val, (list, tuple)):
            return int(val[0]), int(val[1]), int(val[2])
        return int(val), int(val), int(val)

    N = int(entry.get("N", 1))
    C = int(entry.get("Cin", entry.get("C", 1)))
    K = int(entry.get("Cout", entry.get("K", 1)))
    groups = int(entry.get("groups", 1))

    if op == "conv1d":
        return (
            ConvProblem(
                N=N,
                Hi=1,
                Wi=int(entry.get("L", 1)),
                C=C,
                K=K,
                Y=1,
                X=int(entry.get("Kl", 1)),
                sH=1,
                sW=int(entry.get("stride", 1)),
                pH=0,
                pW=int(entry.get("pad", 0)),
                dH=1,
                dW=int(entry.get("dilation", 1)),
                groups=groups,
            ),
            dtype,
        )

    if op == "conv3d":
        sD, sH, sW = _triple(entry.get("strides", entry.get("stride", 1)))
        pD, pH, pW = _triple(entry.get("pads_before", entry.get("pad", 0)))
        dD, dH, dW = _triple(entry.get("dilations", entry.get("dilation", 1)))
        return (
            ConvProblem(
                N=N,
                Di=int(entry.get("D", entry.get("Di", 1))),
                Hi=int(entry.get("H", entry.get("Hi", 1))),
                Wi=int(entry.get("W", entry.get("Wi", 1))),
                C=C,
                K=K,
                Z=int(entry.get("Kd", entry.get("Z", 1))),
                Y=int(entry.get("Kh", entry.get("Y", 1))),
                X=int(entry.get("Kw", entry.get("X", 1))),
                sD=sD,
                sH=sH,
                sW=sW,
                pD=pD,
                pH=pH,
                pW=pW,
                dD=dD,
                dH=dH,
                dW=dW,
                groups=groups,
            ),
            dtype,
        )

    if layout not in ("NHWC", "NWC"):
        raise ValueError(
            f"layout={layout!r} is not supported; only NHWC/NWC inputs are accepted"
        )
    if "pads_before" in entry:
        pH, pW = _scalar_or_pair(entry["pads_before"])
    else:
        pH, pW = _scalar_or_pair(entry.get("pad", 0))
    sH, sW = _scalar_or_pair(entry.get("stride", 1))
    dH, dW = _scalar_or_pair(entry.get("dilation", 1))
    return (
        ConvProblem(
            N=N,
            Hi=int(entry.get("H", entry.get("Hi", 1))),
            Wi=int(entry.get("W", entry.get("Wi", 1))),
            C=C,
            K=K,
            Y=int(entry.get("Kh", entry.get("Y", 1))),
            X=int(entry.get("Kw", entry.get("X", 1))),
            sH=sH,
            sW=sW,
            pH=pH,
            pW=pW,
            dH=dH,
            dW=dW,
            groups=groups,
        ),
        dtype,
    )


def parse_miopen_cmd(cmd: str):
    """Parse a MIOpenDriver command into ``(ConvProblem, dtype, forw)``.

    Accepts the full command line (including the binary path), e.g.::

        ./bin/MIOpenDriver convfp16 -n 8 -c 3 -H 224 -W 224 -k 64 \\
            -y 11 -x 11 -p 2 -q 2 -u 4 -v 4 -l 1 -j 1 -m conv -g 1 -F 1 \\
            -t 1 -in_layout=NHWC

    Only 2-D NHWC convolutions are accepted (``ValueError`` otherwise).
    ``forw`` is the raw ``-F`` value; the caller keeps the cases of the --direction pass.
    """
    import shlex

    tokens = shlex.split(cmd)

    # Strip binary path (anything before the driver keyword).
    driver_kw = None
    driver_idx = None
    for i, t in enumerate(tokens):
        # The driver keyword precedes every flag; stop at the first flag so a
        # flag *value* (e.g. the "conv" in "-m conv") is never taken for it.
        if t.startswith("-"):
            break
        key = t.split("/")[-1].lower()
        if key in _MIOPEN_DTYPE_MAP:
            driver_kw = key
            driver_idx = i
            break
    if driver_kw is None:
        raise ValueError(
            f"No MIOpenDriver keyword found in command "
            f"(expected one of: {list(_MIOPEN_DTYPE_MAP)})"
        )
    dtype = _MIOPEN_DTYPE_MAP[tokens[driver_idx].split("/")[-1].lower()]

    sub = argparse.ArgumentParser(add_help=False)
    for flags, dest, default in (
        (("-n", "--n"), "N", 1),
        (("-c", "--c"), "C", 1),
        (("-H", "--H"), "Hi", 1),
        (("-W", "--W"), "Wi", 1),
        (("-k", "--k"), "K", 1),
        (("-y", "--y"), "Y", 1),
        (("-x", "--x"), "X", 1),
        (("-p", "--p"), "pH", 0),
        (("-q", "--q"), "pW", 0),
        (("-u", "--u"), "sH", 1),
        (("-v", "--v"), "sW", 1),
        (("-l", "--l"), "dH", 1),
        (("-j", "--j"), "dW", 1),
        (("-g", "--g"), "groups", 1),
        (("-F", "--F"), "forw", 1),
    ):
        sub.add_argument(*flags, dest=dest, type=int, default=default)
    sub.add_argument("-in_layout", "--in_layout", dest="in_layout", default="NHWC")
    # Ignored flags -- consumed so they do not fail the parse.
    sub.add_argument("-m", "--m", dest="_mode", default="conv")
    sub.add_argument("-t", "--t", dest="_time", type=int, default=0)
    sub.add_argument("-V", "--V", dest="_verify", type=int, default=1)
    sub.add_argument("-_", "--_", dest="_spatial_dim", type=int, default=2)
    a, _ = sub.parse_known_args(tokens[driver_idx + 1 :])

    if a.in_layout.upper() not in ("NHWC", "NWC"):
        raise ValueError(
            f"Layout {a.in_layout!r} is not supported; only NHWC/NWC inputs are accepted"
        )

    from kernels.common.conv_implicit_gemm import ConvProblem

    problem = ConvProblem(
        N=a.N,
        Hi=a.Hi,
        Wi=a.Wi,
        C=a.C,
        K=a.K,
        Y=a.Y,
        X=a.X,
        sH=a.sH,
        sW=a.sW,
        pH=a.pH,
        pW=a.pW,
        dH=a.dH,
        dW=a.dW,
        groups=a.groups,
    )
    return problem, dtype, a.forw


def _runs_direction(forw: int, direction: str) -> bool:
    """MIOpen ``-F``: 0 runs every direction, bit 0 forward, bit 1 dgrad."""
    return forw == 0 or bool(forw & _FORW_BIT[direction])


def _parse_cases(args) -> list:
    """``[(ConvProblem, dtype), ...]`` from the input flags, or exit code 2."""
    from kernels.common.conv_implicit_gemm import ConvProblem

    cases: list = []
    if args.json_file is not None:
        import json

        entries = json.loads(Path(args.json_file).read_text())
        if not isinstance(entries, list):
            print(f"error: {args.json_file}: expected a JSON array", file=sys.stderr)
            return 2
        for idx, entry in enumerate(entries):
            if args.json_filter_suite and entry.get("suite") != args.json_filter_suite:
                continue
            if (
                args.json_filter_priority
                and entry.get("priority") != args.json_filter_priority
            ):
                continue
            try:
                cases.append(parse_json_case(entry))
            except ValueError as e:
                case_id = entry.get("case_id", f"#{idx + 1}")
                print(f"[warn] case {case_id}: skipping — {e}", file=sys.stderr)
    elif args.miopen_file is not None or args.miopen_cmd is not None:
        if args.miopen_file is not None:
            lines = Path(args.miopen_file).read_text().splitlines()
            where = args.miopen_file
        else:
            lines, where = [args.miopen_cmd], "--miopen-cmd"
        for lineno, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                problem, dtype, forw = parse_miopen_cmd(line)
            except ValueError as e:
                print(f"[warn] {where}:{lineno}: skipping — {e}", file=sys.stderr)
                continue
            if not _runs_direction(forw, args.direction):
                print(
                    f"[skip] {where}:{lineno}: -F {forw} has no {args.direction} "
                    f"pass (--direction {args.direction})",
                    file=sys.stderr,
                )
                continue
            cases.append((problem, dtype))
    else:
        try:
            problem = ConvProblem(
                N=args.N,
                Hi=args.Hi,
                Wi=args.Wi,
                C=args.C,
                K=args.K,
                Y=args.Y,
                X=args.X,
                sH=args.sH,
                sW=args.sW,
                pH=args.pH,
                pW=args.pW,
                dH=args.dH,
                dW=args.dW,
                groups=args.groups,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        cases.append((problem, args.dtype or "fp16"))
    if not cases:
        print("error: no valid cases found", file=sys.stderr)
        return 2
    return cases


# ---------------------------------------------------------------------------
# Instance selection
# ---------------------------------------------------------------------------


def _request_for(problem, dtype: str, arch: str, direction: str):
    from dispatch.grouped_convolution import ConvGroupedRequest
    from kernels.conv.kernel_identity import current_llvm_flavor

    # The flavor this process lowers -- and so loads binaries for -- is the
    # benchmark's to state; the dispatcher never probes for it.
    fields = dict(
        llvm_flavor=current_llvm_flavor(),
        direction=direction,
        N=problem.N,
        C=problem.C,
        K=problem.K,
        Hi=problem.Hi,
        Wi=problem.Wi,
        Y=problem.Y,
        X=problem.X,
        arch=arch,
        G=problem.groups,
        stride_h=problem.sH,
        stride_w=problem.sW,
        pad_h=problem.pH,
        pad_w=problem.pW,
        dilation_h=problem.dH,
        dilation_w=problem.dW,
        dtype=dtype,
    )
    if problem.is_3d:
        fields.update(
            Di=problem.Di,
            Z=problem.Z,
            stride_d=problem.sD,
            pad_d=problem.pD,
            dilation_d=problem.dD,
        )
    return ConvGroupedRequest(**fields)


def _filters(args):
    """The --algorithm / --instance predicate over instance specs."""
    algorithms = (
        {a.strip() for a in args.algorithm.split(",") if a.strip()}
        if args.algorithm
        else None
    )
    pattern = re.compile(args.instance) if args.instance else None

    def keep(spec) -> bool:
        if algorithms is not None and spec.algorithm not in algorithms:
            return False
        return pattern is None or pattern.search(spec.name) is not None

    return keep


@dataclass
class _Selection:
    """What one case runs: the dispatcher's admitted instances, filtered."""

    request: object
    results: list  # DispatchResult, in the dispatcher's priority order
    n_admitted: int
    default_pick: str | None  # the dispatcher's own choice, before filtering


def _select(args, problem, dtype: str, case_idx: int) -> _Selection:
    from dispatch.grouped_convolution import dispatch_conv_grouped_all

    req = _request_for(problem, dtype, args.arch, args.direction)
    admitted = list(dispatch_conv_grouped_all(req))
    keep = _filters(args)
    results = [r for r in admitted if keep(r.spec)]
    if args.sample is not None and results:
        n = max(1, round(len(results) * args.sample))
        rng = random.Random(args.seed + case_idx)
        chosen = set(rng.sample(range(len(results)), min(n, len(results))))
        results = [r for i, r in enumerate(results) if i in chosen]
    return _Selection(
        request=req,
        results=results,
        n_admitted=len(admitted),
        default_pick=admitted[0].spec.name if admitted else None,
    )


def _no_instance_reason(req) -> str:
    from dispatch.grouped_convolution import dispatch_conv_grouped

    try:
        dispatch_conv_grouped(req)
    except ValueError as e:
        return str(e)
    return "every admitted instance was filtered out (--algorithm/--instance)"


def _check_flavor(specs) -> int:
    """0 when this process lowers the flavor the instances were named for."""
    from kernels.conv.kernel_identity import current_llvm_flavor

    wanted = sorted({s.identity.llvm_flavor for s in specs})
    have = current_llvm_flavor()
    if wanted and have not in wanted:
        print(
            f"error: the registered instances are lowered for LLVM flavor "
            f"{'/'.join(wanted)}, this process lowers {have}: rerun with "
            f"ROCKE_LLVM_FLAVOR={wanted[0]}",
            file=sys.stderr,
        )
        return 2
    return 0


# ---------------------------------------------------------------------------
# AOT build
# ---------------------------------------------------------------------------


def _build_jobs(specs):
    """``{(dtype, family): [BuildJob]}`` for the instances, or raise LookupError.

    Each instance -- and each helper kernel it launches, once however many
    instances share it -- is rebuilt from the job ``--compile-all`` of its own
    family builds for its identity, so its binary is the one the implicit-GEMM
    and direct-conv caches would hold for that name.
    """
    from kernels.conv.aot_jobs import job_for_identity
    from kernels.conv.direct_aot_jobs import direct_job_for_identity

    groups: dict = {}
    seen: set = set()
    for spec in specs:
        for ident in (*spec.helpers, spec.identity):
            if ident in seen:
                continue
            seen.add(ident)
            job = (
                direct_job_for_identity(ident)
                if ident.is_direct
                else job_for_identity(ident)
            )
            groups.setdefault((ident.dtype_a, ident.is_direct), []).append(job)
    return groups


def _compile(cache, specs, arch: str, jobs: int, limit: int | None = None) -> int:
    """Bring ``cache`` up to date for ``specs``; compile only what changed."""
    from benchmarks.common.kernel_sweep import compile_jobs
    from kernels.conv.aot_jobs import kernel_for_job
    from kernels.conv.direct_aot_jobs import direct_kernel_for_job

    try:
        groups = _build_jobs(specs)
    except LookupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    rc = 0
    for (dtype, direct), all_jobs in sorted(groups.items()):
        family = "direct conv" if direct else "implicit GEMM"
        print(f"\n== {family} {dtype}: {len(all_jobs)} kernels", flush=True)
        rc = max(
            rc,
            compile_jobs(
                cache=cache,
                all_jobs=all_jobs,
                build=direct_kernel_for_job if direct else kernel_for_job,
                arch=arch,
                dtype=dtype,
                directions=sorted({j.identity.direction for j in all_jobs}),
                jobs=jobs,
                limit=limit,
            ),
        )
    return rc


def _compile_all(args) -> int:
    from benchmarks.common.kernel_cache import KernelCache
    from dispatch.grouped_convolution import registered_conv_grouped_instances

    keep = _filters(args)
    directions = (args.direction,) if args.direction else _DIRECTIONS
    specs = [
        s
        for d in directions
        for s in registered_conv_grouped_instances(args.arch, direction=d)
        if keep(s) and (args.dtype is None or s.dtype == args.dtype)
    ]
    if not specs:
        print(
            f"error: no registered {args.arch} instance matches the filters",
            file=sys.stderr,
        )
        return 2
    rc = _check_flavor(specs)
    if rc:
        return rc
    cache_dir = Path(args.cache_dir or "./kernel_cache")
    print(
        f"AOT build: {len(specs)} registered {args.arch} "
        f"{'/'.join(directions)} instances -> {cache_dir}",
        flush=True,
    )
    return _compile(
        KernelCache(cache_dir, args.arch), specs, args.arch, args.jobs, args.limit
    )


def _describe_cache(cache, specs) -> None:
    """How many of the kernels ``specs`` launch the cache holds, and how fresh."""
    from benchmarks.common.kernel_cache import current_emitter_digest

    digest = current_emitter_digest()
    binaries = {i for s in specs for i in (*s.helpers, s.identity)}
    present = [cache.meta(i) for i in binaries if cache.has(i)]
    stale = sum(1 for m in present if m.get("emitter_digest") != digest)
    print(
        f"  cache holds {len(present)} of the {len(binaries)} kernels these cases "
        f"launch" + (f" ({stale} built from older emitter sources)" if stale else ""),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class _Measured:
    name: str
    tflops: float
    gbps: float
    ms: float
    passed: bool | None


def _verify(rt, run, out_dev, out_t, ref_out, name: str, dump_fail, u8):
    """Launch once and compare with the reference: ``(stop, passed)``.

    Peak-normalised relative error, ``max|out - ref| / max(max|ref|, 1)``.
    With ``dump_fail`` the first failing kernel's output, reference and
    difference are written there and the run stops.
    """
    import torch

    from rocke.runtime import synchronize_and_release

    rt.memset(out_dev, 0, out_t.nbytes)
    run()
    synchronize_and_release(0)
    out_cpu = torch.empty_like(out_t)
    rt.memcpy_d2h(u8(out_cpu), out_dev, out_t.nbytes)
    out_f32 = out_cpu.float().cuda()
    diff = out_f32 - ref_out
    rel_err = float(diff.abs().max() / ref_out.abs().max().clamp(min=1.0))
    passed = rel_err < 5e-2
    if passed:
        return False, True
    print(f"  verify {name}: FAIL(rel_err={rel_err:.2e})", flush=True)
    if not dump_fail:
        return False, False
    import numpy as np

    from benchmarks.common.reproduce_kernel import artifact_stem

    dump_dir = Path(dump_fail)
    dump_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(name)
    for what, t in (("out", out_f32), ("ref", ref_out), ("diff", diff)):
        np.savetxt(dump_dir / f"{stem}_{what}.txt", t.cpu().numpy().ravel(), fmt="%.6f")
    flat = int(diff.abs().argmax())
    print(
        f"  [dump] saved to {dump_dir}/  max_diff at {np.unravel_index(flat, diff.shape)}: "
        f"out={float(out_f32.flatten()[flat]):.6f} ref={float(ref_out.flatten()[flat]):.6f}",
        flush=True,
    )
    return True, False


def _run_case(args, cache, problem, dtype: str, sel) -> int:
    import ctypes

    import numpy as np
    import torch

    from dispatch.grouped_convolution import launch_plan_for
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import HipError, Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def _u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

    p = problem
    direction = sel.request.direction
    torch_dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    torch.manual_seed(42)

    def _make(*shape):
        if args.debug_init is not None:
            return torch.full(shape, args.debug_init, dtype=torch_dt)
        return torch.empty(*shape).uniform_(-1.0, 1.0).to(torch_dt)

    x_shape = (p.N, p.Hi, p.Wi, p.C)
    y_shape = (p.N, p.Ho, p.Wo, p.K)
    W_t = _make(p.K, p.Y, p.X, p.C // p.groups)
    if direction == "fwd":
        in_name, out_name = "x", "y"
        In_t, Out_t = _make(*x_shape), torch.empty(*y_shape, dtype=torch_dt)
    else:
        in_name, out_name = "dy", "dx"
        In_t, Out_t = _make(*y_shape), torch.empty(*x_shape, dtype=torch_dt)
    flop = float(p.flops)
    bytes_xfer = float(In_t.nbytes + W_t.nbytes + Out_t.nbytes)

    ref_out = None
    if args.verify or args.dump_fail:
        from builders.common.conv_reference import conv_reference, dgrad_reference

        if direction == "fwd":
            ref_out = conv_reference(In_t.cuda(), W_t.cuda(), p, out_dtype=torch_dt)
        else:
            # Rounded to the output dtype, as conv_reference's out_dtype does.
            ref_out = dgrad_reference(In_t, W_t, p).to(torch_dt).float()

    plans = {r.spec.name: launch_plan_for(r.request, r.spec) for r in sel.results}
    rt = Runtime()
    sizes = {in_name: In_t.nbytes, "w": W_t.nbytes, out_name: Out_t.nbytes}
    # Scratch is sized for the largest plan and shared, as benchmark_direct_conv
    # does; it needs no initial contents.
    for plan in plans.values():
        for name, nbytes in plan.workspaces.items():
            sizes[name] = max(sizes.get(name, 0), nbytes)
    ptrs = {name: rt.alloc(nbytes) for name, nbytes in sizes.items()}
    rt.memcpy_h2d(ptrs[in_name], _u8(In_t), In_t.nbytes)
    rt.memcpy_h2d(ptrs["w"], _u8(W_t), W_t.nbytes)

    stop = EarlyStop.for_case(args, p, dtype, direction)
    measured: list[_Measured] = []
    n_missing = n_failed = 0
    n = len(sel.results)
    rc = 0
    try:
        for idx, result in enumerate(sel.results, 1):
            spec = result.spec
            plan = plans[spec.name]
            entries = [cache.get(step.identity) for step in plan.steps]
            if any(e is None for e in entries):
                n_missing += 1
                if n_missing <= 5:
                    absent = next(
                        s.identity for s, e in zip(plan.steps, entries) if e is None
                    )
                    print(
                        f"  [missing] {spec.name}: {absent.label()} not in the cache",
                        flush=True,
                    )
                continue
            if "sub_gemms" in ptrs:
                synchronize_and_release(0)
                rt.free(ptrs.pop("sub_gemms"))
            if plan.num_sub_gemms:
                # The tilde record table depends on the instance's tile.
                table = np.array(plan.sub_gemm_table, dtype=np.int32)
                ptrs["sub_gemms"] = rt.alloc(table.nbytes)
                rt.memcpy_h2d(ptrs["sub_gemms"], table.ctypes.data, table.nbytes)
            try:
                # Each step bound once: a timed loop that packs per launch
                # times the host, not a short kernel.
                bound = [
                    KernelLauncher(
                        hsaco=hsaco,
                        kernel_name=meta["kernel_name"],
                        signature=list(step.signature),
                    ).bind(
                        plan.values(step, ptrs, sizes),
                        config=LaunchConfig(grid=step.grid, block=step.block, stream=0),
                    )
                    for step, (hsaco, meta) in zip(plan.steps, entries)
                ]
            except (HipError, KeyError) as e:
                print(f"  [skip] {spec.name}: {e}", flush=True)
                continue

            def run(bound=bound):
                for launch in bound:
                    launch()

            passed = None
            if ref_out is not None:
                stopped, passed = _verify(
                    rt,
                    run,
                    ptrs[out_name],
                    Out_t,
                    ref_out,
                    spec.name,
                    args.dump_fail,
                    _u8,
                )
                if stopped:
                    return 1
                if not passed:
                    n_failed += 1
                    rc = 1
            try:
                ms = stop.measure(
                    run, warmup=args.warmup, iters=args.iters, stream=0, passed=passed
                )
            except (HipError, RuntimeError) as e:
                print(f"  [skip] {spec.name}: {e}", flush=True)
                continue
            if ms is None:
                stop.report(spec.name)
                continue
            m = _Measured(
                spec.name, flop / ms * 1e-9, bytes_xfer / ms * 1e-6, ms, passed
            )
            measured.append(m)
            mark = "*" if spec.name == sel.default_pick else " "
            print(
                f"  [{idx:>{len(str(n))}}/{n}]{mark}{m.tflops:7.1f} TFLOPS  "
                f"{ms:8.3f} ms  {spec.name}",
                flush=True,
            )
    finally:
        synchronize_and_release(0)
        for ptr in ptrs.values():
            rt.free(ptr)

    if n_missing:
        print(
            f"  {n_missing} admitted instances are not in the cache "
            f"(build them with --compile-all, or run without --run-from-cache)",
            flush=True,
        )
    if ref_out is not None:
        print(f"  verify: {n_failed} of the launched instances failed", flush=True)
    if not measured:
        print("  No successful launches.", flush=True)
        if stop.summary():
            print(f"  {stop.summary()}", flush=True)
        return 1

    # Only verified (or unverified) results rank: a wrong-but-fast kernel must
    # not be reported as the best.
    ranked = sorted(
        (m for m in measured if m.passed is not False), key=lambda m: -m.tflops
    )
    if not ranked:
        print("  every launched instance failed verification", flush=True)
        return 1
    top = min(args.top, len(ranked))
    print(
        f"\n  Top-{top} of {len(ranked)} ('*' = dispatcher's default pick):",
        flush=True,
    )
    print(
        f"  {'rank':>4}  {'TFLOPS':>7}  {'ms':>8}  {'GB/s':>7}  {'verify':>6}  instance"
    )
    for rank, m in enumerate(ranked[:top], 1):
        verdict = "-" if m.passed is None else ("PASS" if m.passed else "FAIL")
        mark = "*" if m.name == sel.default_pick else " "
        print(
            f"  {rank:>4}{mark} {m.tflops:>7.1f}  {m.ms:>8.3f}  {m.gbps:>7.1f}  "
            f"{verdict:>6}  {m.name}"
        )
    best = ranked[0]
    pick = next(
        (i for i, m in enumerate(ranked, 1) if m.name == sel.default_pick), None
    )
    if pick is not None:
        m = ranked[pick - 1]
        print(
            f"  dispatcher pick: rank {pick}/{len(ranked)}, {m.tflops:.1f} TFLOPS "
            f"({100.0 * m.tflops / best.tflops:.0f}% of best)",
            flush=True,
        )
    # The line benchmark_conv_compare.py reads, as the other conv benchmarks print.
    print(
        f"\nBest: {best.tflops:.1f} TFLOPS — {best.name}  {best.ms:.3f} ms", flush=True
    )
    return rc


def _list_case(sel) -> None:
    counts: dict[str, int] = {}
    for r in sel.results:
        counts[r.spec.algorithm] = counts.get(r.spec.algorithm, 0) + 1
    print(
        "  "
        + ", ".join(f"{a} {n}" for a, n in sorted(counts.items()))
        + f"  (default pick: {sel.default_pick})"
    )
    for r in sel.results:
        mark = "*" if r.spec.name == sel.default_pick else " "
        helpers = (
            f" (after {len(r.spec.helpers)} weight transforms)"
            if r.spec.helpers
            else ""
        )
        print(f"   {mark} {r.spec.name}  grid={r.grid} block={r.block}{helpers}")


# ---------------------------------------------------------------------------
# ckProfiler comparison
# ---------------------------------------------------------------------------

_CK_DTYPE_STR = {"fp32": "fp32", "fp16": "fp16", "bf16": "bfp16"}


def _run_ckprofiler(
    problem,
    dtype: str,
    direction: str,
    ckprofiler: str,
    converter_script: str,
    verify: int,
) -> None:
    """Run ckProfiler's ``direction`` pass on ``problem`` via convert_miopen_driver_to_profiler.py.

    Builds the args namespace the converter's argparse would produce and
    calls its ``init_const_args`` / ``run_ck_profiler`` directly.
    """
    import importlib.util
    import types

    ck_dtype_str = _CK_DTYPE_STR.get(dtype)
    if ck_dtype_str is None:
        print(
            f"[ckprofiler] dtype={dtype!r} is not supported; skipping", file=sys.stderr
        )
        return
    spec = importlib.util.spec_from_file_location("_ck_converter", converter_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = problem
    args = types.SimpleNamespace(
        ck_profiler_cmd=ckprofiler,
        data_type=ck_dtype_str,
        in_layout="NHWC",
        spatial_dim=2,
        batchsize=p.N,
        in_channels=p.C,
        out_channels=p.K,
        group_count=p.groups,
        in_h=p.Hi,
        in_w=p.Wi,
        fil_h=p.Y,
        fil_w=p.X,
        conv_stride_h=p.sH,
        conv_stride_w=p.sW,
        pad_h=p.pH,
        pad_w=p.pW,
        dilation_h=p.dH,
        dilation_w=p.dW,
        in_d=1,
        fil_d=1,
        conv_stride_d=1,
        pad_d=0,
        dilation_d=1,
        forw=_FORW_BIT[direction],
        verify=verify,
        time=1,
        instance=-1,
        list_instances=False,
    )
    mod.init_const_args(args)
    mod.run_ck_profiler(args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build (AOT) and benchmark the instances registered in the "
        "grouped-convolution dispatcher."
    )
    parser.add_argument("--arch", default="gfx950", help="gfx target (default: gfx950)")
    parser.add_argument(
        "--direction",
        choices=_DIRECTIONS,
        default=None,
        help="pass the cases run (default: fwd); with --compile-all, build only "
        "this direction's instances (default: all)",
    )
    parser.add_argument(
        "--dtype",
        choices=_DTYPES,
        default=None,
        help="data type of the shape-flag case (default: fp16); with --compile-all, "
        "build only this dtype's instances (default: all)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="timed iterations (default: 10)"
    )
    add_early_stop_arg(parser)
    parser.add_argument(
        "--top", type=int, default=10, help="print the top-N instances (default: 10)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check every instance against the torch reference before timing it; "
        "a failing instance never ranks",
    )
    parser.add_argument(
        "--dump-fail",
        default=None,
        metavar="PATH",
        dest="dump_fail",
        help="on the first verify FAIL, dump out/ref/diff to PATH and stop "
        "(implies --verify)",
    )
    parser.add_argument(
        "--debug-init",
        nargs="?",
        const=1.0,
        default=None,
        type=float,
        dest="debug_init",
        metavar="VALUE",
        help="initialise the inputs to a constant (1.0 when given without a value) "
        "instead of uniform random",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the instances the dispatcher admits for each case (after the "
        "filters) and exit; nothing is compiled or launched",
    )

    filt = parser.add_argument_group(
        "Instance filters",
        "Applied on top of the dispatcher's admission, to --compile-all as well "
        "as to the runs.",
    )
    filt.add_argument(
        "--algorithm",
        default=None,
        help="comma-separated instance families to keep (implicit_gemm, "
        "direct_nongrouped, direct_grouped, direct_depthwise, "
        "direct_depthwise_tiled, direct_depthwise_spatial, direct_depthwise_col; "
        "dgrad: direct_depthwise_dgrad, direct_grouped_dgrad_mfma)",
    )
    filt.add_argument(
        "--instance",
        default=None,
        metavar="REGEX",
        help="keep the instances whose name matches REGEX (re.search)",
    )
    filt.add_argument(
        "--sample",
        type=float,
        default=None,
        metavar="FRAC",
        help="run a random FRAC of each case's instances (e.g. 0.1)",
    )
    filt.add_argument(
        "--seed", type=int, default=0, help="RNG seed for --sample (default: 0)"
    )

    cache_grp = parser.add_argument_group(
        "AOT build and kernel cache",
        "The instances are shape-generic AOT binaries kept in a kernel cache "
        "(the format benchmark_implicit_gemm_conv.py and benchmark_direct_conv.py "
        "use, so one cache can serve all three).",
    )
    cache_grp.add_argument(
        "--compile-all",
        action="store_true",
        dest="compile_all",
        help="build every registered instance for --arch into --cache-dir and exit. "
        "No shape needed, no GPU used; parallelised with --jobs.",
    )
    cache_grp.add_argument(
        "--run-from-cache",
        default=None,
        metavar="DIR",
        dest="run_from_cache",
        help="run on the binaries in DIR; nothing is compiled",
    )
    cache_grp.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        dest="cache_dir",
        help="cache to build into: --compile-all defaults to ./kernel_cache; a run "
        "without --run-from-cache builds the instances its cases need here "
        "(default: a temporary directory, deleted at exit)",
    )
    cache_grp.add_argument(
        "--jobs",
        type=int,
        default=0,
        metavar="N",
        help="parallel compile workers (default: 0 = os.cpu_count())",
    )
    cache_grp.add_argument(
        "--limit",
        type=int,
        default=None,
        help="build at most this many out-of-date instances per dtype and family "
        "(smoke tests)",
    )

    miopen_grp = parser.add_argument_group(
        "MIOpen input",
        "Cases from MIOpenDriver commands whose -F includes --direction "
        "(0 runs every pass).",
    )
    miopen_grp.add_argument("--miopen-cmd", default=None, metavar="CMD")
    miopen_grp.add_argument(
        "--miopen-file",
        default=None,
        metavar="FILE",
        help="one MIOpenDriver command per line ('#' comments allowed)",
    )

    json_grp = parser.add_argument_group(
        "JSON input", "Cases from a JSON array, e.g. bench_cases_conv.json."
    )
    json_grp.add_argument("--json-file", default=None, metavar="FILE")
    json_grp.add_argument(
        "--json-filter-suite",
        default=None,
        metavar="SUITE",
        help='only entries whose "suite" is SUITE',
    )
    json_grp.add_argument(
        "--json-filter-priority",
        default=None,
        metavar="PRIORITY",
        help='only entries whose "priority" is PRIORITY',
    )

    ck_grp = parser.add_argument_group(
        "ckProfiler comparison",
        "Run ckProfiler's conv (the --direction pass) on each case before the "
        "instances.",
    )
    ck_grp.add_argument(
        "--ckprofiler", default=None, metavar="PATH", help="ckProfiler binary"
    )
    ck_grp.add_argument(
        "--ckprofiler-script",
        default=None,
        metavar="PATH",
        help="path to convert_miopen_driver_to_profiler.py",
    )

    conv = parser.add_argument_group("Shape", "2-D convolution shape (NHWC / KYXC)")
    conv.add_argument("--N", type=int, default=8, help="batch size")
    conv.add_argument("--Hi", type=int, default=56, help="input height")
    conv.add_argument("--Wi", type=int, default=56, help="input width")
    conv.add_argument("--C", type=int, default=64, help="input channels")
    conv.add_argument("--K", type=int, default=64, help="output channels")
    conv.add_argument("--Y", type=int, default=3, help="filter height")
    conv.add_argument("--X", type=int, default=3, help="filter width")
    conv.add_argument("--sH", type=int, default=1, help="vertical stride")
    conv.add_argument("--sW", type=int, default=1, help="horizontal stride")
    conv.add_argument("--pH", type=int, default=1, help="vertical padding")
    conv.add_argument("--pW", type=int, default=1, help="horizontal padding")
    conv.add_argument("--dH", type=int, default=1, help="vertical dilation")
    conv.add_argument("--dW", type=int, default=1, help="horizontal dilation")
    conv.add_argument(
        "--groups",
        "-g",
        type=int,
        default=1,
        help="conv groups; C and K must divide by it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.jobs <= 0:
        args.jobs = os.cpu_count() or 1
    if args.dump_fail:
        args.verify = True
    if args.sample is not None and not 0.0 < args.sample <= 1.0:
        print("error: --sample takes a fraction in (0, 1]", file=sys.stderr)
        return 2
    if args.ckprofiler is not None and args.ckprofiler_script is None:
        print("error: --ckprofiler requires --ckprofiler-script", file=sys.stderr)
        return 2

    if args.compile_all:
        return _compile_all(args)
    args.direction = args.direction or "fwd"

    cases = _parse_cases(args)
    if isinstance(cases, int):
        return cases

    from benchmarks.common.kernel_cache import KernelCache
    from dispatch.grouped_convolution import registered_conv_grouped_instances

    rc = _check_flavor(
        registered_conv_grouped_instances(args.arch, direction=args.direction)
    )
    if rc:
        return rc

    selections = [
        _select(args, problem, dtype, idx)
        for idx, (problem, dtype) in enumerate(cases, 1)
    ]

    if args.list:
        for idx, ((problem, dtype), sel) in enumerate(zip(cases, selections), 1):
            print(
                f"\nCase {idx}/{len(cases)}: {problem.short()} {dtype} {args.direction} — "
                f"{sel.n_admitted} admitted, {len(sel.results)} after filters"
            )
            if sel.results:
                _list_case(sel)
            else:
                print(f"  no instance: {_no_instance_reason(sel.request)}")
        return 0

    import tempfile

    tmp = None
    if args.run_from_cache:
        cache_dir = Path(args.run_from_cache)
    elif args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="rocke_grouped_conv_")
        cache_dir = Path(tmp.name)
    cache = KernelCache(cache_dir, args.arch)

    try:
        needed = {r.spec.name: r.spec for sel in selections for r in sel.results}
        if not args.run_from_cache and needed:
            print(
                f"Building the {len(needed)} instances these cases run into {cache_dir}",
                flush=True,
            )
            rc = _compile(cache, list(needed.values()), args.arch, args.jobs)
            if rc:
                return rc
        elif needed:
            print(f"Running from {cache_dir}:", flush=True)
            _describe_cache(cache, list(needed.values()))

        overall = 0
        for idx, ((problem, dtype), sel) in enumerate(zip(cases, selections), 1):
            print(f"\n{'#' * 72}", flush=True)
            print(
                f"# Case {idx}/{len(cases)}: {problem.short()} {dtype} {args.direction} — "
                f"{sel.n_admitted} instances admitted"
                + (
                    f", {len(sel.results)} after filters"
                    if len(sel.results) != sel.n_admitted
                    else ""
                )
                + (f"; default pick {sel.default_pick}" if sel.default_pick else ""),
                flush=True,
            )
            print(f"{'#' * 72}", flush=True)
            if args.ckprofiler is not None:
                _run_ckprofiler(
                    problem,
                    dtype,
                    args.direction,
                    args.ckprofiler,
                    args.ckprofiler_script,
                    int(args.verify),
                )
            if not sel.results:
                print(f"  no instance: {_no_instance_reason(sel.request)}", flush=True)
                overall = 1
                continue
            overall = max(overall, _run_case(args, cache, problem, dtype, sel))
        return overall
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
