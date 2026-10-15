# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MLA decode live driver (gfx942, gfx950): resolve, build, verify, time.

Every shape goes ``MLAShape -> MLARequest -> dispatch_mla``; the spec, the
split-KV count and the launch geometry are the dispatcher's, never hand-rolled
here, so the driver measures and verifies exactly what ships.

``--mode resolve`` (default, CPU)
    Shape -> candidate, kernel name(s), grid, split-KV count.
``--mode build`` (CPU)
    Resolve, then build every selected kernel's IR.
``--mode verify`` (GPU)
    The whole op per shape -- ``torch.bmm`` absorb, the decode kernel(s),
    ``torch.bmm`` ``W_UV`` -- against the fp64 reference, through
    :func:`builders.mla.verify_mla_decode.run_case`. Sequence lengths are
    jittered below each shape's ``seqlen_k`` so most end in a page tail.
``--mode time`` (GPU)
    HIP-event timing of the four stages separately -- [A] absorb, [B]+[C] the
    decode kernel(s), [D] ``W_UV`` -- and end to end, with the split-KV count
    used. Numbers are printed, not recorded: absolute measurements belong in
    the access-controlled record, not in this repository.

Run, from ``rocke/library``::

    python -m benchmarks.common.benchmark_mla_decode_live              # resolve
    python -m benchmarks.common.benchmark_mla_decode_live --mode verify --model Kimi
    python -m benchmarks.common.benchmark_mla_decode_live --mode time --arch gfx950

``--shapes`` defaults to ``benchmarks/<arch>/attention/decode/mla_shapes.json``;
only its bf16 ``seqlen_q == 1`` entries are taken (the fp8 rows are a later
phase).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.common.mla_shapes import MLAShape, filter_shapes, load_mla_shapes

_BENCH_ROOT = Path(__file__).resolve().parent.parent


def _default_shapes(arch: str) -> Path:
    return _BENCH_ROOT / arch / "attention" / "decode" / "mla_shapes.json"


def _decode_shapes(path: Path, *, model: str | None, limit: int | None):
    shapes = [s for s in load_mla_shapes(path) if s.dtype == "bf16" and s.seqlen_q == 1]
    return filter_shapes(shapes, model=model, limit=limit)


def _resolve(shape: MLAShape, *, arch: str, build: bool) -> dict:
    """One shape through dispatch; never raises, so one bad shape hides nothing."""
    from dispatch.mla import dispatch_mla, mla_decode_launch

    row = {"shape": shape.signature}
    try:
        result = dispatch_mla(shape.to_request(arch=arch))
    except ValueError as exc:
        return {**row, "status": "unsupported", "detail": str(exc)}
    row.update(
        status="ok",
        candidate=result.candidate.name,
        kernels=list(mla_decode_launch(result).kernel_names()),
        grid=result.grid,
        num_seg=result.spec.num_seg,
    )
    if build:
        try:
            result.candidate.built(result.spec, arch)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            row.update(status="build_failed", detail=f"{type(exc).__name__}: {exc}")
    return row


def _report(rows: list[dict], arch: str) -> int:
    width = max((len(r["shape"]) for r in rows), default=0)
    for r in rows:
        if r["status"] == "ok":
            print(
                f"{r['shape']:<{width}}  {r['candidate']}  grid={r['grid']} "
                f"num_seg={r['num_seg']}  {' + '.join(r['kernels'])}"
            )
        else:
            print(f"{r['shape']:<{width}}  {r['status'].upper()}: {r['detail']}")
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\nSUMMARY {ok}/{len(rows)} shapes resolved on {arch}")
    return 0 if ok == len(rows) else 1


def _verify(shapes, *, arch: str, seed: int) -> int:
    from builders.mla.verify_mla_decode import ATOL, run_case, uniform_problem

    failed = []
    for s in shapes:
        p = uniform_problem(
            heads=s.num_query_heads, batch=s.batch, seqlen_k=s.seqlen_k, seed=seed
        )
        if not run_case(
            s.signature, p, mode="full", num_segs=["auto"], arch=arch, atol=ATOL
        ):
            failed.append(s.signature)
    print(
        f"\nSUMMARY {len(shapes) - len(failed)}/{len(shapes)} shapes passed on {arch}"
    )
    for sig in failed:
        print(f"  FAILED {sig}")
    return 1 if failed else 0


def _time(shapes, *, arch: str, warmup: int, iters: int, seed: int) -> int:
    import torch
    from builders.mla.mla_decode_runner import MlaDecodeRunner
    from builders.mla.verify_mla_decode import device_inputs, to_device, uniform_problem
    from dispatch.mla import dispatch_mla
    from rocke.runtime import time_launches

    stream = int(torch.cuda.current_stream().cuda_stream)
    print(f"device: {torch.cuda.get_device_name(0)}  arch: {arch}")
    print(f"warmup: {warmup}  iters: {iters}  (HIP events, per call)")
    for s in shapes:
        p = uniform_problem(
            heads=s.num_query_heads, batch=s.batch, seqlen_k=s.seqlen_k, seed=seed
        )
        runner = MlaDecodeRunner(dispatch_mla(s.to_request(arch=arch)))
        dev = device_inputs(p)
        q_nope = to_device(p.q_nope, torch.bfloat16)
        q_rope = to_device(p.q_rope, torch.bfloat16)
        w_uk = to_device(p.w_uk, torch.bfloat16)
        d_nope, r_kv = q_nope.shape[-1], runner.spec.r_kv
        qa = torch.empty(
            q_nope.shape[0],
            q_nope.shape[1],
            runner.spec.qa_cols,
            dtype=torch.bfloat16,
            device="cuda",
        )
        latent = torch.empty(
            q_nope.shape[0], q_nope.shape[1], r_kv, dtype=torch.bfloat16, device="cuda"
        )
        lse = torch.empty(q_nope.shape[:2], dtype=torch.float32, device="cuda")
        out = torch.empty(
            q_nope.shape[0],
            q_nope.shape[1],
            w_uk.shape[2] - d_nope,
            dtype=torch.bfloat16,
            device="cuda",
        )

        def absorb():
            runner.absorb(q_nope=q_nope, q_rope=q_rope, w_uk=w_uk, out=qa)

        def attend():
            runner.latent(
                qa=qa, scale=p.scale, out=latent, lse=lse, stream=stream, **dev
            )

        def project():
            runner.project(latent=latent, w_uk=w_uk, d_nope=d_nope, out=out)

        def whole():
            absorb()
            attend()
            project()

        stages = {
            name: time_launches(fn, warmup=warmup, iters=iters, stream=stream)
            for name, fn in (
                ("absorb", absorb),
                ("decode", attend),
                ("w_uv", project),
                ("total", whole),
            )
        }
        print(
            f"{s.signature}  num_seg={runner.spec.num_seg}  "
            + "  ".join(f"{k}={v * 1e3:.1f}us" for k, v in stages.items()),
            flush=True,
        )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode", choices=("resolve", "build", "verify", "time"), default="resolve"
    )
    ap.add_argument(
        "--arch", default=None, help="default: the local device, else gfx942"
    )
    ap.add_argument("--shapes", type=Path, default=None)
    ap.add_argument("--model", default=None, help="substring match on the model name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    # Before anything compiles, with or without --arch: torch binds its
    # bundled comgr, which the IR flavor is resolved against.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    arch = args.arch
    if arch is None:
        from rocke.runtime.hip_module import get_device_arch

        arch = (get_device_arch() or "gfx942").split(":")[0]
    shapes = _decode_shapes(
        args.shapes or _default_shapes(arch), model=args.model, limit=args.limit
    )
    if not shapes:
        print("no bf16 decode shapes matched the filters", file=sys.stderr)
        return 1
    if args.mode in ("resolve", "build"):
        rows = [_resolve(s, arch=arch, build=args.mode == "build") for s in shapes]
        return _report(rows, arch)
    if args.mode == "verify":
        return _verify(shapes, arch=arch, seed=args.seed)
    if args.iterations <= 0:
        print("--iterations must be positive", file=sys.stderr)
        return 1
    return _time(
        shapes, arch=arch, warmup=args.warmup, iters=args.iterations, seed=args.seed
    )


if __name__ == "__main__":
    raise SystemExit(main())
