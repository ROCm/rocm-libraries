# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Direct CUDA-event timing across a list of shape configs.

The caller supplies:

- a ``baseline_fn(shape)``: optional reference callable timed
  alongside the candidate (e.g. Triton);
- a ``candidate_fn(shape)``: the CK DSL kernel to evaluate;
- a list of shape dicts ``SHAPES`` — passed verbatim to the two
  callables.

Both ``baseline_fn`` and ``candidate_fn`` must launch *one* kernel
each per call, on the current torch stream. The harness runs the
warmup, the synchronize, the CUDA-event window, and the per-shape
report.

The reason this script exists separately from ``rocke.runtime.time_launches``
is that ``time_launches`` calls ``torch.cuda.synchronize()`` between
calls, which on Triton interacts badly with `tunableop` autotuning the
first time the function is invoked. The direct CUDA-event window
records one ``start`` / ``end`` pair around the full ``iters`` loop,
which is the methodology AITER/vLLM uses for production benchmarking
and avoids the Triton autotune artifact.

Programmatic use:

    from probe_targeted_bench import bench_shapes

    def candidate(shape):
        run_unified_attention_torch(... shape ...)

    def baseline(shape):
        unified_attention(... shape ..., backend="triton")

    bench_shapes(SHAPES, candidate_fn=candidate, baseline_fn=baseline)

CLI demo (no GPU work — just exercises the shape iterator):

    python probe_targeted_bench.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


def _bootstrap_rocke() -> None:
    try:
        import rocke  # noqa: F401

        return
    except ImportError:
        pass
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Python"
        if (candidate / "rocke" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return
        candidate = parent / "rocke" / "__init__.py"
        if candidate.exists():
            sys.path.insert(0, str(parent))
            return


_bootstrap_rocke()


def time_cuda_event(
    fn: Callable[[], Any],
    *,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """Time ``fn`` with a single CUDA-event window. Returns microseconds/iter."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def bench_shapes(
    shapes: Iterable[tuple[str, dict] | dict],
    *,
    candidate_fn: Callable[[dict], None],
    baseline_fn: Callable[[dict], None] | None = None,
    warmup: int = 5,
    iters: int = 20,
    candidate_label: str = "candidate",
    baseline_label: str = "baseline",
) -> list[dict]:
    """Time candidate and baseline across each shape; report a tidy table."""
    rows: list[dict] = []
    print(
        f"{'shape':<40} {candidate_label + ' (us)':>16} "
        f"{baseline_label + ' (us)':>16} {'speedup':>10}"
    )
    print("-" * 86)
    for entry in shapes:
        if isinstance(entry, tuple):
            name, shape = entry
        else:
            name = entry.get("name", repr(entry))  # type: ignore[union-attr]
            shape = entry
        row = {"name": name}
        try:
            row["candidate_us"] = time_cuda_event(
                lambda s=shape: candidate_fn(s),
                warmup=warmup,
                iters=iters,
            )
        except Exception as e:  # noqa: BLE001
            row["candidate_us"] = None
            row["candidate_error"] = f"{type(e).__name__}: {e}"
        if baseline_fn is not None:
            try:
                row["baseline_us"] = time_cuda_event(
                    lambda s=shape: baseline_fn(s),
                    warmup=warmup,
                    iters=iters,
                )
            except Exception as e:  # noqa: BLE001
                row["baseline_us"] = None
                row["baseline_error"] = f"{type(e).__name__}: {e}"
        else:
            row["baseline_us"] = None
        c = row.get("candidate_us")
        b = row.get("baseline_us")
        sx = (b / c) if (c and b) else None
        line = f"{name:<40} "
        line += f"{c:>16.2f} " if c is not None else f"{'FAIL':>16} "
        if baseline_fn is not None:
            line += f"{b:>16.2f} " if b is not None else f"{'FAIL':>16} "
            line += f"{sx:>9.2f}x" if sx is not None else f"{'-':>10}"
        else:
            line += " " * 16 + f"{'-':>10}"
        print(line)
        rows.append(row)
    return rows


# ---- Demo --------------------------------------------------------------

DEMO_SHAPES = [
    ("bf16_decode_n16_kv1k", dict(nseq=16, q_len=1, kv_len=1024, fp8=False, sw=0)),
    (
        "bf16_decode_n16_kv1k_sw127",
        dict(nseq=16, q_len=1, kv_len=1024, fp8=False, sw=128),
    ),
    (
        "bf16_prefill_n16_q256_kv256",
        dict(nseq=16, q_len=256, kv_len=256, fp8=False, sw=0),
    ),
    (
        "bf16_prefill_n2_q1024_kv1024",
        dict(nseq=2, q_len=1024, kv_len=1024, fp8=False, sw=0),
    ),
    (
        "bf16_prefill_n16_q1024_kv1024",
        dict(nseq=16, q_len=1024, kv_len=1024, fp8=False, sw=0),
    ),
    (
        "bf16_prefill_n16_q1024_kv4096",
        dict(nseq=16, q_len=1024, kv_len=4096, fp8=False, sw=0),
    ),
    (
        "fp8_prefill_n16_q1024_kv4096",
        dict(nseq=16, q_len=1024, kv_len=4096, fp8=True, sw=0),
    ),
    ("fp8_decode_n16_kv1k", dict(nseq=16, q_len=1, kv_len=1024, fp8=True, sw=0)),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="just print the shape list and exit without GPU work",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args(argv)
    if args.dry_run:
        print(
            "Demo shape catalog (n=8 entries, suitable for vLLM-style prefill traces):"
        )
        for name, shape in DEMO_SHAPES:
            print(f"  {name:<40}  {shape}")
        print("\nWire candidate_fn and baseline_fn via the public API to run.")
        return 0
    print(
        "ERROR: this script needs candidate_fn/baseline_fn callables; "
        "import bench_shapes and pass them in."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
