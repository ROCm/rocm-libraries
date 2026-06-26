# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Single-process rocprof-friendly harness for one CK DSL kernel.

The script:

1. accepts a Python callable spec (``--builder=<module:fn>``,
   ``--problem-json=...``);
2. builds the kernel once;
3. warms it up so the COMGR / module-load cost is paid;
4. enters a tight ``run_fn()`` loop for ``--iters`` invocations.

Drop ``rocprof`` / ``rocprofv3`` in front of the invocation and it will
attach to the kernel-execution window without your warmup and build
noise contaminating the trace.

Example (with rocprofv3 PMC counters):

    rocprofv3 -i metrics.txt -o run.csv --kernel-trace -- \\
        python probe_rocprof_single.py \\
        --builder rocke.examples.gfx950.attention.parity_unified_attention:run_one \\
        --problem-json /tmp/problem.json --iters 10

If you just want a CUDA-event-timed single-process run that simulates
what rocprof will see, use ``--no-rocprof``: the script still emits the
per-iter wall time so the kernel timing in this harness can be
cross-checked against rocprof's report.

This script is intentionally minimal. Real probes wire it to project-
specific builders. Treat it as a template.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


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


def _resolve_callable(dotted: str):
    """Resolve ``module.submodule:callable`` to a Python callable."""
    if ":" not in dotted:
        raise SystemExit(f"--builder must look like 'pkg.mod:fn', got {dotted!r}")
    module_name, attr = dotted.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def run(
    builder,
    problem: dict,
    iters: int = 10,
    warmup: int = 5,
    seed: int | None = None,
    sync: bool = True,
) -> dict:
    """Build once, warm up, then time ``iters`` invocations."""
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    runner = builder(problem)
    if not callable(runner):
        raise TypeError(
            f"builder {builder!r} returned {type(runner).__name__}, not callable"
        )

    for _ in range(warmup):
        runner()
    if sync:
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        runner()
    end.record()
    if sync:
        torch.cuda.synchronize()
    ms_per = start.elapsed_time(end) / iters
    print(
        f"warmup={warmup} iters={iters} mean={ms_per * 1000:.1f} us "
        f"(total {start.elapsed_time(end):.2f} ms)"
    )
    return {"iters": iters, "warmup": warmup, "ms_per_iter": ms_per}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--builder",
        required=True,
        help="dotted-path callable, e.g. mypkg.bench:make_runner; it must "
        "accept the parsed problem dict and return a no-arg callable.",
    )
    p.add_argument(
        "--problem-json",
        required=True,
        help="path to a JSON file describing the problem (passed verbatim "
        "to the builder)",
    )
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-sync",
        action="store_true",
        help="skip the trailing torch.cuda.synchronize() (useful if rocprof "
        "is already managing sync semantics)",
    )
    args = p.parse_args(argv)
    builder = _resolve_callable(args.builder)
    problem = json.loads(Path(args.problem_json).read_text())
    result = run(
        builder=builder,
        problem=problem,
        iters=args.iters,
        warmup=args.warmup,
        seed=args.seed,
        sync=not args.no_sync,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
