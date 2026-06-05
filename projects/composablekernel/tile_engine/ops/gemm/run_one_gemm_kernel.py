#!/usr/bin/env python3
"""Worker script for running GEMM kernels in an isolated subprocess.

Mirrors grouped_conv's run_one_grouped_conv_kernel.py:
- Receives kernel config + problem via stdin as JSON
- Loads the .so library ONLY inside this subprocess
- Outputs timing results as JSON to stdout (one line per kernel, flushed)
- A GPU fault kills only this process; the parent driver can continue

Input JSON format:
    Single: {"so_path": "...", "problem": {"M":.., "N":.., "K":..}, "kernel_name": "..."}
    Batch:  {"items": [{"so_path": "...", "problem": {...}, "kernel_name": "..."}, ...]}

Output JSON format (one line per kernel):
    {"idx": 0, "ok": true, "ms": 0.123, "tflops": 456.7, "non_zero": 1, "kernel": "..."}
    {"idx": 1, "ok": false, "error": "...", "kernel": "..."}
"""

import json
import os
import sys

# Add dispatcher python paths from environment (os.pathsep-separated).
gemm_pypath = os.environ.get("GEMM_PYPATH", "")
if gemm_pypath:
    for p in gemm_pypath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)

from gemm_utils import GemmProblem, GpuGemmRunner  # noqa: E402
import numpy as np  # noqa: E402


def _run_one(idx, so_path, prob_dict, kernel_name):
    """Run a single kernel and emit its result as one JSON line."""
    try:
        problem = GemmProblem.from_dict(prob_dict)

        np.random.seed(42)
        A = (np.random.randn(problem.M, problem.K) * 0.1).astype(np.float16)
        B = (np.random.randn(problem.K, problem.N) * 0.1).astype(np.float16)

        # CRITICAL: load the library ONLY inside this subprocess.
        runner = GpuGemmRunner(lib_path=so_path)
        result = runner.run(A, B, problem)

        if result.success:
            non_zero = (
                int(np.count_nonzero(result.output))
                if result.output is not None
                else 0
            )
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "ok": True,
                        "ms": result.time_ms,
                        "tflops": result.tflops,
                        "non_zero": non_zero,
                        "kernel": kernel_name,
                    }
                ),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {
                        "idx": idx,
                        "ok": False,
                        "error": f"kernel returned status {result.status}",
                        "kernel": kernel_name,
                    }
                ),
                flush=True,
            )

    except Exception as e:
        print(
            json.dumps(
                {"idx": idx, "ok": False, "error": str(e), "kernel": kernel_name}
            ),
            flush=True,
        )


def main():
    """Read JSON from stdin, run kernel(s), output results."""
    try:
        d = json.loads(sys.stdin.buffer.read())
    except Exception as e:
        print(
            json.dumps({"idx": 0, "ok": False, "error": f"JSON parse error: {e}"}),
            flush=True,
        )
        sys.exit(1)

    if "items" in d:
        for i, item in enumerate(d["items"]):
            _run_one(
                i, item["so_path"], item["problem"], item.get("kernel_name", "unknown")
            )
    else:
        _run_one(0, d["so_path"], d["problem"], d.get("kernel_name", "unknown"))


if __name__ == "__main__":
    main()
