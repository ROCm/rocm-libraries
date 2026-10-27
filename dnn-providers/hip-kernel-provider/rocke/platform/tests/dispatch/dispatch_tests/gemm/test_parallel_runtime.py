# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Threaded compile+run smoke for the GEMM dispatcher."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rocke.dispatch import GemmRequest, dispatch_gemm_fp16
from rocke.dispatch.gemm import build_kernel
from rocke.helpers import compile_kernel, make_gemm_manifest, write_artifact
from rocke.runtime.hip_module import get_device_arch


# The runtime bodies below build+launch a gfx950 code object, so gate on the exact arch
# (via the rocke HIP runtime, no torch): on any other device the launch would fail with
# hipError(209) no kernel image.
_ARCH = get_device_arch(0)


# Runs one dispatched kernel through its binding (``DispatchResult.bind``), the
# dispatcher's launch recipe for every body (split-K workspaces, skinny bodies,
# swizzled grids), in a fresh process: re-dispatch the same request, check the
# first launch against the reference, then time a few launches.
_BOUND_RUN = """
import sys, time
from rocke.dispatch import GemmRequest, dispatch_gemm_fp16
from rocke.runtime.hip_module import Runtime
hsaco, kernel_name, arch, spec_id = sys.argv[1:5]
M, N, K = (int(x) for x in sys.argv[5].split(","))
result = dispatch_gemm_fp16(GemmRequest(M=M, N=N, K=K, arch=arch, spec_id=spec_id))
rt = Runtime()
fn = rt.load_module(open(hsaco, "rb").read()).get_function(kernel_name)
binding = result.bind(verify=True)
args, ptrs = binding.make_args(rt)
rt.launch_blocking(fn, binding.grid, binding.block, args)
_max_abs, bad, total = binding.check(rt, ptrs)
print(f"bad={bad}/{total}")
t0 = time.perf_counter()
for _ in range(3):
    rt.launch_blocking(fn, binding.grid, binding.block, args)
print(f"Perf: {(time.perf_counter() - t0) / 3 * 1e3:.4f} ms")
sys.exit(1 if bad else 0)
"""


def _compile_and_run(req: GemmRequest, shape: tuple[int, int, int]) -> str:
    result = dispatch_gemm_fp16(req)
    artifact = compile_kernel(build_kernel(result), arch=req.arch)
    spec = result.spec
    work_dir = Path(tempfile.mkdtemp(prefix="rocke_dispatch_parallel_gemm_"))
    env = dict(os.environ)
    py_root = str(Path(__file__).resolve().parents[4] / "python")
    env["PYTHONPATH"] = py_root + os.pathsep + env.get("PYTHONPATH", "")
    if req.spec_id == "auto":
        hsaco = work_dir / f"{artifact.kernel_name}.hsaco"
        hsaco.write_bytes(artifact.hsaco)
        cmd = [
            sys.executable,
            "-c",
            _BOUND_RUN,
            str(hsaco),
            artifact.kernel_name,
            req.arch,
            req.spec_id,
            ",".join(str(x) for x in shape),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=180, env=env)
        output = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode:
            raise RuntimeError(output[-2500:])
        return output
    manifest = make_gemm_manifest(
        artifact=artifact,
        block_m=spec.tile.tile_m,
        block_n=spec.tile.tile_n,
        block_k=spec.tile.tile_k,
        threads_per_block=spec.block_size,
        default_shape=shape,
        warmup_iters=1,
        timed_iters=3,
        args_signature=list(result.signature),
        atoms=["dispatcher-fp16-gemm-parallel-runtime-test"],
        extra={"dispatcher_kernel_id": result.kernel_id.as_dict()},
    )
    write_artifact(artifact, work_dir, manifest)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "rocke.run_manifest",
            str(work_dir / f"{artifact.kernel_name}.hsaco"),
            str(work_dir / "manifest.json"),
            "--shape",
            ",".join(str(x) for x in shape),
            "--verify",
        ],
        text=True,
        capture_output=True,
        timeout=180,
        env=env,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode:
        raise RuntimeError(output[-2500:])
    return output


@unittest.skipUnless(_ARCH == "gfx950", "requires a gfx950 GPU")
class TestGemmParallelRuntime(unittest.TestCase):
    def test_threaded_compile_and_run(self):
        # the auto pick runs through its binding (any body the dispatcher
        # chooses); the pinned universal candidate through the manifest runner
        jobs = (
            (GemmRequest(M=128, N=128, K=64, arch="gfx950"), (128, 128, 64)),
            (
                GemmRequest(
                    M=128,
                    N=128,
                    K=64,
                    arch="gfx950",
                    spec_id="cdna_mem_64x128",
                ),
                (128, 128, 64),
            ),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_compile_and_run, req, shape) for req, shape in jobs]
            for future in as_completed(futures):
                out = future.result()
                self.assertIn("bad=0/", out)
                self.assertIn("Perf:", out)


if __name__ == "__main__":
    unittest.main()
