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


def _has_rocm_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _compile_and_run(req: GemmRequest, shape: tuple[int, int, int]) -> str:
    result = dispatch_gemm_fp16(req)
    artifact = compile_kernel(build_kernel(result), arch=req.arch)
    spec = result.spec
    work_dir = Path(tempfile.mkdtemp(prefix="rocke_dispatch_parallel_gemm_"))
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
    env = dict(os.environ)
    py_root = str(Path(__file__).resolve().parents[4] / "Python")
    env["PYTHONPATH"] = py_root + os.pathsep + env.get("PYTHONPATH", "")
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


@unittest.skipUnless(_has_rocm_gpu(), "requires a ROCm GPU")
class TestGemmParallelRuntime(unittest.TestCase):
    def test_threaded_compile_and_run(self):
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
