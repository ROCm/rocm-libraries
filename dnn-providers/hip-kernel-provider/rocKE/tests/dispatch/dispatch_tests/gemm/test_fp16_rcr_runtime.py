# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU smoke tests for the CK DSL FP16 GEMM dispatcher."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
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


@unittest.skipUnless(_has_rocm_gpu(), "requires a ROCm GPU")
class TestDispatchGemmRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.work_dir = Path(tempfile.mkdtemp(prefix="rocke_dispatch_gemm_test_"))
        req = GemmRequest(M=4096, N=4096, K=4096, arch="gfx950")
        cls.dispatch_result = dispatch_gemm_fp16(req)
        cls.spec = cls.dispatch_result.spec
        artifact = compile_kernel(build_kernel(cls.dispatch_result), arch=req.arch)
        manifest = make_gemm_manifest(
            artifact=artifact,
            block_m=cls.spec.tile.tile_m,
            block_n=cls.spec.tile.tile_n,
            block_k=cls.spec.tile.tile_k,
            threads_per_block=cls.spec.block_size,
            default_shape=(128, 128, 64),
            warmup_iters=2,
            timed_iters=5,
            args_signature=list(cls.dispatch_result.signature),
            atoms=["dispatcher-fp16-gemm-runtime-test"],
            extra={"dispatcher_kernel_id": cls.dispatch_result.kernel_id.as_dict()},
        )
        write_artifact(artifact, cls.work_dir, manifest)
        cls.hsaco = cls.work_dir / f"{artifact.kernel_name}.hsaco"
        cls.manifest = cls.work_dir / "manifest.json"

    def _run_manifest(
        self, shape: tuple[int, int, int], *, verify: bool, timeout: int = 180
    ) -> str:
        env = dict(os.environ)
        # Keep the package under test first even when this test is invoked from
        # a different working directory.
        py_root = str(Path(__file__).resolve().parents[4] / "Python")
        env["PYTHONPATH"] = py_root + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            sys.executable,
            "-m",
            "rocke.run_manifest",
            str(self.hsaco),
            str(self.manifest),
            "--shape",
            ",".join(str(x) for x in shape),
        ]
        if verify:
            cmd.append("--verify")
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        self.assertEqual(proc.returncode, 0, output[-2500:])
        return output

    def test_small_shapes_verify_against_cpu_reference(self):
        for shape in ((128, 128, 64), (256, 256, 128)):
            with self.subTest(shape=shape):
                out = self._run_manifest(shape, verify=True, timeout=120)
                self.assertIn("bad=0/", out)
                self.assertIn("Perf:", out)

    def test_large_example_shapes_launch_without_hang(self):
        # Shapes come from the gfx1151 GEMM README and gfx950 square GEMM study.
        # They are launch-only here because full CPU NumPy references for these
        # sizes are impractical in a smoke test.
        shapes = (
            (512, 512, 8192),
            (1024, 1024, 1024),
            (8192, 512, 512),
            (2048, 2048, 2048),
            (4096, 4096, 512),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                out = self._run_manifest(shape, verify=False, timeout=180)
                self.assertIn("Perf:", out)


if __name__ == "__main__":
    unittest.main()
