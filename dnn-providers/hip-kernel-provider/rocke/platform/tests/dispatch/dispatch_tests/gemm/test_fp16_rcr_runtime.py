# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU smoke tests for the CK DSL FP16 GEMM dispatcher."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from rocke.dispatch import GemmRequest, dispatch_gemm_fp16
from rocke.dispatch.gemm import build_kernel
from rocke.helpers import compile_kernel, make_gemm_manifest, write_artifact
from rocke.runtime.hip_module import get_device_arch
from rocke.tests.arch_marks import arch_specific
from rocke.tests.subprocess_launcher import (
    rocke_python_command,
    rocke_subprocess_env,
)

# Each runtime class builds+launches code objects for one arch, so it gates on the exact
# arch (via the rocke HIP runtime, no torch): on any other device the launch would fail
# with hipError(209) no kernel image.
_ARCH = get_device_arch(0)


def _build_manifest(result, work_dir: Path) -> tuple[Path, Path]:
    """Compile a dispatch result and write its HSACO + manifest under work_dir."""
    spec = result.spec
    artifact = compile_kernel(build_kernel(result), arch=result.request.arch)
    manifest = make_gemm_manifest(
        artifact=artifact,
        block_m=spec.tile.tile_m,
        block_n=spec.tile.tile_n,
        block_k=spec.tile.tile_k,
        threads_per_block=spec.block_size,
        default_shape=(128, 128, 64),
        warmup_iters=2,
        timed_iters=5,
        args_signature=list(result.signature),
        atoms=["dispatcher-fp16-gemm-runtime-test"],
        extra={"dispatcher_kernel_id": result.kernel_id.as_dict()},
    )
    out = work_dir / result.kernel_id.spec_id
    write_artifact(artifact, out, manifest)
    return out / f"{artifact.kernel_name}.hsaco", out / "manifest.json"


def _run_manifest(
    test: unittest.TestCase,
    hsaco: Path,
    manifest: Path,
    shape: tuple[int, int, int],
    *,
    verify: bool,
    timeout: int = 180,
) -> str:
    """Launch through rocke.run_manifest in a subprocess; fail the test on error."""
    cmd = rocke_python_command(
        "-m",
        "rocke.run_manifest",
        str(hsaco),
        str(manifest),
        "--shape",
        ",".join(str(x) for x in shape),
    )
    if verify:
        cmd.append("--verify")
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=rocke_subprocess_env(),
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    test.assertEqual(proc.returncode, 0, output[-2500:])
    return output


@unittest.skipUnless(_ARCH == "gfx950", "requires a gfx950 GPU")
class TestDispatchGemmRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        work_dir = Path(tempfile.mkdtemp(prefix="rocke_dispatch_gemm_test_"))
        result = dispatch_gemm_fp16(GemmRequest(M=4096, N=4096, K=4096, arch="gfx950"))
        cls.hsaco, cls.manifest = _build_manifest(result, work_dir)

    def _run_manifest(
        self, shape: tuple[int, int, int], *, verify: bool, timeout: int = 180
    ) -> str:
        return _run_manifest(
            self, self.hsaco, self.manifest, shape, verify=verify, timeout=timeout
        )

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


@arch_specific(
    ("gfx1201",),
    reason="the size-gated WMMA candidates are registered for gfx1201 only",
)
@unittest.skipUnless(_ARCH == "gfx1201", "requires a gfx1201 GPU")
class TestGfx1201DispatchGemmRuntime(unittest.TestCase):
    """Each size-gated gfx1201 candidate, as auto dispatch selects it, computes
    the full output exactly against the manifest runner's CPU reference."""

    def test_selected_candidates_verify_against_cpu_reference(self):
        work_dir = Path(tempfile.mkdtemp(prefix="rocke_dispatch_gfx1201_test_"))
        cases = (
            ((1024, 1024, 256), "gfx1201_wmma_128x128x16"),
            ((2048, 2048, 256), "gfx1201_wmma_128x64x32"),
            ((2112, 512, 256), "gfx1201_wmma_64x128x32"),
            ((2048, 2048, 272), "gfx1201_wmma_128x128x16_large"),
        )
        for (m, n, k), spec_id in cases:
            with self.subTest(shape=(m, n, k), spec_id=spec_id):
                result = dispatch_gemm_fp16(GemmRequest(M=m, N=n, K=k, arch="gfx1201"))
                self.assertEqual(result.kernel_id.spec_id, spec_id)
                hsaco, manifest = _build_manifest(result, work_dir)
                output = _run_manifest(self, hsaco, manifest, (m, n, k), verify=True)
                self.assertIn(f"bad=0/{m * n}", output)


if __name__ == "__main__":
    unittest.main()
