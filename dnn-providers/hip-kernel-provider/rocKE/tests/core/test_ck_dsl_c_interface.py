# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CI harness for the ck_dsl C interface.

The C interface lives in ``python/ck_dsl_c`` and ships C parity emitters that
mirror Python emitters. These tests keep all orchestration in ``python/test``:

* non-GPU: build ``ckc_core`` and compare selected C/Python LLVM emitters;
* GPU: compile a C-emitted GEMM LLVM module to HSACO and verify it through the
  shared ``ck_dsl.run_manifest`` harness on gfx950.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from ck_dsl.helpers import make_gemm_manifest, write_artifact

_ROCKE = Path(__file__).resolve().parents[2]  # core -> tests -> rocKE
_PY_ROOT = _ROCKE / "Python"
_CKC_ROOT = _ROCKE / "Cpp"
_DEFAULT_BUILD_DIR = Path("/tmp/ck_dsl_c_ci_build")
_LLVM_KERNEL_RE = re.compile(r"define\s+amdgpu_kernel\s+void\s+@([A-Za-z_.$][\w.$]*)")
_DATALAYOUT_RE = re.compile(r'^target datalayout = ".+"$', re.MULTILINE)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PY_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(cmd: list[str | os.PathLike[str]], *, cwd: Path = _ROCKE, timeout: int = 240):
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "command failed\n"
            f"cmd: {' '.join(str(x) for x in cmd)}\n"
            f"rc: {proc.returncode}\n"
            f"stdout:\n{proc.stdout[-2500:]}\n"
            f"stderr:\n{proc.stderr[-2500:]}"
        )
    return proc.stdout


def _build_dir() -> Path:
    return Path(os.environ.get("CK_DSL_C_BUILD_DIR", _DEFAULT_BUILD_DIR))


def _configure_and_build() -> Path:
    build = _build_dir()
    _run(["cmake", "-S", _CKC_ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release"])
    _run(
        [
            "cmake",
            "--build",
            build,
            "--target",
            "ckc_smoke",
            "ckc_emit",
            "ckc_gemm_emit",
            "ckc_attention_unified_emit",
            "ckc_fused_moe_emit",
            "-j",
            os.environ.get("CK_DSL_C_BUILD_JOBS", "8"),
        ],
        timeout=360,
    )
    return build


@dataclass
class CkcArtifact:
    kernel_name: str
    llvm_text: str
    hsaco: bytes
    timings: dict[str, float] = field(default_factory=dict)
    ir_text: str = ""

    @property
    def hsaco_bytes(self) -> int:
        return len(self.hsaco)


def _kernel_name_from_llvm(llvm_text: str) -> str:
    match = _LLVM_KERNEL_RE.search(llvm_text)
    if not match:
        raise AssertionError("could not find amdgpu_kernel definition in C-emitted LLVM")
    return match.group(1)


def _normalize_llvm_for_branch_compare(llvm_text: str) -> str:
    """Ignore LLVM datalayout spelling drift between source branches.

    The C-interface branch carries generated emitters that target an older
    datalayout string. For CI here, the useful contract is that the C API builds
    the same kernels and emits the same body-level LLVM as the current Python
    emitter; the standalone IR parity harness owns exact datalayout checks.
    """

    return _DATALAYOUT_RE.sub('target datalayout = "<normalized>"', llvm_text)


def _detect_gpu_arch() -> tuple[bool, str | None, str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None, "torch reports no ROCm GPU"
        props = torch.cuda.get_device_properties(0)
        return True, props.gcnArchName.split(":", 1)[0], torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, None, f"torch GPU detection failed: {exc!r}"


class TestCkDslCInterfaceStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = _configure_and_build()

    def _c_emit(self, exe: str, *args: str) -> str:
        return _run([self.build / exe, *args], timeout=240)

    def _py_emit(self, script: str, *args: str) -> str:
        return _run([sys.executable, _CKC_ROOT / "tests" / "parity" / script, *args])

    def test_smoke_executable_runs(self):
        out = self._c_emit("ckc_smoke")
        self.assertIn("builder lifecycle + LLVM lower symbols resolved", out)

    def test_core_ir_parity_emitters_match_python(self):
        for case in ("scalar", "memory", "forloop", "vector"):
            with self.subTest(case=case):
                self.assertEqual(
                    _normalize_llvm_for_branch_compare(self._py_emit("emit.py", case)),
                    _normalize_llvm_for_branch_compare(self._c_emit("ckc_emit", case)),
                )

    def test_representative_instance_emitters_match_python(self):
        samples = (
            ("gemm_emit.py", "ckc_gemm_emit", ("0",)),
            ("attention_unified_emit.py", "ckc_attention_unified_emit", ("4",)),
            ("fused_moe_emit.py", "ckc_fused_moe_emit", ("1", "gather")),
            ("fused_moe_emit.py", "ckc_fused_moe_emit", ("1", "topk_weighted_reduce")),
        )
        for py_script, c_exe, args in samples:
            with self.subTest(c_exe=c_exe, args=args):
                self.assertEqual(
                    _normalize_llvm_for_branch_compare(self._py_emit(py_script, *args)),
                    _normalize_llvm_for_branch_compare(self._c_emit(c_exe, *args)),
                )


GPU_AVAILABLE, GPU_ARCH, GPU_REASON = _detect_gpu_arch()


@unittest.skipUnless(
    GPU_AVAILABLE and GPU_ARCH == "gfx950",
    f"needs a gfx950 ROCm GPU; detected {GPU_ARCH or GPU_REASON}",
)
class TestCkDslCInterfaceGfx950(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = _configure_and_build()

    def test_c_emitted_gemm_runs_on_gfx950(self):
        from ck_dsl.runtime.comgr import build_hsaco_from_llvm_ir

        llvm_text = _run([self.build / "ckc_gemm_emit", "0"], timeout=240)
        kernel_name = _kernel_name_from_llvm(llvm_text)
        hsaco, timings = build_hsaco_from_llvm_ir(
            llvm_text,
            isa="amdgcn-amd-amdhsa--gfx950",
            options=["-O3"],
        )
        artifact = CkcArtifact(
            kernel_name=kernel_name,
            llvm_text=llvm_text,
            hsaco=hsaco,
            timings={
                "comgr_bc": timings.bc * 1000.0,
                "comgr_relocatable": timings.relocatable * 1000.0,
                "comgr_executable": timings.executable * 1000.0,
            },
        )

        with tempfile.TemporaryDirectory(prefix="ckdsl_c_gemm_gfx950_") as tmp:
            out_dir = Path(tmp)
            manifest = make_gemm_manifest(
                artifact=artifact,
                block_m=128,
                block_n=128,
                block_k=32,
                threads_per_block=256,
                default_shape=(512, 512, 512),
                warmup_iters=2,
                timed_iters=10,
                atoms=["ck-dsl-c-interface", "gemm-fp16-rcr"],
                notes="C interface emitted LLVM compiled and launched via run_manifest",
            )
            paths = write_artifact(
                artifact,
                out_dir,
                manifest,
                write_ir_text=False,
                write_llvm_text=True,
            )
            result = _run(
                [
                    sys.executable,
                    "-m",
                    "ck_dsl.run_manifest",
                    paths["hsaco"],
                    paths["manifest"],
                    "--shape",
                    "512,512,512",
                    "--verify",
                ],
                timeout=240,
            )
        self.assertIn("bad=0", result)
        self.assertIn("Perf:", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
