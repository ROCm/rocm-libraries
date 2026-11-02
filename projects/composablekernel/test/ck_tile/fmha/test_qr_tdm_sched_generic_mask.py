# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device compile checks for qr_tdm_sched forward kernels with the generic mask.

The default codegen emits SimplifiedGenericAttentionMask kernels; generate.py -m generic
emits the same pipelines with GenericAttentionMask. The dedicated generic-forward suites
numerically check D64 N128 causal instances; this test also checks template compatibility
across the remaining generic pipeline paths. It generates the -m generic
qr_tdm_sched blobs and runs device-side semantic analysis (-fsyntax-only, so every template
is instantiated but no code is generated) on one blob per pipeline path. No GPU work is
launched.
"""

import argparse
import concurrent.futures
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# One blob per pipeline path that reaches the generic-mask predicate: the countdown loop
# (D64, D128, D192 N64), the index loop (D192 N128), K-padding, group mode and fp16. The
# value is (file name pattern, FmhaMasks alias of the blob): "mc" blobs use the causal form
# and "mg" blobs the windowed form of GenericAttentionMask.
# Pin both D64 tile widths so filename ordering cannot replace N64 causal coverage with N128.
_SELECTED_BLOBS = {
    "d64_countdown_window": (
        r"^fmha_fwd_d64_bf16_batch_b128x64x.*_npad_.*_nbias_mg_lse_",
        "GenericMask",
    ),
    "d64_padded_causal": (
        r"^fmha_fwd_d64_bf16_batch_b128x64x.*_pssk_.*_nbias_mc_lse_",
        "CausalMask",
    ),
    "d64_n128_padded_causal": (
        r"^fmha_fwd_d64_bf16_batch_b128x128x.*_pssk_.*_nbias_mc_lse_",
        "CausalMask",
    ),
    "d128_countdown_window": (
        r"^fmha_fwd_d128_bf16_batch_.*_nbias_mg_lse_",
        "GenericMask",
    ),
    "d128_group_causal": (r"^fmha_fwd_d128_bf16_group_.*_nbias_mc_lse_", "CausalMask"),
    "d192_n64_countdown_window": (
        r"^fmha_fwd_d192_bf16_batch_b128x64x.*_nbias_mg_lse_",
        "GenericMask",
    ),
    "d192_n128_index_causal": (
        r"^fmha_fwd_d192_bf16_batch_b128x128x.*_nbias_mc_lse_",
        "CausalMask",
    ),
    "d192_n64_fp16_causal": (
        r"^fmha_fwd_d192_fp16_batch_b128x64x.*_nbias_mc_nlse_",
        "CausalMask",
    ),
}
_ERROR_LINE = re.compile(r"^.*: (?:fatal )?error: .*$", re.MULTILINE)


class TestQrTdmSchedGenericMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (cls.ck_root / "include" / "ck_tile").is_dir():
            raise RuntimeError(f"CK headers not found under {cls.ck_root}")
        if not (cls.ck_build / "include" / "ck" / "config.h").is_file():
            raise RuntimeError(
                f"Configured ck/config.h not found under {cls.ck_build / 'include'}"
            )
        cls.directory = tempfile.TemporaryDirectory(prefix="ck_qr_tdm_sched_generic_")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.blob_dir = Path(cls.directory.name)
        example = cls.ck_root / "example" / "ck_tile" / "01_fmha"
        argv = [
            sys.executable,
            str(example / "generate.py"),
            "--targets",
            cls.gpu_target,
            "--api",
            "fwd",
            "--optdim",
            "64,128,192",
            "--filter",
            "*qr_tdm_sched*",
            "-m",
            "generic",
            "--output_dir",
            str(cls.blob_dir),
        ]
        result = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{shlex.join(argv)} failed:\n{result.stdout}")
        blobs = sorted(path.name for path in cls.blob_dir.glob("fmha_fwd_d*.cpp"))
        cls.selected = {}
        for name, (pattern, _) in _SELECTED_BLOBS.items():
            matches = [blob for blob in blobs if re.search(pattern, blob)]
            if not matches:
                raise RuntimeError(f"No -m generic blob matches {name}: {pattern}")
            cls.selected[name] = cls.blob_dir / matches[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=cls.jobs) as pool:
            futures = {
                name: pool.submit(cls.compile_blob, path)
                for name, path in cls.selected.items()
            }
            cls.results = {name: future.result() for name, future in futures.items()}

    @classmethod
    def compile_blob(cls, path):
        argv = [
            cls.compiler,
            "-x",
            "hip",
            "-std=c++20",
            f"--offload-arch={cls.gpu_target}",
            "--offload-device-only",
            "-fsyntax-only",
            "-fno-color-diagnostics",
            "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
            *("-D" + define for define in cls.defines),
            "-I" + str(cls.ck_build / "include"),
            "-I" + str(cls.ck_root / "include"),
            "-I" + str(cls.ck_root / "library" / "include"),
            "-I" + str(cls.ck_root / "example" / "ck_tile" / "01_fmha"),
            str(path),
        ]
        try:
            result = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=900,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Could not run {shlex.join(argv)}: {error}") from error
        result.stdout = f"Command: {shlex.join(argv)}\n" + result.stdout
        return result

    def check(self, name):
        source = self.selected[name].read_text(encoding="utf-8")
        # Guard against a codegen change that silently stops emitting the generic mask.
        alias = _SELECTED_BLOBS[name][1]
        self.assertIn(
            f"using fmha_mask = FmhaMasks::{alias};", source, self.selected[name].name
        )
        result = self.results[name]
        errors = "\n".join(_ERROR_LINE.findall(result.stdout)[:10])
        self.assertEqual(result.returncode, 0, errors or result.stdout[-4000:])

    def test_d64_countdown_window(self):
        self.check("d64_countdown_window")

    def test_d64_padded_causal(self):
        self.check("d64_padded_causal")

    def test_d64_n128_padded_causal(self):
        self.check("d64_n128_padded_causal")

    def test_d128_countdown_window(self):
        self.check("d128_countdown_window")

    def test_d128_group_causal(self):
        self.check("d128_group_causal")

    def test_d192_n64_countdown_window(self):
        self.check("d192_n64_countdown_window")

    def test_d192_n128_index_causal(self):
        self.check("d192_n128_index_causal")

    def test_d192_n64_fp16_causal(self):
        self.check("d192_n64_fp16_causal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler", required=True, help="Configured CMAKE_HIP_COMPILER"
    )
    parser.add_argument(
        "--ck-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--ck-build", required=True, type=Path, help="Configured CK build directory"
    )
    parser.add_argument(
        "--gpu-target", default="gfx1250", help="gfx125x target to compile for"
    )
    parser.add_argument(
        "--defines-file",
        type=Path,
        help="Configure-level compile definitions, one per line (the GPU feature macros)",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel compilations")
    arguments, unittest_arguments = parser.parse_known_args()
    TestQrTdmSchedGenericMask.compiler = arguments.compiler
    TestQrTdmSchedGenericMask.ck_root = arguments.ck_root.resolve()
    TestQrTdmSchedGenericMask.ck_build = arguments.ck_build.resolve()
    TestQrTdmSchedGenericMask.gpu_target = arguments.gpu_target
    TestQrTdmSchedGenericMask.jobs = max(1, arguments.jobs)
    TestQrTdmSchedGenericMask.defines = (
        [
            line.strip()
            for line in arguments.defines_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if arguments.defines_file
        else []
    )
    unittest.main(argv=[sys.argv[0], *unittest_arguments])
