# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression coverage for MX build entry points and unsupported configurations."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

_DISP = Path(__file__).resolve().parents[1]
_CK = _DISP.parent
sys.path[:0] = [str(_DISP / "python"), str(_DISP / "codegen")]

import mx_gemm_utils as mx  # noqa: E402
import unified_mx_gemm_codegen as codegen  # noqa: E402
from dispatcher_common import arch_feature_defines, unified_framework_flags  # noqa: E402


class TestMxReviewRegressions(unittest.TestCase):
    def test_direct_tile_engine_listing_without_pythonpath(self):
        mx_dir = _CK / "tile_engine/ops/gemm/mx_gemm"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as tmp:
            for cwd in (mx_dir, Path(tmp)):
                with self.subTest(cwd=cwd):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(mx_dir / "mx_gemm_instance_builder.py"),
                            "--working_path",
                            tmp,
                            "--gpu_target",
                            "gfx1250:xnack-",
                            "--datatype",
                            "fp8",
                            "--layout",
                            "rcr",
                            "--list_kernels",
                            "--config_json",
                            str(mx_dir / "configs/default_ci_config_gfx1250.json"),
                        ],
                        cwd=cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertEqual(
                        (Path(tmp) / "mx_gemm_kernel_count.txt").read_text(), "8"
                    )

    def test_suffixed_targets_match_bare_codegen_and_defaults(self):
        for arch, target in (
            ("gfx950", "gfx950:sramecc+:xnack-"),
            ("gfx1250", "gfx1250:xnack-"),
        ):
            for pipeline in (
                None,
                "comp_async",
                "comp_async_eight_waves",
                "weight_preshuffle",
            ):
                with self.subTest(arch=arch, pipeline=pipeline):
                    bare = mx.default_fp8_config(arch, pipeline)
                    suffixed = mx.default_fp8_config(target, pipeline)
                    self.assertEqual(
                        bare.to_codegen_config(), suffixed.to_codegen_config()
                    )
                    # Bypass factory normalization to exercise direct config/codegen callers.
                    raw_config = replace(bare, gpu_target=target)
                    self.assertEqual(raw_config._fallback_name(), bare._fallback_name())
                    raw = bare.to_codegen_config()
                    raw["gpu_target"] = target
                    self.assertEqual(
                        codegen._generate(raw),
                        codegen._generate(bare.to_codegen_config()),
                    )
                    self.assertEqual(codegen.kernel_name(raw), bare.name)
                    self.assertEqual(raw["gpu_target"], target)

    def test_detection_and_setup_normalize_target_suffixes(self):
        for arch, target in (
            ("gfx950", "gfx950:sramecc+:xnack-"),
            ("gfx1250", "gfx1250:xnack-"),
        ):
            with self.subTest(arch=arch):
                with patch(
                    "mx_gemm_utils.subprocess.check_output",
                    return_value=f"Name: {target}\n",
                ):
                    self.assertEqual(mx._get_arch(), arch)
                for explicit in (None, target):
                    configs = [
                        replace(mx.default_fp8_config(arch), gpu_target=target),
                        mx.default_fp8_config(arch),
                    ]
                    with (
                        tempfile.TemporaryDirectory() as tmp,
                        patch(
                            "mx_gemm_utils._compile_kernel", return_value=True
                        ) as compile_kernel,
                    ):
                        results = mx.setup_multiple_mx_gemm_dispatchers(
                            configs, Path(tmp), gfx_arch=explicit, parallel=False
                        )
                        self.assertEqual(results[0], results[1])
                        self.assertIsNotNone(results[0])
                        self.assertEqual(compile_kernel.call_count, 1)
                        self.assertEqual(compile_kernel.call_args.args[2], arch)
                        self.assertTrue(all(cfg.gpu_target == arch for cfg in configs))
        with self.assertRaisesRegex(ValueError, "one architecture"):
            mx.setup_multiple_mx_gemm_dispatchers(
                [
                    replace(
                        mx.default_fp8_config("gfx950"), gpu_target="gfx950:xnack-"
                    ),
                    replace(
                        mx.default_fp8_config("gfx1250"), gpu_target="gfx1250:xnack-"
                    ),
                ]
            )

    def test_tdm_output_lds_boundary(self):
        for pipeline in ("comp_tdm", "comp_tdm_v2"):
            for factory in (mx.default_fp4_config, mx.default_fp8_config):
                for n, expected in ((320, True), (352, False), (512, False)):
                    cfg = replace(
                        factory("gfx1250", pipeline), tile_m=512, tile_n=n, tile_k=128
                    )
                    with self.subTest(pipeline=pipeline, dtype=cfg.datatype, n=n):
                        self.assertEqual(cfg.is_valid(), expected)
                        if expected:
                            codegen._validate(cfg.to_codegen_config())
                        else:
                            with self.assertRaises(ValueError):
                                codegen._validate(cfg.to_codegen_config())

    def test_gfx1250_filters_persistent_and_k_padding(self):
        config = _CK / "tile_engine/ops/gemm/mx_gemm/configs/default_config.json"
        with tempfile.TemporaryDirectory() as tmp:
            for arch in ("gfx950", "gfx1250", "gfx1250:xnack-"):
                builder = codegen._load_mx_builder()(
                    "mx_gemm", tmp, arch, "fp8", "rcr", str(config)
                )
                builder.config["trait_config"]["pad_k"]["values"] = [False, True]
                traits = builder._generate_trait_combinations()
                with self.subTest(arch=arch):
                    self.assertEqual(
                        {t[0] for t in traits},
                        {"comp_async", "comp_async_eight_waves", "weight_preshuffle"},
                    )
                    flags = {(t[5], t[6]) for t in traits}
                    self.assertEqual(
                        flags,
                        {(False, False), (False, True), (True, False), (True, True)}
                        if arch == "gfx950"
                        else {(False, False)},
                    )

    def test_standalone_compiler_uses_arch_feature_definitions(self):
        for arch in ("gfx950", "gfx1250"):
            with (
                self.subTest(arch=arch),
                patch("mx_gemm_utils._mx_codegen_flags", return_value=()),
                patch("mx_gemm_utils.subprocess.run") as run,
            ):
                run.return_value.returncode = 0
                self.assertTrue(
                    mx._compile_kernel(Path("kernel.hpp"), Path("kernel.so"), arch)
                )
                command = run.call_args.args[0]
                for flag in arch_feature_defines(arch) + unified_framework_flags(arch):
                    self.assertIn(flag, command)
                self.assertIn(f"--offload-arch={arch}", command)
                self.assertIn(f'-DGFX_ARCH="{arch}"', command)

    @unittest.skipUnless(
        shutil.which("cmake") and shutil.which("c++"),
        "requires CMake and a host C++ compiler",
    )
    def test_cmake_mx_target_and_effective_features(self):
        # Configure the real bindings CMake with an imported HIP target; no GPU is needed.
        for variable in ("GPU_TARGETS", "CK_TILE_GEMM_GPU_TARGET"):
            for arch in ("gfx950:sramecc+:xnack-", "gfx1250:xnack-", "gfx942"):
                for has_header in (False, True):
                    with (
                        self.subTest(variable=variable, arch=arch, header=has_header),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        tmp = Path(tmp)
                        build = tmp / "build"
                        if has_header:
                            headers = build / "generated_kernels"
                            headers.mkdir(parents=True)
                            (headers / "mx_gemm_test.hpp").touch()
                        (tmp / "CMakeLists.txt").write_text(
                            "cmake_minimum_required(VERSION 3.16)\n"
                            "project(mx_cmake_probe LANGUAGES CXX)\n"
                            "add_library(hip::device INTERFACE IMPORTED)\n"
                            # A conflicting inherited definition must not leak into this target.
                            "add_compile_definitions(CK_USE_FNUZ_FP8 CK_TILE_USE_WMMA=0)\n"
                            f'set({variable} "{arch}")\n'
                            f'add_subdirectory("{_DISP / "bindings/ctypes"}" ctypes)\n'
                        )
                        result = subprocess.run(
                            [
                                "cmake",
                                "-S",
                                str(tmp),
                                "-B",
                                str(build),
                                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        commands = json.loads(
                            (build / "compile_commands.json").read_text()
                        )
                        matches = [
                            c
                            for c in commands
                            if c["file"].endswith("/mx_gemm_ctypes_lib.cpp")
                        ]
                        base_arch = arch.split(":", 1)[0]
                        if base_arch == "gfx942":
                            self.assertFalse(matches)
                            continue
                        self.assertEqual(len(matches), 1)
                        command = shlex.split(matches[0]["command"])
                        self.assertIn(f'-DGFX_ARCH="{base_arch}"', command)
                        self.assertIn("-std=gnu++17", command)
                        self.assertEqual(
                            "-DCK_TILE_SINGLE_KERNEL_INCLUDE" in command, has_header
                        )
                        flags = [
                            arg for arg in command[1:] if arg.startswith(("-D", "-U"))
                        ]
                        preprocessed = subprocess.run(
                            [command[0], *flags, "-dM", "-E", "-x", "c++", "-"],
                            input="",
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        macros = {
                            line.split()[1]: " ".join(line.split()[2:])
                            for line in preprocessed.stdout.splitlines()
                        }
                        self.assertNotIn("CK_USE_FNUZ_FP8", macros)
                        for flag in arch_feature_defines(
                            base_arch
                        ) + unified_framework_flags(base_arch):
                            key, _, value = flag[2:].partition("=")
                            self.assertEqual(macros.get(key), value or "1", key)


if __name__ == "__main__":
    unittest.main()
