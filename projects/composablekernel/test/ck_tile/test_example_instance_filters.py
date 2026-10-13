# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only checks: python3 test/ck_tile/test_example_instance_filters.py -v."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


CK_ROOT = Path(__file__).resolve().parents[2]
LAYERNORM = CK_ROOT / "example/ck_tile/02_layernorm2d"
GEMM = CK_ROOT / "example/ck_tile/38_block_scale_gemm"


class ExampleInstanceFilters(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="ck-instance-filters-")
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)

    def run_command(self, command, succeeds=True):
        result = subprocess.run(command, text=True, capture_output=True)
        output = result.stdout + result.stderr
        if succeeds:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
        return output

    def generate(self, directory, pattern=None, mode="--list_blobs", succeeds=True):
        command = [
            sys.executable,
            str(LAYERNORM / "generate.py"),
            "--api",
            "fwd",
            "--working_path",
            str(directory),
            mode,
        ]
        if pattern is not None:
            command += ["--filter", pattern]
        return self.run_command(command, succeeds)

    def listed_names(self, directory):
        return {
            Path(line).name
            for line in (directory / "layernorm2d_fwd_blobs.txt")
            .read_text()
            .splitlines()
        }

    def test_layernorm_default_empty_and_wildcard_keep_all_instances(self):
        manifests = []
        for index, pattern in enumerate((None, "", "*")):
            directory = self.work / str(index)
            self.generate(directory, pattern)
            manifests.append(self.listed_names(directory))
        self.assertGreater(len(manifests[0]), 3)
        self.assertEqual(manifests[0], manifests[1])
        self.assertEqual(manifests[0], manifests[2])

    def test_layernorm_exact_filter_keeps_manifest_and_dispatcher_in_sync(self):
        directory = self.work / "exact"
        pattern = "layernorm2d_fwd_fp16_n256"
        self.generate(directory, pattern)
        self.generate(directory, pattern, "--gen_blobs")
        self.assertEqual(
            self.listed_names(directory),
            {
                pattern + ".cpp",
                "layernorm2d_fwd_api.cpp",
                "layernorm2d_fwd_api_common.hpp",
            },
        )
        self.assertEqual(
            self.listed_names(directory),
            {p.name for p in directory.iterdir() if p.suffix in (".cpp", ".hpp")},
        )
        dispatcher = (directory / "layernorm2d_fwd_api.cpp").read_text()
        self.assertIn('t.prec_i == "fp16"', dispatcher)
        self.assertIn("a.n <= 256", dispatcher)
        self.assertNotIn('t.prec_i == "bf16"', dispatcher)
        self.assertNotIn("a.n <= 512", dispatcher)

    def test_layernorm_wildcard_selects_multiple_instances(self):
        directory = self.work / "wildcard"
        self.generate(directory, "layernorm2d_fwd_*_n256")
        names = self.listed_names(directory)
        self.assertIn("layernorm2d_fwd_fp16_n256.cpp", names)
        self.assertIn("layernorm2d_fwd_bf16_n256.cpp", names)
        self.assertNotIn("layernorm2d_fwd_fp16_n512.cpp", names)
        self.assertNotIn("layernorm2d_fwd_fp16_n256_pras.cpp", names)

    def test_layernorm_rejects_unmatched_filter_in_both_modes(self):
        for mode in ("--list_blobs", "--gen_blobs"):
            with self.subTest(mode=mode):
                output = self.generate(
                    self.work / "invalid", "no_such_kernel", mode, succeeds=False
                )
                self.assertIn("matches no layernorm2d kernel", output)

    def cmake_fixture(self, example, target):
        if not shutil.which("cmake"):
            self.skipTest("cmake is required")
        source = self.work / "fixture"
        source.mkdir()
        for path in example.glob("*.cpp"):
            shutil.copyfile(path, source / path.name)
        # Include the real example CMake in a small project. Only generation and
        # source selection are exercised, so no HIP compiler or GPU is needed.
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(instance_filters LANGUAGES CXX)\n"
            f'set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
            "set(GPU_TARGETS gfx942)\n"
            f'include("{example.as_posix()}/CMakeLists.txt")\n'
            f"get_target_property(sources {target} SOURCES)\n"
            'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/sources.txt" '
            'CONTENT "$<JOIN:${sources},\\n>\\n")\n'
            "if(LAYERNORM2D_FWD_GEN_BLOBS)\n"
            "  add_custom_target(generate_instances DEPENDS ${LAYERNORM2D_FWD_GEN_BLOBS})\n"
            "endif()\n"
        )
        return source, self.work / "build"

    def configure(self, source, build, variable, pattern, succeeds=True):
        return self.run_command(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                f"-D{variable}={pattern}",
            ],
            succeeds,
        )

    def test_gemm_cmake_source_selection_and_reset(self):
        source, build = self.cmake_fixture(GEMM, "tile_example_gemm_quant")
        variable = "GEMM_QUANT_INSTANCE_FILTER"
        self.configure(source, build, variable, "")
        default_sources = (build / "sources.txt").read_text().splitlines()
        self.assertGreater(len(default_sources), 3)
        self.configure(source, build, variable, "^gemm_bquant_quantgrouped_fp8$")
        self.assertEqual(
            set((build / "sources.txt").read_text().splitlines()),
            {"gemm_quant.cpp", "gemm_bquant_quantgrouped_fp8.cpp"},
        )
        self.configure(source, build, variable, "^gemm_bquant_quantgrouped_(fp8|bf8)$")
        self.assertEqual(
            set((build / "sources.txt").read_text().splitlines()),
            {
                "gemm_quant.cpp",
                "gemm_bquant_quantgrouped_fp8.cpp",
                "gemm_bquant_quantgrouped_bf8.cpp",
            },
        )
        self.configure(source, build, variable, "")
        self.assertEqual(
            (build / "sources.txt").read_text().splitlines(), default_sources
        )
        output = self.configure(
            source, build, variable, "no_such_source", succeeds=False
        )
        self.assertIn("matches no instance source", output)

    def test_layernorm_cmake_regenerates_dispatcher_when_reusing_filter(self):
        source, build = self.cmake_fixture(LAYERNORM, "tile_example_layernorm2d_fwd")
        for size in (256, 512, 256):
            self.configure(
                source,
                build,
                "LAYERNORM2D_FWD_INSTANCE_FILTER",
                f"layernorm2d_fwd_fp16_n{size}",
            )
            self.run_command(
                ["cmake", "--build", str(build), "--target", "generate_instances"]
            )
            dispatcher = (build / "layernorm2d_fwd_api.cpp").read_text()
            self.assertIn(f"a.n <= {size}", dispatcher)
            self.assertNotIn(f"a.n <= {768 - size}", dispatcher)


if __name__ == "__main__":
    unittest.main()
