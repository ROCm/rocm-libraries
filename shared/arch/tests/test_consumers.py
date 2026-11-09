# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise library target policies and canonical names against new entries."""

import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
ARCH = ROOT / "shared/arch"


class ConsumerTests(unittest.TestCase):
    def test_dependency_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "consumer"
            source.mkdir()
            consumer = (
                "include(rocm-arch-dependency)\n"
                "rocm_arch_list(names KIND PROCESSOR)\n"
                'if(NOT TARGET rocm::arch OR NOT "gfx942" IN_LIST names)\n'
                '  message(FATAL_ERROR "Missing target or catalog")\n'
                "endif()\n"
                "include(rocm-arch-dependency)\n"
            )
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.25.2)\n"
                "project(dependency_consumer LANGUAGES NONE)\n"
                "add_subdirectory(first)\nadd_subdirectory(second)\n",
                encoding="utf-8",
            )
            for directory in ("first", "second"):
                (source / directory).mkdir()
                (source / directory / "CMakeLists.txt").write_text(
                    consumer, encoding="utf-8"
                )

            def configure(label, module_path, *options):
                subprocess.run(
                    [
                        "cmake",
                        "-S",
                        str(source),
                        "-B",
                        str(root / label),
                        f"-DCMAKE_MODULE_PATH={module_path}",
                        *options,
                    ],
                    check=True,
                )

            configure(
                "source-build",
                ARCH / "cmake",
                "-DROCM_ARCH_BUILD_TESTING=OFF",
                "-DCMAKE_DISABLE_FIND_PACKAGE_rocm-arch=TRUE",
            )
            # The source dependency installs with its consumer.
            prefix = root / "installed"
            subprocess.run(
                [
                    "cmake",
                    "--install",
                    str(root / "source-build"),
                    "--prefix",
                    str(prefix),
                ],
                check=True,
            )
            configure(
                "source-installed-build",
                ARCH / "cmake",
                f"-DCMAKE_PREFIX_PATH={prefix}",
            )
            self.assertFalse((root / "source-installed-build/first/rocm-arch").exists())
            # Installed inclusion works with no source module, prefix search, or Python.
            configure(
                "installed-build",
                prefix / "share/rocm-arch/cmake",
                "-DCMAKE_DISABLE_FIND_PACKAGE_Python3=TRUE",
            )
            self.assertFalse((root / "installed-build/first/rocm-arch").exists())
            relocated = root / "relocated"
            prefix.rename(relocated)
            configure(
                "relocated-build",
                relocated / "share/rocm-arch/cmake",
                "-DCMAKE_DISABLE_FIND_PACKAGE_Python3=TRUE",
            )

    def test_additional_consumers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "python/rocm_arch"
            shutil.copytree(ARCH / "python/rocm_arch", package)
            standalone = root / "standalone"
            standalone.mkdir()
            shutil.copy(ROOT / "projects/rocrand/rmake.py", standalone / "rmake.py")
            prim_baseline = set(
                "gfx906:xnack- gfx908:xnack- gfx90a:xnack- gfx90a:xnack+ gfx942 gfx950 "
                "gfx1030 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1152 gfx1153 "
                "gfx1170 gfx1171 gfx1200 gfx1201 gfx1250".split()
            )
            rand_baseline = set(
                "gfx906:xnack- gfx1030 gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 "
                "gfx1152 gfx1153 gfx1170 gfx1171 gfx1200 gfx1201".split()
            )
            prim_source = (ROOT / "projects/rocprim/CMakeLists.txt").read_text()
            start = prim_source.index("      rocm_arch_list(_default_processors")
            end = prim_source.index(
                "      rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS TARGETS", start
            )
            policy = prim_source[start : prim_source.index("\n", end)]
            for added in (False, True):
                if added:
                    (package / "architectures/gfx9999.json").write_text(
                        json.dumps(
                            {"schema_version": 1, "id": "gfx9999", "kind": "processor"}
                        )
                    )
                output = root / "generated"
                subprocess.run(
                    [
                        sys.executable,
                        str(ARCH / "tools/generate.py"),
                        "--catalog",
                        str(package / "architectures"),
                        "--output",
                        str(output),
                    ],
                    check=True,
                )
                result = root / "targets.txt"
                script = root / "prim.cmake"
                script.write_text(
                    "cmake_minimum_required(VERSION 3.25.2)\n"
                    f'include("{output}/rocm-arch-catalog.cmake")\n'
                    f'include("{ARCH}/cmake/rocm-arch-functions.cmake")\n'
                    "function(rocm_check_target_ids output)\n"
                    '  cmake_parse_arguments(ARG "" "" "TARGETS" ${ARGN})\n'
                    '  set(${output} "${ARG_TARGETS}" PARENT_SCOPE)\n'
                    "endfunction()\n"
                    + policy
                    + f'\nfile(WRITE "{result}" "${{DEFAULT_AMDGPU_TARGETS}}")\n'
                )
                subprocess.run(["cmake", "-P", str(script)], check=True)
                self.assertEqual(
                    set(result.read_text().split(";")),
                    prim_baseline | ({"gfx9999"} if added else set()),
                )
                env = dict(os.environ, PYTHONPATH=str(root / "python"))
                python_check = (
                    "import sys; import rmake; sys.argv=['rmake']; "
                    f"assert set(rmake.parse_args().gpu_architecture.split(',')) == {rand_baseline | ({'gfx9999'} if added else set())!r}; "
                    "sys.argv=['rmake','-a','gfx942']; "
                    "assert rmake.parse_args().gpu_architecture == 'gfx942'; "
                )
                subprocess.run(
                    [sys.executable, "-c", python_check],
                    env=env,
                    cwd=standalone,
                    check=True,
                )
                cpp = root / "sparse.cpp"
                cpp.write_text(
                    "#include <rocsparse_arch.hpp>\n"
                    'static_assert(rocm::arch::equal(rocsparse::rocsparse_arch_names::gfx908, "gfx908"), "dispatch identity");\n'
                    + (
                        'static_assert(rocm::arch::equal(rocsparse::rocsparse_arch_names::gfx9999, "gfx9999"), "registration");\n'
                        if added
                        else ""
                    )
                    + "int main() {}\n"
                )
                subprocess.run(
                    [
                        shutil.which("c++") or "c++",
                        "-std=c++14",
                        "-I",
                        str(output / "include"),
                        "-I",
                        str(ROOT / "projects/rocsparse/library/src/include"),
                        str(cpp),
                        "-o",
                        str(root / "sparse"),
                    ],
                    check=True,
                )
                subprocess.run([str(root / "sparse")], check=True)

    def test_fft_catalog_adoption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "python/rocm_arch"
            shutil.copytree(ARCH / "python/rocm_arch", package)
            output = root / "generated"
            # Freeze the previous default set in the test, not in production.
            baseline = set(
                "gfx803 gfx900 gfx906 gfx908 gfx90a gfx90c gfx942 gfx950 gfx1030 "
                "gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1152 gfx1153 "
                "gfx1200 gfx1201 gfx1250".split()
            )
            for added in (False, True):
                if added:
                    (package / "architectures/gfx9999.json").write_text(
                        json.dumps(
                            {"schema_version": 1, "id": "gfx9999", "kind": "processor"}
                        ),
                        encoding="utf-8",
                    )
                subprocess.run(
                    [
                        sys.executable,
                        str(ARCH / "tools/generate.py"),
                        "--catalog",
                        str(package / "architectures"),
                        "--output",
                        str(output),
                    ],
                    check=True,
                )
                for project in ("rocfft", "hipfft"):
                    with self.subTest(project=project, added=added):
                        result = root / "result.txt"
                        script = root / "consumer.cmake"
                        cmake_source = (
                            ROOT / f"projects/{project}/CMakeLists.txt"
                        ).read_text()
                        policy = re.search(
                            r"^rocm_arch_list\(DEFAULT_GPUS[^\n]*\n"
                            r"list\(REMOVE_ITEM DEFAULT_GPUS[^\n]*\n",
                            cmake_source,
                            re.MULTILINE,
                        )
                        self.assertIsNotNone(
                            policy, "Missing inline FFT default policy"
                        )
                        script.write_text(
                            "cmake_minimum_required(VERSION 3.25.2)\n"
                            f'include("{output}/rocm-arch-catalog.cmake")\n'
                            f'include("{ARCH}/cmake/rocm-arch-functions.cmake")\n'
                            + policy.group(0)
                            + f'file(WRITE "{result}" "${{DEFAULT_GPUS}}")\n',
                            encoding="utf-8",
                        )
                        subprocess.run(["cmake", "-P", str(script)], check=True)
                        expected = baseline | ({"gfx9999"} if added else set())
                        self.assertEqual(set(result.read_text().split(";")), expected)
                # Launch a fresh interpreter so Python cannot reuse a cached module.
                env = dict(os.environ)
                env["PYTHONPATH"] = os.pathsep.join(
                    [
                        str(root / "python"),
                        str(ROOT / "projects/rocfft/library/src/device"),
                    ]
                )
                script = (
                    "from kernels.configs.config_arch import supported_arch as a; "
                    "import rocm_arch, pickle; "
                    "assert a.GFX_90A.value == 'gfx90a'; "
                    "assert a.GFX_GENERIC.value == 'gfx_generic'; "
                    "assert a.GFX_1250_STRICT.value == 'gfx1250-strict'; "
                    "assert pickle.loads(pickle.dumps(a.GFX_90A)) is a.GFX_90A; "
                    "assert {n.value for n in a} == set(rocm_arch.architectures()) | {'gfx_generic'}; "
                    f"assert hasattr(a, 'GFX_9999') == {added!r}"
                )
                subprocess.run([sys.executable, "-c", script], env=env, check=True)


if __name__ == "__main__":
    unittest.main()
