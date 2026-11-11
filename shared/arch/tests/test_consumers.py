# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise real FFT policy modules and Python configuration against a new entry."""

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
