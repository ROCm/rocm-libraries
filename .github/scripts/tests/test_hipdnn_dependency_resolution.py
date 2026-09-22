# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise production dependency resolution with local CMake consumer fixtures.

Requires CMake >= 3.25.2 and a C++ toolchain, but no upstream dependencies.
Network isolation, when required, must be provided by the test runner.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULES = {
    "hipdnn": REPO_ROOT / "projects/hipdnn/cmake/Dependencies.cmake",
    "provider": REPO_ROOT / "dnn-providers/cmake/Dependencies.cmake",
}
# Package names deliberately differ from FetchContent names in two cases.
DEPENDENCIES = {
    "GTest": ("GOOGLETEST", "GTest", ("gtest", "gmock")),
    "flatbuffers": ("FLATBUFFERS", "flatbuffers", ("flatbuffers",)),
    "spdlog": ("SPDLOG", "spdlog", ("spdlog",)),
    "nlohmann_json": ("JSON", "nlohmann_json", ("nlohmann_json",)),
}


class DependencyResolutionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hipdnn-deps-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cmake = shutil.which("cmake")
        self.assertIsNotNone(self.cmake, "CMake is required for these regression tests")

    def run_command(self, *command):
        return subprocess.run(
            [str(arg) for arg in command],
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout[-8000:])

    def make_fixture(self, directory, package, origin, installed=False):
        """Supply a tiny compiled source library or an installed header-only package."""
        directory.mkdir(parents=True)
        _, namespace, targets = DEPENDENCIES[package]
        declarations = []
        cmake = ["cmake_minimum_required(VERSION 3.25.2)"]
        if not installed:
            cmake.append("project(dependency_fixture LANGUAGES CXX)")
        for target in targets:
            if installed:
                declarations.append(
                    f'inline const char* {target}_origin() {{ return "{origin}"; }}'
                )
                cmake.extend(
                    [
                        f"add_library({namespace}::{target} INTERFACE IMPORTED)",
                        f"set_target_properties({namespace}::{target} PROPERTIES "
                        'INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}")',
                    ]
                )
            else:
                declarations.append(f"const char* {target}_origin();")
                (directory / f"{target}.cpp").write_text(
                    '#include "fixture.h"\n'
                    f'const char* {target}_origin() {{ return "{origin}"; }}\n',
                    encoding="utf-8",
                )
                cmake.extend(
                    [
                        f"add_library({target} STATIC {target}.cpp)",
                        f"add_library({namespace}::{target} ALIAS {target})",
                        f"target_include_directories({target} PUBLIC "
                        '"${CMAKE_CURRENT_SOURCE_DIR}")',
                    ]
                )
        (directory / "fixture.h").write_text(
            "#pragma once\n" + "\n".join(declarations) + "\n", encoding="utf-8"
        )
        filename = f"{package}Config.cmake" if installed else "CMakeLists.txt"
        (directory / filename).write_text("\n".join(cmake) + "\n", encoding="utf-8")
        return directory

    def make_consumer(self, name, resolver, package="GTest", installed=None):
        directory = self.root / name
        directory.mkdir()
        _, namespace, targets = DEPENDENCIES[package]
        lookup = f"set(CMAKE_DISABLE_FIND_PACKAGE_{package} ON)"
        if installed is not None:
            # Use a real package config, without consulting ambient registries,
            # system prefixes or environment-provided package roots.
            lookup = "\n".join(
                f"set(CMAKE_FIND_USE_{kind} OFF)"
                for kind in (
                    "PACKAGE_ROOT_PATH",
                    "CMAKE_ENVIRONMENT_PATH",
                    "SYSTEM_ENVIRONMENT_PATH",
                    "CMAKE_SYSTEM_PATH",
                    "INSTALL_PREFIX",
                    "PACKAGE_REGISTRY",
                    "SYSTEM_PACKAGE_REGISTRY",
                )
            )
            lookup += f'\nset(CMAKE_PREFIX_PATH "{installed.as_posix()}")'
        invocation = "fetch_gtest_dependency()"
        if resolver == "hipdnn":
            # A deliberately unavailable fallback version must not supersede
            # either a supplied source or a matching installed package.
            version = "" if installed is not None else " VERSION 999.0.0"
            invocation = f"hipdnn_add_dependency({package}{version})"
        links = " ".join(f"{namespace}::{target}" for target in targets)
        (directory / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.25.2)\n"
            "project(dependency_consumer LANGUAGES CXX)\n"
            f"{lookup}\n"
            f'include("{MODULES[resolver].as_posix()}")\n'
            f"{invocation}\n"
            "add_executable(consumer main.cpp)\n"
            f"target_link_libraries(consumer PRIVATE {links})\n"
            'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer-$<CONFIG>.txt" '
            'CONTENT "$<TARGET_FILE:consumer>")\n',
            encoding="utf-8",
        )
        statements = "\n".join(
            f"    std::puts({target}_origin());" for target in targets
        )
        (directory / "main.cpp").write_text(
            '#include "fixture.h"\n#include <cstdio>\n'
            f"int main() {{\n{statements}\n}}\n",
            encoding="utf-8",
        )
        return directory

    def configure(self, consumer, **options):
        return self.run_command(
            self.cmake,
            "-S",
            consumer,
            "-B",
            consumer / "build",
            "-DCMAKE_BUILD_TYPE=Release",
            *(f"-D{key}={value}" for key, value in options.items()),
        )

    def assert_origin(self, consumer, package, origin):
        build = consumer / "build"
        self.assert_success(
            self.run_command(
                self.cmake,
                "--build",
                build,
                "--config",
                "Release",
                "--target",
                "consumer",
            )
        )
        executable = (build / "consumer-Release.txt").read_text(encoding="utf-8")
        result = self.run_command(executable)
        self.assert_success(result)
        self.assertEqual(
            result.stdout.splitlines(), [origin] * len(DEPENDENCIES[package][2])
        )

    def assert_missing(self, result, package):
        self.assertNotEqual(result.returncode, 0, result.stdout[-8000:])
        # Require the resolver's named policy error, not a toolchain or download
        # failure that happens to make configuration fail too.
        self.assertIn(f"{package} was not found", result.stdout)
        self.assertIn("ALLOW_FETCH_DEPS=ON", result.stdout)

    def test_explicit_sources_work_with_fetching_disabled(self):
        cases = [
            ("hipdnn", "GTest", {}),
            ("provider", "GTest", {}),
            ("provider", "GTest", {"ALLOW_FETCH_DEPS": "OFF"}),
            *(
                ("hipdnn", package, {"ALLOW_FETCH_DEPS": "OFF"})
                for package in DEPENDENCIES
            ),
        ]
        for index, (resolver, package, options) in enumerate(cases):
            with self.subTest(resolver=resolver, package=package, options=options):
                source = self.make_fixture(
                    self.root / f"source-{index}", package, "source"
                )
                consumer = self.make_consumer(f"consumer-{index}", resolver, package)
                options = {
                    **options,
                    f"FETCHCONTENT_SOURCE_DIR_{DEPENDENCIES[package][0]}": source.as_posix(),
                }
                self.assert_success(self.configure(consumer, **options))
                self.assert_origin(consumer, package, "source")

    def test_installed_package_precedes_conflicting_source(self):
        source = self.make_fixture(self.root / "source", "GTest", "source")
        installed = self.make_fixture(
            self.root / "prefix", "GTest", "installed", installed=True
        )
        for resolver in MODULES:
            with self.subTest(resolver=resolver):
                consumer = self.make_consumer(resolver, resolver, installed=installed)
                self.assert_success(
                    self.configure(
                        consumer,
                        ALLOW_FETCH_DEPS="OFF",
                        FETCHCONTENT_SOURCE_DIR_GOOGLETEST=source.as_posix(),
                    )
                )
                self.assert_origin(consumer, "GTest", "installed")

    def test_missing_or_nonexistent_source_fails_closed(self):
        for resolver in MODULES:
            for supplied in (False, True):
                with self.subTest(resolver=resolver, supplied=supplied):
                    consumer = self.make_consumer(f"{resolver}-{supplied}", resolver)
                    options = {"ALLOW_FETCH_DEPS": "OFF"}
                    if supplied:
                        options["FETCHCONTENT_SOURCE_DIR_GOOGLETEST"] = (
                            self.root / "nonexistent"
                        ).as_posix()
                    self.assert_missing(self.configure(consumer, **options), "GTest")

    def test_removed_source_is_not_reused_on_reconfigure(self):
        for resolver in MODULES:
            with self.subTest(resolver=resolver):
                source = self.make_fixture(
                    self.root / f"{resolver}-source", "GTest", "source"
                )
                consumer = self.make_consumer(resolver, resolver)
                self.assert_success(
                    self.configure(
                        consumer,
                        ALLOW_FETCH_DEPS="OFF",
                        FETCHCONTENT_SOURCE_DIR_GOOGLETEST=source.as_posix(),
                    )
                )
                self.assert_origin(consumer, "GTest", "source")
                shutil.rmtree(source)
                self.assert_missing(
                    self.configure(consumer, ALLOW_FETCH_DEPS="OFF"), "GTest"
                )

    def test_legacy_no_download_policy(self):
        truthy = self.make_consumer("legacy-on", "hipdnn")
        self.assert_missing(
            self.configure(truthy, ALLOW_FETCH_DEPS="ON", HIPDNN_NO_DOWNLOAD="ON"),
            "GTest",
        )
        falsey = self.make_consumer("legacy-off", "hipdnn")
        self.assert_missing(self.configure(falsey, HIPDNN_NO_DOWNLOAD="OFF"), "GTest")


if __name__ == "__main__":
    unittest.main()
