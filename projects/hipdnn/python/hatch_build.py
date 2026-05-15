# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Hatchling custom build hook: compiles the nanobind extension or copies a pre-built one."""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        package_dir = os.path.join(self.root, "hipdnn_frontend")
        so_path = self._find_or_build_extension()
        dest = os.path.join(package_dir, os.path.basename(so_path))
        shutil.copy2(so_path, dest)
        build_data["shared_data"] = {"_extension_path": dest}
        build_data["force_include"] = {
            dest: f"hipdnn_frontend/{os.path.basename(so_path)}",
        }

    def finalize(self, version, build_data, artifact_path):
        if version == "editable":
            return
        path = (build_data.get("shared_data") or {}).get("_extension_path")
        if path and os.path.isfile(path):
            os.remove(path)

    def _find_or_build_extension(self):
        prebuilt = os.environ.get("HIPDNN_PREBUILT_SO")
        if prebuilt:
            if not os.path.isfile(prebuilt):
                raise RuntimeError(f"HIPDNN_PREBUILT_SO points to missing file: {prebuilt}")
            return prebuilt
        return self._compile_extension()

    def _compile_extension(self):
        build_dir = os.path.join(self.root, "_hatch_build")
        os.makedirs(build_dir, exist_ok=True)

        cmake_args = [
            "cmake",
            "-S", self.root,
            "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        prefix_path = os.environ.get("CMAKE_PREFIX_PATH")
        if prefix_path:
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={prefix_path}")

        subprocess.check_call(cmake_args)
        subprocess.check_call(["cmake", "--build", build_dir])

        pattern = os.path.join(build_dir, "hipdnn_frontend_python*.so")
        matches = glob.glob(pattern)
        if not matches:
            raise RuntimeError(
                f"Build succeeded but no .so found matching {pattern}. "
                f"Contents: {os.listdir(build_dir)}"
            )
        return matches[0]
