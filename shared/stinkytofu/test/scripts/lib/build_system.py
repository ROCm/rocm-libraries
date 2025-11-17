#!/usr/bin/env python3
"""
Build system management for hipblaslt/tensilelite
"""

import sys
import tempfile
from pathlib import Path
from typing import Tuple
from .docker_helper import DockerHelper
from .utils import log_info, log_error, log_success, get_build_dir_for_container


class BuildSystem:
    """Handles building hipblaslt/tensilelite binaries"""

    def __init__(self, docker: DockerHelper, docker_hipblaslt_path: str,
                 new_cmake: bool = False, verbose: bool = False):
        self.docker = docker
        self.docker_hipblaslt_path = docker_hipblaslt_path
        self.new_cmake = new_cmake
        self.verbose = verbose

    def get_build_paths(self, build_type: str) -> Tuple[str, str, str]:
        """
        Get build paths for test execution

        Args:
            build_type: 'rel_build' or 'dbg_build'

        Returns:
            Tuple of (build_dir, tensile_sh, client_bin)
        """
        build_dir = get_build_dir_for_container(build_type, self.docker.container_name)
        tensile_sh = f"{self.docker_hipblaslt_path}/tensilelite/{build_dir}/Tensile.sh"

        if self.new_cmake:
            client_bin = f"{self.docker_hipblaslt_path}/tensilelite/{build_dir}/tensilelite/client/tensilelite-client"
        else:
            client_bin = f"{self.docker_hipblaslt_path}/tensilelite/{build_dir}/tensilelite/client/tensile_client"

        return build_dir, tensile_sh, client_bin

    def _exec_build_cmd(self, cmd: str, error_msg: str) -> bool:
        """Execute build command with proper error handling"""
        if self.verbose:
            # In verbose mode, output directly to console
            result = self.docker.exec_command(cmd, workdir=self.docker_hipblaslt_path, capture_output=False)
        else:
            # In non-verbose mode, capture output and only show on error
            result = self.docker.exec_command(cmd, workdir=self.docker_hipblaslt_path, capture_output=True)
            if result.returncode != 0:
                log_error(error_msg)
                if result.stdout:
                    print(result.stdout, file=sys.stderr)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)

        return result.returncode == 0

    def check_and_build_binary(self, build_type_name: str, build_dir: str) -> bool:
        """
        Check and build binary if needed

        Args:
            build_type_name: 'Release' or 'Debug'
            build_dir: 'rel_build' or 'dbg_build'

        Returns:
            True if successful, False otherwise
        """
        build_dir_with_container = get_build_dir_for_container(build_dir, self.docker.container_name)
        build_path = f"{self.docker_hipblaslt_path}/tensilelite/{build_dir_with_container}"

        # Determine client path based on cmake system
        if self.new_cmake:
            client_path = f"{build_path}/tensilelite/client/tensilelite-client"
        else:
            client_path = f"{build_path}/tensilelite/client/tensile_client"

        if self.verbose:
            log_info(f"Checking {build_type_name} build...")

        # Check if client binary exists
        if self.docker.check_path_exists(client_path):
            if self.verbose:
                log_info(f"{build_type_name} binary exists, running incremental build...")

            # Incremental build
            cmd = f"cmake --build tensilelite/{build_dir_with_container} --parallel"
            if not self._exec_build_cmd(cmd, "Incremental build failed"):
                return False
        else:
            if self.verbose:
                log_info(f"{build_type_name} binary not found, running full build...")

            # Check if build directory exists
            if not self.docker.check_path_exists(build_path):
                # Need to configure first
                if self.verbose:
                    log_info(f"Configuring {build_type_name} build...")

                if self.new_cmake:
                    # New cmake system: use preset with TENSILELITE_BUILD_TESTING
                    cmd = f"cmake --preset tensilelite -S . -B tensilelite/{build_dir_with_container} -DTENSILELITE_BUILD_TESTING=ON -DCMAKE_BUILD_TYPE={build_type_name}"
                    if not self._exec_build_cmd(cmd, "CMake configuration failed"):
                        return False
                else:
                    # Old cmake system: configure main system
                    cmd = f"cd tensilelite && cmake -S ./ -B ./{build_dir_with_container} -DDEVELOP_MODE=ON -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE={build_type_name}"
                    if not self._exec_build_cmd(cmd, "CMake configuration failed"):
                        return False

                    # Configure old cmake system (tensilelite-client)
                    cmd = f"cd tensilelite && cmake -S ./Tensile/Source/ -B ./{build_dir_with_container}/tensilelite -DTENSILE_USE_HIP=ON -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DTensile_LIBRARY_FORMAT=yaml -DTensile_ENABLE_MARKER=False -DTENSILE_USE_LLVM=ON -DTENSILE_USE_MSGPACK=ON -DCMAKE_BUILD_TYPE={build_type_name}"
                    if not self._exec_build_cmd(cmd, "CMake tensilelite-client configuration failed"):
                        return False

            # Build directory exists (or just configured), now build
            if self.verbose:
                log_info(f"Building {build_type_name}...")

            # Build
            cmd = f"cmake --build tensilelite/{build_dir_with_container} --parallel"
            if not self._exec_build_cmd(cmd, "Build failed"):
                return False

            # Build old cmake system (tensilelite-client) only if not using new cmake
            if not self.new_cmake:
                cmd = f"cd tensilelite && cmake --build {build_dir_with_container}/tensilelite --parallel"
                if not self._exec_build_cmd(cmd, "Build tensilelite-client failed"):
                    return False

        log_success(f"binary ready: {build_path}")
        return True

