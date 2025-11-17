#!/usr/bin/env python3
"""
Code generation time test executor (Test B)
"""

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple
from .test_executor import TestExecutor
from .utils import log_info, log_error, build_test_env_vars


class CodegenTest(TestExecutor):
    """Executor for code generation time tests"""

    def __init__(self, config: Dict):
        super().__init__(config, 'codegen_time')

        # Get build paths
        self.build_dir, self.tensile_sh, self.client_bin = \
            self.build_system.get_build_paths('rel_build')

    def run_single_test(self, test_entry: Tuple[str, str, str]) -> bool:
        """Run a single code generation time test"""
        source_yaml, active_modifier, all_modifiers = test_entry

        # Setup test paths
        paths = self.setup_test_paths(source_yaml, active_modifier, all_modifiers)
        yaml_name = paths['yaml_name']
        patched_yaml = paths['patched_yaml']
        log_file = paths['log_file']
        out_dir = paths['out_dir']
        docker_work_dir = paths['docker_work_dir']
        docker_yaml_file = paths['docker_yaml_file']
        host_work_dir = paths['host_work_dir']

        # Yappi-specific files
        yappi_file = "yappi_results.txt"
        yappi_profile_file = "yappi_results.profile"
        result_yappi_file = Path(self.logs_dir) / f"{yaml_name}_yappi_results.txt"

        if self.verbose:
            log_info(f"Yappi results: {result_yappi_file}")

        # Clean up old output directory if exists (unless --preserve-build is set)
        if not self.preserve_build:
            self.docker.remove_files(out_dir, docker_work_dir)
        else:
            if self.verbose:
                log_info(f"Preserving existing build directory: {out_dir}")

        # Build environment variables
        env_vars = build_test_env_vars(self.gpu_arch, include_stinkytofu_dump=False)

        # Build environment setup string
        env_setup = ' && '.join([f'export {k}={v}' for k, v in env_vars.items()])
        if env_setup:
            env_setup += ' && '

        # Build the command
        cmd = f"docker exec -w \"{docker_work_dir}\" \"{self.container_name}\" bash -c \"{env_setup}{self.tensile_sh} {docker_yaml_file} {out_dir} --prebuilt-client={self.client_bin}\""

        # Execute the command
        if not self.execute_test_command(cmd, log_file, yaml_name):
            return False

        # Move yappi results to final location
        if self.verbose:
            log_info("Collecting yappi results...")

        host_yappi_file = Path(host_work_dir) / yappi_file
        host_yappi_profile_file = Path(host_work_dir) / yappi_profile_file

        if host_yappi_file.exists():
            shutil.copy(host_yappi_file, result_yappi_file)
            host_yappi_file.unlink()
            if host_yappi_profile_file.exists():
                host_yappi_profile_file.unlink()
            if self.verbose:
                log_info(f"Yappi results saved to: {result_yappi_file}")
        else:
            log_error(f"Yappi results file not found at: {host_yappi_file}")
            return False

        # Parse results and save locally
        if self.verbose:
            log_info("Parsing results...")

        parse_cmd = [
            'python3', str(self.script_dir / 'parse_codegen_time.py'),
            '--yappi-file', str(result_yappi_file),
            '--yaml-file', patched_yaml,
            '--output-dir', self.logs_dir,
            '--git-info', self.git_info,
            '--hostname', self.hostname,
            '--gpu-arch', self.gpu_arch,
            '--test-date', self.test_date,
            '--yaml-name', yaml_name
        ]
        if self.verbose:
            parse_cmd.append('--verbose')

        subprocess.run(parse_cmd, check=True)

        # Check if test passed
        test_passed = self.check_test_result(log_file, yaml_name)

        # Clean up docker workspace (unless --preserve-build is set)
        if not self.preserve_build:
            cleanup_pattern = f"{out_dir} a.out"
            self.docker.remove_files(cleanup_pattern, docker_work_dir)
        else:
            cleanup_pattern = "a.out"
            self.docker.remove_files(cleanup_pattern, docker_work_dir)
            if self.verbose:
                log_info(f"Preserved build directory: {out_dir}")

        return test_passed

