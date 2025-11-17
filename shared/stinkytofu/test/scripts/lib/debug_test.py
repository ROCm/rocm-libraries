#!/usr/bin/env python3
"""
Debug verification test executor (Test C)
"""

from pathlib import Path
from typing import Dict, Tuple
from .test_executor import TestExecutor
from .utils import log_info, build_test_env_vars


class DebugTest(TestExecutor):
    """Executor for debug verification tests"""

    def __init__(self, config: Dict):
        super().__init__(config, 'dbg_verify')

        # Get build paths for debug build
        self.build_dir, self.tensile_sh, self.client_bin = \
            self.build_system.get_build_paths('dbg_build')

    def run_single_test(self, test_entry: Tuple[str, str, str]) -> bool:
        """Run a single debug verification test"""
        source_yaml, active_modifier, all_modifiers = test_entry

        # Setup test paths
        paths = self.setup_test_paths(source_yaml, active_modifier, all_modifiers)
        yaml_name = paths['yaml_name']
        patched_yaml = paths['patched_yaml']
        log_file = paths['log_file']
        out_dir = paths['out_dir']
        docker_work_dir = paths['docker_work_dir']
        docker_yaml_file = paths['docker_yaml_file']

        # Clean up old output directory if exists (unless --preserve-build is set)
        if not self.preserve_build:
            self.docker.remove_files(out_dir, docker_work_dir)
        else:
            if self.verbose:
                log_info(f"Preserving existing build directory: {out_dir}")

        # Build environment variables
        env_vars = build_test_env_vars(self.gpu_arch, include_stinkytofu_dump=True)

        # Build environment setup string
        env_setup = ' && '.join([f'export {k}={v}' for k, v in env_vars.items()])
        if env_setup:
            env_setup += ' && '

        # Build the command
        cmd = f"docker exec -w \"{docker_work_dir}\" -it \"{self.container_name}\" bash -c \"{env_setup}{self.tensile_sh} {docker_yaml_file} {out_dir} --prebuilt-client={self.client_bin}\""

        # Execute the command
        if not self.execute_test_command(cmd, log_file, yaml_name):
            return False

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

    def execute_all(self) -> int:
        """Execute all debug verification tests"""
        # Run tests without database update (debug tests don't generate local.json)
        if self.verbose:
            log_info(f"Starting {self.test_type} tests for GPU: {self.gpu_arch}")

        # Get test list
        test_list = self.test_list_parser.get_test_list(
            self.test_type.replace('_', '-'),
            self.gpu_arch,
            self.test_pattern
        )

        if not test_list:
            from .utils import log_warning
            log_warning(f"No test cases found matching pattern: {self.test_pattern}")
            return 0

        if self.verbose:
            log_info(f"Found {len(test_list)} test case(s)")

        if self.verbose:
            log_info(f"Database files will be saved to: {self.database_dir}")
            log_info(f"Log files will be saved to: {self.logs_dir}")

        # Run each test
        for test_entry in test_list:
            self.total_tests += 1
            if self.run_single_test(test_entry):
                self.passed_tests += 1
            else:
                self.failed_tests += 1

        # Note: Debug tests don't update global database (no performance metrics)

        # Print summary
        self.print_summary()

        # Create summary file in logs directory
        summary_file = Path(self.logs_dir) / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Debug Verification Test Summary\n")
            f.write("================================\n")
            f.write(f"Git commit: {self.git_info}\n")
            f.write(f"Hostname: {self.hostname}\n")
            f.write(f"GPU: {self.gpu_arch}\n")
            f.write(f"Test date: {self.test_date}\n")
            f.write("\n")
            f.write(f"Total tests: {self.total_tests}\n")
            f.write(f"Passed: {self.passed_tests}\n")
            f.write(f"Failed: {self.failed_tests}\n")

        # Create report file in database directory
        report_file = Path(self.database_dir) / f"dbg_verify-{self.hostname}_{self.gpu_arch}_report.txt"
        with open(report_file, 'w') as f:
            f.write("Debug Verification Test Report\n")
            f.write("================================\n")
            f.write(f"Git commit: {self.git_info}\n")
            f.write(f"Hostname: {self.hostname}\n")
            f.write(f"GPU: {self.gpu_arch}\n")
            f.write(f"Test date: {self.test_date}\n")
            f.write("\n")
            f.write(f"Total tests: {self.total_tests}\n")
            f.write(f"Passed: {self.passed_tests}\n")
            f.write(f"Failed: {self.failed_tests}\n")

        if self.verbose:
            log_info(f"Summary saved to: {summary_file}")
            log_info(f"Report saved to: {report_file}")

        # Exit with appropriate status
        if self.failed_tests > 0:
            if self.verbose:
                from .utils import log_warning
                log_warning(f"Some tests failed. Please check the logs in: {self.logs_dir}")
            return self.failed_tests

        if self.verbose:
            from .utils import log_success
            log_success(f"All {self.test_type} tests passed")

        return 0

