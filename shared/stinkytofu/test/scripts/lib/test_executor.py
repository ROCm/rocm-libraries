#!/usr/bin/env python3
"""
Base class for test executors
"""

import os
import sys
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .docker_helper import DockerHelper
from .build_system import BuildSystem
from .test_list import TestListParser
from .yaml_patcher import YAMLPatcher
from .utils import (log_info, log_success, log_warning, log_error,
                   create_result_dir, get_yaml_variant_name, build_test_env_vars)


class TestExecutor(ABC):
    """Base class for all test executors"""

    def __init__(self, config: Dict, test_type: str):
        self.config = config
        self.test_type = test_type
        self.verbose = config.get('verbose', False)

        # Extract configuration
        self.container_name = config['container_name']
        self.docker_hipblaslt_path = config['docker_path']
        self.host_hipblaslt_path = config['host_path']
        self.output_dir = config['output_dir']
        self.gpu_arch = config['gpu_arch']
        self.test_pattern = config.get('pattern', '.*')
        self.git_info = config['git_info']
        self.hostname = config['hostname']
        self.test_date = config['test_date']
        self.tool_path = config.get('tool_path', '')
        self.new_cmake = config.get('new_cmake', False)
        self.preserve_build = config.get('preserve_build', False)

        # Determine script and test directories
        if self.tool_path:
            self.script_dir = Path(self.tool_path) / "scripts"
            self.test_dir = Path(self.tool_path)
        else:
            self.script_dir = Path(__file__).parent.parent
            self.test_dir = self.script_dir.parent

        # Initialize helpers
        self.docker = DockerHelper(self.container_name, self.verbose)
        self.build_system = BuildSystem(self.docker, self.docker_hipblaslt_path,
                                       self.new_cmake, self.verbose)
        self.test_list_parser = TestListParser(str(self.test_dir))
        self.yaml_patcher = YAMLPatcher(str(self.script_dir))

        # Test tracking
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

        # Create result directories
        self.database_dir, self.logs_dir = create_result_dir(
            self.output_dir, self.gpu_arch, self.test_type,
            self.hostname, self.git_info
        )

    @abstractmethod
    def run_single_test(self, test_entry: Tuple[str, str, str]) -> bool:
        """
        Run a single test - must be implemented by subclass

        Args:
            test_entry: Tuple of (yaml_path, active_modifier, all_modifiers)

        Returns:
            True if test passed, False otherwise
        """
        pass

    def setup_test_paths(self, source_yaml: str, active_modifier: str,
                        all_modifiers: str) -> Dict[str, str]:
        """
        Setup test paths for a single test iteration

        Returns:
            Dictionary with test paths and info
        """
        yaml_basename = Path(source_yaml).stem
        yaml_name = get_yaml_variant_name(yaml_basename, active_modifier)

        if self.verbose:
            log_info("-----------------------------------")
            if active_modifier:
                log_info(f"Running test: {yaml_name} (modifier: {active_modifier})")
            else:
                log_info(f"Running test: {yaml_name} (default)")

        # Create test-specific directory in logs
        test_log_dir = Path(self.logs_dir) / yaml_name
        test_log_dir.mkdir(parents=True, exist_ok=True)

        # Generate patched YAML
        patched_yaml = self.yaml_patcher.patch_yaml(
            source_yaml, self.logs_dir, self.test_type,
            active_modifier, all_modifiers
        )

        # Log file
        log_file = Path(self.logs_dir) / f"{yaml_name}.log"

        # Create tmp directory
        tmp_dir = Path(self.logs_dir).parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Output directory (unique temp directory inside tmp/)
        import time
        out_dir = f"../tmp/{yaml_name}_{self.container_name}_{os.getpid()}_{int(time.time())}.out"

        # Map yaml file to docker path
        host_yaml_file = str(Path(patched_yaml).resolve())
        host_work_dir = str(Path(host_yaml_file).parent)
        docker_work_dir = self.docker.map_host_to_docker_path(host_work_dir)
        docker_yaml_file = f"{docker_work_dir}/{Path(patched_yaml).name}"

        if self.verbose:
            log_info(f"YAML file: {docker_yaml_file}")
            log_info(f"Work dir: {docker_work_dir}")
            log_info(f"Output dir: {out_dir}")
            log_info(f"Log file: {log_file}")

        return {
            'yaml_name': yaml_name,
            'patched_yaml': patched_yaml,
            'log_file': str(log_file),
            'out_dir': out_dir,
            'docker_work_dir': docker_work_dir,
            'docker_yaml_file': docker_yaml_file,
            'test_log_dir': str(test_log_dir),
            'host_work_dir': host_work_dir
        }

    def execute_test_command(self, cmd: str, log_file: str, yaml_name: str) -> bool:
        """Execute test command with error handling"""
        if self.verbose:
            log_info("Executing test...")
            log_info(f"Command: {cmd}")

        # Execute the command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Write output to log file
        with open(log_file, 'w') as f:
            if result.stdout:
                f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)

        if result.returncode == 0:
            if self.verbose:
                log_info("Test execution completed")
            return True
        else:
            log_error(f"Test execution failed for {yaml_name}.yaml")
            log_error(f"Command: {cmd}")
            log_error(f"Log file: {log_file}")
            log_error("Last 10 lines of log:")
            # Show last 10 lines
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.rstrip(), file=sys.stderr)
            return False

    def check_test_result(self, log_file: str, yaml_name: str) -> bool:
        """Check if test passed based on log file"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                if "clientExit=0 (PASS)" in content:
                    log_success(f"{yaml_name}.yaml")
                    return True
                else:
                    log_error(f"{yaml_name}.yaml: FAILED (clientExit=0 (PASS) not found in log)")
                    return False
        except Exception as e:
            log_error(f"Error checking test result: {e}")
            return False

    def update_global_database(self) -> None:
        """Update global database with results from this run"""
        if self.verbose:
            log_info("Updating global database...")

        local_json = Path(self.logs_dir) / "local.json"
        db_file = Path(self.database_dir) / f"{self.test_type}-{self.hostname}_{self.gpu_arch}.json"

        if local_json.exists():
            cmd = [
                'python3', str(self.script_dir / 'update_global_database.py'),
                '--local-json', str(local_json),
                '--database-file', str(db_file),
                '--git-hash', self.git_info
            ]
            if self.verbose:
                cmd.append('--verbose')

            subprocess.run(cmd, check=True)

            if self.verbose:
                log_success(f"Global database updated: {db_file}")
        else:
            log_warning("local.json not found, skipping database update")

    def print_summary(self) -> None:
        """Print test summary"""
        if self.verbose:
            log_info("===================================")
            log_info(f"{self.test_type.replace('_', '-').title()} Test Summary")
            log_info("===================================")
            log_info(f"Total tests: {self.total_tests}")
            log_success(f"Passed: {self.passed_tests}")
            if self.failed_tests > 0:
                log_error(f"Failed: {self.failed_tests}")
            else:
                log_info(f"Failed: {self.failed_tests}")
            log_info(f"Database saved to: {self.database_dir}")
            log_info(f"Logs saved to: {self.logs_dir}")

    def execute_all(self) -> int:
        """
        Execute all tests

        Returns:
            Number of failed tests
        """
        if self.verbose:
            log_info(f"Starting {self.test_type} tests for GPU: {self.gpu_arch}")

        # Get test list
        test_list = self.test_list_parser.get_test_list(
            self.test_type.replace('_', '-'),
            self.gpu_arch,
            self.test_pattern
        )

        if not test_list:
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

        # Update global database
        self.update_global_database()

        # Print summary
        self.print_summary()

        # Exit with appropriate status
        if self.failed_tests > 0:
            if self.verbose:
                log_warning(f"Some tests failed. Please check the logs in: {self.logs_dir}")
            return self.failed_tests

        if self.verbose:
            log_success(f"All {self.test_type} tests passed")

        return 0

