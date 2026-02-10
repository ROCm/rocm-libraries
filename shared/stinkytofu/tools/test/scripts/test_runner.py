#!/usr/bin/env python3
"""
Main test orchestration script for hipblaslt/tensilelite testing
This script coordinates exe-time, codegen-time, and dbg-verify tests
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.docker_helper import DockerHelper
from lib.utils import (log_info, log_success, log_error, log_yellow,
                      get_git_info, get_hostname)
from lib.exe_time_test import ExeTimeTest
from lib.codegen_test import CodegenTest
from lib.debug_test import DebugTest


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Main test orchestration script for hipblaslt/tensilelite testing'
    )

    parser.add_argument('--container-name', required=True,
                       help='Docker container name')
    parser.add_argument('--docker-path', required=True,
                       help='Path to hipblaslt inside container')
    parser.add_argument('--host-path', required=True,
                       help='Path to hipblaslt on host')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for logs and results')
    parser.add_argument('--tool-path', default='',
                       help='Path to stinkytofu test tools (default: script location)')
    parser.add_argument('--new-cmake', type=int, default=0, choices=[0, 1],
                       help='Use new ROCm cmake build system: 0=old, 1=new (default: 0)')
    parser.add_argument('--test', default='all',
                       choices=['exe_time', 'codegen_time', 'dbg_verify', 'all'],
                       help='Test type (default: all)')
    parser.add_argument('--gpu-arch', default='',
                       help='GPU architecture (auto-detected if not provided)')
    parser.add_argument('--pattern', default='.*',
                       help='Regex pattern to select test cases (default: .*)')
    parser.add_argument('--preserve-build', action='store_true',
                       help='Preserve build output directory (default: disabled)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output (default: disabled)')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Print header
    log_info("===================================")
    log_info("Starting stinkytofu Test Suite")
    log_info("===================================")
    log_info(f"Container: {args.container_name}")
    log_info(f"Docker path: {args.docker_path}")
    log_info(f"Host path: {args.host_path}")
    log_info(f"Output dir: {args.output_dir}")
    log_info(f"Test type: {args.test}")
    log_info(f"Test pattern: {args.pattern}")
    if args.tool_path:
        log_info(f"Tool path: {args.tool_path}")
    log_info(f"CMake system: {'new' if args.new_cmake else 'old'}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Check container
    if args.verbose:
        log_info("Checking container...")

    docker = DockerHelper(args.container_name, args.verbose)
    if not docker.check_container():
        log_error("Container check failed")
        return 1

    # Start container if not running
    if args.verbose:
        log_info("Starting container...")

    if not docker.start_container():
        log_error(f"Failed to start container {args.container_name}")
        return 1

    log_success("Container started")

    # Check if tensilelite folder exists
    if args.verbose:
        log_info("Checking tensilelite folder...")

    tensilelite_path = f"{args.docker_path}/tensilelite"
    if not docker.check_path_exists(tensilelite_path):
        log_error(f"Tensilelite folder not found at {tensilelite_path}")
        return 1

    log_success("Tensilelite folder found")

    # Detect GPU architecture if not provided
    if not args.gpu_arch:
        if args.verbose:
            log_info("Detecting GPU architecture...")

        args.gpu_arch = docker.detect_gpu_arch()
        if not args.gpu_arch:
            log_error("Failed to detect GPU architecture")
            return 1

        log_success(f"Detected GPU: {args.gpu_arch}")
    else:
        if args.verbose:
            log_info(f"Using specified GPU architecture: {args.gpu_arch}")

    # Check and install yappi if needed (for codegen_time test)
    if args.test in ['codegen_time', 'all']:
        if args.verbose:
            log_info("Checking yappi installation...")

        if not docker.check_yappi():
            if args.verbose:
                log_info("Installing yappi...")
            if not docker.install_yappi():
                log_error("Failed to install yappi")
                return 1

        log_success("yappi is available")

    # Get git info, hostname, and test date (shared across all tests)
    git_info = get_git_info(args.host_path)
    hostname = get_hostname()
    test_date = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_info(f"Git commit: {git_info}")
    log_info(f"Hostname: {hostname}")
    log_info(f"Test date: {test_date}")

    # Prepare configuration dictionary
    config = {
        'container_name': args.container_name,
        'docker_path': args.docker_path,
        'host_path': args.host_path,
        'output_dir': args.output_dir,
        'gpu_arch': args.gpu_arch,
        'pattern': args.pattern,
        'git_info': git_info,
        'hostname': hostname,
        'test_date': test_date,
        'verbose': args.verbose,
        'tool_path': args.tool_path,
        'new_cmake': bool(args.new_cmake),
        'preserve_build': args.preserve_build
    }

    # Track overall test failures
    overall_failed = False

    # Import build system for building binaries
    from lib.build_system import BuildSystem
    build_system = BuildSystem(docker, args.docker_path, bool(args.new_cmake), args.verbose)

    # Run tests based on test type
    if args.test in ['exe_time', 'all']:
        log_yellow("-----------------------------------")
        log_yellow("Running exe_time tests")
        log_yellow("-----------------------------------")

        # Check/build Release binary
        if not build_system.check_and_build_binary('Release', 'rel_build'):
            log_error("Failed to build Release binary")
            return 1

        # Run exe_time tests
        test = ExeTimeTest(config)
        if test.execute_all() != 0:
            overall_failed = True

    if args.test in ['codegen_time', 'all']:
        log_yellow("-----------------------------------")
        log_yellow("Running codegen_time tests")
        log_yellow("-----------------------------------")

        # Check/build Release binary
        if not build_system.check_and_build_binary('Release', 'rel_build'):
            log_error("Failed to build Release binary")
            return 1

        # Run codegen_time tests
        test = CodegenTest(config)
        if test.execute_all() != 0:
            overall_failed = True

    if args.test in ['dbg_verify', 'all']:
        log_yellow("-----------------------------------")
        log_yellow("Running dbg_verify tests")
        log_yellow("-----------------------------------")

        # Check/build Debug binary
        if not build_system.check_and_build_binary('Debug', 'dbg_build'):
            log_error("Failed to build Debug binary")
            return 1

        # Run dbg_verify tests
        test = DebugTest(config)
        if test.execute_all() != 0:
            overall_failed = True

    log_info(f"Completed. Results are in: {args.output_dir}")

    # Exit with error if any test type failed
    if overall_failed:
        log_error("Some tests failed across one or more test types")
        return 1

    # Exit with success
    log_success("All tests passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())

