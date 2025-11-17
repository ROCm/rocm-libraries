#!/usr/bin/env python3
"""
Utility functions for stinkytofu testing
"""

import os
import sys
import subprocess
import socket
from pathlib import Path
from typing import Optional, Tuple


# Color codes for logging
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def log_info(message: str) -> None:
    """Log info message to stderr"""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}", file=sys.stderr)


def log_success(message: str) -> None:
    """Log success message to stderr"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}", file=sys.stderr)


def log_warning(message: str) -> None:
    """Log warning message to stderr"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}", file=sys.stderr)


def log_error(message: str) -> None:
    """Log error message to stderr"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}", file=sys.stderr)


def log_yellow(message: str) -> None:
    """Log yellow-colored message to stderr"""
    print(f"{Colors.YELLOW}{message}{Colors.NC}", file=sys.stderr)


def get_git_info(repo_path: str) -> str:
    """
    Get git commit info (short hash with -dirty if uncommitted changes)

    Args:
        repo_path: Path to git repository

    Returns:
        Git hash string (e.g., "abc1234" or "abc1234-dirty")
    """
    try:
        # Get short hash
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        git_hash = result.stdout.strip()

        # Check if dirty
        result = subprocess.run(
            ['git', 'diff-index', '--quiet', 'HEAD', '--'],
            cwd=repo_path,
            capture_output=True
        )

        if result.returncode != 0:
            git_hash += "-dirty"

        return git_hash
    except Exception:
        return "unknown"


def get_hostname() -> str:
    """Get short hostname"""
    return socket.gethostname().split('.')[0]


def create_result_dir(output_dir: str, gpu_arch: str, test_type: str,
                     hostname: str, git_info: str) -> Tuple[str, str]:
    """
    Create result directory structure

    Returns:
        Tuple of (database_dir, logs_dir)
    """
    # Create database directory
    database_dir = Path(output_dir) / gpu_arch / "database"
    database_dir.mkdir(parents=True, exist_ok=True)

    # Create logs directory with hostname-git_hash
    log_session = f"{hostname}-{git_info}"
    logs_dir = Path(output_dir) / gpu_arch / "logs" / log_session / test_type
    logs_dir.mkdir(parents=True, exist_ok=True)

    return str(database_dir), str(logs_dir)


def get_build_dir_for_container(base_build_dir: str, container_name: str) -> str:
    """Get container-specific build directory name"""
    return f"{base_build_dir}_{container_name}"


def build_test_env_vars(gpu_arch: str, include_stinkytofu_dump: bool = False) -> dict:
    """
    Build environment variables for test execution

    Args:
        gpu_arch: GPU architecture (e.g., 'gfx1250')
        include_stinkytofu_dump: Whether to include STINKYTOFU_DUMP

    Returns:
        Dictionary of environment variables
    """
    env_vars = {}

    if gpu_arch == "gfx1250":
        # gfx1250-specific environment variables
        env_vars.update({
            'ROCM_PATH': '/opt/rocm',
            'PATH': '/root/rocplaycap/rocplaycap-src-4.5.1/bin:$ROCM_PATH/llvm/bin:$ROCM_PATH/bin:~/.local/bin:$PATH',
            'LD_LIBRARY_PATH': '$HOME/rocr/_build/rocr/lib/',
            'HSA_MODEL_LIB': '/root/download_ffm/libhsakmtmodel.so',
            'HSA_ENABLE_SDMA': '0',
            'HSA_ENABLE_ITERRUPT': '0',
            'HSA_MODEL_TOPOLOGY': '/root/download_ffm/mi450',
            'TARGET_ARCH': 'gfx1250',
            'HSA_KMT_MODEL_GPUVM_BASE': '0x3000000000',
            'HSA_KMT_MODEL_GPUVM_SIZE': '0xF00000000',
            'HSA_MODEL_NUM_THREADS': '224'
        })

    if include_stinkytofu_dump:
        env_vars['STINKYTOFU_DUMP'] = '1'

    return env_vars


def get_yaml_variant_name(yaml_basename: str, active_modifier: str) -> str:
    """Determine YAML variant name from basename and modifier"""
    if active_modifier == "+sia3":
        return f"{yaml_basename}-sia3"
    elif active_modifier == "+sparse":
        return f"{yaml_basename}-sparse"
    elif active_modifier == "+sia5":
        return f"{yaml_basename}-sia5"
    else:
        return yaml_basename

