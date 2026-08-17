# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Utility wrappers for ROCm profiling tools.

This module provides Python wrappers around ROCm command-line tools,
making them easier to call from Python scripts and ensuring they're
accessible regardless of PATH configuration.
"""

import os
import subprocess
import shutil
from pathlib import Path


# Default ROCm installation path
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")


def find_rocm_tool(
    tool_name: str, search_paths: list[str] | None = None
) -> Path | None:
    """Find a ROCm tool by searching common installation paths.

    Args:
        tool_name: Name of the tool (e.g., 'rocprofv3', 'llvm-objdump')
        search_paths: Additional paths to search (default: standard ROCm paths)

    Returns:
        Path to tool if found, None otherwise
    """
    if search_paths is None:
        search_paths = [
            f"{ROCM_PATH}/bin",
            f"{ROCM_PATH}/llvm/bin",
            "/usr/bin",
            "/usr/local/bin",
        ]

    # First check if it's in PATH
    tool_in_path = shutil.which(tool_name)
    if tool_in_path:
        return Path(tool_in_path)

    # Search in specific directories
    for search_path in search_paths:
        tool_path = Path(search_path) / tool_name
        if tool_path.exists() and tool_path.is_file():
            # Check if executable
            if os.access(tool_path, os.X_OK):
                return tool_path

    return None


def run_rocprof(
    command: list[str],
    stats: bool = True,
    kernel_trace: bool = True,
    output_file: str | None = None,
    output_format: str = "csv",
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run rocprofv3 with specified options.

    Args:
        command: Command to profile (as list)
        stats: Enable --stats flag
        kernel_trace: Enable --kernel-trace flag
        output_file: Output file prefix
        output_format: Output format ('csv', 'json', 'pftrace')
        timeout: Timeout in seconds

    Returns:
        CompletedProcess object from subprocess.run

    Raises:
        FileNotFoundError: If rocprofv3 not found
        subprocess.TimeoutExpired: If command times out
    """
    rocprof = find_rocm_tool("rocprofv3")
    if not rocprof:
        raise FileNotFoundError(
            "rocprofv3 not found. Please ensure ROCm is installed and "
            "ROCM_PATH is set correctly."
        )

    # Build rocprof command
    rocprof_cmd = [str(rocprof)]

    if stats:
        rocprof_cmd.append("--stats")
    if kernel_trace:
        rocprof_cmd.append("--kernel-trace")

    rocprof_cmd.extend(["-f", output_format])

    if output_file:
        rocprof_cmd.extend(["-o", output_file])

    rocprof_cmd.append("--")
    rocprof_cmd.extend(command)

    return subprocess.run(rocprof_cmd, capture_output=True, text=True, timeout=timeout)


def run_llvm_objdump(
    input_file: Path,
    mcpu: str = "gfx950",
    disassemble: bool = True,
    symbolize_operands: bool = True,
) -> str:
    """Run llvm-objdump to disassemble a code object.

    Args:
        input_file: Path to code object (.co file)
        mcpu: GPU architecture (e.g., 'gfx950', 'gfx940')
        disassemble: Enable disassembly (-d flag)
        symbolize_operands: Symbolize operands (--symbolize-operands)

    Returns:
        Disassembled output as string

    Raises:
        FileNotFoundError: If llvm-objdump not found
    """
    llvm_objdump = find_rocm_tool("llvm-objdump")
    if not llvm_objdump:
        raise FileNotFoundError(
            "llvm-objdump not found. Please ensure ROCm LLVM is installed."
        )

    cmd = [str(llvm_objdump)]

    if disassemble:
        cmd.append("-d")
    if symbolize_operands:
        cmd.append("--symbolize-operands")

    cmd.extend(["--mcpu", mcpu, str(input_file)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"llvm-objdump failed: {result.stderr}")

    return result.stdout


def run_roc_obj_extract(input_file: Path, output_dir: Path) -> list[Path]:
    """Run roc-obj-extract to extract code objects from .so file.

    Args:
        input_file: Path to .so file
        output_dir: Directory for extracted code objects

    Returns:
        List of paths to extracted .co files

    Raises:
        FileNotFoundError: If roc-obj-extract not found
    """
    roc_obj_extract = find_rocm_tool("roc-obj-extract")
    if not roc_obj_extract:
        raise FileNotFoundError(
            "roc-obj-extract not found. Please ensure ROCm is installed."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [str(roc_obj_extract), str(input_file)],
        cwd=output_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"roc-obj-extract failed: {result.stderr}")

    # Return list of extracted .co files
    return list(output_dir.glob("*.co"))


def get_rocm_info() -> dict:
    """Get ROCm installation information.

    Returns:
        Dictionary with ROCm version, path, and available tools
    """
    info = {"rocm_path": ROCM_PATH, "tools": {}, "version": None}

    # Find all tools
    tools = ["rocprofv3", "llvm-objdump", "roc-obj-extract", "rocm-smi", "rocminfo"]
    for tool in tools:
        tool_path = find_rocm_tool(tool)
        info["tools"][tool] = str(tool_path) if tool_path else None

    # Get ROCm version from rocminfo
    rocminfo = find_rocm_tool("rocminfo")
    if rocminfo:
        try:
            result = subprocess.run(
                [str(rocminfo)], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split("\n"):
                if "Runtime Version:" in line:
                    info["version"] = line.split(":")[-1].strip()
                    break
        except Exception:
            pass

    return info


def verify_rocm_tools() -> tuple[bool, list[str]]:
    """Verify all required ROCm tools are available.

    Returns:
        Tuple of (all_found: bool, missing: list of missing tool names)
    """
    required_tools = ["rocprofv3", "llvm-objdump", "roc-obj-extract"]
    missing = []

    for tool in required_tools:
        if not find_rocm_tool(tool):
            missing.append(tool)

    return (len(missing) == 0, missing)


if __name__ == "__main__":
    # Test tool detection
    print("ROCm Tool Detection")
    print("=" * 60)

    info = get_rocm_info()
    print(f"ROCm Path:    {info['rocm_path']}")
    print(f"ROCm Version: {info['version'] or 'Unknown'}")
    print("\nTools:")

    for tool, path in info["tools"].items():
        status = "✓" if path else "✗"
        print(f"  {status} {tool:20s} {path or 'NOT FOUND'}")

    all_found, missing = verify_rocm_tools()
    print("\n" + "=" * 60)
    if all_found:
        print("✓ All required ROCm tools found!")
    else:
        print(f"✗ Missing tools: {', '.join(missing)}")
        print("\nPlease install ROCm or set ROCM_PATH environment variable.")
