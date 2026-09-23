# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Core functionality for rocRoller test management.
"""

import json
import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def build_gtest_filter_args(include: list[str], exclude: list[str]) -> list[str]:
    """
    Build GTest command-line filter arguments.

    Args:
        include: List of patterns to include (empty = include all)
        exclude: List of patterns to exclude

    Returns:
        List of command-line arguments (e.g., ['--gtest_filter=*-Excluded*'])
    """
    if not include and not exclude:
        logger.debug("No GTest filters specified")
        return []

    # Build filter string
    parts = []

    # Include patterns (default to * if not specified)
    if include:
        parts.append(":".join(include))
        logger.debug(f"GTest include patterns: {include}")
    else:
        parts.append("*")

    # Exclude patterns
    if exclude:
        parts.append("-" + ":".join(exclude))
        logger.debug(f"GTest exclude patterns: {exclude}")

    filter_string = "".join(parts)
    logger.info(f"GTest filter string: {filter_string}")
    return [f"--gtest_filter={filter_string}"]


def build_catch2_filter_args(include: list[str], exclude: list[str]) -> list[str]:
    """
    Build Catch2 command-line filter arguments.

    Args:
        include: List of patterns to include
        exclude: List of patterns to exclude

    Returns:
        List of command-line arguments
    """
    args = []

    # Catch2 uses positional arguments for includes and --exclude for excludes
    # For includes, we pass them as positional arguments
    for pattern in include:
        args.append(pattern)

    # For excludes, use ~[pattern] syntax
    for pattern in exclude:
        args.append(f"~{pattern}")

    if include:
        logger.debug(f"Catch2 include patterns: {include}")
    if exclude:
        logger.debug(f"Catch2 exclude patterns: {exclude}")
    if args:
        logger.info(f"Catch2 filter args: {args}")
    else:
        logger.debug("No Catch2 filters specified")

    return args


def build_ctest_filter_args(include: list[str], exclude: list[str]) -> list[str]:
    """
    Build CTest command-line filter arguments.

    Args:
        include: List of argument lists
        exclude: List of argument lists

    Returns:
        List of command-line arguments (e.g., ['-R', 'pattern', '-E', 'exclude'])
    """
    args = []

    # We allow lists of arguments
    if include:
        for x in include:
            args.extend(x)
    if exclude:
        for x in exclude:
            args.extend(x)

    if args:
        logger.info(f"CTest filter args: {args}")
    else:
        logger.debug("No CTest filters specified")

    return args


@dataclass(frozen=True, eq=True)
class Test:
    """Represents a single test case."""

    name: str
    framework: str


def parse_gtest_xml(xml_file: Path) -> set[Test]:
    """
    Parse GTest XML output to extract test names.

    Args:
        xml_file: Path to GTest XML output file

    Returns:
        Set of Test objects
    """
    logger.debug(f"Parsing GTest XML: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    tests = set()

    # GTest XML structure: <testsuites><testsuite name="..."><testcase name="..."/></testsuite></testsuites>
    for testsuite in root.findall(".//testsuite"):
        suite_name = testsuite.get("name", "")
        for testcase in testsuite.findall("testcase"):
            test_name = testcase.get("name", "")
            if test_name and suite_name:
                # Combine suite and test name
                full_name = f"{suite_name}/{test_name}"
                tests.add(Test(name=full_name, framework="gtest"))

    logger.info(f"Parsed {len(tests)} GTest tests")
    return tests


def parse_catch2_xml(xml_file: Path) -> set[Test]:
    """
    Parse Catch2 XML output to extract test names.

    Args:
        xml_file: Path to Catch2 XML output file

    Returns:
        Set of Test objects
    """
    logger.debug(f"Parsing Catch2 XML: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    tests = set()

    # Catch2 XML structure: <TestCase><Name>...</Name></TestCase>
    for testcase in root.findall(".//TestCase"):
        name_elem = testcase.find("Name")
        if name_elem is not None and name_elem.text:
            tests.add(Test(name=name_elem.text, framework="catch2"))

    logger.info(f"Parsed {len(tests)} Catch2 tests")
    return tests


def parse_ctest_json(json_file: Path) -> set[Test]:
    """
    Parse CTest JSON output to extract test names.

    Args:
        json_file: Path to CTest JSON output file

    Returns:
        Set of Test objects
    """
    logger.debug(f"Parsing CTest JSON: {json_file}")
    with open(json_file, "r") as f:
        data = json.load(f)

    tests = set()

    # CTest JSON structure: {"tests": [{"name": "..."}, ...]}
    for test_data in data.get("tests", []):
        test_name = test_data.get("name", "")
        if test_name:
            tests.add(Test(name=test_name, framework="ctest"))

    logger.info(f"Parsed {len(tests)} CTest tests")
    return tests


def discover_gtest_tests(
    executable: str,
    include: list[str] = None,
    exclude: list[str] = None,
    build_dir: Path | None = None,
) -> set[Test]:
    """
    Discover GTest tests by running the executable.

    Args:
        executable: Path to GTest executable
        include: List of patterns to include (empty = include all)
        exclude: List of patterns to exclude
        build_dir: Optional build directory

    Returns:
        Set of Test objects
    """
    if include is None:
        include = []
    if exclude is None:
        exclude = []

    logger.info(f"Discovering GTest tests from {executable}")

    # Determine working directory
    cwd = str(build_dir) if build_dir else None
    if not cwd:
        script_dir = Path(__file__).resolve().parent
        rocroller_root = script_dir.parent.parent.parent
        cwd = str(rocroller_root)

    logger.debug(f"Working directory: {cwd}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        xml_file = tmpdir_path / "gtest-output.xml"

        cmd = [executable, "--gtest_list_tests", f"--gtest_output=xml:{xml_file}"]
        filter_args = build_gtest_filter_args(include, exclude)
        cmd.extend(filter_args)

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)
        logger.debug(f"Command completed with return code: {result.returncode}")

        return parse_gtest_xml(xml_file)


def discover_catch2_tests(
    executable: str,
    include: list[str] = None,
    exclude: list[str] = None,
    build_dir: Path | None = None,
) -> set[Test]:
    """
    Discover Catch2 tests by running the executable.

    Args:
        executable: Path to Catch2 executable
        include: List of patterns to include
        exclude: List of patterns to exclude
        build_dir: Optional build directory

    Returns:
        Set of Test objects
    """
    if include is None:
        include = []
    if exclude is None:
        exclude = []

    logger.info(f"Discovering Catch2 tests from {executable}")

    # Determine working directory
    cwd = str(build_dir) if build_dir else None
    if not cwd:
        script_dir = Path(__file__).resolve().parent
        rocroller_root = script_dir.parent.parent.parent
        cwd = str(rocroller_root)

    logger.debug(f"Working directory: {cwd}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        xml_file = tmpdir_path / "catch2-output.xml"

        cmd = [
            executable,
            "--list-tests",
            "--reporter",
            "xml",
            "--out",
            str(xml_file),
        ]
        filter_args = build_catch2_filter_args(include, exclude)
        cmd.extend(filter_args)

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)
        logger.debug(f"Command completed with return code: {result.returncode}")

        return parse_catch2_xml(xml_file)


def discover_ctest_tests(
    executable: str,
    include: list[str] = None,
    exclude: list[str] = None,
    build_dir: Path | None = None,
) -> set[Test]:
    """
    Discover CTest tests by running ctest.

    Args:
        executable: Path to ctest executable
        include: List of patterns to include (regex)
        exclude: List of patterns to exclude (regex)
        build_dir: Optional build directory

    Returns:
        Set of Test objects
    """
    if include is None:
        include = []
    if exclude is None:
        exclude = []

    logger.info(f"Discovering CTest tests from {executable}")

    # CTest needs to run from build directory
    cwd = str(build_dir) if build_dir else None
    logger.debug(f"Working directory: {cwd}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        cmd = [executable, "--show-only=json-v1"]
        filter_args = build_ctest_filter_args(include, exclude)
        cmd.extend(filter_args)

        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=cwd, check=True, capture_output=True, text=True
        )
        logger.debug(f"Command completed with return code: {result.returncode}")

        json_file = tmpdir_path / "ctest-output.json"
        with open(json_file, "w") as f:
            f.write(result.stdout)

        return parse_ctest_json(json_file)


def list_tests_for_executable(
    executable: str, profile: str, build_dir: Path | None = None
) -> set[Test]:
    """
    List tests for a given executable and profile.

    Args:
        executable: Path to test executable
        profile: Profile name from test-profiles.yaml
        build_dir: Optional build directory (for ctest)

    Returns:
        Set of Test objects matching the profile filters
    """
    # Load config
    script_dir = Path(__file__).resolve().parent
    config_file = script_dir.parent.parent.parent / "test-profiles.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Find profile
    if profile not in config.get("profiles", {}):
        raise ValueError(f"Profile '{profile}' not found in test-profiles.yaml")

    profile_data = config["profiles"][profile]

    # Find the test entry for this executable
    framework = None
    include_patterns = []
    exclude_patterns = []

    tests_list = profile_data.get("tests", [])
    for test_entry in tests_list:
        if test_entry.get("executable") == executable:
            framework = test_entry.get("framework")
            include_patterns = test_entry.get("include", [])
            exclude_patterns = test_entry.get("exclude", [])
            break

    if framework is None:
        raise ValueError(f"Executable '{executable}' not found in profile '{profile}'")

    # Use framework-specific discovery functions
    if framework == "gtest":
        return discover_gtest_tests(
            executable, include_patterns, exclude_patterns, build_dir
        )
    elif framework == "catch2":
        return discover_catch2_tests(
            executable, include_patterns, exclude_patterns, build_dir
        )
    elif framework == "ctest":
        return discover_ctest_tests(
            executable, include_patterns, exclude_patterns, build_dir
        )
    else:
        logger.error(f"Unknown framework: {framework}")
        raise ValueError(f"Unknown framework: {framework}")


def list_tests(profile: str, build_dir: Path | None = None) -> set[Test]:
    """
    List all tests for a given profile (across all executables).

    Args:
        profile: Profile name from test-profiles.yaml
        build_dir: Optional build directory (for ctest)

    Returns:
        Set of Test objects from all executables in the profile
    """
    logger.info(f"Listing all tests for profile='{profile}'")

    # Load config
    script_dir = Path(__file__).resolve().parent
    config_file = script_dir.parent.parent.parent / "test-profiles.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Find profile
    if profile not in config.get("profiles", {}):
        logger.error(f"Profile '{profile}' not found in test-profiles.yaml")
        raise ValueError(f"Profile '{profile}' not found in test-profiles.yaml")

    profile_data = config["profiles"][profile]
    tests_list = profile_data.get("tests", [])

    # Collect tests from all executables
    all_tests = set()

    for test_entry in tests_list:
        executable = test_entry.get("executable")
        logger.debug(f"Processing executable: {executable}")

        try:
            tests = list_tests_for_executable(executable, profile, build_dir)
            all_tests.update(tests)
            logger.info(f"Found {len(tests)} tests from {executable}")
        except Exception as e:
            logger.warning(f"Failed to list tests from {executable}: {e}")

    logger.info(f"Total tests for profile '{profile}': {len(all_tests)}")
    return all_tests


def get_test_commands(profile_name: str, config_file: Path | None = None) -> dict:
    """
    Get command lines for all executables in a profile, organized by framework.

    Args:
        profile_name: Name of the profile
        config_file: Optional path to test-profiles.yaml

    Returns:
        Dictionary mapping framework names to lists of commands.
        Each command is a list of arguments (e.g., ["test/rocroller-tests", "--gtest_filter=..."]).

        Example:
        {
            "gtest": [["test/rocroller-tests", "--gtest_filter=..."]],
            "catch2": [["test/rocroller-tests-catch", "filter"]],
            "ctest": [["ctest", "-R", "pattern"]]
        }
    """
    # Load config
    if config_file is None:
        script_dir = Path(__file__).resolve().parent
        config_file = script_dir.parent.parent.parent / "test-profiles.yaml"

    logger.debug(f"Loading config from: {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Find profile
    if profile_name not in config.get("profiles", {}):
        logger.error(f"Profile '{profile_name}' not found")
        raise ValueError(f"Profile '{profile_name}' not found")

    logger.info(f"Generating test commands for profile '{profile_name}'")
    profile_data = config["profiles"][profile_name]
    tests_list = profile_data.get("tests", [])

    # Commands organized by framework
    commands = {}

    for test_entry in tests_list:
        executable = test_entry.get("executable")
        framework = test_entry.get("framework")
        include = test_entry.get("include", [])
        exclude = test_entry.get("exclude", [])

        # Build base command
        cmd = [executable]

        # Add framework-specific arguments and filters
        if framework == "gtest":
            filter_args = build_gtest_filter_args(include, exclude)
            cmd.extend(filter_args)

        elif framework == "catch2":
            filter_args = build_catch2_filter_args(include, exclude)
            cmd.extend(filter_args)

        elif framework == "ctest":
            filter_args = build_ctest_filter_args(include, exclude)
            cmd.extend(filter_args)

        logger.debug(f"Command for {executable}: {' '.join(cmd)}")

        # Add to commands dict under framework key
        if framework not in commands:
            commands[framework] = []
        commands[framework].append(cmd)

    logger.info(f"Generated commands for {len(commands)} framework(s)")
    return commands
