import yaml
import sys
import re
import platform
import argparse
import contextlib
import shlex


def _format_extra_args(extra_args):
    """Format a list of extra command-line args for CMake add_test().

    Each arg is shell-quoted with shlex.quote so that values containing spaces
    or shell metacharacters are preserved as a single argument by CTest. Empty
    or None inputs produce an empty string (no leading space).
    """
    if not extra_args:
        return ""
    if isinstance(extra_args, str):
        # Allow YAML authors to write a single string instead of a list.
        extra_args = shlex.split(extra_args)
    return " " + " ".join(shlex.quote(str(a)) for a in extra_args)


def _cmake_quote(value):
    """Quote one value as a CMake bracket argument."""
    text = str(value)
    for level in range(10):
        marker = "=" * level
        if f"]{marker}]" not in text:
            return f"[{marker}[{text}]{marker}]"
    raise ValueError("Unable to quote CMake argument")


def _format_cmake_args(args):
    """Format a list of command arguments for generated CMake add_test()."""
    if not args:
        return ""
    return " " + " ".join(_cmake_quote(arg) for arg in args)


def _is_match_everything(pattern):
    """True if a gtest pattern matches every test name.

    `*` matches any (possibly empty) substring, so a pattern made only of `*`
    is match-all -- `**` is just a clumsier `*`. `*.*` is also match-all in
    practice: every gtest full name is `Suite.Test`, so the dot always matches.
    Such a pattern under `exclude_gpu` means "exclude the whole suite", which a
    gtest filter cannot express -- see the DISABLED handling in main().
    """
    if not isinstance(pattern, str) or not pattern:
        return False
    return set(pattern) == {"*"} or pattern == "*.*"


def _dedupe_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _rtest_script_basename(name_prefix: str) -> str:
    """Basename of the rtest driver script next to the gtest binary.

    ``name_prefix`` is normally the gtest executable name used in CTest (for
    example ``mylib-test``). The co-installed rtest driver is expected to be
    ``<stem>_rtest.py`` where ``stem`` is ``name_prefix`` with a trailing
    ``-test`` removed (``mylib-test`` -> ``mylib_rtest.py``). If there is no
    ``-test`` suffix, hyphens in ``name_prefix`` are turned into underscores and
    ``_rtest.py`` is appended.
    """
    if name_prefix.endswith("-test"):
        return name_prefix[: -len("-test")] + "_rtest.py"
    stem = name_prefix.replace("-", "_")
    return stem + "_rtest.py"


def _format_gtest_command_tail(
    executable,
    pattern_string,
    extra_args_string,
    command_args_string="",
    test_yaml=None,
):
    """Gtest binary invocation: optional args, --yaml, --gtest_filter, then extra_args."""
    tail = f"{executable}{command_args_string}"
    if test_yaml:
        tail += f" --yaml {shlex.quote(test_yaml)}"
    if pattern_string:
        tail += f" --gtest_filter={pattern_string}"
    return f"{tail}{extra_args_string}"


def _ctest_python_command(is_windows):
    """Python executable name for CTest rtest invocations at run time.

    Windows Python installs typically expose ``python`` on PATH, not ``python3``.
    """
    return "python" if is_windows else "python3"


def _format_category_command(
    use_rtest_driver,
    name_prefix,
    target_name,
    pattern_string,
    extra_args_string,
    category_name,
    test_yaml=None,
    command_args_string="",
    is_windows=False,
):
    """Return the COMMAND tail for add_test (everything after COMMAND)."""
    if use_rtest_driver:
        rtest_script = _rtest_script_basename(name_prefix)
        rtest_set = f"ctest_{category_name}"
        py = _ctest_python_command(is_windows)
        return f"{py} {rtest_script} -t {rtest_set}{extra_args_string}"
    return _format_gtest_command_tail(
        target_name,
        pattern_string,
        extra_args_string,
        command_args_string,
        test_yaml,
    )


def _format_install_add_test_line(
    use_rtest_driver,
    name_prefix,
    category_name,
    target_name,
    pattern_string,
    extra_args_string,
    cmake_python3,
    test_yaml=None,
    gpu_arch=None,
    install_executable=None,
    install_command_args_string="",
    is_windows=False,
):
    """One-line add_test(...) for install-time CTestTestfile fragments."""
    suffix = f"_{gpu_arch}" if gpu_arch else ""
    test_name = f"{name_prefix}_{category_name}{suffix}_suite"
    if use_rtest_driver:
        py = (
            shlex.quote(cmake_python3)
            if cmake_python3
            else _ctest_python_command(is_windows)
        )
        rtest_script = _rtest_script_basename(name_prefix)
        rtest_set = f"ctest_{category_name}"
        return (
            f'add_test({test_name} {py} "../{rtest_script}" '
            f"-t {rtest_set}{extra_args_string})\n"
        )
    exe = install_executable if install_executable is not None else f"../{target_name}"
    gtest_tail = _format_gtest_command_tail(
        _cmake_quote(exe),
        pattern_string,
        extra_args_string,
        install_command_args_string,
        test_yaml,
    )
    return f"add_test({test_name} {gtest_tail})\n"


# Allowlist patterns for YAML-sourced values
_IDENTIFIER_RE = re.compile(r"^[\w\-\.]+$")
_GTEST_PATTERN_RE = re.compile(r"^[\w\*\.\-/]+$")


def validate_identifier(value):
    """Validate that a value is a safe identifier (alphanumerics, hyphens, dots, underscores).

    Returns an error message string on failure, or None on success.
    """
    if not isinstance(value, str):
        return f"Identifier must be a string, got {type(value).__name__}: {value!r}"
    if not _IDENTIFIER_RE.match(value):
        return (
            f"Identifier contains unsafe characters: {value!r} "
            f"(only alphanumerics, hyphens, dots, and underscores allowed)"
        )
    return None


def validate_gtest_pattern(pattern):
    """Validate that a gtest filter pattern contains only safe characters.

    Returns an error message string on failure, or None on success.
    """
    if not isinstance(pattern, str):
        return f"Pattern must be a string, got {type(pattern).__name__}: {pattern!r}"
    if not _GTEST_PATTERN_RE.match(pattern):
        return (
            f"Invalid gtest pattern: {pattern!r} "
            f"(only alphanumerics, wildcards, dots, hyphens, underscores, and slashes allowed)"
        )
    return None


def validate_config(categories, exclude_gpu_config, is_windows, is_linux):
    """Validate all category and GPU-exclusion entries.

    Returns a list of error message strings; empty if everything is valid.
    All issues are collected so the caller can report them at once rather
    than failing on the first one.
    """
    errors = []

    if not isinstance(categories, dict):
        errors.append(
            f"test_categories must be a mapping, got {type(categories).__name__}"
        )
    else:
        for category_name, category_info in categories.items():
            err = validate_identifier(category_name)
            if err is not None:
                errors.append(f"category name {category_name!r}: {err}")

            if not isinstance(category_info, dict):
                errors.append(
                    f"category {category_name!r}: entry must be a mapping, got "
                    f"{type(category_info).__name__}"
                )
                continue

            patterns = category_info.get("test_patterns", []) or []
            exclude = category_info.get("exclude", []) or []
            if is_windows:
                exclude = exclude + (category_info.get("exclude_windows", []) or [])
            if is_linux:
                exclude = exclude + (category_info.get("exclude_linux", []) or [])

            for p in patterns:
                err = validate_gtest_pattern(p)
                if err is not None:
                    errors.append(f"category {category_name!r} test_patterns: {err}")
            for e in exclude:
                err = validate_gtest_pattern(e)
                if err is not None:
                    errors.append(f"category {category_name!r} exclude: {err}")
            for label in category_info.get("labels", []) or []:
                err = validate_identifier(label)
                if err is not None:
                    errors.append(f"category {category_name!r} label: {err}")

            test_yaml = category_info.get("test_yaml")
            if test_yaml is not None:
                err = validate_identifier(test_yaml)
                if err is not None:
                    errors.append(f"category {category_name!r} test_yaml: {err}")

            if not patterns and not test_yaml:
                errors.append(
                    f"category {category_name!r}: must define test_patterns and/or test_yaml"
                )

    if exclude_gpu_config is None:
        return errors
    if not isinstance(exclude_gpu_config, dict):
        errors.append(
            f"exclude_gpu must be a mapping, got {type(exclude_gpu_config).__name__}"
        )
        return errors

    for gpu_key, gpu_config in exclude_gpu_config.items():
        err = validate_identifier(gpu_key)
        if err is not None:
            errors.append(f"exclude_gpu key {gpu_key!r}: {err}")

        if not isinstance(gpu_config, dict):
            errors.append(
                f"exclude_gpu {gpu_key!r}: entry must be a mapping, got "
                f"{type(gpu_config).__name__}"
            )
            continue

        for p in gpu_config.get("test_patterns", []) or []:
            # test_patterns may be either a flat list or list-of-lists.
            sub_patterns = p if isinstance(p, list) else [p]
            for sp in sub_patterns:
                err = validate_gtest_pattern(sp)
                if err is not None:
                    errors.append(f"exclude_gpu {gpu_key!r} test_patterns: {err}")

        for label in gpu_config.get("labels", []) or []:
            err = validate_identifier(label)
            if err is not None:
                errors.append(f"exclude_gpu {gpu_key!r} label: {err}")

    return errors


def gpu_arch_matches(specific_arch, pattern_arch):
    """
    Check if a specific GPU architecture matches a pattern with X wildcards.
    E.g., gfx1150 matches gfx1150 (exact), gfx115X, gfx11X, etc.
    """
    if specific_arch == pattern_arch:
        return True

    # Check if pattern_arch has X wildcards
    if "X" not in pattern_arch:
        return False

    # Split at the first X and check if specific_arch starts with the prefix
    prefix = pattern_arch.split("X")[0]
    return specific_arch.startswith(prefix)


# Recognised key shapes:
#   exclude_gpu_<arch>             -> OS-agnostic
#   exclude_gpu_<arch>_windows     -> only when configuring on Windows
#   exclude_gpu_<arch>_linux       -> only when configuring on Linux
_EXCLUDE_GPU_KEY_RE = re.compile(r"^exclude_gpu_(gfx\w+?)(?:_(windows|linux))?$")


def parse_exclude_gpu_key(key):
    """Return (gpu_arch, os_suffix) for a YAML key under ``exclude_gpu``.

    ``os_suffix`` is one of ``"windows"``, ``"linux"`` or ``None``.
    Returns ``(None, None)`` if the key does not match the expected shape.
    """
    m = _EXCLUDE_GPU_KEY_RE.match(key)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def exclude_gpu_key_applies(os_suffix, is_windows, is_linux):
    """Return True if a key with the given OS suffix should be honoured on the
    current host. ``None`` means OS-agnostic and always applies. eg component: rocprim
    """
    if os_suffix is None:
        return True
    if os_suffix == "windows":
        return is_windows
    if os_suffix == "linux":
        return is_linux
    return False


def load_yaml(yaml_file):
    """Load and parse a YAML file, exiting with a descriptive error on failure."""
    try:
        with open(yaml_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: YAML file not found: {yaml_file}", file=sys.stderr)
    except PermissionError:
        print(
            f"Error: Permission denied reading YAML file: {yaml_file}", file=sys.stderr
        )
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax in {yaml_file}: {e}", file=sys.stderr)
    except Exception as e:
        print(
            f"Error: Unexpected failure loading {yaml_file}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Parse test_categories.yaml and generate CMake test definitions"
    )
    parser.add_argument("yaml_file", help="Path to the test_categories.yaml file")
    parser.add_argument(
        "target_name", help="Name of the test target (e.g., miopen_gtest)"
    )
    parser.add_argument("working_dir", help="Working directory for running tests")
    parser.add_argument(
        "install_test_file",
        nargs="?",
        default=None,
        help="Optional: Path to write install-time test definitions with relative paths",
    )
    parser.add_argument(
        "--resource-group",
        default=None,
        help=(
            "Optional CTest RESOURCE_GROUPS token (e.g. 'gfx942' or 'gpus'). "
            "When set, generated test names get a '_<resource>' segment after "
            'the target name and each suite gets RESOURCE_GROUPS "1,<resource>:1" applied.'
        ),
    )
    parser.add_argument(
        "--test-name-prefix",
        default=None,
        help="Optional prefix for generated CTest names. Defaults to target_name.",
    )
    parser.add_argument(
        "--command-arg",
        action="append",
        default=[],
        help="Additional build-tree command argument. May be repeated.",
    )
    parser.add_argument(
        "--install-command-arg",
        action="append",
        default=[],
        help="Additional install-tree command argument. May be repeated.",
    )
    parser.add_argument(
        "--install-executable",
        default=None,
        help="Install-tree executable path. Defaults to ../target_name.",
    )
    parser.add_argument(
        "--additional-label",
        action="append",
        default=[],
        help="Additional CTest label to append to every generated suite. May be repeated.",
    )
    parser.add_argument(
        "--environment",
        action="append",
        default=[],
        help=(
            "Additional ENVIRONMENT entry (KEY=VALUE) applied to every generated "
            "suite, e.g. CMake-side TEST_ENVIRONMENT (ASAN symbolizer path, "
            "coverage LLVM_PROFILE_FILE). May be repeated. Overrides a "
            "same-keyed execution_settings.environment entry from the YAML."
        ),
    )
    parser.add_argument(
        "--environment-modification",
        action="append",
        default=[],
        dest="environment_modification",
        help=(
            "Additional ENVIRONMENT_MODIFICATION entry (e.g. "
            "PATH=path_list_prepend:<dir>) applied to every generated suite. May "
            "be repeated. Used to extend PATH for runtime DLL discovery without "
            "clobbering the inherited value."
        ),
    )
    parser.add_argument(
        "--fixtures-required",
        action="append",
        default=[],
        dest="fixtures_required",
        help=(
            "CTest FIXTURES_REQUIRED entry applied to every generated suite (e.g. "
            "a setup fixture that must run before the suite). May be repeated. "
            "Currently applied only to the build tree; not emitted into the install "
            "tree, since no use case requires it there yet."
        ),
    )
    parser.add_argument(
        "--use-rtest-driver",
        action="store_true",
        dest="use_rtest_driver",
        help=(
            "Generate CTest commands that run the rtest driver script "
            "(basename derived from name_prefix; typically mylib-test -> "
            "mylib_rtest.py) with -t ctest_<category> instead of invoking the "
            "gtest binary with --gtest_filter=... "
            '(requires matching <test sets="ctest_<category>"> entries in the '
            "project's rtest XML)."
        ),
    )
    parser.add_argument(
        "--cmake-python3",
        default=None,
        help="Absolute path to Python3 interpreter (for install-tree add_test lines).",
    )

    args = parser.parse_args()

    yaml_file = args.yaml_file
    target_name = args.target_name
    working_dir = args.working_dir
    install_test_file = args.install_test_file
    resource_group = args.resource_group
    use_rtest_driver = args.use_rtest_driver
    cmake_python3 = args.cmake_python3
    if args.test_name_prefix is not None:
        err = validate_identifier(args.test_name_prefix)
        if err is not None:
            print(f"Error: invalid --test-name-prefix value: {err}", file=sys.stderr)
            sys.exit(1)
    for label in args.additional_label:
        err = validate_identifier(label)
        if err is not None:
            print(f"Error: invalid --additional-label value: {err}", file=sys.stderr)
            sys.exit(1)
    cli_env_overrides = {}
    for entry in args.environment:
        if "=" not in entry:
            print(
                f"Error: invalid --environment value {entry!r}: expected KEY=VALUE",
                file=sys.stderr,
            )
            sys.exit(1)
        key, value = entry.split("=", 1)
        cli_env_overrides[key] = value
    if resource_group is not None:
        err = validate_identifier(resource_group)
        if err is not None:
            print(f"Error: invalid --resource-group value: {err}", file=sys.stderr)
            sys.exit(1)
    base_name_prefix = args.test_name_prefix if args.test_name_prefix else target_name
    name_prefix = (
        f"{base_name_prefix}_{resource_group}" if resource_group else base_name_prefix
    )
    resource_groups_prop = (
        f' RESOURCE_GROUPS "1,{resource_group}:1"' if resource_group else ""
    )
    command_args_string = _format_cmake_args(args.command_arg)
    install_command_args_string = _format_cmake_args(args.install_command_arg)
    install_executable = (
        args.install_executable if args.install_executable else f"../{target_name}"
    )

    config = load_yaml(yaml_file)

    # Open install test file if provided, using context manager for automatic cleanup
    try:
        install_cm = (
            open(install_test_file, "a", buffering=1)
            if install_test_file
            else contextlib.nullcontext()
        )
    except OSError as e:
        print(
            f"Warning: I/O error opening install test file {install_test_file}: {e}",
            file=sys.stderr,
        )
        install_cm = contextlib.nullcontext()
    except Exception as e:
        print(
            f"Warning: Unexpected error opening install test file {install_test_file}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        install_cm = contextlib.nullcontext()

    with install_cm as install_file_handle:
        # ===============================================================================================================
        # Parse the YAML file, add excludes (including OS-specific), and write tests to CMake and install file.
        # ===============================================================================================================
        categories = config.get("test_categories", {})
        execution_settings = config.get("execution_settings", {})
        timeouts = execution_settings.get("category_timeouts", {})
        timeout_multiplier = execution_settings.get("timeout_multiplier", 1)
        env_dict = dict(execution_settings.get("environment", {}) or {})
        env_dict.update(cli_env_overrides)
        env_string = (
            ";".join(f"{k}={v}" for k, v in env_dict.items()) if env_dict else None
        )
        env_mod_string = (
            ";".join(args.environment_modification)
            if args.environment_modification
            else None
        )
        fixtures_string = (
            ";".join(args.fixtures_required) if args.fixtures_required else None
        )
        exclude_gpu_config = config.get("exclude_gpu", {})

        # Detect OS
        is_windows = platform.system() == "Windows"
        is_linux = platform.system() == "Linux"

        # Validate the categories before generating CMake code.
        # If validation fails, no partial or intermediate CMake file will be written.
        validation_errors = validate_config(
            categories, exclude_gpu_config, is_windows, is_linux
        )
        if validation_errors:
            print(
                f"Error: {len(validation_errors)} validation error(s) in {yaml_file}:",
                file=sys.stderr,
            )
            for msg in validation_errors:
                print(f"  - {msg}", file=sys.stderr)
            sys.exit(1)

        print("# Generated CMake code for test categories")
        print(f"# Detected OS: {platform.system()}")
        print(f"# Timeout multiplier: {timeout_multiplier}")

        # Store category information for later use with GPU exclusions
        category_data = {}

        for category_name, category_info in categories.items():
            patterns = category_info.get("test_patterns", []) or []
            test_yaml = category_info.get("test_yaml")
            if not patterns and not test_yaml:
                print(
                    f"Warning: Category '{category_name}' has no test_patterns or test_yaml defined, skipping.",
                    file=sys.stderr,
                )
                continue
            labels = category_info.get("labels", [])
            exclude = category_info.get("exclude", [])
            if exclude is None:
                exclude = []
            extra_args = category_info.get("extra_args", []) or []

            # Add OS-specific exclusions
            if is_windows:
                exclude_windows = category_info.get("exclude_windows", [])
                if exclude_windows:
                    exclude.extend(exclude_windows)

            if is_linux:
                exclude_linux = category_info.get("exclude_linux", [])
                if exclude_linux:
                    exclude.extend(exclude_linux)

            base_timeout = timeouts.get(category_name, 300)
            timeout = int(base_timeout * timeout_multiplier)
            print(f"# Category: {category_name}")
            print(f'# Description: {category_info.get("description", "")}')

            # Build positive pattern string and exclude string
            positive_string = ":".join(patterns)
            exclude_string = ":".join(exclude) if exclude else ""

            # Store positive and exclude strings separately for GPU exclusion processing
            category_data[category_name] = {
                "positive_string": positive_string,
                "exclude_string": exclude_string,
                "labels": labels[:],  # Make a copy
                "timeout": timeout,
                "test_yaml": test_yaml,
                "extra_args": (
                    list(extra_args) if isinstance(extra_args, list) else extra_args
                ),
            }
            extra_args_string = _format_extra_args(extra_args)

            # Build complete pattern string for this category test
            if exclude_string:
                pattern_string = positive_string + "-" + exclude_string
            else:
                pattern_string = positive_string

            combined_labels = _dedupe_preserve_order(labels + args.additional_label)
            label_string = '"' + ";".join(combined_labels) + '"'

            # =======================================================================
            # Write category test to CMake file and install file.
            # =======================================================================
            print("add_test(")
            print(f"  NAME {name_prefix}_{category_name}_suite")
            cmd_tail = _format_category_command(
                use_rtest_driver,
                name_prefix,
                target_name,
                pattern_string,
                extra_args_string,
                category_name,
                test_yaml,
                command_args_string,
                is_windows=is_windows,
            )
            print(f"  COMMAND {cmd_tail}")
            print(f"  WORKING_DIRECTORY {working_dir}")
            print(")")

            print(
                f"set_tests_properties({name_prefix}_{category_name}_suite PROPERTIES"
            )
            print(f"  LABELS {label_string}")
            print(f"  TIMEOUT {timeout}")
            if env_string:
                print(f'  ENVIRONMENT "{env_string}"')
            if env_mod_string:
                print(f'  ENVIRONMENT_MODIFICATION "{env_mod_string}"')
            if fixtures_string:
                print(f'  FIXTURES_REQUIRED "{fixtures_string}"')
            if resource_group:
                print(f'  RESOURCE_GROUPS "1,{resource_group}:1"')
            print(")")
            print()

            # Write install-time test with relative path if install file is provided
            if install_file_handle:
                try:
                    install_file_handle.write(
                        _format_install_add_test_line(
                            use_rtest_driver,
                            name_prefix,
                            category_name,
                            target_name,
                            pattern_string,
                            extra_args_string,
                            cmake_python3,
                            test_yaml,
                            install_executable=install_executable,
                            install_command_args_string=install_command_args_string,
                            is_windows=is_windows,
                        )
                    )
                    env_prop = f' ENVIRONMENT "{env_string}"' if env_string else ""
                    env_mod_prop = (
                        f' ENVIRONMENT_MODIFICATION "{env_mod_string}"'
                        if env_mod_string
                        else ""
                    )
                    # FIXTURES_REQUIRED currently applies only to the build tree; no
                    # use case requires it in the install tree yet.
                    install_file_handle.write(
                        f"set_tests_properties({name_prefix}_{category_name}_suite PROPERTIES LABELS {label_string} TIMEOUT {timeout}{env_prop}{env_mod_prop}{resource_groups_prop})\n\n"
                    )
                    install_file_handle.flush()
                except OSError as e:
                    print(
                        f"Warning: I/O error writing category {category_name} to install test file: {e}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"Warning: Unexpected error writing category {category_name} to install test file: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )

        # ========================================================================
        # GPU Exclusion Tests with Hierarchical Pattern Matching
        # ========================================================================
        #
        # This section generates GPU-specific exclusion tests

        # - Uses wildcard 'X' for pattern matching (e.g., gfx11X matches gfx1100, gfx1150, etc.)

        # - Generates one test per category (quick, standard, etc.) per unique ex_gpu_* label
        # - Test name format: {target_name}_{category}_{gpu_arch}_suite
        # - Uses gtest filter: "{category_patterns}:-{gpu_exclusion_patterns}"
        # ========================================================================

        # Collect all ex_gpu labels and their corresponding GPU architectures

        ex_gpu_labels_to_process = set()
        for gpu_key, gpu_config in exclude_gpu_config.items():
            arch, os_suffix = parse_exclude_gpu_key(gpu_key)
            if arch is None:
                continue
            if not exclude_gpu_key_applies(os_suffix, is_windows, is_linux):
                print(f"# Skipping {gpu_key}: not applicable on {platform.system()}")
                continue
            gpu_labels = gpu_config.get("labels", [])
            for label in gpu_labels:
                if label.startswith("ex_gpu_"):
                    ex_gpu_labels_to_process.add(label)

        # For each unique ex_gpu label, create tests with hierarchical pattern matching
        # Sort to ensure consistent test order
        for ex_gpu_label in sorted(ex_gpu_labels_to_process):
            # Extract the GPU architecture from the label (e.g., ex_gpu_gfx1150 -> gfx1150)
            gpu_arch = ex_gpu_label.replace("ex_gpu_", "")

            # Collect all patterns that apply to this GPU architecture
            # This includes exact matches and hierarchical matches (e.g., gfx1150 matches gfx115X, gfx11X)
            all_applicable_patterns = []
            all_applicable_categories = set()
            # Categories a match-everything pattern applies to. Tracked per
            # category, not per arch: patterns from several matching keys are
            # unioned, so a key contributing "*" for one category must not
            # disable a category that only a narrower key labelled.
            fully_excluded_categories = set()

            for gpu_key, gpu_config in exclude_gpu_config.items():
                config_arch, os_suffix = parse_exclude_gpu_key(gpu_key)
                if config_arch is None:
                    continue
                if not exclude_gpu_key_applies(os_suffix, is_windows, is_linux):
                    continue

                # Check if this config applies to our target GPU architecture
                if gpu_arch_matches(gpu_arch, config_arch):
                    # test_patterns may be a flat list or a list-of-lists.
                    flat_patterns = []
                    for p in gpu_config.get("test_patterns", []) or []:
                        if isinstance(p, list):
                            flat_patterns.extend(p)
                        else:
                            flat_patterns.append(p)
                    all_applicable_patterns.extend(flat_patterns)

                    excludes_everything = any(
                        _is_match_everything(p) for p in flat_patterns
                    )

                    # Collect applicable categories from this config
                    gpu_labels = gpu_config.get("labels", [])
                    for label in gpu_labels:
                        if label in category_data:
                            all_applicable_categories.add(label)
                            if excludes_everything:
                                fully_excluded_categories.add(label)

            if not all_applicable_patterns:
                continue

            seen = set()
            unique_patterns = []
            for pattern in all_applicable_patterns:
                if pattern not in seen:
                    seen.add(pattern)
                    unique_patterns.append(pattern)

            # Build GPU exclusion pattern string - format: pattern1:pattern2
            gpu_exclude_string = ":".join(unique_patterns)

            # A match-everything exclusion means "this suite does not run on
            # this arch". That intent is NOT expressible as a gtest filter: the
            # emitted "<positive>-<excludes>:*" has a negative half matching
            # everything, so the suite still registers, still launches the
            # binary, and selects 0 tests. That read as a silent pass until a
            # zero-tests-ran guard turned it into a hard failure
            # (dnn-providers/integration-tests/src/main.cpp) -- see
            # https://github.com/ROCm/rocm-libraries/issues/11163.
            #
            # Emitting the suite DISABLED expresses it directly: CTest reports
            # "Not Run (Disabled)" and never launches the binary. Two properties
            # of DISABLED matter here, both verified against ctest 3.28:
            #   - `ctest --print-labels` still publishes a disabled test's
            #     labels, so arch discovery that greps for ex_gpu_<arch>
            #     (TheRock's test_runner.py) keeps finding this arch.
            #   - The skip stays visible in console output and in --output-junit
            #     (status="disabled"), rather than the suite vanishing.
            #
            # Note this does NOT make the selection non-empty: CTest counts only
            # enabled tests, so a selection of nothing but disabled suites still
            # prints "No tests were found!!!" -- exit 0 under the default
            # --no-tests=ignore, but exit 8 under --no-tests=error. Today the
            # gfx110X lane stays non-empty because the sibling
            # test_categories.yaml contributes enabled suites to the same
            # directory; neither consumer passes --no-tests=error.

            # Create one test for each applicable category
            for category_name in all_applicable_categories:
                cat_data = category_data[category_name]
                suite_disabled = category_name in fully_excluded_categories
                positive_string = cat_data["positive_string"]
                cat_exclude_string = cat_data["exclude_string"]
                cat_labels = cat_data["labels"]
                timeout = cat_data["timeout"]
                cat_extra_args_string = _format_extra_args(cat_data.get("extra_args"))
                cat_test_yaml = cat_data.get("test_yaml")

                # Build combined pattern string: positive - category_excludes:gpu_excludes
                combined_exclude_string = ""
                if cat_exclude_string:
                    combined_exclude_string = (
                        cat_exclude_string + ":" + gpu_exclude_string
                    )
                else:
                    combined_exclude_string = gpu_exclude_string

                pattern_string = positive_string + "-" + combined_exclude_string

                combined_labels = _dedupe_preserve_order(
                    cat_labels + args.additional_label + [ex_gpu_label]
                )
                label_string = '"' + ";".join(combined_labels) + '"'

                # =======================================================================
                # Write GPU exclusion tests to CMake file and install file.
                # =======================================================================
                print(f"# GPU exclusion for {gpu_arch} - {category_name} category")
                print("add_test(")
                print(f"  NAME {name_prefix}_{category_name}_{gpu_arch}_suite")
                # GPU-specific gtest slices are not represented in rtest XML; always use the binary.
                cmd_tail = _format_category_command(
                    False,
                    name_prefix,
                    target_name,
                    pattern_string,
                    cat_extra_args_string,
                    category_name,
                    cat_test_yaml,
                    command_args_string,
                    is_windows=is_windows,
                )
                print(f"  COMMAND {cmd_tail}")
                print(f"  WORKING_DIRECTORY {working_dir}")
                print(")")

                print(
                    f"set_tests_properties({name_prefix}_{category_name}_{gpu_arch}_suite PROPERTIES"
                )
                print(f"  LABELS {label_string}")
                print(f"  TIMEOUT {timeout}")
                if env_string:
                    print(f'  ENVIRONMENT "{env_string}"')
                if env_mod_string:
                    print(f'  ENVIRONMENT_MODIFICATION "{env_mod_string}"')
                if fixtures_string:
                    print(f'  FIXTURES_REQUIRED "{fixtures_string}"')
                if resource_group:
                    print(f'  RESOURCE_GROUPS "1,{resource_group}:1"')
                if suite_disabled:
                    print("  DISABLED TRUE")
                print(")")
                print()

                # Write install-time test with relative path if install file is provided
                if install_file_handle:
                    try:
                        install_file_handle.write(
                            _format_install_add_test_line(
                                False,
                                name_prefix,
                                category_name,
                                target_name,
                                pattern_string,
                                cat_extra_args_string,
                                cmake_python3,
                                cat_test_yaml,
                                gpu_arch=gpu_arch,
                                install_executable=install_executable,
                                install_command_args_string=install_command_args_string,
                                is_windows=is_windows,
                            )
                        )
                        env_prop = f' ENVIRONMENT "{env_string}"' if env_string else ""
                        env_mod_prop = (
                            f' ENVIRONMENT_MODIFICATION "{env_mod_string}"'
                            if env_mod_string
                            else ""
                        )
                        # FIXTURES_REQUIRED currently applies only to the build tree; no
                        # use case requires it in the install tree yet.
                        disabled_prop = " DISABLED TRUE" if suite_disabled else ""
                        install_file_handle.write(
                            f"set_tests_properties({name_prefix}_{category_name}_{gpu_arch}_suite PROPERTIES LABELS {label_string} TIMEOUT {timeout}{env_prop}{env_mod_prop}{resource_groups_prop}{disabled_prop})\n\n"
                        )
                        install_file_handle.flush()
                    except OSError as e:
                        print(
                            f"Warning: I/O error writing GPU exclude {category_name}_{gpu_arch} to install test file: {e}",
                            file=sys.stderr,
                        )
                    except Exception as e:
                        print(
                            f"Warning: Unexpected error writing GPU exclude {category_name}_{gpu_arch} to install test file: {type(e).__name__}: {e}",
                            file=sys.stderr,
                        )


if __name__ == "__main__":
    main()
