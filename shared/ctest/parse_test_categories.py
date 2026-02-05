import yaml
import sys
import re
import platform
import argparse


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

    args = parser.parse_args()

    yaml_file = args.yaml_file
    target_name = args.target_name
    working_dir = args.working_dir
    install_test_file = args.install_test_file

    try:
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML: {e}", file=sys.stderr)
        sys.exit(1)

    # Open install test file if provided
    install_file_handle = None
    if install_test_file:
        try:
            install_file_handle = open(
                install_test_file, "a", buffering=1
            )  # Line buffered
            print(
                f"# DEBUG: Opened install test file: {install_test_file}",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"Warning: Could not open install test file {install_test_file}: {e}",
                file=sys.stderr,
            )
            install_file_handle = None
    else:
        print(f"# DEBUG: No install test file provided", file=sys.stderr)

    categories = config.get("test_categories", {})
    execution_settings = config.get("execution_settings", {})
    timeouts = execution_settings.get("category_timeouts", {})
    timeout_multiplier = execution_settings.get("timeout_multiplier", 1)
    exclude_gpu_config = config.get("exclude_gpu", {})

    # Detect OS
    is_windows = platform.system() == "Windows"
    is_linux = platform.system() == "Linux"

    print("# Generated CMake code for test categories")
    print(f"# Detected OS: {platform.system()}")
    print(f"# Timeout multiplier: {timeout_multiplier}")

    # Store category information for later use with GPU exclusions
    category_data = {}

    for category_name, category_info in categories.items():
        patterns = category_info.get("test_patterns", [])
        labels = category_info.get("labels", [])
        exclude = category_info.get("exclude", [])
        if exclude == None:
            exclude = []

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

        # Build positive pattern string
        positive_string = ""
        for pattern in patterns:
            positive_string = positive_string + ":" + pattern
        positive_string = positive_string[1:]  # Remove leading colon

        # Build negative pattern string for exclusions
        exclude_string = ""
        if exclude:
            for excluded_pattern in exclude:
                exclude_string = exclude_string + ":" + excluded_pattern
            exclude_string = exclude_string[1:]  # Remove leading colon

        # Store positive and exclude strings separately for GPU exclusion processing
        category_data[category_name] = {
            "positive_string": positive_string,
            "exclude_string": exclude_string,
            "labels": labels[:],  # Make a copy
            "timeout": timeout,
        }

        # Build complete pattern string for this category test
        if exclude_string:
            pattern_string = positive_string + "-" + exclude_string
        else:
            pattern_string = positive_string

        label_string = ""
        for label in labels:
            label_string = label_string + ";" + label
        label_string = '"' + label_string[1:] + '"'
        print("add_test(")
        print(f"  NAME {target_name}-{category_name}-suite")
        print(f"  COMMAND {target_name} --gtest_filter={pattern_string}")
        print(f"  WORKING_DIRECTORY {working_dir}")
        print(")")

        print(f"set_tests_properties({target_name}-{category_name}-suite PROPERTIES")
        print(f"  LABELS {label_string}")
        print(f"  TIMEOUT {timeout}")
        print(")")
        print()

        # Write install-time test with relative path if install file is provided
        if install_file_handle:
            try:
                print(
                    f"# DEBUG: Writing category test {category_name}", file=sys.stderr
                )
                install_file_handle.write(
                    f'add_test({target_name}-{category_name}-suite "../{target_name}" --gtest_filter={pattern_string})\n'
                )
                install_file_handle.write(
                    f"set_tests_properties({target_name}-{category_name}-suite PROPERTIES LABELS {label_string} TIMEOUT {timeout})\n\n"
                )
                install_file_handle.flush()
            except Exception as e:
                print(
                    f"Warning: Failed to write category {category_name} to install test file: {e}",
                    file=sys.stderr,
                )

    # ========================================================================
    # GPU Exclusion Tests with Hierarchical Pattern Matching
    # ========================================================================
    #
    # This section generates GPU-specific exclusion tests
    #
    # Hierarchical Matching:
    # - Uses wildcard 'X' for pattern matching (e.g., gfx11X matches gfx1100, gfx1150, etc.)
    # - More specific GPUs inherit exclusions from general patterns
    # - Example: gfx1150 will exclude patterns from BOTH:
    #   * exclude_gpu_gfx11X (general gfx11 family)
    #   * exclude_gpu_gfx1150 (specific to gfx1150)
    #
    # Generated Tests:
    # - One test per category (quick, standard, etc.) per unique ex_gpu_* label
    # - Test name format: {target_name}-{category}-{gpu_arch}-suite
    # - Uses gtest filter: "{category_patterns}:-{gpu_exclusion_patterns}"
    # - Labels include both category labels and ex_gpu_* label
    #
    # Usage Examples:
    # - On gfx1150 hardware:
    #   ctest -L quick -L ex_gpu_gfx1150
    #   (runs ONLY the gfx1150 GPU exclusion test with gfx11X and gfx1150 patterns excluded)
    #
    # - On gfx950 hardware:
    #   ctest -L quick -L ex_gpu_gfx950
    #   (runs ONLY the gfx950 GPU exclusion test with gfx950 patterns excluded)
    #
    # - On generic/other hardware:
    #   ctest -L quick -LE ex_gpu
    #   (runs main quick suite with ALL patterns included, excludes GPU-specific tests)
    #
    # Note: Using "-L quick" alone will run BOTH main suite and ALL GPU exclusion tests
    # ========================================================================

    def gpu_arch_matches(specific_arch, pattern_arch):
        """
        Check if a specific GPU architecture matches a pattern with X wildcards.
        E.g., gfx1150 matches gfx1150 (exact), gfx115X, gfx11X, etc.
        X acts as a wildcard for any remaining characters after that point.
        """
        if specific_arch == pattern_arch:
            return True

        # Check if pattern_arch has X wildcards
        if "X" not in pattern_arch:
            return False

        # X means "any characters after this point"
        # So gfx11X matches gfx110, gfx1150, gfx1151, etc.
        # Split at the first X and check if specific_arch starts with the prefix
        prefix = pattern_arch.split("X")[0]
        return specific_arch.startswith(prefix)

    # Collect all ex_gpu labels and their corresponding GPU architectures
    # We need to process each unique ex_gpu_* label
    ex_gpu_labels_to_process = set()
    for gpu_key, gpu_config in exclude_gpu_config.items():
        match = re.match(r"exclude_gpu_(gfx\w+)", gpu_key)
        if match:
            gpu_labels = gpu_config.get("labels", [])
            for label in gpu_labels:
                if label.startswith("ex_gpu_"):
                    ex_gpu_labels_to_process.add(label)

    # Process top-level exclude_gpu section
    # For each unique ex_gpu label, create tests with hierarchical pattern matching
    for ex_gpu_label in ex_gpu_labels_to_process:
        # Extract the GPU architecture from the label (e.g., ex_gpu_gfx1150 -> gfx1150)
        gpu_arch = ex_gpu_label.replace("ex_gpu_", "")

        # Collect all patterns that apply to this GPU architecture
        # This includes exact matches and hierarchical matches (e.g., gfx1150 matches gfx115X, gfx11X)
        all_applicable_patterns = []
        all_applicable_categories = set()

        for gpu_key, gpu_config in exclude_gpu_config.items():
            match = re.match(r"exclude_gpu_(gfx\w+)", gpu_key)
            if not match:
                continue

            config_arch = match.group(1)

            # Check if this config applies to our target GPU architecture
            if gpu_arch_matches(gpu_arch, config_arch):
                patterns = gpu_config.get("test_patterns", [])
                if patterns:
                    all_applicable_patterns.extend(patterns)

                # Collect applicable categories from this config
                gpu_labels = gpu_config.get("labels", [])
                for label in gpu_labels:
                    if label in category_data:
                        all_applicable_categories.add(label)

        if not all_applicable_patterns:
            continue

        # Remove duplicates from all_applicable_patterns while preserving order
        seen = set()
        unique_patterns = []
        for pattern in all_applicable_patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)

        # Build GPU exclusion pattern string - format: pattern1:pattern2
        gpu_exclude_string = ""
        for pattern in unique_patterns:
            gpu_exclude_string = gpu_exclude_string + ":" + pattern
        gpu_exclude_string = gpu_exclude_string[1:]  # Remove leading colon

        # Create one test for each applicable category
        for category_name in all_applicable_categories:
            cat_data = category_data[category_name]
            positive_string = cat_data["positive_string"]
            cat_exclude_string = cat_data["exclude_string"]
            cat_labels = cat_data["labels"]
            timeout = cat_data["timeout"]

            # Build combined pattern string: positive - category_excludes:gpu_excludes
            # Combine all negative patterns
            combined_exclude_string = ""
            if cat_exclude_string:
                combined_exclude_string = cat_exclude_string + ":" + gpu_exclude_string
            else:
                combined_exclude_string = gpu_exclude_string

            pattern_string = positive_string + "-" + combined_exclude_string

            # Build label string: category_labels + ex_gpu_<arch> label
            combined_labels = cat_labels + [ex_gpu_label]
            label_string = ""
            for label in combined_labels:
                label_string = label_string + ";" + label
            label_string = '"' + label_string[1:] + '"'

            print(f"# GPU exclusion for {gpu_arch} - {category_name} category")
            print("add_test(")
            print(f"  NAME {target_name}-{category_name}-{gpu_arch}-suite")
            print(f"  COMMAND {target_name} --gtest_filter={pattern_string}")
            print(f"  WORKING_DIRECTORY {working_dir}")
            print(")")

            print(
                f"set_tests_properties({target_name}-{category_name}-{gpu_arch}-suite PROPERTIES"
            )
            print(f"  LABELS {label_string}")
            print(f"  TIMEOUT {timeout}")
            print(")")
            print()

            # Write install-time test with relative path if install file is provided
            if install_file_handle:
                try:
                    print(
                        f"# DEBUG: Writing GPU exclude test {category_name}-{gpu_arch}",
                        file=sys.stderr,
                    )
                    install_file_handle.write(
                        f'add_test({target_name}-{category_name}-{gpu_arch}-suite "../{target_name}" --gtest_filter={pattern_string})\n'
                    )
                    install_file_handle.write(
                        f"set_tests_properties({target_name}-{category_name}-{gpu_arch}-suite PROPERTIES LABELS {label_string} TIMEOUT {timeout})\n\n"
                    )
                    install_file_handle.flush()
                except Exception as e:
                    print(
                        f"Warning: Failed to write GPU exclude {category_name}-{gpu_arch} to install test file: {e}",
                        file=sys.stderr,
                    )

    # Close install test file if it was opened
    if install_file_handle:
        try:
            install_file_handle.flush()  # Ensure all data is written
            install_file_handle.close()
            print(f"# DEBUG: Closed install test file successfully", file=sys.stderr)
        except Exception as e:
            print(f"# DEBUG: Error closing install test file: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
