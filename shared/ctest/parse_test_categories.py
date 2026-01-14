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
        "target_name", help="Name of the test target (e.g., hipblas-test)"
    )
    parser.add_argument("working_dir", help="Working directory for running tests")

    args = parser.parse_args()

    yaml_file = args.yaml_file
    target_name = args.target_name
    working_dir = args.working_dir

    try:
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML: {e}", file=sys.stderr)
        sys.exit(1)

    categories = config.get("test_categories", {})
    timeouts = config.get("execution_settings", {}).get("category_timeouts", {})

    # Detect OS
    is_windows = platform.system() == "Windows"
    is_linux = platform.system() == "Linux"

    print("# Generated CMake code for test categories")
    print(f"# Detected OS: {platform.system()}")

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

        # Collect GPU-specific exclusions
        gpu_excludes = {}
        for key in category_info.keys():
            match = re.match(r"exclude_gpu_(gfx\w+)", key)
            if match:
                gpu_arch = match.group(1)
                gpu_patterns = category_info.get(key, [])
                if gpu_patterns:
                    gpu_excludes[gpu_arch] = gpu_patterns
                    # Add these patterns to the general exclude list
                    exclude.extend(gpu_patterns)

        timeout = timeouts.get(category_name, 300)
        print(f"# Category: {category_name}")
        print(f'# Description: {category_info.get("description", "")}')
        patterns[:] = [x for x in patterns if x not in exclude]

        pattern_string = ""
        for pattern in patterns:
            pattern_string = pattern_string + ":" + pattern
        pattern_string = '"' + pattern_string[1:] + '"'

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

        # Generate tests for GPU-specific exclusions
        # Taking miopen as an example, if it has got exclude_gpu_gfx90a for the quick category,
        # a new test with excluded patterns will be added  with name miopen_gtest-quick-gfx90a-exclude
        # and an additional label ex_gpu_gfx90a. These patterns will be removed from quick category.
        # The tests are run with 'ctest -L quick -LE ex_gpu_gfx90a' when run from gfx90a.
        # For other platforms 'ctest -L quick' will be used, which will run tests from both 'quick' and
        # 'exclude_gpu_gfx90a' categories (tests miopen_gtest-quick-suite and miopen_gtest-quick-gfx90a-exclude).

        for gpu_arch, gpu_patterns in gpu_excludes.items():
            gpu_pattern_string = ""
            for pattern in gpu_patterns:
                gpu_pattern_string = gpu_pattern_string + ":" + pattern
            gpu_pattern_string = '"' + gpu_pattern_string[1:] + '"'

            # Create labels with category name and ex_gpu_<arch>
            # gpu_label_string = f'"{category_name};ex_gpu_{gpu_arch}"'
            gpu_label_string = label_string[:-1] + f';ex_gpu_{gpu_arch}"'

            print(f"# GPU-specific exclusion for {gpu_arch}")
            print("add_test(")
            print(f"  NAME {target_name}-{category_name}-{gpu_arch}-exclude")
            print(f"  COMMAND {target_name} --gtest_filter={gpu_pattern_string}")
            print(f"  WORKING_DIRECTORY {working_dir}")
            print(")")

            print(
                f"set_tests_properties({target_name}-{category_name}-{gpu_arch}-exclude PROPERTIES"
            )
            print(f"  LABELS {gpu_label_string}")
            print(f"  TIMEOUT {timeout}")
            # print(f"  DISABLED TRUE")
            print(")")
            print()


if __name__ == "__main__":
    main()
