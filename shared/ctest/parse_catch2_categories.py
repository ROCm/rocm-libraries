import yaml
import sys
import re
import platform
import argparse
import contextlib

# Allowlist patterns for YAML-sourced values
_TAG_RE = re.compile(r"^~?\[[\w\-\.\*\/]*\]$")
_IDENTIFIER_RE = re.compile(r"^[\w\-\.]+$")


def validate_tag(tag):
    """Validate a tag matches Catch2 syntax: [name], ~[name], or [].

    Returns an error message string on failure, or None on success.
    """
    if not isinstance(tag, str):
        return f"Tag must be a string, got {type(tag).__name__}: {tag!r}"
    if tag == "[]":
        return None
    if not _TAG_RE.match(tag):
        return (
            f"Invalid tag syntax: {tag!r} "
            f"(expected pattern like [name], ~[name], or [])"
        )
    return None


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


def validate_categories(categories, is_windows, is_linux):
    """Validate all category names, tags, and labels.

    Returns a list of error message strings; empty if everything is valid.
    All validation issues are collected so the caller can report them at once
    rather than failing on the first one.
    """
    errors = []

    if not isinstance(categories, dict):
        errors.append(
            f"test_categories must be a mapping, got {type(categories).__name__}"
        )
        return errors

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

        test_tags = category_info.get("test_tags", []) or []
        exclude_tags = category_info.get("exclude_tags", []) or []
        if is_windows:
            exclude_tags = exclude_tags + (
                category_info.get("exclude_tags_windows", []) or []
            )
        if is_linux:
            exclude_tags = exclude_tags + (
                category_info.get("exclude_tags_linux", []) or []
            )

        for tag in test_tags:
            err = validate_tag(tag)
            if err is not None:
                errors.append(f"category {category_name!r} test_tags: {err}")
        for tag in exclude_tags:
            err = validate_tag(tag)
            if err is not None:
                errors.append(f"category {category_name!r} exclude_tags: {err}")

        for label in category_info.get("labels", []) or []:
            err = validate_identifier(label)
            if err is not None:
                errors.append(f"category {category_name!r} label: {err}")

    return errors


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


def build_catch2_tag_expression(test_tags, exclude_tags):
    """
    Build a Catch2 tag expression from include and exclude tag lists.

    Catch2 test spec grammar:
      - Comma ',' separates OR clauses
      - Space separates AND conditions within a clause
      - '~' negates the next condition

    Because comma binds looser than space, excludes must be duplicated
    across each include clause to get correct semantics:
      [a] ~[x],[b] ~[x]  means  ([a] AND NOT [x]) OR ([b] AND NOT [x])
    """
    include_tags = [tag for tag in test_tags if tag != "[]"] if test_tags else []

    exclude_parts = [f"~{tag}" for tag in exclude_tags] if exclude_tags else []

    exclude_str = " ".join(exclude_parts) if exclude_parts else ""

    if not include_tags:
        return exclude_str or ""

    if not exclude_str:
        return ",".join(include_tags)

    # Duplicate excludes per include clause for correct precedence
    clauses = []
    for tag in include_tags:
        clauses.append(f"{tag} {exclude_str}")
    return ",".join(clauses)


def main():
    parser = argparse.ArgumentParser(
        description="Parse Catch2 test_categories YAML and generate CMake test definitions"
    )
    parser.add_argument("yaml_file", help="Path to the test_categories YAML file")
    parser.add_argument(
        "target_name", help="Name of the test target (e.g., rocroller-tests-catch)"
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

    config = load_yaml(yaml_file)

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
        categories = config.get("test_categories", {})
        execution_settings = config.get("execution_settings", {})
        timeouts = execution_settings.get("category_timeouts", {})
        timeout_multiplier = execution_settings.get("timeout_multiplier", 1)
        env_dict = execution_settings.get("environment", {}) or {}
        env_string = (
            ";".join(f"{k}={v}" for k, v in env_dict.items()) if env_dict else None
        )

        is_windows = platform.system() == "Windows"
        is_linux = platform.system() == "Linux"

        # Validate the categories before generating CMake code.
        # If validation fails, no partial or intermediate CMake file will be written.
        validation_errors = validate_categories(categories, is_windows, is_linux)
        if validation_errors:
            print(
                f"Error: {len(validation_errors)} validation error(s) in {yaml_file}:",
                file=sys.stderr,
            )
            for msg in validation_errors:
                print(f"  - {msg}", file=sys.stderr)
            sys.exit(1)

        print("# Generated CMake code for Catch2 test categories (tag-based)")
        print(f"# Detected OS: {platform.system()}")
        print(f"# Timeout multiplier: {timeout_multiplier}")

        for category_name, category_info in categories.items():
            test_tags = category_info.get("test_tags", [])
            exclude_tags = category_info.get("exclude_tags", [])
            if exclude_tags is None:
                exclude_tags = []

            if is_windows:
                exclude_tags_win = category_info.get("exclude_tags_windows", [])
                if exclude_tags_win:
                    exclude_tags = exclude_tags + exclude_tags_win

            if is_linux:
                exclude_tags_linux = category_info.get("exclude_tags_linux", [])
                if exclude_tags_linux:
                    exclude_tags = exclude_tags + exclude_tags_linux

            labels = category_info.get("labels", [])

            base_timeout = timeouts.get(category_name, 300)
            timeout = int(base_timeout * timeout_multiplier)

            tag_expression = build_catch2_tag_expression(test_tags, exclude_tags)

            print(f"# Category: {category_name}")
            print(f'# Description: {category_info.get("description", "")}')
            print(
                f"# Tag expression: {tag_expression if tag_expression else '(all tests)'}"
            )

            if tag_expression:
                command_line = f'  COMMAND {target_name} "{tag_expression}"'
            else:
                command_line = f"  COMMAND {target_name}"

            label_string = '"' + ";".join(labels) + '"'

            print("add_test(")
            print(f"  NAME {target_name}_{category_name}_suite")
            print(command_line)
            print(f"  WORKING_DIRECTORY {working_dir}")
            print(")")

            print(
                f"set_tests_properties({target_name}_{category_name}_suite PROPERTIES"
            )
            print(f"  LABELS {label_string}")
            print(f"  TIMEOUT {timeout}")
            if env_string:
                print(f'  ENVIRONMENT "{env_string}"')
            print(")")
            print()

            if install_file_handle:
                try:
                    if tag_expression:
                        install_cmd = f'add_test({target_name}_{category_name}_suite "../{target_name}" "{tag_expression}")\n'
                    else:
                        install_cmd = f'add_test({target_name}_{category_name}_suite "../{target_name}")\n'
                    install_file_handle.write(install_cmd)
                    env_prop = f' ENVIRONMENT "{env_string}"' if env_string else ""
                    install_file_handle.write(
                        f"set_tests_properties({target_name}_{category_name}_suite PROPERTIES LABELS {label_string} TIMEOUT {timeout}{env_prop})\n\n"
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


if __name__ == "__main__":
    main()
