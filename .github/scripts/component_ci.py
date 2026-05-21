"""
Determines which component CI jobs to run based on changed files.

Outputs per-platform JSON matrices via GITHUB_OUTPUT:
  - linux_components: components to run on Linux
  - windows_components: components to run on Windows

Each component entry has:
  - name: display name
  - dir: working directory relative to repo root
  - build: whether to run invoke build
  - test: whether to run ctest
  - build_gcc: whether to run GCC build (Linux only)
  - build_static: whether to run static build (Windows only)
  - requirements: path to requirements.txt relative to dir (or empty)
  - pip_test_path: pytest directory relative to dir (or empty)

Component config format:
  Each component has common fields (name, dir, paths) and per-platform
  config under "linux" and "windows" keys. A platform value can be:
    - a dict of step flags for that platform
    - a string referencing another platform (e.g. "linux") to reuse its config
    - omitted to skip that platform entirely

Usage:
  python component_ci.py
"""

import fnmatch
import json
import os
import subprocess

COMPONENTS = {
    "stinkytofu": {
        "name": "StinkyTofu",
        "dir": "shared/stinkytofu",
        "paths": [
            "shared/stinkytofu/**",
        ],
        "linux": {
            "build": True,
            "test": True,
            "build_gcc": True,
            "requirements": "requirements.txt",
            "pip_test_path": "python_module/tests",
        },
        "windows": {
            "build": True,
            "test": True,
            "build_static": True,
            "requirements": "requirements.txt",
            "pip_test_path": "python_module/tests",
        },
    },
    "rocisa": {
        "name": "rocISA",
        "dir": "projects/hipblaslt/tensilelite/rocisa",
        "paths": [
            "projects/hipblaslt/tensilelite/rocisa/**",
            "shared/stinkytofu/**",
        ],
        "linux": {
            "pip_test_path": "test",
        },
        "windows": "linux",
    },
}

WORKFLOW_FILE = ".github/workflows/component-ci.yml"

STEP_FIELDS = (
    "build",
    "test",
    "build_gcc",
    "build_static",
    "requirements",
    "pip_test_path",
)
STEP_DEFAULTS = {
    "build": False,
    "test": False,
    "build_gcc": False,
    "build_static": False,
    "requirements": "",
    "pip_test_path": "",
}


def get_changed_files(base_ref: str) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base_ref],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return set(result.stdout.splitlines())


def matches_paths(changed_files: set[str], patterns: list[str]) -> bool:
    for f in changed_files:
        for pattern in patterns:
            if fnmatch.fnmatch(f, pattern):
                return True
    return False


def detect_changed_components(changed_files: set[str]) -> dict[str, bool]:
    results = {}
    for key, config in COMPONENTS.items():
        component_patterns = config["paths"] + [WORKFLOW_FILE]
        results[key] = matches_paths(changed_files, component_patterns)
    return results


def resolve_platform_config(config: dict, platform: str) -> dict | None:
    """Resolve platform config, following string references."""
    platform_config = config.get(platform)
    if platform_config is None:
        return None
    if isinstance(platform_config, str):
        platform_config = config[platform_config]
    return platform_config


def make_entry(config: dict, platform: str) -> dict | None:
    platform_config = resolve_platform_config(config, platform)
    if platform_config is None:
        return None

    entry = {
        "name": config["name"],
        "dir": config["dir"],
    }
    for field in STEP_FIELDS:
        entry[field] = platform_config.get(field, STEP_DEFAULTS[field])
    return entry


def build_outputs(changed: dict[str, bool]) -> dict[str, str]:
    linux_components = []
    windows_components = []

    for key, is_changed in changed.items():
        if not is_changed:
            continue
        config = COMPONENTS[key]

        linux_entry = make_entry(config, "linux")
        if linux_entry is not None:
            linux_components.append(linux_entry)

        windows_entry = make_entry(config, "windows")
        if windows_entry is not None:
            windows_components.append(windows_entry)

    return {
        "linux_components": json.dumps(linux_components),
        "windows_components": json.dumps(windows_components),
    }


def set_github_output(outputs: dict[str, str]):
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        for k, v in outputs.items():
            print(f"{k}={v}")
        return
    with open(output_file, "a") as f:
        for k, v in outputs.items():
            f.write(f"{k}={v}\n")


def main():
    base_ref = os.environ.get("BASE_REF", "HEAD^")
    is_workflow_dispatch = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

    if is_workflow_dispatch:
        changed = {key: True for key in COMPONENTS}
    else:
        changed_files = get_changed_files(base_ref)
        changed = detect_changed_components(changed_files)

    print(f"Changed components: {changed}")
    outputs = build_outputs(changed)
    print(f"Outputs: {outputs}")
    set_github_output(outputs)


if __name__ == "__main__":
    main()
