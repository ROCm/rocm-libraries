"""
Determines which component CI jobs to run based on changed files.

Outputs boolean flags per component via GITHUB_OUTPUT:
  - stinkytofu=true/false
  - rocisa=true/false

Each component defines a set of path patterns. If any changed file matches,
that component is marked as triggered.

Usage:
  python component_ci.py
"""

import fnmatch
import os
import subprocess

COMPONENTS = {
    "stinkytofu": [
        "shared/stinkytofu/**",
    ],
    "rocisa": [
        "projects/hipblaslt/tensilelite/rocisa/**",
        "shared/stinkytofu/**",
    ],
}

WORKFLOW_FILE = ".github/workflows/component-ci.yml"


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
    for key, patterns in COMPONENTS.items():
        all_patterns = patterns + [WORKFLOW_FILE]
        results[key] = matches_paths(changed_files, all_patterns)
    return results


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
    outputs = {k: str(v).lower() for k, v in changed.items()}
    set_github_output(outputs)


if __name__ == "__main__":
    main()
