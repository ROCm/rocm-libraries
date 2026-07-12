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

import os

from ci_utils import get_modified_paths, matches_paths, set_github_output

COMPONENTS = {
    "stinkytofu": [
        "shared/stinkytofu/**",
    ],
    "rocisa": [
        "projects/hipblaslt/tensilelite/rocisa/**",
        "shared/stinkytofu/**",
    ],
    "miopen": [
        "projects/miopen/**",
        ".github/workflows/component-ci-miopen.yml",
    ],
}

# Paths that affect every component's CI run: the workflow, this script,
# and the shared ci-env action. A change here re-triggers all components.
INFRA_PATHS = [
    ".github/workflows/component-ci.yml",
    ".github/scripts/component_ci.py",
    ".github/actions/ci-env/**",
]


# detect_changed_components is not shared with therock_configure_ci.py because
# TheRock uses prefix-based subtree matching (via repos-config.json) while
# component CI uses fnmatch glob patterns with cross-component triggers
# (e.g. rocisa triggers when stinkytofu changes because it depends on it).
def detect_changed_components(changed_files: set[str]) -> dict[str, bool]:
    results = {}
    for key, patterns in COMPONENTS.items():
        all_patterns = patterns + INFRA_PATHS
        results[key] = matches_paths(changed_files, all_patterns)
    return results


def main():
    base_ref = os.environ.get("BASE_REF", "HEAD^")
    is_workflow_dispatch = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

    if is_workflow_dispatch:
        changed = {key: True for key in COMPONENTS}
    else:
        changed_files = get_modified_paths(base_ref)
        changed = detect_changed_components(changed_files)

    print(f"Changed components: {changed}")
    outputs = {k: str(v).lower() for k, v in changed.items()}
    set_github_output(outputs)


if __name__ == "__main__":
    main()
