#!/usr/bin/env python3
"""Get changed projects based on git diff."""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Map subtrees to project names (simplified from therock_matrix.py)
SUBTREE_TO_PROJECT = {
    "projects/composablekernel": "miopen",
    "projects/hipblas": "blas",
    "projects/hipblas-common": "blas",
    "projects/hipblaslt": "blas",
    "projects/hipcub": "prim",
    "projects/hipfft": "fft",
    "projects/hiprand": "rand",
    "projects/hipsolver": "solver",
    "projects/hipsparse": "sparse",
    "projects/hipsparselt": "sparselt",
    "projects/miopen": "miopen",
    "projects/rocblas": "blas",
    "projects/rocfft": "fft",
    "projects/rocprim": "prim",
    "projects/rocrand": "rand",
    "projects/rocsolver": "solver",
    "projects/rocsparse": "sparse",
    "projects/rocthrust": "prim",
    "projects/rocwmma": "rocwmma",
    "shared/tensile": "blas",
}


def get_changed_files(base_ref: str) -> list[str]:
    """Get list of changed files from git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError:
        return []


def get_changed_projects(base_ref: str) -> str:
    """Get comma-separated list of changed projects."""
    changed_files = get_changed_files(base_ref)
    if not changed_files:
        return ""

    projects = set()
    for file_path in changed_files:
        for subtree, project in SUBTREE_TO_PROJECT.items():
            if file_path.startswith(subtree + "/"):
                projects.add(project)
                break

    return ",".join(sorted(projects))


if __name__ == "__main__":
    base_ref = os.environ.get("BASE_REF", "HEAD^")
    result = get_changed_projects(base_ref)
    print(result)
