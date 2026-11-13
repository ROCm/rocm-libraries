"""
hipCCL-scoped trim of rocm-libraries' .github/scripts/therock_matrix.py.

This dictionary is used to map specific file directory changes to the
corresponding build flag and tests. The upstream rocm-libraries version
covers every project in that monorepo (blas, fft, miopen, prim, rand, ...);
this copy only knows about hipCCL's own components, and paths are relative
to this repo's root (no "projects/" prefix - see ../../README.md for the
hipccl3 layout).

NOTE(hipccl3): this is a first-pass placeholder, not validated against a
real TheRock checkout. In particular:
  - "-DTHEROCK_ENABLE_PRIM=ON" is TheRock's existing flag for
    rocprim+hipcub+rocthrust and should still be correct once hipCCL is
    consumed from its own repo instead of rocm-libraries' projects/rocprim
    etc. - but this has not been confirmed with the TheRock maintainers.
  - libhipcxx has no known TheRock-side cmake flag yet (it isn't built by
    TheRock at all today), so it has no entry here. Adding one is blocked on
    TheRock integration work, not anything in this file.
"""

import copy
import os

subtree_to_project_map = {
    "rocprim": "prim",
    "hipcub": "prim",
    "rocthrust": "prim",
    # "libhipcxx": "libhipcxx",  # NOTE(hipccl3): no TheRock build flag yet, see module docstring.
}

project_map = {
    "prim": {
        "cmake_options": ["-DTHEROCK_ENABLE_PRIM=ON"],
        "projects_to_test": ["rocprim", "rocthrust", "hipcub"],
    },
}

additional_options = {}
dependency_graph = {}
SUBTREE_EXTRA_MATRIX_PROJECTS = {}


def collect_projects_to_run(subtrees):
    """Trimmed version of rocm-libraries' function of the same name, minus the
    additional_options/dependency_graph/SUBTREE_EXTRA_MATRIX_PROJECTS handling
    (hipCCL has no optional/dependent components yet, unlike rocm-libraries'
    sparse/solver/hipdnn-on-miopen style additions)."""
    subtrees = list(subtrees)
    platform = os.getenv("PLATFORM")
    projects = set()
    local_project_map = copy.deepcopy(project_map)

    for subtree in subtrees:
        if subtree in subtree_to_project_map:
            projects.add(subtree_to_project_map.get(subtree))

    project_to_run = []
    for project in projects:
        if project in local_project_map:
            project_map_data = local_project_map.get(project)

            supported_platforms = project_map_data.pop("platforms", None)
            if supported_platforms is not None and platform not in supported_platforms:
                continue

            project_map_data["cmake_options"].extend(["-DTHEROCK_ENABLE_ALL=OFF"])
            project_map_data["cmake_options"] = list(set(project_map_data["cmake_options"]))
            project_map_data["projects_to_test"] = list(set(project_map_data["projects_to_test"]))
            cmake_flag_options = " ".join(project_map_data["cmake_options"])
            projects_to_test_options = ",".join(project_map_data["projects_to_test"])
            project_map_data["cmake_options"] = cmake_flag_options
            project_map_data["projects_to_test"] = projects_to_test_options
            project_to_run.append(project_map_data)

    return project_to_run
