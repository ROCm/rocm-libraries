"""
Determines which component CI jobs to run based on changed files.

Outputs boolean flags per component via GITHUB_OUTPUT:
  - stinkytofu=true/false
  - rocisa=true/false
  - geko=true/false
  - miopen=true/false
  - tensilelite_coverage=true/false
  - hipblaslt_logic_corpus=true/false
  - hipblaslt_rocjitsu_scripts=true/false
  - hipblaslt_jit=true/false

Each component defines a set of path patterns. If any changed file matches,
that component is marked as triggered. A component may also list exclusions:
changed files that match one (and none of its exceptions) are ignored for that
component only.

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
    "geko": [
        "projects/hipblaslt/utilities/geko/**",
        ".github/workflows/component-ci-geko.yml",
    ],
    "miopen": [
        "projects/miopen/**",
        # The hipDNN MIOpen provider carries a copy of MIOpen's rename list,
        # generated from miopen.h, and hand-written declarations of a few
        # MIOpen entry points. MIOpen CI checks both against MIOpen, so a PR
        # touching them has to trigger this component. List only the provider
        # files those checks read (generate_wrapper_sources.py and
        # check_public_abi.py in projects/miopen/script); add any new one.
        "dnn-providers/miopen-provider/MiopenApi.hpp",
        "dnn-providers/miopen-provider/MiopenApiPrivateRename.hpp",
        "dnn-providers/miopen-provider/.clang-format",
        ".github/workflows/component-ci-miopen.yml",
    ],
    "tensilelite_coverage": [
        "projects/hipblaslt/tensilelite/**",
        ".github/workflows/component-ci-tensilelite-coverage.yml",
    ],
    # The logic corpus is tuning data outside tensilelite/, so a change to it
    # alone triggers no TensileLite lane. This one validates the whole corpus
    # and runs the unit tests that read it; the validator, the tests' corpus
    # plumbing, and the SolutionUID codec that the uniqueness test reads
    # trigger it too.
    "hipblaslt_logic_corpus": [
        "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/**",
        "projects/hipblaslt/scripts/run_tensile_logic_check.py",
        "projects/hipblaslt/tensilelite/tensilelite/logic/**",
        "projects/hipblaslt/tensilelite/tensilelite/CustomYamlLoader.py",
        "projects/hipblaslt/tensilelite/tensilelite/Tests/unit/logic_corpus.py",
        "projects/hipblaslt/tensilelite/tensilelite/Tests/unit/"
        "test_solution_uid_uniqueness.py",
        "projects/hipblaslt/tensilelite/tensilelite/SolutionIdGen.py",
        "projects/hipblaslt/tensilelite/tox.ini",
        ".github/workflows/component-ci-logic-corpus.yml",
    ],
    "hipblaslt_rocjitsu_scripts": [
        "projects/hipblaslt/clients/scripts/rocjitsu/**",
        ".github/scripts/run_rocjitsu_hipblaslt_gemm_race_checks.sh",
        ".github/workflows/component-ci-hipblaslt-rocjitsu-scripts.yml",
    ],
    # GPU build and tests of the JIT library. The workflow itself selects the
    # gfx942 and gfx950 runners.
    "hipblaslt_jit": [
        "projects/hipblaslt/library/src/amd_detail/hipblaslt-jit-*",
        "projects/hipblaslt/library/src/amd_detail/hipkittens/**",
        "projects/hipblaslt/cmake/hipblaslt-jit-hipkittens.cmake",
        "projects/hipblaslt/CMakeLists.txt",
        "projects/hipblaslt/library/src/amd_detail/CMakeLists.txt",
        "projects/hipblaslt/library/src/amd_detail/hipblaslt.cpp",
        "projects/hipblaslt/library/src/amd_detail/hipblaslt-ext.cpp",
        "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/tensile_host.cpp",
        "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/include/tensile_host.hpp",
        "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/rocblaslt_auxiliary.cpp",
        "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/rocblaslt_mat.cpp",
        "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/handle.cpp",
        "projects/hipblaslt/clients/tests/jit/**",
        "projects/hipblaslt/scripts/jit/**",
        "projects/hipblaslt/tensilelite/include/Tensile/AMDGPUPredicates.hpp",
        "projects/hipblaslt/tensilelite/include/Tensile/CachingLibrary.hpp",
        "projects/hipblaslt/tensilelite/include/Tensile/ExactLogicLibrary.hpp",
        "projects/hipblaslt/tensilelite/include/Tensile/LocalKernelIndex.hpp",
        "projects/hipblaslt/tensilelite/include/Tensile/ProviderRows.hpp",
        "projects/hipblaslt/tensilelite/src/ProviderRows.cpp",
        "projects/hipblaslt/tensilelite/include/Tensile/Tensile.hpp",
        "projects/hipblaslt/tensilelite/include/Tensile/ContractionProblem.hpp",
        "projects/hipblaslt/tensilelite/include/Tensile/ContractionSolution.hpp",
        "projects/hipblaslt/tensilelite/include/Tensile/Serialization/**",
        "projects/hipblaslt/tensilelite/include/Tensile/msgpack/**",
        "projects/hipblaslt/tensilelite/src/ContractionProblem.cpp",
        "projects/hipblaslt/tensilelite/src/ContractionSolution.cpp",
        "projects/hipblaslt/tensilelite/src/msgpack/**",
        "projects/hipblaslt/tensilelite/tensilelite/Common/GlobalParameters.py",
        ".github/workflows/component-ci-hipblaslt-jit.yml",
    ],
}

# Prose that no step of a component's CI reads. A change confined to these
# files does not trigger the component; any other matching change still does.
# fnmatch's * crosses "/", so "projects/miopen/*.md" covers nested files.
EXCLUSIONS = {
    "miopen": [
        "projects/miopen/*.md",
        "projects/miopen/*.rst",
    ],
}

# Files matching an exclusion that still trigger the component.
EXCLUSION_EXCEPTIONS = {
    "miopen": [
        # rocm_create_package installs the top-level license into the package
        # that the MIOpen workflow builds.
        "projects/miopen/LICENSE.md",
    ],
}

# Paths that affect every component's CI run: the workflow, this script,
# and the shared ci-env action. A change here re-triggers all components.
INFRA_PATHS = [
    ".github/workflows/component-ci.yml",
    ".github/scripts/component_ci.py",
    ".github/actions/ci-env/**",
]


# detect_changed_components is deliberately separate from TheRock's own change
# detection: TheRock uses prefix-based subtree matching (via repos-config.json)
# while component CI uses fnmatch glob patterns with cross-component triggers
# (e.g. rocisa triggers when stinkytofu changes because it depends on it).
def detect_changed_components(changed_files: set[str]) -> dict[str, bool]:
    infra_changed = matches_paths(changed_files, INFRA_PATHS)
    results = {}
    for key, patterns in COMPONENTS.items():
        excluded = EXCLUSIONS.get(key, [])
        exceptions = EXCLUSION_EXCEPTIONS.get(key, [])
        relevant = [
            f
            for f in changed_files
            if not matches_paths([f], excluded) or matches_paths([f], exceptions)
        ]
        results[key] = infra_changed or matches_paths(relevant, patterns)
    return results


def main():
    base_ref = os.environ.get("BASE_REF", "HEAD^")
    is_workflow_dispatch = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

    if is_workflow_dispatch:
        changed = {key: True for key in COMPONENTS}
    else:
        # Both sides of a rename: moving a source file to an excluded or
        # unrelated path must still trigger the component it left.
        changed_files = get_modified_paths(base_ref, detect_renames=False)
        changed = detect_changed_components(changed_files)

    print(f"Changed components: {changed}")
    outputs = {k: str(v).lower() for k, v in changed.items()}
    set_github_output(outputs)


if __name__ == "__main__":
    main()
