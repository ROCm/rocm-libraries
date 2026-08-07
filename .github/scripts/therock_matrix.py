"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""

import copy
import os

subtree_to_project_map = {
    "dnn-providers/hipblaslt-provider": "hipblaslt-provider",
    "dnn-providers/hip-kernel-provider": "hip-kernel-provider",
    "dnn-providers/miopen-provider": "miopen-provider",
    "dnn-providers/integration-tests": "dnn-provider-integration-tests",
    "projects/composablekernel": "miopen",
    "projects/hipblas": "blas",
    "projects/hipblas-common": "blas",
    "projects/hipblaslt": "blas",
    "projects/hipcub": "prim",
    "projects/hipdnn": "hipdnn",
    "projects/hipfft": "fft",
    "projects/hiprand": "rand",
    "projects/hiptensor": "hiptensor",
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
    "projects/rocalution": "rocalution",
    "projects/rocwmma": "rocwmma",
    "projects/hipthreads": "hipthreads",
    "shared/mxdatagenerator": "blas",
    "shared/origami": "blas",
    "shared/rocroller": "rocroller",
    "shared/stinkytofu": "blas",
    "shared/tensile": "blas",
}

project_map = {
    "prim": {
        "cmake_options": ["-DTHEROCK_ENABLE_PRIM=ON"],
        "projects_to_test": ["rocprim", "rocthrust", "hipcub"],
    },
    "rand": {
        "cmake_options": ["-DTHEROCK_ENABLE_RAND=ON"],
        "projects_to_test": ["rocrand", "hiprand"],
    },
    "blas": {
        "cmake_options": ["-DTHEROCK_ENABLE_BLAS=ON"],
        "projects_to_test": ["hipblaslt", "rocblas", "hipblas", "tensilelite"],
    },
    "miopen": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_MIOPEN=ON",
            "-DTHEROCK_ENABLE_MIOPENPROVIDER=ON",
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
            "-DTHEROCK_COMPOSABLE_KERNEL_FOR_MIOPEN_ONLY=ON",
        ],
        "projects_to_test": ["miopen", "miopenprovider"],
    },
    "fft": {
        "cmake_options": ["-DTHEROCK_ENABLE_FFT=ON", "-DTHEROCK_ENABLE_RAND=ON"],
        "projects_to_test": ["hipfft", "rocfft"],
    },
    "hiptensor": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_HIPTENSOR=ON",
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
            "-DTHEROCK_ENABLE_RAND=ON",
        ],
        "additional_flags": {
            "linux": ["-DTHEROCK_ENABLE_ROCPROFV3=ON"],
        },
        "projects_to_test": ["hiptensor"],
    },
    "hip-kernel-provider": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_HIPKERNELPROVIDER=ON",
            "-DHIP_KERNEL_PROVIDER_ENABLE=ON",
            "-DTHEROCK_FLAG_HIPKERNELPROVIDER_ENABLE_ROCKE=ON",
        ],
        "projects_to_test": ["hipkernelprovider"],
    },
    "hipthreads": {
        "cmake_options": ["-DTHEROCK_ENABLE_HIPTHREADS=ON"],
        "projects_to_test": ["hipthreads"],
    },
}

# For certain math components, they are optional during building and testing.
# As they are optional, we do not want to include them as default as this takes more time in the CI.
# However, if we run a separate build for optional components, those files will be overriden as these components share the same umbrella as other projects
# Example: SPARSE is included in BLAS, but a separate build would cause overwriting of the blas_lib.tar.xz and blas_test.tar.xz and be missing libraries and tests
additional_options = {
    "sparse": {
        "cmake_options": ["-DTHEROCK_ENABLE_SPARSE=ON"],
        "projects_to_test": ["rocsparse", "hipsparse"],
        "project_to_add": "blas",
    },
    "sparselt": {
        "cmake_options": ["-DTHEROCK_ENABLE_SPARSE=ON"],
        "projects_to_test": ["hipsparselt"],
        "project_to_add": "blas",
    },
    "solver": {
        "cmake_options": ["-DTHEROCK_ENABLE_SOLVER=ON"],
        "projects_to_test": ["rocsolver", "hipsolver"],
        "project_to_add": "blas",
    },
    "hipdnn": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_HIPBLASLTPROVIDER=ON",
            "-DTHEROCK_ENABLE_HIPKERNELPROVIDER=ON",
            "-DHIP_KERNEL_PROVIDER_ENABLE=ON",
            "-DTHEROCK_ENABLE_MIOPENPROVIDER=ON",
            "-DTHEROCK_ENABLE_HIPDNN_SAMPLES=ON",
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
            "-DTHEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS=ON",
            "-DTHEROCK_COMPOSABLE_KERNEL_FOR_MIOPEN_ONLY=ON",
        ],
        "projects_to_test": [
            "hipdnn",
            "hipdnn_install",
            "hipdnn-samples",
            "miopenprovider",
            "hipblasltprovider",
            "hipkernelprovider",
            "hipdnn-integration-tests",
        ],
        "project_to_add": "miopen",
    },
    "miopen-provider": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_MIOPENPROVIDER=ON",
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
            "-DTHEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS=ON",
        ],
        "projects_to_test": ["miopenprovider"],
        "project_to_add": "miopen",
    },
    "dnn-provider-integration-tests": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS=ON",
            "-DTHEROCK_ENABLE_MIOPENPROVIDER=ON",
            "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON",
        ],
        "projects_to_test": ["hipdnn-integration-tests", "miopenprovider"],
        "project_to_add": "miopen",
    },
    "hipblaslt-provider": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_HIPBLASLTPROVIDER=ON",
        ],
        "projects_to_test": ["hipblasltprovider"],
        "project_to_add": "blas",
    },
    "rocwmma": {
        "cmake_options": ["-DTHEROCK_ENABLE_ROCWMMA=ON"],
        "projects_to_test": ["rocwmma"],
        "project_to_add": "blas",
    },
    "rocalution": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_ROCALUTION=ON",
            "-DTHEROCK_ENABLE_SPARSE=ON",
            "-DTHEROCK_ENABLE_RAND=ON",
        ],
        "projects_to_test": ["rocalution"],
        "project_to_add": "blas",
    },
    # rocRoller is built under the BLAS umbrella but only tested when its own
    # subtree changes. Merges into the "blas" job when a PR touches both, which
    # avoids a redundant BLAS build and S3 artifact overlap.
    "rocroller": {
        "cmake_options": ["-DTHEROCK_ENABLE_BLAS=ON"],
        "projects_to_test": ["rocroller"],
        "project_to_add": "blas",
    },
}

# If a project has dependencies that are also being built, we combine build options and test options
# This way, there will be no S3 upload overlap and we save redundant builds
dependency_graph = {
    "miopen": ["blas", "rand"],
}

# When these subtrees change, also activate the given optional matrix project so
# its additional_options merge into the parent job (e.g. hipSPARSELt depends on hipBLASLt).
SUBTREE_EXTRA_MATRIX_PROJECTS = {
    "projects/hipblaslt": "sparselt",
}

ROCJITSU_RACE_CHECK_SUBTREES = {
    "projects/hipblaslt",
}

# PR labels that inject an extra cmake option into a specific project's build.
# The option is only added when the gating label is present on the PR AND that
# project is actually being built, so the default build is unchanged. This lets a
# branch opt a single superbuild into a feature flag without adding a second,
# colliding job: the existing job builds with the flag on and uploads its
# artifact once, in place of the flag-off one, so there is no artifact overlap or
# job-name clash.
#
# To add an entry, map a (manually applied) GitHub label to a project and the
# cmake options to inject. The label must exist in the repo's label set; it is not
# auto-applied via labeler.yml. Example:
#
#   LABEL_GATED_CMAKE_OPTIONS = {
#       "ci:my-feature": {
#           "project": "myproject",
#           "cmake_options": ["-DTHEROCK_FLAG_MY_FEATURE=ON"],
#       },
#   }
#
# `project` must name an entry in `project_map` or `additional_options`; anything
# else is rejected up front by `validate_label_gated_cmake_options` rather than
# failing obscurely later.
#
# A target does not have to survive as its own job. Projects get merged into one
# another twice -- an optional component into its `project_to_add` parent, and a
# dependency into the project that absorbs it -- and an injection follows its
# target through both, landing on whichever job ends up doing that project's
# build. If the target is not built at all, nothing is injected.
#
# Injected options are appended last, after every default and merged-in option,
# so injecting a value that contradicts a default wins (cmake takes the last -D
# for a given name).
LABEL_GATED_CMAKE_OPTIONS = {}


def validate_label_gated_cmake_options(gated_options=None):
    """Check LABEL_GATED_CMAKE_OPTIONS entries before they are used.

    This map is hand-edited and its only feedback channel is a CI run, so a typo
    would otherwise surface as a confusing mid-run failure or, worse, as a green
    build that silently did not get the flag. Fail loudly and early instead.
    """
    if gated_options is None:
        gated_options = LABEL_GATED_CMAKE_OPTIONS
    valid_projects = set(project_map) | set(additional_options)
    for label, gated in gated_options.items():
        for key in ("project", "cmake_options"):
            if key not in gated:
                raise ValueError(
                    f"LABEL_GATED_CMAKE_OPTIONS['{label}'] is missing required key '{key}'"
                )
        if gated["project"] not in valid_projects:
            raise ValueError(
                f"LABEL_GATED_CMAKE_OPTIONS['{label}'] targets unknown project "
                f"'{gated['project']}'. Valid projects: {sorted(valid_projects)}"
            )
        if not isinstance(gated["cmake_options"], list):
            # Anything other than a list either blows up mid-run or, for a string
            # or dict, is silently consumed as a sequence of characters or keys and
            # produces a different build than the one that was asked for.
            raise ValueError(
                f"LABEL_GATED_CMAKE_OPTIONS['{label}']['cmake_options'] must be a "
                f"list of options, not {type(gated['cmake_options']).__name__}"
            )

def collect_projects_to_run(subtrees, pr_labels=None):
    subtrees = list(subtrees)
    platform = os.getenv("PLATFORM")
    projects = set()
    # Record why the BLAS row was selected before dependency folding loses the
    # original subtree identity. Workflows consume this marker after the matrix
    # is assembled to attach instrumentation to the final merged product row.
    run_rocjitsu_race_check = bool(ROCJITSU_RACE_CHECK_SUBTREES.intersection(subtrees))
    # Work on per-call deep copies so module-level state stays immutable across calls.
    local_project_map = copy.deepcopy(project_map)
    local_additional_options = copy.deepcopy(additional_options)

    # collect the associated subtree to project
    for subtree in subtrees:
        if subtree in subtree_to_project_map:
            projects.add(subtree_to_project_map.get(subtree))

        extra_matrix = SUBTREE_EXTRA_MATRIX_PROJECTS.get(subtree)
        if extra_matrix:
            projects.add(extra_matrix)

    # Collect the label-gated cmake options to inject, keyed by target project.
    # They are not written into the option lists here: the merge passes below
    # rewrite those lists, so an option injected now can be reordered behind a
    # merged-in default or dropped outright when a parent's options are replaced.
    # Instead the target is followed through each merge and the options are
    # appended at emit time, which keeps them last and keeps them attached to the
    # job that actually builds the target.
    validate_label_gated_cmake_options()
    pending_injections = {}
    for label in pr_labels or []:
        gated = LABEL_GATED_CMAKE_OPTIONS.get(label)
        if not gated:
            continue
        pending_injections.setdefault(gated["project"], []).extend(
            gated["cmake_options"]
        )

    def redirect_injection(absorbed, absorbed_into):
        """Follow an injection when `absorbed`'s build moves onto another job."""
        options = pending_injections.pop(absorbed, None)
        if options:
            pending_injections.setdefault(absorbed_into, []).extend(options)

    for project in list(projects):
        # Check if an optional math component was included.
        if project in local_additional_options:
            project_options_to_add = local_additional_options[project]

            project_to_add = project_options_to_add["project_to_add"]
            # If `project_to_add` is in included, add options to the existing `local_project_map` entry
            if project_to_add in projects:
                local_project_map[project_to_add]["cmake_options"].extend(
                    project_options_to_add["cmake_options"]
                )
                local_project_map[project_to_add]["projects_to_test"].extend(
                    project_options_to_add["projects_to_test"]
                )
            # If `project_to_add` is not included, only run build and tests for the optional project
            else:
                projects.add(project_to_add)
                local_project_map[project_to_add]["cmake_options"] = (
                    project_options_to_add["cmake_options"]
                )
                local_project_map[project_to_add]["projects_to_test"] = (
                    project_options_to_add["projects_to_test"]
                )

            # Either way the component is now built by `project_to_add`'s job, and
            # the component itself never produces one, so an injection aimed at it
            # has to move with it.
            redirect_injection(project, project_to_add)

    # Check for potential dependencies
    to_remove_from_project_map = []
    for project in list(projects):
        # Check if project has a dependency combine
        if project in dependency_graph:
            for dependency in dependency_graph[project]:
                # If the dependency is also included, let's combine to avoid overlap
                if dependency in projects:
                    local_project_map[project]["cmake_options"].extend(
                        local_project_map[dependency]["cmake_options"]
                    )
                    local_project_map[project]["projects_to_test"].extend(
                        local_project_map[dependency]["projects_to_test"]
                    )
                    to_remove_from_project_map.append(dependency)
                    redirect_injection(dependency, project)

    # if dependency is included in projects and parent is found, we delete the dependency as the parent will build and test
    for to_remove_item in to_remove_from_project_map:
        projects.remove(to_remove_item)
        del local_project_map[to_remove_item]

    # retrieve the subtrees to checkout, cmake options to build, and projects to test
    project_to_run = []
    for project in projects:
        if project in local_project_map:
            project_map_data = local_project_map.get(project)

            # Check if platform-based additional flags are needed
            if (
                "additional_flags" in project_map_data
                and platform in project_map_data["additional_flags"]
            ):
                project_map_data["cmake_options"].extend(
                    project_map_data["additional_flags"][platform]
                )

            # To save time, only build what is needed
            project_map_data["cmake_options"].extend(["-DTHEROCK_ENABLE_ALL=OFF"])

            # Label-gated options go on last so they override anything above them,
            # including a default this job absorbed from another project. Targets
            # that are not built never reach here, which is what keeps an unrelated
            # PR's build identical to the default one.
            project_map_data["cmake_options"].extend(
                pending_injections.get(project, [])
            )

            # To ensure uniqueness of flags and tests. dict.fromkeys dedupes while
            # preserving insertion order; set() does not, and its iteration order
            # varies with PYTHONHASHSEED between runs. Order matters because cmake
            # takes the last -D for a given name, so a set() here would make an
            # option that overrides an earlier default resolve nondeterministically.
            project_map_data["cmake_options"] = list(
                dict.fromkeys(project_map_data["cmake_options"])
            )
            project_map_data["projects_to_test"] = list(
                dict.fromkeys(project_map_data["projects_to_test"])
            )
            project_map_data["run_rocjitsu_race_check"] = (
                run_rocjitsu_race_check
                and "tensilelite" in project_map_data["projects_to_test"]
            )

            cmake_flag_options = " ".join(project_map_data["cmake_options"])
            projects_to_test_options = ",".join(project_map_data["projects_to_test"])
            project_map_data["cmake_options"] = cmake_flag_options
            project_map_data["projects_to_test"] = projects_to_test_options
            project_to_run.append(project_map_data)

    return project_to_run
