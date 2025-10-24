"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/hipblas": "blas",
    "projects/hipblas-common": "blas",
    "projects/hipblaslt": "blas",
    "projects/hipcub": "prim",
    "projects/hipdnn": "hipdnn",
    "projects/hipfft": "fft",
    "projects/hiprand": "rand",
    "projects/hipsolver": "solver",
    "projects/hipsparse": "sparse",
    "projects/miopen": "miopen",
    "projects/rocblas": "blas",
    "project/rocfft": "fft",
    "projects/rocprim": "prim",
    "projects/rocrand": "rand",
    "projects/rocsolver": "solver",
    "projects/rocsparse": "sparse",
    "projects/rocthrust": "prim",
    "shared/mxdatagenerator": "blas",
    "shared/origami": "blas",
    "shared/rocroller": "blas",
    "shared/tensile": "blas"
}

project_map = {
    "prim": {
        "cmake_options": "-DTHEROCK_ENABLE_PRIM=ON",
        "project_to_test": "rocprim, rocthrust, hipcub",
    },
    "rand": {
        "cmake_options": "-DTHEROCK_ENABLE_RAND=ON",
        "project_to_test": "rocrand, hiprand",
    },
    "blas": {
        "cmake_options": "-DTHEROCK_ENABLE_BLAS=ON",
        "project_to_test": "hipblaslt, rocblas, hipblas",
    },
    "miopen": {
        "cmake_options": "-DTHEROCK_ENABLE_MIOPEN=ON -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON -DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON -DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../composable_kernel",
        "project_to_test": "miopen",
    },
    "fft": {
        "cmake_options": "-DTHEROCK_ENABLE_FFT=ON",
        "project_to_test": "hipfft, rocfft",
    },
    "hipdnn": {
        "cmake_options": "-DTHEROCK_ENABLE_HIPDNN=ON",
        "project_to_test": "hipdnn",
    }
}

# For certain math components, they are optional during building and testing.
# As they are optional, we do not want to include them as default as this takes more time in the CI.
# However, if we run a separate build for optional components, those files will be overriden as these components share the same umbrella as other projects
# Example: SPARSE is included in BLAS, but a separate build would cause overwriting of the blas_lib.tar.xz and blas_test.tar.xz and be missing libraries and tests
additional_options = {
    "sparse": {
        "cmake_options": "-DTHEROCK_ENABLE_SPARSE=ON",
        "project_to_test": "rocsparse, hipsparse",
        "project_to_add": "blas"
    },
    "solver": {
        "cmake_options": "-DTHEROCK_ENABLE_SOLVER=ON",
        "project_to_test": "rocsolver, hipsolver",
        "project_to_add": "blas"
    }
}

def collect_projects_to_run(subtrees):
    projects = set()
    # collect the associated subtree to project
    for subtree in subtrees:
        if subtree in subtree_to_project_map:
            projects.add(subtree_to_project_map.get(subtree))
            
    # Check if an optional math component was included.
    for project in list(projects):
        if project in additional_options:
            project_options_to_add = additional_options[project]
            
            project_to_add = project_options_to_add["project_to_add"]
            # If `project_to_add` is in included, add options to the existing `project_map` entry
            if project_to_add in projects:
                project_map[project_to_add]["cmake_options"] += f" {project_options_to_add["cmake_options"]}"
                project_map[project_to_add]["project_to_test"] += f", {project_options_to_add["project_to_test"]}"
            # If `project_to_add` is not included, only run build and tests for the optional project
            else:
                projects.add(project_to_add)
                project_map[project_to_add]["cmake_options"] = project_options_to_add["cmake_options"]
                project_map[project_to_add]["project_to_test"] = project_options_to_add["project_to_test"]

    # retrieve the subtrees to checkout, cmake options to build, and projects to test
    project_to_run = []
    for project in projects:
        if project in project_map:
            project_map_data = project_map.get(project)
            # To save time, only build what is needed
            project_map_data["cmake_options"] += " -DTHEROCK_ENABLE_ALL=OFF"
            project_to_run.append(project_map_data)

    return project_to_run
