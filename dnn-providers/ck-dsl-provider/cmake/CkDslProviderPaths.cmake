# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Resolve the absolute paths of:
#   1. The in-tree ck_dsl Python package
#      (projects/composablekernel/python/ck_dsl/).
#   2. The provider-local ck_dsl_provider Python package
#      (dnn-providers/ck-dsl-provider/python/).
#
# These paths are baked into a generated header
# (ckdsl_provider_paths.h) so the embedded interpreter can prepend them
# to sys.path at startup. Required because:
#   - ck_dsl has no pyproject.toml / setup.py, so we cannot pip-install
#     it into the embedded interpreter's site-packages (PREP_FINDINGS §P-4).
#   - The provider's own Python package is shipped beside the .so, not
#     into site-packages.
#
# Outputs (set in parent scope):
#   CK_DSL_PYTHON_PACKAGE_PATH           absolute dir containing ck_dsl/
#   CK_DSL_PROVIDER_PYTHON_PACKAGE_PATH  absolute dir containing
#                                        ck_dsl_provider/

# Snapshot the directory this .cmake file lives in *at include time*.
# Inside a CMake function body, CMAKE_CURRENT_LIST_DIR reflects the
# caller's list file, not the file that defined the function, so we
# cannot rely on it inside ck_dsl_provider_resolve_python_paths().
set(_ckDslProviderPathsCmakeDir "${CMAKE_CURRENT_LIST_DIR}")

# Resolve CK_DSL_PYTHON_PACKAGE_PATH and CK_DSL_PROVIDER_PYTHON_PACKAGE_PATH
# in the caller's scope. Walks up from the provider source directory until it
# locates projects/composablekernel/python/ck_dsl/__init__.py; aborts with
# FATAL_ERROR if the ck_dsl package cannot be found.
function(ck_dsl_provider_resolve_python_paths)
    # Walk up from the provider source directory until we find
    # projects/composablekernel/python/ck_dsl/__init__.py. The walk
    # stops at the filesystem root.
    set(_searchDir "${_ckDslProviderPathsCmakeDir}/..")
    get_filename_component(_searchDir "${_searchDir}" ABSOLUTE)

    set(_ckDslRelPath "projects/composablekernel/python/ck_dsl/__init__.py")
    set(_resolvedCkDslDir "")

    while(NOT _resolvedCkDslDir AND NOT _searchDir STREQUAL "/")
        if(EXISTS "${_searchDir}/${_ckDslRelPath}")
            set(_resolvedCkDslDir "${_searchDir}/projects/composablekernel/python")
            break()
        endif()
        get_filename_component(_parent "${_searchDir}" DIRECTORY)
        if(_parent STREQUAL _searchDir)
            # get_filename_component returns the same dir for "/", "/."
            break()
        endif()
        set(_searchDir "${_parent}")
    endwhile()

    if(NOT _resolvedCkDslDir OR NOT EXISTS "${_resolvedCkDslDir}/ck_dsl/__init__.py")
        message(FATAL_ERROR
            "CK DSL provider: failed to locate the in-tree ck_dsl Python "
            "package. Walked up from "
            "${CMAKE_CURRENT_LIST_DIR}/.. looking for "
            "${_ckDslRelPath}. Either the source tree layout has changed "
            "or the provider was extracted outside the rocm-libraries "
            "workspace. Set CK_DSL_PYTHON_PACKAGE_PATH explicitly to "
            "override the search.")
    endif()

    # The provider-local Python package lives beside the source dir of
    # this .cmake file, in ../python/.
    set(_providerPyDir "${_ckDslProviderPathsCmakeDir}/../python")
    get_filename_component(_providerPyDir "${_providerPyDir}" ABSOLUTE)

    if(NOT EXISTS "${_providerPyDir}/ck_dsl_provider/__init__.py")
        message(FATAL_ERROR
            "CK DSL provider: provider-local Python package missing at "
            "${_providerPyDir}/ck_dsl_provider/__init__.py.")
    endif()

    set(CK_DSL_PYTHON_PACKAGE_PATH "${_resolvedCkDslDir}" PARENT_SCOPE)
    set(CK_DSL_PROVIDER_PYTHON_PACKAGE_PATH "${_providerPyDir}" PARENT_SCOPE)

    message(STATUS
        "CK DSL provider Python paths: "
        "ck_dsl=${_resolvedCkDslDir}, "
        "ck_dsl_provider=${_providerPyDir}")
endfunction()
