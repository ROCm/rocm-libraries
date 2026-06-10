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

# Resolve the include directories needed to consume the header-only
# CK-Tile dispatcher key/problem types (fmha_kernel_key.hpp /
# fmha_problem.hpp). Two directories are required because fmha_types.hpp
# (pulled in transitively) includes ck_tile/core and ck_tile/ops headers:
#   1. projects/composablekernel/dispatcher/include
#   2. projects/composablekernel/include
#
# Both are HIP-free (the key/problem headers do NOT transitively include
# kernel_launch.hpp), so the consuming TUs need no HIP compile mode.
#
# Walks up from the provider source directory the same way the Python
# package resolver does. Outputs (set in caller's scope):
#   CK_DISPATCHER_INCLUDE_DIR  dispatcher/include
#   CK_TILE_INCLUDE_DIR        composablekernel/include
function(ck_dsl_provider_resolve_dispatcher_include)
    set(_searchDir "${_ckDslProviderPathsCmakeDir}/..")
    get_filename_component(_searchDir "${_searchDir}" ABSOLUTE)

    set(_ckRelMarker "projects/composablekernel/dispatcher/include/ck_tile/dispatcher/fmha_kernel_key.hpp")
    set(_resolvedCkRoot "")

    while(NOT _resolvedCkRoot AND NOT _searchDir STREQUAL "/")
        if(EXISTS "${_searchDir}/${_ckRelMarker}")
            set(_resolvedCkRoot "${_searchDir}/projects/composablekernel")
            break()
        endif()
        get_filename_component(_parent "${_searchDir}" DIRECTORY)
        if(_parent STREQUAL _searchDir)
            break()
        endif()
        set(_searchDir "${_parent}")
    endwhile()

    if(NOT _resolvedCkRoot OR NOT EXISTS "${_resolvedCkRoot}/dispatcher/include")
        message(FATAL_ERROR
            "CK DSL provider: failed to locate the CK-Tile dispatcher "
            "include directory. Walked up from "
            "${CMAKE_CURRENT_LIST_DIR}/.. looking for ${_ckRelMarker}. "
            "Set CK_DISPATCHER_INCLUDE_DIR explicitly to override the search.")
    endif()

    set(CK_DISPATCHER_INCLUDE_DIR "${_resolvedCkRoot}/dispatcher/include" PARENT_SCOPE)
    set(CK_TILE_INCLUDE_DIR "${_resolvedCkRoot}/include" PARENT_SCOPE)

    message(STATUS
        "CK DSL provider dispatcher includes: "
        "dispatcher=${_resolvedCkRoot}/dispatcher/include, "
        "ck_tile=${_resolvedCkRoot}/include")
endfunction()

# Resolve the absolute path of the in-tree LightGBM model the
# FMHA-forward scorer loads
# (projects/composablekernel/dispatcher/heuristics/models/
#  fmha_fwd_gfx950/model_tflops.lgbm). Walks up from the provider source
# directory the same way the other resolvers do. The model points into
# the SOURCE tree for the POC (it is not installed beside the .so).
#
# Output (set in caller's scope):
#   CK_DSL_FMHA_FWD_MODEL_PATH  absolute path to model_tflops.lgbm
function(ck_dsl_provider_resolve_fmha_fwd_model)
    set(_searchDir "${_ckDslProviderPathsCmakeDir}/..")
    get_filename_component(_searchDir "${_searchDir}" ABSOLUTE)

    set(_modelRelPath
        "projects/composablekernel/dispatcher/heuristics/models/fmha_fwd_gfx950/model_tflops.lgbm")
    set(_resolvedModelPath "")

    while(NOT _resolvedModelPath AND NOT _searchDir STREQUAL "/")
        if(EXISTS "${_searchDir}/${_modelRelPath}")
            set(_resolvedModelPath "${_searchDir}/${_modelRelPath}")
            break()
        endif()
        get_filename_component(_parent "${_searchDir}" DIRECTORY)
        if(_parent STREQUAL _searchDir)
            break()
        endif()
        set(_searchDir "${_parent}")
    endwhile()

    if(NOT _resolvedModelPath OR NOT EXISTS "${_resolvedModelPath}")
        message(FATAL_ERROR
            "CK DSL provider: failed to locate the FMHA-forward gfx950 "
            "LightGBM model. Walked up from "
            "${CMAKE_CURRENT_LIST_DIR}/.. looking for ${_modelRelPath}. "
            "The model must be decompressed in-tree for the POC scorer. "
            "Set CK_DSL_FMHA_FWD_MODEL_PATH explicitly to override the search.")
    endif()

    set(CK_DSL_FMHA_FWD_MODEL_PATH "${_resolvedModelPath}" PARENT_SCOPE)

    message(STATUS
        "CK DSL provider FMHA-forward model: ${_resolvedModelPath}")
endfunction()

# Resolve the absolute path of the in-tree LightGBM model the
# implicit-GEMM conv-forward scorer loads
# (projects/composablekernel/dispatcher/heuristics/models/
#  grouped_conv_forward_2d3d_suffix_bf16_gfx950/model_tflops.lgbm). The
# model is bf16/gfx950 only -- the scorer's caller short-circuits to the
# analytic fallback for any other dtype/arch.
#
# Same walk-up search as the other resolvers. The conv models ship
# gzipped in-tree; this resolver expects the decompressed .lgbm to exist
# (the scorer prints a "decompress first" hint if it does not load).
#
# Output (set in caller's scope):
#   CK_DSL_GROUPED_CONV_FWD_MODEL_PATH  absolute path to model_tflops.lgbm
function(ck_dsl_provider_resolve_grouped_conv_fwd_model)
    set(_searchDir "${_ckDslProviderPathsCmakeDir}/..")
    get_filename_component(_searchDir "${_searchDir}" ABSOLUTE)

    set(_modelRelPath
        "projects/composablekernel/dispatcher/heuristics/models/grouped_conv_forward_2d3d_suffix_bf16_gfx950/model_tflops.lgbm")
    set(_resolvedModelPath "")

    while(NOT _resolvedModelPath AND NOT _searchDir STREQUAL "/")
        if(EXISTS "${_searchDir}/${_modelRelPath}")
            set(_resolvedModelPath "${_searchDir}/${_modelRelPath}")
            break()
        endif()
        get_filename_component(_parent "${_searchDir}" DIRECTORY)
        if(_parent STREQUAL _searchDir)
            break()
        endif()
        set(_searchDir "${_parent}")
    endwhile()

    if(NOT _resolvedModelPath OR NOT EXISTS "${_resolvedModelPath}")
        message(FATAL_ERROR
            "CK DSL provider: failed to locate the grouped-conv-forward "
            "gfx950 bf16 LightGBM model. Walked up from "
            "${CMAKE_CURRENT_LIST_DIR}/.. looking for ${_modelRelPath}. "
            "The model ships gzipped in-tree; decompress with "
            "`gunzip ${_modelRelPath}.gz`. Set "
            "CK_DSL_GROUPED_CONV_FWD_MODEL_PATH explicitly to override the search.")
    endif()

    set(CK_DSL_GROUPED_CONV_FWD_MODEL_PATH "${_resolvedModelPath}" PARENT_SCOPE)

    message(STATUS
        "CK DSL provider grouped-conv-forward model: ${_resolvedModelPath}")
endfunction()

# Resolve the absolute path of the in-tree LightGBM model for the
# fp16/gfx942 implicit-GEMM conv-forward scorer
# (projects/composablekernel/dispatcher/heuristics/models/
#  grouped_conv_forward_fp16_gfx942/model_tflops.lgbm).
#
# Same walk-up search as the other resolvers. Like the bf16/gfx950
# model, this model is uncompressed (not gzipped) so it can be loaded
# directly by the scorer at runtime.
#
# Output (set in caller's scope):
#   CK_DSL_GROUPED_CONV_FWD_FP16_GFX942_MODEL_PATH  absolute path to model_tflops.lgbm
function(ck_dsl_provider_resolve_grouped_conv_fwd_fp16_gfx942_model)
    set(_searchDir "${_ckDslProviderPathsCmakeDir}/..")
    get_filename_component(_searchDir "${_searchDir}" ABSOLUTE)

    set(_modelRelPath
        "projects/composablekernel/dispatcher/heuristics/models/grouped_conv_forward_fp16_gfx942/model_tflops.lgbm")
    set(_resolvedModelPath "")

    while(NOT _resolvedModelPath AND NOT _searchDir STREQUAL "/")
        if(EXISTS "${_searchDir}/${_modelRelPath}")
            set(_resolvedModelPath "${_searchDir}/${_modelRelPath}")
            break()
        endif()
        get_filename_component(_parent "${_searchDir}" DIRECTORY)
        if(_parent STREQUAL _searchDir)
            break()
        endif()
        set(_searchDir "${_parent}")
    endwhile()

    if(NOT _resolvedModelPath OR NOT EXISTS "${_resolvedModelPath}")
        message(FATAL_ERROR
            "CK DSL provider: failed to locate the grouped-conv-forward "
            "fp16/gfx942 LightGBM model. Walked up from "
            "${CMAKE_CURRENT_LIST_DIR}/.. looking for ${_modelRelPath}. "
            "The model must exist in-tree (run the fp16/gfx942 training "
            "pipeline to produce it). Set "
            "CK_DSL_GROUPED_CONV_FWD_FP16_GFX942_MODEL_PATH explicitly to override the search.")
    endif()

    set(CK_DSL_GROUPED_CONV_FWD_FP16_GFX942_MODEL_PATH "${_resolvedModelPath}" PARENT_SCOPE)

    message(STATUS
        "CK DSL provider grouped-conv-forward fp16/gfx942 model: ${_resolvedModelPath}")
endfunction()
