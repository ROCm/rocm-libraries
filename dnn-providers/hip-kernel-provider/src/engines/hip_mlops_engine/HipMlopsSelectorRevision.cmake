# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/HkpSourceDigest.cmake")

# ============================================================================
# hkp_hip_mlops_revision(<out-revision> <engine-dir> [CONFIGURE_DEPENDS])
# ============================================================================
# The HIP_MLOPS_ENGINE selector revision: 16 hex digits over everything that decides which
# kernel a graph gets and the code it runs. L1 models record it and the loader refuses a
# mismatch, so a missing input lets a stale model estimate a kernel that changed.
#
# Every source under the engine directory is an input: the plan builders choose the kernel
# and its compile options and launch geometry, kernels/ is the source hiprtc compiles, and
# the compiler and module cache are the path between them. So are the provider's compiler
# (compilation/Program.*, which passes the options to hiprtc) and its compile-option
# builder (compilation/KernelCompileOptions.hpp). Not inputs: CMakeLists.txt, and
# HipMlopsL1Models.hpp, the table of declared model ids -- adding a model for one
# architecture must not expire the others.
#
# CONFIGURE_DEPENDS makes every input a configure dependency so an edit or a new source
# reconfigures.
function(hkp_hip_mlops_revision _out_revision _engine_dir)
    cmake_parse_arguments(PARSE_ARGV 2 _arg "CONFIGURE_DEPENDS" "" "")
    set(_track "")
    if(_arg_CONFIGURE_DEPENDS)
        set(_track CONFIGURE_DEPENDS)
    endif()

    file(GLOB_RECURSE _inputs ${_track}
        "${_engine_dir}/*.cpp"
        "${_engine_dir}/*.hpp"
        "${_engine_dir}/*.h"
        "${_engine_dir}/*.hip")
    list(REMOVE_ITEM _inputs "${_engine_dir}/HipMlopsL1Models.hpp")

    get_filename_component(_src_dir "${_engine_dir}/../.." ABSOLUTE)
    foreach(_shared IN ITEMS
            compilation/Program.cpp
            compilation/Program.hpp
            compilation/KernelCompileOptions.hpp)
        if(NOT EXISTS "${_src_dir}/${_shared}")
            message(FATAL_ERROR
                "hip_mlops selector revision: input ${_shared} does not exist under "
                "${_src_dir}. A moved or renamed compiler source must move here too, or the "
                "revision silently stops naming it.")
        endif()
        list(APPEND _inputs "${_src_dir}/${_shared}")
    endforeach()

    hkp_source_digest(_revision "${_src_dir}" ${_inputs})
    set(${_out_revision} "${_revision}" PARENT_SCOPE)
    if(_arg_CONFIGURE_DEPENDS)
        # The calling directory's property: a function runs in its caller's directory.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_inputs})
    endif()
endfunction()
