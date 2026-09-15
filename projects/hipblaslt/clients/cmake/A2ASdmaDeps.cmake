# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# Resolves the SDMA-queue dependency chain shared by the fused-A2A sample and
# the fused-A2A benchmark. Sets HIPBLASLT_SDMA_QUEUE_INCLUDE_DIR in the caller's
# scope.
function(hipblaslt_require_a2a_sdma_deps)
    if(NOT DEFINED ROCM_PATH)
        if(DEFINED ENV{ROCM_PATH})
            set(ROCM_PATH "$ENV{ROCM_PATH}")
        else()
            set(ROCM_PATH "/opt/rocm")
        endif()
    endif()

    if(NOT TARGET numa::numa)
        find_library(NUMA_LIBRARY
            NAMES numa
            HINTS "${ROCM_PATH}/lib/rocm_sysdeps/lib"
            PATHS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
        find_path(NUMA_INCLUDE_DIR
            NAMES numa.h
            HINTS "${ROCM_PATH}/include/rocm_sysdeps/include"
            PATHS "${ROCM_PATH}/include/rocm_sysdeps/include")
        if(NUMA_LIBRARY)
            add_library(numa::numa UNKNOWN IMPORTED)
            set_target_properties(numa::numa PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARY}")
            if(NUMA_INCLUDE_DIR)
                set_target_properties(numa::numa PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
            endif()
        else()
            message(FATAL_ERROR
                "libnuma not found but required by hsakmt::hsakmt. Searched the "
                "ROCm-vendored path '${ROCM_PATH}/lib/rocm_sysdeps/lib' and the "
                "default system library paths. Set ROCM_PATH if your ROCm is "
                "elsewhere, or point NUMA_LIBRARY at the libnuma.so to use.")
        endif()
    endif()

    find_package(hsa-runtime64 REQUIRED)
    find_package(hsakmt REQUIRED)

    if(TARGET hsakmt::hsakmt)
        include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../tensilelite/cmake/HsakmtLinkInterface.cmake")
        get_target_property(_hsakmt_ill hsakmt::hsakmt INTERFACE_LINK_LIBRARIES)
        if(_hsakmt_ill)
            tensilelite_sanitize_hsakmt_link_interface(_hsakmt_ill_clean
                "${ROCM_PATH}/lib/rocm_sysdeps/lib" _hsakmt_ill)
            set_target_properties(hsakmt::hsakmt PROPERTIES
                INTERFACE_LINK_LIBRARIES "${_hsakmt_ill_clean}")
        endif()
    endif()

    set(HIPBLASLT_SDMA_QUEUE_INCLUDE_DIR
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../tensilelite/client/include"
        PARENT_SCOPE)
endfunction()
