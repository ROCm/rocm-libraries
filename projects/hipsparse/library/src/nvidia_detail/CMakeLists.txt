# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Auxiliary function sources
target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_auxiliary.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_generic_auxiliary.cpp"
)

# Add CUDA backend sources organized by category
add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
add_subdirectory(extra)
add_subdirectory(precond)
add_subdirectory(reorder)
add_subdirectory(conversion)
add_subdirectory(generic)

# CUDA-specific compile definitions
target_compile_definitions(hipsparse PRIVATE __HIP_PLATFORM_NVIDIA__)

# CUDA-specific includes
target_include_directories(hipsparse PRIVATE
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

# Check for cusparseCreateSpVec symbol to determine CUDA version
set(SEARCH_SYMBOL "cusparseCreateSpVec")
set(SEARCH_RESULTS "")
if(WIN32)
    execute_process(
        COMMAND dumpbin /EXPORTS ${CUDA_cusparse_LIBRARY}
        COMMAND findstr ${SEARCH_SYMBOL}
        OUTPUT_VARIABLE SEARCH_RESULTS
        ERROR_VARIABLE ERR
    )
else()
    execute_process(
        COMMAND objdump -T ${CUDA_cusparse_LIBRARY}
        COMMAND grep ${SEARCH_SYMBOL}
        OUTPUT_VARIABLE SEARCH_RESULTS
        ERROR_VARIABLE ERR
    )
endif()

if(SEARCH_RESULTS STREQUAL "")
    # symbol not found
    target_compile_definitions(hipsparse PRIVATE CUDART_10_1_UPDATE_VERSION=0)
else()
    # symbol found
    target_compile_definitions(hipsparse PRIVATE CUDART_10_1_UPDATE_VERSION=1)
endif()

# Link CUDA dependencies with proper visibility
target_link_libraries(hipsparse PRIVATE ${CUDA_cusparse_LIBRARY})
