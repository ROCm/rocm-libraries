# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_auxiliary.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_generic_auxiliary.cpp"
)

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
add_subdirectory(extra)
add_subdirectory(precond)
add_subdirectory(reorder)
add_subdirectory(conversion)
add_subdirectory(generic)

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
    target_compile_definitions(hipsparse PRIVATE CUDART_10_1_UPDATE_VERSION=0)
else()
    target_compile_definitions(hipsparse PRIVATE CUDART_10_1_UPDATE_VERSION=1)
endif()

target_link_libraries(hipsparse PRIVATE ${CUDA_cusparse_LIBRARY})
