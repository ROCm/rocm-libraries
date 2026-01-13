# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_auxiliary.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_generic_auxiliary.cpp"
)

target_link_libraries(hipsparse
    PRIVATE
        roc::rocsparse
)

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
add_subdirectory(extra)
add_subdirectory(precond)
add_subdirectory(reorder)
add_subdirectory(conversion)
add_subdirectory(generic)
