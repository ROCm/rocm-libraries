# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Auxiliary function sources
target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_auxiliary.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_generic_auxiliary.cpp"
)

# Link AMD dependencies with proper visibility
target_link_libraries(hipsparse
    PUBLIC hip::host
    PRIVATE roc::rocsparse
)

# AMD-specific compile definitions
target_compile_definitions(hipsparse PRIVATE __HIP_PLATFORM_AMD__)

# Add AMD backend sources organized by category
add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
add_subdirectory(extra)
add_subdirectory(precond)
add_subdirectory(reorder)
add_subdirectory(conversion)
add_subdirectory(generic)
