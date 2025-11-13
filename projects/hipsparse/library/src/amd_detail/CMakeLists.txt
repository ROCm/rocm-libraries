# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Add AMD backend sources organized by category
add_subdirectory(auxiliary)
add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
add_subdirectory(extra)
add_subdirectory(precond)
add_subdirectory(reorder)
add_subdirectory(conversion)
add_subdirectory(generic)

# AMD-specific compile definitions
target_compile_definitions(hipsparse PRIVATE __HIP_PLATFORM_AMD__)

# Link AMD dependencies with proper visibility
target_link_libraries(hipsparse
    PUBLIC hip::host
    PRIVATE roc::rocsparse
)
