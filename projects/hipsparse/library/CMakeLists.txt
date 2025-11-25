# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Define library target with explicit type
if(HIPSPARSE_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    add_library(hipsparse SHARED)
else()
    add_library(hipsparse STATIC)
endif()
add_library(roc::hipsparse ALIAS hipsparse)

# Target properties
rocm_set_soversion(hipsparse ${hipsparse_SOVERSION})
set_target_properties(hipsparse PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    DEBUG_POSTFIX "-d"
)

# Following boost conventions of prefixing 'lib' on static built libraries
if(NOT (HIPSPARSE_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS))
    set_target_properties(hipsparse PROPERTIES PREFIX "lib")
endif()

# Compile features (NOT global CMAKE_CXX_STANDARD)
target_compile_features(hipsparse PRIVATE cxx_std_17)

# Common compile options
target_compile_options(hipsparse PRIVATE -Wno-unused-command-line-argument -Wall)

# Conditional flags for ASAN
if(HIPSPARSE_ENABLE_ASAN)
    target_compile_options(hipsparse PRIVATE -fsanitize=address -shared-libasan)
    target_link_options(hipsparse PRIVATE -fuse-ld=lld)
endif()

# Conditional flags for coverage
if(HIPSPARSE_BUILD_COVERAGE)
    target_compile_options(hipsparse PRIVATE -fprofile-arcs -ftest-coverage)
    target_link_options(hipsparse PRIVATE --coverage)
endif()

# Add subdirectories for header management and sources
add_subdirectory(include)
add_subdirectory(src)

# Create Fortran module if not on Windows
if(NOT WIN32)
    # Set Fortran module output directory
    set(CMAKE_Fortran_MODULE_DIRECTORY "${PROJECT_BINARY_DIR}/include/hipsparse")
    
    # Create hipSPARSE Fortran module
    add_library(hipsparse_fortran OBJECT)
    
    target_sources(hipsparse_fortran PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/hipsparse.f90"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/hipsparse_enums.f90"
    )
    
    # Target compile options
    target_compile_options(hipsparse_fortran PRIVATE -ffree-form -cpp)
endif()

# Installation
rocm_install_targets(
    TARGETS hipsparse
    INCLUDE ${CMAKE_BINARY_DIR}/include
)

# Export targets
if(HIPSPARSE_ENABLE_HIP)
    rocm_export_targets(
        TARGETS roc::hipsparse
        DEPENDS PACKAGE hip
        STATIC_DEPENDS PACKAGE rocsparse
        NAMESPACE roc::
    )
elseif(HIPSPARSE_ENABLE_CUDA)
    rocm_export_targets(
        TARGETS roc::hipsparse
        NAMESPACE roc::
    )
endif()

# Windows-specific post-build copy for shared libraries
if(WIN32 AND (HIPSPARSE_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS))
    add_custom_command(
        TARGET hipsparse POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:hipsparse>
            ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:hipsparse>
    )
    # Copy PDB file for Debug builds (using generator expression to avoid CMAKE_BUILD_TYPE check)
    add_custom_command(
        TARGET hipsparse POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${PROJECT_BINARY_DIR}/staging/hipsparse$<$<CONFIG:Debug>:.pdb>
            ${PROJECT_BINARY_DIR}/clients/staging/hipsparse$<$<CONFIG:Debug>:.pdb>
    )
endif()
