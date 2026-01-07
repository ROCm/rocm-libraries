# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Configure version header
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rocsparse-version.h.in"
    "${PROJECT_BINARY_DIR}/include/rocsparse/rocsparse-version.h"
)

# Generate export header
include(GenerateExportHeader)
generate_export_header(rocsparse
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/rocsparse/rocsparse-export.h"
)

# Public headers will be installed separately via rocm_install in root CMakeLists.txt
# We don't use FILE_SET here to maintain compatibility with rocm_install_targets

# Copy headers to build directory for backward compatibility
file(GLOB_RECURSE ROCSPARSE_SOURCE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/include/rocsparse/.header_stamp
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${PROJECT_BINARY_DIR}/include/rocsparse
    COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_BINARY_DIR}/include/rocsparse/.header_stamp
    DEPENDS ${ROCSPARSE_SOURCE_HEADERS}
    COMMENT "Copying rocSPARSE headers to build directory"
)
add_custom_target(rocsparse_headers
    DEPENDS ${PROJECT_BINARY_DIR}/include/rocsparse/.header_stamp
)
add_dependencies(rocsparse rocsparse_headers)

# Add source files from src subdirectory
add_subdirectory(src)

# Create Fortran module OBJECT library if Fortran is enabled
if(ROCSPARSE_ENABLE_FORTRAN)
    # Set Fortran module output directory
    set(CMAKE_Fortran_MODULE_DIRECTORY "${PROJECT_BINARY_DIR}/include/rocsparse")

    # Create Fortran module OBJECT library
    add_library(rocsparse-fortran-module OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/src/rocsparse_enums.f90
        ${CMAKE_CURRENT_SOURCE_DIR}/src/rocsparse.f90
    )
endif()

# Target compile options
target_compile_options(rocsparse PRIVATE
    -Wno-unused-command-line-argument
    -Wall
)

# Target include directories
target_include_directories(rocsparse
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocsparse>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Compile definitions based on options
if(ROCSPARSE_ENABLE_MEMSTAT)
    target_compile_definitions(rocsparse PRIVATE ROCSPARSE_WITH_MEMSTAT)
endif()

if(ROCSPARSE_ENABLE_ILP64)
    target_compile_definitions(rocsparse PRIVATE rocsparse_ILP64)
endif()

if(rocblas_FOUND AND ROCSPARSE_ENABLE_ROCBLAS)
    target_compile_definitions(rocsparse PRIVATE ROCSPARSE_WITH_ROCBLAS)
endif()

# Link dependencies
target_link_libraries(rocsparse
    PRIVATE
        roc::rocprim
        hip::device
)

# Link rocBLAS if enabled
if(ROCSPARSE_ENABLE_ROCBLAS AND rocblas_FOUND)
    target_link_libraries(rocsparse PRIVATE roc::rocblas)
endif()

if(ROCSPARSE_ENABLE_MARKER)
    target_compile_definitions(rocsparse PRIVATE ROCSPARSE_BUILT_WITH_ROCTX)
    target_include_directories(rocsparse PRIVATE ${ROCTRACER_INCLUDE_DIR})
    target_link_libraries(rocsparse PRIVATE ${rocTracer})
endif()

# Target properties
set_target_properties(rocsparse PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    DEBUG_POSTFIX "-d"
)

# Windows: Copy DLL to clients staging for easier testing
if(WIN32 AND ROCSPARSE_ENABLE_CLIENT)
    add_custom_command(
        TARGET rocsparse POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:rocsparse>
                ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:rocsparse>
    )
    # Copy PDB file for Debug builds using generator expression
    add_custom_command(
        TARGET rocsparse POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy,true>
                ${PROJECT_BINARY_DIR}/staging/rocsparse.pdb
                ${PROJECT_BINARY_DIR}/clients/staging/rocsparse.pdb
        COMMAND_EXPAND_LISTS
    )
endif()

# Install Fortran module files
rocm_install(
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/rocsparse.f90"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/rocsparse_enums.f90"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocsparse"
)
