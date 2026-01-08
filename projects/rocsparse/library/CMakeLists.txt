# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rocsparse-version.h.in"
    "${PROJECT_BINARY_DIR}/include/rocsparse/rocsparse-version.h"
)

include(GenerateExportHeader)
generate_export_header(rocsparse
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/rocsparse/rocsparse-export.h"
)

add_subdirectory(src)

if(ROCSPARSE_ENABLE_FORTRAN)
    set(CMAKE_Fortran_MODULE_DIRECTORY "${PROJECT_BINARY_DIR}/include/rocsparse")
    add_library(rocsparse-fortran-module OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/src/rocsparse_enums.f90
        ${CMAKE_CURRENT_SOURCE_DIR}/src/rocsparse.f90
    )
endif()

target_compile_options(rocsparse PRIVATE
    -Wno-unused-command-line-argument
    -Wall
)

target_include_directories(rocsparse
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocsparse>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocsparse>
)

if(ROCSPARSE_ENABLE_MEMSTAT)
    target_compile_definitions(rocsparse PRIVATE ROCSPARSE_WITH_MEMSTAT)
endif()

if(ROCSPARSE_ENABLE_ILP64)
    target_compile_definitions(rocsparse PRIVATE rocsparse_ILP64)
endif()

if(rocblas_FOUND AND ROCSPARSE_ENABLE_ROCBLAS)
    target_compile_definitions(rocsparse PRIVATE ROCSPARSE_WITH_ROCBLAS)
endif()

target_link_libraries(rocsparse
    PRIVATE
        roc::rocprim
        hip::device
)

if(ROCSPARSE_ENABLE_ROCBLAS)
    target_link_libraries(rocsparse PRIVATE roc::rocblas)
endif()

if(ROCSPARSE_ENABLE_MARKER)
    target_compile_definitions(rocsparse PRIVATE ROCSPARSE_BUILT_WITH_ROCTX)
    target_include_directories(rocsparse PRIVATE ${ROCTRACER_INCLUDE_DIR})
    target_link_libraries(rocsparse PRIVATE ${rocTracer})
endif()

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
