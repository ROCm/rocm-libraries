# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(HIPSPARSE_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    add_library(hipsparse SHARED)
else()
    add_library(hipsparse STATIC)
endif()
add_library(roc::hipsparse ALIAS hipsparse)

rocm_set_soversion(hipsparse ${hipsparse_SOVERSION})
set_target_properties(hipsparse PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    DEBUG_POSTFIX "-d"
)

target_compile_features(hipsparse PRIVATE cxx_std_17)

target_compile_options(hipsparse PRIVATE -Wno-unused-command-line-argument -Wall)

if(HIPSPARSE_ENABLE_CUDA)
    target_compile_definitions(hipsparse PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_link_libraries(hipsparse PUBLIC hip::host)
elseif(HIPSPARSE_ENABLE_HIP)
    target_compile_definitions(hipsparse PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(hipsparse PUBLIC hip::host)
endif()

add_subdirectory(include)
add_subdirectory(src)

if(HIPSPARSE_ENABLE_FORTRAN)
    set(CMAKE_Fortran_MODULE_DIRECTORY "${PROJECT_BINARY_DIR}/include/hipsparse")
    
    add_library(hipsparse_fortran OBJECT)
    
    target_sources(hipsparse_fortran PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/hipsparse.f90"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/hipsparse_enums.f90"
    )
    
    target_compile_options(hipsparse_fortran PRIVATE -ffree-form -cpp)
endif()

rocm_install_targets(
    TARGETS hipsparse
    INCLUDE "${CMAKE_BINARY_DIR}/include"
)


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
    # Copy the DLL to clients/staging
    add_custom_command(
        TARGET hipsparse POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:hipsparse>"
            "${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:hipsparse>"
    )
    
    # Copy PDB file for Debug builds (only when it exists)
    add_custom_command(
        TARGET hipsparse POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
            "$<$<CONFIG:Debug>:$<TARGET_PDB_FILE:hipsparse>>"
            "$<$<CONFIG:Debug>:${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_PDB_FILE_NAME:hipsparse>>"
    )
endif()
