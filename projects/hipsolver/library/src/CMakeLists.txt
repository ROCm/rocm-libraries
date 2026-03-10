# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# --- Fortran bindings ---

if(HIPSOLVER_ENABLE_FORTRAN)
    add_library(hipsolver_fortran)
    target_sources(hipsolver_fortran PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipsolver_module.f90"
    )
    rocm_set_soversion(hipsolver_fortran ${hipsolver_SOVERSION})
    rocm_install(TARGETS hipsolver_fortran)
endif()

# --- Main library target ---

add_library(hipsolver)
add_library(roc::hipsolver ALIAS hipsolver)

target_compile_features(hipsolver PRIVATE cxx_std_17)
set_target_properties(hipsolver PROPERTIES CXX_EXTENSIONS OFF)

# Warning suppression (was in hipsolver-common INTERFACE library)
target_compile_options(hipsolver PRIVATE -Wno-unused-result)

# --- Sources ---

if(NOT HIPSOLVER_ENABLE_CUDA)
    target_sources(hipsolver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipsolver_conversions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipsolver.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipsolver_dense.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipsolver_dense64.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipsolver_refactor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipsolver_sparse.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/hipsolver_dense_common.cpp"
    )
    if(NOT HIPSOLVER_ENABLE_SPARSE)
        target_sources(hipsolver PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/dlopen/cholmod.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/dlopen/rocsparse.cpp"
        )
    endif()
else()
    target_sources(hipsolver PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/nvidia_detail/hipsolver_conversions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvidia_detail/hipsolver.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvidia_detail/hipsolver_dense.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvidia_detail/hipsolver_dense64.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvidia_detail/hipsolver_refactor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvidia_detail/hipsolver_sparse.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/hipsolver_dense_common.cpp"
    )
endif()

# --- Static library define ---

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(hipsolver PRIVATE HIPSOLVER_STATIC_LIB)
    set_target_properties(hipsolver PROPERTIES PREFIX "lib")
endif()

# --- WIN32 compile definitions ---

if(WIN32)
    target_compile_definitions(hipsolver PRIVATE
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
endif()

# --- ASAN flags (was in hipsolver-common INTERFACE library) ---

if(HIPSOLVER_ENABLE_ASAN)
    target_compile_options(hipsolver PRIVATE -fsanitize=address -shared-libasan)
    target_link_options(hipsolver PRIVATE -fsanitize=address -shared-libasan -fuse-ld=lld)
endif()

# --- Code coverage flags (was in hipsolver-common INTERFACE library) ---

if(HIPSOLVER_BUILD_COVERAGE)
    target_compile_options(hipsolver PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipsolver PRIVATE -fprofile-instr-generate)
endif()

# --- Armor flags ---

add_armor_flags(hipsolver "${HIPSOLVER_ARMOR_LEVEL}")

# --- Dynamic loading for dlopen path ---

if(NOT HIPSOLVER_ENABLE_SPARSE)
    target_link_libraries(hipsolver PRIVATE $<$<PLATFORM_ID:Linux>:${CMAKE_DL_LIBS}>)
endif()

# --- Dependencies ---

set(static_depends)

if(NOT HIPSOLVER_ENABLE_CUDA)
    find_package(rocblas REQUIRED CONFIG)
    list(APPEND static_depends PACKAGE rocblas)

    find_package(rocsolver REQUIRED CONFIG)
    list(APPEND static_depends PACKAGE rocsolver)

    target_link_libraries(hipsolver PRIVATE roc::rocblas roc::rocsolver)
    target_link_libraries(hipsolver PUBLIC hip::host)

    if(CUSTOM_TARGET)
        target_link_libraries(hipsolver PRIVATE hip::${CUSTOM_TARGET})
    endif()

    if(HIPSOLVER_ENABLE_SPARSE)
        find_package(rocsparse REQUIRED CONFIG)
        list(APPEND static_depends PACKAGE rocsparse)

        find_package(CHOLMOD QUIET)
        if(NOT TARGET SuiteSparse::CHOLMOD)
            # try again with the custom find modules for older versions of suitesparse
            list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/suitesparse")
            find_package(CHOLMOD REQUIRED)
        endif()

        target_link_libraries(hipsolver PRIVATE roc::rocsparse SuiteSparse::CHOLMOD)
        target_compile_definitions(hipsolver PRIVATE HAVE_ROCSPARSE)
    endif()
else()
    target_compile_definitions(hipsolver PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_link_libraries(hipsolver PRIVATE ${CUDA_cusolver_LIBRARY})

    target_include_directories(hipsolver
        SYSTEM PRIVATE
            $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
            $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
endif()

# --- Include directories ---

target_include_directories(hipsolver
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include/internal>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipsolver>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipsolver/internal>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --- Target properties ---

rocm_set_soversion(hipsolver ${hipsolver_SOVERSION})
set_target_properties(hipsolver PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

# Symbol visibility
include(GenerateExportHeader)
set_target_properties(hipsolver PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
generate_export_header(hipsolver
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/hipsolver/internal/hipsolver-export.h"
)

# Copy public headers to build tree for consumers
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/library/include"
    "${PROJECT_BINARY_DIR}/include/hipsolver"
)

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
    rocm_wrap_header_file(
        internal/hipsolver-version.h internal/hipsolver-export.h
        GUARDS SYMLINK WRAPPER
        WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR} hipsolver/${CMAKE_INSTALL_INCLUDEDIR}
        ORIGINAL_FILES "${PROJECT_BINARY_DIR}/include/hipsolver/internal/hipsolver-version.h"
    )
endif()

# --- Windows DLL copy for clients ---

if(WIN32)
    if(HIPSOLVER_ENABLE_BENCHMARKS OR HIPSOLVER_BUILD_TESTING)
        add_custom_command(TARGET hipsolver POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                "${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:hipsolver>"
                "${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:hipsolver>"
        )
        # Copy PDB in Debug config
        add_custom_command(TARGET hipsolver POST_BUILD
            COMMAND "$<$<CONFIG:Debug>:${CMAKE_COMMAND}>" -E copy
                "${PROJECT_BINARY_DIR}/staging/$<TARGET_PDB_FILE_NAME:hipsolver>"
                "${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_PDB_FILE_NAME:hipsolver>"
        )
    endif()
endif()

# --- Installation ---

rocm_install_targets(
    TARGETS hipsolver
    INCLUDE "${CMAKE_BINARY_DIR}/include"
)

if(NOT HIPSOLVER_ENABLE_CUDA)
    rocm_export_targets(
        TARGETS roc::hipsolver
        DEPENDS PACKAGE hip
        STATIC_DEPENDS ${static_depends}
        NAMESPACE roc::
    )
else()
    rocm_export_targets(
        TARGETS roc::hipsolver
        DEPENDS PACKAGE HIP
        NAMESPACE roc::
    )
endif()

if(HIPSOLVER_ENABLE_FORTRAN)
    if(NOT HIPSOLVER_ENABLE_ASAN)
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipsolver_module.f90"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hipsolver"
        )
    endif()
endif()

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
    rocm_install(
        DIRECTORY "${PROJECT_BINARY_DIR}/hipsolver"
        DESTINATION "."
    )

    if(NOT WIN32)
        rocm_install(
            CODE "
                set(PREFIX \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX})
                set(INPUT_FILE \${PREFIX}/include/hipsolver/hipsolver_module.f90)
                set(SYMLINK_LOCATIONS \${PREFIX}/hipsolver/include \${PREFIX}/include)
                foreach(LOCATION IN LISTS SYMLINK_LOCATIONS)
                    file(MAKE_DIRECTORY \${LOCATION})
                    execute_process(COMMAND ln -sfr \${INPUT_FILE} \${LOCATION})
                    message(STATUS \"Created symlink in \${LOCATION} to \${INPUT_FILE}.\")
                endforeach()
            "
        )
    endif()
endif()
