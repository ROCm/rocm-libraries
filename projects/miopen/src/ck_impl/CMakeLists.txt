# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Per-GPU-target shared libraries for CK grouped convolution operations.
# Each library contains the extern "C" implementations that link against
# CK device code for a single GPU architecture.

if(NOT MIOPEN_USE_COMPOSABLEKERNEL)
    return()
endif()

set(CK_IMPL_SOURCES
    ck_grouped_conv_common.cpp
    ck_grouped_conv_fwd_impl.cpp
    ck_grouped_conv_bwd_impl.cpp
    ck_grouped_conv_wrw_impl.cpp
)

# Strip target features (e.g. ":sramecc+:xnack-") to get the base arch name.
# The loader uses the same stripping logic when looking up the .so.
function(ck_impl_base_arch ARCH OUT_VAR)
    string(FIND "${ARCH}" ":" _colon_pos)
    if(_colon_pos GREATER -1)
        string(SUBSTRING "${ARCH}" 0 ${_colon_pos} _base)
    else()
        set(_base "${ARCH}")
    endif()
    set(${OUT_VAR} "${_base}" PARENT_SCOPE)
endfunction()

foreach(gpu_target IN LISTS GPU_TARGETS)
    ck_impl_base_arch("${gpu_target}" arch_base)
    miopen_sanitize_arch("${gpu_target}" arch_safe)

    set(lib_name "MIOpenCKGroupedConv_${arch_base}")

    if(TARGET ${lib_name})
        continue()
    endif()

    add_library(${lib_name} SHARED ${CK_IMPL_SOURCES})

    # Place the library alongside MIOpen in the build tree so the
    # runtime loader (which searches MIOpen's directory) can find
    # them without installation.
    set_target_properties(${lib_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )

    # Full target string needed for correct device codegen
    target_compile_options(${lib_name} PRIVATE
        --offload-arch=${gpu_target}
    )

    # Mark as building the DLL for dllexport on Windows
    target_compile_definitions(${lib_name} PRIVATE CK_GROUPED_CONV_BUILDING_DLL)

    # Per-target compile definitions
    if(gpu_target MATCHES "gfx942" OR gpu_target MATCHES "gfx950")
        target_compile_definitions(${lib_name} PRIVATE CK_ENABLE_TF32)
    endif()
    if(gpu_target MATCHES "gfx950")
        target_compile_definitions(${lib_name} PRIVATE CK_USE_GFX95)
    endif()

    # Link against per-arch unbundled CK library
    target_link_libraries(${lib_name} PRIVATE ck_conv_${arch_safe})

    # CK include directories
    if(MIOPEN_BUILD_CK)
        target_include_directories(${lib_name} SYSTEM PRIVATE
            ${MIOPEN_CK_INCLUDE_DIR}
            ${MIOPEN_CK_BUILD_INCLUDE_DIR}
            ${MIOPEN_CK_LIBRARY_INCLUDE_DIR}
        )
    else()
        target_link_libraries(${lib_name} PRIVATE composable_kernel::device_conv_operations)
    endif()

    # MIOpen internal headers (for ProblemDescription, ConvSolution, etc.)
    target_include_directories(${lib_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/src/include
        ${PROJECT_BINARY_DIR}/include
    )

    # HIP runtime (hip::device required for compiling CK device code)
    target_link_libraries(${lib_name} PRIVATE hip::device)

    # Link against MIOpen for shared types (ConvSolution, InvokerFactory, etc.)
    target_link_libraries(${lib_name} PRIVATE MIOpen)

    # Install alongside MIOpen
    install(TARGETS ${lib_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endforeach()
