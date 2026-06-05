# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Per-GPU-target shared libraries for CK grouped convolution operations.
# Libraries keep per-arch naming for runtime selection while supporting
# two CK linkage modes:
#   - MIOPEN_CK_UNBUNDLE_PER_ARCH=ON: link per-arch unbundled CK archives.
#   - MIOPEN_CK_UNBUNDLE_PER_ARCH=OFF: link full CK device archive/target.

if(NOT MIOPEN_USE_COMPOSABLEKERNEL)
    return()
endif()

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

# GPU base archs for which MIOpen has CK grouped conv kernels.
# Note: this is narrower than CK's own supported arch list — e.g. gfx103x and
# gfx11x are CK-capable but MIOpen does not yet provide CK grouped conv kernels
# for them. Add arches here when MIOpen adds grouped conv support for them.
set(_CK_SUPPORTED_ARCHS
    gfx908 gfx90a gfx942 gfx950
    gfx1100 gfx1101 gfx1102
    gfx1150 gfx1151 gfx1152 gfx1153
    gfx1200 gfx1201
)

# Filter GPU_TARGETS to only those with MIOpen CK grouped conv support.
set(_CK_FILTERED_TARGETS)
foreach(_gpu_target IN LISTS GPU_TARGETS)
    ck_impl_base_arch("${_gpu_target}" _base)
    if(_base IN_LIST _CK_SUPPORTED_ARCHS)
        list(APPEND _CK_FILTERED_TARGETS "${_gpu_target}")
    else()
        message(STATUS "MIOpen CK: skipping ${_gpu_target} (no MIOpen CK grouped conv support for this arch)")
    endif()
endforeach()

if(NOT _CK_FILTERED_TARGETS)
    message(STATUS "MIOpen CK: no supported GPU targets found in GPU_TARGETS; "
                   "skipping CK grouped conv library build.")
    return()
endif()

set(CK_IMPL_SOURCES
    ck_grouped_conv_common.cpp
    ck_grouped_conv_fwd_impl.cpp
    ck_grouped_conv_bwd_impl.cpp
    ck_grouped_conv_wrw_impl.cpp
    ck_grouped_conv_3d_fwd_impl.cpp
    ck_grouped_conv_3d_bwd_impl.cpp
    ck_grouped_conv_3d_wrw_impl.cpp
    ck_fused_bias_activ_impl.cpp
    ck_fused_bias_res_add_activ_impl.cpp
    ck_fused_grp_activ_impl.cpp
    ck_fused_grp_bias_activ_impl.cpp
    ck_depthwise_fwd_impl.cpp
)

# Collect plugin target names for the MIOpen_with_plugins proxy (see src/CMakeLists.txt).
set(CK_PLUGIN_TARGETS)
foreach(gpu_target IN LISTS _CK_FILTERED_TARGETS)
    ck_impl_base_arch("${gpu_target}" arch_base)
    if(MIOPEN_CK_UNBUNDLE_PER_ARCH)
        miopen_sanitize_arch("${gpu_target}" arch_safe)
    endif()

    set(lib_name "MIOpenCKGroupedConv_${arch_base}")

    if(TARGET ${lib_name})
        continue()
    endif()

    list(APPEND CK_PLUGIN_TARGETS ${lib_name})
    add_library(${lib_name} SHARED ${CK_IMPL_SOURCES})

    # Place the library alongside MIOpen in the build tree so the
    # runtime loader (which searches MIOpen's directory) can find
    # them without installation.
    set_target_properties(${lib_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )

    set_target_properties(${lib_name} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN 1
    )
    if(NOT WIN32 AND NOT APPLE)
        target_link_libraries(${lib_name} PRIVATE "-Wl,--exclude-libs,ALL")
    endif()

    # Full target string needed for correct device codegen
    target_compile_options(${lib_name} PRIVATE
        --offload-arch=${gpu_target}
    )

    # Mark as building the DLL for dllexport on Windows
    target_compile_definitions(${lib_name} PRIVATE CK_IMPL_BUILDING_DLL)

    # Per-target compile definitions
    if(gpu_target MATCHES "gfx942" OR gpu_target MATCHES "gfx950")
        target_compile_definitions(${lib_name} PRIVATE CK_ENABLE_TF32)
    endif()
    if(gpu_target MATCHES "gfx950")
        target_compile_definitions(${lib_name} PRIVATE CK_USE_GFX95)
    endif()

    # Choose CK linkage mode:
    #   - Per-arch unbundled archive
    #   - Full CK device_conv_operations archive/target
    if(MIOPEN_CK_UNBUNDLE_PER_ARCH)
        target_link_libraries(${lib_name} PRIVATE ck_conv_${arch_safe})
    elseif(MIOPEN_BUILD_CK)
        target_link_libraries(${lib_name} PRIVATE device_conv_operations)
    else()
        target_link_libraries(${lib_name} PRIVATE composable_kernel::device_conv_operations)
    endif()

    # CK include directories
    if(MIOPEN_BUILD_CK)
        target_include_directories(${lib_name} SYSTEM PRIVATE
            ${MIOPEN_CK_INCLUDE_DIR}
            ${MIOPEN_CK_BUILD_INCLUDE_DIR}
            ${MIOPEN_CK_LIBRARY_INCLUDE_DIR}
        )
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
    set_target_properties(${lib_name} PROPERTIES INSTALL_RPATH "$ORIGIN")
    install(TARGETS ${lib_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endforeach()

# Propagate the list to the parent scope so src/CMakeLists.txt can wire
# build-order dependencies into the MIOpen_with_plugins proxy target.
set(CK_PLUGIN_TARGETS "${CK_PLUGIN_TARGETS}" PARENT_SCOPE)
