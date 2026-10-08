# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Per-GPU-target shared libraries for CK grouped convolution operations.
# Libraries keep per-arch naming for runtime selection while supporting
# two CK linkage modes:
#   - MIOPEN_CK_UNBUNDLE_PER_ARCH=ON: link per-arch unbundled CK archives.
#   - MIOPEN_CK_UNBUNDLE_PER_ARCH=OFF: link the CK conv library built for this plugin's
#     target, or the unified CK device archive when CK has no per-target libraries.

if(NOT MIOPEN_USE_COMPOSABLEKERNEL)
    return()
endif()

# MIOPEN_CK_PLUGIN_TARGETS (top-level CMakeLists.txt) holds the full target ids that get a
# plugin, one per base arch; MIOPEN_CK_SELECTED_TARGETS is the selection before targets
# without a CK library were dropped.
if(NOT MIOPEN_CK_PLUGIN_TARGETS)
    if(MIOPEN_CK_SELECTED_TARGETS)
        message(STATUS "MIOpen CK: no CK library for any selected GPU target; "
                       "skipping CK grouped conv library build.")
    else()
        message(STATUS "MIOpen CK: no supported GPU targets found in GPU_TARGETS; "
                       "skipping CK grouped conv library build.")
    endif()
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
    ck_depthwise_bwd_data_impl.cpp
)

# Collect plugin target names for the MIOpen_with_plugins proxy (see src/CMakeLists.txt).
set(CK_PLUGIN_TARGETS)
foreach(gpu_target IN LISTS MIOPEN_CK_PLUGIN_TARGETS)
    ck_impl_base_arch("${gpu_target}" arch_base)
    miopen_sanitize_arch("${gpu_target}" arch_safe)

    set(lib_name "MIOpenCKGroupedConv_${arch_base}")

    if(TARGET ${lib_name})
        continue()
    endif()

    list(APPEND CK_PLUGIN_TARGETS ${lib_name})
    if(MIOPEN_CK_UNBUNDLE_PER_ARCH)
        set(_ck_link_desc "ck_conv_${arch_safe}")
    elseif(MIOPEN_CK_PER_TARGET)
        set(_ck_link_desc "device_conv_operations_${arch_safe}")
    else()
        set(_ck_link_desc "unified device_conv_operations (fallback)")
    endif()
    message(STATUS "MIOpen CK: ${lib_name} links ${_ck_link_desc}")
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

    # Leave AMDGPU device code uninstrumented under a host UBSan build (see the detection in the
    # top-level CMakeLists.txt). These CK-impl plugins device-compile the same way the main
    # MIOpen library does, so without this their device objects fail to link with undefined
    # __ubsan_handle_*_abort symbols. -Xarch_device scopes -fno-sanitize=undefined to the device
    # sub-compile only, so host code keeps UBSan.
    if(MIOPEN_DISABLE_DEVICE_UBSAN)
        target_compile_options(${lib_name} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xarch_device -fno-sanitize=undefined>")
    endif()

    # Mark as building the DLL for dllexport on Windows
    target_compile_definitions(${lib_name} PRIVATE CK_IMPL_BUILDING_DLL)

    # Per-target CK compile definitions. Use the shared helper (passing this
    # plugin's single GPU arch) so the plugin's CK macro set stays in sync with
    # the main MIOpen target and with how CK itself was built.
    miopen_apply_ck_compile_definitions(${lib_name} ${gpu_target})

    # Select the wavefront-size-specialized depthwise instance set for this GPU.
    # CDNA (gfx9xx) uses 64-lane wavefronts; RDNA (gfx10xx/gfx11xx/gfx12xx) uses 32.
    if(arch_base MATCHES "^gfx9")
        target_compile_definitions(${lib_name} PRIVATE MIOPEN_CK_DEPTHWISE_WAVE_SIZE=64)
    else()
        target_compile_definitions(${lib_name} PRIVATE MIOPEN_CK_DEPTHWISE_WAVE_SIZE=32)
    endif()

    # Choose CK linkage mode:
    #   - Per-arch unbundled archive
    #   - This target's own CK conv library (CK built or installed per-target)
    #   - Unified CK device_conv_operations (CK without per-target libraries)
    if(MIOPEN_CK_UNBUNDLE_PER_ARCH)
        target_link_libraries(${lib_name} PRIVATE ck_conv_${arch_safe})
    elseif(MIOPEN_CK_PER_TARGET AND MIOPEN_BUILD_CK)
        target_link_libraries(${lib_name} PRIVATE device_conv_operations_${arch_safe})
    elseif(MIOPEN_CK_PER_TARGET)
        target_link_libraries(${lib_name} PRIVATE composable_kernel::device_conv_operations_${arch_safe})
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

    # These plugins consume MIOpen's internal C++ types (ConvSolution,
    # InvokerFactory, ProblemDescription, etc.), not its C API, so they must link
    # whichever library carries those symbols. _MIOPEN_IMPL_TARGET names it in
    # either configuration: the single MIOpen library by default, or
    # MIOpen_private when MIOPEN_ENABLE_HIPDNN_WRAPPER=ON splits the internals out
    # behind a public wrapper that exports only the C API.
    target_link_libraries(${lib_name} PRIVATE ${_MIOPEN_IMPL_TARGET})

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
