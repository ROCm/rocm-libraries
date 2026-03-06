# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# OBJECT library with all implementation
add_library(
    hip_kernel_plugin_impl OBJECT
    HipKernelHandle.cpp
    HipKernelContainer.cpp
    HipKernelPluginPublic.cpp
)
target_include_directories(hip_kernel_plugin_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hip_kernel_plugin_impl PUBLIC hipdnn_data_sdk hipdnn_plugin_sdk)

target_compile_options(hip_kernel_plugin_impl PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(hip_kernel_plugin_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
clang_tidy_check(hip_kernel_plugin_impl)
add_clang_tidy_custom_target()

# Static library for unit tests to link against (provides access to internal implementation)
add_library(hip_kernel_plugin_private STATIC $<TARGET_OBJECTS:hip_kernel_plugin_impl>)
target_include_directories(hip_kernel_plugin_private PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hip_kernel_plugin_private PUBLIC hipdnn_data_sdk hipdnn_plugin_sdk hip_kernel_plugin_impl)

# hip_kernel_plugin - SHARED library (the actual plugin)
add_library(hip_kernel_plugin SHARED $<TARGET_OBJECTS:hip_kernel_plugin_impl>)
target_link_libraries(hip_kernel_plugin PRIVATE hipdnn_data_sdk hipdnn_plugin_sdk)

if(NOT WIN32)
    target_link_libraries(hip_kernel_plugin PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

target_compile_options(hip_kernel_plugin PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    hip_kernel_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

# ============================================================================
# Tests
# ============================================================================
if(HIPKERNELPROVIDER_ENABLE_TESTS)
    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    # Keep this after all build folders have been added so they have a chance to register their tests
    # using add_*_test_target()
    finalize_test_targets()

    install_hip_kernel_plugin_ctest_files()
endif()
