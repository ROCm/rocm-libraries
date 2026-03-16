# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

# SDK dependencies - use targets directly in superbuild, find_package otherwise
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()

if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    sdpa_kernel_plugin_tests
    main.cpp
    TestAsmSdpaFwdKernelArgs.cpp
    TestSdpaKernelContainer.cpp
    TestSdpaKernelHandle.cpp
    TestSdpaKernelContext.cpp
)

target_compile_options(sdpa_kernel_plugin_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    sdpa_kernel_plugin_tests PUBLIC GTest::gtest GTest::gmock hip::host
                                   sdpa_kernel_plugin_private hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# make it so tests inside folders can find files without having to use a relative path.
target_include_directories(sdpa_kernel_plugin_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

clang_tidy_check(sdpa_kernel_plugin_tests)

add_unit_test_target(sdpa_kernel_plugin_tests ${CMAKE_CURRENT_BINARY_DIR})
