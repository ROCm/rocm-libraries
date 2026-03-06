# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_compile_definitions(__HIP_PLATFORM_AMD__)
find_package(hip REQUIRED)
find_package(Threads REQUIRED)

find_package(hipdnn_test_sdk CONFIG REQUIRED)
find_package(hipdnn_plugin_sdk CONFIG REQUIRED)

add_executable(
    hip_kernel_plugin_tests
    main.cpp
    TestHipKernelContainer.cpp
    TestHipKernelHandle.cpp
    TestHipKernelContext.cpp
)

target_compile_options(hip_kernel_plugin_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_link_libraries(
    hip_kernel_plugin_tests PUBLIC GTest::gtest GTest::gmock hip::host
                                   hip_kernel_plugin_private hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# make it so tests inside folders can find files without having to use a relative path.
target_include_directories(hip_kernel_plugin_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

clang_tidy_check(hip_kernel_plugin_tests)

add_unit_test_target(hip_kernel_plugin_tests ${CMAKE_CURRENT_BINARY_DIR})
