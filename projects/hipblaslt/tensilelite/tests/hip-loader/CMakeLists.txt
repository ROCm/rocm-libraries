# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest REQUIRED)

# Compile the production adapter into this executable so --wrap intercepts its
# HIP calls even when tensilelite-host is a shared library. Keep the wrappers
# out of the library and the GPU test executables.
add_executable(tensilelite-loader-tests
    HipSolutionAdapter_test.cpp
    ../../src/hip/HipSolutionAdapter.cpp
    # launchKernel's diagnostic stream operator is private to the host library.
    ../../src/KernelArguments.cpp
)
# Match TheRock's bin/hipblaslt-test* test-artifact include rule.
set_target_properties(tensilelite-loader-tests PROPERTIES
    OUTPUT_NAME hipblaslt-test-tensilelite-loader
)
target_link_libraries(tensilelite-loader-tests PRIVATE
    roc::tensilelite-host hip::device GTest::gtest_main
)
target_link_options(tensilelite-loader-tests PRIVATE
    "LINKER:--wrap=hipModuleLoad"
    "LINKER:--wrap=hipModuleUnload"
    "LINKER:--wrap=hipGetLastError"
)
if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(tensilelite-loader-tests PRIVATE)
endif()

enable_testing()
add_test(NAME TensileLiteLoader COMMAND tensilelite-loader-tests)
set_tests_properties(TensileLiteLoader PROPERTIES TIMEOUT 30)
install(TARGETS tensilelite-loader-tests
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tests
)
