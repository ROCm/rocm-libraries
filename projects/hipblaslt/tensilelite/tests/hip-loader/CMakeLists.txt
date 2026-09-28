# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest REQUIRED)

# Compile the adapter and its support code into this executable so --wrap can
# intercept HIP calls. Linking tensilelite-host as well would define the adapter
# twice and fail ASan's ODR check when the host library is also instrumented.
add_executable(tensilelite-loader-tests
    HipSolutionAdapter_test.cpp
    ../../src/hip/HipSolutionAdapter.cpp
    ../../src/Debug.cpp
    ../../src/KernelArguments.cpp
    # Provides the base SolutionAdapter destructor. Library readers are unused,
    # so this target does not enable TENSILE_MSGPACK or TENSILE_YAML.
    ../../src/Tensile.cpp
)
# Match TheRock's bin/hipblaslt-test* test-artifact include rule.
set_target_properties(tensilelite-loader-tests PROPERTIES
    OUTPUT_NAME hipblaslt-test-tensilelite-loader
)
target_link_libraries(tensilelite-loader-tests PRIVATE
    rocisa::rocisa-cpp roc::origami hip::device GTest::gtest_main
)
target_include_directories(tensilelite-loader-tests PRIVATE
    ../../include
    "${CMAKE_CURRENT_BINARY_DIR}/../../include"
)
target_compile_features(tensilelite-loader-tests PRIVATE cxx_std_17)
target_compile_definitions(tensilelite-loader-tests PRIVATE __HIP_HCC_COMPAT_MODE__=1)
# Preserve host sanitizer/coverage options without linking the host library.
target_compile_options(tensilelite-loader-tests PRIVATE
    $<TARGET_PROPERTY:tensilelite-host,INTERFACE_COMPILE_OPTIONS>
)
target_link_options(tensilelite-loader-tests PRIVATE
    $<TARGET_PROPERTY:tensilelite-host,INTERFACE_LINK_OPTIONS>
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
