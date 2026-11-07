# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Probe only the test target's C++23 support; consumers still require C++20.
function(hostnumerics_add_expected_parity)
    if(NOT "cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        return()
    endif()

    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    check_cxx_source_compiles("
        #include <expected>
        #if !defined(__cpp_lib_expected) || __cpp_lib_expected < 202202L
        #error std::expected is unavailable
        #endif
        int main() { return *std::expected<int, int>(0); }
    " HOSTNUMERICS_HAS_STD_EXPECTED)
    unset(CMAKE_CXX_STANDARD)
    unset(CMAKE_CXX_STANDARD_REQUIRED)
    cmake_pop_check_state()

    if(HOSTNUMERICS_HAS_STD_EXPECTED)
        add_executable(hostnumerics-std-expected-tests
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tests/expected_test.cpp")
        target_compile_features(hostnumerics-std-expected-tests PRIVATE cxx_std_23)
        target_compile_definitions(hostnumerics-std-expected-tests PRIVATE HOSTNUMERICS_TEST_STD_EXPECTED)
        target_link_libraries(hostnumerics-std-expected-tests PRIVATE GTest::gtest_main)
        gtest_discover_tests(
            hostnumerics-std-expected-tests
            DISCOVERY_MODE PRE_TEST
            TEST_PREFIX "hostnumerics-std-expected."
        )
    endif()
endfunction()

hostnumerics_add_expected_parity()
