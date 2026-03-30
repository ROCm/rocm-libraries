# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Expected compilation failure tests for rocm_ck schema types.
#
# Each .cpp file must FAIL to compile. The CTest test PASSES when
# compilation FAILS (WILL_FAIL property). This verifies that consteval
# validation catches invalid configs at compile time.
#
# Usage:
#   ctest --test-dir build -R compile_fail        # run all compile-fail tests
#   ctest --test-dir build -L compile_fail         # same, via label
#
# Adding a new test:
#   1. Create a new .cpp file with the invalid constexpr expression
#   2. It is automatically picked up by the glob below

file(GLOB COMPILE_FAIL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(source ${COMPILE_FAIL_SOURCES})
    get_filename_component(test_name ${source} NAME_WE)
    set(target_name "compile_fail_${test_name}")

    add_library(${target_name} OBJECT EXCLUDE_FROM_ALL ${source})
    target_link_libraries(${target_name} PRIVATE rocm_ck)
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    target_compile_options(${target_name} PRIVATE
        -Wno-global-constructors
        -Wno-undef
    )

    add_test(
        NAME ${target_name}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${target_name}
    )
    set_tests_properties(${target_name} PROPERTIES
        WILL_FAIL TRUE
        LABELS "rocm_ck;compile_fail"
    )
endforeach()
