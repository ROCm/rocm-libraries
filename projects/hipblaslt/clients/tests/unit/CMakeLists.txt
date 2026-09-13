# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Host-only client unit tests, compiled into both hipblaslt-test and the
# standalone hipblaslt-test-hostunit binary below. See README.md for why, for
# the *HostUnit* selection wiring, and for how to add a new test here.
set(HIPBLASLT_HOST_UNIT_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/parser_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scaling_format_gtest.cpp
)

# hipblaslt-test is created in clients/CMakeLists.txt before add_subdirectory(tests),
# so the target already exists here.
target_sources(hipblaslt-test PRIVATE ${HIPBLASLT_HOST_UNIT_TEST_SOURCES})

# hip::device (not hip::host) is required for compilation as HIP; see
# "Why hip::device" in README.md. This target does not use OpenMP directly, so
# it is not linked here.
add_executable(hipblaslt-test-hostunit ${HIPBLASLT_HOST_UNIT_TEST_SOURCES})

target_link_libraries(hipblaslt-test-hostunit
    PRIVATE
        hipblaslt::hipblaslt-clients-common
        hip::device
        GTest::gtest
        GTest::gtest_main
)

target_compile_definitions(hipblaslt-test-hostunit PRIVATE GOOGLE_TEST)
