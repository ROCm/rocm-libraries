# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Host-only client unit tests, compiled into both hipblaslt-test and the
# standalone hipblaslt-test-hostunit binary below. See README.md for why, for
# the *HostUnit* selection wiring, and for how to add a new test here.
set(HIPBLASLT_HOST_UNIT_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/alpha_mode_logging_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/adaptive_sm_count_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allclose_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_options_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_verify_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cblas_reference_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mx_scale_format_predicate_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mx_scale_setup_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mx_tile128x4_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parser_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocroller_kernel_match_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocroller_kernel_select_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scaling_format_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yaml_conversion_gtest.cpp
)

# The unbuildable-configuration lookup is compiled in from the library, so its
# table is tested on a machine without a GPU. Only these two sources see the
# rocRoller ranker's headers and the table.
set(HIPBLASLT_ROCROLLER_UNBUILDABLE_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/rocroller_unbuildable_gtest.cpp
    ${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller_unbuildable.cpp
)
list(APPEND HIPBLASLT_HOST_UNIT_TEST_SOURCES ${HIPBLASLT_ROCROLLER_UNBUILDABLE_TEST_SOURCES})

# hipblaslt-test is created in clients/CMakeLists.txt before add_subdirectory(tests),
# so the target already exists here.
target_sources(hipblaslt-test PRIVATE ${HIPBLASLT_HOST_UNIT_TEST_SOURCES})

# hip::device (not hip::host) is required for compilation as HIP; see
# "Why hip::device" in README.md. This target does not use OpenMP directly, so
# it is not linked here.
add_executable(hipblaslt-test-hostunit ${HIPBLASLT_HOST_UNIT_TEST_SOURCES})

# YAML conversion finds its script and template next to the executable.
set_target_properties(hipblaslt-test-hostunit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients"
)

# roc::tensilelite-host arrives here transitively through roc::hipblaslt anyway;
# naming it is what puts TensileLite's include dirs on this target, including the
# generated tensilelitehost/export.h that Tensile/ContractionProblem.hpp needs.
# hipblaslt-test already links it the same way (see ../../CMakeLists.txt).
target_link_libraries(hipblaslt-test-hostunit
    PRIVATE
        hipblaslt::hipblaslt-clients-common
        hip::device
        roc::tensilelite-host
        GTest::gtest
        GTest::gtest_main
)

target_compile_definitions(hipblaslt-test-hostunit PRIVATE GOOGLE_TEST)

set_source_files_properties(${HIPBLASLT_ROCROLLER_UNBUILDABLE_TEST_SOURCES}
    TARGET_DIRECTORY hipblaslt-test hipblaslt-test-hostunit
    PROPERTIES INCLUDE_DIRECTORIES
        "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include;${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include;${PROJECT_SOURCE_DIR}/tensilelite/tensilelite/CustomKernels/rocroller"
)
