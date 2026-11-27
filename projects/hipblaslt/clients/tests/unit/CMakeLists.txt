# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Host-only client unit tests, compiled into both hipblaslt-test and the
# standalone hipblaslt-test-hostunit binary below. See README.md for why, for
# the *HostUnit* selection wiring, and for how to add a new test here.
set(HIPBLASLT_HOST_UNIT_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/alpha_mode_logging_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/adaptive_sm_count_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allclose_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/arch_name_cache_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/arguments_init_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_options_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_verify_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cblas_reference_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exception_logging_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grouped_predicate_problem_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mx_integer_exact_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mx_scale_format_predicate_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mx_scale_setup_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mx_tile128x4_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parser_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocroller_kernel_match_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocroller_kernel_select_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocroller_streamk_status_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scaling_format_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mxdatagen_maf1_0_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mxdatagen_maf2_0_gtest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yaml_conversion_gtest.cpp
)

# hipblaslt-test is created in clients/CMakeLists.txt before add_subdirectory(tests),
# so the target already exists here.
target_sources(hipblaslt-test PRIVATE ${HIPBLASLT_HOST_UNIT_TEST_SOURCES})

# rocroller_selected.hpp and solution_selection.hpp include "rocblaslt.h".
# That header lives next to the library sources, not on hipblaslt-test's path.
# The internal src/include directory must not be added: it holds a utility.hpp
# that shadows the clients' header of the same name.
set(HIPBLASLT_ROCBLASLT_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../library/src/amd_detail/rocblaslt/include")
target_include_directories(hipblaslt-test PRIVATE "${HIPBLASLT_ROCBLASLT_INCLUDE_DIR}")

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
target_include_directories(hipblaslt-test-hostunit PRIVATE "${HIPBLASLT_ROCBLASLT_INCLUDE_DIR}")

target_compile_definitions(hipblaslt-test-hostunit PRIVATE GOOGLE_TEST)
