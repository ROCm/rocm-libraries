# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(Catch2 3 QUIET CONFIG)
if(NOT Catch2_FOUND)
    if(NOT TILEWRIGHT_ENABLE_FETCH)
        message(FATAL_ERROR "tilewright tests need Catch2 3: install it or set TILEWRIGHT_ENABLE_FETCH=ON")
    endif()
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.16.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()
include(Catch)

add_executable(tilewright-tests)
target_sources(
    tilewright-tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/model_writer.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_concurrency.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_env.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_loader.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_pool.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_rank.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_real_models.cpp"
)
# White-box tests (kernel variants, CRC) include the engine's private headers.
target_include_directories(tilewright-tests PRIVATE "${tilewright_SOURCE_DIR}/src/tilewright")
target_link_libraries(tilewright-tests PRIVATE roc::tilewright Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(
    tilewright-tests
    PRIVATE TILEWRIGHT_TEST_WEIGHTS_DIR="${tilewright_SOURCE_DIR}/weights/hipblaslt"
            TILEWRIGHT_TEST_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/data/sample_kernels.tsv"
)

catch_discover_tests(tilewright-tests TEST_SPEC "~[env]" DISCOVERY_MODE PRE_TEST)

# The engine reads its environment knobs once per process, so each setting
# runs in its own process.
add_test(NAME tilewright-env-forced COMMAND tilewright-tests "[env]")
set_tests_properties(
    tilewright-env-forced
    PROPERTIES ENVIRONMENT "TILEWRIGHT_FORCE_CELL=1;TILEWRIGHT_PICK_LOG=1;TILEWRIGHT_DIAG=1"
)
add_test(NAME tilewright-env-out-of-range COMMAND tilewright-tests "[env]")
set_tests_properties(
    tilewright-env-out-of-range PROPERTIES ENVIRONMENT "TILEWRIGHT_FORCE_CELL=96;TILEWRIGHT_PICK_LOG=0"
)
add_test(NAME tilewright-env-invalid COMMAND tilewright-tests "[env]")
set_tests_properties(
    tilewright-env-invalid PROPERTIES ENVIRONMENT "TILEWRIGHT_FORCE_CELL=1x;TILEWRIGHT_DIAG="
)
