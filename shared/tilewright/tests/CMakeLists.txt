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

set(_tw_test_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/model_writer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_concurrency.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_env.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_loader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_pool.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rank.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_real_models.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_v3.cpp"
)
add_executable(tilewright-tests)
target_sources(tilewright-tests PRIVATE ${_tw_test_sources})
# White-box tests (kernel variants, CRC) include the engine's private headers.
target_include_directories(tilewright-tests PRIVATE "${tilewright_SOURCE_DIR}/src/tilewright")
target_link_libraries(tilewright-tests PRIVATE roc::tilewright Catch2::Catch2WithMain Threads::Threads)
target_compile_definitions(
    tilewright-tests
    PRIVATE TILEWRIGHT_TEST_WEIGHTS_DIR="${tilewright_SOURCE_DIR}/weights/hipblaslt"
            TILEWRIGHT_TEST_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/data/sample_kernels.tsv"
)

catch_discover_tests(tilewright-tests TEST_SPEC "~[env]" DISCOVERY_MODE PRE_TEST)

get_target_property(_tw_engine_sources tilewright SOURCES)
# The engine again with the selection-time options (TW_LEVER_PREFETCH and
# TW_LEVER_MEMO) as library `name`, the memo's plan check TW_LEVER_MEMO_CHECK
# set to `memo_check`, and the whole suite plus test_levers.cpp against it in
# one process (ctest entry `name`), so that the thread's memo also outlives
# the rankings of one test.
function(_tw_levers_suite name memo_check)
    add_library(${name} STATIC ${_tw_engine_sources})
    set_target_properties(
        ${name}
        PROPERTIES POSITION_INDEPENDENT_CODE ON
                   CXX_VISIBILITY_PRESET hidden
                   VISIBILITY_INLINES_HIDDEN ON
    )
    target_compile_features(${name} PUBLIC cxx_std_17)
    target_include_directories(${name} PUBLIC "${tilewright_SOURCE_DIR}/include")
    target_link_libraries(${name} PUBLIC Threads::Threads)
    target_compile_options(${name} PRIVATE $<TARGET_PROPERTY:tilewright,COMPILE_OPTIONS>)
    target_compile_definitions(
        ${name}
        PUBLIC TW_LEVER_PREFETCH=1 TW_LEVER_MEMO=1 TW_LEVER_MEMO_CHECK=${memo_check}
        PRIVATE $<TARGET_PROPERTY:tilewright,COMPILE_DEFINITIONS>
    )
    add_executable(${name}-tests)
    target_sources(
        ${name}-tests PRIVATE ${_tw_test_sources} "${CMAKE_CURRENT_SOURCE_DIR}/test_levers.cpp"
    )
    target_include_directories(${name}-tests PRIVATE "${tilewright_SOURCE_DIR}/src/tilewright")
    target_link_libraries(${name}-tests PRIVATE ${name} Catch2::Catch2WithMain Threads::Threads)
    target_compile_definitions(
        ${name}-tests
        PRIVATE TILEWRIGHT_TEST_WEIGHTS_DIR="${tilewright_SOURCE_DIR}/weights/hipblaslt"
                TILEWRIGHT_TEST_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/data/sample_kernels.tsv"
    )
    add_test(NAME ${name} COMMAND ${name}-tests "~[env]")
endfunction()
# With the plan check (the default where assertions are), and without it, as
# a release build compiles the memo.
_tw_levers_suite(tilewright-levers 1)
_tw_levers_suite(tilewright-levers-nocheck 0)

# The engine reads its environment knobs once per process, so each setting
# runs in its own process.
add_test(NAME tilewright-env-forced COMMAND tilewright-tests "[env]")
set_tests_properties(
    tilewright-env-forced
    PROPERTIES ENVIRONMENT
               "TILEWRIGHT_FORCE_CELL=1;TILEWRIGHT_PICK_LOG=1;TILEWRIGHT_DIAG=1;TILEWRIGHT_CACHE_MB=7"
)
add_test(NAME tilewright-env-out-of-range COMMAND tilewright-tests "[env]")
set_tests_properties(
    tilewright-env-out-of-range
    PROPERTIES ENVIRONMENT "TILEWRIGHT_FORCE_CELL=96;TILEWRIGHT_PICK_LOG=0;TILEWRIGHT_CACHE_MB=0"
)
add_test(NAME tilewright-env-invalid COMMAND tilewright-tests "[env]")
set_tests_properties(
    tilewright-env-invalid
    PROPERTIES ENVIRONMENT "TILEWRIGHT_FORCE_CELL=1x;TILEWRIGHT_DIAG=;TILEWRIGHT_CACHE_MB=12x"
)
