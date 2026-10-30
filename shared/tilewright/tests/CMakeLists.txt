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

# The real-model tests ([real] and tilewright-levers-parity) load every shipped
# model under this directory, hipBLASLt's library logic, whose
# <arch>/<subtree>/Tilewright directories hold them, and are skipped when it
# holds none.
cmake_path(
    SET _tw_default_weights NORMALIZE
    "${tilewright_SOURCE_DIR}/../../projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
)
set(TILEWRIGHT_TEST_WEIGHTS_DIR
    "${_tw_default_weights}"
    CACHE PATH "Directory the real-model tests search for shipped tilewright models"
)

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
    PRIVATE TILEWRIGHT_TEST_WEIGHTS_DIR="${TILEWRIGHT_TEST_WEIGHTS_DIR}"
            TILEWRIGHT_TEST_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/data/sample_kernels.tsv"
)

catch_discover_tests(tilewright-tests TEST_SPEC "~[env]" DISCOVERY_MODE PRE_TEST)

get_target_property(_tw_engine_sources tilewright SOURCES)
# The engine again as library `name`, with the selection-time options set
# whatever the CMake options say: TW_LEVER_PREFETCH to `prefetch` and
# TW_LEVER_MEMO to `memo` (1 or 0), and the memo's plan check
# TW_LEVER_MEMO_CHECK to a fourth argument when one is given (else its
# default, on where assertions are). The engine's other definitions are taken
# as they are.
function(_tw_engine name prefetch memo)
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
    set(_tw_levers TW_LEVER_PREFETCH=${prefetch} TW_LEVER_MEMO=${memo})
    if(ARGC GREATER 3)
        list(APPEND _tw_levers TW_LEVER_MEMO_CHECK=${ARGV3})
    endif()
    target_compile_definitions(
        ${name}
        PUBLIC ${_tw_levers}
        PRIVATE $<FILTER:$<TARGET_PROPERTY:tilewright,COMPILE_DEFINITIONS>,EXCLUDE,^TW_LEVER_>
    )
endfunction()

# The whole suite, plus the sources given after `name`, against the engine
# library `name` in one process (ctest entry `name`), so that the thread's
# memo also outlives the rankings of one test.
function(_tw_suite name)
    add_executable(${name}-tests)
    target_sources(${name}-tests PRIVATE ${_tw_test_sources} ${ARGN})
    target_include_directories(${name}-tests PRIVATE "${tilewright_SOURCE_DIR}/src/tilewright")
    target_link_libraries(${name}-tests PRIVATE ${name} Catch2::Catch2WithMain Threads::Threads)
    target_compile_definitions(
        ${name}-tests
        PRIVATE TILEWRIGHT_TEST_WEIGHTS_DIR="${TILEWRIGHT_TEST_WEIGHTS_DIR}"
                TILEWRIGHT_TEST_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/data/sample_kernels.tsv"
    )
    add_test(NAME ${name} COMMAND ${name}-tests "~[env]")
endfunction()

# tilewright-tests tests the engine as the CMake options build it, with both
# options by default. Whatever they say, the suite also runs against the
# engine with both options and the plan check (the default where assertions
# are) and with both and without the check (as a release build compiles the
# memo), each with test_levers.cpp, and with neither option.
_tw_engine(tilewright-levers 1 1 1)
_tw_suite(tilewright-levers "${CMAKE_CURRENT_SOURCE_DIR}/test_levers.cpp")
_tw_engine(tilewright-levers-nocheck 1 1 0)
_tw_suite(tilewright-levers-nocheck "${CMAKE_CURRENT_SOURCE_DIR}/test_levers.cpp")
_tw_engine(tilewright-levers-off 0 0)
_tw_suite(tilewright-levers-off)

# rerank_check and memo_check of training_pipeline/scripts/experiments, which
# hash TensileLite's re-rank and pairs of rankings the memo may or must not
# answer, against the engine with neither option (the reference), each option
# alone, and both without and with the plan check. The ctest entry
# tilewright-levers-parity (levers_parity.cmake) runs them on every shipped
# model with the sample kernel pool and requires every build to print the
# reference's hashes, and the memo to answer rankings where it is built in.
set(_tw_experiments "${tilewright_SOURCE_DIR}/training_pipeline/scripts/experiments")
if(EXISTS "${_tw_experiments}/rerank_check.cpp" AND EXISTS "${_tw_experiments}/memo_check.cpp")
    _tw_engine(tilewright-levers-prefetch 1 0)
    _tw_engine(tilewright-levers-memo 0 1)
    set(_tw_parity_args)
    foreach(_tw_build off prefetch memo nocheck check)
        set(_tw_lib tilewright-levers-${_tw_build})
        if(_tw_build STREQUAL "check")
            set(_tw_lib tilewright-levers)
        endif()
        foreach(_tw_check rerank_check memo_check)
            set(_tw_exe tilewright-parity-${_tw_build}-${_tw_check})
            add_executable(${_tw_exe} "${_tw_experiments}/${_tw_check}.cpp")
            # The checks count the memo's answers with the engine's private
            # counters (model_impl.hpp).
            target_include_directories(${_tw_exe} PRIVATE "${tilewright_SOURCE_DIR}/src/tilewright")
            target_link_libraries(${_tw_exe} PRIVATE ${_tw_lib})
            list(APPEND _tw_parity_args "-D${_tw_check}_${_tw_build}=$<TARGET_FILE:${_tw_exe}>")
        endforeach()
    endforeach()
    # Copies of the shipped models with the alignment veto, which the entry
    # ranks as well.
    add_executable(tilewright-veto-models "${CMAKE_CURRENT_SOURCE_DIR}/veto_models.cpp"
                                          "${CMAKE_CURRENT_SOURCE_DIR}/model_writer.cpp")
    target_include_directories(tilewright-veto-models PRIVATE "${tilewright_SOURCE_DIR}/src/tilewright")
    target_link_libraries(tilewright-veto-models PRIVATE roc::tilewright)
    # The models served with the drift fallback as well: copies of every shipped
    # model that takes it, but the TF32 TT one.
    add_executable(
        tilewright-drift-flag "${CMAKE_CURRENT_SOURCE_DIR}/drift_flag.cpp"
                              "${CMAKE_CURRENT_SOURCE_DIR}/model_writer.cpp"
    )
    target_include_directories(tilewright-drift-flag PRIVATE "${tilewright_SOURCE_DIR}/src/tilewright")
    target_link_libraries(tilewright-drift-flag PRIVATE roc::tilewright Threads::Threads)
    add_test(
        NAME tilewright-levers-parity
        COMMAND
            "${CMAKE_COMMAND}" "-DWEIGHTS=${TILEWRIGHT_TEST_WEIGHTS_DIR}"
            "-DKERNELS=${CMAKE_CURRENT_SOURCE_DIR}/data/sample_kernels.tsv"
            "-DSHAPES=${_tw_experiments}/bench_default_shapes.txt"
            "-DOUT=${CMAKE_CURRENT_BINARY_DIR}/levers-parity"
            "-DBUILDS=off,prefetch,memo,nocheck,check" "-DMEMO_BUILDS=memo,nocheck,check"
            "-DVETO_MODELS=$<TARGET_FILE:tilewright-veto-models>"
            "-DDRIFT_FLAG=$<TARGET_FILE:tilewright-drift-flag>"
            "-DDRIFT_SKIP=gfx950_Cijk_Alik_Bjlk_S_MX_B_Bias_HAS_SAV_UserArgs"
            ${_tw_parity_args} -P "${CMAKE_CURRENT_SOURCE_DIR}/levers_parity.cmake"
    )
    set_tests_properties(
        tilewright-levers-parity PROPERTIES SKIP_REGULAR_EXPRESSION "no materialized shipped models"
    )
endif()

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
