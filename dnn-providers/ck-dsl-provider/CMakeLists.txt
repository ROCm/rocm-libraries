# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# ck-dsl-provider: a hipDNN engine plugin that executes ck_dsl-generated kernels
# (precompiled HSACO + comgr-from-.ll JIT fallback) via the bundled, Python-free
# ck_dsl_runtime. Mirrors ck-fmha-provider's structure; backend = ck_dsl_runtime
# (HIP + libamd_comgr), no CK template dispatcher / ck_host.
cmake_minimum_required(VERSION 3.21)
project(ck_dsl_provider LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CK_DSL_PROVIDER_BUILD_TESTS "Build ck-dsl-provider tests" OFF)

# C-JIT path: compile the kernels at plan-build time from the pure-C ck_dsl
# engine (libckc_core.a + ckc/* headers), no shipped artifact, no Python.
# Enabled by default; the runtime behaviour is gated at runtime by env
# CK_DSL_C_JIT=1 (the ArtifactStore path remains the fallback). Set OFF to drop
# the libckc_core link dependency entirely.
option(CK_DSL_PROVIDER_C_JIT "Compile in the pure-C ck_dsl C-JIT path" ON)

include(cmake/Dependencies.cmake)

if(CK_DSL_PROVIDER_C_JIT)
    # ckc/* headers (ir.h, lower_llvm.h, instance_*.h) consumed by c_engine.hpp.
    set(CKC_INCLUDE_DIR "${CK_DSL_C_ROOT}/include" CACHE PATH "ckc headers dir")
    if(NOT EXISTS "${CKC_INCLUDE_DIR}/ckc/ir.h")
        # Default to the in-repo ck_dsl_c.
        set(CKC_INCLUDE_DIR
            "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/composablekernel/python/ck_dsl_c/include")
    endif()
    # libckc_core: the engine static archive linked --whole-archive below.
    #
    # We DO NOT silently fall back to a checked-in build directory. A stale
    # archive shadowing a fresh build was a recurring source of phantom
    # failures ("Conv C-JIT broken"). There are exactly two supported sources,
    # and the one chosen is always reported:
    #
    #   (1) explicit  -DCKC_LIB=/path/to/libckc_core.a
    #       Use a specific archive (e.g. one the canonical build.sh just built).
    #       The file MUST exist or configuration fails immediately.
    #
    #   (2) default (no -DCKC_LIB)
    #       Build the engine FRESH from source as a sub-build of this configure,
    #       so the linked archive is always in lockstep with the engine source.
    #       `CKC_LINK_TARGET` then resolves to the freshly-built archive path.
    set(CKC_ENGINE_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/composablekernel/python/ck_dsl_c"
        CACHE PATH "ck_dsl C engine source dir")
    if(DEFINED CKC_LIB)
        if(NOT EXISTS "${CKC_LIB}")
            message(FATAL_ERROR
                "CKC_LIB was set to '${CKC_LIB}' but no such file exists. "
                "Point -DCKC_LIB at a freshly-built libckc_core.a (see the "
                "canonical build script tools/build.sh in the engine tree), or "
                "omit -DCKC_LIB to build the engine fresh as part of this build.")
        endif()
        set(CKC_LINK_TARGET "${CKC_LIB}")
        message(STATUS "ck-dsl-provider C-JIT: using explicit CKC_LIB=${CKC_LIB}")
    else()
        if(NOT EXISTS "${CKC_ENGINE_SRC}/CMakeLists.txt")
            message(FATAL_ERROR
                "CK_DSL_PROVIDER_C_JIT=ON and no -DCKC_LIB given, but the engine "
                "source was not found at '${CKC_ENGINE_SRC}'. Set -DCKC_ENGINE_SRC "
                "to the ck_dsl_c source dir, pass -DCKC_LIB=/path/to/libckc_core.a, "
                "or set CK_DSL_PROVIDER_C_JIT=OFF.")
        endif()
        # Build the engine fresh as a fully isolated sub-build. ExternalProject
        # runs the engine's own cmake/compiler/language configuration verbatim
        # (the engine compiles its .c through the C++ frontend with custom flags
        # that must not be mixed with this project's flags) and yields a fresh
        # libckc_core.a that the provider then links. This guarantees the linked
        # archive is always in lockstep with the engine source, with zero chance
        # of a stale checked-in archive being used.
        include(ExternalProject)
        set(_CKC_EP_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/_ckc_engine")
        set(_CKC_EP_BUILD  "${_CKC_EP_PREFIX}/build")
        set(_CKC_FRESH_LIB "${_CKC_EP_BUILD}/libckc_core.a")
        if(NOT TARGET ckc_engine_ep)
            ExternalProject_Add(ckc_engine_ep
                SOURCE_DIR        "${CKC_ENGINE_SRC}"
                PREFIX            "${_CKC_EP_PREFIX}"
                BINARY_DIR        "${_CKC_EP_BUILD}"
                CMAKE_ARGS        -DCMAKE_BUILD_TYPE=Release
                BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR>
                                  --target ckc_core --config Release
                BUILD_ALWAYS      TRUE       # always rebuild -> never stale
                INSTALL_COMMAND   ""
                BUILD_BYPRODUCTS  "${_CKC_FRESH_LIB}")
        endif()
        set(CKC_LINK_TARGET "${_CKC_FRESH_LIB}")
        message(STATUS
            "ck-dsl-provider C-JIT: building engine fresh (ExternalProject) from "
            "${CKC_ENGINE_SRC} (no -DCKC_LIB given)")
    endif()
    find_library(AMD_COMGR_LIB NAMES amd_comgr HINTS /opt/rocm/lib /opt/rocm-*/lib)
    message(STATUS "ck-dsl-provider C-JIT: comgr=${AMD_COMGR_LIB} inc=${CKC_INCLUDE_DIR}")
endif()

# LightGBM for the trained-model heuristic (mirrors CK Tile's ML heuristic).
# Optional: when not found the plugin still builds (FirstFit only) -- but the
# LGBM_* symbols in ml_heuristic.hpp must resolve, so we link it when present.
find_library(LIGHTGBM_LIB NAMES _lightgbm lightgbm lib_lightgbm
    HINTS /workspace/.local/lib/python3.12/site-packages/lightgbm/lib
          ${LIGHTGBM_ROOT}/lib /opt/rocm/lib)
if(NOT LIGHTGBM_LIB)
    # Fall back to the explicit path if the NAMES probe misses the lib_ prefix.
    find_file(LIGHTGBM_LIB lib_lightgbm.so
        HINTS /workspace/.local/lib/python3.12/site-packages/lightgbm/lib)
endif()
message(STATUS "ck-dsl-provider LightGBM: ${LIGHTGBM_LIB}")

# Bundled runtime library (header-only INTERFACE target ck_dsl_runtime).
set(CK_DSL_RUNTIME_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(runtime)

add_library(ck_dsl_plugin_impl OBJECT
    src/CkDslPluginPublic.cpp
    src/CkDslContainer.cpp
    src/CkDslHandle.cpp
    src/engines/CkDslGemmEngine.cpp
    src/engines/CkDslParamParser.cpp
    src/engines/plans/CkDslGemmPlanBuilder.cpp
    src/engines/plans/CkDslGemmPlan.cpp
    src/engines/CkDslAttentionEngine.cpp
    src/engines/CkDslAttnParamParser.cpp
    src/engines/plans/CkDslAttnPlanBuilder.cpp
    src/engines/plans/CkDslAttnPlan.cpp
    src/engines/CkDslConvEngine.cpp
    src/engines/CkDslConvParamParser.cpp
    src/engines/plans/CkDslConvPlanBuilder.cpp
    src/engines/plans/CkDslConvPlan.cpp)

target_include_directories(ck_dsl_plugin_impl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ck_dsl_plugin_impl
    PUBLIC hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk
    PRIVATE ck_dsl_runtime hip::host)
if(CK_DSL_PROVIDER_C_JIT)
    target_compile_definitions(ck_dsl_plugin_impl PRIVATE CK_DSL_PROVIDER_C_JIT=1)
    target_include_directories(ck_dsl_plugin_impl PRIVATE ${CKC_INCLUDE_DIR})
endif()
set_target_properties(ck_dsl_plugin_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ck_dsl_plugin SHARED $<TARGET_OBJECTS:ck_dsl_plugin_impl>)
target_link_libraries(ck_dsl_plugin PRIVATE hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk ck_dsl_runtime hip::host)
if(CK_DSL_PROVIDER_C_JIT)
    # Whole-archive so the comgr-from-.ll lowering TUs are retained. The link
    # target is either an explicit -DCKC_LIB path or the freshly-built engine
    # archive ($<TARGET_FILE:ckc_core>); never a stale checked-in archive.
    target_link_libraries(ck_dsl_plugin PRIVATE
        -Wl,--whole-archive ${CKC_LINK_TARGET} -Wl,--no-whole-archive)
    # When the engine is built fresh (ExternalProject), build it before linking.
    if(TARGET ckc_engine_ep)
        add_dependencies(ck_dsl_plugin ckc_engine_ep)
    endif()
    if(AMD_COMGR_LIB)
        target_link_libraries(ck_dsl_plugin PRIVATE ${AMD_COMGR_LIB})
    endif()
endif()
if(LIGHTGBM_LIB)
    target_link_libraries(ck_dsl_plugin PRIVATE ${LIGHTGBM_LIB})
endif()
set_target_properties(ck_dsl_plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    OUTPUT_NAME "ck_dsl_provider_plugin")

# Host-only unit tests for parser pure functions -- no GPU, no hipDNN frontend.
if(CK_DSL_PROVIDER_BUILD_TESTS)
    add_executable(ck_dsl_gemm_blayout_test
        integration_tests/GemmBLayoutTest.cpp
        src/engines/CkDslParamParser.cpp)
    target_include_directories(ck_dsl_gemm_blayout_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(ck_dsl_gemm_blayout_test
        PRIVATE hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk ck_dsl_runtime hip::host)

    add_executable(ck_dsl_attn_layout_test
        integration_tests/AttnLayoutTest.cpp
        src/engines/CkDslAttnParamParser.cpp)
    target_include_directories(ck_dsl_attn_layout_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(ck_dsl_attn_layout_test
        PRIVATE hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk ck_dsl_runtime hip::host)

    # Host-only fuzz/robustness test for all three param parsers. Feeds
    # randomized + edge-case hipDNN graph flatbuffers through the parsers and
    # asserts no crash / OOB (clean parse or std::exception only). No GPU.
    add_executable(ck_dsl_param_fuzz_test
        integration_tests/ParamParserFuzzTest.cpp
        src/engines/CkDslParamParser.cpp
        src/engines/CkDslAttnParamParser.cpp
        src/engines/CkDslConvParamParser.cpp)
    target_include_directories(ck_dsl_param_fuzz_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(ck_dsl_param_fuzz_test
        PRIVATE hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk ck_dsl_runtime hip::host)

    enable_testing()
    add_test(NAME ck_dsl_gemm_blayout_test COMMAND ck_dsl_gemm_blayout_test)
    add_test(NAME ck_dsl_attn_layout_test COMMAND ck_dsl_attn_layout_test)
    add_test(NAME ck_dsl_param_fuzz_test COMMAND ck_dsl_param_fuzz_test)
endif()

if(DEFINED HIPDNN_BUILD_PLUGIN_ENGINE_DIR)
    set_target_properties(ck_dsl_plugin PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})
endif()

if(DEFINED HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR)
    install(TARGETS ck_dsl_plugin
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR})
endif()
