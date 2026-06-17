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
    # libckc_core.a: prefer an explicit -DCKC_LIB=..., else a hint file written
    # by the ck_dsl_c build, else a probe of common build dirs.
    if(NOT DEFINED CKC_LIB)
        if(EXISTS "/tmp/ckc_libdir.txt")
            file(READ "/tmp/ckc_libdir.txt" _ckc_libdir)
            string(STRIP "${_ckc_libdir}" _ckc_libdir)
            find_library(CKC_LIB NAMES ckc_core HINTS "${_ckc_libdir}")
        endif()
    endif()
    if(NOT CKC_LIB)
        find_library(CKC_LIB NAMES ckc_core
            HINTS ${CKC_LIB_DIR}
                  "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/composablekernel/python/ck_dsl_c/build")
    endif()
    if(NOT CKC_LIB)
        message(FATAL_ERROR
            "CK_DSL_PROVIDER_C_JIT=ON but libckc_core not found. "
            "Pass -DCKC_LIB=/path/to/libckc_core.a (build via "
            "cmake .../ck_dsl_c) or set CK_DSL_PROVIDER_C_JIT=OFF.")
    endif()
    find_library(AMD_COMGR_LIB NAMES amd_comgr HINTS /opt/rocm/lib /opt/rocm-*/lib)
    message(STATUS "ck-dsl-provider C-JIT: lib=${CKC_LIB} comgr=${AMD_COMGR_LIB} inc=${CKC_INCLUDE_DIR}")
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
    # Whole-archive so the comgr-from-.ll lowering TUs are retained.
    target_link_libraries(ck_dsl_plugin PRIVATE
        -Wl,--whole-archive ${CKC_LIB} -Wl,--no-whole-archive)
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
    enable_testing()
    add_test(NAME ck_dsl_gemm_blayout_test COMMAND ck_dsl_gemm_blayout_test)
    add_test(NAME ck_dsl_attn_layout_test COMMAND ck_dsl_attn_layout_test)
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
