# heuristics/sweep/CMakeLists.txt
#
# Standalone oracle sweep binary for the ck_dsl implicit-GEMM conv path.
#
# This CMakeLists is designed to be added via add_subdirectory() from a
# superbuild that has already defined ck_dsl_provider_private and the
# generated ckdsl_provider_paths.h (see build.sh for the wrapper).
#
# Targets defined here:
#   conv_oracle_sweep   -- data collection binary; run on a gfx942 device

cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(ConvOracleSweep LANGUAGES CXX)
    message(WARNING
        "heuristics/sweep configured as a standalone project. "
        "ck_dsl_provider_private will not be found unless you ran "
        "build.sh which sets up the superbuild wrapper.")
endif()

find_package(Threads REQUIRED)
if(NOT TARGET hip::host)
    find_package(hip REQUIRED)
endif()
if(NOT TARGET Python3::Python)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
endif()
if(NOT TARGET pybind11::embed)
    find_package(pybind11 REQUIRED)
endif()
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()

if(NOT DEFINED CK_DSL_PROVIDER_SOURCE_DIR)
    set(CK_DSL_PROVIDER_SOURCE_DIR
        "/rocm-libraries/dnn-providers/ck-dsl-provider/src"
        CACHE PATH "Path to ck-dsl-provider/src (private headers)")
endif()

if(NOT DEFINED CK_DSL_PROVIDER_BINARY_DIR)
    set(CK_DSL_PROVIDER_BINARY_DIR "${CMAKE_BINARY_DIR}/provider_build"
        CACHE PATH "Provider binary dir containing ckdsl_provider_paths.h")
endif()

add_executable(conv_candidate_sweep
    main.cpp
    ConvCandidateSweep.cpp
)

target_compile_features(conv_candidate_sweep PRIVATE cxx_std_17)

target_include_directories(conv_candidate_sweep
    PRIVATE
        ${CK_DSL_PROVIDER_SOURCE_DIR}   # CkDslContainer.hpp, engines/*, adapters/*
        ${CK_DSL_PROVIDER_BINARY_DIR}   # ckdsl_provider_paths.h (generated)
)

target_link_libraries(conv_candidate_sweep
    PRIVATE
        hip::host
        ck_dsl_provider_private
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        Python3::Python
        pybind11::embed
        Threads::Threads
)
