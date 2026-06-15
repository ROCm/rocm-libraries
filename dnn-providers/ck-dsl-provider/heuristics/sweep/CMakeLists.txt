# heuristics/sweep/CMakeLists.txt
#
# Standalone candidate sweep for the ck_dsl implicit-GEMM conv path.
#
# Dependencies:
#   - hip::host       (HIP runtime)
#   - ckc_core        (pure-C ck_dsl engine; provides CEngine::build_conv)
#   - amd_comgr       (LLVM IR -> HSACO compilation; used by Kernel::ensure_compiled)
#   - ck_dsl_runtime  (header-only: c_engine.hpp, kernel.hpp, manifest.hpp, comgr.hpp)
#
# Targets defined here:
#   conv_candidate_sweep  -- data collection binary; run on a gfx942/gfx950 device

cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(ConvCandidateSweep LANGUAGES CXX)
endif()

find_package(Threads REQUIRED)
if(NOT TARGET hip::host)
    find_package(hip REQUIRED)
endif()

if(NOT DEFINED CK_DSL_RUNTIME_INCLUDE_DIR)
    set(CK_DSL_RUNTIME_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/include"
        CACHE PATH "Path to ck_dsl_runtime headers (runtime/include)")
endif()

if(NOT DEFINED CKC_INCLUDE_DIR)
    set(CKC_INCLUDE_DIR
        "/rocm-libraries/projects/composablekernel/python/ck_dsl_c/include"
        CACHE PATH "Path to ckc/ headers (instance_conv_implicit_gemm.h etc.)")
endif()

if(NOT DEFINED CKC_CORE_LIB)
    set(CKC_CORE_LIB "" CACHE FILEPATH "Path to libckc_core.a")
endif()

add_executable(conv_candidate_sweep
    main.cpp
    ConvCandidateSweep.cpp
)

target_compile_features(conv_candidate_sweep PRIVATE cxx_std_17)

target_include_directories(conv_candidate_sweep
    PRIVATE
        ${CK_DSL_RUNTIME_INCLUDE_DIR}   # ck_dsl_runtime/*.hpp
        ${CKC_INCLUDE_DIR}              # ckc/ir.h, ckc/instance_conv_implicit_gemm.h, etc.
)

target_link_libraries(conv_candidate_sweep
    PRIVATE
        hip::host
        Threads::Threads
        amd_comgr
)

if(CKC_CORE_LIB)
    target_link_libraries(conv_candidate_sweep PRIVATE ${CKC_CORE_LIB})
else()
    find_library(CKC_CORE_FOUND ckc_core HINTS /opt/rocm/lib)
    if(CKC_CORE_FOUND)
        target_link_libraries(conv_candidate_sweep PRIVATE ${CKC_CORE_FOUND})
    else()
        message(FATAL_ERROR
            "libckc_core not found. Set -DCKC_CORE_LIB=/path/to/libckc_core.a")
    endif()
endif()
