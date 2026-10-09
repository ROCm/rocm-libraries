# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shared core for the TensileLite client and its tests.
add_library(tensilelite-client-common STATIC)
add_library(tensilelite::client-common ALIAS tensilelite-client-common)

if(NOT WIN32)
    find_package(amd_smi REQUIRED)
else()
    find_package(amd_smi)
endif()

# The GPU-initiated SDMA path for fused A2A (SdmaQueue.hpp) needs hsakmt (KFD
# thunk) + hsa-runtime64. It is OFF by default so the ordinary client build
# carries no hsakmt/hsa-runtime64/libnuma dependency: environments that never
# built these can still configure and build the client unchanged. Turn it on
# with -DTENSILELITE_ENABLE_SDMA=ON.
option(TENSILELITE_ENABLE_SDMA "Build the GPU-initiated SDMA transport path" OFF)

if(TENSILELITE_ENABLE_SDMA AND NOT WIN32)
    # hsakmt::hsakmt's exported link interface names numa::numa, so libnuma has
    # to be found as an imported target before that export is loaded. The
    # project's cmake/FindNUMA.cmake does that and documents when it can be
    # dropped.
    find_package(NUMA REQUIRED)

    find_package(hsa-runtime64 REQUIRED)
    find_package(hsakmt REQUIRED)

    # ROCm 7.x's hsakmtTargets.cmake bakes BUILD-HOST absolute paths into
    # hsakmt::hsakmt's INTERFACE_LINK_LIBRARIES that do not exist on a fresh
    # container, so every executable linking hsakmt fails. The repair lives in
    # cmake/HsakmtLinkInterface.cmake, which documents the two broken forms.
    include(HsakmtLinkInterface)
    hipblaslt_sanitize_hsakmt_link_interface()
endif()

target_link_libraries(tensilelite-client-common
    PUBLIC
        hip::device
        hip::host
        roc::hostnumerics
        roc::hostnumerics-amd-gpu-layout
        roc::tensilelite-host
        rocisa::rocisa-cpp
        OpenMP::OpenMP_CXX
)
if(amd_smi_FOUND)
    target_link_libraries(tensilelite-client-common PUBLIC amd_smi)
endif()
# PRIVATE: hsakmt/hsa are needed only by TUs in this target that include the
# header-only SdmaQueue.hpp -- today just FusedA2AClient.cpp, itself gated on
# the macro below. The two executables compile one source each and never
# include it, so they get neither the headers nor the libs.
if(TENSILELITE_ENABLE_SDMA AND NOT WIN32)
    target_link_libraries(tensilelite-client-common
        PRIVATE hsakmt::hsakmt hsa-runtime64::hsa-runtime64)
    # FusedA2AClient.cpp gates its SdmaQueueSet setup on this macro; without it
    # the fused run fails fast instead of handing the kernel a null queue array.
    target_compile_definitions(tensilelite-client-common
        PRIVATE TENSILELITE_ENABLE_SDMA)
endif()

target_include_directories(tensilelite-client-common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(tensilelite-client-common
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(tensilelite-client-common PUBLIC)
endif()

# Enable coverage instrumentation for client code when TENSILELITE_ENABLE_COVERAGE is ON
if(TENSILELITE_ENABLE_COVERAGE)
    message(STATUS "Enabling C++ code coverage instrumentation for tensilelite-client-common")
    target_compile_options(tensilelite-client-common PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(tensilelite-client-common PUBLIC -fprofile-instr-generate -fcoverage-mapping)
endif()

# f16c gives the validator hot loop hardware Half<->float conversion, which is
# what lets the auto-vectorizer turn the per-element comparison into AVX2 code.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mf16c HIPBLASLT_CXX_HAS_MF16C)
if(HIPBLASLT_CXX_HAS_MF16C)
    target_compile_options(tensilelite-client-common PRIVATE -mf16c)
endif()

add_library(tensilelite-hostnumerics-adapter STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hostnumerics/DataInitialization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hostnumerics/GemmInvocationAdapter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hostnumerics/Reference.cpp"
)
add_library(tensilelite::hostnumerics-adapter ALIAS tensilelite-hostnumerics-adapter)

target_sources(tensilelite-hostnumerics-adapter
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TensileLite/Client/hostnumerics/DataInitializationHelpers.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TensileLite/Client/hostnumerics/GemmInvocationAdapter.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TensileLite/Client/hostnumerics/HostNumericsBridge.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TensileLite/Client/hostnumerics/Reference.hpp"
)

target_include_directories(tensilelite-hostnumerics-adapter
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(tensilelite-hostnumerics-adapter
    PUBLIC
        roc::hostnumerics
    PRIVATE
        hip::device
        hip::host
        roc::tensilelite-host
        rocisa::rocisa-cpp
        OpenMP::OpenMP_CXX
)
set_target_properties(tensilelite-hostnumerics-adapter
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)
if(HIPBLASLT_CXX_HAS_MF16C)
    target_compile_options(tensilelite-hostnumerics-adapter PRIVATE -mf16c)
endif()
if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(tensilelite-hostnumerics-adapter PUBLIC)
endif()
if(TENSILELITE_ENABLE_COVERAGE)
    target_compile_options(
        tensilelite-hostnumerics-adapter PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(
        tensilelite-hostnumerics-adapter PUBLIC -fprofile-instr-generate -fcoverage-mapping)
endif()

target_link_libraries(tensilelite-client-common
    PUBLIC
        tensilelite::hostnumerics-adapter
)

option(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK "Enable in-process ROCprofiler-SDK counters" OFF)
target_compile_definitions(tensilelite-client-common
    PUBLIC
        TENSILELITE_CLIENT_ENABLE_ROCPROFSDK=$<BOOL:${TENSILELITE_CLIENT_ENABLE_ROCPROFSDK}>
)
if(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)
    find_package(rocprofiler-sdk)
    if(NOT rocprofiler-sdk_FOUND)
        message(FATAL_ERROR "ROCprofiler-SDK not found.")
    endif()
endif()

add_executable(tensilelite-client "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(tensilelite-client PRIVATE tensilelite-client-common)

# rocprofiler-sdk is only used by tensilelite-client (in-process counters).
if(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)
    target_link_libraries(tensilelite-client PRIVATE rocprofiler-sdk)
endif()

add_subdirectory(include)
add_subdirectory(src)
