# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shared core for tensilelite-client and cpu-gemm-driver.
#
# Both executables consume nearly the same sources, link line, include
# directories, compile definitions, C++ standard, and sanitizer flags. Owning
# them on a single STATIC helper target removes a `foreach(target ...)` loop
# that applied identical properties to two targets, a `COMMON_CLIENT_SOURCES`
# variable that hid the source list from `target_sources` (and from IDE
# navigation), and three duplicated `target_include_directories` calls.
#
# Each executable is now a thin wrapper that adds only its `main` translation
# unit and links to this helper.
add_library(tensilelite-client-common STATIC)
add_library(tensilelite::client-common ALIAS tensilelite-client-common)

if(NOT WIN32)
    find_package(amd_smi REQUIRED)
else()
    find_package(amd_smi)
endif()

# The GPU-initiated SDMA path for fused A2A (SdmaQueue.cpp) needs hsakmt (KFD
# thunk) + hsa-runtime64. It is OFF by default so the ordinary client build
# carries no hsakmt/hsa-runtime64/libnuma dependency: environments that never
# built these can still configure and build the client unchanged. Turn it on
# with -DTENSILELITE_ENABLE_SDMA_A2A=ON.
option(TENSILELITE_ENABLE_SDMA_A2A "Build GPU-initiated SDMA path for fused A2A" OFF)

if(TENSILELITE_ENABLE_SDMA_A2A AND NOT WIN32)
    # ROCm 7.x's hsakmt cmake config declares a link dependency on the imported
    # target numa::numa but never provides it, so find_package(hsakmt) fails at
    # generate time ("target numa::numa not found") unless we define it first.
    # ROCm 7.x also vendors libnuma into ${ROCM_PATH}/lib/rocm_sysdeps/lib
    # (renamed librocm_sysdeps_numa.so.1 with a libnuma.so symlink), which is
    # NOT on the default library search path, so a bare find_library misses it.
    # Search that directory explicitly in addition to the system paths, so both
    # a vendored ROCm libnuma and a system-installed one are found.
    if(NOT TARGET numa::numa)
        if(NOT DEFINED ROCM_PATH)
            if(DEFINED ENV{ROCM_PATH})
                set(ROCM_PATH "$ENV{ROCM_PATH}")
            else()
                set(ROCM_PATH "/opt/rocm")
            endif()
        endif()
        find_library(NUMA_LIBRARY
            NAMES numa
            HINTS "${ROCM_PATH}/lib/rocm_sysdeps/lib"
            PATHS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
        find_path(NUMA_INCLUDE_DIR
            NAMES numa.h
            HINTS "${ROCM_PATH}/include/rocm_sysdeps/include"
            PATHS "${ROCM_PATH}/include/rocm_sysdeps/include")
        if(NUMA_LIBRARY)
            add_library(numa::numa UNKNOWN IMPORTED)
            set_target_properties(numa::numa PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARY}")
            if(NUMA_INCLUDE_DIR)
                set_target_properties(numa::numa PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
            endif()
        else()
            message(FATAL_ERROR
                "libnuma not found but required by hsakmt::hsakmt for "
                "TENSILELITE_ENABLE_SDMA_A2A. Searched the ROCm-vendored path "
                "'${ROCM_PATH}/lib/rocm_sysdeps/lib' and the default system "
                "library paths. Set ROCM_PATH if your ROCm is elsewhere, or "
                "point NUMA_LIBRARY at the libnuma.so to use.")
        endif()
    endif()
    find_package(hsa-runtime64 REQUIRED)
    find_package(hsakmt REQUIRED)

    # ROCm 7.x's hsakmtTargets.cmake bakes BUILD-HOST absolute paths into
    # hsakmt::hsakmt's INTERFACE_LINK_LIBRARIES that do not exist on a fresh
    # container, so every executable linking hsakmt fails. Two forms appear:
    #   * an absolute /usr/lib64/libc.so entry -> make reports "No rule to make
    #     target '/usr/lib64/libc.so'" (libc is linked implicitly anyway), and
    #   * a `-L/home/runner/.../rocm_sysdeps/lib` search dir feeding `-ldrm` /
    #     `-ldrm_amdgpu` -> ld.lld "unable to find library -ldrm".
    # The libdrm/numa the interface wants are actually vendored in
    # ${ROCM_PATH}/lib/rocm_sysdeps/lib. Drop the phantom libc entry and repoint
    # every dead `-L` search dir at the real vendored sysdeps lib dir.
    if(TARGET hsakmt::hsakmt)
        set(_sysdeps_lib "${ROCM_PATH}/lib/rocm_sysdeps/lib")
        get_target_property(_hsakmt_ill hsakmt::hsakmt INTERFACE_LINK_LIBRARIES)
        if(_hsakmt_ill)
            set(_hsakmt_ill_clean "")
            foreach(_lib IN LISTS _hsakmt_ill)
                if(_lib MATCHES "/libc\\.so$" AND NOT EXISTS "${_lib}")
                    message(STATUS "hsakmt: dropping nonexistent libc path '${_lib}' "
                                   "from INTERFACE_LINK_LIBRARIES (libc is linked implicitly)")
                elseif(_lib MATCHES "^-L(.+)$" AND NOT IS_DIRECTORY "${CMAKE_MATCH_1}")
                    message(STATUS "hsakmt: repointing dead search dir '${CMAKE_MATCH_1}' "
                                   "to vendored '${_sysdeps_lib}'")
                    list(APPEND _hsakmt_ill_clean "-L${_sysdeps_lib}")
                else()
                    list(APPEND _hsakmt_ill_clean "${_lib}")
                endif()
            endforeach()
            set_target_properties(hsakmt::hsakmt PROPERTIES
                INTERFACE_LINK_LIBRARIES "${_hsakmt_ill_clean}")
        endif()
    endif()
endif()

target_link_libraries(tensilelite-client-common
    PUBLIC
        hip::device
        hip::host
        roc::tensilelite-host
        rocisa::rocisa-cpp
        OpenMP::OpenMP_CXX
)
if(amd_smi_FOUND)
    target_link_libraries(tensilelite-client-common PUBLIC amd_smi)
endif()
# PRIVATE: only SdmaQueue.cpp (compiled into this target) calls hsakmt/hsa;
# downstream consumers use our own wrapper, so the dependency must not leak.
if(TENSILELITE_ENABLE_SDMA_A2A AND NOT WIN32)
    target_link_libraries(tensilelite-client-common
        PRIVATE hsakmt::hsakmt hsa-runtime64::hsa-runtime64)
endif()

# When mxDataGenerator is enabled, hipblaslt::mxdatagen propagates the
# HIPBLASLT_ENABLE_MXDATAGENERATOR macro, the C++20 floor, the include path
# for `<mxDataGen.hpp>`, hipblaslt::headers (so `<hipblaslt/...>` API headers
# resolve), and the link to roc::mxDataGenerator. PUBLIC so the executables
# below pick the macro up too.
if(HIPBLASLT_ENABLE_MXDATAGENERATOR)
    target_link_libraries(tensilelite-client-common PUBLIC hipblaslt::mxdatagen)
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

add_executable(cpu-gemm-driver "${CMAKE_CURRENT_SOURCE_DIR}/cpu_gemm_driver.cpp")
target_link_libraries(cpu-gemm-driver PRIVATE tensilelite-client-common)

# rocprofiler-sdk is only used by tensilelite-client (in-process counters).
if(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)
    target_link_libraries(tensilelite-client PRIVATE rocprofiler-sdk)
endif()

add_subdirectory(include)
add_subdirectory(src)
