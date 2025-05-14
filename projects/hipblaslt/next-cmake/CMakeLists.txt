# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Need a list of old compilation flags, definitions and link flags
# to confirm if they are required and document what we dropped if not.

cmake_minimum_required(VERSION 3.25.0)
project(hipblaslt VERSION 0.15.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

include(Dependencies)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(CMakeDependentOption)
include(hipblaslt_python)
include(hipblaslt_target_configure_sanitizers)

rocm_setup_version(VERSION ${PROJECT_VERSION})
set(hipblaslt_SOVERSION 0.15)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
      set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
    else()
      set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
    endif()
endif()

set(HIPBLASLT_ENABLE_DEVICE ON CACHE BOOL "Build hipBLASLt device libraries.")
set(HIPBLASLT_ENABLE_CLIENT ON CACHE BOOL "Build hipBLASLt client apps.")
cmake_dependent_option(HIPBLASLT_ENABLE_HOST "Build hipBLASLt host library." ON "HIPBLASLT_ENABLE_CLIENT" OFF)
set(HIPBLASLT_ENABLE_COVERAGE OFF CACHE BOOL "Build gcov support")

if(HIPBLASLT_ENABLE_CLIENT)
    set(HIPBLASLT_BUILD_TESTING ON CACHE BOOL "Build hipblaslt client tests.")
    set(HIPBLASLT_ENABLE_SAMPLES ON CACHE BOOL "Build client samples.")
    # rocm-smi is not presently available on Windows so we do not require it.
    cmake_dependent_option(HIPBLASLT_REQUIRE_ROCM_SMI "Require rocm_smi." ON "NOT WIN32" OFF)
endif()

set(HIPBLASLT_ENABLE_ASAN OFF CACHE BOOL "Build hipBLASLt with address sanitizer enabled")

if(HIPBLASLT_ENABLE_HOST)
    set(HIPBLASLT_BUILD_SHARED_LIBS ON CACHE BOOL "Build the hipblaslt library as shared vs static")
    set(HIPBLASLT_ENABLE_HIP ON CACHE BOOL "Use the HIP runtime.")
    set(HIPBLASLT_ENABLE_MSGPACK ON CACHE BOOL "Use msgpack for parsing configuration files.")
    set(HIPBLASLT_ENABLE_OPENMP ON CACHE BOOL "Use OpenMP to improve performance.")
    set(HIPBLASLT_ENABLE_LLVM OFF CACHE BOOL "Use msgpack for parsing configuration files.")
    set(HIPBLASLT_ENABLE_ROCROLLER OFF CACHE BOOL "Use RocRoller library.")
    set(HIPBLASLT_ENABLE_BLIS ON CACHE BOOL "Enable BLIS support.") # I don't know that we can build with this OFF
    set(HIPBLASLT_ENABLE_LAZY_LOAD ON CACHE BOOL "Enable lazy loading of runtime code oject files to reduce ram usage.")
    cmake_dependent_option(HIPBLASLT_ENABLE_MARKER "Use the marker library." ON "NOT WIN32" OFF)
    set(HIPBLASLT_ENABLE_HIPBLAS_DIRECT OFF CACHE BOOL "Use the hipblas header directly.")
endif()

if(HIPBLASLT_ENABLE_DEVICE)
    set(HIPBLASLT_BUNDLE_PYTHON_DEPS ON CACHE BOOL "Build python dependencies requied for device code generation.")
    set(HIPBLASLT_DEVICE_JOBS "" CACHE STRING "Number of CPU threads to use for building device libraries (will use nproc if unset).")
    set(HIPBLASLT_KEEP_TMP OFF CACHE STRING "Keep temporary build directory for device libraries (turning this ON  bloat the build size).")
    set(HIPBLASLT_LIBLOGIC_PATH "" CACHE STRING "Path to library logic files (will use 'library' if unset).")
    set(device_library_dir "${CMAKE_BINARY_DIR}/Tensile/library")
    file(MAKE_DIRECTORY "${device_library_dir}")
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE CACHE BOOL "Skip build RPATH")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE CACHE BOOL "Build with install RPATH")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib" CACHE STRING "Install RPATH")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Use link path for RPATH")

mark_as_advanced(CMAKE_SKIP_BUILD_RPATH)
mark_as_advanced(CMAKE_BUILD_WITH_INSTALL_RPATH)
mark_as_advanced(CMAKE_INSTALL_RPATH)
mark_as_advanced(CMAKE_INSTALL_RPATH_USE_LINK_PATH)

set(TEMP_TENSILE_HOST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite/Tensile/Source/lib")
set(HIPBLASLT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../library")
set(HIPBLASLT_CLIENTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../clients")
set(ROCBLASLT_LIB_DIR "${HIPBLASLT_LIB_DIR}/src/amd_detail/rocblaslt")

find_package(hip REQUIRED)
if(HIPBLASLT_ENABLE_HOST)
    if(HIPBLASLT_ENABLE_HIPBLAS_DIRECT)
        set(hipblas_target hipblas)
    else()
        set(hipblas_target hipblas-common)
    endif()
    if(NOT TARGET ${hipblas_target})
        find_package(${hipblas_target} REQUIRED)
    endif()
    if(HIPBLASLT_ENABLE_BLIS)
        find_package(BLIS REQUIRED)
    endif()
    if(HIPBLASLT_ENABLE_MARKER)
        find_library(rocTracer roctx64 REQUIRED)
    endif()
endif()
if(HIPBLASLT_REQUIRE_ROCM_SMI)
    find_package(rocm_smi REQUIRED)
else()
    find_package(rocm_smi)
endif()
if(HIPBLASLT_ENABLE_CLIENT)
    find_package(BLAS REQUIRED)
    find_package(cblas REQUIRED)
    find_package(LAPACK REQUIRED)
endif()
if(HIPBLASLT_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()
if(HIPBLASLT_ENABLE_MSGPACK)
    # See: https://github.com/msgpack/msgpack-c/wiki/Q%26A#how-to-support-both-msgpack-c-c-version-5x-and-6x-
    # Prefer 6.x (msgpack-cxx)
    find_package(msgpack-cxx CONFIG)
    if(msgpack-cxx_FOUND)
        message(STATUS "Found msgpack-cxx (>=6.x)")
    else()
        find_package(msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
        message(STATUS "Found msgpack (<=5.x)")
    endif()
endif()

if(HIPBLASLT_ENABLE_LLVM)
    find_package(LLVM REQUIRED)
endif()
if(HIPBLASLT_ENABLE_DEVICE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    if(HIPBLASLT_ENABLE_ASAN AND NOT WIN32)
        set(ASAN_LIB_PATH "unset")
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so
            OUTPUT_VARIABLE ASAN_LIB_PATH
        )
        # If ASAN_LIB_PATH is libclang_rt.asan-x86_64.so
        # rather than /path/to/libclang_rt.asan-x86_64.so then
        # we failed to locate it and HAS_PARENT_PATH is false.
        string(STRIP ${ASAN_LIB_PATH} ASAN_LIB_PATH)
        cmake_path(HAS_PARENT_PATH ASAN_LIB_PATH result)
        if(NOT result)
            message(FATAL_ERROR "Failed to locate libclang_rt.asan-x86_64.so ")
        endif()
        # Disable a few asan options to get builds going but these should be addressed
        set(asan_opts "LD_PRELOAD=${ASAN_LIB_PATH}" "ASAN_OPTIONS=detect_leaks=0,new_delete_type_mismatch=0")
    endif()

    if(HIPBLASLT_BUNDLE_PYTHON_DEPS)
        # Build the rocisa python extension.
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite/rocisa"
            "${CMAKE_CURRENT_BINARY_DIR}/rocisa_pyext")
        set(HIPBLASLT_PYTHON_DEPS "rocisa")
        # TODO: The rocisa build explicitly sets the LIBRARY_OUTPUT_DIR to the
        # (top-level!) CMAKE_BINARY_DIR/lib directory. This is not a great idea
        # and should be removed/changed to a locally scoped directory. However,
        # that is hard to do while this is all glued together with the old venv
        # setup. In the meantime, the directory passed here must match the
        # what is set there. Failure to do so will cause a ModuleNotFoundError
        # trying to load rocisa.
        hipblaslt_configure_bundled_python_command("${CMAKE_BINARY_DIR}/lib" "${asan_opts}")
    else()
        # Just pass through to the found python executable and trust that the
        # user set it up properly.
        set(HIPBLASLT_PYTHON_COMMAND "${Python3_EXECUTABLE}")
        set(HIPBLASLT_PYTHON_DEPS)
    endif()
endif()

if(HIPBLASLT_ENABLE_HOST)
    if(HIPBLASLT_BUILD_SHARED_LIBS)
        add_library(hipblaslt SHARED)
    else()
        add_library(hipblaslt STATIC)
        target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_STATIC_LIB)
    endif()
    add_library(roc::hipblaslt ALIAS hipblaslt)

    target_link_libraries(hipblaslt
        PUBLIC
            ${hipblas_target}
        PRIVATE
            hip::device
            rocisa::rocisa-cpp
            ${CMAKE_DL_LIBS}
    )

    if(HIPBLASLT_ENABLE_ASAN)
        hipblaslt_target_configure_sanitizers(hipblaslt PRIVATE)
    endif()

    if(HIPBLASLT_ENABLE_OPENMP)
        target_link_libraries(hipblaslt PRIVATE OpenMP::OpenMP_CXX)
    endif()

    target_compile_features(hipblaslt PUBLIC cxx_std_17) # I would like to make this configurable

    target_compile_definitions(hipblaslt
        PRIVATE
            TENSILE_DEFAULT_SERIALIZATION
        PUBLIC
            ROCM_USE_FLOAT16
            __HIP_PLATFORM_AMD__
    )

    if(HIPBLASLT_ENABLE_HIPBLAS_DIRECT)
        target_compile_definitions(hipblaslt PRIVATE LEGACY_HIPBLAS_DIRECT)
    endif()

    if(HIPBLASLT_ENABLE_LAZY_LOAD)
        target_compile_definitions(hipblaslt PRIVATE ROCBLASLT_TENSILE_LAZY_LOAD)
    endif()

    if(HIPBLASLT_ENABLE_HIP)
        target_compile_definitions(hipblaslt PRIVATE TENSILE_USE_HIP)
    endif()

    if(HIPBLASLT_ENABLE_MSGPACK)
        if(msgpack-cxx_FOUND)
            target_link_libraries(hipblaslt PRIVATE msgpack-cxx)
        else()
            target_link_libraries(hipblaslt PRIVATE msgpackc)
        endif()
        target_compile_definitions(hipblaslt PRIVATE TENSILE_MSGPACK)
    endif()

    if(HIPBLASLT_ENABLE_MARKER)
        target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_ENABLE_MARKER)
    endif()

    target_include_directories(hipblaslt
        PUBLIC
            $<BUILD_INTERFACE:${HIPBLASLT_LIB_DIR}/src/include>
            $<BUILD_INTERFACE:${HIPBLASLT_LIB_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/host-library/include>
            $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            "${ROCBLASLT_LIB_DIR}/include"
            "${ROCBLASLT_LIB_DIR}/src/include"
            "${CMAKE_CURRENT_BINARY_DIR}/include"
    )

    set_target_properties(hipblaslt
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/host-library
    )

    add_subdirectory(rocisa/rocisa-cpp)
    add_subdirectory(host-library)
endif()

if(HIPBLASLT_ENABLE_DEVICE)
    add_subdirectory(device-library)
endif()
if(HIPBLASLT_ENABLE_CLIENT)
    add_subdirectory(clients)
endif()


if(HIPBLASLT_ENABLE_HOST)
    # work around code object stripping failure if using /usr/bin/strip
    set(CPACK_RPM_SPEC_MORE_DEFINE "%define __strip /opt/rocm/llvm/bin/llvm-strip") # need to generalize this
    get_target_property(hipblaslt_public_headers roc::hipblaslt INTERFACE_SOURCES)

    install(
        FILES ${hipblaslt_public_headers}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt"
        COMPONENT hipBLASLt_Development
    )
    # temporary workaround to avoid updating downstream components
    # TODO: remove the follow file install once rocblas is updated
    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/host-library/include/hipblaslt/hipblaslt-export.h"
            "${CMAKE_CURRENT_BINARY_DIR}/host-library/include/hipblaslt/hipblaslt-version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT hipBLASLt_Development
    )

    rocm_install_targets(TARGETS hipblaslt)

    rocm_export_targets(
        TARGETS roc::hipblaslt
        DEPENDS PACKAGE hip
        DEPENDS PACKAGE hipblas-common
        NAMESPACE roc::
    )
endif()

if(HIPBLASLT_ENABLE_DEVICE)
    if(WIN32)
        set(HIPBLASLT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}hipblaslt/bin" CACHE PATH "path to tensile library")
    else()
        set(HIPBLASLT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/hipblaslt" CACHE PATH "path to tensile library")
    endif()
    rocm_install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Tensile/library
        DESTINATION ${HIPBLASLT_TENSILE_LIBRARY_DIR}
        COMPONENT devel #may need to conditionally set this
    )
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

include(ROCMCreatePackage)
include(ROCMClients)

#set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib")

set(GFORTRAN_RPM "libgfortran4")
set(GFORTRAN_DEB "libgfortran4")
if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel")
    if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
        set(GFORTRAN_RPM "libgfortran")
    endif()
elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
    set(GFORTRAN_RPM "gfortran")
endif()

if(HIPBLASLT_ENABLE_CLIENT)
    rocm_install(
        FILES 
            "${HIPBLASLT_CLIENTS_DIR}/include/hipblaslt_common.yaml"
            "${HIPBLASLT_CLIENTS_DIR}/include/hipblaslt_template.yaml"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT clients-common
    )
    rocm_install(
        PROGRAMS "${HIPBLASLT_CLIENTS_DIR}/common/hipblaslt_gentest.py"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT clients-common
    )
    rocm_install(TARGETS hipblaslt-bench COMPONENT benchmarks)
    rocm_install(
        FILES "${HIPBLASLT_CLIENTS_DIR}/benchmarks/sequence.yaml"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT benchmarks
    )

    if(HIPBLASLT_ENABLE_MARKER)
        rocm_package_add_dependencies(DEPENDS "roctracer >= 1.0.0")
    endif()

    if(NOT CLIENTS_OS)
        rocm_set_os_id(CLIENTS_OS)
        string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
        rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
    endif()

    rocm_package_setup_component(clients)
    rocm_package_setup_client_component(clients-common)
    rocm_package_setup_client_component(benchmarks
        DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}"
    )

endif()

if(HIPBLASLT_BUILD_TESTING OR BUILD_TESTING)
    rocm_install(
        TARGETS hipblaslt-test
        COMPONENT tests
    )
    rocm_install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/clients/hipblaslt_gtest.data"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT tests
    )
    rocm_package_setup_client_component(tests
        DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}"
    )
endif()

if(HIPBLASLT_ENABLE_SAMPLES)
    install(
        TARGETS
            sample_hipblaslt_gemm
            sample_hipblaslt_gemm_ext
            sample_hipblaslt_gemm_batched
            sample_hipblaslt_gemm_batched_ext
            sample_hipblaslt_gemm_tuning_splitk_ext
            sample_hipblaslt_gemm_bias
            sample_hipblaslt_gemm_bias_ext
            sample_hipblaslt_gemm_get_all_algos
            sample_hipblaslt_gemm_get_all_algos_ext
            sample_hipblaslt_gemm_get_algo_by_index_ext
            sample_hipblaslt_gemm_alphavec_ext
            sample_hipblaslt_gemm_gelu_aux_bias
            sample_hipblaslt_gemm_gelu_aux_bias_ext
            sample_hipblaslt_gemm_amax
            sample_hipblaslt_gemm_amax_ext
            sample_hipblaslt_gemm_amax_with_scale
            sample_hipblaslt_gemm_amax_with_scale_ext
            sample_hipblaslt_gemm_bgradb
            sample_hipblaslt_gemm_ext_bgradb
            sample_hipblaslt_gemm_dgelu_bgrad
            sample_hipblaslt_gemm_dgelu_bgrad_ext
            sample_hipblaslt_gemm_is_tuned_ext
            sample_hipblaslt_gemm_tuning_wgm_ext
            sample_hipblaslt_gemm_with_scale_a_b
            sample_hipblaslt_gemm_with_scale_a_b_ext
            sample_hipblaslt_groupedgemm_ext
            sample_hipblaslt_groupedgemm_fixed_mk_ext
            sample_hipblaslt_groupedgemm_get_all_algos_ext
            sample_hipblaslt_gemm_mix_precision
            sample_hipblaslt_gemm_mix_precision_ext
            sample_hipblaslt_gemm_mix_precision_with_amax_ext
            sample_hipblaslt_gemm_attr_tciA_tciB
            sample_hipblaslt_ext_op_layernorm
            sample_hipblaslt_ext_op_amax
            sample_hipblaslt_ext_op_amax_with_scale
            sample_hipblaslt_gemm_with_TF32
            sample_hipblaslt_gemm_swizzle_a
            sample_hipblaslt_gemm_bias_swizzle_a_ext
            sample_hipblaslt_weight_swizzle_padding
            sample_hipblaslt_gemm_swish_bias
        RUNTIME
        COMPONENT samples
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/hipblaslt-samples
    )
    rocm_package_setup_client_component(samples)
endif()

set(HIPBLASLT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")
rocm_create_package(
    NAME hipblaslt
    DESCRIPTION "HIP library for GEMM operations with extended functionality"
    MAINTAINER "hipBLASLt Maintainer <hipblaslt-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPBLASLT_CONFIG_DIR}
)

if(HIPBLASLT_ENABLE_COVERAGE)
    target_compile_options(hipblaslt PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipblaslt PRIVATE -fprofile-instr-generate)

    target_compile_options(hipblaslt-clients-common PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipblaslt-clients-common PRIVATE -fprofile-instr-generate)

    target_compile_options(hipblaslt-bench PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipblaslt-bench PRIVATE -fprofile-instr-generate)

    target_compile_options(hipblaslt-test PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipblaslt-test PRIVATE -fprofile-instr-generate)

    set(coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")
    add_custom_target(
        code_cov_tests
        DEPENDS hipblaslt-test
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${coverage_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${coverage_dir}/profraw"
        COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_%p.profraw" 
                                        GTEST_LISTENER=NO_PASS_LINE_IN_LOG 
                                        $<TARGET_FILE:hipblaslt-test> 
                                        --precompile=hipblaslt-test-precompile.db
      )
      find_program(
        LLVM_PROFDATA
        llvm-profdata
        HINTS llvm/bin
        REQUIRED
      )
      find_program(
        LLVM_COV
        llvm-cov
        REQUIRED
        HINTS llvm/bin
      )
      add_custom_target(
        coverage
        DEPENDS code_cov_tests
        COMMAND ${LLVM_PROFDATA} merge -sparse "${coverage_dir}/profraw/hipblaslt-coverage_*.profraw" -o "${coverage_dir}/hipblaslt.profdata"
        COMMAND ${LLVM_COV} report -object $<TARGET_FILE:hipblaslt> -instr-profile="${coverage_dir}/hipblaslt.profdata"
        COMMAND ${LLVM_COV} show -object $<TARGET_FILE:hipblaslt> -instr-profile="${coverage_dir}/hipblaslt.profdata" -format=html -output-dir="${coverage_dir}"
      )
endif()
