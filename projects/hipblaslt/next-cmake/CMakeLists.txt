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

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(CMakeDependentOption)
include(hipblaslt_python)

set(HIPBLASLT_ENABLE_DEVICE ON CACHE BOOL "Build hipBLASLt device libraries.")
set(HIPBLASLT_ENABLE_CLIENT ON CACHE BOOL "Build hipBLASLt client apps.")
cmake_dependent_option(HIPBLASLT_ENABLE_HOST "Build hipBLASLt host library." ON "HIPBLASLT_ENABLE_CLIENT" OFF)

if(HIPBLASLT_ENABLE_CLIENT)
    set(HIPBLASLT_BUILD_TESTING ON CACHE BOOL "Build hipblaslt client tests.")
    set(HIPBLASLT_ENABLE_SAMPLES ON CACHE BOOL "Build client samples.")
    # rocm-smi is not presently available on Windows so we do not require it.
    cmake_dependent_option(HIPBLASLT_REQUIRE_ROCM_SMI "Require rocm_smi." ON "NOT WIN32" OFF)
endif()

if(HIPBLASLT_ENABLE_HOST)
    set(HIPBLASLT_BUILD_SHARED ON CACHE BOOL "Build the hipblaslt library as shared vs static")
    set(HIPBLASLT_ENABLE_HIP ON CACHE BOOL "Use the HIP runtime.")
    set(HIPBLASLT_ENABLE_MSGPACK ON CACHE BOOL "Use msgpack for parsing configuration files.")
    set(HIPBLASLT_ENABLE_OPENMP ON CACHE BOOL "Use OpenMP to improve performance.")
    set(HIPBLASLT_ENABLE_LLVM OFF CACHE BOOL "Use msgpack for parsing configuration files.")
    set(HIPBLASLT_ENABLE_ROCROLLER OFF CACHE BOOL "Use RocRoller library.")
    set(HIPBLASLT_ENABLE_BLIS ON CACHE BOOL "Enable BLIS support.") # I don't know that we can build with this OFF
    set(HIPBLASLT_ENABLE_LAZY_LOAD ON CACHE BOOL "Enable lazy loading of runtime code oject files to reduce ram usage.")
endif()

if(HIPBLASLT_ENABLE_DEVICE)
    set(HIPBLASLT_BUNDLE_PYTHON_DEPS ON CACHE BOOL "Build python dependencies requied for device code generation.")
    set(HIPBLASLT_DEVICE_JOBS "" CACHE STRING "Number of CPU threads to use for building device libraries (will use nproc if unset).")
    set(HIPBLASLT_DEVICE_KEEP_TMP OFF CACHE STRING "Keep temporary build directory for device libraries (turning this ON  bloat the build size).")
    set(HIPBLASLT_DEVICE_LIBLOGIC_PATH "" CACHE STRING "Path to library logic files (will use 'library' if unset).")
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
set(ROCBLASLT_LIB_DIR "${HIPBLASLT_LIB_DIR}/src/amd_detail/rocblaslt")

find_package(hip REQUIRED)
if(HIPBLASLT_ENABLE_HOST)
    if(NOT TARGET roc::hipblas-common)
        find_package(hipblas-common REQUIRED)
    endif()
    if(HIPBLASLT_ENABLE_BLIS)
        find_package(BLIS REQUIRED)
    endif()
endif()
if(HIPBLASLT_REQUIRE_ROCM_SMI)
    find_package(rocm_smi REQUIRED)
else()
    find_package(rocm_smi)
endif()
if(HIPBLASLT_ENABLE_CLIENT)
    #find_package(Threads REQUIRED) # TODO: We don't fail to build or link without Threads remove?
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()
if(HIPBLASLT_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()
if(HIPBLASLT_ENABLE_MSGPACK)
    # See: https://github.com/msgpack/msgpack-c/wiki/Q%26A#how-to-support-both-msgpack-c-c-version-5x-and-6x-
    # Prefer 6.x (msgpack-cxx) as that is what we bundle in the build.
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
        hipblaslt_configure_bundled_python_command("${CMAKE_BINARY_DIR}/lib")
    else()
        # Just pass through to the found python executable and trust that the
        # user set it up properly.
        set(HIPBLASLT_PYTHON_COMMAND "${Python3_EXECUTABLE}")
        set(HIPBLASLT_PYTHON_DEPS)
    endif()
endif()
if(HIPBLASLT_ENABLE_HOST)
    if(HIPBLASLT_BUILD_SHARED)
        add_library(hipblaslt SHARED)
    else()
        add_library(hipblaslt STATIC)
        target_compile_definitions(hipblaslt PRIVATE HIPBLASLT_STATIC_LIB)
    endif()
    add_library(roc::hipblaslt ALIAS hipblaslt)

    target_link_libraries(hipblaslt
        PUBLIC
            roc::hipblas-common
        PRIVATE
            hip::device
            rocisa::rocisa-cpp
            ${CMAKE_DL_LIBS}
    )

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

    if(HIPBLASLT_ENABLE_LAZY_LOAD)
        target_compile_definitions(hipblaslt
            PRIVATE
                ROCBLASLT_TENSILE_LAZY_LOAD
        )
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
    install(TARGETS hipblaslt rocisa-cpp
        EXPORT hipblaslt-targets
        LIBRARY
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT hipBLASLt_Runtime
        ARCHIVE
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT hipBLASLt_Development
        RUNTIME
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT hipBLASLt_Runtime
        PUBLIC_HEADER
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            COMPONENT hipBLASLt_Development
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
    install(
        EXPORT hipblaslt-targets
        NAMESPACE roc::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipblaslt
    )
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
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-config-version.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipblaslt-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-config.cmake" @ONLY)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-config-version.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipblaslt"
        COMPONENT hipBLASLt_Development
    )
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}"
        COMPONENT hipBLASLt_Docs
    )

    if(HIPBLASLT_ENABLE_DEVICE)
        install(
            DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Tensile/library"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/hipblaslt"
            COMPONENT hipBLASLt_Runtime
        )
    endif()

    if(HIPBLASLT_ENABLE_CLIENT)
        install(
            TARGETS hipblaslt-bench
            EXPORT hipblaslt-targets
            COMPONENT hipBLASLt_Runtime
        )
        if(BUILD_TESTING OR HIPBLASLT_BUILD_TESTING)
            install(
                TARGETS hipblaslt-test
                EXPORT hipblaslt-targets
                COMPONENT hipBLASLt_Runtime
            )
            install(
                FILES "${CMAKE_CURRENT_BINARY_DIR}/clients/hipblaslt_gtest.data"
                DESTINATION "${CMAKE_INSTALL_BINDIR}"
                COMPONENT hipBLASLt_Runtime
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
                EXPORT hipblaslt-targets
                COMPONENT hipBLASLt_Runtime
            )
        endif()
    endif()

    set(CPACK_PACKAGE_NAME "hipBLASLt")
    set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
    set(CPACK_PACKAGE_CONTACT "hipBLASLt Maintainer <hipblaslt-maintainer@amd.com>")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "HIP library for GEMM operation with extended functionality")
    set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
    set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
    set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
    set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_LIST_DIR}/../LICENSE.md)
    set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_LIST_DIR}/../README.md)
    set(CPACK_VERBATIM_VARIABLES YES)
    set(CPACK_COMPONENTS_ALL "hipBLASLt_Runtime;hipBLASLt_Development;hipBLASLt_Docs")

    set(CPACK_COMPONENTS_GROUPING IGNORE)

    set(CPACK_DEB_COMPONENT_INSTALL ON)
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
    set(CPACK_DEBIAN_HIPBLASLT_RUNTIME_PACKAGE_NAME "hipblaslt")
    set(CPACK_DEBIAN_HIPBLASLT_RUNTIME_PACKAGE_SECTION "runtime")
    set(CPACK_DEBIAN_HIPBLASLT_DEVELOPMENT_PACKAGE_NAME "hipblaslt-dev")
    set(CPACK_DEBIAN_HIPBLASLT_DOCS_PACKAGE_NAME "hipblaslt-docs")
    set(CPACK_DEBIAN_PACKAGE_PROVIDES "hipblaslt (= ${PROJECT_VERSION})")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "hipblas-common (>=1.0)")

    set(CPACK_RPM_COMPONENT_INSTALL ON)
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
    set(CPACK_RPM_HIPBLASLT_RUNTIME_PACKAGE_NAME "hipblaslt")
    set(CPACK_RPM_HIPBLASLT_DEVELOPMENT_PACKAGE_NAME "hipblaslt-devel")
    set(CPACK_RPM_HIPBLASLT_DOCS_PACKAGE_NAME "hipblaslt-docs")
    set(CPACK_RPM_HIPBLASLT_RUNTIME_PACKAGE_REQUIRES "hipblaslt (= ${PROJECT_VERSION})")
    set(CPACK_RPM_PACKAGE_PROVIDES "hipblaslt (= ${PROJECT_VERSION})")
    set(CPACK_RPM_PACKAGE_DEPENDS "hipblas-common (>=1.0)")
    
    include(CPack)
    
    cpack_add_component(hipBLASLt_Runtime
        DISPLAY_NAME "Runtime"
        DESCRIPTION "Runtime libraries for hipBLASLt"
        # GROUP Runtime
        REQUIRED
    )
    cpack_add_component(hipBLASLt_Development
        DISPLAY_NAME "Development"
        DESCRIPTION "Headers and CMake files for hipBLASLt"
        # GROUP Development
        DEPENDS hipBLASLt_Runtime
    )
    cpack_add_component(hipBLASLt_Docs
        DISPLAY_NAME "Documentation"
        DESCRIPTION "Documentation for hipBLASLt"
        # GROUP Documentation
    )
endif()