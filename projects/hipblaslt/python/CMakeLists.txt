# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Builds the hipblaslt._core extension against an installed hipBLASLt. This project is opt-in
# and source-build only: hipBLASLt's own CMake never adds or installs it, and TheRock never
# builds it.
cmake_minimum_required(VERSION 3.25)
project(hipblaslt_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# The ROCm that provides HIP. hipblaslt_ROOT also counts as the ROCm root when it's the only one
# set.
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(HIPBLASLT_PY_ROCM "$ENV{ROCM_PATH}")
elseif(DEFINED ENV{ROCM_HOME} AND NOT "$ENV{ROCM_HOME}" STREQUAL "")
    set(HIPBLASLT_PY_ROCM "$ENV{ROCM_HOME}")
elseif(DEFINED ENV{hipblaslt_ROOT} AND NOT "$ENV{hipblaslt_ROOT}" STREQUAL "")
    set(HIPBLASLT_PY_ROCM "$ENV{hipblaslt_ROOT}")
elseif(NOT WIN32)
    set(HIPBLASLT_PY_ROCM "/opt/rocm")
else()
    message(FATAL_ERROR "Set ROCM_PATH to the ROCm install to build against "
                        "(for the rocm-sdk wheels: rocm-sdk path --root).")
endif()
file(TO_CMAKE_PATH "${HIPBLASLT_PY_ROCM}" HIPBLASLT_PY_ROCM)
list(APPEND CMAKE_PREFIX_PATH "${HIPBLASLT_PY_ROCM}")

# hip::host only. Never enable the HIP language: it breaks the Windows CI lanes.
find_package(hip CONFIG REQUIRED)

# The hipBLASLt the package version came from. The metadata provider resolves it, and
# find_package looks nowhere else, so both steps use one install even when another one with the
# same version (and a different git revision) is on CMAKE_PREFIX_PATH.
execute_process(
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/skbuild_plugins/hipblaslt_version_provider.py"
    OUTPUT_VARIABLE HIPBLASLT_PY_HIPBLASLT_ROOT
    ERROR_VARIABLE _provider_error
    RESULT_VARIABLE _provider_result
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _provider_result EQUAL 0)
    message(FATAL_ERROR "${_provider_error}")
endif()
file(TO_CMAKE_PATH "${HIPBLASLT_PY_HIPBLASLT_ROOT}" HIPBLASLT_PY_HIPBLASLT_ROOT)
find_package(hipblaslt CONFIG REQUIRED PATHS "${HIPBLASLT_PY_HIPBLASLT_ROOT}" NO_DEFAULT_PATH)

get_target_property(_hipblaslt_type roc::hipblaslt TYPE)
if(NOT _hipblaslt_type STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR
        "The bindings need a shared libhipblaslt, but roc::hipblaslt in ${hipblaslt_DIR} is a "
        "${_hipblaslt_type}. Build hipBLASLt without --static.")
endif()

if(DEFINED SKBUILD_PROJECT_VERSION
   AND NOT hipblaslt_VERSION VERSION_EQUAL SKBUILD_PROJECT_VERSION)
    message(FATAL_ERROR
        "The package version (${SKBUILD_PROJECT_VERSION}, from hipblaslt-version.h) doesn't match "
        "the hipBLASLt CMake found (${hipblaslt_VERSION}, in ${hipblaslt_DIR}).")
endif()

# Where the build-time hipBLASLt lives. hipblaslt/_build_info.py records it for the import-time
# lookup's last fallback, and Linux RUNPATH points at it.
get_target_property(_hipblaslt_lib roc::hipblaslt IMPORTED_LOCATION_RELEASE)
if(NOT _hipblaslt_lib)
    get_target_property(_hipblaslt_lib roc::hipblaslt LOCATION)
endif()
get_filename_component(HIPBLASLT_PY_BUILD_LIB_DIR "${_hipblaslt_lib}" DIRECTORY)
# The name _core's NEEDED entry uses: the SONAME on Linux, the DLL name on Windows.
get_target_property(HIPBLASLT_PY_LIB_NAME roc::hipblaslt IMPORTED_SONAME_RELEASE)
if(NOT HIPBLASLT_PY_LIB_NAME)
    get_filename_component(HIPBLASLT_PY_LIB_NAME "${_hipblaslt_lib}" NAME)
endif()
# The HIP runtime the build used, which may live outside hipBLASLt's prefix.
get_target_property(HIPBLASLT_PY_HIP_RUNTIME hip::amdhip64 IMPORTED_LOCATION_RELEASE)
if(NOT HIPBLASLT_PY_HIP_RUNTIME)
    get_target_property(HIPBLASLT_PY_HIP_RUNTIME hip::amdhip64 LOCATION)
endif()
get_target_property(_hipblaslt_inc roc::hipblaslt INTERFACE_INCLUDE_DIRECTORIES)
list(GET _hipblaslt_inc 0 HIPBLASLT_PY_BUILD_INCLUDE_DIR)
# hipblas-common can live in a different prefix (invoke build installs hipBLASLt on its own).
find_path(HIPBLASLT_PY_HIPBLAS_COMMON_INCLUDE_DIR hipblas-common/hipblas-common.h
    HINTS ${_hipblaslt_inc} "${HIPBLASLT_PY_ROCM}/include" NO_DEFAULT_PATH)
if(NOT HIPBLASLT_PY_HIPBLAS_COMMON_INCLUDE_DIR)
    message(FATAL_ERROR "Can't find hipblas-common/hipblas-common.h next to hipBLASLt or in "
                        "${HIPBLASLT_PY_ROCM}/include.")
endif()
if(WIN32)
    set(HIPBLASLT_PY_BUILD_BIN_DIR "${HIPBLASLT_PY_ROCM}/bin")
else()
    set(HIPBLASLT_PY_BUILD_BIN_DIR "")
endif()

nanobind_add_module(_core NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/module.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/enums.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/descriptors.cpp"
)
# hipBLASLt's headers first: hip::host's include directory may hold an older hipBLASLt.
target_include_directories(_core BEFORE PRIVATE "${HIPBLASLT_PY_BUILD_INCLUDE_DIR}")
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
target_link_libraries(_core PRIVATE hip::host roc::hipblaslt)
if(WIN32)
    target_compile_definitions(_core PRIVATE NOMINMAX)
else()
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "${HIPBLASLT_PY_BUILD_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH ON)
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/csrc/_build_info.py.in"
               "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py" @ONLY)

install(TARGETS _core DESTINATION hipblaslt)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py" DESTINATION hipblaslt)
