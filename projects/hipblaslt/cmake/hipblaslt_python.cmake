# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

macro(hipblaslt_find_python minimum_version python_dev_components)
    find_package(Python3 ${minimum_version} COMPONENTS Interpreter ${python_dev_components} REQUIRED)
    set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
    find_package(Python ${minimum_version} COMPONENTS Interpreter ${python_dev_components} REQUIRED)
    if(NOT "${Python_EXECUTABLE}" STREQUAL "${Python3_EXECUTABLE}")
        message(WARNING
            "FindPython and FindPython3 found different executables. Pin "
            "-DPython_EXECUTABLE and -DPython3_EXECUTABLE if needed "
            "(${Python_EXECUTABLE} vs ${Python3_EXECUTABLE})")
    endif()
endmacro()

# Keep discovery in the caller's scope, including Python's imported targets.
macro(hipblaslt_find_build_python)
    if(HIPBLASLT_ENABLE_DEVICE OR HIPBLASLT_PREPARE_TENSILELITE_GENERATOR OR ROCISA_BUILD_PYTHON
            OR HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS)
        if(ROCISA_USE_STABLE_ABI)
            hipblaslt_find_python(3.12 "Development.Module;Development.SABIModule")
        else()
            hipblaslt_find_python(3.10 "Development.Module")
        endif()
    elseif(TENSILELITE_ENABLE_CLIENT OR HIPBLASLT_ENABLE_CLIENT)
        hipblaslt_find_python(3.10 "")
    endif()
endmacro()

function(hipblaslt_resolve_build_rocm_root output)
    if(HIPBLASLT_ENABLE_THEROCK)
        if(NOT THEROCK_TOOLCHAIN_ROOT)
            message(FATAL_ERROR
                "HIPBLASLT_ENABLE_THEROCK requires THEROCK_TOOLCHAIN_ROOT")
        endif()
        set(_root "${THEROCK_TOOLCHAIN_ROOT}")
    elseif(ROCM_PATH)
        set(_root "${ROCM_PATH}")
    elseif(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(_root "$ENV{ROCM_PATH}")
    elseif(WIN32)
        message(FATAL_ERROR
            "A standalone Windows build requires ROCM_PATH to select the build SDK")
    else()
        set(_root "/opt/rocm")
    endif()
    cmake_path(ABSOLUTE_PATH _root NORMALIZE)
    set(${output} "${_root}" PARENT_SCOPE)
endfunction()

# Sets ${output} to the ROCm identity stamped into the tensilelite wheel, or to
# an empty string for a development build (version +g<commit>). Only TheRock
# builds pass one: THEROCK_PACKAGE_VERSION, when it is set and is not the
# local-build placeholder "git". Standalone builds never stamp an identity;
# the ROCm the client is compiled against is recorded on the client instead
# (tensilelite-client --build-rocm).
function(hipblaslt_resolve_build_rocm_version output)
    set(_version "")
    if(HIPBLASLT_ENABLE_THEROCK AND THEROCK_PACKAGE_VERSION
            AND NOT THEROCK_PACKAGE_VERSION STREQUAL "git")
        set(_version "${THEROCK_PACKAGE_VERSION}")
    endif()
    set(${output} "${_version}" PARENT_SCOPE)
endfunction()

# Adapter selection is command-scoped and shared by dense generation, ExtOps and
# embedded consumers. Python/native search paths belong to the generator target.
function(hipblaslt_tensilelite_python_environment output)
    set(_environment "ROCM_VERSION=${hip_VERSION}")
    if(HIPBLASLT_ENABLE_THEROCK)
        list(APPEND _environment "TENSILELITE_ROCM_ADAPTER=therock-ci")
        if(HIPBLASLT_BUILD_ROCM_VERSION)
            list(APPEND _environment
                "THEROCK_PACKAGE_VERSION=${HIPBLASLT_BUILD_ROCM_VERSION}")
        endif()
    else()
        list(APPEND _environment "ROCM_PATH=${HIPBLASLT_BUILD_ROCM_ROOT}")
    endif()
    set(${output} "${_environment}" PARENT_SCOPE)
endfunction()

function(hipblaslt_configure_tensilelite_python asan_options)
    if(NOT HIPBLASLT_ENABLE_DEVICE AND NOT HIPBLASLT_PREPARE_TENSILELITE_GENERATOR)
        return()
    endif()
    if(NOT TARGET _rocisa)
        message(FATAL_ERROR "Packaged generation requires the in-tree _rocisa target")
    endif()
    include("${hipblaslt_SOURCE_DIR}/tensilelite/cmake/TensileLiteGenerator.cmake")
    hipblaslt_tensilelite_python_environment(_environment)
    if(TENSILELITE_GENERATOR_WHEEL)
        set(_selection WHEEL "${TENSILELITE_GENERATOR_WHEEL}")
    else()
        set(_selection WHEEL_MANIFEST "${TENSILELITE_CANONICAL_WHEEL_MANIFEST}")
    endif()
    get_filename_component(_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    tensilelite_prepare_generator(
        TARGET tensilelite-generator
        ${_selection}
        METADATA "${TENSILELITE_METADATA_FILE}"
        PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
        TOOL_BIN_DIRS "${_compiler_dir}"
        ENVIRONMENT ${_environment} ${asan_options})
    add_dependencies(tensilelite-generator tensilelite-metadata)
    if(NOT TENSILELITE_GENERATOR_WHEEL)
        add_dependencies(tensilelite-generator tensilelite-canonical-release-wheel)
    endif()
    if(HIPBLASLT_PREPARE_TENSILELITE_GENERATOR)
        add_custom_target(tensilelite-prepared-generator ALL DEPENDS tensilelite-generator)
    endif()
endfunction()

# Install the artifact handoff independently of tests and generated devices.
# Native extensions/resources retain their existing devel install rules.
function(hipblaslt_install_generator_support)
    if(NOT TARGET _rocisa OR (NOT HIPBLASLT_ENABLE_DEVICE
            AND NOT HIPBLASLT_PREPARE_TENSILELITE_GENERATOR))
        return()
    endif()
    include(CMakePackageConfigHelpers)
    set(_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/tensilelite")
    set(_support_source "${hipblaslt_SOURCE_DIR}/tensilelite")
    configure_package_config_file(
        "${_support_source}/cmake/tensilelite-generator-support.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-generator-support.cmake.gen"
        INSTALL_DESTINATION "${_config_dir}"
        PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_BINDIR)
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-generator-support.cmake"
        INPUT "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-generator-support.cmake.gen")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-generator-support.cmake"
        DESTINATION "${_config_dir}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT})
    install(FILES "${TENSILELITE_METADATA_FILE}"
        DESTINATION "${_config_dir}" RENAME tensilelite-generator-metadata.json COMPONENT ${TENSILELITE_DEVEL_COMPONENT})
    install(FILES
        "${_support_source}/cmake/TensileLiteGenerator.cmake"
        "${_support_source}/cmake/hipblaslt_codegen.cmake"
        DESTINATION "${_config_dir}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT})
    install(FILES
        "${_support_source}/scripts/prepare_generator.py"
        "${_support_source}/scripts/check_release_wheel_contents.py"
        "${_support_source}/scripts/install_release_wheels.py"
        DESTINATION "${_config_dir}/generator-scripts" COMPONENT ${TENSILELITE_DEVEL_COMPONENT})
    install(FILES "${_support_source}/scripts/generator-bootstrap/sitecustomize.py"
        DESTINATION "${_config_dir}/generator-scripts/generator-bootstrap"
        COMPONENT ${TENSILELITE_DEVEL_COMPONENT})
endfunction()
