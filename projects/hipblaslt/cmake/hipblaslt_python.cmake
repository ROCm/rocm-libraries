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

function(hipblaslt_resolve_build_rocm_version output)
    if(HIPBLASLT_ENABLE_THEROCK)
        if(THEROCK_PACKAGE_VERSION AND NOT THEROCK_PACKAGE_VERSION STREQUAL "git")
            set(_version "${THEROCK_PACKAGE_VERSION}")
        else()
            message(FATAL_ERROR
                "HIPBLASLT_ENABLE_THEROCK requires a release THEROCK_PACKAGE_VERSION")
        endif()
    else()
        hipblaslt_resolve_build_rocm_root(_root)
        set(_version_file "${_root}/.info/version")
        file(READ "${_version_file}" _version)
        string(STRIP "${_version}" _version)
    endif()
    set(${output} "${_version}" PARENT_SCOPE)
endfunction()

function(hipblaslt_tensilelite_python_environment output)
    set(_path "$ENV{PATH}")
    if(WIN32)
        string(REPLACE ";" "$<SEMICOLON>" _path "${_path}")
    endif()

    set(_environment
        "TENSILELITE_ROCM_VERSION=${HIPBLASLT_BUILD_ROCM_VERSION}"
        "PYTHONPATH=$<TARGET_FILE_DIR:_rocisa>/.."
        "PATH=${_path}"
    )
    if(HIPBLASLT_ENABLE_THEROCK)
        list(APPEND _environment
            "THEROCK_PACKAGE_VERSION=${HIPBLASLT_BUILD_ROCM_VERSION}")
    else()
        list(APPEND _environment
            "ROCM_PATH=${HIPBLASLT_BUILD_ROCM_ROOT}")
    endif()
    set(${output} "${_environment}" PARENT_SCOPE)
endfunction()

function(hipblaslt_configure_tensilelite_python asan_options)
    if(NOT HIPBLASLT_ENABLE_DEVICE)
        set(HIPBLASLT_PYTHON_COMMAND "${Python3_EXECUTABLE}" PARENT_SCOPE)
        set(HIPBLASLT_PYTHON_DEPS "" PARENT_SCOPE)
        return()
    endif()

    if(NOT TARGET _rocisa)
        message(FATAL_ERROR
            "Device generation requires the in-tree _rocisa target")
    endif()

    set(_canonical_wheel
        "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-release-wheels/tensilelite-${TENSILELITE_DISTRIBUTION_VERSION}-py3-none-any.whl")
    set(_install_stamp "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-wheel-install.stamp")
    add_custom_command(
        OUTPUT "${_install_stamp}"
        COMMAND "${Python3_EXECUTABLE}" -m pip install
            --disable-pip-version-check --force-reinstall --no-deps
            "${_canonical_wheel}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_install_stamp}"
        DEPENDS "${_canonical_wheel}"
        COMMENT "Installing the canonical TensileLite wheel into the build Python"
        VERBATIM
        USES_TERMINAL
    )
    add_custom_target(tensilelite-python-build-environment
        DEPENDS "${_install_stamp}" _rocisa)

    hipblaslt_tensilelite_python_environment(_python_environment)
    set(_python_command "${CMAKE_COMMAND}" -E env ${_python_environment})
    if(asan_options)
        list(APPEND _python_command ${asan_options})
    endif()
    list(APPEND _python_command --
        "${Python3_EXECUTABLE}")

    set(HIPBLASLT_PYTHON_COMMAND "${_python_command}" PARENT_SCOPE)
    set(HIPBLASLT_PYTHON_DEPS "tensilelite-python-build-environment" PARENT_SCOPE)
endfunction()
