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

# Sets ${output} to the environment of the code-generation commands. Python
# imports tensilelite from the wheel unpacked into the build tree (never from
# the source tree, and never by installing it into the build Python), rocisa
# from the in-tree build, and rocisa_stinkytofu_adaptor, which is not part of
# the wheel yet, from the source tree.
function(hipblaslt_tensilelite_python_environment output)
    set(_path "$ENV{PATH}")
    if(WIN32)
        string(REPLACE ";" "$<SEMICOLON>" _path "${_path}")
    endif()

    if(WIN32)
        set(_ds "$<SEMICOLON>")
    else()
        set(_ds ":")
    endif()
    set(_python_path
        "${HIPBLASLT_TENSILELITE_WHEEL_ROOT}"
        "$<TARGET_FILE_DIR:_rocisa>/.."
        "${hipblaslt_SOURCE_DIR}/tensilelite/rocisa_stinkytofu_adaptor"
    )
    list(JOIN _python_path "${_ds}" _python_path)

    set(_environment
        "PYTHONPATH=${_python_path}"
        "PATH=${_path}"
    )
    if(HIPBLASLT_ENABLE_THEROCK)
        # Internal marker: only these commands select TensileLite's TheRock
        # build-time ROCm adapter; installed wheels never see it. The version
        # is passed only when TheRock supplied a real one.
        list(APPEND _environment "TENSILELITE_ROCM_ADAPTER=therock-ci")
        if(HIPBLASLT_BUILD_ROCM_VERSION)
            list(APPEND _environment
                "THEROCK_PACKAGE_VERSION=${HIPBLASLT_BUILD_ROCM_VERSION}")
        endif()
    else()
        list(APPEND _environment
            "ROCM_PATH=${HIPBLASLT_BUILD_ROCM_ROOT}")
    endif()
    set(${output} "${_environment}" PARENT_SCOPE)
endfunction()

# Sets HIPBLASLT_PYTHON_COMMAND, HIPBLASLT_PYTHON_DEPS (targets) and
# HIPBLASLT_PYTHON_INPUTS (files) in the parent scope for the code-generation
# commands. HIPBLASLT_PYTHON_COMMAND may be a multi-token list, so use it
# without quotes in COMMAND lists.
function(hipblaslt_configure_tensilelite_python asan_options)
    if(NOT HIPBLASLT_ENABLE_DEVICE)
        set(HIPBLASLT_PYTHON_COMMAND "${Python3_EXECUTABLE}" PARENT_SCOPE)
        set(HIPBLASLT_PYTHON_DEPS "" PARENT_SCOPE)
        set(HIPBLASLT_PYTHON_INPUTS "" PARENT_SCOPE)
        return()
    endif()

    if(NOT TARGET _rocisa)
        message(FATAL_ERROR
            "Device generation requires the in-tree _rocisa target")
    endif()

    # Code generation runs the built wheel: unpack it into the build tree. The
    # build Python's site-packages are never modified.
    set(_canonical_wheel
        "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-release-wheels/tensilelite-${TENSILELITE_DISTRIBUTION_VERSION}-py3-none-any.whl")
    set(HIPBLASLT_TENSILELITE_WHEEL_ROOT "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-wheel")
    set(_unpack_stamp "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-wheel.stamp")
    add_custom_command(
        OUTPUT "${_unpack_stamp}"
        COMMAND "${Python3_EXECUTABLE}"
            "${hipblaslt_SOURCE_DIR}/tensilelite/scripts/install_release_wheels.py"
            --destination "${HIPBLASLT_TENSILELITE_WHEEL_ROOT}"
            "${_canonical_wheel}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_unpack_stamp}"
        DEPENDS
            "${_canonical_wheel}"
            "${hipblaslt_SOURCE_DIR}/tensilelite/scripts/install_release_wheels.py"
        COMMENT "Unpacking the TensileLite wheel for device generation"
        VERBATIM
    )
    add_custom_target(tensilelite-python-build-environment
        DEPENDS "${_unpack_stamp}" _rocisa)

    hipblaslt_tensilelite_python_environment(_python_environment)
    set(_python_command "${CMAKE_COMMAND}" -E env ${_python_environment})
    if(asan_options)
        list(APPEND _python_command ${asan_options})
    endif()
    list(APPEND _python_command --
        "${Python3_EXECUTABLE}")

    set(HIPBLASLT_TENSILELITE_WHEEL_ROOT "${HIPBLASLT_TENSILELITE_WHEEL_ROOT}" PARENT_SCOPE)
    set(HIPBLASLT_PYTHON_COMMAND "${_python_command}" PARENT_SCOPE)
    set(HIPBLASLT_PYTHON_DEPS "tensilelite-python-build-environment" PARENT_SCOPE)
    set(HIPBLASLT_PYTHON_INPUTS "${_unpack_stamp}" PARENT_SCOPE)
endfunction()
