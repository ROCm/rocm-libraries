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

function(_hipblaslt_python_command output python rocm_root asan_options)
    set(_command
        "${CMAKE_COMMAND}" -E env
        "ROCM_PATH=${rocm_root}"
    )
    if(asan_options)
        list(APPEND _command ${asan_options})
    endif()
    list(APPEND _command -- "${python}")
    set(${output} "${_command}" PARENT_SCOPE)
endfunction()

function(hipblaslt_configure_tensilelite_python mode asan_options)
    if(NOT HIPBLASLT_ENABLE_DEVICE AND NOT TENSILELITE_ENABLE_CLIENT)
        set(HIPBLASLT_PYTHON_COMMAND "${Python3_EXECUTABLE}" PARENT_SCOPE)
        set(HIPBLASLT_PYTHON_DEPS "" PARENT_SCOPE)
        return()
    endif()

    set(_source_root "${CMAKE_CURRENT_SOURCE_DIR}/tensilelite")
    if(ROCM_PATH)
        set(_base_rocm "${ROCM_PATH}")
    elseif(DEFINED ENV{ROCM_PATH})
        set(_base_rocm "$ENV{ROCM_PATH}")
    else()
        set(_base_rocm "/opt/rocm")
    endif()

    if(mode STREQUAL "SYSTEM")
        _hipblaslt_python_command(_python_command "${Python3_EXECUTABLE}" "${_base_rocm}" "${asan_options}")
        execute_process(
            COMMAND ${_python_command} -c "import rocisa, tensilelite"
            RESULT_VARIABLE _import_status
            OUTPUT_VARIABLE _import_stdout
            ERROR_VARIABLE _import_stderr
        )
        if(NOT _import_status EQUAL 0)
            message(FATAL_ERROR
                "SYSTEM TensileLite Python environment validation failed.\n"
                "Python: ${Python3_EXECUTABLE}\nROCm: ${_base_rocm}\n"
                "${_import_stdout}${_import_stderr}")
        endif()
        set(HIPBLASLT_PYTHON_COMMAND "${_python_command}" PARENT_SCOPE)
        set(HIPBLASLT_PYTHON_DEPS "" PARENT_SCOPE)
        set(HIPBLASLT_TENSILELITE_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" PARENT_SCOPE)
        set(HIPBLASLT_TENSILELITE_STAGE "${_base_rocm}" PARENT_SCOPE)
        return()
    endif()

    if(NOT mode STREQUAL "BUILD")
        message(FATAL_ERROR
            "HIPBLASLT_TENSILELITE_PYTHON_MODE must be BUILD or SYSTEM, got: ${mode}")
    endif()
    if(NOT TARGET tensilelite-client)
        message(FATAL_ERROR
            "BUILD Python mode requires tensilelite-client. "
            "Do not disable TENSILELITE_ENABLE_CLIENT when device generation is enabled.")
    endif()

    set(_stage "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-rocm")
    set(_venv "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-venv")
    if(WIN32)
        set(_venv_python "${_venv}/Scripts/python.exe")
    else()
        set(_venv_python "${_venv}/bin/python")
    endif()
    set(_client_stage_dir "${_stage}/${CMAKE_INSTALL_LIBEXECDIR}/hipblaslt/tensilelite")
    set(_runtime_stamp "${_stage}/.python-runtime.stamp")

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import sys; print(repr(list(dict.fromkeys(path for path in sys.path if path and ('site-packages' in path or 'dist-packages' in path)))))"
        OUTPUT_VARIABLE _bootstrap_site_packages
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )

    set(_rocm_version_file "${_base_rocm}/.info/version")
    if(NOT EXISTS "${_rocm_version_file}")
        message(FATAL_ERROR "ROCm version file not found: ${_rocm_version_file}")
    endif()
    file(READ "${_rocm_version_file}" _rocm_version)
    string(STRIP "${_rocm_version}" _rocm_version)
    file(MAKE_DIRECTORY "${_stage}/.info")
    file(WRITE "${_stage}/.info/version" "${_rocm_version}\n")

    set(_toolchain_commands)
    if(WIN32)
        list(APPEND _toolchain_commands
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${_stage}/bin"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${_base_rocm}/bin/amdclang++.exe" "${_stage}/bin/amdclang++.exe"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${_base_rocm}/bin/amdclang.exe" "${_stage}/bin/amdclang.exe"
        )
    else()
        list(APPEND _toolchain_commands
            COMMAND "${CMAKE_COMMAND}" -E create_symlink "${_base_rocm}/bin" "${_stage}/bin"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${_stage}/${CMAKE_INSTALL_LIBDIR}"
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                "${_base_rocm}/${CMAKE_INSTALL_LIBDIR}/llvm"
                "${_stage}/${CMAKE_INSTALL_LIBDIR}/llvm"
            COMMAND "${CMAKE_COMMAND}" -E create_symlink "${_base_rocm}/llvm" "${_stage}/llvm"
            COMMAND "${CMAKE_COMMAND}" -E create_symlink "${_base_rocm}/include" "${_stage}/include"
        )
    endif()
    add_custom_command(
        OUTPUT "${_runtime_stamp}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_client_stage_dir}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:tensilelite-client>"
            "${_client_stage_dir}/$<TARGET_FILE_NAME:tensilelite-client>"
        ${_toolchain_commands}
        COMMAND "${Python3_EXECUTABLE}" -m venv "${_venv}"
        COMMAND "${_venv_python}" -c
            "from pathlib import Path; import site; Path(site.getsitepackages()[0], 'hipblaslt-bootstrap.pth').write_text(chr(10).join(${_bootstrap_site_packages}) + chr(10), encoding='utf-8')"
        COMMAND "${CMAKE_COMMAND}" -E env
            "ROCM_PATH=${_stage}"
            -- "${_venv_python}" -m pip install
            --disable-pip-version-check --no-build-isolation --no-deps
            --config-settings editable_mode=compat
            --editable "${_source_root}"
        COMMAND "${CMAKE_COMMAND}" -E env
            "ROCM_PATH=${_stage}"
            -- "${_venv_python}" -c
            "import filelock, joblib, msgpack, numpy, packaging, yaml, rocisa, tensilelite"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_runtime_stamp}"
        DEPENDS
            tensilelite-client
            "${_source_root}/build_backend.py"
            "${_source_root}/pyproject.toml"
            "${_source_root}/setup.py"
            "${_source_root}/release_metadata.py"
        COMMENT "Staging the TensileLite client and private Python environment"
        VERBATIM
        USES_TERMINAL
    )
    add_custom_target(hipblaslt-tensilelite-python-env ALL DEPENDS "${_runtime_stamp}")

    _hipblaslt_python_command(_python_command "${_venv_python}" "${_stage}" "${asan_options}")
    set(HIPBLASLT_PYTHON_COMMAND "${_python_command}" PARENT_SCOPE)
    set(HIPBLASLT_PYTHON_DEPS "hipblaslt-tensilelite-python-env" PARENT_SCOPE)
    set(HIPBLASLT_TENSILELITE_STAGE "${_stage}" PARENT_SCOPE)
    set(HIPBLASLT_TENSILELITE_PYTHON_EXECUTABLE "${_venv_python}" PARENT_SCOPE)
endfunction()
