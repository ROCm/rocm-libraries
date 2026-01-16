# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

macro(hipblaslt_find_python python_dev_component)
    find_package(Python3 3.8 COMPONENTS Interpreter ${python_dev_component} REQUIRED)
    set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
    find_package(Python 3.8 COMPONENTS Interpreter ${python_dev_component} REQUIRED)
    if(NOT "${Python_EXECUTABLE}" STREQUAL "${Python3_EXECUTABLE}")
        message(WARNING "FindPython and FindPython3 found different executables. You may need to pin -DPython_EXECUTABLE and -DPython3_EXECUTABLE (${Python_EXECUTABLE} vs ${Python3_EXECUTABLE})")
    endif()
endmacro()

# Creates or updates a Python virtual environment with required dependencies.
# This ensures a reproducible build environment isolated from system Python.
#
# Args:
#   venv_dir: Directory where the venv will be created (typically in CMAKE_BINARY_DIR)
#   requirements_file: Path to requirements.txt with package dependencies
#
# Sets in parent scope:
#   HIPBLASLT_VENV_PYTHON: Path to the venv's Python interpreter
#   HIPBLASLT_VENV_SITE_PACKAGES: Path to the venv's site-packages directory
#
function(hipblaslt_setup_python_venv venv_dir requirements_file)
    set(venv_python "${venv_dir}/bin/python3")
    set(venv_pip "${venv_dir}/bin/pip")
    
    if(WIN32)
        set(venv_python "${venv_dir}/Scripts/python.exe")
        set(venv_pip "${venv_dir}/Scripts/pip.exe")
    endif()

    # Create venv if it doesn't exist
    if(NOT EXISTS "${venv_python}")
        message(STATUS "Creating Python virtual environment at ${venv_dir}")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m venv "${venv_dir}"
            RESULT_VARIABLE _venv_result
            ERROR_VARIABLE _venv_error
            OUTPUT_QUIET
        )
        if(_venv_result)
            message(FATAL_ERROR "Failed to create Python virtual environment:\n${_venv_error}")
        endif()
        message(STATUS "Python virtual environment created successfully")
    else()
        message(STATUS "Using existing Python virtual environment at ${venv_dir}")
    endif()

    # Upgrade pip to avoid warnings and ensure compatibility
    message(STATUS "Upgrading pip in virtual environment")
    execute_process(
        COMMAND "${venv_python}" -m pip install --quiet --upgrade pip
        RESULT_VARIABLE _pip_upgrade_result
        ERROR_VARIABLE _pip_upgrade_error
        OUTPUT_QUIET
    )
    if(_pip_upgrade_result)
        message(WARNING "Failed to upgrade pip (non-fatal):\n${_pip_upgrade_error}")
    endif()

    # Install or upgrade requirements
    # Use a stamp file to avoid reinstalling on every configure
    set(requirements_stamp "${CMAKE_BINARY_DIR}/python_requirements.stamp")
    set(requirements_hash "")
    
    if(EXISTS "${requirements_file}")
        file(MD5 "${requirements_file}" requirements_hash)
    endif()
    
    set(needs_install FALSE)
    if(EXISTS "${requirements_stamp}")
        file(READ "${requirements_stamp}" cached_hash)
        if(NOT "${cached_hash}" STREQUAL "${requirements_hash}")
            set(needs_install TRUE)
        endif()
    else()
        set(needs_install TRUE)
    endif()

    if(needs_install)
        message(STATUS "Installing Python dependencies from ${requirements_file}")
        execute_process(
            COMMAND "${venv_python}" -m pip install --quiet -r "${requirements_file}"
            RESULT_VARIABLE _pip_install_result
            ERROR_VARIABLE _pip_install_error
        )
        if(_pip_install_result)
            message(FATAL_ERROR "Failed to install Python requirements:\n${_pip_install_error}")
        endif()
        
        # Write stamp file to avoid reinstalling next time
        file(WRITE "${requirements_stamp}" "${requirements_hash}")
        message(STATUS "Python dependencies installed successfully")
    else()
        message(STATUS "Python dependencies are up to date")
    endif()

    # Get the site-packages directory for later use
    execute_process(
        COMMAND "${venv_python}" -c "import sysconfig; print(sysconfig.get_path('purelib'))"
        OUTPUT_VARIABLE _site_packages
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _site_result
    )
    if(_site_result)
        message(WARNING "Could not determine venv site-packages directory")
        set(_site_packages "${venv_dir}/lib/python3/site-packages")
    endif()

    # Export to parent scope
    set(HIPBLASLT_VENV_PYTHON "${venv_python}" PARENT_SCOPE)
    set(HIPBLASLT_VENV_SITE_PACKAGES "${_site_packages}" PARENT_SCOPE)
    
    message(VERBOSE "Venv Python: ${venv_python}")
    message(VERBOSE "Venv site-packages: ${_site_packages}")
endfunction()

# Sets the HIPBLASLT_PYTHON_COMMAND variable in the parent scope such that it
# can invoke the Python interpreter valid for the build parameters. Because
# this may involve a multi token list, it must be used without quotes in
# COMMAND lists.
#
# When using bundled Python dependencies, this will use the venv Python and
# set up PYTHONPATH to include both the venv site-packages and any additional
# build directories (like rocisa).
function(hipblaslt_configure_bundled_python_command python_binary_dir asan_options)
    # Set up a python command which sets PYTHONPATH and copies the current
    # PATH to the build time invocation, invoking python with the -P option
    # to enable additional environment protections.
    if(WIN32)
        set(_ds "$<SEMICOLON>")
    else()
        set(_ds ":")
    endif()
    
    # Determine which Python to use
    if(DEFINED HIPBLASLT_VENV_PYTHON)
        set(_python_exe "${HIPBLASLT_VENV_PYTHON}")
        message(STATUS "Using venv Python: ${_python_exe}")
        
        # When using venv, we still need to add rocisa and tensilelite to PYTHONPATH
        # because they're built into the build directory, not installed into the venv
        set(_python_path
            # TODO: Order is important because the tensilelite directory incorrectly
            # contains a "rocisa" directory which could be incorrectly inferred
            # to be a namespace package. That tree should be re-organized to have
            # a discrete python src dir.
            "${python_binary_dir}"
            # TODO: This should not need to traverse to the parent directory once
            # moved to the root.
            "${hipblaslt_SOURCE_DIR}/tensilelite"
        )
    else
        set(_python_exe "${Python3_EXECUTABLE}")
        set(_python_path
            "${python_binary_dir}"
            "${hipblaslt_SOURCE_DIR}/tensilelite"
        )
    endif()
    
    list(JOIN _python_path "${_ds}" _python_path)

    # Capture the configure time path so that the build environment is always
    # fixed to what we saw at configure time.
    set(_path "$ENV{PATH}")
    if(WIN32)
        string(REPLACE ";" "${_ds}" _path "${_path}")
    endif()
    set(_python_command
        "${CMAKE_COMMAND}" -E env
        "PYTHONPATH=${_python_path}"
        "PATH=${_path}"
        "${asan_options}"
        --
        "${_python_exe}"
    )
    message(VERBOSE "Python command: ${_python_command}")
    set(HIPBLASLT_PYTHON_COMMAND "${_python_command}" PARENT_SCOPE)
endfunction()
