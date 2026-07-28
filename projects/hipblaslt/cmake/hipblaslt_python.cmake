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

# Collects the loader settings that have to be pinned into the build time
# environment. The dynamic loader resolves the transitive dependencies of
# anything named in LD_PRELOAD through LD_LIBRARY_PATH, so a build started from
# a shell that does not have it would drop the preload and, with it, the
# sanitizer coverage, without failing.
function(hipblaslt_pinned_loader_environment out_var)
    set(_environment "")
    if(NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
        list(APPEND _environment "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}")
    endif()
    set(${out_var} "${_environment}" PARENT_SCOPE)
endfunction()

# Sets the HIPBLASLT_PYTHON_COMMAND variable in the parent scope such that it
# can invoke the Python interpreter valid for the build parameters. Because
# this may involve a multi token list, it must be used without quotes in
# COMMAND lists.
function(hipblaslt_configure_bundled_python_command python_binary_dir asan_options)
    # Set up a python command which sets PYTHONPATH and copies the current
    # PATH to the build time invocation.
    if(WIN32)
        set(_ds "$<SEMICOLON>")
    else()
        set(_ds ":")
    endif()
    set(_python_path
        "${python_binary_dir}"
        "${hipblaslt_SOURCE_DIR}/tensilelite"
    )
    list(JOIN _python_path "${_ds}" _python_path)

    # Capture the configure time path so that the build environment is always
    # fixed to what we saw at configure time.
    set(_path "$ENV{PATH}")
    if(WIN32)
        string(REPLACE ";" "${_ds}" _path "${_path}")
    endif()
    # Only a sanitized build needs the loader pinned; doing it unconditionally
    # would bake the configure time LD_LIBRARY_PATH into every build.
    set(_loader_environment "")
    if(asan_options)
        hipblaslt_pinned_loader_environment(_loader_environment)
    endif()
    set(_python_command
        "${CMAKE_COMMAND}" -E env
        "PYTHONPATH=${_python_path}"
        "PATH=${_path}"
        ${_loader_environment}
        "${asan_options}"
        --
        "${Python3_EXECUTABLE}"
    )
    message(VERBOSE "Python command: ${_python_command}")
    set(HIPBLASLT_PYTHON_COMMAND "${_python_command}" PARENT_SCOPE)
endfunction()

# Sets HIPBLASLT_PYTHON_COMMAND in the parent scope for builds that use the
# ambient interpreter rather than a bundled one. The sanitizer environment is
# still applied, because the code generation and test data generation steps run
# through this command too and would otherwise be uninstrumented.
function(hipblaslt_configure_python_command sanitizer_options)
    if(sanitizer_options)
        hipblaslt_pinned_loader_environment(_loader_environment)
        set(_python_command
            "${CMAKE_COMMAND}" -E env
            ${_loader_environment}
            "${sanitizer_options}"
            --
            "${Python_EXECUTABLE}"
        )
    else()
        set(_python_command "${Python_EXECUTABLE}")
    endif()
    message(VERBOSE "Python command: ${_python_command}")
    set(HIPBLASLT_PYTHON_COMMAND "${_python_command}" PARENT_SCOPE)
endfunction()
