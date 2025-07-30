# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Sets the HIPBLASLT_PYTHON_COMMAND variable in the parent scope such that it
# can invoke the Python interpreter valid for the build parameters. Because
# this may involve a multi token list, it must be used without quotes in
# COMMAND lists.
function(hipblaslt_configure_bundled_python_command python_binary_dir asan_options)
    # Set up a python command which sets PYTHONPATH and copies the current
    # PATH to the build time invocation, invoking python with the -P option
    # to enable additional environment protections.
    if(WIN32)
        set(_ds "$<SEMICOLON>")
    else()
        set(_ds ":")
    endif()
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
        "${Python_EXECUTABLE}"
    )
    message(VERBOSE "Python command: ${_python_command}")
    set(HIPBLASLT_PYTHON_COMMAND "${_python_command}" PARENT_SCOPE)
endfunction()
