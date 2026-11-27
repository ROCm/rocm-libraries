# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/hipblaslt_codegen.cmake")

# Prepare one explicitly selected generator. The target owns its command and
# dependencies; consumers use tensilelite_generator_command() or GENERATOR in
# create_device_library(), rather than reading its properties or private paths.
function(tensilelite_prepare_generator)
    cmake_parse_arguments(PARSE_ARGV 0 arg "ASAN;TSAN"
        "TARGET;WHEEL;WHEEL_MANIFEST;METADATA;PYTHON_EXECUTABLE"
        "ENVIRONMENT;TOOL_BIN_DIRS")
    if(arg_UNPARSED_ARGUMENTS OR arg_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "tensilelite_prepare_generator: invalid arguments: ${arg_UNPARSED_ARGUMENTS};${arg_KEYWORDS_MISSING_VALUES}")
    endif()
    if(NOT arg_TARGET OR TARGET "${arg_TARGET}")
        message(FATAL_ERROR "tensilelite_prepare_generator requires a new TARGET")
    endif()
    if((arg_WHEEL AND arg_WHEEL_MANIFEST) OR (NOT arg_WHEEL AND NOT arg_WHEEL_MANIFEST))
        message(FATAL_ERROR "Select exactly one WHEEL or WHEEL_MANIFEST")
    endif()
    if(arg_ASAN AND arg_TSAN)
        message(FATAL_ERROR "ASAN and TSAN are mutually exclusive")
    endif()
    if(NOT arg_PYTHON_EXECUTABLE)
        find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
        set(arg_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
    endif()

    set(_native_dependencies "")
    set(_runtime_dirs "")
    if(TARGET _rocisa)
        set(_native_python "$<TARGET_FILE_DIR:_rocisa>/..")
        list(APPEND _native_dependencies _rocisa)
        list(APPEND _runtime_dirs "$<TARGET_FILE_DIR:_rocisa>")
        if(TARGET stinkytofu::stinkytofu)
            list(APPEND _native_dependencies stinkytofu::stinkytofu)
            list(APPEND _runtime_dirs "$<TARGET_FILE_DIR:stinkytofu::stinkytofu>")
        endif()
        # configure_file stages these wrappers without relinking the extension.
        # A wrapper edit must still invalidate generated kernels.
        get_target_property(_native_source _rocisa SOURCE_DIR)
        file(GLOB _wrappers CONFIGURE_DEPENDS
            "${_native_source}/rocisa/*.py" "${_native_source}/rocisa/*.py.in")
        list(APPEND _native_dependencies ${_wrappers})
        if(NOT arg_METADATA)
            set(arg_METADATA "${TENSILELITE_METADATA_FILE}")
        endif()
    else()
        # Installed by the native producer. All its paths are relative to the
        # installation prefix; no source or producer build directory is needed.
        include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/tensilelite-generator-support.cmake")
        if(NOT arg_METADATA)
            set(arg_METADATA "${_native_metadata}")
        endif()
    endif()
    if(NOT EXISTS "${arg_METADATA}")
        message(FATAL_ERROR "Generator native identity metadata is missing: ${arg_METADATA}")
    endif()
    if(TARGET roc::origami)
        list(APPEND _native_dependencies roc::origami)
        list(APPEND _runtime_dirs "$<TARGET_FILE_DIR:roc::origami>")
    endif()

    set(_scripts "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../scripts")
    if(EXISTS "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/generator-scripts")
        set(_scripts "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/generator-scripts")
    endif()
    set(_prepare "${_scripts}/prepare_generator.py")
    set(_selection --metadata "${arg_METADATA}")
    if(arg_WHEEL)
        cmake_path(ABSOLUTE_PATH arg_WHEEL NORMALIZE)
        list(APPEND _selection --wheel "${arg_WHEEL}")
        set(_artifact "${arg_WHEEL}")
        execute_process(COMMAND "${arg_PYTHON_EXECUTABLE}" -B "${_prepare}"
            ${_selection} --check-only COMMAND_ERROR_IS_FATAL ANY)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${arg_WHEEL}" "${arg_METADATA}")
    else()
        list(APPEND _selection --manifest "${arg_WHEEL_MANIFEST}")
        set(_artifact "${arg_WHEEL_MANIFEST}")
    endif()
    set(_root "${CMAKE_CURRENT_BINARY_DIR}/${arg_TARGET}/python")
    set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/${arg_TARGET}/unpacked.stamp")
    add_custom_command(OUTPUT "${_stamp}"
        COMMAND "${arg_PYTHON_EXECUTABLE}" -B "${_prepare}" ${_selection}
            --destination "${_root}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
        DEPENDS "${_artifact}" "${arg_METADATA}" "${_prepare}"
            "${_scripts}/check_release_wheel_contents.py"
            "${_scripts}/install_release_wheels.py"
        COMMENT "Preparing the packaged TensileLite generator ${arg_TARGET}"
        VERBATIM)
    add_custom_target(${arg_TARGET} DEPENDS "${_stamp}" ${_native_dependencies})
    set(_sanitizer "")
    if(arg_ASAN)
        set(_sanitizer ASAN)
    elseif(arg_TSAN)
        set(_sanitizer TSAN)
    endif()
    hipblaslt_make_python_command(_command
        PYTHON_EXECUTABLE "${arg_PYTHON_EXECUTABLE}"
        PYTHONPATH_DIRS "${_scripts}/generator-bootstrap" "${_root}" "${_native_python}"
        RUNTIME_LIB_DIRS ${_runtime_dirs}
        TOOL_BIN_DIRS ${arg_TOOL_BIN_DIRS}
        ${_sanitizer})
    list(APPEND arg_ENVIRONMENT
        "TENSILELITE_GENERATOR_PYTHON_ROOT=${_root}"
        "TENSILELITE_GENERATOR_NATIVE_ROOT=${_native_python}"
        "PYTHONDONTWRITEBYTECODE=1")
    # Nest env commands to retain the common path/sanitizer handling while
    # carrying the caller's selected ROCm adapter and version.
    if(arg_ENVIRONMENT)
        set(_command "${CMAKE_COMMAND}" -E env ${arg_ENVIRONMENT} -- ${_command})
    endif()
    set_target_properties(${arg_TARGET} PROPERTIES
        TENSILELITE_GENERATOR_COMMAND "${_command}"
        TENSILELITE_GENERATOR_INPUTS "${_stamp};${_native_dependencies};${_scripts}/generator-bootstrap/sitecustomize.py")
endfunction()

function(tensilelite_generator_command target out_command out_dependencies)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Unknown TensileLite generator target: ${target}")
    endif()
    get_target_property(_command "${target}" TENSILELITE_GENERATOR_COMMAND)
    get_target_property(_inputs "${target}" TENSILELITE_GENERATOR_INPUTS)
    if(NOT _command OR NOT _inputs)
        message(FATAL_ERROR "Target is not a prepared TensileLite generator: ${target}")
    endif()
    set(${out_command} "${_command}" PARENT_SCOPE)
    set(${out_dependencies} "${target};${_inputs}" PARENT_SCOPE)
endfunction()
