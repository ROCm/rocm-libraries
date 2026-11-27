# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# Builds the canonical wheel from METADATA (default: this build's refreshed
# TENSILELITE_METADATA_FILE). A separately configured consumer passes the
# producer's installed metadata so both wheels carry the same identity.
function(hipblaslt_add_tensilelite_release_wheel source_root)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "METADATA" "")
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "hipblaslt_add_tensilelite_release_wheel: invalid arguments: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT arg_METADATA)
        set(arg_METADATA "${TENSILELITE_METADATA_FILE}")
    endif()
    set(_metadata_dependencies "${arg_METADATA}")
    if(TARGET tensilelite-metadata)
        list(PREPEND _metadata_dependencies tensilelite-metadata)
    endif()
    set(_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-release-wheels")
    set(_stamp "${_wheel_dir}/.published")
    file(GLOB_RECURSE _sources CONFIGURE_DEPENDS
        "${source_root}/tensilelite/*.py"
        "${source_root}/tensilelite/CustomKernels/*.s"
        "${source_root}/tensilelite/Source/*.h"
        "${source_root}/tensilelite/logic/*.yaml"
        "${source_root}/tensilelite/ductile/config/*.yaml")
    set(_inputs ${_sources}
        "${source_root}/_tensilelite_client_binding.py"
        "${source_root}/tensilelite_configure_client.py"
        "${source_root}/VERSION" "${source_root}/pyproject.toml"
        "${source_root}/setup.py" "${source_root}/MANIFEST.in"
        "${source_root}/release_metadata.py" "${source_root}/requirements.txt"
        "${source_root}/scripts/build_release_wheel.py"
        "${source_root}/scripts/check_release_wheel_contents.py"
        "${source_root}/scripts/stage_release_source.py"
        "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
    set(_inputs_file "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-wheel-inputs.txt")
    list(JOIN _inputs "\n" _input_lines)
    file(CONFIGURE OUTPUT "${_inputs_file}" CONTENT "${_input_lines}\n" @ONLY)
    # Check publication on each requested build: the wheel's dynamic filename
    # cannot be a fixed byproduct, and a surviving manifest must not hide its loss.
    add_custom_target(tensilelite-canonical-release-wheel
        COMMAND "${Python3_EXECUTABLE}" -B "${source_root}/scripts/build_release_wheel.py"
            --source "${source_root}" --metadata "${arg_METADATA}"
            --inputs "${_inputs_file}"
            --stage "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-wheel-canonical"
            --destination "${_wheel_dir}"
        BYPRODUCTS "${_stamp}"
        DEPENDS ${_metadata_dependencies} "${_inputs_file}" ${_inputs}
        COMMENT "Checking the TensileLite release wheel against refreshed metadata"
        VERBATIM USES_TERMINAL)
    add_custom_target(tensilelite-build-release-wheels DEPENDS tensilelite-canonical-release-wheel)
    set(TENSILELITE_CANONICAL_WHEEL_MANIFEST "${_stamp}" PARENT_SCOPE)
endfunction()

# Installs the wheel the build generated with into the test artifact: the
# explicit TENSILELITE_GENERATOR_WHEEL when supplied, otherwise the canonical
# wheel built by hipblaslt_add_tensilelite_release_wheel().
function(hipblaslt_install_tensilelite_test_wheel source_root destination)
    if(TENSILELITE_GENERATOR_WHEEL)
        set(_wheel "${TENSILELITE_GENERATOR_WHEEL}")
        cmake_path(ABSOLUTE_PATH _wheel NORMALIZE)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -B
                "${source_root}/scripts/prepare_generator.py"
                --wheel "${_wheel}" --metadata "${TENSILELITE_METADATA_FILE}" --check-only
            COMMAND_ERROR_IS_FATAL ANY)
        set(_selection "\"${_wheel}\"")
    elseif(TENSILELITE_CANONICAL_WHEEL_MANIFEST)
        set(_selection "--manifest \"${TENSILELITE_CANONICAL_WHEEL_MANIFEST}\"")
    else()
        message(FATAL_ERROR "No TensileLite wheel is available for the test artifact")
    endif()
    # TheRock runs pytest with this artifact root on PYTHONPATH. Extract the
    # validated pure-Python wheel so its packages are importable there while
    # retaining the wheel file for installation tests.
    install(CODE "
        execute_process(
            COMMAND \"${Python3_EXECUTABLE}\"
                \"${source_root}/scripts/install_release_wheels.py\"
                --destination \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${destination}\"
                ${_selection}
                --copy-wheels \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${destination}/wheels\"
            RESULT_VARIABLE _tensilelite_wheel_install_result
        )
        if(NOT _tensilelite_wheel_install_result EQUAL 0)
            message(FATAL_ERROR \"Failed to install TensileLite wheels into the test artifact\")
        endif()
    " COMPONENT tests)
endfunction()
