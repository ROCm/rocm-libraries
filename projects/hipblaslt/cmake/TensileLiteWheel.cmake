# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

function(hipblaslt_add_tensilelite_release_wheel source_root)
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
            --source "${source_root}" --metadata "${TENSILELITE_METADATA_FILE}"
            --inputs "${_inputs_file}"
            --stage "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-wheel-canonical"
            --destination "${_wheel_dir}"
        BYPRODUCTS "${_stamp}"
        DEPENDS tensilelite-metadata "${TENSILELITE_METADATA_FILE}" "${_inputs_file}" ${_inputs}
        COMMENT "Checking the TensileLite release wheel against refreshed metadata"
        VERBATIM USES_TERMINAL)
    add_custom_target(tensilelite-build-release-wheels DEPENDS tensilelite-canonical-release-wheel)
    set(TENSILELITE_CANONICAL_WHEEL_MANIFEST "${_stamp}" PARENT_SCOPE)
endfunction()
