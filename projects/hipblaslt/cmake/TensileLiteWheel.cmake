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
    add_custom_command(OUTPUT "${_stamp}"
        COMMAND "${Python3_EXECUTABLE}" -B "${source_root}/scripts/build_release_wheel.py"
            --source "${source_root}" --metadata "${TENSILELITE_METADATA_FILE}"
            --stage "${CMAKE_CURRENT_BINARY_DIR}/tensilelite-wheel-canonical"
            --destination "${_wheel_dir}"
        DEPENDS tensilelite-metadata "${TENSILELITE_METADATA_FILE}" ${_sources}
            "${source_root}/_tensilelite_client_binding.py"
            "${source_root}/tensilelite_configure_client.py"
            "${source_root}/VERSION" "${source_root}/pyproject.toml"
            "${source_root}/setup.py" "${source_root}/MANIFEST.in"
            "${source_root}/release_metadata.py" "${source_root}/requirements.txt"
            "${source_root}/scripts/build_release_wheel.py"
            "${source_root}/scripts/check_release_wheel_contents.py"
            "${source_root}/scripts/stage_release_source.py"
        COMMENT "Building the TensileLite release wheel from refreshed metadata"
        VERBATIM USES_TERMINAL)
    add_custom_target(tensilelite-canonical-release-wheel DEPENDS "${_stamp}")
    add_dependencies(tensilelite-canonical-release-wheel tensilelite-metadata)
    add_custom_target(tensilelite-build-release-wheels DEPENDS tensilelite-canonical-release-wheel)
    set(TENSILELITE_CANONICAL_WHEEL_MANIFEST "${_stamp}" PARENT_SCOPE)
endfunction()
