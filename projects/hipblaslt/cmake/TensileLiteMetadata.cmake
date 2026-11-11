# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# Shared by the native client and canonical wheel. The initial resolution makes
# configure-time diagnostics available; the target refreshes it on every build.
function(hipblaslt_configure_tensilelite_metadata source_root output_dir)
    set(_generator "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenerateTensileLiteMetadata.cmake")
    set(_command "${CMAKE_COMMAND}"
        "-DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
        "-DSOURCE_ROOT=${source_root}"
        "-DOUTPUT_DIR=${output_dir}"
        "-DBUILD_ROCM_ROOT=${HIPBLASLT_BUILD_ROCM_ROOT}"
        "-DSOURCE_REVISION_OVERRIDE=${TENSILELITE_SOURCE_REVISION}"
        "-DROCM_IDENTITY=${HIPBLASLT_BUILD_ROCM_VERSION}"
        -P "${_generator}")
    execute_process(COMMAND ${_command} COMMAND_ERROR_IS_FATAL ANY)
    set(TENSILELITE_METADATA_FILE "${output_dir}/metadata.json" PARENT_SCOPE)
    file(READ "${output_dir}/metadata.json" _metadata)
    foreach(_pair IN ITEMS
            "distribution_version;TENSILELITE_DISTRIBUTION_VERSION"
            "source_revision;TENSILELITE_CLIENT_SOURCE_REVISION"
            "build_rocm;TENSILELITE_CLIENT_BUILD_ROCM")
        list(GET _pair 0 _key)
        list(GET _pair 1 _variable)
        string(JSON _value GET "${_metadata}" "${_key}")
        set(${_variable} "${_value}" PARENT_SCOPE)
    endforeach()
    add_custom_target(tensilelite-metadata
        COMMAND ${_command}
        BYPRODUCTS "${output_dir}/metadata.json" "${output_dir}/TensileLiteClientVersion.hpp"
        COMMENT "Refreshing TensileLite source identity"
        VERBATIM)
endfunction()

function(hipblaslt_attach_tensilelite_metadata target)
    get_filename_component(_metadata_dir "${TENSILELITE_METADATA_FILE}" DIRECTORY)
    add_dependencies(${target} tensilelite-metadata)
    target_include_directories(${target} PRIVATE "${_metadata_dir}")
    target_sources(${target} PRIVATE "${_metadata_dir}/TensileLiteClientVersion.hpp")
endfunction()
