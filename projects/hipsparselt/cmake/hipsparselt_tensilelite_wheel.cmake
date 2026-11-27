# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# TheRock configures hipBLASLt and hipSPARSELt separately and does not yet hand
# hipBLASLt's canonical TensileLite wheel to hipSPARSELt. Build an equivalent
# wheel here from the shared source and hipBLASLt's installed identity metadata,
# after checking that this source tree and ROCm identity match that producer.
#
# Sets TENSILELITE_CANONICAL_WHEEL_MANIFEST and ${out_identity} (the ROCm
# identity generation is checked against, or empty for a development build).
function(hipsparselt_add_staged_tensilelite_wheel tensilelite_root metadata out_identity)
    file(READ "${metadata}" _producer)
    string(JSON _producer_version GET "${_producer}" distribution_version)
    string(JSON _producer_revision GET "${_producer}" source_revision)
    string(JSON _producer_identity ERROR_VARIABLE _no_identity GET "${_producer}" rocm_identity)
    if(_no_identity)
        set(_producer_identity "")
    endif()

    # TODO: once TheRock forwards THEROCK_PACKAGE_VERSION to hipSPARSELt, remove
    # the fallback to the producer's identity and require it as hipBLASLt does.
    if(THEROCK_PACKAGE_VERSION STREQUAL "git")
        set(_identity "")
    elseif(THEROCK_PACKAGE_VERSION)
        set(_identity "${THEROCK_PACKAGE_VERSION}")
    else()
        set(_identity "${_producer_identity}")
    endif()

    set(_command "${Python3_EXECUTABLE}" -B "${tensilelite_root}/release_metadata.py" metadata)
    if(_identity)
        list(APPEND _command --rocm-identity "${_identity}")
    endif()
    if(TENSILELITE_SOURCE_REVISION)
        set(_command "${CMAKE_COMMAND}" -E env
            "TENSILELITE_SOURCE_REVISION=${TENSILELITE_SOURCE_REVISION}" ${_command})
    endif()
    execute_process(COMMAND ${_command}
        OUTPUT_VARIABLE _local OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    string(JSON _local_version GET "${_local}" distribution_version)
    string(JSON _local_revision GET "${_local}" source_revision)
    if(NOT _local_version STREQUAL _producer_version
            OR NOT _local_revision STREQUAL _producer_revision)
        message(FATAL_ERROR
            "hipSPARSELt's TensileLite does not match the staged hipBLASLt producer.\n"
            "  hipBLASLt: ${_producer_version}, source ${_producer_revision}\n"
            "  hipSPARSELt: ${_local_version}, source ${_local_revision}\n"
            "Build both from the same rocm-libraries checkout and THEROCK_PACKAGE_VERSION.")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${metadata}")

    cmake_path(GET tensilelite_root PARENT_PATH _hipblaslt_source)
    include("${_hipblaslt_source}/cmake/TensileLiteWheel.cmake")
    hipblaslt_add_tensilelite_release_wheel("${tensilelite_root}" METADATA "${metadata}")
    set(TENSILELITE_CANONICAL_WHEEL_MANIFEST "${TENSILELITE_CANONICAL_WHEEL_MANIFEST}" PARENT_SCOPE)
    set(${out_identity} "${_identity}" PARENT_SCOPE)
endfunction()
