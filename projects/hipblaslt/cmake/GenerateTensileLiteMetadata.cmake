# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Resolve once per build. configure_file and file(CONFIGURE) preserve timestamps
# when contents agree, so an unchanged identity does not recompile the client.
set(_python_command "${PYTHON_EXECUTABLE}")
if(SOURCE_REVISION_OVERRIDE)
    set(_python_command "${CMAKE_COMMAND}" -E env
        "TENSILELITE_SOURCE_REVISION=${SOURCE_REVISION_OVERRIDE}" "${PYTHON_EXECUTABLE}")
endif()
set(_identity_args)
if(ROCM_IDENTITY)
    set(_identity_args --rocm-identity "${ROCM_IDENTITY}")
endif()
execute_process(
    COMMAND ${_python_command} "${SOURCE_ROOT}/release_metadata.py" metadata ${_identity_args}
    OUTPUT_VARIABLE _metadata
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
string(JSON TENSILELITE_DISTRIBUTION_VERSION GET "${_metadata}" distribution_version)
string(JSON TENSILELITE_CLIENT_SOURCE_REVISION GET "${_metadata}" source_revision)
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "${SOURCE_ROOT}/tensilelite/_build_rocm.py" "${BUILD_ROCM_ROOT}"
    OUTPUT_VARIABLE TENSILELITE_CLIENT_BUILD_ROCM
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
string(JSON _metadata SET "${_metadata}" build_rocm "\"${TENSILELITE_CLIENT_BUILD_ROCM}\"")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
configure_file("${SOURCE_ROOT}/client/TensileLiteClientVersion.hpp.in"
    "${OUTPUT_DIR}/TensileLiteClientVersion.hpp" @ONLY)
file(CONFIGURE OUTPUT "${OUTPUT_DIR}/metadata.json" CONTENT "${_metadata}\n" @ONLY)
