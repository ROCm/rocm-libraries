# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_fixture_target "hipblaslt-bespoke-fixture-${_fixture_arch}")
set(_fixture_output "${HIPBLASLT_BESPOKE_FIXTURE_ROOT}/${_fixture_arch}")
set(_fixture_sanitizer_args "")
if(HIPBLASLT_ENABLE_ASAN
        OR THEROCK_SANITIZER STREQUAL "ASAN"
        OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
    list(APPEND _fixture_sanitizer_args HOST_ASAN)
elseif(HIPBLASLT_ENABLE_TSAN OR THEROCK_SANITIZER STREQUAL "TSAN")
    list(APPEND _fixture_sanitizer_args HOST_TSAN)
endif()
create_device_library(
    TARGET "${_fixture_target}"
    CODEGEN_ROOT "${PROJECT_SOURCE_DIR}/tensilelite"
    LOGIC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${_fixture_arch}"
    OUTPUT_DIR "${_fixture_output}"
    ARCHES ${_fixture_targets}
    LOGIC_FILTER "*"
    YAML_FORMAT "${HIPBLASLT_ENABLE_YAML}"
    LAZY_LOAD "${HIPBLASLT_ENABLE_LAZY_LOAD}"
    ASAN "${HIPBLASLT_ENABLE_ASAN}"
    GEMM_A2A_FUSION "${HIPBLASLT_ENABLE_GEMM_A2A_FUSION}"
    ${_fixture_sanitizer_args}
    KEEP_BUILD_TMP ON
)
set(_fixture_indices "${_fixture_output}/indices.txt")
add_custom_command(
    OUTPUT "${_fixture_indices}"
    COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${CMAKE_CURRENT_SOURCE_DIR}/check_fixture.py"
            "${_fixture_output}" "${CMAKE_CURRENT_BINARY_DIR}/MatchTable.yaml"
            "${_fixture_indices}"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${_fixture_target}.stamp"
            "${CMAKE_CURRENT_SOURCE_DIR}/check_fixture.py"
    VERBATIM
)
add_custom_target("${_fixture_target}-checked" DEPENDS "${_fixture_indices}")
add_dependencies(hipblaslt-test "${_fixture_target}-checked")
install(DIRECTORY "${_fixture_output}/library/"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/bespoke-fixture/${_fixture_arch}/library"
        COMPONENT tests)
install(DIRECTORY "${_fixture_output}/carrier-library/"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/bespoke-fixture/${_fixture_arch}/carrier-library"
        COMPONENT tests)
install(FILES "${_fixture_indices}"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/bespoke-fixture/${_fixture_arch}"
        COMPONENT tests)
