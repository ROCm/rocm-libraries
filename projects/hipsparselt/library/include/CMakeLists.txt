# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(GenerateExportHeader)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparselt-version.h.in"
    "${PROJECT_BINARY_DIR}/include/hipsparselt/hipsparselt-version.h"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparselt.h"
    "${PROJECT_BINARY_DIR}/include/hipsparselt/hipsparselt.h"
    COPYONLY
)

generate_export_header(hipsparselt
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/hipsparselt/hipsparselt-export.h"
)
