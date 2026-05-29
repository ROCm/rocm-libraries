# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_program(HIPSPARSE_MTX2CSR hipsparse_mtx2csr PATHS "/opt/rocm/bin" "${ROCM_PATH}/bin")

if(NOT CMAKE_MATRICES_DIR)
  set(CMAKE_MATRICES_DIR "${PROJECT_BINARY_DIR}/clients/matrices")
  message(WARNING "Unspecified CMAKE_MATRICES_DIR, the default value of CMAKE_MATRICES_DIR is set to '${CMAKE_MATRICES_DIR}'")
endif()

# convert relative path to absolute
get_filename_component(PROJECT_BINARY_DIR "${PROJECT_BINARY_DIR}"
                       ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
get_filename_component(CMAKE_MATRICES_DIR "${CMAKE_MATRICES_DIR}"
                       ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}")

include("${CMAKE_CURRENT_LIST_DIR}/hipsparse_test_matrices_common.cmake")

hipsparse_prepare_test_matrices("${HIPSPARSE_MTX2CSR}" "${HIPSPARSE_MTX2CSR}")
