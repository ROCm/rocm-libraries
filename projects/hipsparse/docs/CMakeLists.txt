# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(ROCMDoxygenDoc)

rocm_add_doxygen_doc(
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html"
  DESTINATION "${CMAKE_INSTALL_DOCDIR}")
