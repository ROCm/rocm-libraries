# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(ROCMSphinxDoc)

file(COPY
  "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

rocm_add_sphinx_doc(
    "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_DIR html
  BUILDER html
  USES_DOXYGEN
  VARS
    "doxyfile=${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
)
