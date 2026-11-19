# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Reads the ex_gpu_* labels the shared test-category parser puts on an enabled test.
#
# Reads MIOPEN_TEST_CATEGORIES_YAML, MIOPEN_TEST_CATEGORIES_PARSER and Python3_EXECUTABLE, sets
# MIOPEN_FORWARDING_PARITY_GPU_LABELS. Its own file so a `cmake -P` driver can exercise it
# against a fixture YAML: getting this wrong drops parity coverage on an architecture without
# failing anything, so it needs a test cheaper than a configure.
set(MIOPEN_FORWARDING_PARITY_GPU_LABELS "")
# A tree with no test_categories.yaml is supported and has no exclusion sets, so an empty list is
# the right answer there.
if(EXISTS "${MIOPEN_TEST_CATEGORIES_YAML}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/forwarding_parity_gpu_labels.py"
            "${MIOPEN_TEST_CATEGORIES_PARSER}" "${MIOPEN_TEST_CATEGORIES_YAML}"
    OUTPUT_VARIABLE MIOPEN_FORWARDING_PARITY_GPU_LABELS
    ERROR_VARIABLE MIOPEN_EX_GPU_LABEL_ERROR
    RESULT_VARIABLE MIOPEN_EX_GPU_LABEL_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  # Fatal, although the category registration only warns on the same failure: carrying on would
  # register only the bare entry and drop every per-architecture entry without failing anything.
  if(NOT MIOPEN_EX_GPU_LABEL_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Could not read the ex_gpu_* labels from ${MIOPEN_TEST_CATEGORIES_YAML} with "
      "${MIOPEN_TEST_CATEGORIES_PARSER}: ${MIOPEN_EX_GPU_LABEL_ERROR}")
  endif()
  unset(MIOPEN_EX_GPU_LABEL_ERROR)
  unset(MIOPEN_EX_GPU_LABEL_RESULT)
endif()
