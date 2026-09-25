# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Reads the ex_gpu_* labels the exclusion sets in a test_categories.yaml declare.
#
# Reads MIOPEN_TEST_CATEGORIES_YAML, sets MIOPEN_FORWARDING_PARITY_GPU_LABELS. Its own file so a
# `cmake -P` driver can exercise it against a fixture YAML: getting this wrong drops parity
# coverage on an architecture without failing anything, so it needs a test cheaper than a
# configure.
#
# Taken from the labels an exclusion set declares, not from its key: a key may carry an OS
# suffix (exclude_gpu_<arch>_linux) that the label it declares does not, so a label rebuilt from
# the key would name something no runner ever selects.
set(MIOPEN_FORWARDING_PARITY_GPU_LABELS "")
# A tree with no test_categories.yaml is supported and has no exclusion sets, so an empty list is
# the right answer there. Guarded because file(STRINGS) on a missing file is fatal.
if(EXISTS "${MIOPEN_TEST_CATEGORIES_YAML}")
  file(STRINGS "${MIOPEN_TEST_CATEGORIES_YAML}" MIOPEN_EX_GPU_LABEL_LINES
    REGEX "^ +- \"?ex_gpu_[A-Za-z0-9_]+\"?$")
  foreach(MIOPEN_EX_GPU_LABEL_LINE IN LISTS MIOPEN_EX_GPU_LABEL_LINES)
    string(REGEX REPLACE "^ +- \"?(ex_gpu_[A-Za-z0-9_]+)\"?$" "\\1" MIOPEN_EX_GPU_LABEL
      "${MIOPEN_EX_GPU_LABEL_LINE}")
    list(APPEND MIOPEN_FORWARDING_PARITY_GPU_LABELS "${MIOPEN_EX_GPU_LABEL}")
  endforeach()
  unset(MIOPEN_EX_GPU_LABEL_LINES)
  unset(MIOPEN_EX_GPU_LABEL_LINE)
  unset(MIOPEN_EX_GPU_LABEL)
  # A file that exists but yields nothing means its format moved away from what the regex above
  # expects. Carrying on would register only the bare entry and drop every per-architecture
  # entry without failing anything.
  if(NOT MIOPEN_FORWARDING_PARITY_GPU_LABELS)
    message(FATAL_ERROR
      "No ex_gpu_* labels found in ${MIOPEN_TEST_CATEGORIES_YAML}. Each exclusion set must "
      "declare its label on its own line, as - \"ex_gpu_<arch>\" under labels:.")
  endif()
  # One architecture's Windows and Linux exclusion sets declare the same label, and mirroring it
  # once per declaration would register two ctest entries under one name.
  list(REMOVE_DUPLICATES MIOPEN_FORWARDING_PARITY_GPU_LABELS)
endif()
