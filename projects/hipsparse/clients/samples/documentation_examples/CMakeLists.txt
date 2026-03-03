# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Set output directory for all sample executables
# This variable will be inherited by all subdirectories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")

add_subdirectory(level1)
add_subdirectory(level2)
add_subdirectory(level3)
add_subdirectory(extra)
add_subdirectory(precond)
add_subdirectory(conversion)
add_subdirectory(generic)
