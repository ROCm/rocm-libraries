# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(hipsparselt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/hipsparselt.cpp")

add_subdirectory(rocsparselt)
