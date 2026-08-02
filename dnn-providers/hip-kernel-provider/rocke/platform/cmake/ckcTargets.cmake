# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Compatibility entry point for consumers that included the previously installed
# target export directly. New consumers should use find_package(ckc CONFIG
# REQUIRED).
include(CMakeFindDependencyMacro)
find_dependency(Threads)

include("${CMAKE_CURRENT_LIST_DIR}/ckcTargetsImpl.cmake")
