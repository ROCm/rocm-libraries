# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Origami is bundled with hipBLASLt, not distributed as an independent ROCm
# package. Resolve the co-installed config even when the caller supplies only
# hipblaslt_DIR instead of the whole prefix in CMAKE_PREFIX_PATH.
find_dependency(origami CONFIG
    PATHS "${CMAKE_CURRENT_LIST_DIR}/../origami"
          "${CMAKE_CURRENT_LIST_DIR}/origami"
    NO_DEFAULT_PATH)

# Codegen is an explicit capability. Consumers opt in with
# include(hipblaslt_codegen) after find_package(hipblaslt).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
