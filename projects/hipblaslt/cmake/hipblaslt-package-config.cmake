# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_dependency(origami CONFIG
    PATHS "${CMAKE_CURRENT_LIST_DIR}/../origami"
          "${CMAKE_CURRENT_LIST_DIR}/origami"
    NO_DEFAULT_PATH)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
