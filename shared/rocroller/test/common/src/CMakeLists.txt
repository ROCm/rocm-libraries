# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(common-test-utilities
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/CommonGraphs.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/mxDataGen.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Scheduling.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utilities.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/WidenAddrExprTo64bit.cpp"
)
