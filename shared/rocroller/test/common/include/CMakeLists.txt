# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(common-test-utilities
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/common/CommonGraphs.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/CommonGraphs_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/F8Values.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/GEMMProblem.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/mxDataGen.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/Scheduling.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/SourceMatcher.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/TestValues.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/Utilities.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/WidenAddrExprTo64bit.hpp"
)
