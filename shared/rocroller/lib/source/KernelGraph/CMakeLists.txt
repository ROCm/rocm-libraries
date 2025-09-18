# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/ControlFlowArgumentTracer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ControlFlowRWTracer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ControlToCoordinateMapper.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/GlobalConstraints.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelGraph.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelGraph_serialization.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LastRWTracer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerFromCommand.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RegisterTagManager.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Reindexer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ScopeManager.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp"
)

add_subdirectory(Transformations)
add_subdirectory(CoordinateGraph)
add_subdirectory(ControlGraph)
