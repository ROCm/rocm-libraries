# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/AddInstruction.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ArgumentLoader.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Buffer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CrashKernelGenerator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Instruction.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LoadStoreTileGenerator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerFromKernelGraph.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MemoryInstructions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/SubInstruction.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/WaitCount.cpp"
)

add_subdirectory(Arithmetic)
