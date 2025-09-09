# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/BlockScale.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CommandArgument.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Command.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Operation.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RandomNumberGenerator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorScalar.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/T_LoadStore.cpp"
)
