# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/LoadOption.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/StreamK.cpp"
)
