# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/VALUReadDAfterWMMAOrSWMMAC.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/VALUWriteAfterWMMAOrSWMMACRead.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/VALUWriteAfterWMMAOrSWMMACWrite.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/WMMAOrSWMMACReadDAfterWMMA.cpp"
)
