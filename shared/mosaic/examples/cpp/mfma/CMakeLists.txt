# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(mfma_layout mfma_layout.cpp)
target_link_libraries(mfma_layout PRIVATE mosaic)
target_compile_definitions(mfma_layout PRIVATE
    MFMA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/matrix_calculator_layouts")

if(MOSAIC_BUILD_HIP_EXAMPLES)
    add_executable(mfma_example mfma_example.cpp)
    target_link_libraries(mfma_example PRIVATE mosaic)
endif()
