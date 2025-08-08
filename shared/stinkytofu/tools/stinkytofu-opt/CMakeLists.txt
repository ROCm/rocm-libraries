# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# stinkytofu-opt tool
add_executable(stinkytofu-opt
    stinkytofu-opt.cpp
)

target_include_directories(stinkytofu-opt PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    "${TABLEGEN_OUTPUT_DIR}"
)

target_link_libraries(stinkytofu-opt PRIVATE
    stinkytofu
)
