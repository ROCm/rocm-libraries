# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# stinkytofu-opt tool
add_executable(stinkytofu-opt
    stinkytofu-opt.cpp
)

target_include_directories(stinkytofu-opt PRIVATE
    "${TABLEGEN_OUTPUT_DIR}"
)

target_link_libraries(stinkytofu-opt PRIVATE
    stinkytofu
)

# Install stinkytofu-opt tool
install(TARGETS stinkytofu-opt
    RUNTIME DESTINATION bin
    COMPONENT tools
)
