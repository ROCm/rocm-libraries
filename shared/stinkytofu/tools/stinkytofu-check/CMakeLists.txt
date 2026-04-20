# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(stinkytofu-check
    stinkytofu-check.cpp
)

# stinkytofu-check needs exceptions for std::regex
target_compile_options(stinkytofu-check PRIVATE -fexceptions -frtti)

install(TARGETS stinkytofu-check
    RUNTIME DESTINATION bin
    COMPONENT tools
)
