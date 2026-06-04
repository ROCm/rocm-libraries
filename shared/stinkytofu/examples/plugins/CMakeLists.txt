# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Example plugin — built as a shared library for dynamic loading.
# Links stinkytofu normally. At runtime, the host process must have
# libstinkytofu.so loaded with RTLD_GLOBAL before dlopen-ing this plugin.
add_library(stinkytofu-plugin-helloworld SHARED
    HelloWorldPass.cpp
)

target_include_directories(stinkytofu-plugin-helloworld PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(stinkytofu-plugin-helloworld PRIVATE stinkytofu)

set_target_properties(stinkytofu-plugin-helloworld PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
)
