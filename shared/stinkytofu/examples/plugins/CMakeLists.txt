# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# LLVM-style plugin build: MODULE (shared) for dynamic loading,
# OBJECT for static linking into the host binary.
if(BUILD_SHARED_LIBS)
    add_library(stinkytofu-plugin-helloworld MODULE
        HelloWorldPass.cpp
    )

    target_link_libraries(stinkytofu-plugin-helloworld PRIVATE stinkytofu)

    set_target_properties(stinkytofu-plugin-helloworld PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins"
    )
else()
    add_library(stinkytofu-plugin-helloworld OBJECT
        HelloWorldPass.cpp
    )

    target_link_libraries(stinkytofu-plugin-helloworld PRIVATE stinkytofu)
endif()

target_include_directories(stinkytofu-plugin-helloworld PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
