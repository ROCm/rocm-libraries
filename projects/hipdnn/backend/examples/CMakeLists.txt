# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# UHD Selection Example
add_executable(uhd_selection_example uhd_selection_example.cpp)

target_link_libraries(uhd_selection_example
    PRIVATE
        hipdnn_backend_private
        hipdnn_data_sdk
        hipdnn_flatbuffers_sdk
)

target_include_directories(uhd_selection_example
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Install example binary
install(TARGETS uhd_selection_example
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
)
