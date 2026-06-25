# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tilewright hipBLASLt backend: header-only glue that converts hipBLASLt/TensileLite
# GEMM structs (origami's problem_t/config_t/hardware_t) into tilewright's neutral
# types and calls the engine. It is the ONLY tilewright component that references
# framework (origami) headers, so it is gated OFF for standalone tilewright builds
# and enabled when tilewright is built inside hipBLASLt (where roc::origami-headers
# exists).

add_library(tilewright-hipblaslt INTERFACE)
add_library(roc::tilewright-hipblaslt ALIAS tilewright-hipblaslt)

target_compile_features(tilewright-hipblaslt INTERFACE cxx_std_17)

target_include_directories(tilewright-hipblaslt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# The engine (roc::tilewright) plus origami's header-only types it converts from.
target_link_libraries(tilewright-hipblaslt INTERFACE roc::tilewright)
if(TARGET roc::origami-headers)
    target_link_libraries(tilewright-hipblaslt INTERFACE roc::origami-headers)
endif()

if(TILEWRIGHT_ENABLE_INSTALL)
    install(
        DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.hpp"
    )
endif()
