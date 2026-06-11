# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# mosaic hipBLASLt backend: header-only glue that converts hipBLASLt/TensileLite
# GEMM structs (origami's problem_t/config_t/hardware_t) into mosaic's neutral
# types and calls the engine. It is the ONLY mosaic component that references
# framework (origami) headers, so it is gated OFF for standalone mosaic builds
# and enabled when mosaic is built inside hipBLASLt (where roc::origami-headers
# exists).

add_library(mosaic-hipblaslt INTERFACE)
add_library(roc::mosaic-hipblaslt ALIAS mosaic-hipblaslt)

target_compile_features(mosaic-hipblaslt INTERFACE cxx_std_17)

target_include_directories(mosaic-hipblaslt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# The engine (roc::mosaic) plus origami's header-only types it converts from.
target_link_libraries(mosaic-hipblaslt INTERFACE roc::mosaic)
if(TARGET roc::origami-headers)
    target_link_libraries(mosaic-hipblaslt INTERFACE roc::origami-headers)
endif()

if(MOSAIC_ENABLE_INSTALL)
    install(
        DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.hpp"
    )
endif()
