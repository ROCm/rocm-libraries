# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(hip REQUIRED)

# Build formocast as a static library (internal dependency only)
add_library(formocast STATIC)
add_library(formocast::formocast ALIAS formocast)

set_target_properties(formocast PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(formocast PUBLIC cxx_std_17)

add_library(formocast-headers INTERFACE)
add_library(formocast::formocast-headers ALIAS formocast-headers)

target_include_directories(formocast-headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../rocisa>
)

target_sources(formocast-headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/formocast.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/formocast_simulator.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/formocast_utils.hpp>
)

target_link_libraries(formocast PUBLIC formocast::formocast-headers)

target_sources(formocast
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/formocast.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/formocast_simulator.cpp"
)

target_link_libraries(formocast PUBLIC hip::host)
