# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Add ROCm ISA to StinkyTofu conversion sources to rocisa target
target_sources(rocisa PRIVATE AllHwMappings.cpp ToStinkyTofuUtils.cpp)

target_include_directories(rocisa PRIVATE
    "${STINKYTOFU_TOP_DIR}/include"
    "${CMAKE_BINARY_DIR}/tensilelite/rocisa/stinkytofu"
)

target_link_libraries(rocisa PRIVATE stinkytofu)
