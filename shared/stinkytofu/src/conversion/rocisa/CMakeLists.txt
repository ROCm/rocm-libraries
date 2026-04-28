# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Add ROCm ISA to StinkyTofu conversion sources to rocisa target
target_sources(_rocisa PRIVATE AllHwMappings.cpp ToStinkyTofuUtils.cpp)

target_include_directories(_rocisa PRIVATE
    "${STINKYTOFU_TOP_DIR}/include"
    "${CMAKE_BINARY_DIR}/tensilelite/rocisa/stinkytofu"
)

target_link_libraries(_rocisa PRIVATE stinkytofu)
