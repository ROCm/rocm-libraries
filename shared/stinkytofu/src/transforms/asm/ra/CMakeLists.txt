# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    PhysRegMatrix.cpp
    AllocationConstraints.cpp
    AllocationScope.cpp
    AllocationVerifier.cpp
    AllocatorRegistry.cpp
    LegacyColoring.cpp
    RegisterAllocationPass.cpp
    RegisterSymbolSync.cpp
    allocators/LegacyIdentityAllocator.cpp
    allocators/GreedyAllocator.cpp
)
