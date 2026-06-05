# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Add backend sources to the stinkytofu target
target_sources(stinkytofu PRIVATE
    Backend.cpp
    BackendRegistry.cpp
    Gfx1250Backend.cpp
)
