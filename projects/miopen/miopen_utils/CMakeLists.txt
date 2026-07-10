# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# INTERNAL BUILD-ONLY library — not installed, not exported, not part of the public MIOpen API.
# MIOpen Utilities (Layer 3): above-the-library code shared by MIOpenDriver and the tests
# (TensorDesc RAII wrapper, CPU reference impls, tensor_holder/verify, etc.).
#
# Depends ONLY on common_utils (Layer 1) and the MIOpen public C API (miopen/miopen.h).
# It must NOT include or link any internal miopen/ header (the sole, documented exception is
# tensor_desc.hpp's MIOPEN_BUILD_TESTING-gated bridge to the internal type — see that header).
#
# Header-only, so INTERFACE (not STATIC): there are no .cpp translation units to compile, and a
# STATIC archive only earns its keep with real sources. If the CPU reference impls are ever
# converted to compiled TUs, revisit the STATIC choice then.

add_library(miopen_utils INTERFACE)
add_library(MIOpen::miopen_utils ALIAS miopen_utils)
set_target_properties(miopen_utils PROPERTIES EXCLUDE_FROM_ALL TRUE)

target_include_directories(miopen_utils INTERFACE
    # BUILD_INTERFACE only — these headers are not installed.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Layer-3 depends on Layer-1 only. The MIOpen public API is picked up by the consuming
# targets (driver/tests) that already link MIOpen; miopen_utils itself pulls in only
# common_utils so it stays free of internal library coupling.
target_link_libraries(miopen_utils INTERFACE miopen_common_utils)
