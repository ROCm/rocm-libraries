# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# INTERNAL BUILD-ONLY library — not installed, not exported, not part of the public MIOpen API.
# Header-only pure C++ utilities shared by MIOpen, MIOpenDriver, and tests.
# Contains NO MIOpen or GPU dependencies.
# Do NOT add install(TARGETS miopen_common_utils ...) — headers live in the build tree only.

add_library(miopen_common_utils INTERFACE)
add_library(MIOpen::common_utils ALIAS miopen_common_utils)
set_target_properties(miopen_common_utils PROPERTIES EXCLUDE_FROM_ALL TRUE)

target_include_directories(miopen_common_utils INTERFACE
    # BUILD_INTERFACE only — no install interface; these headers are not installed.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# bfloat16.hpp needs MIOPEN_USE_RNE_BFLOAT16 at compile time. It no longer pulls
# miopen/config.h, so the value is propagated as an INTERFACE compile definition.
# The option is declared in the top-level CMakeLists.txt.
if(MIOPEN_USE_RNE_BFLOAT16)
    target_compile_definitions(miopen_common_utils INTERFACE MIOPEN_USE_RNE_BFLOAT16=1)
else()
    target_compile_definitions(miopen_common_utils INTERFACE MIOPEN_USE_RNE_BFLOAT16=0)
endif()
