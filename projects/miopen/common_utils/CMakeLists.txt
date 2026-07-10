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

# Exception to the "not installed" rule above (headers only; the target stays
# build-only — no install(TARGETS)/export). Many installed MIOpen C++ headers
# (src/include/miopen/*.hpp) include <common_utils/*.hpp>, so these headers are
# part of the installed public surface transitively. Install the whole directory
# so external consumers building against the install tree (e.g. ROCm/MIFin) can
# resolve them. Gated on the same option that installs MIOpen's C++ headers, so
# the two move together.
if(MIOPEN_INSTALL_CXX_HEADERS)
    # float8.hpp is a forwarding shim into src/kernels/ (not installed), so exclude it
    # from the installed set — its relative include would not resolve externally, and no
    # installed MIOpen header includes it (only build-only miopen_utils/driver/test do).
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/common_utils
            DESTINATION include
            PATTERN "float8.hpp" EXCLUDE)
endif()
