# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

# Standalone entry point for provider-owned rocKE AOT artifacts. No compiled
# targets live here yet; the tree only wires Python-driven custom commands and
# CTest entries, so enabling a C/C++ language would only add toolchain coupling.
project(rocke-client LANGUAGES NONE)

# CTest defines BUILD_TESTING for standalone configures. The hip-kernel-provider
# superbuild may also enable tests through HIPKERNELPROVIDER_ENABLE_TESTS.
include(CTest)

# rocKE-client consumes the sibling rocKE source tree as Python source. Keep this
# override cached so standalone builds and parent builds can point at the same
# rocKE checkout without relying on the developer shell's PYTHONPATH.
set(ROCKE_CLIENT_ROCKE_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../rocKE"
    CACHE PATH "Path to the rocKE source tree used by rocKE-client AOT builds"
)
get_filename_component(ROCKE_CLIENT_ROCKE_SOURCE_DIR
    "${ROCKE_CLIENT_ROCKE_SOURCE_DIR}" ABSOLUTE
)
if(NOT EXISTS "${ROCKE_CLIENT_ROCKE_SOURCE_DIR}/Python/rocke")
    message(FATAL_ERROR
        "ROCKE_CLIENT_ROCKE_SOURCE_DIR must point to a rocKE source tree with Python/rocke: "
        "${ROCKE_CLIENT_ROCKE_SOURCE_DIR}"
    )
endif()

add_subdirectory(aot)
