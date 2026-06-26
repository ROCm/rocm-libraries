# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

project(rocke-client LANGUAGES NONE)

include(CTest)

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
