# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(tensilelite-host
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/YAML.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Loading.cpp"
)
