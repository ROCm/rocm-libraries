# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(tensilelite-host
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/AnalyticalGemm.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Hardware.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/StreamK.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utils.hpp"
)
