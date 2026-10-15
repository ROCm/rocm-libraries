# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Build-time UKD/KMD/KDP -> kpack packaging. Provider-internal.

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HkpPackaging)

option(HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES
    "CI-only, non-shipping: register packaging probes (build-tree ctest entries hkp-probe-*)" OFF)

hkp_add_packaging()
