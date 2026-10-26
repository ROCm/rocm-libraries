# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Build-time UKD/KMD/KDP -> kpack packaging. Provider-internal.

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HkpPackaging)

option(HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES
    "CI-only, non-shipping: register packaging probes (build-tree ctest entries hkp-probe-*)" OFF)
set(HIPKERNELPROVIDER_MIN_PACKAGING_PROBES 0 CACHE STRING
    "Fail configure when fewer than this many packaging probes are declared. A lane that never builds every architecture the descriptors name sets 1, so broken probe discovery cannot pass as a lane with nothing to probe")
if(HIPKERNELPROVIDER_MIN_PACKAGING_PROBES GREATER 0 AND NOT HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES)
    message(FATAL_ERROR
        "hkp: HIPKERNELPROVIDER_MIN_PACKAGING_PROBES=${HIPKERNELPROVIDER_MIN_PACKAGING_PROBES} "
        "needs -DHIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES=ON: with probes off none are "
        "declared, so the minimum could never be met.")
endif()

hkp_add_packaging()
