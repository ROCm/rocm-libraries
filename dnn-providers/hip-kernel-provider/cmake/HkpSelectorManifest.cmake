# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# ============================================================================
# Selector dump of a checkout, without configuring it
# ============================================================================
#   cmake -DHKP_ENGINE=asm_sdpa
#         -DHKP_PROVIDER_DIR=<checkout>/dnn-providers/hip-kernel-provider
#         -DHKP_OUT=<file>
#         -P <checkout>/dnn-providers/hip-kernel-provider/cmake/HkpSelectorManifest.cmake
#
# Writes the same file the build writes to <build>/selector_revisions/<engine>.selector.json,
# from the recipe in HKP_PROVIDER_DIR run on that tree. Run it from the tree it describes:
# the recipe it includes decides which sources are inputs, and that is the revision a model
# trained at that commit recorded. Comparing the dump of the commit a model was trained at
# with this build's names the inputs that changed, which is what decides whether the model
# must be retrained (the steps are in the engine's recipe header).

# A -P script starts with every policy unset; the provider's minimum sets them as a build does.
cmake_minimum_required(VERSION 3.25.2)

foreach(_required HKP_ENGINE HKP_PROVIDER_DIR HKP_OUT)
    if(NOT ${_required})
        message(FATAL_ERROR "HkpSelectorManifest: -D${_required}=... is required.")
    endif()
endforeach()

set(_engines "${HKP_PROVIDER_DIR}/src/engines")
if(HKP_ENGINE STREQUAL "asm_sdpa")
    include("${_engines}/asm_sdpa_engine/AsmSdpaSelectorRevision.cmake")
    hkp_asm_sdpa_revision(_revision "${_engines}/asm_sdpa_engine" MANIFEST "${HKP_OUT}")
else()
    message(FATAL_ERROR "HkpSelectorManifest: HKP_ENGINE '${HKP_ENGINE}' is not asm_sdpa.")
endif()
message(STATUS "${HKP_ENGINE} selector revision ${_revision} -> ${HKP_OUT}")
