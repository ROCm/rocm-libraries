# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# ============================================================================
# Selector dump of a checkout, without configuring it
# ============================================================================
#   cmake -DHKP_ENGINE=<hip_mlops|asm_sdpa_fwd>
#         -DHKP_PROVIDER_DIR=<checkout>/dnn-providers/hip-kernel-provider
#         -DHKP_OUT=<file>
#         -P <checkout>/dnn-providers/hip-kernel-provider/cmake/HkpSelectorManifest.cmake
#
# Writes the same file the build writes to <build>/selector_revisions/<engine>.selector.json,
# from the recipe in HKP_PROVIDER_DIR run on that tree. Run it from the tree it describes:
# the recipe it includes decides which sources are inputs, and that is the revision a model
# trained at that commit recorded. `uhd_gen restamp` takes the dump of the training commit
# as --selector-before and this build's dump as --selector-after.
foreach(_required HKP_ENGINE HKP_PROVIDER_DIR HKP_OUT)
    if(NOT ${_required})
        message(FATAL_ERROR "HkpSelectorManifest: -D${_required}=... is required.")
    endif()
endforeach()

set(_engines "${HKP_PROVIDER_DIR}/src/engines")
if(HKP_ENGINE STREQUAL "hip_mlops")
    include("${_engines}/hip_mlops_engine/HipMlopsSelectorRevision.cmake")
    hkp_hip_mlops_revision(_revision "${_engines}/hip_mlops_engine" MANIFEST "${HKP_OUT}")
elseif(HKP_ENGINE STREQUAL "asm_sdpa_fwd")
    include("${_engines}/asm_sdpa_engine/AsmSdpaSelectorRevision.cmake")
    hkp_asm_sdpa_fwd_revision(_revision "${_engines}/asm_sdpa_engine" MANIFEST "${HKP_OUT}")
else()
    message(FATAL_ERROR
        "HkpSelectorManifest: HKP_ENGINE '${HKP_ENGINE}' is not hip_mlops or asm_sdpa_fwd.")
endif()
message(STATUS "${HKP_ENGINE} selector revision ${_revision} -> ${HKP_OUT}")
