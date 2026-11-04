# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Only the provider owns the surrounding repository's hipDNN test layout.
# Pass optional defaults into rocKE without overriding explicit caller inputs.
function(hipkernelprovider_configure_rocke_test_assets repository_root)
    # Header discovery must not depend on graph inputs or their explicit opt-out.
    if(NOT DEFINED ROCKE_HKP_PACKS_DIR)
        set(packs "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/engines/kernel_ingestor_engine/packs")
        if(EXISTS "${packs}/Gfx950AttentionDenseGeometry.hpp"
           AND NOT IS_DIRECTORY "${packs}/Gfx950AttentionDenseGeometry.hpp"
           AND EXISTS "${packs}/Gfx950AttentionUnified3DGeometry.hpp"
           AND NOT IS_DIRECTORY "${packs}/Gfx950AttentionUnified3DGeometry.hpp")
            set(ROCKE_HKP_PACKS_DIR "${packs}" PARENT_SCOPE)
        endif()
    endif()
    if(DEFINED ROCKE_HIPDNN_SDPA_TEMPLATE OR DEFINED ROCKE_HIPDNN_SDPA_BUNDLES_DIR)
        return()
    endif()
    set(template
        "${repository_root}/projects/hipdnn/tools/IngestorGenerator/configs/templates/attention_sdpa_fwd.graph.json")
    set(bundles "${repository_root}/dnn-providers/integration-tests/integration-test-bundles")
    if(EXISTS "${template}" AND NOT IS_DIRECTORY "${template}" AND IS_DIRECTORY "${bundles}")
        set(ROCKE_HIPDNN_SDPA_TEMPLATE "${template}" PARENT_SCOPE)
        set(ROCKE_HIPDNN_SDPA_BUNDLES_DIR "${bundles}" PARENT_SCOPE)
    endif()
endfunction()
