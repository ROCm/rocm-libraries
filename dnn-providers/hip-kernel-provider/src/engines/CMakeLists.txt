# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(ENABLE_ASM_SDPA_ENGINE)
    add_subdirectory(asm_sdpa_engine)
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    add_subdirectory(hip_mlops_engine)
endif()

if(ENABLE_AOT_CATALOG_ENGINE)
    add_subdirectory(aot_catalog_engine)
endif()
