# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(ENABLE_ASM_SDPA_ENGINE)
    add_subdirectory(asm_sdpa_engine)
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    add_subdirectory(hip_mlops_engine)
endif()

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    add_subdirectory(kernel_ingestor_engine)
endif()
