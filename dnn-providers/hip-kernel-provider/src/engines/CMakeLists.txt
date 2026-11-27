# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(ENABLE_ASM_SDPA_ENGINE)
    add_subdirectory(asm_sdpa_engine)
endif()

# The HIP MLOps operations migrated to the kernel ingestor keep their kernel sources beside the
# descriptors of the engine that serves each (<engine>/kernels/), and the headers those share
# under include/. Both engines compile them, so both read them from here.
set(HKP_HIP_MLOPS_INGESTOR_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_ingestor_engine/descriptors/HipMlops")

# The ingestor's HipMlops packs share this engine's applicability checks and compile
# options, so its shared sources build in either mode.
if(ENABLE_HIP_MLOPS_ENGINE OR HIPKERNELPROVIDER_MLOPS_ENABLE_INGESTOR)
    add_subdirectory(hip_mlops_engine)
endif()

if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    add_subdirectory(kernel_ingestor_engine)
endif()
