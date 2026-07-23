include(GpuArch)

hipconv_get_matching_gpu("${HIPCONV_LOGICAL_TO_GPU}" "gfx950" CDNA4_GPUS)

if(CDNA4_GPUS)
    hipconv_add_arch_lib(cdna4 "${CDNA4_GPUS}")
    hipconv_autoshard(ARCH         cdna4
                      NAMESPACE    direct
                      KERNEL_CLASS Direct_F16_ConvKernel
                      CONFIG_TABLE direct/direct/config_table.hpp
                      KERNEL       direct/direct/kernel.hpp
                      NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
    hipconv_autoshard(ARCH         cdna4
                      NAMESPACE    direct_l1
                      KERNEL_CLASS DirectL1_ConvKernel
                      CONFIG_TABLE direct/direct_l1/config_table.h
                      KERNEL       direct/direct_l1/kernel.h
                      NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
endif()
