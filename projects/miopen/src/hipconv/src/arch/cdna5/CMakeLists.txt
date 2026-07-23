include(GpuArch)

hipconv_get_matching_gpu("${HIPCONV_LOGICAL_TO_GPU}" "gfx1250" CDNA5_GPUS)

if(CDNA5_GPUS)
    hipconv_add_arch_lib(cdna5 "${CDNA5_GPUS}")
    hipconv_autoshard(ARCH         cdna5
                      NAMESPACE    direct
                      KERNEL_CLASS Direct_F16_ConvKernel
                      CONFIG_TABLE direct/config_table.hpp
                      KERNEL       direct/kernel.hpp
                      NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
    hipconv_autoshard(ARCH         cdna5
                      NAMESPACE    grouped_multi_g
                      KERNEL_CLASS Grouped_MultiG_ConvKernel
                      CONFIG_TABLE grouped/grouped_multi_g/config_table.hpp
                      KERNEL       grouped/grouped_multi_g/kernel.hpp
                      NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
    hipconv_autoshard(ARCH         cdna5
                      NAMESPACE    grouped_multi_g_wgrad
                      KERNEL_CLASS Grouped_Multi_G_WgradConvKernel
                      CONFIG_TABLE grouped/grouped_multi_g_wgrad/config_table.hpp
                      KERNEL       grouped/grouped_multi_g_wgrad/kernel.hpp
                      NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
endif()
