include(GpuArch)

hipconv_add_arch_lib(cdna4 TARGETS gfx950)
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
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    direct_wgrad
                  KERNEL_CLASS DirectWgrad_ConvKernel
                  CONFIG_TABLE direct/direct_wgrad/config_table.h
                  KERNEL       direct/direct_wgrad/kernel.h
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    depthwise_1d_toeplitz
                  KERNEL_CLASS Depthwise_1D_Toeplitz_ConvKernel
                  CONFIG_TABLE depthwise/depthwise_1d_toeplitz/config_table.h
                  KERNEL       depthwise/depthwise_1d_toeplitz/kernel.h
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
if(HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ)
    target_compile_definitions(hipconv_arch_cdna4 PRIVATE HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ)
    hipconv_autoshard(ARCH         cdna4
                      NAMESPACE    depthwise_2d_toeplitz
                      KERNEL_CLASS Depthwise_2D_Toeplitz_ConvKernel
                      CONFIG_TABLE depthwise/depthwise_2d_toeplitz/config_table.h
                      KERNEL       depthwise/depthwise_2d_toeplitz/kernel.h
                      NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
endif()
