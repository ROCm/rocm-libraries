include(GpuArch)

hipconv_add_arch_lib(cdna4 TARGETS gfx950)
# The depthwise families carry the 1-D filter sizes on top of the squares, so they split finer.
math(EXPR HIPCONV_CDNA4_DW_SHARDS "${HIPCONV_NUM_SHARDS} * 3")
# The 1D Toeplitz family also carries conv3d depth-tap capacities 3, 5 and 7.
math(EXPR HIPCONV_CDNA4_DW_TOEPLITZ_SHARDS "${HIPCONV_NUM_SHARDS} * 5")
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    direct
                  KERNEL_CLASS Direct_F16_ConvKernel
                  CONFIG_TABLE direct/direct/config_table.hpp
                  KERNEL       direct/direct/kernel.hpp
                  MATCHER_HPP  direct/direct/config_matcher.hpp
                  MATCHER_CPP  direct/direct/config_matcher.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    direct_l1
                  KERNEL_CLASS DirectL1_ConvKernel
                  CONFIG_TABLE direct/direct_l1/config_table.h
                  KERNEL       direct/direct_l1/kernel.h
                  MATCHER_HPP  direct/direct_l1/config_desc.h
                  MATCHER_CPP  direct/direct_l1/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    patch_embed
                  KERNEL_CLASS PatchEmbed_ConvKernel
                  CONFIG_TABLE direct/patch_embed/config_table.h
                  KERNEL       direct/patch_embed/kernel.h
                  MATCHER_HPP  direct/patch_embed/config_desc.h
                  MATCHER_CPP  direct/patch_embed/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
# direct_flex's unrolled sweeps spill K256 under the default allocation order; allocating local live
# ranges in reverse fits them in 236-246 VGPRs (docs/algorithms/direct/direct-flex-cdna4.md,
# Positions inner).
#
# gfx950 preloads up to 16 dwords of leading scalar kernel arguments into SGPRs at wave launch;
# Hot (kernel_args.h) uses 14.
hipconv_autoshard(ARCH            cdna4
                  NAMESPACE       direct_flex
                  KERNEL_CLASS    DirectFlex_ConvKernel
                  CONFIG_TABLE    direct/direct_flex/config_table.h
                  KERNEL          direct/direct_flex/kernel.h
                  MATCHER_HPP     direct/direct_flex/config_desc.h
                  MATCHER_CPP     direct/direct_flex/config_desc.cpp
                  NUM_SHARDS      ${HIPCONV_NUM_SHARDS}
                  EXTRA_HIP_FLAGS "-mllvm=--greedy-reverse-local-assignment"
                                  "-mllvm=-amdgpu-kernarg-preload-count=16")
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    direct_wgrad
                  KERNEL_CLASS DirectWgrad_ConvKernel
                  CONFIG_TABLE direct/direct_wgrad/config_table.h
                  KERNEL       direct/direct_wgrad/kernel.h
                  MATCHER_HPP  direct/direct_wgrad/config_desc.h
                  MATCHER_CPP  direct/direct_wgrad/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    depthwise_1d_toeplitz
                  KERNEL_CLASS Depthwise_1D_Toeplitz_ConvKernel
                  CONFIG_TABLE depthwise/depthwise_1d_toeplitz/config_table.h
                  KERNEL       depthwise/depthwise_1d_toeplitz/kernel.h
                  MATCHER_HPP  depthwise/depthwise_1d_toeplitz/config_desc.h
                  MATCHER_CPP  depthwise/depthwise_1d_toeplitz/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_CDNA4_DW_TOEPLITZ_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    depthwise_wgrad_hankel
                  KERNEL_CLASS Depthwise_Wgrad_Hankel_ConvKernel
                  CONFIG_TABLE depthwise/depthwise_wgrad_hankel/config_table.h
                  MATCHER_HPP  depthwise/depthwise_wgrad_hankel/config_desc.h
                  MATCHER_CPP  depthwise/depthwise_wgrad_hankel/config_desc.cpp
                  KERNEL       depthwise/depthwise_wgrad_hankel/kernel.h
                  NUM_SHARDS   ${HIPCONV_CDNA4_DW_SHARDS})
if(HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ)
    target_compile_definitions(hipconv_arch_cdna4 PRIVATE HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ)
    hipconv_autoshard(ARCH         cdna4
                      NAMESPACE    depthwise_2d_toeplitz
                      KERNEL_CLASS Depthwise_2D_Toeplitz_ConvKernel
                      CONFIG_TABLE depthwise/depthwise_2d_toeplitz/config_table.h
                      KERNEL       depthwise/depthwise_2d_toeplitz/kernel.h
                      MATCHER_HPP  depthwise/depthwise_2d_toeplitz/config_desc.h
                      MATCHER_CPP  depthwise/depthwise_2d_toeplitz/config_desc.cpp
                      NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
endif()

hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_4c
                  KERNEL_CLASS Grouped_4C_ConvKernel
                  CONFIG_TABLE grouped/grouped_4c/config_table.hpp
                  KERNEL       grouped/grouped_4c/kernel.hpp
                  MATCHER_HPP  grouped/grouped_4c/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_4c/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_4c_wgrad
                  KERNEL_CLASS Grouped_4C_F16_WgradConvKernel
                  CONFIG_TABLE grouped/grouped_4c_wgrad_fp16bf16/config_table.hpp
                  KERNEL       grouped/grouped_4c_wgrad_fp16bf16/kernel.hpp
                  MATCHER_HPP  grouped/grouped_4c_wgrad_fp16bf16/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_4c_wgrad_fp16bf16/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_4c_wgrad_tf32
                  KERNEL_CLASS Grouped_4C_Wgrad_TF32_ConvKernel
                  CONFIG_TABLE grouped/grouped_4c_wgrad_tf32/config_table.hpp
                  KERNEL       grouped/grouped_4c_wgrad_tf32/kernel.hpp
                  MATCHER_HPP  grouped/grouped_4c_wgrad_tf32/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_4c_wgrad_tf32/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_8c
                  KERNEL_CLASS Grouped_8C_ConvKernel
                  CONFIG_TABLE grouped/grouped_8c/config_table.hpp
                  KERNEL       grouped/grouped_8c/kernel.hpp
                  MATCHER_HPP  grouped/grouped_8c/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_8c/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_8c_wgrad
                  KERNEL_CLASS Grouped_8C_F16_WgradConvKernel
                  CONFIG_TABLE grouped/grouped_8c_wgrad_fp16bf16/config_table.hpp
                  KERNEL       grouped/grouped_8c_wgrad_fp16bf16/kernel.hpp
                  MATCHER_HPP  grouped/grouped_8c_wgrad_fp16bf16/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_8c_wgrad_fp16bf16/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_8c_wgrad_tf32
                  KERNEL_CLASS Grouped_8C_Wgrad_TF32_ConvKernel
                  CONFIG_TABLE grouped/grouped_8c_wgrad_tf32/config_table.hpp
                  KERNEL       grouped/grouped_8c_wgrad_tf32/kernel.hpp
                  MATCHER_HPP  grouped/grouped_8c_wgrad_tf32/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_8c_wgrad_tf32/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_16c
                  KERNEL_CLASS Grouped_16C_ConvKernel
                  CONFIG_TABLE grouped/grouped_16c/config_table.hpp
                  KERNEL       grouped/grouped_16c/kernel.hpp
                  MATCHER_HPP  grouped/grouped_16c/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_16c/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_16c_wgrad
                  KERNEL_CLASS Grouped_16C_F16_WgradConvKernel
                  CONFIG_TABLE grouped/grouped_16c_wgrad_fp16bf16/config_table.hpp
                  KERNEL       grouped/grouped_16c_wgrad_fp16bf16/kernel.hpp
                  MATCHER_HPP  grouped/grouped_16c_wgrad_fp16bf16/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_16c_wgrad_fp16bf16/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_16c_wgrad_tf32
                  KERNEL_CLASS Grouped_16C_Wgrad_TF32_ConvKernel
                  CONFIG_TABLE grouped/grouped_16c_wgrad_tf32/config_table.hpp
                  KERNEL       grouped/grouped_16c_wgrad_tf32/kernel.hpp
                  MATCHER_HPP  grouped/grouped_16c_wgrad_tf32/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_16c_wgrad_tf32/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_32c
                  KERNEL_CLASS Grouped_32C_ConvKernel
                  CONFIG_TABLE grouped/grouped_32c/config_table.hpp
                  KERNEL       grouped/grouped_32c/kernel.hpp
                  MATCHER_HPP  grouped/grouped_32c/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_32c/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_32c_wgrad
                  KERNEL_CLASS Grouped_32C_F16_WgradConvKernel
                  CONFIG_TABLE grouped/grouped_32c_wgrad_fp16bf16/config_table.hpp
                  KERNEL       grouped/grouped_32c_wgrad_fp16bf16/kernel.hpp
                  MATCHER_HPP  grouped/grouped_32c_wgrad_fp16bf16/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_32c_wgrad_fp16bf16/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
hipconv_autoshard(ARCH         cdna4
                  NAMESPACE    grouped_32c_wgrad_tf32
                  KERNEL_CLASS Grouped_32C_Wgrad_TF32_ConvKernel
                  CONFIG_TABLE grouped/grouped_32c_wgrad_tf32/config_table.hpp
                  KERNEL       grouped/grouped_32c_wgrad_tf32/kernel.hpp
                  MATCHER_HPP  grouped/grouped_32c_wgrad_tf32/config_desc.hpp
                  MATCHER_CPP  grouped/grouped_32c_wgrad_tf32/config_desc.cpp
                  NUM_SHARDS   ${HIPCONV_NUM_SHARDS})
