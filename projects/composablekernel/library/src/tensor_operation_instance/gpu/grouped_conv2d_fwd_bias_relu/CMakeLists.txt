# ONLY XDL_KERNELS
add_instance_library(device_grouped_conv2d_fwd_bias_relu_instance
   xdl/device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_instance.cpp
   xdl/device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_16x16_instance.cpp

   xdl/large_tensor/device_grouped_conv2d_fwd_bias_relu_xdl_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp

   xdl/merged_groups/device_grouped_conv2d_fwd_bias_relu_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instance.cpp

   xdl/mem/device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_intra_instance.cpp
   xdl/mem/device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_inter_instance.cpp

   xdl/comp/device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instance.cpp
   xdl/comp/device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_2x_instance.cpp
   xdl/comp/device_grouped_conv2d_fwd_bias_relu_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_part2_instance.cpp
)
