add_custom_target(example_grouped_conv_bwd_weight)
add_example_executable(example_grouped_conv_bwd_weight_xdl_fp16 grouped_conv_bwd_weight_xdl_fp16.cpp)
add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_xdl_fp16)

add_example_executable(example_grouped_conv_bwd_weight_xdl_bf16 grouped_conv_bwd_weight_xdl_bf16.cpp)
add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_xdl_bf16)

add_example_executable(example_grouped_conv_bwd_weight_xdl_fp16_comp_bf8_fp8 grouped_conv_bwd_weight_xdl_fp16_comp_bf8_fp8.cpp)
add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_xdl_fp16_comp_bf8_fp8)

add_example_executable(example_grouped_conv_bwd_weight_wmma_fp16 grouped_conv_bwd_weight_wmma_fp16.cpp)
add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_wmma_fp16)

add_example_executable(example_grouped_conv_bwd_weight_dl_fp16 grouped_conv_bwd_weight_dl_fp16.cpp)
add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_dl_fp16)

add_example_executable(example_grouped_conv_bwd_weight_v3_xdl_bf16 grouped_conv_bwd_weight_v3_xdl_bf16.cpp)
add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_v3_xdl_bf16)

add_example_executable(example_grouped_conv_bwd_weight_v3_xdl_fp16 grouped_conv_bwd_weight_v3_xdl_fp16.cpp)
add_example_dependencies(example_grouped_conv_bwd_weight example_grouped_conv_bwd_weight_v3_xdl_fp16)
