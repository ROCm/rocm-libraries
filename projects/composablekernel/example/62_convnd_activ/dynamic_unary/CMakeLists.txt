list(APPEND gpu_list gfx908 gfx90a gfx942 gfx950)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
 if(gpu IN_LIST gpu_list AND target EQUAL 0)
      add_custom_target(example_convnd_activ_dynamic_unary_xdl)
      # Sigmoid
      add_example_executable(example_convnd_fwd_xdl_dynamic_sigmoid_fp16 convnd_fwd_xdl_dynamic_sigmoid_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_sigmoid_fp16)
      # Tanh
      add_example_executable(example_convnd_fwd_xdl_dynamic_tanh_fp16 convnd_fwd_xdl_dynamic_tanh_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_tanh_fp16)
      # Relu
      add_example_executable(example_convnd_fwd_xdl_dynamic_relu_fp16 convnd_fwd_xdl_dynamic_relu_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_relu_fp16)
      # SoftRelu
      add_example_executable(example_convnd_fwd_xdl_dynamic_softrelu_fp16 convnd_fwd_xdl_dynamic_softrelu_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_softrelu_fp16)
      # Abs
      add_example_executable(example_convnd_fwd_xdl_dynamic_abs_fp16 convnd_fwd_xdl_dynamic_abs_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_abs_fp16)
      # Pow
      add_example_executable(example_convnd_fwd_xdl_dynamic_pow_fp16 convnd_fwd_xdl_dynamic_pow_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_pow_fp16)
      # Clipped Relu
      add_example_executable(example_convnd_fwd_xdl_dynamic_clippedrelu_fp16 convnd_fwd_xdl_dynamic_clippedrelu_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_clippedrelu_fp16)
      # Leaky Relu
      add_example_executable(example_convnd_fwd_xdl_dynamic_leakyrelu_fp16 convnd_fwd_xdl_dynamic_leakyrelu_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_leakyrelu_fp16)
      # Elu
      add_example_executable(example_convnd_fwd_xdl_dynamic_elu_fp16 convnd_fwd_xdl_dynamic_elu_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_elu_fp16)
      # Swish
      add_example_executable(example_convnd_fwd_xdl_dynamic_swish_fp16 convnd_fwd_xdl_dynamic_swish_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_swish_fp16)
      # PassThrough
      add_example_executable(example_convnd_fwd_xdl_dynamic_passthrough_fp16 convnd_fwd_xdl_dynamic_passthrough_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_passthrough_fp16)
      # Logistic
      add_example_executable(example_convnd_fwd_xdl_dynamic_logistic_fp16 convnd_fwd_xdl_dynamic_logistic_fp16.cpp)
      add_example_dependencies(example_convnd_activ_dynamic_unary_xdl example_convnd_fwd_xdl_dynamic_logistic_fp16)
   
   set(target 1)
 endif()
endforeach()
