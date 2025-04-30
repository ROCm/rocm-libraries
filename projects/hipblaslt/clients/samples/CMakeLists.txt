# ########################################################################
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Samples
add_executable( sample_hipblaslt_gemm 01_basic_gemm/sample_hipblaslt_gemm.cpp)
add_executable( sample_hipblaslt_gemm_ext 01_basic_gemm_ext/sample_hipblaslt_gemm_ext.cpp)
add_executable( sample_hipblaslt_gemm_batched 02_batched_gemm/sample_hipblaslt_gemm_batched.cpp)
add_executable( sample_hipblaslt_gemm_batched_ext 02_batched_gemm_ext/sample_hipblaslt_gemm_batched_ext.cpp)
add_executable( sample_hipblaslt_gemm_tuning_splitk_ext 03_splitk_gemm_ext/sample_hipblaslt_gemm_tuning_splitk_ext.cpp)
add_executable( sample_hipblaslt_gemm_bias 04_gemm_bias/sample_hipblaslt_gemm_bias.cpp)
add_executable( sample_hipblaslt_gemm_bias_ext 04_gemm_bias_ext/sample_hipblaslt_gemm_bias_ext.cpp)
add_executable( sample_hipblaslt_gemm_get_all_algos 05_get_all_algos_gemm/sample_hipblaslt_gemm_get_all_algos.cpp)
add_executable( sample_hipblaslt_gemm_get_all_algos_ext 05_get_all_algos_gemm_ext/sample_hipblaslt_gemm_get_all_algos_ext.cpp)
add_executable( sample_hipblaslt_gemm_get_algo_by_index_ext 06_get_algo_by_index_gemm_ext/sample_hipblaslt_gemm_get_algo_by_index_ext.cpp)
add_executable( sample_hipblaslt_gemm_alphavec_ext 07_gemm_alpha_vec_ext/sample_hipblaslt_gemm_alphavec_ext.cpp)
add_executable( sample_hipblaslt_gemm_gelu_aux_bias 08_gemm_gelu_aux_bias/sample_hipblaslt_gemm_gelu_aux_bias.cpp)
add_executable( sample_hipblaslt_gemm_gelu_aux_bias_ext 08_gemm_gelu_aux_bias_ext/sample_hipblaslt_gemm_gelu_aux_bias_ext.cpp)
add_executable( sample_hipblaslt_gemm_amax 09_gemm_amax/sample_hipblaslt_gemm_amax.cpp)
add_executable( sample_hipblaslt_gemm_amax_ext 09_gemm_amax_ext/sample_hipblaslt_gemm_amax_ext.cpp)
add_executable( sample_hipblaslt_gemm_amax_with_scale 10_gemm_amax_with_scale/sample_hipblaslt_gemm_amax_with_scale.cpp)
add_executable( sample_hipblaslt_gemm_amax_with_scale_ext 10_gemm_amax_with_scale_ext/sample_hipblaslt_gemm_amax_with_scale_ext.cpp)
add_executable( sample_hipblaslt_gemm_bgradb 11_gemm_bgradb/sample_hipblaslt_gemm_bgradb.cpp)
add_executable( sample_hipblaslt_gemm_ext_bgradb 11_gemm_bgradb_ext/sample_hipblaslt_gemm_ext_bgradb.cpp)
add_executable( sample_hipblaslt_gemm_dgelu_bgrad 12_gemm_dgelu_bgrad/sample_hipblaslt_gemm_dgelu_bgrad.cpp)
add_executable( sample_hipblaslt_gemm_dgelu_bgrad_ext 12_gemm_dgelu_bgrad_ext/sample_hipblaslt_gemm_dgelu_bgrad_ext.cpp)
add_executable( sample_hipblaslt_gemm_is_tuned_ext 13_is_tuned_gemm_ext/sample_hipblaslt_gemm_is_tuned_ext.cpp)
add_executable( sample_hipblaslt_gemm_tuning_wgm_ext 14_tuning_wgm_gemm_ext/sample_hipblaslt_gemm_tuning_wgm_ext.cpp)
add_executable( sample_hipblaslt_gemm_with_scale_a_b 15_gemm_scale_a_b/sample_hipblaslt_gemm_with_scale_a_b.cpp)
add_executable( sample_hipblaslt_gemm_with_scale_a_b_ext 15_gemm_scale_a_b_ext/sample_hipblaslt_gemm_with_scale_a_b_ext.cpp)
add_executable( sample_hipblaslt_groupedgemm_ext 16_gemm_grouped_ext/sample_hipblaslt_groupedgemm_ext.cpp)
add_executable( sample_hipblaslt_groupedgemm_fixed_mk_ext 17_fixed_mk_gemm_grouped_ext/sample_hipblaslt_groupedgemm_fixed_mk_ext.cpp)
add_executable( sample_hipblaslt_groupedgemm_get_all_algos_ext 18_get_all_algos_gemm_grouped_ext/sample_hipblaslt_groupedgemm_get_all_algos_ext.cpp)
add_executable( sample_hipblaslt_gemm_mix_precision 19_mix_precision_gemm/sample_hipblaslt_gemm_mix_precision.cpp)
add_executable( sample_hipblaslt_gemm_mix_precision_ext 19_mix_precision_gemm_ext/sample_hipblaslt_gemm_mix_precision_ext.cpp)
add_executable( sample_hipblaslt_gemm_mix_precision_with_amax_ext 20_mix_precision_gemm_amax_ext/sample_hipblaslt_gemm_mix_precision_with_amax_ext.cpp)
add_executable( sample_hipblaslt_gemm_attr_tciA_tciB 21_gemm_cvtA_cvtB/sample_hipblaslt_gemm_attr_tciA_tciB.cpp)
add_executable( sample_hipblaslt_ext_op_layernorm 22_layernorm_ext/sample_hipblaslt_ext_op_layernorm.cpp)
add_executable( sample_hipblaslt_gemm_ext_deprecated deprecated/sample_hipblaslt_gemm_ext_deprecated.cpp)
add_executable( sample_hipblaslt_ext_op_amax 23_amax_ext/sample_hipblaslt_ext_op_amax.cpp)
add_executable( sample_hipblaslt_ext_op_amax_with_scale 24_amax_with_scale_ext/sample_hipblaslt_ext_op_amax_with_scale.cpp)
add_executable( sample_hipblaslt_gemm_with_TF32 25_gemm_with_TF32/sample_hipblaslt_gemm_with_TF32.cpp)
add_executable( sample_hipblaslt_gemm_swizzle_a 26_gemm_swizzle_a/sample_hipblaslt_gemm_swizzle_a.cpp)
add_executable( sample_hipblaslt_gemm_bias_swizzle_a_ext 26_gemm_swizzle_a/sample_hipblaslt_gemm_bias_swizzle_a_ext.cpp)
add_executable( sample_hipblaslt_weight_swizzle_padding 26_gemm_swizzle_a/sample_hipblaslt_weight_swizzle_padding.cpp)
add_executable( sample_hipblaslt_gemm_swish_bias 27_gemm_swish_bias/sample_hipblaslt_gemm_swish_bias.cpp)


set(samples sample_hipblaslt_gemm
            sample_hipblaslt_gemm_ext
            sample_hipblaslt_gemm_batched
            sample_hipblaslt_gemm_batched_ext
            sample_hipblaslt_gemm_tuning_splitk_ext
            sample_hipblaslt_gemm_bias
            sample_hipblaslt_gemm_bias_ext
            sample_hipblaslt_gemm_get_all_algos
            sample_hipblaslt_gemm_get_all_algos_ext
            sample_hipblaslt_gemm_get_algo_by_index_ext
            sample_hipblaslt_gemm_alphavec_ext
            sample_hipblaslt_gemm_gelu_aux_bias
            sample_hipblaslt_gemm_gelu_aux_bias_ext
            sample_hipblaslt_gemm_amax
            sample_hipblaslt_gemm_amax_ext
            sample_hipblaslt_gemm_amax_with_scale
            sample_hipblaslt_gemm_amax_with_scale_ext
            sample_hipblaslt_gemm_bgradb
            sample_hipblaslt_gemm_ext_bgradb
            sample_hipblaslt_gemm_dgelu_bgrad
            sample_hipblaslt_gemm_dgelu_bgrad_ext
            sample_hipblaslt_gemm_is_tuned_ext
            sample_hipblaslt_gemm_tuning_wgm_ext
            sample_hipblaslt_gemm_with_scale_a_b
            sample_hipblaslt_gemm_with_scale_a_b_ext
            sample_hipblaslt_groupedgemm_ext
            sample_hipblaslt_groupedgemm_fixed_mk_ext
            sample_hipblaslt_groupedgemm_get_all_algos_ext
            sample_hipblaslt_gemm_mix_precision
            sample_hipblaslt_gemm_mix_precision_ext
            sample_hipblaslt_gemm_mix_precision_with_amax_ext
            sample_hipblaslt_gemm_attr_tciA_tciB
            sample_hipblaslt_ext_op_layernorm
            sample_hipblaslt_gemm_ext_deprecated
            sample_hipblaslt_ext_op_amax
            sample_hipblaslt_ext_op_amax_with_scale
            sample_hipblaslt_gemm_with_TF32
            sample_hipblaslt_gemm_swizzle_a
            sample_hipblaslt_gemm_bias_swizzle_a_ext
            sample_hipblaslt_weight_swizzle_padding
            sample_hipblaslt_gemm_swish_bias)

set( sample_list_all ${samples})

set( sample_list_hip_device ${sample_list_all} )

foreach( exe ${sample_list_all} )
  target_link_libraries( ${exe} PRIVATE roc::hipblaslt )

  set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
  )

  target_compile_definitions( ${exe} PRIVATE ROCM_USE_FLOAT16 )

  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
      )

  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      )

  if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # GCC or hip-clang needs specific flags to turn on f16c intrinsics
    target_compile_options( ${exe} PRIVATE -mf16c -Wno-unused-result )
    target_compile_definitions( ${exe} PRIVATE ROCBLASLT_INTERNAL_API )
  endif( )
  if( LEGACY_HIPBLAS_DIRECT )
    target_compile_definitions( ${exe} PUBLIC LEGACY_HIPBLAS_DIRECT )
  endif()
endforeach( )

foreach( exe ${sample_list_hip_device} )
  target_link_libraries( ${exe} PRIVATE hip::device )
endforeach( )

# Install executables to libexec
foreach( exe ${samples} )
  install(
    TARGETS ${exe}
    RUNTIME
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/hipblaslt-samples
    COMPONENT samples
  )
endforeach( )

# Install source files to share
install(
  DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/hipblaslt/samples
  COMPONENT samples
  FILES_MATCHING
  PATTERN "*.cpp"
  PATTERN "*.h"
  PATTERN "CMakeLists.txt" EXCLUDE
  PATTERN "build" EXCLUDE
)

# Add tests for samples
foreach( exe ${samples} )
  add_test(NAME ${exe} COMMAND ${exe})
endforeach()
