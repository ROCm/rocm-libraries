###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 ###############################################################################

 set(CK_CONTRACTION_INSTANCE_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_kknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_knnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_mknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_mnnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_cf32_compute_cf32_kknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_cf32_compute_cf32_knnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_cf32_compute_cf32_mknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_cf32_compute_cf32_mnnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_cf64_compute_cf64_kknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_cf64_compute_cf64_knnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_cf64_compute_cf64_mknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_cf64_compute_cf64_mnnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_kknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_knnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_mknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_mnnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_kknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_knnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_mknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_mnnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_kknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_knnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_mknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_mnnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_kknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_knnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_mknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_mnnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_kknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_knnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_mknn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_mnnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_knn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_mnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_kkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_knn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_mkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_mnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_kkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_knn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_mkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_mnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_kkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_knn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_mkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_compute_f32_mnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_kkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_knn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_mkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_bf16_mnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_kkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_knn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_mkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_compute_f16_mnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_kkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_knn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_mkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_mnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_kkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_knn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_mkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_compute_f32_mnn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_kkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_knn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_mkn_instance.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/device_contraction_scale_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_mnn_instance.cpp
     )

add_hiptensor_component(hiptensor_contraction_instances ${CK_CONTRACTION_INSTANCE_SOURCES})
target_include_directories(hiptensor_contraction_instances PRIVATE ${composable_kernel_INCLUDES})
