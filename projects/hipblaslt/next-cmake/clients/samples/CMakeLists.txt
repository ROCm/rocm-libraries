# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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

set(TEMP_HIPBLASLT_SAMPLE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/samples)

add_library(sample-helper INTERFACE)

if(HIPBLASLT_ENABLE_BLIS)
    target_link_libraries(sample-helper INTERFACE BLIS::BLIS)
endif()
target_link_libraries(sample-helper INTERFACE hip::device)

target_compile_features(sample-helper INTERFACE cxx_std_17)

target_sources(sample-helper
    INTERFACE
    $<BUILD_INTERFACE:${TEMP_CLIENTS_SOURCE_DIR}/samples/common/helper.h>
)

target_include_directories(sample-helper
    INTERFACE
        $<BUILD_INTERFACE:${TEMP_CLIENTS_SOURCE_DIR}/samples/common>
        $<BUILD_INTERFACE:${TEMP_CLIENTS_SOURCE_DIR}/include>
)

if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(sample-helper INTERFACE)
endif()

target_link_libraries(sample-helper
    INTERFACE
        roc::hipblaslt
)

add_subdirectory(01_hipblaslt_gemm)
add_subdirectory(01_hipblaslt_gemm_ext)
add_subdirectory(01_hipblaslt_gemm_ext_deprecated)
add_subdirectory(02_hipblaslt_gemm_batched)
add_subdirectory(02_hipblaslt_gemm_batched_ext)
add_subdirectory(03_hipblaslt_gemm_tuning_splitk_ext)
add_subdirectory(04_hipblaslt_gemm_bias)
add_subdirectory(04_hipblaslt_gemm_bias_ext)
add_subdirectory(05_hipblaslt_gemm_get_all_algos)
add_subdirectory(05_hipblaslt_gemm_get_all_algos_ext)
add_subdirectory(06_hipblaslt_gemm_get_algo_by_index_ext)
add_subdirectory(07_hipblaslt_gemm_alphavec_ext)
add_subdirectory(08_hipblaslt_gemm_gelu_aux_bias)
add_subdirectory(08_hipblaslt_gemm_gelu_aux_bias_ext)
add_subdirectory(09_hipblaslt_gemm_amax)
add_subdirectory(09_hipblaslt_gemm_amax_ext)
add_subdirectory(10_hipblaslt_gemm_amax_with_scale)
add_subdirectory(10_hipblaslt_gemm_amax_with_scale_ext)
add_subdirectory(11_hipblaslt_gemm_bgradb)
add_subdirectory(11_hipblaslt_gemm_ext_bgradb)
add_subdirectory(12_hipblaslt_gemm_dgelu_bgrad)
add_subdirectory(12_hipblaslt_gemm_dgelu_bgrad_ext)
add_subdirectory(13_hipblaslt_gemm_is_tuned_ext)
add_subdirectory(14_hipblaslt_gemm_tuning_wgm_ext)
add_subdirectory(15_hipblaslt_gemm_with_scale_a_b)
add_subdirectory(15_hipblaslt_gemm_with_scale_a_b_ext)
add_subdirectory(16_hipblaslt_groupedgemm_ext)
add_subdirectory(17_hipblaslt_groupedgemm_fixed_mk_ext)
add_subdirectory(18_hipblaslt_groupedgemm_get_all_algos_ext)
add_subdirectory(19_hipblaslt_gemm_mix_precision)
add_subdirectory(19_hipblaslt_gemm_mix_precision_ext)
add_subdirectory(20_hipblaslt_gemm_mix_precision_with_amax_ext)
add_subdirectory(21_hipblaslt_gemm_attr_tciA_tciB)
add_subdirectory(22_hipblaslt_ext_op_layernorm)
add_subdirectory(23_hipblaslt_ext_op_amax)
add_subdirectory(24_hipblaslt_ext_op_amax_with_scale)
add_subdirectory(25_hipblaslt_gemm_with_TF32)
add_subdirectory(26_hipblaslt_gemm_swizzle_a)
add_subdirectory(26_hipblaslt_gemm_bias_swizzle_a_ext)
add_subdirectory(26_hipblaslt_weight_swizzle_padding)
add_subdirectory(27_hipblaslt_gemm_swish_bias)
