/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "hipsparselt_data.hpp"
#include "hipsparselt_datatype2string.hpp"
#include "hipsparselt_test.hpp"
#include "testing_auxiliary.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{

    // ----------------------------------------------------------------------------
    // aux
    // ----------------------------------------------------------------------------

    // In the general case of <Ti, To, Tc>, these tests do not apply, and if this
    // functor is called, an internal error message is generated. When converted
    // to bool, this functor returns false.
    template <typename Ti, typename To = Ti, typename Tc = To, typename TBias = Ti, typename = void>
    struct aux_testing : hipsparselt_test_invalid
    {
    };

    // When Ti = To = Tc != void, this test applies.
    // When converted to bool, this functor returns true.
    template <typename Ti, typename To, typename Tc, typename TBias>
    struct aux_testing<
        Ti,
        To,
        Tc,
        TBias,
        std::enable_if_t<std::is_same<Ti, __half>{} || std::is_same<Ti, hip_bfloat16>{}
                         || std::is_same<Ti, int8_t>{}>> : hipsparselt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            // --- non-aggregated tests ---
            if(!strcmp(arg.function, "aux_handle_init_bad_arg"))
                testing_aux_handle_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_handle"))
                testing_aux_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_dense_init_arg"))
                testing_aux_mat_dense_init(arg);
            else if(!strcmp(arg.function, "aux_mat_structured_init"))
                testing_aux_mat_structured_init(arg);
            else if(!strcmp(arg.function, "aux_matmul_init"))
                testing_aux_matmul_init(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_get_bias_vector"))
                testing_aux_matmul_set_get_bias_vector(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init"))
                testing_aux_matmul_alg_init(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init"))
                testing_aux_matmul_plan_init(arg);
            else if(!strcmp(arg.function, "aux_get_workspace_size"))
                testing_aux_get_workspace_size(arg);
            // --- individual sub-tests: aux_get_version ---
            else if(!strcmp(arg.function, "aux_get_version_match"))
                testing_aux_get_version_match(arg);
            else if(!strcmp(arg.function, "aux_get_version_git_rev_null"))
                testing_aux_get_version_git_rev_null(arg);
            // --- individual sub-tests: aux_handle_destroy_bad_arg ---
            else if(!strcmp(arg.function, "aux_handle_destroy_bad_arg_uninit"))
                testing_aux_handle_destroy_bad_arg_uninit(arg);
            else if(!strcmp(arg.function, "aux_handle_destroy_bad_arg_null"))
                testing_aux_handle_destroy_bad_arg_null(arg);
            // --- individual sub-tests: aux_mat_init_dense_bad_arg ---
            else if(!strcmp(arg.function, "aux_mat_init_dense_bad_arg_uninit_handle"))
                testing_aux_mat_init_dense_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_init_dense_bad_arg_null_handle"))
                testing_aux_mat_init_dense_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_init_dense_bad_arg_null_descr"))
                testing_aux_mat_init_dense_bad_arg_null_descr(arg);
            else if(!strcmp(arg.function, "aux_mat_init_dense_bad_arg_zero_row"))
                testing_aux_mat_init_dense_bad_arg_zero_row(arg);
            else if(!strcmp(arg.function, "aux_mat_init_dense_bad_arg_zero_col"))
                testing_aux_mat_init_dense_bad_arg_zero_col(arg);
            else if(!strcmp(arg.function, "aux_mat_init_dense_bad_arg_zero_ld"))
                testing_aux_mat_init_dense_bad_arg_zero_ld(arg);
            else if(!strcmp(arg.function, "aux_mat_init_dense_bad_arg_large_ld"))
                testing_aux_mat_init_dense_bad_arg_large_ld(arg);
            else if(!strcmp(arg.function, "aux_mat_init_dense_bad_arg_large_alignment"))
                testing_aux_mat_init_dense_bad_arg_large_alignment(arg);
            // --- individual sub-tests: aux_mat_init_structured_bad_arg ---
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_uninit_handle"))
                testing_aux_mat_init_structured_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_null_handle"))
                testing_aux_mat_init_structured_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_null_descr"))
                testing_aux_mat_init_structured_bad_arg_null_descr(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_zero_row"))
                testing_aux_mat_init_structured_bad_arg_zero_row(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_small_row"))
                testing_aux_mat_init_structured_bad_arg_small_row(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_zero_col"))
                testing_aux_mat_init_structured_bad_arg_zero_col(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_small_col"))
                testing_aux_mat_init_structured_bad_arg_small_col(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_zero_ld"))
                testing_aux_mat_init_structured_bad_arg_zero_ld(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_unaligned_ld"))
                testing_aux_mat_init_structured_bad_arg_unaligned_ld(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_unaligned_row"))
                testing_aux_mat_init_structured_bad_arg_unaligned_row(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_unsupported_type"))
                testing_aux_mat_init_structured_bad_arg_unsupported_type(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_large_ld"))
                testing_aux_mat_init_structured_bad_arg_large_ld(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg_large_alignment"))
                testing_aux_mat_init_structured_bad_arg_large_alignment(arg);
            // --- individual sub-tests: aux_mat_set_get_attr ---
            else if(!strcmp(arg.function, "aux_mat_set_get_attr_num_batches"))
                testing_aux_mat_set_get_attr_num_batches(arg);
            else if(!strcmp(arg.function, "aux_mat_set_get_attr_batch_stride"))
                testing_aux_mat_set_get_attr_batch_stride(arg);
            // --- individual sub-tests: aux_mat_destroy_bad_arg ---
            else if(!strcmp(arg.function, "aux_mat_destroy_bad_arg_uninit"))
                testing_aux_mat_destroy_bad_arg_uninit(arg);
            else if(!strcmp(arg.function, "aux_mat_destroy_bad_arg_null"))
                testing_aux_mat_destroy_bad_arg_null(arg);
            // --- individual sub-tests: aux_mat_assign ---
            else if(!strcmp(arg.function, "aux_mat_assign_copy_value"))
                testing_aux_mat_assign_copy_value(arg);
            else if(!strcmp(arg.function, "aux_mat_assign_not_reference"))
                testing_aux_mat_assign_not_reference(arg);
            // --- individual sub-tests: aux_mat_set_attr_bad_arg ---
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg_null_handle"))
                testing_aux_mat_set_attr_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg_uninit_handle"))
                testing_aux_mat_set_attr_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg_null_descr"))
                testing_aux_mat_set_attr_bad_arg_null_descr(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg_uninit_descr"))
                testing_aux_mat_set_attr_bad_arg_uninit_descr(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg_null_data"))
                testing_aux_mat_set_attr_bad_arg_null_data(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg_zero_batches"))
                testing_aux_mat_set_attr_bad_arg_zero_batches(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg_wrong_size_batches"))
                testing_aux_mat_set_attr_bad_arg_wrong_size_batches(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg_null_stride"))
                testing_aux_mat_set_attr_bad_arg_null_stride(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg_invalid_stride"))
                testing_aux_mat_set_attr_bad_arg_invalid_stride(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg_wrong_size_stride"))
                testing_aux_mat_set_attr_bad_arg_wrong_size_stride(arg);
            // --- individual sub-tests: aux_mat_get_attr_bad_arg ---
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg_null_handle"))
                testing_aux_mat_get_attr_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg_uninit_handle"))
                testing_aux_mat_get_attr_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg_null_descr"))
                testing_aux_mat_get_attr_bad_arg_null_descr(arg);
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg_uninit_descr"))
                testing_aux_mat_get_attr_bad_arg_uninit_descr(arg);
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg_null_data"))
                testing_aux_mat_get_attr_bad_arg_null_data(arg);
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg_wrong_size_batches"))
                testing_aux_mat_get_attr_bad_arg_wrong_size_batches(arg);
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg_null_stride"))
                testing_aux_mat_get_attr_bad_arg_null_stride(arg);
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg_wrong_size_stride"))
                testing_aux_mat_get_attr_bad_arg_wrong_size_stride(arg);
            // --- individual sub-tests: aux_matmul_init_bad_arg ---
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_null_handle"))
                testing_aux_matmul_init_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_uninit_handle"))
                testing_aux_matmul_init_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_null_descr"))
                testing_aux_matmul_init_bad_arg_null_descr(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_conj_opA"))
                testing_aux_matmul_init_bad_arg_conj_opA(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_conj_opB"))
                testing_aux_matmul_init_bad_arg_conj_opB(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_uninit_matA"))
                testing_aux_matmul_init_bad_arg_uninit_matA(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_two_sparse"))
                testing_aux_matmul_init_bad_arg_two_sparse(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_wrong_compute_type"))
                testing_aux_matmul_init_bad_arg_wrong_compute_type(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_structured_C"))
                testing_aux_matmul_init_bad_arg_structured_C(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_order"))
                testing_aux_matmul_init_bad_arg_mismatched_order(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_null_matA"))
                testing_aux_matmul_init_bad_arg_null_matA(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_uninit_matB"))
                testing_aux_matmul_init_bad_arg_uninit_matB(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_null_matB"))
                testing_aux_matmul_init_bad_arg_null_matB(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_uninit_matC"))
                testing_aux_matmul_init_bad_arg_uninit_matC(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_null_matC"))
                testing_aux_matmul_init_bad_arg_null_matC(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_uninit_matD"))
                testing_aux_matmul_init_bad_arg_uninit_matD(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_null_matD"))
                testing_aux_matmul_init_bad_arg_null_matD(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_same_op_int8"))
                testing_aux_matmul_init_bad_arg_same_op_int8(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_K"))
                testing_aux_matmul_init_bad_arg_mismatched_K(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_N"))
                testing_aux_matmul_init_bad_arg_mismatched_N(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_C_dim"))
                testing_aux_matmul_init_bad_arg_mismatched_C_dim(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_D_dim"))
                testing_aux_matmul_init_bad_arg_mismatched_D_dim(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_unsupported_A_type"))
                testing_aux_matmul_init_bad_arg_unsupported_A_type(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_B_type"))
                testing_aux_matmul_init_bad_arg_mismatched_B_type(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_C_type"))
                testing_aux_matmul_init_bad_arg_mismatched_C_type(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_D_type"))
                testing_aux_matmul_init_bad_arg_mismatched_D_type(arg);
            // --- individual sub-tests: aux_matmul_assign ---
            else if(!strcmp(arg.function, "aux_matmul_assign_copy_value"))
                testing_aux_matmul_assign_copy_value(arg);
            else if(!strcmp(arg.function, "aux_matmul_assign_not_reference"))
                testing_aux_matmul_assign_not_reference(arg);
            // --- individual sub-tests: aux_matmul_set_attr_bad_arg ---
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_null_handle"))
                testing_aux_matmul_set_attr_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_uninit_handle"))
                testing_aux_matmul_set_attr_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_null_matmul"))
                testing_aux_matmul_set_attr_bad_arg_null_matmul(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_uninit_matmul"))
                testing_aux_matmul_set_attr_bad_arg_uninit_matmul(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_null_data"))
                testing_aux_matmul_set_attr_bad_arg_null_data(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_wrong_size"))
                testing_aux_matmul_set_attr_bad_arg_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_relu_upperbound_wrong_size"))
                testing_aux_matmul_set_attr_bad_arg_relu_upperbound_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_sigmoid_int8"))
                testing_aux_matmul_set_attr_bad_arg_sigmoid_int8(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_bias_pointer_wrong_size"))
                testing_aux_matmul_set_attr_bad_arg_bias_pointer_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_bias_stride_invalid"))
                testing_aux_matmul_set_attr_bad_arg_bias_stride_invalid(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg_bias_type"))
                testing_aux_matmul_set_attr_bad_arg_bias_type(arg);
            // --- individual sub-tests: aux_matmul_get_attr_bad_arg ---
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_null_handle"))
                testing_aux_matmul_get_attr_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_uninit_handle"))
                testing_aux_matmul_get_attr_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_null_matmul"))
                testing_aux_matmul_get_attr_bad_arg_null_matmul(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_uninit_matmul"))
                testing_aux_matmul_get_attr_bad_arg_uninit_matmul(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_null_data"))
                testing_aux_matmul_get_attr_bad_arg_null_data(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_wrong_size"))
                testing_aux_matmul_get_attr_bad_arg_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_relu_upperbound_null"))
                testing_aux_matmul_get_attr_bad_arg_relu_upperbound_null(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_relu_upperbound_wrong_size"))
                testing_aux_matmul_get_attr_bad_arg_relu_upperbound_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_relu_threshold_null"))
                testing_aux_matmul_get_attr_bad_arg_relu_threshold_null(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_relu_threshold_wrong_size"))
                testing_aux_matmul_get_attr_bad_arg_relu_threshold_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_leakyrelu_alpha_null"))
                testing_aux_matmul_get_attr_bad_arg_leakyrelu_alpha_null(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_leakyrelu_alpha_wrong_size"))
                testing_aux_matmul_get_attr_bad_arg_leakyrelu_alpha_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_tanh_alpha_null"))
                testing_aux_matmul_get_attr_bad_arg_tanh_alpha_null(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_tanh_alpha_wrong_size"))
                testing_aux_matmul_get_attr_bad_arg_tanh_alpha_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_tanh_beta_null"))
                testing_aux_matmul_get_attr_bad_arg_tanh_beta_null(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_tanh_beta_wrong_size"))
                testing_aux_matmul_get_attr_bad_arg_tanh_beta_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_bias_pointer_wrong_size"))
                testing_aux_matmul_get_attr_bad_arg_bias_pointer_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_bias_stride_null"))
                testing_aux_matmul_get_attr_bad_arg_bias_stride_null(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_bias_stride_wrong_size"))
                testing_aux_matmul_get_attr_bad_arg_bias_stride_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_bias_type_null"))
                testing_aux_matmul_get_attr_bad_arg_bias_type_null(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_bias_type_wrong_size"))
                testing_aux_matmul_get_attr_bad_arg_bias_type_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_alpha_vector_scaling_null"))
                testing_aux_matmul_get_attr_bad_arg_alpha_vector_scaling_null(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_alpha_vector_scaling_wrong_size"))
                testing_aux_matmul_get_attr_bad_arg_alpha_vector_scaling_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg_beta_vector_scaling"))
                testing_aux_matmul_get_attr_bad_arg_beta_vector_scaling(arg);
            // --- individual sub-tests: aux_matmul_set_get_attr ---
            else if(!strcmp(arg.function, "aux_matmul_set_get_attr_relu"))
                testing_aux_matmul_set_get_attr_relu(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_get_attr_relu_upperbound"))
                testing_aux_matmul_set_get_attr_relu_upperbound(arg);
            // --- individual sub-tests: aux_matmul_alg_set_attr_bad_arg ---
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_null_handle"))
                testing_aux_matmul_alg_set_attr_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_uninit_handle"))
                testing_aux_matmul_alg_set_attr_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_null_alg_sel"))
                testing_aux_matmul_alg_set_attr_bad_arg_null_alg_sel(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_uninit_alg_sel"))
                testing_aux_matmul_alg_set_attr_bad_arg_uninit_alg_sel(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_config_max_id"))
                testing_aux_matmul_alg_set_attr_bad_arg_config_max_id(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_split_k"))
                testing_aux_matmul_alg_set_attr_bad_arg_split_k(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_null_data"))
                testing_aux_matmul_alg_set_attr_bad_arg_null_data(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_wrong_size"))
                testing_aux_matmul_alg_set_attr_bad_arg_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_config_id_out_of_range"))
                testing_aux_matmul_alg_set_attr_bad_arg_config_id_out_of_range(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_search_iterations_wrong_size"))
                testing_aux_matmul_alg_set_attr_bad_arg_search_iterations_wrong_size(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_search_iterations_zero"))
                testing_aux_matmul_alg_set_attr_bad_arg_search_iterations_zero(arg);
            // --- individual sub-tests: aux_matmul_alg_get_attr_bad_arg ---
            else if(!strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_null_handle"))
                testing_aux_matmul_alg_get_attr_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_uninit_handle"))
                testing_aux_matmul_alg_get_attr_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_null_alg_sel"))
                testing_aux_matmul_alg_get_attr_bad_arg_null_alg_sel(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_uninit_alg_sel"))
                testing_aux_matmul_alg_get_attr_bad_arg_uninit_alg_sel(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_null_data"))
                testing_aux_matmul_alg_get_attr_bad_arg_null_data(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_wrong_size"))
                testing_aux_matmul_alg_get_attr_bad_arg_wrong_size(arg);
            // --- individual sub-tests: aux_get_workspace_size_bad_arg ---
            else if(!strcmp(arg.function, "aux_get_workspace_size_bad_arg_uninit_plan"))
                testing_aux_get_workspace_size_bad_arg_uninit_plan(arg);
            // --- individual sub-tests: aux_matmul_alg_init_bad_arg ---
            else if(!strcmp(arg.function, "aux_matmul_alg_init_bad_arg_null_handle"))
                testing_aux_matmul_alg_init_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init_bad_arg_uninit_handle"))
                testing_aux_matmul_alg_init_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init_bad_arg_null_alg_sel"))
                testing_aux_matmul_alg_init_bad_arg_null_alg_sel(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init_bad_arg_null_matmul"))
                testing_aux_matmul_alg_init_bad_arg_null_matmul(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init_bad_arg_uninit_matmul"))
                testing_aux_matmul_alg_init_bad_arg_uninit_matmul(arg);
            // --- individual sub-tests: aux_matmul_alg_assign ---
            else if(!strcmp(arg.function, "aux_matmul_alg_assign_copy_value"))
                testing_aux_matmul_alg_assign_copy_value(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_assign_not_reference"))
                testing_aux_matmul_alg_assign_not_reference(arg);
            // --- individual sub-tests: aux_matmul_plan_destroy_bad_arg ---
            else if(!strcmp(arg.function, "aux_matmul_plan_destroy_bad_arg_null"))
                testing_aux_matmul_plan_destroy_bad_arg_null(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_destroy_bad_arg_uninit"))
                testing_aux_matmul_plan_destroy_bad_arg_uninit(arg);
            // --- individual sub-tests: aux_matmul_plan_init_bad_arg ---
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg_null_handle"))
                testing_aux_matmul_plan_init_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg_uninit_handle"))
                testing_aux_matmul_plan_init_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg_null_plan"))
                testing_aux_matmul_plan_init_bad_arg_null_plan(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg_null_matmul"))
                testing_aux_matmul_plan_init_bad_arg_null_matmul(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg_uninit_matmul"))
                testing_aux_matmul_plan_init_bad_arg_uninit_matmul(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg_null_alg_sel"))
                testing_aux_matmul_plan_init_bad_arg_null_alg_sel(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg_uninit_alg_sel"))
                testing_aux_matmul_plan_init_bad_arg_uninit_alg_sel(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg_mismatched_batches"))
                testing_aux_matmul_plan_init_bad_arg_mismatched_batches(arg);
            // --- individual sub-tests: aux_get_workspace_size_bad_arg ---
            else if(!strcmp(arg.function, "aux_get_workspace_size_bad_arg_null_handle"))
                testing_aux_get_workspace_size_bad_arg_null_handle(arg);
            else if(!strcmp(arg.function, "aux_get_workspace_size_bad_arg_uninit_handle"))
                testing_aux_get_workspace_size_bad_arg_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_get_workspace_size_bad_arg_null_plan"))
                testing_aux_get_workspace_size_bad_arg_null_plan(arg);
            else if(!strcmp(arg.function, "aux_get_workspace_size_bad_arg_null_size"))
                testing_aux_get_workspace_size_bad_arg_null_size(arg);
            // --- missing coverage: GetVersion bad-arg ---
            else if(!strcmp(arg.function, "aux_get_version_null_handle"))
                testing_aux_get_version_null_handle(arg);
            else if(!strcmp(arg.function, "aux_get_version_null_version"))
                testing_aux_get_version_null_version(arg);
            // --- missing coverage: GetProperty bad-arg ---
            else if(!strcmp(arg.function, "aux_get_property_null_value"))
                testing_aux_get_property_null_value(arg);
            // --- missing coverage: GetGitRevision ---
            else if(!strcmp(arg.function, "aux_get_git_revision_uninit_handle"))
                testing_aux_get_git_revision_uninit_handle(arg);
            else if(!strcmp(arg.function, "aux_get_git_revision_valid"))
                testing_aux_get_git_revision_valid(arg);
            // --- missing coverage: GetArchName ---
            else if(!strcmp(arg.function, "aux_get_arch_name"))
                testing_aux_get_arch_name(arg);
            else if(!strcmp(arg.function, "aux_get_arch_name_null"))
                testing_aux_get_arch_name_null(arg);
            // --- missing coverage: MatmulAlgSelectionDestroy ---
            else if(!strcmp(arg.function, "aux_matmul_alg_sel_destroy"))
                testing_aux_matmul_alg_sel_destroy(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_sel_destroy_bad_arg_null"))
                testing_aux_matmul_alg_sel_destroy_bad_arg_null(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_sel_destroy_bad_arg_uninit"))
                testing_aux_matmul_alg_sel_destroy_bad_arg_uninit(arg);
            // --- missing coverage: DenseDescriptorInit row order ---
            else if(!strcmp(arg.function, "aux_mat_dense_init_row_order"))
                testing_aux_mat_dense_init_row_order(arg);
            // --- missing coverage: matmul init with matB as sparse ---
            else if(!strcmp(arg.function, "aux_matmul_init_matB_sparse"))
                testing_aux_matmul_init_matB_sparse(arg);
            // --- missing coverage: CONFIG_MAX_ID get ---
            else if(!strcmp(arg.function, "aux_matmul_alg_get_attr_max_id"))
                testing_aux_matmul_alg_get_attr_max_id(arg);
            // --- missing coverage: SPLIT_K_MODE / SPLIT_K_BUFFERS ---
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_split_k_mode"))
                testing_aux_matmul_alg_set_attr_bad_arg_split_k_mode(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_split_k_buffers"))
                testing_aux_matmul_alg_set_attr_bad_arg_split_k_buffers(arg);
            // --- missing coverage: ALPHA_VECTOR_SCALING set ---
            else if(!strcmp(arg.function, "aux_matmul_set_attr_alpha_vector_scaling"))
                testing_aux_matmul_set_attr_alpha_vector_scaling(arg);
            // --- missing coverage: activation set/get round-trips ---
            else if(!strcmp(arg.function, "aux_matmul_set_get_attr_gelu"))
                testing_aux_matmul_set_get_attr_gelu(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_get_attr_abs"))
                testing_aux_matmul_set_get_attr_abs(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_get_attr_leakyrelu"))
                testing_aux_matmul_set_get_attr_leakyrelu(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_get_attr_tanh"))
                testing_aux_matmul_set_get_attr_tanh(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct aux_test : RocSparseLt_Test<aux_test, aux_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return hipsparselt_spmm_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            // Non-aggregated tests
            return !strcmp(arg.function, "aux_handle_init_bad_arg")
                   || !strcmp(arg.function, "aux_handle")
                   || !strcmp(arg.function, "aux_mat_dense_init_arg")
                   || !strcmp(arg.function, "aux_mat_structured_init")
                   || !strcmp(arg.function, "aux_matmul_init")
                   || !strcmp(arg.function, "aux_matmul_set_get_bias_vector")
                   || !strcmp(arg.function, "aux_matmul_alg_init")
                   || !strcmp(arg.function, "aux_matmul_plan_init")
                   || !strcmp(arg.function, "aux_get_workspace_size")
                   // Individual sub-tests: aux_get_version
                   || !strcmp(arg.function, "aux_get_version_match")
                   || !strcmp(arg.function, "aux_get_version_git_rev_null")
                   // Individual sub-tests: aux_handle_destroy_bad_arg
                   || !strcmp(arg.function, "aux_handle_destroy_bad_arg_uninit")
                   || !strcmp(arg.function, "aux_handle_destroy_bad_arg_null")
                   // Individual sub-tests: aux_mat_init_dense_bad_arg
                   || !strcmp(arg.function, "aux_mat_init_dense_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_mat_init_dense_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_mat_init_dense_bad_arg_null_descr")
                   || !strcmp(arg.function, "aux_mat_init_dense_bad_arg_zero_row")
                   || !strcmp(arg.function, "aux_mat_init_dense_bad_arg_zero_col")
                   || !strcmp(arg.function, "aux_mat_init_dense_bad_arg_zero_ld")
                   || !strcmp(arg.function, "aux_mat_init_dense_bad_arg_large_ld")
                   || !strcmp(arg.function, "aux_mat_init_dense_bad_arg_large_alignment")
                   // Individual sub-tests: aux_mat_init_structured_bad_arg
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_null_descr")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_zero_row")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_small_row")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_zero_col")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_small_col")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_zero_ld")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_unaligned_ld")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_unaligned_row")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_unsupported_type")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_large_ld")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg_large_alignment")
                   // Individual sub-tests: aux_mat_set_get_attr
                   || !strcmp(arg.function, "aux_mat_set_get_attr_num_batches")
                   || !strcmp(arg.function, "aux_mat_set_get_attr_batch_stride")
                   // Individual sub-tests: aux_mat_destroy_bad_arg
                   || !strcmp(arg.function, "aux_mat_destroy_bad_arg_uninit")
                   || !strcmp(arg.function, "aux_mat_destroy_bad_arg_null")
                   // Individual sub-tests: aux_mat_assign
                   || !strcmp(arg.function, "aux_mat_assign_copy_value")
                   || !strcmp(arg.function, "aux_mat_assign_not_reference")
                   // Individual sub-tests: aux_mat_set_attr_bad_arg
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg_null_descr")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg_uninit_descr")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg_null_data")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg_zero_batches")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg_wrong_size_batches")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg_null_stride")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg_invalid_stride")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg_wrong_size_stride")
                   // Individual sub-tests: aux_mat_get_attr_bad_arg
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg_null_descr")
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg_uninit_descr")
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg_null_data")
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg_wrong_size_batches")
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg_null_stride")
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg_wrong_size_stride")
                   // Individual sub-tests: aux_matmul_init_bad_arg
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_null_descr")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_conj_opA")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_conj_opB")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_uninit_matA")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_two_sparse")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_wrong_compute_type")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_structured_C")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_order")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_null_matA")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_uninit_matB")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_null_matB")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_uninit_matC")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_null_matC")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_uninit_matD")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_null_matD")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_same_op_int8")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_K")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_N")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_C_dim")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_D_dim")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_unsupported_A_type")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_B_type")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_C_type")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg_mismatched_D_type")
                   // Individual sub-tests: aux_matmul_assign
                   || !strcmp(arg.function, "aux_matmul_assign_copy_value")
                   || !strcmp(arg.function, "aux_matmul_assign_not_reference")
                   // Individual sub-tests: aux_matmul_set_attr_bad_arg
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_null_matmul")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_uninit_matmul")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_null_data")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_relu_upperbound_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_sigmoid_int8")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_bias_pointer_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_bias_stride_invalid")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg_bias_type")
                   // Individual sub-tests: aux_matmul_get_attr_bad_arg
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_null_matmul")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_uninit_matmul")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_null_data")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_relu_upperbound_null")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_relu_upperbound_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_relu_threshold_null")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_relu_threshold_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_leakyrelu_alpha_null")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_leakyrelu_alpha_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_tanh_alpha_null")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_tanh_alpha_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_tanh_beta_null")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_tanh_beta_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_bias_pointer_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_bias_stride_null")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_bias_stride_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_bias_type_null")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_bias_type_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_alpha_vector_scaling_null")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_alpha_vector_scaling_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg_beta_vector_scaling")
                   // Individual sub-tests: aux_matmul_set_get_attr
                   || !strcmp(arg.function, "aux_matmul_set_get_attr_relu")
                   || !strcmp(arg.function, "aux_matmul_set_get_attr_relu_upperbound")
                   // Individual sub-tests: aux_matmul_alg_set_attr_bad_arg
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_null_alg_sel")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_uninit_alg_sel")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_config_max_id")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_split_k")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_null_data")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_config_id_out_of_range")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_search_iterations_wrong_size")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_search_iterations_zero")
                   // Individual sub-tests: aux_matmul_alg_get_attr_bad_arg
                   || !strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_null_alg_sel")
                   || !strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_uninit_alg_sel")
                   || !strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_null_data")
                   || !strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg_wrong_size")
                   // Individual sub-tests: aux_get_workspace_size_bad_arg
                   || !strcmp(arg.function, "aux_get_workspace_size_bad_arg_uninit_plan")
                   // Individual sub-tests: aux_matmul_alg_init_bad_arg
                   || !strcmp(arg.function, "aux_matmul_alg_init_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_matmul_alg_init_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_matmul_alg_init_bad_arg_null_alg_sel")
                   || !strcmp(arg.function, "aux_matmul_alg_init_bad_arg_null_matmul")
                   || !strcmp(arg.function, "aux_matmul_alg_init_bad_arg_uninit_matmul")
                   // Individual sub-tests: aux_matmul_alg_assign
                   || !strcmp(arg.function, "aux_matmul_alg_assign_copy_value")
                   || !strcmp(arg.function, "aux_matmul_alg_assign_not_reference")
                   // Individual sub-tests: aux_matmul_plan_destroy_bad_arg
                   || !strcmp(arg.function, "aux_matmul_plan_destroy_bad_arg_null")
                   || !strcmp(arg.function, "aux_matmul_plan_destroy_bad_arg_uninit")
                   // Individual sub-tests: aux_matmul_plan_init_bad_arg
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg_null_plan")
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg_null_matmul")
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg_uninit_matmul")
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg_null_alg_sel")
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg_uninit_alg_sel")
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg_mismatched_batches")
                   // Individual sub-tests: aux_get_workspace_size_bad_arg
                   || !strcmp(arg.function, "aux_get_workspace_size_bad_arg_null_handle")
                   || !strcmp(arg.function, "aux_get_workspace_size_bad_arg_uninit_handle")
                   || !strcmp(arg.function, "aux_get_workspace_size_bad_arg_null_plan")
                   || !strcmp(arg.function, "aux_get_workspace_size_bad_arg_null_size")
                   // Missing coverage: GetVersion bad-arg
                   || !strcmp(arg.function, "aux_get_version_null_handle")
                   || !strcmp(arg.function, "aux_get_version_null_version")
                   // Missing coverage: GetProperty bad-arg
                   || !strcmp(arg.function, "aux_get_property_null_value")
                   // Missing coverage: GetGitRevision
                   || !strcmp(arg.function, "aux_get_git_revision_uninit_handle")
                   || !strcmp(arg.function, "aux_get_git_revision_valid")
                   // Missing coverage: GetArchName
                   || !strcmp(arg.function, "aux_get_arch_name")
                   || !strcmp(arg.function, "aux_get_arch_name_null")
                   // Missing coverage: MatmulAlgSelectionDestroy
                   || !strcmp(arg.function, "aux_matmul_alg_sel_destroy")
                   || !strcmp(arg.function, "aux_matmul_alg_sel_destroy_bad_arg_null")
                   || !strcmp(arg.function, "aux_matmul_alg_sel_destroy_bad_arg_uninit")
                   // Missing coverage: DenseDescriptorInit row order
                   || !strcmp(arg.function, "aux_mat_dense_init_row_order")
                   // Missing coverage: matmul init with matB as sparse
                   || !strcmp(arg.function, "aux_matmul_init_matB_sparse")
                   // Missing coverage: CONFIG_MAX_ID get
                   || !strcmp(arg.function, "aux_matmul_alg_get_attr_max_id")
                   // Missing coverage: SPLIT_K_MODE / SPLIT_K_BUFFERS
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_split_k_mode")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg_split_k_buffers")
                   // Missing coverage: ALPHA_VECTOR_SCALING set
                   || !strcmp(arg.function, "aux_matmul_set_attr_alpha_vector_scaling")
                   // Missing coverage: activation set/get round-trips
                   || !strcmp(arg.function, "aux_matmul_set_get_attr_gelu")
                   || !strcmp(arg.function, "aux_matmul_set_get_attr_abs")
                   || !strcmp(arg.function, "aux_matmul_set_get_attr_leakyrelu")
                   || !strcmp(arg.function, "aux_matmul_set_get_attr_tanh");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocSparseLt_TestName<aux_test> name(arg.name);

            name << hip_datatype_to_string(arg.a_type) << hip_datatype_to_string(arg.b_type)
                 << hip_datatype_to_string(arg.c_type) << hip_datatype_to_string(arg.d_type);

            return std::move(name);
        }
    };

    TEST_P(aux_test, conversion)
    {
        RUN_TEST_ON_THREADS_STREAMS(hipsparselt_spmm_dispatch<aux_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(aux_test);

} // namespace
