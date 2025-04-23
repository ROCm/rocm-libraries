#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
import sys
if sys.version_info < (3, 0):
    sys.exit()

import argparse
import os
import unittest
import onnx
import onnx.backend.test
import numpy as np
from onnx_migraphx.backend import MIGraphXBackend as c2
from packaging import version

pytest_plugins = 'onnx.backend.test.report',


class MIGraphXBackendTest(onnx.backend.test.BackendTest):
    def __init__(self, backend, parent_module=None):
        super(MIGraphXBackendTest, self).__init__(backend, parent_module)

    @classmethod
    def assert_similar_outputs(cls,
                               ref_outputs,
                               outputs,
                               rtol,
                               atol,
                               model_dir=None):
        prog_string = c2.get_program()
        np.testing.assert_equal(len(ref_outputs),
                                len(outputs),
                                err_msg=prog_string)
        for i in range(len(outputs)):
            np.testing.assert_equal(ref_outputs[i].dtype,
                                    outputs[i].dtype,
                                    err_msg=prog_string)
            if ref_outputs[i].dtype == object:
                np.testing.assert_array_equal(ref_outputs[i],
                                              outputs[i],
                                              err_msg=prog_string)
            else:
                np.testing.assert_allclose(ref_outputs[i],
                                           outputs[i],
                                           rtol=1e-3,
                                           atol=1e-5,
                                           err_msg=prog_string)


def disabled_tests_onnx_1_7_0(backend_test):
    # fails
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_maxpool_with_argmax_2d_precomputed_pads_cpu')
    backend_test.exclude(
        r'test_maxpool_with_argmax_2d_precomputed_strides_cpu')
    backend_test.exclude(r'test_nonmaxsuppression_center_point_box_format_cpu')
    backend_test.exclude(r'test_nonmaxsuppression_flipped_coordinates_cpu')
    backend_test.exclude(r'test_nonmaxsuppression_identical_boxes_cpu')
    backend_test.exclude(r'test_nonmaxsuppression_limit_output_size_cpu')
    backend_test.exclude(
        r'test_nonmaxsuppression_suppress_by_IOU_and_scores_cpu')
    backend_test.exclude(r'test_nonmaxsuppression_suppress_by_IOU_cpu')
    backend_test.exclude(r'test_nonmaxsuppression_two_batches_cpu')
    backend_test.exclude(r'test_nonmaxsuppression_two_classes_cpu')
    backend_test.exclude(r'test_nonzero_example_cpu')

    # from OnnxBackendPyTorchConvertedModelTest
    backend_test.exclude(r'test_ConvTranspose2d_cpu')
    backend_test.exclude(r'test_ConvTranspose2d_no_bias_cpu')

    # from OnnxBackendPyTorchOperatorModelTest
    backend_test.exclude(r'test_operator_add_broadcast_cpu')
    backend_test.exclude(r'test_operator_add_size1_right_broadcast_cpu')
    backend_test.exclude(r'test_operator_addconstant_cpu')
    backend_test.exclude(r'test_operator_convtranspose_cpu')

    # errors
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_bitshift_left_uint16_cpu')
    backend_test.exclude(r'test_bitshift_left_uint32_cpu')
    backend_test.exclude(r'test_bitshift_left_uint64_cpu')
    backend_test.exclude(r'test_bitshift_left_uint8_cpu')
    backend_test.exclude(r'test_bitshift_right_uint16_cpu')
    backend_test.exclude(r'test_bitshift_right_uint32_cpu')
    backend_test.exclude(r'test_bitshift_right_uint64_cpu')
    backend_test.exclude(r'test_bitshift_right_uint8_cpu')
    backend_test.exclude(r'test_cast_FLOAT_to_STRING_cpu')
    backend_test.exclude(r'test_cast_STRING_to_FLOAT_cpu')
    backend_test.exclude(r'test_compress_0_cpu')
    backend_test.exclude(r'test_compress_1_cpu')
    backend_test.exclude(r'test_compress_default_axis_cpu')
    backend_test.exclude(r'test_compress_negative_axis_cpu')
    backend_test.exclude(r'test_constant_pad_cpu')
    backend_test.exclude(r'test_convinteger_with_padding_cpu')
    backend_test.exclude(r'test_convtranspose_1d_cpu')
    backend_test.exclude(r'test_det_2d_cpu')
    backend_test.exclude(r'test_det_nd_cpu')
    backend_test.exclude(r'test_edge_pad_cpu')
    backend_test.exclude(r'test_maxunpool_export_with_output_shape_cpu')
    backend_test.exclude(r'test_maxunpool_export_without_output_shape_cpu')
    backend_test.exclude(r'test_qlinearmatmul_2D_cpu')
    backend_test.exclude(r'test_qlinearmatmul_3D_cpu')
    backend_test.exclude(r'test_range_float_type_positive_delta_expanded_cpu')
    backend_test.exclude(r'test_range_int32_type_negative_delta_expanded_cpu')
    backend_test.exclude(r'test_reflect_pad_cpu')
    backend_test.exclude(
        r'test_resize_downsample_scales_cubic_A_n0p5_exclude_outside_cpu')
    backend_test.exclude(
        r'test_resize_downsample_scales_cubic_align_corners_cpu')
    backend_test.exclude(r'test_resize_downsample_scales_cubic_cpu')
    backend_test.exclude(
        r'test_resize_downsample_scales_linear_align_corners_cpu')
    backend_test.exclude(r'test_resize_downsample_scales_linear_cpu')
    backend_test.exclude(r'test_resize_downsample_sizes_cubic_cpu')
    backend_test.exclude(
        r'test_resize_downsample_sizes_linear_pytorch_half_pixel_cpu')
    backend_test.exclude(r'test_resize_tf_crop_and_resize_cpu')
    backend_test.exclude(
        r'test_resize_upsample_scales_cubic_A_n0p5_exclude_outside_cpu')
    backend_test.exclude(
        r'test_resize_upsample_scales_cubic_align_corners_cpu')
    backend_test.exclude(r'test_resize_upsample_scales_cubic_asymmetric_cpu')
    backend_test.exclude(r'test_resize_upsample_scales_cubic_cpu')
    backend_test.exclude(
        r'test_resize_upsample_scales_linear_align_corners_cpu')
    backend_test.exclude(r'test_resize_upsample_scales_linear_cpu')
    backend_test.exclude(r'test_resize_upsample_sizes_cubic_cpu')
    backend_test.exclude(r'test_reversesequence_batch_cpu')
    backend_test.exclude(r'test_reversesequence_time_cpu')
    backend_test.exclude(r'test_scan_sum_cpu')
    backend_test.exclude(r'test_slice_cpu')
    backend_test.exclude(r'test_slice_end_out_of_bounds_cpu')
    backend_test.exclude(r'test_slice_neg_cpu')
    backend_test.exclude(r'test_slice_neg_steps_cpu')
    backend_test.exclude(r'test_slice_start_out_of_bounds_cpu')
    backend_test.exclude(
        r'test_strnormalizer_export_monday_casesensintive_lower_cpu')
    backend_test.exclude(
        r'test_strnormalizer_export_monday_casesensintive_nochangecase_cpu')
    backend_test.exclude(
        r'test_strnormalizer_export_monday_casesensintive_upper_cpu')
    backend_test.exclude(r'test_strnormalizer_export_monday_empty_output_cpu')
    backend_test.exclude(
        r'test_strnormalizer_export_monday_insensintive_upper_twodim_cpu')
    backend_test.exclude(r'test_strnormalizer_nostopwords_nochangecase_cpu')
    backend_test.exclude(
        r'test_tfidfvectorizer_tf_batch_onlybigrams_skip0_cpu')
    backend_test.exclude(
        r'test_tfidfvectorizer_tf_batch_onlybigrams_skip5_cpu')
    backend_test.exclude(
        r'test_tfidfvectorizer_tf_batch_uniandbigrams_skip5_cpu')
    backend_test.exclude(r'test_tfidfvectorizer_tf_only_bigrams_skip0_cpu')
    backend_test.exclude(r'test_tfidfvectorizer_tf_onlybigrams_levelempty_cpu')
    backend_test.exclude(r'test_tfidfvectorizer_tf_onlybigrams_skip5_cpu')
    backend_test.exclude(r'test_tfidfvectorizer_tf_uniandbigrams_skip5_cpu')
    backend_test.exclude(r'test_top_k_cpu')
    backend_test.exclude(r'test_top_k_negative_axis_cpu')
    backend_test.exclude(r'test_top_k_smallest_cpu')
    backend_test.exclude(r'test_unique_sorted_with_axis_3d_cpu')
    backend_test.exclude(r'test_unique_sorted_with_negative_axis_cpu')
    backend_test.exclude(r'test_upsample_nearest_cpu')

    # from OnnxBackendPyTorchConvertedModelTest
    backend_test.exclude(r'test_PReLU_1d_multiparam_cpu')
    backend_test.exclude(r'test_PReLU_2d_multiparam_cpu')
    backend_test.exclude(r'test_PReLU_3d_multiparam_cpu')
    backend_test.exclude(r'test_ReplicationPad2d_cpu')

    # from OnnxBackendPyTorchOperatorModelTest
    backend_test.exclude(r'test_operator_add_size1_broadcast_cpu')
    backend_test.exclude(r'test_operator_add_size1_singleton_broadcast_cpu')

    # from OnnxBackendSimpleModelTest
    backend_test.exclude(r'test_gradient_of_add_and_mul_cpu')
    backend_test.exclude(r'test_gradient_of_add_cpu')
    backend_test.exclude(r'test_sequence_model1_cpu')
    backend_test.exclude(r'test_sequence_model2_cpu')
    backend_test.exclude(r'test_sequence_model3_cpu')
    backend_test.exclude(r'test_sequence_model4_cpu')
    backend_test.exclude(r'test_sequence_model5_cpu')
    backend_test.exclude(r'test_sequence_model6_cpu')
    backend_test.exclude(r'test_sequence_model7_cpu')
    backend_test.exclude(r'test_sequence_model8_cpu')
    backend_test.exclude(r'test_strnorm_model_monday_casesensintive_lower_cpu')
    backend_test.exclude(
        r'test_strnorm_model_monday_casesensintive_nochangecase_cpu')
    backend_test.exclude(r'test_strnorm_model_monday_casesensintive_upper_cpu')
    backend_test.exclude(r'test_strnorm_model_monday_empty_output_cpu')
    backend_test.exclude(
        r'test_strnorm_model_monday_insensintive_upper_twodim_cpu')
    backend_test.exclude(r'test_strnorm_model_nostopwords_nochangecase_cpu')


def disabled_tests_onnx_1_8_0(backend_test):
    # errors
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_cast_BFLOAT16_to_FLOAT_cpu')
    backend_test.exclude(r'test_cast_FLOAT_to_BFLOAT16_cpu')
    backend_test.exclude(r'test_if_seq_cpu')
    backend_test.exclude(r'test_loop11_cpu')
    backend_test.exclude(r'test_loop13_seq_cpu')
    backend_test.exclude(r'test_nllloss_NC_cpu')
    backend_test.exclude(r'test_nllloss_NCd1_cpu')
    backend_test.exclude(r'test_nllloss_NCd1_ii_cpu')
    backend_test.exclude(r'test_nllloss_NCd1_mean_weight_negative_ii_cpu')
    backend_test.exclude(r'test_nllloss_NCd1_weight_cpu')
    backend_test.exclude(r'test_nllloss_NCd1_weight_ii_cpu')
    backend_test.exclude(r'test_nllloss_NCd1d2_cpu')
    backend_test.exclude(
        r'test_nllloss_NCd1d2_no_weight_reduction_mean_ii_cpu')
    backend_test.exclude(r'test_nllloss_NCd1d2_reduction_mean_cpu')
    backend_test.exclude(r'test_nllloss_NCd1d2_reduction_sum_cpu')
    backend_test.exclude(r'test_nllloss_NCd1d2_with_weight_cpu')
    backend_test.exclude(r'test_nllloss_NCd1d2_with_weight_reduction_mean_cpu')
    backend_test.exclude(r'test_nllloss_NCd1d2_with_weight_reduction_sum_cpu')
    backend_test.exclude(
        r'test_nllloss_NCd1d2_with_weight_reduction_sum_ii_cpu')
    backend_test.exclude(
        r'test_nllloss_NCd1d2d3_none_no_weight_negative_ii_cpu')
    backend_test.exclude(r'test_nllloss_NCd1d2d3_sum_weight_high_ii_cpu')
    backend_test.exclude(r'test_nllloss_NCd1d2d3d4d5_mean_weight_cpu')
    backend_test.exclude(r'test_nllloss_NCd1d2d3d4d5_none_no_weight_cpu')
    backend_test.exclude(r'test_sce_NCd1_mean_weight_negative_ii_cpu')
    backend_test.exclude(r'test_sce_NCd1_mean_weight_negative_ii_expanded_cpu')
    backend_test.exclude(r'test_sce_NCd1_mean_weight_negative_ii_log_prob_cpu')
    backend_test.exclude(
        r'test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_NCd1d2d3_none_no_weight_negative_ii_cpu')
    backend_test.exclude(
        r'test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded_cpu')
    backend_test.exclude(
        r'test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_cpu')
    backend_test.exclude(
        r'test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_NCd1d2d3_sum_weight_high_ii_cpu')
    backend_test.exclude(r'test_sce_NCd1d2d3_sum_weight_high_ii_expanded_cpu')
    backend_test.exclude(r'test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_cpu')
    backend_test.exclude(
        r'test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_NCd1d2d3d4d5_mean_weight_cpu')
    backend_test.exclude(r'test_sce_NCd1d2d3d4d5_mean_weight_expanded_cpu')
    backend_test.exclude(r'test_sce_NCd1d2d3d4d5_mean_weight_log_prob_cpu')
    backend_test.exclude(
        r'test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_NCd1d2d3d4d5_none_no_weight_cpu')
    backend_test.exclude(r'test_sce_NCd1d2d3d4d5_none_no_weight_expanded_cpu')
    backend_test.exclude(r'test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_cpu')
    backend_test.exclude(
        r'test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_3d_cpu')
    backend_test.exclude(r'test_sce_mean_3d_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_3d_log_prob_cpu')
    backend_test.exclude(r'test_sce_mean_3d_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_cpu')
    backend_test.exclude(r'test_sce_mean_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_log_prob_cpu')
    backend_test.exclude(r'test_sce_mean_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_no_weight_ii_3d_cpu')
    backend_test.exclude(r'test_sce_mean_no_weight_ii_3d_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_no_weight_ii_3d_log_prob_cpu')
    backend_test.exclude(
        r'test_sce_mean_no_weight_ii_3d_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_no_weight_ii_4d_cpu')
    backend_test.exclude(r'test_sce_mean_no_weight_ii_4d_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_no_weight_ii_4d_log_prob_cpu')
    backend_test.exclude(
        r'test_sce_mean_no_weight_ii_4d_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_no_weight_ii_cpu')
    backend_test.exclude(r'test_sce_mean_no_weight_ii_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_no_weight_ii_log_prob_cpu')
    backend_test.exclude(r'test_sce_mean_no_weight_ii_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_weight_cpu')
    backend_test.exclude(r'test_sce_mean_weight_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_3d_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_3d_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_3d_log_prob_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_3d_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_4d_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_4d_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_4d_log_prob_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_4d_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_log_prob_cpu')
    backend_test.exclude(r'test_sce_mean_weight_ii_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_mean_weight_log_prob_cpu')
    backend_test.exclude(r'test_sce_mean_weight_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_none_cpu')
    backend_test.exclude(r'test_sce_none_expanded_cpu')
    backend_test.exclude(r'test_sce_none_log_prob_cpu')
    backend_test.exclude(r'test_sce_none_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_none_weights_cpu')
    backend_test.exclude(r'test_sce_none_weights_expanded_cpu')
    backend_test.exclude(r'test_sce_none_weights_log_prob_cpu')
    backend_test.exclude(r'test_sce_none_weights_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sce_sum_cpu')
    backend_test.exclude(r'test_sce_sum_expanded_cpu')
    backend_test.exclude(r'test_sce_sum_log_prob_cpu')
    backend_test.exclude(r'test_sce_sum_log_prob_expanded_cpu')
    backend_test.exclude(r'test_sequence_insert_at_back_cpu')
    backend_test.exclude(r'test_sequence_insert_at_front_cpu')


def disabled_tests_onnx_1_9_0(backend_test):
    # fails
    # from OnnxBackendPyTorchConvertedModelTest
    # MaxPool dialtion is partially supported on GPU by a workaround
    # But these tests require too large allocations to work properly
    backend_test.exclude(r'test_MaxPool1d_stride_padding_dilation_cpu')
    backend_test.exclude(r'test_MaxPool2d_stride_padding_dilation_cpu')

    # errors
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_convinteger_without_padding_cpu')
    backend_test.exclude(r'test_convtranspose_autopad_same_cpu')
    backend_test.exclude(r'test_identity_sequence_cpu')
    backend_test.exclude(r'test_tril_neg_cpu')
    backend_test.exclude(r'test_tril_out_neg_cpu')
    backend_test.exclude(r'test_tril_out_pos_cpu')
    backend_test.exclude(r'test_tril_pos_cpu')
    backend_test.exclude(r'test_tril_square_neg_cpu')
    backend_test.exclude(r'test_tril_zero_cpu')
    backend_test.exclude(r'test_triu_neg_cpu')
    backend_test.exclude(r'test_triu_one_row_cpu')
    backend_test.exclude(r'test_triu_out_neg_out_cpu')
    backend_test.exclude(r'test_triu_out_pos_cpu')
    backend_test.exclude(r'test_triu_pos_cpu')
    backend_test.exclude(r'test_triu_square_neg_cpu')
    backend_test.exclude(r'test_triu_zero_cpu')


def disabled_tests_onnx_1_10_0(backend_test):
    # fails
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_bernoulli_double_expanded_cpu')
    backend_test.exclude(r'test_bernoulli_expanded_cpu')
    backend_test.exclude(r'test_bernoulli_seed_expanded_cpu')

    # errors
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_bernoulli_cpu')
    backend_test.exclude(r'test_bernoulli_double_cpu')
    backend_test.exclude(r'test_bernoulli_seed_cpu')
    backend_test.exclude(r'test_castlike_BFLOAT16_to_FLOAT_cpu')
    backend_test.exclude(r'test_castlike_BFLOAT16_to_FLOAT_expanded_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_BFLOAT16_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_BFLOAT16_expanded_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_STRING_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_STRING_expanded_cpu')
    backend_test.exclude(r'test_castlike_STRING_to_FLOAT_cpu')
    backend_test.exclude(r'test_castlike_STRING_to_FLOAT_expanded_cpu')
    backend_test.exclude(r'test_optional_get_element_sequence_cpu')


def disabled_tests_onnx_1_11_0(backend_test):
    # errors
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_identity_opt_cpu')
    backend_test.exclude(r'test_if_opt_cpu')
    backend_test.exclude(r'test_loop16_seq_none_cpu')


def disabled_tests_onnx_1_12_0(backend_test):
    # errors
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_blackmanwindow_cpu')
    backend_test.exclude(r'test_blackmanwindow_expanded_cpu')
    backend_test.exclude(r'test_blackmanwindow_symmetric_cpu')
    backend_test.exclude(r'test_blackmanwindow_symmetric_expanded_cpu')
    backend_test.exclude(r'test_dft_axis_cpu')
    backend_test.exclude(r'test_dft_cpu')
    backend_test.exclude(r'test_dft_inverse_cpu')
    backend_test.exclude(r'test_hammingwindow_cpu')
    backend_test.exclude(r'test_hammingwindow_expanded_cpu')
    backend_test.exclude(r'test_hammingwindow_symmetric_cpu')
    backend_test.exclude(r'test_hammingwindow_symmetric_expanded_cpu')
    backend_test.exclude(r'test_hannwindow_cpu')
    backend_test.exclude(r'test_hannwindow_expanded_cpu')
    backend_test.exclude(r'test_hannwindow_symmetric_cpu')
    backend_test.exclude(r'test_hannwindow_symmetric_expanded_cpu')
    backend_test.exclude(r'test_melweightmatrix_cpu')
    backend_test.exclude(r'test_sequence_map_add_1_sequence_1_tensor_cpu')
    backend_test.exclude(
        r'test_sequence_map_add_1_sequence_1_tensor_expanded_cpu')
    backend_test.exclude(r'test_sequence_map_add_2_sequences_cpu')
    backend_test.exclude(r'test_sequence_map_add_2_sequences_expanded_cpu')
    backend_test.exclude(r'test_sequence_map_extract_shapes_cpu')
    backend_test.exclude(r'test_sequence_map_extract_shapes_expanded_cpu')
    backend_test.exclude(r'test_sequence_map_identity_1_sequence_1_tensor_cpu')
    backend_test.exclude(
        r'test_sequence_map_identity_1_sequence_1_tensor_expanded_cpu')
    backend_test.exclude(r'test_sequence_map_identity_1_sequence_cpu')
    backend_test.exclude(r'test_sequence_map_identity_1_sequence_expanded_cpu')
    backend_test.exclude(r'test_sequence_map_identity_2_sequences_cpu')
    backend_test.exclude(
        r'test_sequence_map_identity_2_sequences_expanded_cpu')
    backend_test.exclude(r'test_stft_cpu')
    backend_test.exclude(r'test_stft_with_window_cpu')


def disabled_tests_onnx_1_13_0(backend_test):
    # fails
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_reduce_l1_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_l1_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_l1_keep_dims_example_cpu')
    backend_test.exclude(r'test_reduce_l1_keep_dims_random_cpu')
    backend_test.exclude(r'test_reduce_l1_negative_axes_keep_dims_example_cpu')
    backend_test.exclude(r'test_reduce_l1_negative_axes_keep_dims_random_cpu')
    backend_test.exclude(r'test_reduce_l2_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_l2_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_l2_keep_dims_example_cpu')
    backend_test.exclude(r'test_reduce_l2_keep_dims_random_cpu')
    backend_test.exclude(r'test_reduce_l2_negative_axes_keep_dims_example_cpu')
    backend_test.exclude(r'test_reduce_l2_negative_axes_keep_dims_random_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_log_sum_exp_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_log_sum_exp_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_log_sum_exp_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_negative_axes_keepdims_example_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_square_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_square_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_square_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_square_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_negative_axes_keepdims_example_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_negative_axes_keepdims_random_cpu')

    # errors
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_bitwise_and_i16_3d_cpu')
    backend_test.exclude(r'test_bitwise_and_i32_2d_cpu')
    backend_test.exclude(r'test_bitwise_and_ui64_bcast_3v1d_cpu')
    backend_test.exclude(r'test_bitwise_and_ui8_bcast_4v3d_cpu')
    backend_test.exclude(r'test_bitwise_not_2d_cpu')
    backend_test.exclude(r'test_bitwise_not_3d_cpu')
    backend_test.exclude(r'test_bitwise_not_4d_cpu')
    backend_test.exclude(r'test_bitwise_or_i16_4d_cpu')
    backend_test.exclude(r'test_bitwise_or_i32_2d_cpu')
    backend_test.exclude(r'test_bitwise_or_ui64_bcast_3v1d_cpu')
    backend_test.exclude(r'test_bitwise_or_ui8_bcast_4v3d_cpu')
    backend_test.exclude(r'test_bitwise_xor_i16_3d_cpu')
    backend_test.exclude(r'test_bitwise_xor_i32_2d_cpu')
    backend_test.exclude(r'test_bitwise_xor_ui64_bcast_3v1d_cpu')
    backend_test.exclude(r'test_bitwise_xor_ui8_bcast_4v3d_cpu')
    backend_test.exclude(r'test_center_crop_pad_crop_and_pad_cpu')
    backend_test.exclude(r'test_center_crop_pad_crop_and_pad_expanded_cpu')
    backend_test.exclude(r'test_center_crop_pad_crop_axes_chw_cpu')
    backend_test.exclude(r'test_center_crop_pad_crop_axes_chw_expanded_cpu')
    backend_test.exclude(r'test_center_crop_pad_crop_axes_hwc_cpu')
    backend_test.exclude(r'test_center_crop_pad_crop_axes_hwc_expanded_cpu')
    backend_test.exclude(r'test_center_crop_pad_crop_cpu')
    backend_test.exclude(r'test_center_crop_pad_crop_expanded_cpu')
    backend_test.exclude(r'test_center_crop_pad_pad_cpu')
    backend_test.exclude(r'test_center_crop_pad_pad_expanded_cpu')
    backend_test.exclude(r'test_col2im_5d_cpu')
    backend_test.exclude(r'test_col2im_cpu')
    backend_test.exclude(r'test_col2im_dilations_cpu')
    backend_test.exclude(r'test_col2im_pads_cpu')
    backend_test.exclude(r'test_col2im_strides_cpu')
    backend_test.exclude(r'test_constant_pad_axes_cpu')
    backend_test.exclude(r'test_mish_cpu')
    backend_test.exclude(r'test_optional_get_element_optional_sequence_cpu')
    backend_test.exclude(r'test_optional_get_element_optional_tensor_cpu')
    backend_test.exclude(r'test_optional_get_element_tensor_cpu')
    backend_test.exclude(
        r'test_optional_has_element_empty_no_input_name_optional_input_cpu')
    backend_test.exclude(
        r'test_optional_has_element_empty_no_input_name_tensor_input_cpu')
    backend_test.exclude(
        r'test_optional_has_element_empty_no_input_optional_input_cpu')
    backend_test.exclude(
        r'test_optional_has_element_empty_no_input_tensor_input_cpu')
    backend_test.exclude(r'test_optional_has_element_empty_optional_input_cpu')
    backend_test.exclude(r'test_optional_has_element_optional_input_cpu')
    backend_test.exclude(r'test_optional_has_element_tensor_input_cpu')
    backend_test.exclude(r'test_reduce_l1_default_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_l1_default_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_l2_default_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_l2_default_axes_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_default_axes_keepdims_example_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_default_axes_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_default_axes_keepdims_example_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_default_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_resize_downsample_scales_cubic_antialias_cpu')
    backend_test.exclude(r'test_resize_downsample_scales_linear_antialias_cpu')
    backend_test.exclude(r'test_resize_downsample_sizes_cubic_antialias_cpu')
    backend_test.exclude(r'test_resize_downsample_sizes_linear_antialias_cpu')
    backend_test.exclude(
        r'test_resize_downsample_sizes_nearest_not_larger_cpu')
    backend_test.exclude(
        r'test_resize_downsample_sizes_nearest_not_smaller_cpu')
    backend_test.exclude(r'test_resize_tf_crop_and_resize_axes_2_3_cpu')
    backend_test.exclude(r'test_resize_tf_crop_and_resize_axes_3_2_cpu')
    backend_test.exclude(r'test_resize_upsample_scales_nearest_axes_2_3_cpu')
    backend_test.exclude(r'test_resize_upsample_scales_nearest_axes_3_2_cpu')
    backend_test.exclude(r'test_resize_upsample_sizes_nearest_axes_2_3_cpu')
    backend_test.exclude(r'test_resize_upsample_sizes_nearest_axes_3_2_cpu')
    backend_test.exclude(r'test_resize_upsample_sizes_nearest_not_larger_cpu')


def disabled_tests_onnx_1_14_0(backend_test):
    # errors
    # from OnnxBackendNodeModelTest
    backend_test.exclude(r'test_basic_deform_conv_with_padding_cpu')
    backend_test.exclude(r'test_basic_deform_conv_without_padding_cpu')
    backend_test.exclude(r'test_center_crop_pad_crop_negative_axes_hwc_cpu')
    backend_test.exclude(
        r'test_center_crop_pad_crop_negative_axes_hwc_expanded_cpu')
    backend_test.exclude(r'test_constant_pad_negative_axes_cpu')
    backend_test.exclude(r'test_deform_conv_with_mask_bias_cpu')
    backend_test.exclude(r'test_deform_conv_with_multiple_offset_groups_cpu')
    backend_test.exclude(r'test_equal_string_broadcast_cpu')
    backend_test.exclude(r'test_equal_string_cpu')
    backend_test.exclude(r'test_lppool_1d_default_cpu')
    backend_test.exclude(r'test_lppool_2d_default_cpu')
    backend_test.exclude(r'test_lppool_2d_dilations_cpu')
    backend_test.exclude(r'test_lppool_2d_pads_cpu')
    backend_test.exclude(r'test_lppool_2d_same_lower_cpu')
    backend_test.exclude(r'test_lppool_2d_same_upper_cpu')
    backend_test.exclude(r'test_lppool_2d_strides_cpu')
    backend_test.exclude(r'test_lppool_3d_default_cpu')
    backend_test.exclude(
        r'test_resize_downsample_scales_linear_half_pixel_symmetric_cpu')
    backend_test.exclude(
        r'test_resize_upsample_scales_linear_half_pixel_symmetric_cpu')
    backend_test.exclude(r'test_split_to_sequence_1_cpu')
    backend_test.exclude(r'test_split_to_sequence_2_cpu')
    backend_test.exclude(r'test_split_to_sequence_nokeepdims_cpu')
    backend_test.exclude(r'test_wrap_pad_cpu')


def disabled_tests_onnx_1_16_0(backend_test):
    backend_test.exclude(r'test_dft_axis_opset19_cpu')
    backend_test.exclude(r'test_dft_inverse_opset19_cpu')
    backend_test.exclude(r'test_dft_opset19_cpu')
    backend_test.exclude(
        r'test_gridsample_bicubic_align_corners_0_additional_1_cpu')
    backend_test.exclude(
        r'test_gridsample_bicubic_align_corners_1_additional_1_cpu')
    backend_test.exclude(
        r'test_gridsample_volumetric_bilinear_align_corners_0_cpu')
    backend_test.exclude(
        r'test_gridsample_volumetric_bilinear_align_corners_1_cpu')
    backend_test.exclude(
        r'test_gridsample_volumetric_nearest_align_corners_0_cpu')
    backend_test.exclude(
        r'test_gridsample_volumetric_nearest_align_corners_1_cpu')
    backend_test.exclude(r'test_group_normalization_epsilon_cpu')
    backend_test.exclude(r'test_group_normalization_example_cpu')
    backend_test.exclude(r'test_quantizelinear_int16_cpu')
    backend_test.exclude(r'test_quantizelinear_uint16_cpu')
    backend_test.exclude(r'test_qlinearmatmul_2D_int8_float16_cpu')
    backend_test.exclude(r'test_qlinearmatmul_2D_int8_float32_cpu')
    backend_test.exclude(r'test_qlinearmatmul_2D_uint8_float16_cpu')
    backend_test.exclude(r'test_qlinearmatmul_2D_uint8_float32_cpu')
    backend_test.exclude(r'test_qlinearmatmul_3D_int8_float16_cpu')
    backend_test.exclude(r'test_qlinearmatmul_3D_int8_float32_cpu')
    backend_test.exclude(r'test_qlinearmatmul_3D_uint8_float16_cpu')
    backend_test.exclude(r'test_qlinearmatmul_3D_uint8_float32_cpu')
    backend_test.exclude(r'test_reduce_l1_empty_set_cpu')
    backend_test.exclude(r'test_reduce_l1_empty_set_expanded_cpu')
    backend_test.exclude(r'test_reduce_l2_empty_set_cpu')
    backend_test.exclude(r'test_reduce_l2_empty_set_expanded_cpu')
    backend_test.exclude(r'test_reduce_log_sum_empty_set_cpu')
    backend_test.exclude(r'test_reduce_log_sum_empty_set_expanded_cpu')
    backend_test.exclude(r'test_reduce_log_sum_exp_empty_set_cpu')
    backend_test.exclude(r'test_reduce_log_sum_exp_empty_set_expanded_cpu')
    backend_test.exclude(r'test_reduce_max_bool_inputs_cpu')
    backend_test.exclude(r'test_reduce_min_bool_inputs_cpu')
    # TODO: empty set ReduceOps tests are generating dynamic shapes
    backend_test.exclude(r'test_reduce_min_empty_set_cpu')
    backend_test.exclude(r'test_reduce_prod_empty_set_cpu')
    backend_test.exclude(r'test_reduce_sum_empty_set_cpu')
    backend_test.exclude(
        r'test_reduce_sum_empty_set_non_reduced_axis_zero_cpu')
    backend_test.exclude(r'test_reduce_sum_square_empty_set_cpu')
    backend_test.exclude(r'test_reduce_sum_square_empty_set_expanded_cpu')
    # TODO: Pooling tests are failing with shape mismatches, look into it.
    backend_test.exclude(
        r'test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_True_cpu'
    )
    backend_test.exclude(
        r'test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True_cpu'
    )
    backend_test.exclude(r'test_maxpool_2d_ceil_output_size_reduce_by_one_cpu')
    backend_test.exclude(r'test_maxpool_3d_dilations_use_ref_impl_large_cpu')


def disabled_tests_onnx_1_17_0(backend_test):
    # TODO: empty set ReduceOps tests are generating dynamic shapes
    backend_test.exclude(r'test_reduce_max_empty_set_cpu')
    backend_test.exclude(r'test_reduce_sum_empty_axes_input_noop_cpu')
    # tf_crop_and_resize not supported
    backend_test.exclude(
        r'test_resize_tf_crop_and_resize_extrapolation_value_cpu')
    # keep_aspect_ratio_policy not supported
    backend_test.exclude(r'test_resize_upsample_sizes_nearest_not_smaller_cpu')


def disabled_tests_int4(backend_test):
    # quantizelinear
    backend_test.exclude(r'test_quantizelinear_int4')
    backend_test.exclude(r'test_quantizelinear_uint4')
    # dequantizelinear
    backend_test.exclude(r'test_dequantizelinear_int4')
    backend_test.exclude(r'test_dequantizelinear_uint4')
    # cast
    backend_test.exclude(r'test_cast_FLOAT16_to_UINT4')
    backend_test.exclude(r'test_cast_FLOAT16_to_INT4')
    backend_test.exclude(r'test_cast_FLOAT_to_INT4')
    backend_test.exclude(r'test_cast_FLOAT_to_UINT4')
    backend_test.exclude(r'test_cast_INT4_to_FLOAT')
    backend_test.exclude(r'test_cast_INT4_to_FLOAT16')
    backend_test.exclude(r'test_cast_INT4_to_INT8')
    backend_test.exclude(r'test_cast_UINT4_to_FLOAT')
    backend_test.exclude(r'test_cast_UINT4_to_FLOAT16')
    backend_test.exclude(r'test_cast_UINT4_to_INT8')
    backend_test.exclude(r'test_cast_UINT4_to_UINT8_cpu')


def disabled_tests_float8(backend_test):
    # e4m3fn (Prototensor data type 17 not supported)
    backend_test.exclude(r'test_dequantizelinear_e4m3fn_cpu')
    backend_test.exclude(r'test_dequantizelinear_e4m3fn_float16_cpu')
    backend_test.exclude(r'test_dequantizelinear_e4m3fn_zero_point_cpu')
    backend_test.exclude(r'test_quantizelinear_e4m3fn_cpu')
    backend_test.exclude(r'test_cast_FLOAT16_to_FLOAT8E4M3FN_cpu')
    backend_test.exclude(r'test_cast_FLOAT8E4M3FN_to_FLOAT16_cpu')
    backend_test.exclude(r'test_cast_FLOAT8E4M3FN_to_FLOAT_cpu')
    backend_test.exclude(r'test_cast_FLOAT_to_FLOAT8E4M3FN_cpu')
    backend_test.exclude(r'test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN_cpu')
    backend_test.exclude(r'test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_FLOAT8E4M3FN_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded_cpu')
    # e4m3fnuz (Prototensor data type 18 not supported)
    backend_test.exclude(r'test_cast_FLOAT16_to_FLOAT8E4M3FNUZ_cpu')
    backend_test.exclude(r'test_cast_FLOAT8E4M3FNUZ_to_FLOAT16_cpu')
    backend_test.exclude(r'test_cast_FLOAT8E4M3FNUZ_to_FLOAT_cpu')
    backend_test.exclude(r'test_cast_FLOAT_to_FLOAT8E4M3FNUZ_cpu')
    backend_test.exclude(
        r'test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_cpu')
    backend_test.exclude(r'test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_cpu')
    backend_test.exclude(r'test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_cpu')
    backend_test.exclude(r'test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded_cpu')
    backend_test.exclude(r'test_castlike_FLOAT8E4M3FN_to_FLOAT_cpu')
    backend_test.exclude(r'test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded_cpu')
    # e5m2 ( Prototensor data type 19 not supported )
    backend_test.exclude(r'test_dequantizelinear_e5m2_cpu')
    backend_test.exclude(r'test_quantizelinear_e5m2_cpu')
    backend_test.exclude(r'test_cast_FLOAT16_to_FLOAT8E5M2_cpu')
    backend_test.exclude(r'test_cast_FLOAT8E5M2_to_FLOAT16_cpu')
    backend_test.exclude(r'test_cast_FLOAT8E5M2_to_FLOAT_cpu')
    backend_test.exclude(r'test_cast_FLOAT_to_FLOAT8E5M2_cpu')
    backend_test.exclude(r'test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2_cpu')
    backend_test.exclude(r'test_cast_no_saturate_FLOAT_to_FLOAT8E5M2_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_FLOAT8E5M2_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_FLOAT8E5M2_expanded_cpu')
    # e5m2fnuz (Prototensor data type 20 not supported)
    backend_test.exclude(r'test_cast_FLOAT16_to_FLOAT8E5M2FNUZ_cpu')
    backend_test.exclude(r'test_cast_FLOAT8E5M2FNUZ_to_FLOAT16_cpu')
    backend_test.exclude(r'test_cast_FLOAT8E5M2FNUZ_to_FLOAT_cpu')
    backend_test.exclude(r'test_cast_FLOAT_to_FLOAT8E5M2FNUZ_cpu')
    backend_test.exclude(
        r'test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_cpu')
    backend_test.exclude(r'test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_cpu')
    backend_test.exclude(r'test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_cpu')
    backend_test.exclude(r'test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded_cpu')
    backend_test.exclude(r'test_castlike_FLOAT8E5M2_to_FLOAT_cpu')
    backend_test.exclude(r'test_castlike_FLOAT8E5M2_to_FLOAT_expanded_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_cpu')
    backend_test.exclude(r'test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded_cpu')


def disabled_tests_dynamic_shape(backend_test):
    # constantofshape
    backend_test.exclude(r'test_constantofshape_float_ones_cpu')
    backend_test.exclude(r'test_constantofshape_int_shape_zero_cpu')
    backend_test.exclude(r'test_constantofshape_int_zeros_cpu')
    # cumsum
    backend_test.exclude(r'test_cumsum_1d_cpu')
    backend_test.exclude(r'test_cumsum_1d_exclusive_cpu')
    backend_test.exclude(r'test_cumsum_1d_reverse_cpu')
    backend_test.exclude(r'test_cumsum_1d_reverse_exclusive_cpu')
    backend_test.exclude(r'test_cumsum_2d_axis_0_cpu')
    backend_test.exclude(r'test_cumsum_2d_axis_1_cpu')
    backend_test.exclude(r'test_cumsum_2d_negative_axis_cpu')
    # expand
    backend_test.exclude(r'test_expand_dim_changed_cpu')
    backend_test.exclude(r'test_expand_dim_unchanged_cpu')
    backend_test.exclude(r'test_expand_shape_model1_cpu')
    backend_test.exclude(r'test_expand_shape_model2_cpu')
    backend_test.exclude(r'test_expand_shape_model3_cpu')
    backend_test.exclude(r'test_expand_shape_model4_cpu')
    # onehot
    backend_test.exclude(r'test_onehot_negative_indices_cpu')
    backend_test.exclude(r'test_onehot_with_axis_cpu')
    backend_test.exclude(r'test_onehot_with_negative_axis_cpu')
    backend_test.exclude(r'test_onehot_without_axis_cpu')
    # range
    backend_test.exclude(r'test_range_float_type_positive_delta_cpu')
    backend_test.exclude(r'test_range_int32_type_negative_delta_cpu')
    # slice
    backend_test.exclude(r'test_slice_default_axes_cpu')
    backend_test.exclude(r'test_slice_default_steps_cpu')
    backend_test.exclude(r'test_slice_negative_axes_cpu')
    # split
    backend_test.exclude(r'test_split_variable_parts_1d_opset13_cpu')
    backend_test.exclude(r'test_split_variable_parts_1d_opset18_cpu')
    backend_test.exclude(r'test_split_variable_parts_2d_opset13_cpu')
    backend_test.exclude(r'test_split_variable_parts_2d_opset18_cpu')
    backend_test.exclude(r'test_split_variable_parts_default_axis_opset13_cpu')
    backend_test.exclude(r'test_split_variable_parts_default_axis_opset18_cpu')
    backend_test.exclude(r'test_split_zero_size_splits_opset13_cpu')
    backend_test.exclude(r'test_split_zero_size_splits_opset18_cpu')
    # squeeze
    backend_test.exclude(r'test_squeeze_cpu')
    backend_test.exclude(r'test_squeeze_negative_axes_cpu')
    # unsqueeze
    backend_test.exclude(r'test_unsqueeze_axis_0_cpu')
    backend_test.exclude(r'test_unsqueeze_axis_1_cpu')
    backend_test.exclude(r'test_unsqueeze_axis_2_cpu')
    backend_test.exclude(r'test_unsqueeze_negative_axes_cpu')
    backend_test.exclude(r'test_unsqueeze_three_axes_cpu')
    backend_test.exclude(r'test_unsqueeze_two_axes_cpu')
    backend_test.exclude(r'test_unsqueeze_unsorted_axes_cpu')
    # unique
    backend_test.exclude(r'test_unique_not_sorted_without_axis_cpu')
    backend_test.exclude(r'test_unique_sorted_with_axis_cpu')
    backend_test.exclude(r'test_unique_sorted_without_axis_cpu')
    # tile
    backend_test.exclude(r'test_tile_cpu')
    backend_test.exclude(r'test_tile_precomputed_cpu')
    # resize
    backend_test.exclude(r'test_resize_upsample_scales_nearest_cpu')
    backend_test.exclude(r'test_resize_downsample_scales_nearest_cpu')
    backend_test.exclude(r'test_resize_upsample_sizes_nearest_cpu')
    backend_test.exclude(r'test_resize_downsample_sizes_nearest_cpu')
    backend_test.exclude(
        r'test_resize_upsample_sizes_nearest_floor_align_corners_cpu')
    backend_test.exclude(
        r'test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric_cpu')
    backend_test.exclude(
        r'test_resize_upsample_sizes_nearest_ceil_half_pixel_cpu')
    # reshape
    backend_test.exclude(r'test_reshape_allowzero_reordered_cpu')
    backend_test.exclude(r'test_reshape_extended_dims_cpu')
    backend_test.exclude(r'test_reshape_negative_dim_cpu')
    backend_test.exclude(r'test_reshape_negative_extended_dims_cpu')
    backend_test.exclude(r'test_reshape_one_dim_cpu')
    backend_test.exclude(r'test_reshape_reduced_dims_cpu')
    backend_test.exclude(r'test_reshape_reordered_all_dims_cpu')
    backend_test.exclude(r'test_reshape_reordered_last_dims_cpu')
    backend_test.exclude(r'test_reshape_zero_and_negative_dim_cpu')
    backend_test.exclude(r'test_reshape_zero_dim_cpu')
    # reduce
    backend_test.exclude(
        r'test_reduce_l1_default_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l1_default_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l1_do_not_keepdims_example_expanded_cpu')
    backend_test.exclude(r'test_reduce_l1_do_not_keepdims_random_expanded_cpu')
    backend_test.exclude(r'test_reduce_l1_keep_dims_example_expanded_cpu')
    backend_test.exclude(r'test_reduce_l1_keep_dims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l1_negative_axes_keep_dims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l1_negative_axes_keep_dims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l2_default_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l2_default_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l2_do_not_keepdims_example_expanded_cpu')
    backend_test.exclude(r'test_reduce_l2_do_not_keepdims_random_expanded_cpu')
    backend_test.exclude(r'test_reduce_l2_keep_dims_example_expanded_cpu')
    backend_test.exclude(r'test_reduce_l2_keep_dims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l2_negative_axes_keep_dims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_l2_negative_axes_keep_dims_random_expanded_cpu')
    backend_test.exclude(r'test_reduce_log_sum_asc_axes_cpu')
    backend_test.exclude(r'test_reduce_log_sum_asc_axes_expanded_cpu')
    backend_test.exclude(r'test_reduce_log_sum_default_cpu')
    backend_test.exclude(r'test_reduce_log_sum_default_expanded_cpu')
    backend_test.exclude(r'test_reduce_log_sum_desc_axes_cpu')
    backend_test.exclude(r'test_reduce_log_sum_desc_axes_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_default_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_default_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_do_not_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_do_not_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(r'test_reduce_log_sum_negative_axes_cpu')
    backend_test.exclude(r'test_reduce_log_sum_negative_axes_expanded_cpu')
    backend_test.exclude(r'test_reduce_max_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_max_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_max_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_max_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_max_negative_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_max_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_mean_default_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_mean_default_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_mean_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_mean_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_mean_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_mean_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_mean_negative_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_mean_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_min_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_min_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_min_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_min_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_min_negative_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_min_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_prod_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_prod_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_prod_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_prod_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_prod_negative_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_prod_negative_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_default_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_default_axes_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_do_not_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_do_not_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_empty_axes_input_noop_example_cpu')
    backend_test.exclude(r'test_reduce_sum_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_keepdims_random_cpu')
    backend_test.exclude(r'test_reduce_sum_negative_axes_keepdims_example_cpu')
    backend_test.exclude(r'test_reduce_sum_negative_axes_keepdims_random_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_default_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_default_axes_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_do_not_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_do_not_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_keepdims_random_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_negative_axes_keepdims_example_expanded_cpu')
    backend_test.exclude(
        r'test_reduce_sum_square_negative_axes_keepdims_random_expanded_cpu')


def create_backend_test(testname=None, target_device=None):
    if target_device is not None:
        c2.set_device(target_device)
    backend_test = MIGraphXBackendTest(c2, __name__)

    if testname:
        backend_test.include(testname + '.*')
    else:
        # Onnx Operator tests
        backend_test.include(r'.*test_abs.*')
        backend_test.include(r'.*test_acos.*')
        backend_test.include(r'.*test_acosh.*')
        backend_test.include(r'.*test_add.*')
        backend_test.include(r'.*test_and.*')
        backend_test.include(r'.*test_argmax.*')
        backend_test.include(r'.*test_argmin.*')
        backend_test.include(r'.*test_asin.*')
        backend_test.include(r'.*test_asinh.*')
        backend_test.include(r'.*test_atan.*')
        backend_test.include(r'.*test_atanh.*')
        backend_test.include(r'.*test_averagepool.*')
        backend_test.include(r'.*test_AvgPool.*')
        backend_test.include(r'.*test_[bB]atch[nN]orm(?!.*training).*')
        backend_test.include(r'.*test_bitshift.*')
        backend_test.include(r'.*test_bitwise.*')
        backend_test.include(r'.*test_ceil.*')
        backend_test.include(r'.*test_cast_.*')
        backend_test.include(r'.*test_col2im.*')
        backend_test.include(r'.*test_compress.*')
        backend_test.include(r'.*test_concat.*')
        backend_test.include(r'.*test_constant_.*')
        backend_test.include(r'.*test_Constant.*')
        backend_test.include(r'.*test_constantofshape.*')
        backend_test.include(r'.*test_(basic_)?conv_.*')
        backend_test.include(r'.*test_Conv[1-3]d.*')
        backend_test.include(r'.*test_convinteger.*')
        backend_test.include(r'.*test_convtranspose.*')
        backend_test.include(r'.*test_ConvTranspose[1-3]d.*')
        backend_test.include(r'.*test_cos.*')
        backend_test.include(r'.*test_cosh.*')
        backend_test.include(r'.*test_cumsum.*')
        backend_test.include(r'.*test_(basic_)?deform_conv.*')
        backend_test.include(r'.*test_depthtospace.*')
        backend_test.include(r'.*test_dequantizelinear.*')
        backend_test.include(r'.*test_det.*')
        backend_test.include(r'.*test_dft.*')
        backend_test.include(r'.*test_div.*')
        backend_test.include(r'.*test_dropout.*')
        backend_test.include(r'.*test_einsum.*')
        backend_test.include(r'.*test_equal.*')
        backend_test.include(r'.*test_Embedding.*')
        backend_test.include(r'.*test_erf.*')
        backend_test.include(r'.*test_exp_.*')
        backend_test.include(r'.*test_expand.*')
        backend_test.include(r'.*test_eyelike.*')
        backend_test.include(r'.*test_flatten.*')
        backend_test.include(r'.*test_floor.*')
        backend_test.include(r'.*test_gru.*')
        backend_test.include(r'.*test_gather.*')
        backend_test.include(r'.*test_gemm.*')
        backend_test.include(r'.*test_globalaveragepool.*')
        backend_test.include(r'.*test_globallppool.*')
        backend_test.include(r'.*test_globalmaxpool.*')
        backend_test.include(r'.*test_greater.*')
        backend_test.include(r'.*test_gridsample.*')
        backend_test.include(r'.*test_hardmax.*')
        backend_test.include(r'.*test_identity.*')
        backend_test.include(r'.*test_if.*')
        backend_test.include(r'.*test_instancenorm.*')
        backend_test.include(r'.*test_isinf.*')
        backend_test.include(r'.*test_isnan.*')
        backend_test.include(r'.*test_lrn.*')
        backend_test.include(r'.*test_lstm.*')
        backend_test.include(r'.*test_log.*')
        backend_test.include(r'.*test_log2.*')
        backend_test.include(r'.*test_loop.*')
        backend_test.include(r'.*test_lpnorm.*')
        backend_test.include(r'.*test_lppool.*')
        backend_test.include(r'.*test_matmul.*')
        backend_test.include(r'.*test_max_.*')
        backend_test.include(r'.*test_maxpool.*')
        backend_test.include(r'.*test_MaxPool[1-3]d.*')
        backend_test.include(r'.*test_maxroipool.*')
        backend_test.include(r'.*test_maxunpool.*')
        backend_test.include(r'.*test_mean.*')
        backend_test.include(r'.*test_melweightmatrix.*')
        backend_test.include(r'.*test_min.*')
        backend_test.include(r'.*test_mod.*')
        backend_test.include(r'.*test_mul.*')
        backend_test.include(r'.*test_[mM]ultinomial.*')
        backend_test.include(r'.*test_neg.*')
        backend_test.include(r'.*test_nonmaxsuppression.*')
        backend_test.include(r'.*test_nonzero.*')
        backend_test.include(r'.*test_not.*')
        backend_test.include(r'.*test_onehot.*')
        backend_test.include(r'.*optional_get_element.*')
        backend_test.include(r'.*optional_has_element.*')
        backend_test.include(r'.*test_or.*')
        backend_test.include(r'.*test_(constant_|edge_|reflect_|wrap_)?pad.*')
        backend_test.include(
            r'.*test_(Constant|Reflection|Replication|Zero)+Pad2d.*')
        backend_test.include(r'.*test_pow.*')
        backend_test.include(r'.*test_qlinearconv.*')
        backend_test.include(r'.*test_qlinearmatmul.*')
        backend_test.include(r'.*test_quantizelinear.*')
        backend_test.include(r'.*test_(simple_)?rnn.*')
        backend_test.include(r'.*test_randomnormal.*')
        backend_test.include(r'.*test_randomuniform.*')
        backend_test.include(r'.*test_reciprocal.*')
        backend_test.include(r'.*test_reduce_max.*')
        backend_test.include(r'.*test_reduce_mean.*')
        backend_test.include(r'.*test_reduce_min.*')
        backend_test.include(r'.*test_reduce_prod.*')
        backend_test.include(r'.*test_reduce_sum.*')
        backend_test.include(r'.*test_reshape.*')
        backend_test.include(r'.*test_resize.*')
        backend_test.include(r'.*test_reversesequence.*')
        backend_test.include(r'.*test_roialign.*')
        backend_test.include(r'.*test_round.*')
        backend_test.include(r'.*test_stft.*')
        backend_test.include(r'.*test_scan.*')
        backend_test.include(r'.*test_scatter.*')
        backend_test.include(r'.*test_sequence_at.*')
        backend_test.include(r'.*test_sequence_construct.*')
        backend_test.include(r'.*test_sequence_empty.*')
        backend_test.include(r'.*test_sequence_erase.*')
        backend_test.include(r'.*test_sequence_insert.*')
        backend_test.include(r'.*test_sequence_length.*')
        backend_test.include(r'.*test_shape.*')
        backend_test.include(r'.*test_[sS]igmoid.*')
        backend_test.include(r'.*test_sign.*')
        backend_test.include(r'.*test_sin_.*')
        backend_test.include(r'.*test_sinh.*')
        backend_test.include(r'.*test_size.*')
        backend_test.include(r'.*test_slice.*')
        backend_test.include(r'.*test_spacetodepth.*')
        backend_test.include(r'.*test_split.*')
        backend_test.include(r'.*test_split_to_sequence.*')
        backend_test.include(r'.*test_sqrt.*')
        backend_test.include(r'.*test_squeeze.*')
        backend_test.include(r'.*test_squeeze.*')
        backend_test.include(r'.*test_strnorm.*')
        backend_test.include(r'.*test_sub.*')
        backend_test.include(r'.*test_sum.*')
        backend_test.include(r'.*test_tan_.*')
        backend_test.include(r'.*test_[tT]anh.*')
        backend_test.include(r'.*test_tfidfvectorizer.*')
        backend_test.include(r'.*test_tile.*')
        backend_test.include(r'.*test_top_k.*')
        backend_test.include(r'.*test_transpose.*')
        backend_test.include(r'.*test_tril.*')
        backend_test.include(r'.*test_triu.*')
        backend_test.include(r'.*test_unique.*')
        backend_test.include(r'.*test_unsqueeze.*')
        backend_test.include(r'.*test_upsample.*')
        backend_test.include(r'.*test_where.*')
        backend_test.include(r'.*test_xor.*')

        # Onnx Function tests
        backend_test.include(r'.*test_bernoulli.*')
        backend_test.include(r'.*test_blackmanwindow.*')
        backend_test.include(r'.*test_castlike.*')
        backend_test.include(r'.*test_celu.*')
        backend_test.include(r'.*test_center_crop_pad.*')
        backend_test.include(r'.*test_clip.*')
        backend_test.include(r'.*test_dynamicquantizelinear.*')
        backend_test.include(r'.*test_elu.*')
        backend_test.include(r'.*test_ELU.*')
        backend_test.include(r'.*test_GLU.*')
        backend_test.include(r'.*test_greater_equal.*')
        backend_test.include(r'.*test_group_normalization.*')
        backend_test.include(r'.*test_hammingwindow.*')
        backend_test.include(r'.*test_hannwindow.*')
        backend_test.include(r'.*test_hardsigmoid.*')
        backend_test.include(r'.*test_hardswish.*')
        backend_test.include(r'.*test_layer_normalization.*')
        backend_test.include(r'.*test_LeakyReLU.*')
        backend_test.include(r'.*test_leakyrelu.*')
        backend_test.include(r'.*test_less.*')
        backend_test.include(r'.*test_Linear.*')
        backend_test.include(r'.*test_logsoftmax.*')
        backend_test.include(r'.*test_log_softmax.*')
        backend_test.include(r'.*test_LogSoftmax.*')
        backend_test.include(r'.*test_mvn.*')
        backend_test.include(r'.*test_mish.*')
        backend_test.include(r'.*test_nllloss.*')
        backend_test.include(r'.*test_PixelShuffle.*')
        backend_test.include(r'.*test_PoissonNLLLLoss_no_reduce.*')
        backend_test.include(r'.*test_prelu.*')
        backend_test.include(r'.*test_PReLU.*')
        backend_test.include(r'.*test_range.*')
        backend_test.include(r'.*test_reduce_l1.*')
        backend_test.include(r'.*test_reduce_l2.*')
        backend_test.include(r'.*test_reduce_log.*')
        backend_test.include(r'.*test_ReLU.*')
        backend_test.include(r'.*test_relu.*')
        backend_test.include(r'.*test_selu.*')
        backend_test.include(r'.*test_SELU.*')
        backend_test.include(r'.*test_sequence_map.*')
        backend_test.include(r'.*test_shrink.*')
        backend_test.include(r'.*test_[sS]oftmax.*')
        backend_test.include(r'.*test_[sS]oftmin.*')
        backend_test.include(r'.*test_[sS]oftplus.*')
        backend_test.include(r'.*test_[sS]oftsign.*')
        backend_test.include(r'.*test_sce.*')
        backend_test.include(r'.*test_thresholdedrelu.*')

        # OnnxBackendPyTorchOperatorModelTest
        backend_test.include(r'.*test_operator_add_broadcast.*')
        backend_test.include(r'.*test_operator_addconstant.*')
        backend_test.include(r'.*test_operator_addmm.*')
        backend_test.include(r'.*test_operator_add_size1.*')
        backend_test.include(r'.*test_operator_basic.*')
        backend_test.include(r'.*test_operator_chunk.*')
        backend_test.include(r'.*test_operator_clip.*')
        backend_test.include(r'.*test_operator_concat2.*')
        backend_test.include(r'.*test_operator_conv_.*')
        backend_test.include(r'.*test_operator_convtranspose.*')
        backend_test.include(r'.*test_operator_exp.*')
        backend_test.include(r'.*test_operator_flatten.*')
        backend_test.include(r'.*test_operator_index.*')
        backend_test.include(r'.*test_operator_max_.*')
        backend_test.include(r'.*test_operator_maxpool.*')
        backend_test.include(r'.*test_operator_min.*')
        backend_test.include(r'.*test_operator_mm.*')
        backend_test.include(r'.*test_operator_negativeloglikelihoodloss.*')
        backend_test.include(r'.*test_operator_non_float_params.*')
        backend_test.include(r'.*test_operator_pad.*')
        backend_test.include(r'.*test_operator_params.*')
        backend_test.include(r'.*test_operator_permute2.*')
        backend_test.include(r'.*test_operator_pow.*')
        backend_test.include(r'.*test_operator_reduced_mean_.*')
        backend_test.include(r'.*test_operator_reduced_mean_keepdim.*')
        backend_test.include(r'.*test_operator_reduced_sum_.*')
        backend_test.include(r'.*test_operator_reduced_sum_keepdim.*')
        backend_test.include(r'.*test_operator_repeat.*')
        backend_test.include(r'.*test_operator_selu.*')
        backend_test.include(r'.*test_operator_softmaxcrossentropyloss.*')
        backend_test.include(r'.*test_operator_sqrt.*')
        backend_test.include(r'.*test_operator_symbolic_override.*')
        backend_test.include(r'.*test_operator_symbolic_override_nested.*')
        backend_test.include(r'.*test_operator_view.*')

        # OnnxBackendSimpleModelTest
        backend_test.include(r'.*test_gradient_of.*')
        backend_test.include(r'.*test_sequence_model.*')
        backend_test.include(r'.*test_single_relu_model.*')

        # OnnxBackendRealModelTest
        backend_test.include(r'.*test_bvlc_alexnet.*')
        backend_test.include(r'.*test_densenet121.*')
        backend_test.include(r'.*test_inception_v1.*')
        backend_test.include(r'.*test_inception_v2.*')
        backend_test.include(r'.*test_resnet50.*')
        backend_test.include(r'.*test_shufflenet.*')
        backend_test.include(r'.*test_squeezenet.*')
        backend_test.include(r'.*test_vgg19.*')
        backend_test.include(r'.*test_zfnet512.*')

        # Skipped tests
        # backend_test.include(r'.*test_adagrad.*')
        # backend_test.include(r'.*test_adam.*')
        # backend_test.include(r'.*test_ai_onnx_ml.*')
        # backend_test.include(r'.*test_batchnorm_epsilon_training.*')
        # backend_test.include(r'.*test_batchnorm_example_training.*')
        # backend_test.include(r'.*test_momentum.*')
        # backend_test.include(r'.*test_nesterov_momentum.*')
        # backend_test.include(r'.*test_training_dropout.*')

        # Exclude failing tests

        # from OnnxBackendRealModelTest
        backend_test.exclude(r'test_inception_v1_cpu')

        # PRelu OnnxBackendPyTorchConvertedModelTest has wrong dim for broadcasting
        backend_test.exclude(r'[a-z,_]*PReLU_[0-9]d_multiparam[a-z,_]*')

        # Remove when float8 is supported
        disabled_tests_float8(backend_test)

        # Remove Int4 tests
        disabled_tests_int4(backend_test)

        # Remove when dynamic shapes are supported
        disabled_tests_dynamic_shape(backend_test)

        # additional cases disabled for a specific onnx version
        if version.parse(onnx.__version__) >= version.parse("1.7.0"):
            disabled_tests_onnx_1_7_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.8.0"):
            disabled_tests_onnx_1_8_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.9.0"):
            disabled_tests_onnx_1_9_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.10.0"):
            disabled_tests_onnx_1_10_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.11.0"):
            disabled_tests_onnx_1_11_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.12.0"):
            disabled_tests_onnx_1_12_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.13.0"):
            disabled_tests_onnx_1_13_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.14.0"):
            disabled_tests_onnx_1_14_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.16.0"):
            disabled_tests_onnx_1_16_0(backend_test)

        if version.parse(onnx.__version__) >= version.parse("1.17.0"):
            disabled_tests_onnx_1_17_0(backend_test)


# import all test cases at global scope to make
# them visible to python.unittest.
    globals().update(backend_test.enable_report().test_cases)

    return backend_test


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='Run the ONNX backend tests using MIGraphX.')

    # Add an argument to match a single test name, by adding the name to the 'include' filter.
    # Using -k with python unittest (https://docs.python.org/3/library/unittest.html#command-line-options)
    # doesn't work as it filters on the test method name (Runner._add_model_test) rather than inidividual
    # test case names.
    parser.add_argument(
        '-t',
        '--test-name',
        dest='testname',
        type=str,
        help=
        "Only run tests that match this value. Matching is regex based, and '.*' is automatically appended"
    )
    parser.add_argument('-d',
                        '--device',
                        dest='device',
                        type=str,
                        help="Specify the device to run test on")

    # parse just our args. python unittest has its own args and arg parsing, and that runs inside unittest.main()
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    if args.device is not None:
        print("run on {} device....".format(args.device))
    else:
        print("Default GPU device is used ....")

    return args


if __name__ == '__main__':
    if sys.version_info < (3, 0):
        sys.exit()

    args = parse_args()
    backend_test = create_backend_test(args.testname, args.device)
    unittest.main()
