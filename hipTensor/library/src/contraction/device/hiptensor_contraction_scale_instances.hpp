/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef CONTRACTION_SCALE_HPP
#define CONTRACTION_SCALE_HPP

#include "common.hpp"

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {
            namespace instance
            {
                using F32         = float;
                using CF32        = hipFloatComplex;
                using Empty_Tuple = ck::Tuple<>;

                using F64  = double;
                using CF64 = hipDoubleComplex;

                using ScaleComplex = element_wise::ScaleComplex;

                void
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_kkn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                                               6,
                                                                               6,
                                                                               CF32,
                                                                               CF32,
                                                                               Empty_Tuple,
                                                                               CF32,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               ScaleComplex,
                                                                               CF32>>>& instances);

                void
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_knn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                                               6,
                                                                               6,
                                                                               CF32,
                                                                               CF32,
                                                                               Empty_Tuple,
                                                                               CF32,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               ScaleComplex,
                                                                               CF32>>>& instances);

                void
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_mkn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                                               6,
                                                                               6,
                                                                               CF32,
                                                                               CF32,
                                                                               Empty_Tuple,
                                                                               CF32,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               ScaleComplex,
                                                                               CF32>>>& instances);

                void
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_mnn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                                               6,
                                                                               6,
                                                                               CF32,
                                                                               CF32,
                                                                               Empty_Tuple,
                                                                               CF32,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               ScaleComplex,
                                                                               CF32>>>& instances);

                void
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_kkn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                                               6,
                                                                               6,
                                                                               CF64,
                                                                               CF64,
                                                                               Empty_Tuple,
                                                                               CF64,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               ScaleComplex,
                                                                               CF64>>>& instances);

                void
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_knn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                                               6,
                                                                               6,
                                                                               CF64,
                                                                               CF64,
                                                                               Empty_Tuple,
                                                                               CF64,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               ScaleComplex,
                                                                               CF64>>>& instances);

                void
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_mkn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                                               6,
                                                                               6,
                                                                               CF64,
                                                                               CF64,
                                                                               Empty_Tuple,
                                                                               CF64,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               ScaleComplex,
                                                                               CF64>>>& instances);

                void
                    add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_mnn_instance(
                        std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
                                                                               6,
                                                                               6,
                                                                               CF64,
                                                                               CF64,
                                                                               Empty_Tuple,
                                                                               CF64,
                                                                               PassThrough,
                                                                               PassThrough,
                                                                               ScaleComplex,
                                                                               CF64>>>& instances);

                // Contraction + Scale
                template <index_t NumDimM,
                          index_t NumDimN,
                          index_t NumDimK,
                          typename ADataType,
                          typename BDataType,
                          typename EDataType,
                          typename ComputeDataType>
                struct DeviceOperationInstanceFactory<
                    ck::tensor_operation::device::DeviceContractionMultipleD<
                        NumDimM,
                        NumDimN,
                        NumDimK,
                        HIP_vector_type<ADataType, 2>,
                        HIP_vector_type<BDataType, 2>,
                        ck::Tuple<>,
                        HIP_vector_type<EDataType, 2>,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::ScaleComplex,
                        HIP_vector_type<ComputeDataType, 2>>>
                {
                    using DeviceOp = DeviceContractionMultipleD<
                        NumDimM,
                        NumDimN,
                        NumDimK,
                        HIP_vector_type<ADataType, 2>,
                        HIP_vector_type<BDataType, 2>,
                        ck::Tuple<>,
                        HIP_vector_type<EDataType, 2>,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::PassThrough,
                        ck::tensor_operation::element_wise::ScaleComplex,
                        HIP_vector_type<ComputeDataType, 2>>;

                    static auto GetInstances()
                    {
                        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

                        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float>
                                     && is_same_v<EDataType, float>)
                        {
                            if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6)
                            {
                                add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_kkn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_knn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_mkn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf32_cf32_cf32_compute_cf32_mnn_instance(
                                    op_ptrs);
                            }
                        }

                        if constexpr(is_same_v<ADataType, double> && is_same_v<BDataType, double>
                                     && is_same_v<EDataType, double>)
                        {
                            if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6)
                            {
                                add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_kkn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_knn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_mkn_instance(
                                    op_ptrs);
                                add_device_contraction_scale_m6_n6_k6_xdl_c_shuffle_cf64_cf64_cf64_compute_cf64_mnn_instance(
                                    op_ptrs);
                            }
                        }

                        return op_ptrs;
                    }
                };

            } // namespace instance
        } // namespace device
    } // namespace tensor_operation
} // namespace ck

#endif // CONTRACTION_SCALE_HPP
