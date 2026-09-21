/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <ck/ck.hpp>
#include <ck/library/tensor_operation_instance/add_device_operation_instance.hpp>
#include <ck/library/tensor_operation_instance/gpu/contraction/device_contraction_instance.hpp>
#include <ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

#include "hiptensor_ck_types.hpp"

namespace ck
{
    namespace tensor_operation
    {
        namespace device
        {
            namespace instance
            {

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_kknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_knnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_mknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_mnnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_kknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       BF16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_knnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       BF16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_mknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       BF16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_mnnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F32,
                                                       F32,
                                                       F32_Tuple,
                                                       F32,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       BF16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_kknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F64,
                                                       F64,
                                                       F64_Tuple,
                                                       F64,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F64>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_knnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F64,
                                                       F64,
                                                       F64_Tuple,
                                                       F64,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F64>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_mknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F64,
                                                       F64,
                                                       F64_Tuple,
                                                       F64,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F64>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_mnnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F64,
                                                       F64,
                                                       F64_Tuple,
                                                       F64,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F64>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_kknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F64,
                                                       F64,
                                                       F64_Tuple,
                                                       F64,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_knnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F64,
                                                       F64,
                                                       F64_Tuple,
                                                       F64,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_mknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F64,
                                                       F64,
                                                       F64_Tuple,
                                                       F64,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_mnnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F64,
                                                       F64,
                                                       F64_Tuple,
                                                       F64,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_kknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F16,
                                                       F16,
                                                       F16_Tuple,
                                                       F16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_knnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F16,
                                                       F16,
                                                       F16_Tuple,
                                                       F16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_mknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F16,
                                                       F16,
                                                       F16_Tuple,
                                                       F16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_mnnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F16,
                                                       F16,
                                                       F16_Tuple,
                                                       F16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_kknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       BF16,
                                                       BF16,
                                                       BF16_Tuple,
                                                       BF16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_knnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       BF16,
                                                       BF16,
                                                       BF16_Tuple,
                                                       BF16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_mknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       BF16,
                                                       BF16,
                                                       BF16_Tuple,
                                                       BF16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_mnnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       BF16,
                                                       BF16,
                                                       BF16_Tuple,
                                                       BF16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F32>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_kknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F16,
                                                       F16,
                                                       F16_Tuple,
                                                       F16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_knnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F16,
                                                       F16,
                                                       F16_Tuple,
                                                       F16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_mknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F16,
                                                       F16,
                                                       F16_Tuple,
                                                       F16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_mnnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       F16,
                                                       F16,
                                                       F16_Tuple,
                                                       F16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       F16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_kknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       BF16,
                                                       BF16,
                                                       BF16_Tuple,
                                                       BF16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       BF16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_knnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       BF16,
                                                       BF16,
                                                       BF16_Tuple,
                                                       BF16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       BF16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_mknn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       BF16,
                                                       BF16,
                                                       BF16_Tuple,
                                                       BF16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       BF16>>>& instances);

                void
                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_mnnn_instance(
                        std::vector<std::unique_ptr<
                            DeviceContractionMultipleD<6,
                                                       6,
                                                       6,
                                                       BF16,
                                                       BF16,
                                                       BF16_Tuple,
                                                       BF16,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkHiptensorUnaryOp,
                                                       hiptensor::CkBilinearUnary,
                                                       BF16>>>& instances);

                // Contraction + Bilinear
                template <index_t NumDimM,
                          index_t NumDimN,
                          index_t NumDimK,
                          typename AElementwiseOperation,
                          typename BElementwiseOperation,
                          typename CDEElementwiseOperation,
                          typename ADataType,
                          typename BDataType,
                          typename DDataType,
                          typename EDataType,
                          typename ComputeDataType>
                struct DeviceOperationInstanceFactory<
                    ck::tensor_operation::device::DeviceContractionMultipleD<
                        NumDimM,
                        NumDimN,
                        NumDimK,
                        ADataType,
                        BDataType,
                        ck::Tuple<DDataType>,
                        EDataType,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CDEElementwiseOperation,
                        ComputeDataType>>
                {
                    using DeviceOp = DeviceContractionMultipleD<NumDimM,
                                                                NumDimN,
                                                                NumDimK,
                                                                ADataType,
                                                                BDataType,
                                                                ck::Tuple<DDataType>,
                                                                EDataType,
                                                                AElementwiseOperation,
                                                                BElementwiseOperation,
                                                                CDEElementwiseOperation,
                                                                ComputeDataType>;
                    static auto GetInstances()
                    {
                        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
                        if constexpr(
                            is_same_v<
                                ADataType,
                                float> && is_same_v<BDataType, float> && is_same_v<EDataType, float>)
                        {
                            if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6)
                            {
                                if constexpr(is_same_v<ComputeDataType, float>)
                                {
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance(
                                        op_ptrs);
                                }
                                else if constexpr(is_same_v<ComputeDataType, ck::half_t>)
                                {
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_kknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_knnn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_mknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_mnnn_instance(
                                        op_ptrs);
                                }
                                else if constexpr(is_same_v<ComputeDataType, ck::bhalf_t>)
                                {
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_kknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_knnn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_mknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_mnnn_instance(
                                        op_ptrs);
                                }
                            }
                        }
                        if constexpr(
                            is_same_v<
                                ADataType,
                                double> && is_same_v<BDataType, double> && is_same_v<EDataType, double>)
                        {
                            if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6)
                            {
                                if constexpr(is_same_v<ComputeDataType, double>)
                                {
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_kknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_knnn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_mknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_mnnn_instance(
                                        op_ptrs);
                                }
                                else if constexpr(is_same_v<ComputeDataType, float>)
                                {
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_kknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_knnn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_mknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_mnnn_instance(
                                        op_ptrs);
                                }
                            }
                        }
                        if constexpr(
                            is_same_v<
                                ADataType,
                                ck::half_t> && is_same_v<BDataType, ck::half_t> && is_same_v<EDataType, ck::half_t>)
                        {
                            if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6)
                            {
                                if constexpr(is_same_v<ComputeDataType, ck::half_t>)
                                {
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_kknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_knnn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_mknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_mnnn_instance(
                                        op_ptrs);
                                }
                                else if constexpr(is_same_v<ComputeDataType, float>)
                                {
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_kknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_knnn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_mknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_mnnn_instance(
                                        op_ptrs);
                                }
                            }
                        }
                        if constexpr(
                            is_same_v<
                                ADataType,
                                ck::bhalf_t> && is_same_v<BDataType, ck::bhalf_t> && is_same_v<EDataType, ck::bhalf_t>)
                        {
                            if constexpr(NumDimM == 6 && NumDimN == 6 && NumDimK == 6)
                            {
                                if constexpr(is_same_v<ComputeDataType, ck::bhalf_t>)
                                {
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_kknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_knnn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_mknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_mnnn_instance(
                                        op_ptrs);
                                }
                                else if constexpr(is_same_v<ComputeDataType, float>)
                                {
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_kknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_knnn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_mknn_instance(
                                        op_ptrs);
                                    add_device_contraction_bilinear_unary_m6_n6_k6_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_mnnn_instance(
                                        op_ptrs);
                                }
                            }
                        }

                        return op_ptrs;
                    }
                };

            } // namespace instance
        } // namespace device
    } // namespace tensor_operation
} // namespace ck
