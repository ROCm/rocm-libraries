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

#include "contraction_solution_instances.hpp"
#include "contraction_solution.hpp"

// CK data types and utilities
#include <ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp>
#include <ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/tuple.hpp>
#include <hip/hip_complex.h>
//#include <ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp>

//#include "ck/ck.hpp"
//#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
//#include "ck/library/tensor_operation_instance/gpu/contraction/device_contraction_instance.hpp"
//#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
//#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

// Ensure access to
#include "device/hiptensor_contraction_bilinear_instances.hpp"
#include "device/hiptensor_contraction_scale_instances.hpp"
#include "device/hiptensor_contraction_bilinear_unary_ops_instances.hpp"

//using CkPassThrough       = ck::tensor_operation::element_wise::PassThrough;
//using CkHiptensorUnaryOp  = ck::tensor_operation::element_wise::HiptensorUnaryOp;
//using CkBilinearUnary  = ck::tensor_operation::element_wise::BilinearUnary;

//namespace ck {
//namespace tensor_operation {
//namespace device {
//namespace instance {
//
//using device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance_unary =
//    device_contraction_kk_instance<F32,
//                                   F32,
//                                   F32,
//                                   F32,
//                                   F32_Tuple,
//                                   F32,
//                                   F32,
//                                   CkHiptensorUnaryOp,
//                                   CkHiptensorUnaryOp,
//                                   CkBilinearUnary,
//                                   6>;
//
//void add_device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance_unary(
//    std::vector<std::unique_ptr<DeviceContractionMultipleD<6,
//                                                           6,
//                                                           6,
//                                                           F32,
//                                                           F32,
//                                                           F32_Tuple,
//                                                           F32,
//                                                           CkHiptensorUnaryOp,
//                                                           CkHiptensorUnaryOp,
//                                                           CkBilinearUnary,
//                                                           F32>>>& instances)
//{
//    add_device_operation_instances(
//        instances,
//        device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance_unary{});
//}
//
//// Contraction + Bilinear
//template <index_t NumDimM,
//          index_t NumDimN,
//          index_t NumDimK,
//          typename AElementwiseOperation,
//          typename BElementwiseOperation,
//          typename CDEElementwiseOperation,
//          typename ADataType,
//          typename BDataType,
//          typename DDataType,
//          typename EDataType,
//          typename ComputeDataType>
//struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceContractionMultipleD<
//    NumDimM,
//    NumDimN,
//    NumDimK,
//    ADataType,
//    BDataType,
//    ck::Tuple<DDataType>,
//    EDataType,
//    AElementwiseOperation,
//    BElementwiseOperation,
//    CDEElementwiseOperation,
//    ComputeDataType>>
//{
//    using DeviceOp = DeviceContractionMultipleD<NumDimM,
//                                                NumDimN,
//                                                NumDimK,
//                                                ADataType,
//                                                BDataType,
//                                                ck::Tuple<DDataType>,
//                                                EDataType,
//                                                AElementwiseOperation,
//                                                BElementwiseOperation,
//                                                CDEElementwiseOperation,
//                                                ComputeDataType>;
//    static auto GetInstances()
//    {
//        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
//        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float> &&
//                     is_same_v<EDataType, float>)
//        {
//            if constexpr(is_same_v<ComputeDataType, float>)
//            {
//                add_device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance_unary(
//                    op_ptrs);
//            }
//        }
//        return op_ptrs;
//    }
//};
//
//} // namespace instance
//} // namespace device
//} // namespace tensor_operation
//} // namespace ck

namespace hiptensor
{
    ContractionSolutionInstances::ContractionSolutionInstances()
    {
        // Register all the solutions exactly once

        // Bilinear bf16
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          ck::bhalf_t,
                                          ck::bhalf_t,
                                          ck::Tuple<ck::bhalf_t>,
                                          ck::bhalf_t,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Bilinear,
                                          float>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          ck::bhalf_t,
                                          ck::bhalf_t,
                                          ck::Tuple<ck::bhalf_t>,
                                          ck::bhalf_t,
                                          CkHiptensorUnaryOp,
                                          CkHiptensorUnaryOp,
                                          CkBilinearUnary,
                                          float>());

        // Bilinear f16
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          ck::half_t,
                                          ck::half_t,
                                          ck::Tuple<ck::half_t>,
                                          ck::half_t,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Bilinear,
                                          float>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          ck::half_t,
                                          ck::half_t,
                                          ck::Tuple<ck::half_t>,
                                          ck::half_t,
                                          CkHiptensorUnaryOp,
                                          CkHiptensorUnaryOp,
                                          CkBilinearUnary,
                                          float>());

        // Bilinear f32
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          float,
                                          float,
                                          ck::Tuple<float>,
                                          float,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Bilinear,
                                          float>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          float,
                                          float,
                                          ck::Tuple<float>,
                                          float,
                                          CkHiptensorUnaryOp,
                                          CkHiptensorUnaryOp,
                                          CkBilinearUnary,
                                          float>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          float,
                                          float,
                                          ck::Tuple<float>,
                                          float,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Bilinear,
                                          ck::half_t>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          float,
                                          float,
                                          ck::Tuple<float>,
                                          float,
                                          CkHiptensorUnaryOp,
                                          CkHiptensorUnaryOp,
                                          CkBilinearUnary,
                                          ck::half_t>());


        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          float,
                                          float,
                                          ck::Tuple<float>,
                                          float,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Bilinear,
                                          ck::bhalf_t>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          float,
                                          float,
                                          ck::Tuple<float>,
                                          float,
                                          CkHiptensorUnaryOp,
                                          CkHiptensorUnaryOp,
                                          CkBilinearUnary,
                                          ck::bhalf_t>());


        // Bilinear complex f32
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          hipFloatComplex,
                                          hipFloatComplex,
                                          ck::Tuple<hipFloatComplex>,
                                          hipFloatComplex,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::BilinearComplex,
                                          hipFloatComplex>());

        // Bilinear f64
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          double,
                                          double,
                                          ck::Tuple<double>,
                                          double,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Bilinear,
                                          float>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          double,
                                          double,
                                          ck::Tuple<double>,
                                          double,
                                          CkHiptensorUnaryOp,
                                          CkHiptensorUnaryOp,
                                          CkBilinearUnary,
                                          float>());
                                        

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          double,
                                          double,
                                          ck::Tuple<double>,
                                          double,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Bilinear,
                                          double>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          double,
                                          double,
                                          ck::Tuple<double>,
                                          double,
                                          CkHiptensorUnaryOp,
                                          CkHiptensorUnaryOp,
                                          CkBilinearUnary,
                                          double>());

        // Bilinear complex f64
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          hipDoubleComplex,
                                          hipDoubleComplex,
                                          ck::Tuple<hipDoubleComplex>,
                                          hipDoubleComplex,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::BilinearComplex,
                                          hipDoubleComplex>());

        // Scale bf16
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          ck::bhalf_t,
                                          ck::bhalf_t,
                                          ck::Tuple<>,
                                          ck::bhalf_t,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Scale,
                                          float>());

        // Scale f16
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          ck::half_t,
                                          ck::half_t,
                                          ck::Tuple<>,
                                          ck::half_t,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Scale,
                                          float>());

        // Scale f32
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          float,
                                          float,
                                          ck::Tuple<>,
                                          float,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Scale,
                                          float>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          float,
                                          float,
                                          ck::Tuple<>,
                                          float,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Scale,
                                          ck::half_t>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          float,
                                          float,
                                          ck::Tuple<>,
                                          float,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Scale,
                                          ck::bhalf_t>());

        // scale complex f32
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          hipFloatComplex,
                                          hipFloatComplex,
                                          ck::Tuple<>,
                                          hipFloatComplex,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::ScaleComplex,
                                          hipFloatComplex>());

        // Scale f64
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          double,
                                          double,
                                          ck::Tuple<>,
                                          double,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Scale,
                                          float>());

        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          double,
                                          double,
                                          ck::Tuple<>,
                                          double,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Scale,
                                          double>());
        // scale complex f64
        registerSolutions(
            enumerateContractionSolutions<6,
                                          6,
                                          6,
                                          hipDoubleComplex,
                                          hipDoubleComplex,
                                          ck::Tuple<>,
                                          hipDoubleComplex,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::ScaleComplex,
                                          hipDoubleComplex>());
    }
} // namespace hiptensor
