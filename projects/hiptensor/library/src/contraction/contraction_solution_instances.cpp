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

#include "contraction_solution_instances.hpp"
#include "contraction_solution.hpp"

// CK data types and utilities
#include <ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp>
#include <ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/tuple.hpp>
#include <hip/hip_complex.h>

// Ensure access to
#include "device/hiptensor_contraction_bilinear_instances.hpp"
#include "device/hiptensor_contraction_bilinear_unary_ops_instances.hpp"
#include "device/hiptensor_contraction_scale_instances.hpp"
#include "device/hiptensor_contraction_scale_unary_ops_instances.hpp"

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
                                          ck::bhalf_t>());

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        ck::bhalf_t,
                                                        ck::bhalf_t,
                                                        ck::Tuple<ck::bhalf_t>,
                                                        ck::bhalf_t,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
                                                        CkBilinearUnary,
                                                        ck::bhalf_t>());

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

        registerSolutions(enumerateContractionSolutions<6,
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
                                          ck::half_t>());

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        ck::half_t,
                                                        ck::half_t,
                                                        ck::Tuple<ck::half_t>,
                                                        ck::half_t,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
                                                        CkBilinearUnary,
                                                        ck::half_t>());

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

        registerSolutions(enumerateContractionSolutions<6,
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

        registerSolutions(enumerateContractionSolutions<6,
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

        registerSolutions(enumerateContractionSolutions<6,
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

        registerSolutions(enumerateContractionSolutions<6,
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

        registerSolutions(enumerateContractionSolutions<6,
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

        registerSolutions(enumerateContractionSolutions<6,
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
                                          ck::bhalf_t>());

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        ck::bhalf_t,
                                                        ck::bhalf_t,
                                                        ck::Tuple<>,
                                                        ck::bhalf_t,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
                                                        ck::tensor_operation::element_wise::Scale,
                                                        ck::bhalf_t>());

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

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        ck::bhalf_t,
                                                        ck::bhalf_t,
                                                        ck::Tuple<>,
                                                        ck::bhalf_t,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
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
                                          ck::half_t>());

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        ck::half_t,
                                                        ck::half_t,
                                                        ck::Tuple<>,
                                                        ck::half_t,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
                                                        ck::tensor_operation::element_wise::Scale,
                                                        ck::half_t>());

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

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        ck::half_t,
                                                        ck::half_t,
                                                        ck::Tuple<>,
                                                        ck::half_t,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
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

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        float,
                                                        float,
                                                        ck::Tuple<>,
                                                        float,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
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

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        float,
                                                        float,
                                                        ck::Tuple<>,
                                                        float,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
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

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        float,
                                                        float,
                                                        ck::Tuple<>,
                                                        float,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
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

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        double,
                                                        double,
                                                        ck::Tuple<>,
                                                        double,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
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

        registerSolutions(enumerateContractionSolutions<6,
                                                        6,
                                                        6,
                                                        double,
                                                        double,
                                                        ck::Tuple<>,
                                                        double,
                                                        CkHiptensorUnaryOp,
                                                        CkHiptensorUnaryOp,
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
