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

#include "contraction_cpu_reference_instances.hpp"
#include "contraction_cpu_reference_impl.hpp"

namespace hiptensor
{
    ContractionCpuReferenceInstances::ContractionCpuReferenceInstances()
    {
        // Register all the solutions exactly once
        // Bilinear f16
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        ck::half_t,
                                        ck::half_t,
                                        float,
                                        ck::Tuple<ck::half_t>,
                                        ck::half_t,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Bilinear,
                                        float>());

        // Bilinear bf16
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        ck::bhalf_t,
                                        ck::bhalf_t,
                                        float,
                                        ck::Tuple<ck::bhalf_t>,
                                        ck::bhalf_t,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Bilinear,
                                        float>());

        // Bilinear f32
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        float,
                                        float,
                                        float,
                                        ck::Tuple<float>,
                                        float,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Bilinear,
                                        ck::half_t>());

        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        float,
                                        float,
                                        float,
                                        ck::Tuple<float>,
                                        float,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Bilinear,
                                        ck::bhalf_t>());

        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        float,
                                        float,
                                        float,
                                        ck::Tuple<float>,
                                        float,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Bilinear,
                                        float>());

        // Bilinear complex f32
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        float,
                                        ck::Tuple<hipFloatComplex>,
                                        hipFloatComplex,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::BilinearComplex,
                                        hipFloatComplex>());

        // Bilinear f64
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        double,
                                        double,
                                        float,
                                        ck::Tuple<double>,
                                        double,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Bilinear,
                                        float>());

        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        double,
                                        double,
                                        double,
                                        ck::Tuple<double>,
                                        double,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Bilinear,
                                        double>());

        // Bilinear complex f64
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        double,
                                        ck::Tuple<hipDoubleComplex>,
                                        hipDoubleComplex,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::BilinearComplex,
                                        hipDoubleComplex>());

        // Scale f16
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        ck::half_t,
                                        ck::half_t,
                                        float,
                                        ck::Tuple<>,
                                        ck::half_t,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Scale,
                                        float>());

        // Scale bf16
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        ck::bhalf_t,
                                        ck::bhalf_t,
                                        float,
                                        ck::Tuple<>,
                                        ck::bhalf_t,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Scale,
                                        float>());

        // Scale f32
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        float,
                                        float,
                                        float,
                                        ck::Tuple<>,
                                        float,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Scale,
                                        ck::half_t>());

        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        float,
                                        float,
                                        float,
                                        ck::Tuple<>,
                                        float,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Scale,
                                        ck::bhalf_t>());

        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        float,
                                        float,
                                        float,
                                        ck::Tuple<>,
                                        float,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Scale,
                                        float>());

        // Scale complex f32
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        hipFloatComplex,
                                        hipFloatComplex,
                                        float,
                                        ck::Tuple<>,
                                        hipFloatComplex,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::ScaleComplex,
                                        hipFloatComplex>());

        // Scale f64
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        double,
                                        double,
                                        float,
                                        ck::Tuple<>,
                                        double,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Scale,
                                        float>());

        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        double,
                                        double,
                                        double,
                                        ck::Tuple<>,
                                        double,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::Scale,
                                        double>());

        // Scale complex f64
        registerSolutions(
            enumerateReferenceSolutions<6,
                                        6,
                                        6,
                                        hipDoubleComplex,
                                        hipDoubleComplex,
                                        double,
                                        ck::Tuple<>,
                                        hipDoubleComplex,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::PassThrough,
                                        ck::tensor_operation::element_wise::ScaleComplex,
                                        hipDoubleComplex>());
    }
} // namespace hiptensor
