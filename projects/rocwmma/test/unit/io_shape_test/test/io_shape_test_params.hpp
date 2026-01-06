/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef IO_SHAPE_TEST_PARAMS_HPP
#define IO_SHAPE_TEST_PARAMS_HPP

#include <type_traits>

#include "detail/io_shape.hpp"
#include "kernel_generator.hpp"
#include "unit_test.hpp"

namespace rocwmma
{
    template <typename BlockSizes, typename DataTypes>
    struct IoShapeTestParams : public UnitTestParams
    {
        using Base        = UnitTestParams;
        using MatrixTypes = std::tuple<matrix_a, matrix_b, accumulator>;
        using DataLayouts = typename Base::TestLayoutsAll;
        using KernelParams =
            typename CombineLists<MatrixTypes, BlockSizes, DataTypes, DataLayouts>::Result;

        // Assemble the kernel generator
        using GeneratorImpl   = IOShapeGenerator;
        using KernelGenerator = KernelGenerator<KernelParams, GeneratorImpl>;

        // Sanity check for kernel generator
        static_assert(std::is_same<typename GeneratorImpl::ResultT, typename Base::KernelT>::value,
                      "Kernels from this generator do not match testing interface");

        static inline typename KernelGenerator::ResultT kernels()
        {
            return KernelGenerator::generate();
        }

        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            auto warpSize = HipDevice::instance()->warpSize();
            // clang-format off
            return { {warpSize, 1} };
            // clang-format on
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            // clang-format off
            return { {1024, 1024} };
            // clang-format on
        }
    };

} // namespace rocwmma
#endif // IO_SHAPE_TEST_PARAMS_HPP
