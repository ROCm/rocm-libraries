/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_REDUCTION_EPILOGUE_HPP
#define HIPTENSOR_REDUCTION_EPILOGUE_HPP

#include <cstddef>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hiptensor/hiptensor_types.h>

namespace hiptensor
{
    // Largest rank of the output the reduction epilogue addresses.
    constexpr std::size_t kReductionEpilogueMaxRank = 8;

    // The parts of a reduction that do not reduce, on the caller's stream:
    //   D = alpha * opA(A) + beta * opC(C)
    // at every index of D. `lengths` are D's extents; `stridesD`, `stridesA` and `stridesC` are the
    // strides of the three tensors along those modes (A, when given, carries D's modes). A or C
    // may be nullptr, which drops the term and does not read the tensor. The terms are computed in
    // double for f64 data and in float otherwise, as the reduction kernels do.
    // Used for reductions whose output keeps every mode of A (nothing to reduce), for alpha == 0
    // (A is not read), and to copy C to D before a reduction kernel that updates D in place.
    hiptensorStatus_t reductionEpilogue(hiptensorDataType_t             type,
                                        std::vector<std::size_t> const& lengths,
                                        std::vector<std::size_t> const& stridesD,
                                        void*                           D,
                                        std::vector<std::size_t> const& stridesA,
                                        void const*                     A,
                                        hiptensorOperator_t             opA,
                                        double                          alpha,
                                        std::vector<std::size_t> const& stridesC,
                                        void const*                     C,
                                        hiptensorOperator_t             opC,
                                        double                          beta,
                                        hipStream_t                     stream);

    // Copies the tensor C to D, which has the same extents and strides, on the stream: one
    // asynchronous copy when the elements occupy a contiguous range, otherwise element by
    // element, so that memory between the elements of a padded or strided D is not written.
    hiptensorStatus_t reductionCopyTensor(hiptensorDataType_t             type,
                                          std::vector<std::size_t> const& lengths,
                                          std::vector<std::size_t> const& strides,
                                          void*                           D,
                                          void const*                     C,
                                          hipStream_t                     stream);
} // namespace hiptensor

#endif // HIPTENSOR_REDUCTION_EPILOGUE_HPP
