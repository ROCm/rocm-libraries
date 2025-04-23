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

#ifndef HIPTENSOR_PERMUTATION_TYPES_HPP
#define HIPTENSOR_PERMUTATION_TYPES_HPP

#include <ck/ck.hpp>
#include <ck/utility/sequence.hpp>
#include <ostream>
#include <vector>

#include "permutation_types.hpp"
#include <combined_element_wise_operation.hpp>
#include <hiptensor/hiptensor_types.hpp>

namespace hiptensor
{
    using Uid = std::size_t;

    /**
     * \brief This enum respresents the solution instance is of permutation, elementwise binary operation
     * or elementwise trinary operation
     */
    enum struct ElementwiseInstanceType_t : int32_t
    {
        PERMUTATION,
        ELEMENTWISE_BINARY_OP,
        ELEMENTWISE_TRINARY_OP,
        UNKNOWN,
    };
    /**
     * \brief This enum decides the over the operation based on the inputs.
     * \details This enum decides the operation based on the in puts passed in the
     * hipTensorPermutationGetWorkspaceSize
     */
    enum struct PermutationOpId_t : int32_t
    {
        SCALE,
        PASS_THROUGH,
        UNKNOWN,
    };

    /**
     * \brief This enum categorizes the elementwise instance
     * \details Device instances run on GPUs, while host instances run on CPUs.
     */
    enum struct ElementwiseExecutionSpaceType_t : int32_t
    {
        DEVICE,
        HOST,
        UNKNOWN,
    };

    struct InstanceHyperParams
    {
        ck::index_t      mBlockSize;
        ck::index_t      mM0PerBlock;
        ck::index_t      mM1PerBlock;
        ck::index_t      mM0PerThread;
        ck::index_t      mM1PerThread;
        std::vector<int> mThreadClusterArrangeOrder;
        std::vector<int> mInScalarPerVectorSeq;
        std::vector<int> mOutScalarPerVectorSeq;
    };
} // namespace hiptensor

namespace std
{
    ostream& operator<<(ostream& os, hiptensor::PermutationOpId_t const&);

} // namespace std

#endif // HIPTENSOR_PERMUTATION_TYPES_HPP
