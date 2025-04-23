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

#include "contraction_cpu_reference.hpp"
#include "contraction_cpu_reference_impl.hpp"
#include "contraction_cpu_reference_instances.hpp"

hiptensorStatus_t hiptensorContractionReference(const hiptensorContractionPlan_t* plan,
                                                void const*                       alpha,
                                                void const*                       A,
                                                void const*                       B,
                                                void const*                       beta,
                                                void const*                       C,
                                                void*                             D,
                                                std::vector<size_t> const&        a_ms_ks_lengths,
                                                std::vector<size_t> const&        a_ms_ks_strides,
                                                std::vector<int32_t> const&       a_ms_ks_modes,
                                                std::vector<size_t> const&        b_ns_ks_lengths,
                                                std::vector<size_t> const&        b_ns_ks_strides,
                                                std::vector<int32_t> const&       b_ns_ks_modes,
                                                std::vector<size_t> const&        c_ms_ns_lengths,
                                                std::vector<size_t> const&        c_ms_ns_strides,
                                                std::vector<int32_t> const&       c_ms_ns_modes,
                                                std::vector<size_t> const&        d_ms_ns_lengths,
                                                std::vector<size_t> const&        d_ms_ns_strides,
                                                std::vector<int32_t> const&       d_ms_ns_modes,
                                                hipDataType                       typeA,
                                                hipDataType                       typeB,
                                                hipDataType                       typeC,
                                                hipDataType                       typeD,
                                                void*                             workspace)
{
    auto& instances   = hiptensor::ContractionCpuReferenceInstances::instance();
    auto  computeType = plan->mContractionDesc.mComputeType;
    auto  candidates
        = (C == nullptr) ? instances->allSolutions().query(
                               typeA, typeB, hiptensor::NONE_TYPE, typeD, computeType)
                         : instances->allSolutions().query(typeA, typeB, typeC, typeD, computeType);

    auto toCKVec
        = [](auto& inputVec) { return std::vector<ck::index_t>(inputVec.begin(), inputVec.end()); };

    if(candidates.solutionCount() != 1)
    {
        return HIPTENSOR_STATUS_INTERNAL_ERROR;
    }
    else
    {
        auto refCandidate      = candidates.solutions().begin()->second;
        auto [errorCode, time] = (*refCandidate)(alpha,
                                                 A,
                                                 B,
                                                 beta,
                                                 C,
                                                 D,
                                                 a_ms_ks_lengths,
                                                 a_ms_ks_strides,
                                                 a_ms_ks_modes,
                                                 b_ns_ks_lengths,
                                                 b_ns_ks_strides,
                                                 b_ns_ks_modes,
                                                 c_ms_ns_lengths,
                                                 c_ms_ns_strides,
                                                 c_ms_ns_modes,
                                                 d_ms_ns_lengths,
                                                 d_ms_ns_strides,
                                                 d_ms_ns_modes,
                                                 workspace,
                                                 0);
        return errorCode;
    }
}
