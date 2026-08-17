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

#include "contraction_solution.hpp"
#include "hash.hpp"
#include "hiptensor_options.hpp"

namespace hiptensor
{
    std::array<std::vector<std::size_t>, 8>
        normalizeTensorModes(std::vector<std::size_t> const& a_ms_ks_lengths,
                             std::vector<std::size_t> const& a_ms_ks_strides,
                             std::vector<int32_t> const&     a_ms_ks_modes,
                             std::vector<std::size_t> const& b_ns_ks_lengths,
                             std::vector<std::size_t> const& b_ns_ks_strides,
                             std::vector<int32_t> const&     b_ns_ks_modes,
                             std::vector<std::size_t> const& e_ms_ns_lengths,
                             std::vector<std::size_t> const& e_ms_ns_strides,
                             std::vector<int32_t> const&     e_ms_ns_modes)
    {
        std::vector<std::size_t> normal_a_ms_ks_lengths(MaxNumDimsM + MaxNumDimsK, 1);
        std::vector<std::size_t> normal_a_ms_ks_strides(MaxNumDimsM + MaxNumDimsK, 1);
        std::vector<int32_t>     normal_a_ms_ks_modes(MaxNumDimsM + MaxNumDimsK, -1);
        std::vector<std::size_t> normal_b_ns_ks_lengths(MaxNumDimsK + MaxNumDimsN, 1);
        std::vector<std::size_t> normal_b_ns_ks_strides(MaxNumDimsK + MaxNumDimsN, 1);
        std::vector<int32_t>     normal_b_ns_ks_modes(MaxNumDimsK + MaxNumDimsN, -1);
        std::vector<std::size_t> normal_e_ms_ns_lengths(MaxNumDimsM + MaxNumDimsN, 1);
        std::vector<std::size_t> normal_e_ms_ns_strides(MaxNumDimsM + MaxNumDimsN, 1);
        std::vector<int32_t>     normal_e_ms_ns_modes(MaxNumDimsM + MaxNumDimsN, -1);
        int                      mOffset = 0;
        int                      nOffset = 0;

        // reorder m, n in A, B
        for(int i = 0; i < e_ms_ns_modes.size(); i++)
        {
            if(auto aIt = std::find(a_ms_ks_modes.cbegin(), a_ms_ks_modes.cend(), e_ms_ns_modes[i]);
               aIt != a_ms_ks_modes.cend())
            {
                auto offset                     = std::distance(a_ms_ks_modes.cbegin(), aIt);
                normal_a_ms_ks_modes[mOffset]   = a_ms_ks_modes[offset];
                normal_a_ms_ks_lengths[mOffset] = a_ms_ks_lengths[offset];
                normal_a_ms_ks_strides[mOffset] = a_ms_ks_strides[offset];
                mOffset++;
            }
            else
            {
                auto bIt
                    = std::find(b_ns_ks_modes.cbegin(), b_ns_ks_modes.cend(), e_ms_ns_modes[i]);
                auto offset                     = std::distance(b_ns_ks_modes.cbegin(), bIt);
                normal_b_ns_ks_modes[nOffset]   = b_ns_ks_modes[offset];
                normal_b_ns_ks_lengths[nOffset] = b_ns_ks_lengths[offset];
                normal_b_ns_ks_strides[nOffset] = b_ns_ks_strides[offset];
                nOffset++;
            }
        }

        assert(mOffset > 0 && nOffset > 0);
        for(; mOffset < MaxNumDimsM; mOffset++)
        {
            normal_a_ms_ks_lengths[mOffset] = 1;
            normal_a_ms_ks_strides[mOffset] = normal_a_ms_ks_strides[mOffset - 1];
        }
        for(; nOffset < MaxNumDimsN; nOffset++)
        {
            normal_b_ns_ks_lengths[nOffset] = 1;
            normal_b_ns_ks_strides[nOffset] = normal_b_ns_ks_strides[nOffset - 1];
        }

        // reorder k in A, B - Do not check if A and B have same k here.
        for(int i = 0; i < a_ms_ks_modes.size(); i++)
        {
            if(auto it = std::find(b_ns_ks_modes.cbegin(), b_ns_ks_modes.cend(), a_ms_ks_modes[i]);
               it != b_ns_ks_modes.cend())
            {
                normal_a_ms_ks_modes[mOffset]   = a_ms_ks_modes[i];
                normal_a_ms_ks_lengths[mOffset] = a_ms_ks_lengths[i];
                normal_a_ms_ks_strides[mOffset] = a_ms_ks_strides[i];
                mOffset++;

                auto bOffset                    = std::distance(b_ns_ks_modes.cbegin(), it);
                normal_b_ns_ks_modes[nOffset]   = b_ns_ks_modes[bOffset];
                normal_b_ns_ks_lengths[nOffset] = b_ns_ks_lengths[bOffset];
                normal_b_ns_ks_strides[nOffset] = b_ns_ks_strides[bOffset];
                nOffset++;
            }
        }

        assert(mOffset > 0 && nOffset > 0);
        for(; mOffset < MaxNumDimsM + MaxNumDimsK; mOffset++)
        {
            normal_a_ms_ks_lengths[mOffset] = 1;
            normal_a_ms_ks_strides[mOffset] = normal_a_ms_ks_strides[mOffset - 1];
        }
        for(; nOffset < MaxNumDimsN + MaxNumDimsK; nOffset++)
        {
            normal_b_ns_ks_lengths[nOffset] = 1;
            normal_b_ns_ks_strides[nOffset] = normal_b_ns_ks_strides[nOffset - 1];
        }

        // reorder m, n in D, E
        std::vector<int32_t> contraction_result_modes(MaxNumDimsM + MaxNumDimsN, -1);
        std::copy(normal_a_ms_ks_modes.cbegin(),
                  normal_a_ms_ks_modes.cbegin() + MaxNumDimsM,
                  contraction_result_modes.begin());
        std::copy(normal_b_ns_ks_modes.cbegin(),
                  normal_b_ns_ks_modes.cbegin() + MaxNumDimsN,
                  contraction_result_modes.begin() + MaxNumDimsM);

        for(int i = 0; i < contraction_result_modes.size(); i++)
        {
            auto it = std::find(
                e_ms_ns_modes.cbegin(), e_ms_ns_modes.cend(), contraction_result_modes[i]);
            if(it != e_ms_ns_modes.cend())
            {
                auto offset               = std::distance(e_ms_ns_modes.cbegin(), it);
                normal_e_ms_ns_lengths[i] = e_ms_ns_lengths[offset];
                normal_e_ms_ns_strides[i] = e_ms_ns_strides[offset];
            }
            else
            {
                normal_e_ms_ns_lengths[i] = 1;
                normal_e_ms_ns_strides[i] = normal_e_ms_ns_strides[i - 1];
            }
        }

        return {
            normal_a_ms_ks_lengths,
            normal_a_ms_ks_strides,
            normal_b_ns_ks_lengths,
            normal_b_ns_ks_strides,
            normal_e_ms_ns_lengths,
            normal_e_ms_ns_strides,
            normal_e_ms_ns_lengths,
            normal_e_ms_ns_strides,
        };
    }

    bool isColMajorLayout(std::vector<std::size_t> const& strides,
                          std::vector<std::size_t> const& lengths)
    {
        if(strides.empty())
        {
            using hiptensor::HiptensorOptions;
            auto& options = HiptensorOptions::instance();
            return options->isColMajorStrides();
        }

        if(strides.size() > 0 && strides[0] != 1)
            return false;
        for(int i = 1; i < static_cast<int>(strides.size()); i++)
        {
            if(strides[i] != strides[i - 1] * lengths[i - 1])
            {
                return false;
            }
        }
        return true;
    }

    std::vector<std::size_t>
        applyCKColMajorStridesOptimizationForContraction(std::vector<std::size_t> const& lengths)
    {
        std::vector<std::size_t> strides(lengths.size(), 1);
        // Assign second half of strides
        int stride = 1;
        for(int s = strides.size() / 2 - 1; s >= 0; s--)
        {
            strides[s] = stride;
            stride *= lengths[s];
        }

        // Assign first half of strides
        for(int s = strides.size() - 1; s > static_cast<int>(strides.size()) / 2 - 1; s--)
        {
            strides[s] = stride;
            stride *= lengths[s];
        }
        return strides;
    }

    ContractionSolution::ContractionSolution(
        std::unique_ptr<ck::tensor_operation::device::BaseOperator>&& deviceOp,
        std::unique_ptr<ContractionSolutionParams>&&                  params)
        : mM(0)
        , mN(0)
        , mK(0)
        , mBytes(0)
        , mValid(false)
        , mDeviceOp(std::move(deviceOp))
        , mParams(std::move(params))
    {
    }

    std::tuple<hiptensorStatus_t, float>
        ContractionSolution::operator()(void const*                alpha,
                                        void const*                A,
                                        void const*                B,
                                        void const*                beta,
                                        void const*                D,
                                        void*                      E,
                                        std::vector<std::size_t>   a_ms_ns_lengths,
                                        std::vector<std::size_t>   a_ms_ks_strides,
                                        std::vector<int32_t>       a_ms_ks_modes,
                                        std::vector<std::size_t>   b_ns_ks_lengths,
                                        std::vector<std::size_t>   b_ns_ks_strides,
                                        std::vector<int32_t>       b_ns_ks_modes,
                                        std::vector<std::size_t>   ds_ms_ns_lengths,
                                        std::vector<std::size_t>   ds_ms_ns_strides,
                                        std::vector<int32_t>       ds_ms_ns_modes,
                                        std::vector<std::size_t>   e_ms_ns_lengths,
                                        std::vector<std::size_t>   e_ms_ns_strides,
                                        std::vector<int32_t>       e_ms_ns_modes,
                                        ContractionUnaryOps const& unaryOps,
                                        void*                      workspacePtr,
                                        unsigned long              workspaceSize,
                                        StreamConfig const& streamConfig /*= StreamConfig{}*/)
    {
        if(!initArgs(alpha,
                     A,
                     B,
                     beta,
                     D,
                     E,
                     a_ms_ns_lengths,
                     a_ms_ks_strides,
                     a_ms_ks_modes,
                     b_ns_ks_lengths,
                     b_ns_ks_strides,
                     b_ns_ks_modes,
                     ds_ms_ns_lengths,
                     ds_ms_ns_strides,
                     ds_ms_ns_modes,
                     e_ms_ns_lengths,
                     e_ms_ns_strides,
                     e_ms_ns_modes,
                     unaryOps,
                     workspacePtr))
        {
            return {HIPTENSOR_STATUS_INTERNAL_ERROR, -1.0f};
        }

        if(this->workspaceSize() > workspaceSize)
        {
            resetInvokerArgs();
            return {HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE, -1.0f};
        }
        auto time = mInvokerPtr->Run(mInvokerArgPtr.get(), streamConfig);
        resetInvokerArgs();

        return {HIPTENSOR_STATUS_SUCCESS, time};
    }

    bool ContractionSolution::isValid() const
    {
        return mValid;
    }

    std::unique_ptr<ContractionSolutionParams> const& ContractionSolution::params() const
    {
        return mParams;
    }

    size_t ContractionSolution::uid() const
    {
        // Platform-stable uid: hash the kernel type string (via FNV-1a in Hash{})
        // together with the data-type / op parameters so that distinct CK template
        // instantiations that share the same geometry (GetTypeString) but differ in
        // data types still produce distinct uids.
        auto const& params = mParams;
        return Hash{}(mDeviceOp->GetTypeString(),
                      params->typeA(),
                      params->typeB(),
                      params->typeC(),
                      params->typeD(),
                      params->typeCompute(),
                      params->opA(),
                      params->opB(),
                      params->opCDE());
    }

    std::tuple<ck::index_t, ck::index_t, ck::index_t> ContractionSolution::problemDims() const
    {
        return std::make_tuple(mM, mN, mK);
    }

    std::size_t ContractionSolution::problemBytes() const
    {
        return mBytes;
    }

    std::string ContractionSolution::kernelName() const
    {
        return mDeviceOp->GetTypeString();
    }

    size_t ContractionSolution::workspaceSize() const
    {
        if(mValid)
        {
            return mDeviceOp->GetWorkSpaceSize(mInvokerArgPtr.get());
        }
        else
        {
            return 0;
        }
    }

    void ContractionSolution::resetArgs()
    {
        mM     = 0;
        mN     = 0;
        mK     = 0;
        mBytes = 0;

        mInvokerArgPtr.reset(nullptr);
        mInvokerPtr.reset(nullptr);

        mValid = false;
    }

    void ContractionSolution::resetInvokerArgs()
    {
        mInvokerArgPtr.reset(nullptr);
    }
} // namespace hiptensor
