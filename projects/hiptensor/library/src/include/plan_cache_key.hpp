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

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "data_types.hpp"
#include "hash.hpp"

namespace hiptensor
{
    namespace detail
    {
        inline void appendTensor(std::vector<std::size_t>& out, hiptensorTensorDescriptor_t desc)
        {
            if(desc != nullptr)
            {
                out.insert(out.end(), desc->mLengths.begin(), desc->mLengths.end());
            }
        }

        inline void appendStrides(std::vector<std::size_t>& out, hiptensorTensorDescriptor_t desc)
        {
            if(desc != nullptr)
            {
                out.insert(out.end(), desc->mStrides.begin(), desc->mStrides.end());
            }
        }
    } // namespace detail

    // The plan cache key of earlier versions: tag, data types of A to D, compute type, operation
    // type, contraction kind, and the lengths, strides and modes of A to D. Composable Kernel
    // plans keep it, so plan cache files written by earlier versions still apply to them.
    inline std::size_t legacyPlanCacheKey(hiptensorOperationDescriptor_t desc)
    {
        auto typeOf = [](hiptensorTensorDescriptor_t d) { return d ? d->mType : NONE_TYPE; };

        std::vector<std::size_t> arrLSM;
        for(auto d : {desc->mDescA, desc->mDescB, desc->mDescC, desc->mDescD})
        {
            detail::appendTensor(arrLSM, d);
        }
        for(auto d : {desc->mDescA, desc->mDescB, desc->mDescC, desc->mDescD})
        {
            detail::appendStrides(arrLSM, d);
        }
        arrLSM.insert(arrLSM.end(), desc->mModeA.begin(), desc->mModeA.end());
        arrLSM.insert(arrLSM.end(), desc->mModeB.begin(), desc->mModeB.end());
        arrLSM.insert(arrLSM.end(), desc->mModeC.begin(), desc->mModeC.end());
        arrLSM.insert(arrLSM.end(), desc->mModeD.begin(), desc->mModeD.end());

        return Hash{}(desc->mTag,
                      typeOf(desc->mDescA),
                      typeOf(desc->mDescB),
                      typeOf(desc->mDescC),
                      typeOf(desc->mDescD),
                      desc->mDescCompute,
                      desc->mOperationType,
                      desc->mContractionOpId,
                      arrLSM);
    }

    // The plan cache key of the other backends: the backend's cache name and the legacy key, with
    // what the legacy key leaves out: the unary operators, the operators of the elementwise and
    // reduction operations (opAC, opABC), the alignments, and the trinary contraction's output E.
    inline std::size_t fullPlanCacheKey(hiptensorOperationDescriptor_t desc,
                                        std::string const&             cacheName)
    {
        std::vector<std::size_t> rest;
        auto add = [&rest](auto value) { rest.push_back(static_cast<std::size_t>(value)); };
        for(auto op : {desc->mOpA, desc->mOpB, desc->mOpC, desc->mOpD, desc->mOpAC, desc->mOpABC})
        {
            add(op);
        }
        for(auto d : {desc->mDescA, desc->mDescB, desc->mDescC, desc->mDescD, desc->mDescE})
        {
            add(d != nullptr ? d->mAlignmentRequirement : 0u);
        }
        auto const descE = desc->mDescE;
        add(descE != nullptr ? descE->mType : NONE_TYPE);
        add(descE != nullptr ? descE->mLengths.size() : 0u);
        detail::appendTensor(rest, descE);
        detail::appendStrides(rest, descE);
        add(desc->mModeE.size());
        rest.insert(rest.end(), desc->mModeE.begin(), desc->mModeE.end());

        return Hash{}(Hash{}(cacheName), legacyPlanCacheKey(desc), rest);
    }
} // namespace hiptensor
