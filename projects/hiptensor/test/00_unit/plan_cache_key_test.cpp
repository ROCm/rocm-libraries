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

// The plan cache keys (CPU only):
//  - Composable Kernel plans keep the key of earlier versions, so plan cache files written by
//    them still apply;
//  - the other backends' keys separate what that key leaves out (the unary, elementwise and
//    reduction operators, the alignments, the trinary contraction's output E) and the backend's
//    cache name.

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "plan_cache_key.hpp"

namespace
{
    using hiptensor::fullPlanCacheKey;
    using hiptensor::legacyPlanCacheKey;

    // The key of earlier versions as they computed it (PlanCache::getHashID before the backends).
    std::size_t earlierVersionKey(hiptensorOperationDescriptor_t desc)
    {
        auto lengths = [](hiptensorTensorDescriptor_t d) {
            return d ? d->mLengths : std::vector<std::size_t>{};
        };
        auto strides = [](hiptensorTensorDescriptor_t d) {
            return d ? d->mStrides : std::vector<std::size_t>{};
        };
        auto type
            = [](hiptensorTensorDescriptor_t d) { return d ? d->mType : hiptensor::NONE_TYPE; };

        std::vector<std::size_t> arrLSM;
        for(auto d : {desc->mDescA, desc->mDescB, desc->mDescC, desc->mDescD})
        {
            auto v = lengths(d);
            arrLSM.insert(arrLSM.end(), v.begin(), v.end());
        }
        for(auto d : {desc->mDescA, desc->mDescB, desc->mDescC, desc->mDescD})
        {
            auto v = strides(d);
            arrLSM.insert(arrLSM.end(), v.begin(), v.end());
        }
        arrLSM.insert(arrLSM.end(), desc->mModeA.begin(), desc->mModeA.end());
        arrLSM.insert(arrLSM.end(), desc->mModeB.begin(), desc->mModeB.end());
        arrLSM.insert(arrLSM.end(), desc->mModeC.begin(), desc->mModeC.end());
        arrLSM.insert(arrLSM.end(), desc->mModeD.begin(), desc->mModeD.end());

        return hiptensor::Hash{}(desc->mTag,
                                 type(desc->mDescA),
                                 type(desc->mDescB),
                                 type(desc->mDescC),
                                 type(desc->mDescD),
                                 desc->mDescCompute,
                                 desc->mOperationType,
                                 desc->mContractionOpId,
                                 arrLSM);
    }

    hiptensorTensorDescriptor tensor(std::vector<std::size_t> lengths,
                                     std::vector<std::size_t> strides,
                                     hiptensorDataType_t      type = HIPTENSOR_R_32F)
    {
        return hiptensorTensorDescriptor{type, std::move(lengths), std::move(strides), 16};
    }

    struct Reduction
    {
        hiptensorTensorDescriptor      a = tensor({8, 6}, {1, 8}), d = tensor({8}, {1});
        hiptensorOperationDescriptor   desc{};
        hiptensorOperationDescriptor_t operator()()
        {
            desc.mOperationType = HIPTENSOR_REDUCTION;
            desc.mDescA         = &a;
            desc.mModeA         = {'m', 'n'};
            desc.mOpA           = HIPTENSOR_OP_IDENTITY;
            desc.mDescD         = &d;
            desc.mModeD         = {'m'};
            desc.mOpD           = HIPTENSOR_OP_IDENTITY;
            desc.mOpAC          = HIPTENSOR_OP_ADD;
            desc.mOpABC         = HIPTENSOR_OP_IDENTITY;
            desc.mDescCompute   = HIPTENSOR_COMPUTE_DESC_32F;
            return &desc;
        }
    };

    struct TrinaryContraction
    {
        hiptensorTensorDescriptor a = tensor({8, 5}, {1, 8}), b = tensor({5, 4}, {1, 5}),
                                  c = tensor({4, 6}, {1, 4}), d = tensor({8, 6}, {1, 8}),
                                  e = tensor({8, 6}, {1, 8});
        hiptensorOperationDescriptor   desc{};
        hiptensorOperationDescriptor_t operator()()
        {
            desc.mOperationType = HIPTENSOR_CONTRACTION_TRINARY;
            desc.mDescA         = &a;
            desc.mModeA         = {'m', 'k'};
            desc.mDescB         = &b;
            desc.mModeB         = {'k', 'u'};
            desc.mDescC         = &c;
            desc.mModeC         = {'u', 'n'};
            desc.mDescD         = &d;
            desc.mModeD         = {'m', 'n'};
            desc.mDescE         = &e;
            desc.mModeE         = {'m', 'n'};
            desc.mDescCompute   = HIPTENSOR_COMPUTE_DESC_32F;
            return &desc;
        }
    };
} // namespace

TEST(PlanCacheKey, LegacyKeyIsTheKeyOfEarlierVersions)
{
    Reduction          reduction;
    TrinaryContraction trinary;
    EXPECT_EQ(legacyPlanCacheKey(reduction()), earlierVersionKey(reduction()));
    EXPECT_EQ(legacyPlanCacheKey(trinary()), earlierVersionKey(trinary()));

    // A contraction without C
    trinary();
    trinary.desc.mOperationType = HIPTENSOR_CONTRACTION;
    trinary.desc.mDescC         = nullptr;
    trinary.desc.mModeC.clear();
    trinary.desc.mDescE = nullptr;
    trinary.desc.mModeE.clear();
    EXPECT_EQ(legacyPlanCacheKey(&trinary.desc), earlierVersionKey(&trinary.desc));
}

TEST(PlanCacheKey, FullKeySeparatesOperators)
{
    Reduction sum, max;
    sum();
    max();
    max.desc.mOpAC = HIPTENSOR_OP_MAX;
    // the legacy key does not see the reduction operator
    EXPECT_EQ(legacyPlanCacheKey(&sum.desc), legacyPlanCacheKey(&max.desc));
    EXPECT_NE(fullPlanCacheKey(&sum.desc, "rocke"), fullPlanCacheKey(&max.desc, "rocke"));

    Reduction sqrtA;
    sqrtA();
    sqrtA.desc.mOpA = HIPTENSOR_OP_SQRT;
    EXPECT_NE(fullPlanCacheKey(&sum.desc, "rocke"), fullPlanCacheKey(&sqrtA.desc, "rocke"));

    Reduction aligned;
    aligned();
    aligned.a.mAlignmentRequirement = 256;
    EXPECT_NE(fullPlanCacheKey(&sum.desc, "rocke"), fullPlanCacheKey(&aligned.desc, "rocke"));
}

TEST(PlanCacheKey, FullKeySeparatesTheTrinaryOutput)
{
    TrinaryContraction packed, strided, renamed, wider;
    packed();
    strided();
    strided.e.mStrides = {6, 1};
    renamed();
    renamed.desc.mModeE = {'n', 'm'};
    wider();
    wider.e.mType = HIPTENSOR_R_64F;
    EXPECT_EQ(legacyPlanCacheKey(&packed.desc), legacyPlanCacheKey(&strided.desc));
    for(auto* other : {&strided, &renamed, &wider})
    {
        EXPECT_NE(fullPlanCacheKey(&packed.desc, "rocke"), fullPlanCacheKey(&other->desc, "rocke"));
    }
}

TEST(PlanCacheKey, FullKeySeparatesBackends)
{
    Reduction reduction;
    auto*     desc = reduction();
    EXPECT_NE(fullPlanCacheKey(desc, "rocke"), fullPlanCacheKey(desc, "rocke-jit"));
    EXPECT_NE(fullPlanCacheKey(desc, "rocke"), legacyPlanCacheKey(desc));
    EXPECT_EQ(fullPlanCacheKey(desc, "rocke"), fullPlanCacheKey(desc, "rocke"));
}
