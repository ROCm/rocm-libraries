// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <gtest/gtest.h>
#include <hipdnn-gpu-ref/GpuFpReferenceValidation.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gpu_matmul_ref_test
{

namespace
{

std::vector<int64_t> calculateCDims(const std::vector<int64_t>& aDims,
                                    const std::vector<int64_t>& bDims)
{
    if(aDims.size() != bDims.size())
    {
        throw std::invalid_argument("Matmul requires A and B tensor rank to be equal.");
    }
    if(aDims.size() < 2)
    {
        throw std::invalid_argument("Matmul requires A and B tensor rank to be at least 2.");
    }
    if(aDims[aDims.size() - 1] != bDims[bDims.size() - 2])
    {
        throw std::invalid_argument("Matmul requires that the last dimension of A and the "
                                    "second to last dimension of B (i.e. K) are equal.");
    }
    if(std::any_of(aDims.begin(), aDims.end(), [](int64_t d) { return d <= 0; }))
    {
        throw std::invalid_argument("Matmul requires A tensor dimension to be 1 or greater.");
    }
    if(std::any_of(bDims.begin(), bDims.end(), [](int64_t d) { return d <= 0; }))
    {
        throw std::invalid_argument("Matmul requires B tensor dimension to be 1 or greater.");
    }
    std::vector<int64_t> out;
    for(size_t i = 0; i < aDims.size() - 2; ++i)
    {
        if(std::max(aDims[i], bDims[i]) % std::min(aDims[i], bDims[i]) != 0)
        {
            throw std::invalid_argument(
                "Matmul requires that A and B tensor are broadcast-compatible.");
        }
        out.push_back(std::max(aDims[i], bDims[i]));
    }
    out.push_back(aDims[aDims.size() - 2]);
    out.push_back(bDims[bDims.size() - 1]);
    return out;
}

} // namespace

struct MatmulTestCase
{
    std::vector<int64_t> aDims;
    std::vector<int64_t> bDims;
    std::vector<int64_t> cDims;
    std::vector<int64_t> aStrides;
    std::vector<int64_t> bStrides;
    std::vector<int64_t> cStrides;

    MatmulTestCase(const std::vector<int64_t>& aDimsLocal,
                   const std::vector<int64_t>& bDimsLocal,
                   const std::vector<int64_t>& cDimsLocal,
                   const std::vector<int64_t>& aStridesLocal,
                   const std::vector<int64_t>& bStridesLocal,
                   const std::vector<int64_t>& cStridesLocal)
        : aDims(aDimsLocal)
        , bDims(bDimsLocal)
        , cDims(cDimsLocal)
        , aStrides(aStridesLocal)
        , bStrides(bStridesLocal)
        , cStrides(cStridesLocal)
    {
    }

    MatmulTestCase(const std::vector<int64_t>& aDimsLocal, const std::vector<int64_t>& bDimsLocal)
        : MatmulTestCase(
              aDimsLocal,
              bDimsLocal,
              calculateCDims(aDimsLocal, bDimsLocal),
              hipdnn_data_sdk::utilities::generateStrides(aDimsLocal),
              hipdnn_data_sdk::utilities::generateStrides(bDimsLocal),
              hipdnn_data_sdk::utilities::generateStrides(calculateCDims(aDimsLocal, bDimsLocal)))
    {
    }

    friend std::ostream& operator<<(std::ostream& ss, const MatmulTestCase& testCase)
    {
        ss << "(aDims:";
        hipdnn_data_sdk::utilities::vecToStream(ss, testCase.aDims);
        ss << " bDims:";
        hipdnn_data_sdk::utilities::vecToStream(ss, testCase.bDims);
        ss << " cDims:";
        hipdnn_data_sdk::utilities::vecToStream(ss, testCase.cDims);
        ss << " aStrides:";
        hipdnn_data_sdk::utilities::vecToStream(ss, testCase.aStrides);
        ss << " bStrides:";
        hipdnn_data_sdk::utilities::vecToStream(ss, testCase.bStrides);
        ss << " cStrides:";
        hipdnn_data_sdk::utilities::vecToStream(ss, testCase.cStrides);
        ss << ")";
        return ss;
    }
};

template <typename T>
void assertAllClose(hipdnn_data_sdk::utilities::TensorBase<T>& expected,
                    hipdnn_data_sdk::utilities::TensorBase<T>& actual,
                    float tolerance,
                    const std::string& tensorName = "Tensor")
{
    auto validator = hipdnn_gpu_ref::GpuFpReferenceValidation<T>(tolerance, tolerance);
    ASSERT_TRUE(validator.allClose(expected, actual)) << tensorName << " failed verification";
}

} // namespace gpu_matmul_ref_test
