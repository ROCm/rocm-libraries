// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_integration_tests::golden
{

struct ComparisonResult
{
    bool passed = true;
    size_t totalElements = 0;
    size_t mismatchCount = 0;
    double maxAbsError = 0.0;
    double maxRelError = 0.0;
    int64_t worstFlatIndex = -1;
    double worstExpected = 0.0;
    double worstActual = 0.0;
    float usedAtol = 0.0f;
    float usedRtol = 0.0f;
};

inline std::vector<int64_t> flatIndexToMultiDim(int64_t flatIndex,
                                                 const std::vector<int64_t>& dims)
{
    std::vector<int64_t> result(dims.size());
    for(auto i = static_cast<int64_t>(dims.size()) - 1; i >= 0; --i)
    {
        auto idx = static_cast<size_t>(i);
        result[idx] = flatIndex % dims[idx];
        flatIndex /= dims[idx];
    }
    return result;
}

inline std::string formatMultiDimIndex(const std::vector<int64_t>& indices)
{
    std::ostringstream oss;
    oss << "(";
    for(size_t i = 0; i < indices.size(); ++i)
    {
        if(i > 0)
        {
            oss << ", ";
        }
        oss << indices[i];
    }
    oss << ")";
    return oss.str();
}

inline std::string formatShape(const std::vector<int64_t>& dims)
{
    std::ostringstream oss;
    oss << "[";
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(i > 0)
        {
            oss << ", ";
        }
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

template <typename T>
ComparisonResult compareTensors(const hipdnn_data_sdk::utilities::ITensor& expected,
                                const hipdnn_data_sdk::utilities::ITensor& actual,
                                float atol,
                                float rtol)
{
    ComparisonResult result;
    result.usedAtol = atol;
    result.usedRtol = rtol;
    result.totalElements = expected.elementCount();

    auto expectedIt = expected.cbegin();
    auto actualIt = actual.cbegin();
    auto expectedEnd = expected.cend();

    int64_t idx = 0;
    while(expectedIt != expectedEnd)
    {
        auto expVal = static_cast<double>(static_cast<T>(*expectedIt));
        auto actVal = static_cast<double>(static_cast<T>(*actualIt));

        double absErr = std::abs(expVal - actVal);
        double denom = std::max(std::abs(expVal), std::abs(actVal));
        double relErr = (denom > 0.0) ? absErr / denom : 0.0;

        bool elementPassed = absErr <= static_cast<double>(atol)
                             || absErr <= static_cast<double>(rtol) * denom;

        if(!elementPassed)
        {
            ++result.mismatchCount;
            if(absErr > result.maxAbsError)
            {
                result.maxAbsError = absErr;
                result.maxRelError = relErr;
                result.worstFlatIndex = idx;
                result.worstExpected = expVal;
                result.worstActual = actVal;
            }
        }

        ++expectedIt;
        ++actualIt;
        ++idx;
    }

    result.passed = (result.mismatchCount == 0);
    return result;
}

inline std::string formatComparisonFailure(
    const std::filesystem::path& bundlePath,
    int64_t tensorUid,
    const std::string& tensorName,
    const std::vector<int64_t>& shape,
    const std::string& dtype,
    const ComparisonResult& result)
{
    std::ostringstream oss;
    oss << std::setprecision(8);
    oss << "\n=== Golden Reference Mismatch ===\n";
    oss << "  Bundle:        " << bundlePath << "\n";
    oss << "  Tensor UID:    " << tensorUid;
    if(!tensorName.empty())
    {
        oss << " (" << tensorName << ")";
    }
    oss << "\n";
    oss << "  Shape:         " << formatShape(shape) << "\n";
    oss << "  Dtype:         " << dtype << "\n";
    oss << "  Tolerance:     atol=" << result.usedAtol << " rtol=" << result.usedRtol << "\n";
    oss << "  Max abs error: " << result.maxAbsError << "\n";
    oss << "  Max rel error: " << result.maxRelError << "\n";

    if(result.worstFlatIndex >= 0)
    {
        auto multiIdx = flatIndexToMultiDim(result.worstFlatIndex, shape);
        oss << "  Worst element: flat=" << result.worstFlatIndex
            << " " << formatMultiDimIndex(multiIdx) << "\n";
        oss << "    expected:    " << result.worstExpected << "\n";
        oss << "    actual:      " << result.worstActual << "\n";
    }

    double pct = 0.0;
    if(result.totalElements > 0)
    {
        pct = 100.0 * static_cast<double>(result.mismatchCount)
              / static_cast<double>(result.totalElements);
    }
    oss << "  Mismatched:    " << result.mismatchCount << " / " << result.totalElements
        << " (" << std::setprecision(2) << std::fixed << pct << "%)\n";
    oss << "=================================\n";
    return oss.str();
}

} // namespace hipdnn_integration_tests::golden
