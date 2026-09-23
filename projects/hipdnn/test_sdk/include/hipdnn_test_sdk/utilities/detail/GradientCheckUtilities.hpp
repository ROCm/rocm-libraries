// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace hipdnn_test_sdk::detail
{

struct GradCheckResult
{
    double maxRelErr;
    double maxAbsErr;
    int64_t failCount;
};

/// Convert a flat linear index to N-dimensional indices given tensor dimensions.
/// Uses row-major (C-order) stride decomposition.
inline std::vector<int64_t> flatToNdIndex(int64_t flatIdx, const std::vector<int64_t>& dims)
{
    const auto rank = dims.size();
    std::vector<int64_t> indices(rank);
    for(auto i = static_cast<int64_t>(rank) - 1; i >= 0; --i)
    {
        indices[static_cast<size_t>(i)] = flatIdx % dims[static_cast<size_t>(i)];
        flatIdx /= dims[static_cast<size_t>(i)];
    }
    return indices;
}

/// Compute total number of elements from a dimensions vector.
inline int64_t totalElements(const std::vector<int64_t>& dims)
{
    int64_t total = 1;
    for(const auto d : dims)
    {
        total *= d;
    }
    return total;
}

/// Compute the scalar dot-product loss L = sum(a[i] * b[i]) for any rank tensor.
/// Both tensors must have the same dimensions.
template <typename T>
double computeDotProductLoss(const hipdnn_data_sdk::utilities::TensorBase<T>& a,
                             const hipdnn_data_sdk::utilities::TensorBase<T>& b)
{
    if(a.dims() != b.dims())
    {
        throw std::invalid_argument("computeDotProductLoss: tensor dimensions must match");
    }

    double loss = 0.0;
    const auto numElements = totalElements(a.dims());
    for(int64_t i = 0; i < numElements; ++i)
    {
        const auto indices = flatToNdIndex(i, a.dims());
        loss += safeConvert<double>(a.getHostValue(indices))
                * safeConvert<double>(b.getHostValue(indices));
    }
    return loss;
}

/// Compare analytical and numerical gradient tensors element-wise.
/// Returns the maximum relative error, maximum absolute error, and the number
/// of elements that exceed both tolerance thresholds simultaneously.
template <typename T>
GradCheckResult compareGradients(const hipdnn_data_sdk::utilities::TensorBase<T>& analytical,
                                 const hipdnn_data_sdk::utilities::TensorBase<T>& numerical,
                                 double relTol,
                                 double absTol)
{
    if(analytical.dims() != numerical.dims())
    {
        throw std::invalid_argument("compareGradients: tensor dimensions must match");
    }

    GradCheckResult result{0.0, 0.0, 0};
    const auto numElements = totalElements(analytical.dims());
    for(int64_t i = 0; i < numElements; ++i)
    {
        const auto indices = flatToNdIndex(i, analytical.dims());
        const auto a = safeConvert<double>(analytical.getHostValue(indices));
        const auto n = safeConvert<double>(numerical.getHostValue(indices));
        const double absErr = std::abs(a - n);
        const double denom = std::max({std::abs(a), std::abs(n), 1e-8});
        const double relErr = absErr / denom;

        result.maxAbsErr = std::max(result.maxAbsErr, absErr);
        result.maxRelErr = std::max(result.maxRelErr, relErr);

        if(relErr > relTol && absErr > absTol)
        {
            ++result.failCount;
        }
    }
    return result;
}

/// Compute numerical gradients via central finite differences.
///
/// For each element of @p perturbInput, perturbs by +/-eps, calls
/// @p computeForwardLoss() to get the scalar loss, and stores
/// (L+ - L-) / (2*eps) into @p numericalGrad.
///
/// @p computeForwardLoss must be a callable with signature () -> double that
/// runs the forward pass using the current state of @p perturbInput and returns
/// a scalar loss value.
///
/// @note Execution is sequential because each iteration mutates @p perturbInput.
template <typename T, typename ForwardLossFunc>
void numericalGradient(hipdnn_data_sdk::utilities::TensorBase<T>& perturbInput,
                       hipdnn_data_sdk::utilities::TensorBase<T>& numericalGrad,
                       double eps,
                       ForwardLossFunc&& computeForwardLoss)
{
    if(perturbInput.dims() != numericalGrad.dims())
    {
        throw std::invalid_argument("numericalGradient: tensor dimensions must match");
    }

    const auto numElements = totalElements(perturbInput.dims());
    for(int64_t i = 0; i < numElements; ++i)
    {
        const auto indices = flatToNdIndex(i, perturbInput.dims());
        const auto orig = safeConvert<double>(perturbInput.getHostValue(indices));

        // L+
        perturbInput.setHostValue(safeConvert<T>(orig + eps), indices);
        perturbInput.memory().markHostModified();
        const double lPlus = computeForwardLoss();

        // L-
        perturbInput.setHostValue(safeConvert<T>(orig - eps), indices);
        perturbInput.memory().markHostModified();
        const double lMinus = computeForwardLoss();

        // Restore
        perturbInput.setHostValue(safeConvert<T>(orig), indices);
        perturbInput.memory().markHostModified();

        numericalGrad.setHostValue(safeConvert<T>((lPlus - lMinus) / (2.0 * eps)), indices);
    }
    numericalGrad.memory().markHostModified();
}

} // namespace hipdnn_test_sdk::detail
