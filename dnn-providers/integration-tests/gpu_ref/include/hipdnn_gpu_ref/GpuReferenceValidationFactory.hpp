// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_gpu_ref/GpuFpReferenceValidation.hpp>
#include <hipdnn_gpu_ref/GpuIntReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>

#include <limits>
#include <memory>
#include <type_traits>

namespace hipdnn_gpu_ref
{

// Factory function to create a GPU allClose validator for the given data type.
// Mirrors the createAllCloseValidator() API from CpuFpReferenceValidation.hpp.
std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createGpuAllCloseValidator(hipdnn_frontend::DataType dataType,
                               float absoluteTolerance = std::numeric_limits<float>::epsilon(),
                               float relativeTolerance = std::numeric_limits<float>::epsilon());

// Templated factory function to create a GPU allClose validator.
template <typename T>
inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    // NOLINTNEXTLINE(readability-redundant-casting) - cast needed for non-float T types
    createGpuAllCloseValidator(float absoluteTolerance = float(std::numeric_limits<T>::epsilon()),
                               // NOLINTNEXTLINE(readability-redundant-casting)
                               float relativeTolerance = float(std::numeric_limits<T>::epsilon()))
{
    if constexpr(std::is_integral_v<T>)
    {
        return std::make_unique<GpuIntReferenceValidation<T>>();
    }
    else
    {
        return std::make_unique<GpuFpReferenceValidation<T>>(absoluteTolerance, relativeTolerance);
    }
}

} // namespace hipdnn_gpu_ref
