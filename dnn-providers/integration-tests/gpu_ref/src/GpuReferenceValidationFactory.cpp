// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_gpu_ref/GpuReferenceValidationFactory.hpp>

#include <cstdint>
#include <stdexcept>

namespace hipdnn_gpu_ref
{

std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation> createGpuAllCloseValidator(
    hipdnn_frontend::DataType dataType, float absoluteTolerance, float relativeTolerance)
{
    switch(dataType)
    {
    case hipdnn_frontend::DataType::FLOAT:
        return std::make_unique<GpuFpReferenceValidation<float>>(absoluteTolerance,
                                                                 relativeTolerance);
    case hipdnn_frontend::DataType::HALF:
        return std::make_unique<GpuFpReferenceValidation<hipdnn_data_sdk::types::half>>(
            absoluteTolerance, relativeTolerance);
    case hipdnn_frontend::DataType::BFLOAT16:
        return std::make_unique<GpuFpReferenceValidation<hipdnn_data_sdk::types::bfloat16>>(
            absoluteTolerance, relativeTolerance);
    case hipdnn_frontend::DataType::DOUBLE:
        return std::make_unique<GpuFpReferenceValidation<double>>(absoluteTolerance,
                                                                  relativeTolerance);
    case hipdnn_frontend::DataType::INT8:
        return std::make_unique<GpuIntReferenceValidation<int8_t>>();
    case hipdnn_frontend::DataType::UINT8:
        return std::make_unique<GpuIntReferenceValidation<uint8_t>>();
    case hipdnn_frontend::DataType::INT32:
        return std::make_unique<GpuIntReferenceValidation<int32_t>>();
    default:
        throw std::runtime_error("Unsupported data type for GPU allClose validator");
    }
}

} // namespace hipdnn_gpu_ref
