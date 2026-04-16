// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider
{

// --- Tensor Descriptor Value Object ---

struct TensorDescriptor
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> strideOrder;

    explicit TensorDescriptor(const hipdnn_data_sdk::data_objects::TensorAttributes* attr);

    size_t numDims() const
    {
        return dims.size();
    }
    bool isPacked() const;
};

// --- Abstract Base Class For Plan Validation ---

class IValidator
{
public:
    IValidator(
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap)
        : _tensorMap(tensorMap) {};

    virtual ~IValidator() = default;

protected:
    // --- Validation Utilities ---

    virtual void validateDimensionCount(size_t numDims);

    virtual void validateConsistentDimensions(const std::vector<TensorDescriptor>& tensors);

    virtual void validatePackedTensors(const std::vector<TensorDescriptor>& tensors);

    virtual void validateSupportedLayout(const std::vector<int64_t>& strideOrder, size_t numDims);

    virtual void validateConsistentLayouts(const std::vector<TensorDescriptor>& tensors);

    virtual void validateDataTypeIsSupported(
        hipdnn_data_sdk::data_objects::DataType dataType,
        const std::unordered_set<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
        const std::string& errorMessage);

    virtual void validateConsistentDataTypes(
        const std::vector<int64_t>& tensorIds,
        const std::unordered_set<hipdnn_data_sdk::data_objects::DataType>& allowedTypes,
        const std::string& typeErrorMessage,
        const std::string& consistencyErrorMessage);

    virtual void validateFixedDataType(const std::vector<int64_t>& tensorIds,
                                       hipdnn_data_sdk::data_objects::DataType expectedType,
                                       const std::string& errorMessage);

    virtual void validateConsistentShapes(const std::vector<int64_t>& tensorIds,
                                          const std::vector<int64_t>& referenceShape,
                                          const std::string& errorMessage);

    // --- Component Validators ---

    virtual void checkTensorLayoutsAndDimsSupported() = 0;

    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        _tensorMap;
};

} // namespace hip_kernel_provider
