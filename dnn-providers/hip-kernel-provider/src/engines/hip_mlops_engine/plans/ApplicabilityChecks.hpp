// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider
{

// --- Tensor Descriptor Value Object ---

struct TensorDescriptor
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> strideOrder;

    explicit TensorDescriptor(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attr);

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
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
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
        hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
        const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>& allowedTypes,
        const std::string& errorMessage);

    virtual void validateConsistentDataTypes(
        const std::vector<int64_t>& tensorIds,
        const std::unordered_set<hipdnn_flatbuffers_sdk::data_objects::DataType>& allowedTypes,
        const std::string& typeErrorMessage,
        const std::string& consistencyErrorMessage);

    virtual void validateFixedDataType(const std::vector<int64_t>& tensorIds,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType expectedType,
                                       const std::string& errorMessage);

    virtual void validateConsistentShapes(const std::vector<int64_t>& tensorIds,
                                          const std::vector<int64_t>& referenceShape,
                                          const std::string& errorMessage);

    // Kernel buffers are device tensors, so a virtual tensor in a buffer role has no storage.
    virtual void validateNoVirtualTensors(const std::vector<int64_t>& tensorIds,
                                          const std::string& errorMessage);

    // The kernels index and size their problem with 32-bit unsigned compile-time constants and
    // arithmetic (element count, per-axis extents, grid), so the element count of `dims` must stay
    // below UINT32_MAX. The product is checked in 64 bits.
    virtual void validateElementCountFitsUint32(const std::vector<int64_t>& dims,
                                                const std::string& errorMessage);

    // Plans resolve epsilon as a scalar operand when they are built; an epsilon uid the graph does
    // not carry, or a tensor that holds no readable scalar, must decline here instead. Throws
    // BAD_PARAM, which the builder logs and turns into a decline.
    void checkEpsilonSupported(int64_t epsilonTensorId) const;

    // --- Component Validators ---

    virtual void checkTensorLayoutsAndDimsSupported(const std::vector<int64_t>& tensorIds) = 0;

    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        _tensorMap;
};

} // namespace hip_kernel_provider
