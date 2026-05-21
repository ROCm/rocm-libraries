// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/custom_op_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_backend
{

class CustomOpOperationDescriptor : public HipdnnBackendDescriptorImpl<CustomOpOperationDescriptor>,
                                    public IGraphOperation
{
public:
    void finalize() override;

    void getAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const override;

    void setAttribute(hipdnnBackendAttributeName_t attributeName,
                      hipdnnBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      const void* arrayOfElements) override;

    const hipdnn_flatbuffers_sdk::data_objects::CustomOpAttributesT& getData() const
    {
        return _data;
    }

    const std::vector<std::shared_ptr<TensorDescriptor>>& getInputDescs() const
    {
        return _inputDescs;
    }

    const std::vector<std::shared_ptr<TensorDescriptor>>& getOutputDescs() const
    {
        return _outputDescs;
    }

    hipdnn_flatbuffers_sdk::data_objects::DataType getComputeDataType() const
    {
        return _computeDataType;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const override;

    // Creates a finalized CustomOpOperationDescriptor directly from a FlatBuffer NodeT.
    // Casts nodeT.attributes to CustomOpAttributesT internally, then directly assigns
    // the data struct, looks up tensor descriptors from the tensor map, and calls finalize().
    static std::shared_ptr<CustomOpOperationDescriptor>
        fromNode(const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
                 const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnn_flatbuffers_sdk::data_objects::CustomOpAttributesT _data;

    std::vector<std::shared_ptr<TensorDescriptor>> _inputDescs;
    std::vector<std::shared_ptr<TensorDescriptor>> _outputDescs;

    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;

    std::string _name;
};

} // namespace hipdnn_backend
