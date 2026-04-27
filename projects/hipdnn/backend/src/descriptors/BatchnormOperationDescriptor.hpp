// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <unordered_map>

namespace hipdnn_backend
{

class BatchnormOperationDescriptor
    : public HipdnnBackendDescriptorImpl<BatchnormOperationDescriptor>,
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

    // Direct access to the underlying T struct for OperationGraphBuilder
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributesT& getData() const
    {
        return _data;
    }

    // Access to tensor descriptor references for graph building
    std::shared_ptr<TensorDescriptor> getXDesc() const
    {
        return _xDesc;
    }
    std::shared_ptr<TensorDescriptor> getScaleDesc() const
    {
        return _scaleDesc;
    }
    std::shared_ptr<TensorDescriptor> getBiasDesc() const
    {
        return _biasDesc;
    }
    std::shared_ptr<TensorDescriptor> getEpsilonDesc() const
    {
        return _epsilonDesc;
    }
    std::shared_ptr<TensorDescriptor> getYDesc() const
    {
        return _yDesc;
    }
    std::shared_ptr<TensorDescriptor> getPrevRunningMeanDesc() const
    {
        return _prevRunningMeanDesc;
    }
    std::shared_ptr<TensorDescriptor> getPrevRunningVarianceDesc() const
    {
        return _prevRunningVarianceDesc;
    }
    std::shared_ptr<TensorDescriptor> getMomentumDesc() const
    {
        return _momentumDesc;
    }
    std::shared_ptr<TensorDescriptor> getMeanDesc() const
    {
        return _meanDesc;
    }
    std::shared_ptr<TensorDescriptor> getInvVarianceDesc() const
    {
        return _invVarianceDesc;
    }
    std::shared_ptr<TensorDescriptor> getNextRunningMeanDesc() const
    {
        return _nextRunningMeanDesc;
    }
    std::shared_ptr<TensorDescriptor> getNextRunningVarianceDesc() const
    {
        return _nextRunningVarianceDesc;
    }

    // Get compute data type for the operation (used when building graph nodes)
    hipdnn_flatbuffers_sdk::data_objects::DataType getComputeDataType() const
    {
        return _computeDataType;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const override;

    // Creates a finalized BatchnormOperationDescriptor directly from a FlatBuffer NodeT.
    // Casts nodeT.attributes to BatchnormAttributesT internally, then directly assigns
    // the data struct, looks up tensor descriptors from the tensor map, and calls finalize().
    static std::shared_ptr<BatchnormOperationDescriptor>
        fromNode(const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
                 const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributesT _data;

    // Required tensor descriptor references
    std::shared_ptr<TensorDescriptor> _xDesc;
    std::shared_ptr<TensorDescriptor> _scaleDesc;
    std::shared_ptr<TensorDescriptor> _biasDesc;
    std::shared_ptr<TensorDescriptor> _epsilonDesc;
    std::shared_ptr<TensorDescriptor> _yDesc;

    // Optional tensor descriptor references
    std::shared_ptr<TensorDescriptor> _prevRunningMeanDesc;
    std::shared_ptr<TensorDescriptor> _prevRunningVarianceDesc;
    std::shared_ptr<TensorDescriptor> _momentumDesc;
    std::shared_ptr<TensorDescriptor> _meanDesc;
    std::shared_ptr<TensorDescriptor> _invVarianceDesc;
    std::shared_ptr<TensorDescriptor> _nextRunningMeanDesc;
    std::shared_ptr<TensorDescriptor> _nextRunningVarianceDesc;

    // Tensor array: peer_stats
    std::vector<std::shared_ptr<TensorDescriptor>> _peerStatsDescs;

    // Compute data type for this operation (stored at node level in graph)
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;

    std::string _name;
};

} // namespace hipdnn_backend
