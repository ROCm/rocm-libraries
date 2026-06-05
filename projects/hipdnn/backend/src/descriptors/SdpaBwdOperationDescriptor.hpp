// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>

namespace hipdnn_backend
{

class SdpaBwdOperationDescriptor : public HipdnnBackendDescriptorImpl<SdpaBwdOperationDescriptor>,
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
    const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributesT& getData() const
    {
        return _data;
    }

    // Access to tensor descriptor references for graph building
    std::shared_ptr<TensorDescriptor> getQDesc() const
    {
        return _qDesc;
    }
    std::shared_ptr<TensorDescriptor> getKDesc() const
    {
        return _kDesc;
    }
    std::shared_ptr<TensorDescriptor> getVDesc() const
    {
        return _vDesc;
    }
    std::shared_ptr<TensorDescriptor> getODesc() const
    {
        return _oDesc;
    }
    std::shared_ptr<TensorDescriptor> getDoDesc() const
    {
        return _doDesc;
    }
    std::shared_ptr<TensorDescriptor> getStatsDesc() const
    {
        return _statsDesc;
    }
    std::shared_ptr<TensorDescriptor> getDqDesc() const
    {
        return _dqDesc;
    }
    std::shared_ptr<TensorDescriptor> getDkDesc() const
    {
        return _dkDesc;
    }
    std::shared_ptr<TensorDescriptor> getDvDesc() const
    {
        return _dvDesc;
    }
    std::shared_ptr<TensorDescriptor> getScaleDesc() const
    {
        return _scaleDesc;
    }
    std::shared_ptr<TensorDescriptor> getAttnMaskDesc() const
    {
        return _attnMaskDesc;
    }
    std::shared_ptr<TensorDescriptor> getSeqLenQDesc() const
    {
        return _seqLenQDesc;
    }
    std::shared_ptr<TensorDescriptor> getSeqLenKvDesc() const
    {
        return _seqLenKvDesc;
    }
    std::shared_ptr<TensorDescriptor> getSeedDesc() const
    {
        return _seedDesc;
    }
    std::shared_ptr<TensorDescriptor> getOffsetDesc() const
    {
        return _offsetDesc;
    }
    std::shared_ptr<TensorDescriptor> getDropoutMaskDesc() const
    {
        return _dropoutMaskDesc;
    }
    std::shared_ptr<TensorDescriptor> getDropoutScaleDesc() const
    {
        return _dropoutScaleDesc;
    }
    std::shared_ptr<TensorDescriptor> getDropoutScaleInvDesc() const
    {
        return _dropoutScaleInvDesc;
    }
    std::shared_ptr<TensorDescriptor> getDBiasDesc() const
    {
        return _dbiasDesc;
    }

    // Get compute data type for the operation (used when building graph nodes)
    hipdnn_flatbuffers_sdk::data_objects::DataType getComputeDataType() const
    {
        return _computeDataType;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const override;

    // Creates a finalized SdpaBwdOperationDescriptor directly from a FlatBuffer NodeT.
    // Casts nodeT.attributes to SdpaBackwardAttributesT internally, then directly assigns
    // the data struct, looks up tensor descriptors from the tensor map, and calls finalize().
    static std::shared_ptr<SdpaBwdOperationDescriptor>
        fromNode(const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
                 const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributesT _data;

    // Required input tensor descriptor references
    std::shared_ptr<TensorDescriptor> _qDesc;
    std::shared_ptr<TensorDescriptor> _kDesc;
    std::shared_ptr<TensorDescriptor> _vDesc;
    std::shared_ptr<TensorDescriptor> _oDesc;
    std::shared_ptr<TensorDescriptor> _doDesc;
    std::shared_ptr<TensorDescriptor> _statsDesc;

    // Required output tensor descriptor references
    std::shared_ptr<TensorDescriptor> _dqDesc;
    std::shared_ptr<TensorDescriptor> _dkDesc;
    std::shared_ptr<TensorDescriptor> _dvDesc;

    // Optional tensor descriptor references
    std::shared_ptr<TensorDescriptor> _scaleDesc;
    std::shared_ptr<TensorDescriptor> _attnMaskDesc;
    std::shared_ptr<TensorDescriptor> _seqLenQDesc;
    std::shared_ptr<TensorDescriptor> _seqLenKvDesc;
    std::shared_ptr<TensorDescriptor> _seedDesc;
    std::shared_ptr<TensorDescriptor> _offsetDesc;
    std::shared_ptr<TensorDescriptor> _dropoutMaskDesc;
    std::shared_ptr<TensorDescriptor> _dropoutScaleDesc;
    std::shared_ptr<TensorDescriptor> _dropoutScaleInvDesc;
    std::shared_ptr<TensorDescriptor> _dbiasDesc;

    // Compute data type for this operation (stored at node level in graph)
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;

    // Optional human-readable name for this operation
    std::string _name;
};

} // namespace hipdnn_backend
