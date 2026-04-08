// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "BackendDescriptor.hpp"
#include "IGraphOperation.hpp"
#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <unordered_map>

namespace hipdnn_backend
{

class SdpaFwdOperationDescriptor : public HipdnnBackendDescriptorImpl<SdpaFwdOperationDescriptor>,
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
    const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT& getData() const
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
    std::shared_ptr<TensorDescriptor> getAttnMaskDesc() const
    {
        return _attnMaskDesc;
    }
    std::shared_ptr<TensorDescriptor> getScaleDesc() const
    {
        return _scaleDesc;
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
    std::shared_ptr<TensorDescriptor> getPageTableKDesc() const
    {
        return _pageTableKDesc;
    }
    std::shared_ptr<TensorDescriptor> getPageTableVDesc() const
    {
        return _pageTableVDesc;
    }
    std::shared_ptr<TensorDescriptor> getBlockMaskDesc() const
    {
        return _blockMaskDesc;
    }
    std::shared_ptr<TensorDescriptor> getSinkTokenDesc() const
    {
        return _sinkTokenDesc;
    }
    std::shared_ptr<TensorDescriptor> getDescaleQDesc() const
    {
        return _descaleQDesc;
    }
    std::shared_ptr<TensorDescriptor> getDescaleKDesc() const
    {
        return _descaleKDesc;
    }
    std::shared_ptr<TensorDescriptor> getDescaleVDesc() const
    {
        return _descaleVDesc;
    }
    std::shared_ptr<TensorDescriptor> getDescaleSDesc() const
    {
        return _descaleSDesc;
    }
    std::shared_ptr<TensorDescriptor> getScaleSDesc() const
    {
        return _scaleSDesc;
    }
    std::shared_ptr<TensorDescriptor> getScaleODesc() const
    {
        return _scaleODesc;
    }
    std::shared_ptr<TensorDescriptor> getStatsDesc() const
    {
        return _statsDesc;
    }
    std::shared_ptr<TensorDescriptor> getMaxDesc() const
    {
        return _maxDesc;
    }
    std::shared_ptr<TensorDescriptor> getSumExpDesc() const
    {
        return _sumExpDesc;
    }
    std::shared_ptr<TensorDescriptor> getRngDumpDesc() const
    {
        return _rngDumpDesc;
    }
    std::shared_ptr<TensorDescriptor> getAmaxSDesc() const
    {
        return _amaxSDesc;
    }
    std::shared_ptr<TensorDescriptor> getAmaxODesc() const
    {
        return _amaxODesc;
    }

    // Get compute data type for the operation (used when building graph nodes)
    hipdnn_flatbuffers_sdk::data_objects::DataType getComputeDataType() const
    {
        return _computeDataType;
    }

    // IGraphOperation interface
    std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const override;
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const override;

    // Creates a finalized SdpaFwdOperationDescriptor directly from a FlatBuffer NodeT.
    // Casts nodeT.attributes to SdpaAttributesT internally, then directly assigns
    // the data struct, looks up tensor descriptors from the tensor map, and calls finalize().
    static std::shared_ptr<SdpaFwdOperationDescriptor>
        fromNode(const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
                 const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap);

    static hipdnnBackendDescriptorType_t getStaticType();

    std::string toString() const override;

private:
    hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT _data;

    // Optional human-readable name for this operation
    std::string _name;

    // Store tensor descriptor references for validation and graph building
    std::shared_ptr<TensorDescriptor> _qDesc;
    std::shared_ptr<TensorDescriptor> _kDesc;
    std::shared_ptr<TensorDescriptor> _vDesc;
    std::shared_ptr<TensorDescriptor> _oDesc;
    std::shared_ptr<TensorDescriptor> _attnMaskDesc;
    std::shared_ptr<TensorDescriptor> _scaleDesc;
    std::shared_ptr<TensorDescriptor> _seqLenQDesc;
    std::shared_ptr<TensorDescriptor> _seqLenKvDesc;
    std::shared_ptr<TensorDescriptor> _seedDesc;
    std::shared_ptr<TensorDescriptor> _offsetDesc;
    std::shared_ptr<TensorDescriptor> _dropoutMaskDesc;
    std::shared_ptr<TensorDescriptor> _dropoutScaleDesc;
    std::shared_ptr<TensorDescriptor> _pageTableKDesc;
    std::shared_ptr<TensorDescriptor> _pageTableVDesc;
    std::shared_ptr<TensorDescriptor> _blockMaskDesc;
    std::shared_ptr<TensorDescriptor> _sinkTokenDesc;
    std::shared_ptr<TensorDescriptor> _descaleQDesc;
    std::shared_ptr<TensorDescriptor> _descaleKDesc;
    std::shared_ptr<TensorDescriptor> _descaleVDesc;
    std::shared_ptr<TensorDescriptor> _descaleSDesc;
    std::shared_ptr<TensorDescriptor> _scaleSDesc;
    std::shared_ptr<TensorDescriptor> _scaleODesc;
    std::shared_ptr<TensorDescriptor> _statsDesc;
    std::shared_ptr<TensorDescriptor> _maxDesc;
    std::shared_ptr<TensorDescriptor> _sumExpDesc;
    std::shared_ptr<TensorDescriptor> _rngDumpDesc;
    std::shared_ptr<TensorDescriptor> _amaxSDesc;
    std::shared_ptr<TensorDescriptor> _amaxODesc;

    // Compute data type for this operation (stored at node level in graph)
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
};

} // namespace hipdnn_backend
