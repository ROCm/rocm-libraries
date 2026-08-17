// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "TensorDescriptor.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <memory>
#include <vector>

namespace hipdnn_backend
{

// Interface for operation descriptors that can be composed into an operation graph.
// Implemented by concrete operation descriptors (e.g., ConvolutionFwdOperationDescriptor)
// and consumed by GraphDescriptor for type-agnostic graph construction.
class IGraphOperation
{
public:
    virtual ~IGraphOperation() = default;

    // Returns all tensor descriptors referenced by this operation.
    // Used by GraphDescriptor for tensor deduplication when building the graph.
    virtual std::vector<std::shared_ptr<TensorDescriptor>> getTensorDescriptors() const = 0;

    // Builds a NodeT representing this operation in the graph.
    // Sets the correct flatbuffer union type for the node's attributes.
    // Implementations must set node->compute_data_type from their stored compute type.
    virtual std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> buildNode() const = 0;
};

} // namespace hipdnn_backend
