// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;

inline flatbuffers::FlatBufferBuilder createMatmulGraph(const int64_t aUid,
                                                        const int64_t bUid,
                                                        const int64_t cUid,
                                                        const std::vector<int64_t>& aDims,
                                                        const std::vector<int64_t>& aStrides,
                                                        const std::vector<int64_t>& bDims,
                                                        const std::vector<int64_t>& bStrides,
                                                        const std::vector<int64_t>& cDims,
                                                        const std::vector<int64_t>& cStrides,
                                                        const DataType aDataType,
                                                        const DataType bDataType,
                                                        const DataType cDataType,
                                                        const DataType computeDataType)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, aUid, "a", aDataType, &aStrides, &aDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, bUid, "b", bDataType, &bStrides, &bDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, cUid, "c", cDataType, &cStrides, &cDims));

    auto matmulAttrs = CreateMatmulAttributes(builder, aUid, bUid, cUid);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "matmul_node",
                                     computeDataType,
                                     NodeAttributes::MatmulAttributes,
                                     matmulAttrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "MatmulTestGraph", computeDataType, computeDataType, aDataType, &tensors, &nodes);

    builder.Finish(graph);
    return builder;
}

} // namespace hipdnn_integration_tests::test_utils
