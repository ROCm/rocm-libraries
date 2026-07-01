// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Shared, op-agnostic execute-path test kit (consolidation R1).
//
// Every GPU execute-path frontend integration test reinvents the same three
// pieces: (1) a tensor bundle that owns host/device buffers and produces a
// UID-keyed variant pack, (2) the validate -> ... -> execute compile pipeline,
// and (3) a graph builder. This kit centralizes all three so a new op needs no
// per-op tensor wiring.
//
// The bundle is OP-AGNOSTIC: given a BUILT graph (UIDs assigned), it walks the
// graph's node tree via the public INode::visit() API, collects each node's
// input/output TensorAttributes, allocates one host Tensor<float> per physical
// tensor using that tensor's own dims/strides, and keys the variant pack by
// get_uid(). No op knows or cares which tensors it has — adding matmul/sdpa/
// norms reuses this verbatim.
//
// Graph structure is NOT hand-rolled here: builders delegate to the existing
// FrontendGraphFactory, which already constructs 15 op graphs.
//
// Header-only because the helpers are small and consumed by a handful of
// integration translation units.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

#include <hipdnn_test_sdk/utilities/FrontendGraphFactory.hpp>

namespace hipdnn_test_sdk::utilities
{

/// Owns one host Tensor<float> per PHYSICAL tensor of a built graph and exposes
/// a UID-keyed variant pack. Op-agnostic: the set of tensors is discovered from
/// the graph itself, not declared per op.
///
/// Construct AFTER build_operation_graph()/build() has assigned UIDs — the
/// enumeration relies on UID-bearing tensors and keys the variant pack by UID.
class GraphTensorBundle
{
public:
    /// Walks the graph's node tree, collects each node's physical (non-virtual,
    /// UID-bearing) input/output tensors, allocates a host Tensor<float> per
    /// unique UID from that tensor's own dims/strides, and fills it with
    /// @p fillValue. Virtual (intermediate) tensors are skipped: the backend
    /// materializes them, so they must not appear in the variant pack. A tensor
    /// shared across nodes (same UID) gets exactly one buffer.
    explicit GraphTensorBundle(hipdnn_frontend::graph::Graph& graph, float fillValue = 1.0f)
    {
        const auto addTensor
            = [&](const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& tensorAttr) {
                  if(!tensorAttr || tensorAttr->get_is_virtual() || !tensorAttr->has_uid())
                  {
                      return;
                  }
                  if(_variantPack.count(tensorAttr->get_uid()) != 0)
                  {
                      return;
                  }

                  auto tensor = std::make_unique<hipdnn_data_sdk::utilities::Tensor<float>>(
                      tensorAttr->get_dim(), tensorAttr->get_stride());
                  tensor->fillWithValue(fillValue);
                  _variantPack[tensorAttr->get_uid()] = tensor->memory().deviceData();
                  _tensors.push_back(std::move(tensor));
              };

        graph.visit([&](hipdnn_frontend::graph::INode& node) {
            for(const auto& t : node.getNodeInputTensorAttributes())
            {
                addTensor(t);
            }
            for(const auto& t : node.getNodeOutputTensorAttributes())
            {
                addTensor(t);
            }
        });
    }

    /// Maps tensor UID -> device pointer for graph->execute()/autotune().
    const std::unordered_map<int64_t, void*>& variantPack() const
    {
        return _variantPack;
    }

private:
    // Tensors are heap-owned so their device pointers stay stable as the vector
    // grows and for the bundle's lifetime (Tensor is move-only).
    std::vector<std::unique_ptr<hipdnn_data_sdk::utilities::Tensor<float>>> _tensors;
    std::unordered_map<int64_t, void*> _variantPack;
};

/// Builds a graph for @p op by delegating to FrontendGraphFactory (no graph
/// structure is hand-rolled in the kit). Returned by shared_ptr so it can flow
/// through the frontend pipeline APIs that bind a shared_ptr lifetime.
inline std::shared_ptr<hipdnn_frontend::graph::Graph> buildGraphForOp(OperationType op)
{
    return std::make_shared<hipdnn_frontend::graph::Graph>(FrontendGraphFactory::create(op));
}

/// Runs the canonical compile+execute pipeline against an already-operation-built
/// @p graph, asserting OK at each step: create_execution_plans -> check_support
/// -> build_plans -> workspace -> execute. The graph must already have been
/// through validate() + build_operation_graph() so its UIDs are assigned and
/// match @p variantPack (use a GraphTensorBundle built from this graph). gtest
/// ASSERT_* requires a void return.
inline void runGraphPipeline(hipdnn_frontend::graph::Graph& graph,
                             hipdnnHandle_t handle,
                             const std::unordered_map<int64_t, void*>& variantPack)
{
    using hipdnn_frontend::ErrorCode;

    auto result = graph.create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph.check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph.build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t workspaceSize = 0;
    result = graph.get_workspace_size(workspaceSize);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_GE(workspaceSize, 0);
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    // execute() binds the variant pack by non-const reference; copy so the kit's
    // API can take it by const reference.
    std::unordered_map<int64_t, void*> pack = variantPack;
    result = graph.execute(handle, pack, workspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

} // namespace hipdnn_test_sdk::utilities
