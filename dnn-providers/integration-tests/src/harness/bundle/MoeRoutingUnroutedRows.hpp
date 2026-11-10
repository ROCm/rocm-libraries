// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "harness/bundle/OutputComparison.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace detail
{

/// first_token_offset[E]: the number of routed rows, read from an INT32 or INT64 offsets tensor.
inline int64_t moeRoutedRowCount(hipdnn_data_sdk::utilities::ITensor& offsets,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                 int64_t numExperts)
{
    if(dataType == hipdnn_flatbuffers_sdk::data_objects::DataType::INT64)
    {
        return hipdnn_data_sdk::utilities::TensorView<int64_t>(offsets).getHostValue(
            {numExperts, 0, 0});
    }
    return hipdnn_data_sdk::utilities::TensorView<int32_t>(offsets).getHostValue(
        {numExperts, 0, 0});
}

} // namespace detail

/// MoE routing leaves the token_index / token_ks rows past first_token_offset[E] untouched
/// (moe_routing_attributes.fbs): with expert parallelism some ids are out of range, so fewer
/// than S * top_k rows are routed. Those rows still hold the output sentinel on both sides, and
/// the integer validator rejects any sentinel as an unwritten element.
///
/// For each MoE routing node, this checks that the engine left every unrouted row exactly as
/// the expected side has it (an engine that writes past the routed rows fails here, by row),
/// then zeroes those rows on both sides so the regular comparison grades only the routed rows
/// and the offsets. The routed count is the expected side's first_token_offset[E]; a wrong
/// engine count still fails that comparison. Returns one line per unrouted row the engine
/// wrote, empty when there is none.
inline std::string
    settleMoeRoutingUnroutedRows(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                                 OutputTensors& engineOutputs,
                                 const ExpectedTensorLookup& expectedFor)
{
    const auto& tensorMap = graph.getTensorMap();
    std::ostringstream problems;

    for(uint32_t i = 0; i < graph.nodeCount(); ++i)
    {
        const auto* routing = graph.getNode(i).attributes_as_MoeRoutingAttributes();
        if(routing == nullptr)
        {
            continue;
        }

        const auto offsetsUid = routing->first_token_offset_tensor_uid();
        const auto offsetsAttrs = tensorMap.find(offsetsUid);
        if(offsetsAttrs == tensorMap.end())
        {
            continue;
        }
        const int64_t routed = detail::moeRoutedRowCount(
            expectedFor(offsetsUid), offsetsAttrs->second->data_type(), routing->num_experts());

        for(const int64_t rowsUid :
            {routing->token_index_tensor_uid(), routing->token_ks_tensor_uid()})
        {
            const auto engineIt = engineOutputs.find(rowsUid);
            const auto rowsAttrs = tensorMap.find(rowsUid);
            if(engineIt == engineOutputs.end() || rowsAttrs == tensorMap.end())
            {
                continue;
            }

            auto& engineRows = *engineIt->second;
            auto& expectedRows = expectedFor(rowsUid);
            const int64_t capacity = engineRows.dims().at(1);
            const int64_t firstUnrouted = std::max<int64_t>(routed, 0);
            // Every row routed: nothing to settle. Marking the host side modified here
            // would also discard device results the host copy never received.
            if(firstUnrouted >= capacity)
            {
                continue;
            }
            hipdnn_data_sdk::utilities::TensorView<int32_t> engineView(engineRows);
            hipdnn_data_sdk::utilities::TensorView<int32_t> expectedView(expectedRows);

            // Reading through the views first brings each side's latest values to the host,
            // so the host copy is complete before it is marked as the current one.
            for(int64_t row = firstUnrouted; row < capacity; ++row)
            {
                auto& engineValue = engineView.getHostValue({0, row, 0});
                auto& expectedValue = expectedView.getHostValue({0, row, 0});
                if(engineValue != expectedValue)
                {
                    problems << "  " << tensorLabel(rowsUid, *rowsAttrs->second) << " row " << row
                             << " (unrouted, past first_token_offset[E] = " << routed
                             << "): engine wrote " << engineValue << ", expected untouched "
                             << expectedValue << "\n";
                }
                engineValue = 0;
                expectedValue = 0;
            }
            engineRows.markHostModified();
            expectedRows.markHostModified();
        }
    }

    const auto report = problems.str();
    return report.empty() ? std::string{}
                          : "MoE routing rows past first_token_offset[E] must be left "
                            "untouched:\n"
                                + report;
}

} // namespace hipdnn_integration_tests::bundle
