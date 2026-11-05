// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace sdpa_case_flow
{

// A ragged view of a dense output buffer, read in place on the host. Positions past each
// sequence's live tokens are out of block, so a comparison skips them as the bundle harness does.
// `lengths` is the seq_len tensor of the output's side, or null when the graph has none.
inline std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    raggedView(hipdnn_data_sdk::utilities::ITensor& dense,
               hipdnn_flatbuffers_sdk::data_objects::DataType type,
               const std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>& offsets,
               int64_t offsetMultiplier,
               const std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>& lengths)
{
    using hipdnn_data_sdk::utilities::ITensor;
    void* data = dense.rawHostData();
    return hipdnn_test_sdk::detail::visitNativeType(
        type, [&](auto tag) -> std::unique_ptr<ITensor> {
            using T = typename decltype(tag)::type;
            return std::make_unique<hipdnn_data_sdk::utilities::ShallowRaggedTensor<T>>(
                data,
                dense.dims(),
                dense.strides(),
                hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS,
                offsets,
                std::nullopt,
                offsetMultiplier,
                lengths);
        });
}

} // namespace sdpa_case_flow
