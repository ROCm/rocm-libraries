// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/input-init/FillInputs.hpp"
#include "harness/input-init/PhiloxDropout.hpp"

namespace hipdnn_integration_tests
{

/// What an SDPA backward node's Philox dropout binds (SdpaSemanticsRegister.md SEM-015): a
/// probability in (0, 1), a seed and an offset, and no explicit dropout mask. No reference
/// reproduces an engine's random stream from these, but the mask bits are pinned (see
/// PhiloxDropout.hpp), so the harness draws the mask itself and grades the engine against the
/// explicit-mask gradient. The uids below are the ones the explicit-mask form of the graph
/// gives its mask, scale and inverse scale; they are fresh uids of the rewritten graph.
struct PhiloxBackwardDropout
{
    float probability = 0.0F;
    int64_t seedUid = 0;
    int64_t offsetUid = 0;
    int64_t maskUid = 0;
    int64_t scaleUid = 0;
    int64_t scaleInvUid = 0;
    philox::MaskExtent extent; // [B, Hq, Sq, Skv]: the declared extents of Q and K
};

/// The Philox dropout of the graph's SDPA backward node, if it has one the harness can model.
/// Returns nullopt for a graph without Philox dropout (none, probability 0, or an explicit
/// mask). When the graph has Philox dropout the harness cannot model, returns nullopt and sets
/// @p unsupported to why (a ragged operand, a Q or K that is not rank 4).
std::optional<PhiloxBackwardDropout>
    findPhiloxBackwardDropout(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                              std::string& unsupported);

/// Reads the seed and the offset: from the filled input tensor when the graph has one (a device
/// tensor, or a runtime pass-by-value scalar), else from the value baked into the graph. INT32
/// values are sign-extended to 64 bits, as the engines read them. Returns why they cannot be
/// read, or nullopt after setting @p seed and @p offset.
std::optional<std::string>
    resolvePhiloxSeedAndOffset(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                               const PhiloxBackwardDropout& dropout,
                               const InputTensorMap& inputs,
                               uint64_t& seed,
                               uint64_t& offset);

/// The dense FLOAT keep mask [B, Hq, Sq, Skv] (1 kept, 0 dropped) the graph's Philox dropout draws.
std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    makePhiloxKeepMask(const PhiloxBackwardDropout& dropout, uint64_t seed, uint64_t offset);

/// The explicit-mask form of an SDPA backward graph with Philox dropout: the probability, seed
/// and offset replaced by a dropout mask, a dropout scale (1 / (1 - p)) and a dropout scale
/// inverse (1 - p) as cuDNN's custom dropout takes them, so that the gradient is the exact one
/// (SEM-015). The mask is a FLOAT [B, Hq, Sq, Skv] graph input the caller supplies
/// (makePhiloxKeepMask); the scales are baked into the graph. The seed and offset tensors
/// are dropped. Returns the flatbuffer of the rewritten graph.
std::vector<uint8_t>
    rewritePhiloxBackwardWithExplicitMask(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                                          const PhiloxBackwardDropout& dropout);

} // namespace hipdnn_integration_tests
