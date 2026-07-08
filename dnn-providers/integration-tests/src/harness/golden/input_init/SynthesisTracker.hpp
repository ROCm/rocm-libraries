// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::golden
{

// Pre-allocated input tensors keyed by uid, handed to a fill function to populate.
using InputTensorMap
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

// Result of a synthesis step — returned by per-node fill functions and by
// tracker.finish(). filled==true means synthesis can proceed; filled==false
// means at least one input could not be synthesized — reason says which and why.
struct SynthesisResult
{
    bool filled = false;
    std::string reason;

    static SynthesisResult ok()
    {
        return {true, {}};
    }
    static SynthesisResult unsupported(std::string why)
    {
        return {false, std::move(why)};
    }
};

// Tracks which leaf inputs of a bundle's graph have been accounted for by the
// per-node fill functions. A bundle contains a graph of one or more nodes — a
// single conv, or a fused chain like conv → bias_add → relu. One tracker is
// created for the entire graph's leaf inputs (non-virtual, non-output tensors),
// shared across all fill functions, and finish() is called once at the end.
//
// Graph structure (conv + bias + relu fused graph):
//
//   Data flows top-down. Roots are the leaf input tensors that the tracker
//   owns; the sink is the graph output tensor.
//
//        x (root/leaf)  w (root/leaf)  bias (root/leaf)
//         uid=1          uid=2           uid=4
//           \             /                |
//            \           /                 |
//         ┌──────────────┐                 |
//         │   ConvFwd    │  (internal)     |
//         └──────┬───────┘                 |
//                |                         |
//          conv_y (virtual, uid=10)        |
//                |                         |
//                \                        /
//              ┌──────────────────────┐
//              │   Pointwise ADD      │  (internal)
//              └──────────┬───────────┘
//                         |
//                   bias_out (virtual, uid=11)
//                         |
//              ┌──────────┴───────────┐
//              │   Pointwise RELU     │  (internal)
//              └──────────┬───────────┘
//                         |
//                    out (sink/leaf, uid=6)
//
//   Roots  = leaf input tensors, owned by tracker: {1, 2, 4}
//   Virtual = inter-node edges, not owned → fillFree/markDerived skip them
//   Sink   = graph output tensor, not owned
//
// Each leaf input must be declared as one of three mutually exclusive roles:
//
//   FREE       — random values in a range work. The range can be tight (e.g.
//                variance in [0.5, 1.5] to stay positive) or wide (e.g. x in
//                [-1, 1]). What matters is that any value in the range is valid.
//
//   STRUCTURED — random values in any range won't work. The data needs to be
//                consistent with other state or follow a specific format.
//
//                Example 1: dropout seeds — forward and backward must use the
//                same seed so they generate the same drop pattern. A randomly
//                synthesized seed for a standalone backward won't match any
//                forward pass, producing wrong gradients.
//
//                Example 2: page table indices (paged attention) — when serving
//                multiple users, each user's K and V data grows at different
//                rates. Instead of pre-allocating a large contiguous block per
//                user, GPU memory is pooled into equal-size chunks handed out
//                on demand. A user's data ends up scattered across
//                non-contiguous chunks. The page table tensor holds chunk
//                indices telling the kernel where each user's data lives.
//                Randomly generated indices would not correspond to valid
//                allocated chunks, producing incorrect reads or crashes.
//
//                Example 3: peer_stats (multi-GPU batchnorm) — when a batch
//                is split across multiple GPUs, each GPU computes local
//                statistics (mean, variance) for its chunk. To produce
//                correct global statistics, each GPU must read the others'
//                partial results. The peer_stats tensor holds references to
//                other GPUs' memory regions. Randomly generated values would
//                point to invalid cross-device memory.
//
//   DERIVED    — the value must come from another op's output, not from random
//                generation. In a fused fwd+bwd graph the forward output flows
//                to the backward input as a virtual tensor (not owned, silently
//                skipped). In a standalone backward, the same tensor is a leaf
//                input — markDerived records it, and finish() refuses because
//                no forward pass produced it.
//
// finish() succeeds only when every owned leaf input was declared as some role
// AND none were STRUCTURED or DERIVED. Undeclared inputs and refused inputs both
// produce a diagnostic message so the caller knows what went wrong.
//
// PRECONDITION — a validated, well-formed graph. The tracker trusts the leaf
// set it is handed and the virtual_ flag on every tensor:
//
//   * A required input referenced by a node is assumed to be a real leaf tensor
//     (not mislabeled virtual or aliased to an output). If it were, fillFree
//     would silently no-op on a non-owned uid and finish() would never see it.
//   * A virtual tensor is assumed to genuinely have a producer node. A standalone
//     backward whose `o`/`stats` were erroneously flagged virtual would skip the
//     markDerived refusal and "succeed" with garbage.
//
// Both of those malformed-graph states are rejected upstream — at bundle load
// (the flatbuffer build in loadIntegrationTestBundle) and again by the engine's
// own graph validation (from_binary / check_support / build_plans), which
// requires every virtual tensor to have a producer. By the time synthesis runs,
// the graph is well-formed, so the tracker does not re-validate topology.
class SynthesisTracker
{
public:
    SynthesisTracker(const std::vector<int64_t>& ownedLeafInputUids, InputTensorMap& inputs)
        : _inputs(inputs)
        , _owned(ownedLeafInputUids.begin(), ownedLeafInputUids.end())
    {
    }

    // Declares `uid` as FREE — fills it with random values in [lo, hi] and accounts for it.
    void fillFree(int64_t uid, float lo, float hi, std::mt19937& rng)
    {
        if(!isOwned(uid))
        {
            return;
        }
        const auto seed = static_cast<unsigned int>(rng());
        _inputs.at(uid)->fillTensorWithRandomValues(lo, hi, seed);
        _accounted.insert(uid);
    }

    void fillFree(flatbuffers::Optional<int64_t> uid, float lo, float hi, std::mt19937& rng)
    {
        if(uid.has_value())
        {
            fillFree(*uid, lo, hi, rng);
        }
    }

    // Declares `uid` as STRUCTURED — accounts for it but records a refusal.
    void markStructured(int64_t uid, const char* role)
    {
        if(!isOwned(uid))
        {
            return;
        }
        _accounted.insert(uid);
        _refusals.push_back(std::string(role) + " (structured input)");
    }

    void markStructured(flatbuffers::Optional<int64_t> uid, const char* role)
    {
        if(uid.has_value())
        {
            markStructured(*uid, role);
        }
    }

    // Declares `uid` as DERIVED — accounts for it but records a refusal.
    void markDerived(int64_t uid, const char* role)
    {
        if(!isOwned(uid))
        {
            return;
        }
        _accounted.insert(uid);
        _refusals.push_back(std::string(role) + " (derived from another computation)");
    }

    void markDerived(flatbuffers::Optional<int64_t> uid, const char* role)
    {
        if(uid.has_value())
        {
            markDerived(*uid, role);
        }
    }

    // Returns ok() when all owned leaf inputs were filled with random data.
    // Returns unsupported() when synthesis cannot produce valid data for
    // this graph — either because a leaf input is STRUCTURED/DERIVED
    // (we know about it but can't fill it), or because a leaf input was
    // never declared by any node's fill function.
    // Note: virtual inter-node tensors are not owned, so STRUCTURED/DERIVED
    // calls on them are silently ignored. Absent optional tensors are handled
    // by the flatbuffers::Optional overloads of fillFree/markStructured/
    // markDerived — they silently skip when the optional has no value.
    SynthesisResult finish(const char* opName) const
    {
        std::vector<std::string> reasons = _refusals;
        for(const int64_t uid : _owned)
        {
            if(_accounted.count(uid) == 0)
            {
                reasons.push_back("tensor uid=" + std::to_string(uid)
                                  + " (no role declared by initializer)");
            }
        }

        if(reasons.empty())
        {
            return SynthesisResult::ok();
        }

        std::ostringstream os;
        os << opName << " inputs cannot be synthesized: ";
        for(size_t i = 0; i < reasons.size(); ++i)
        {
            os << (i == 0 ? "" : ", ") << reasons[i];
        }
        return SynthesisResult::unsupported(os.str());
    }

private:
    bool isOwned(int64_t uid) const
    {
        return _owned.count(uid) != 0;
    }

    InputTensorMap& _inputs; // leaf inputs only (non-virtual, non-output tensors)
    std::set<int64_t> _owned;
    std::set<int64_t> _accounted;
    std::vector<std::string> _refusals;
};

} // namespace hipdnn_integration_tests::golden
