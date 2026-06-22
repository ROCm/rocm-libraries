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

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::golden
{

// Pre-allocated input tensors keyed by uid, handed to a fill function to populate.
using InputTensorMap
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

// The outcome of trying to synthesize a node's inputs (tier-3 graph-only path).
//
//   filled == true  : every leaf input the node owns was given valid data.
//   filled == false : at least one owned input is STRUCTURED, DERIVED, or
//                     unrecognized. `reason` explains which — the harness SKIPs.
struct FillOutcome
{
    bool filled = false;
    std::string reason;

    static FillOutcome ok()
    {
        return {true, {}};
    }
    static FillOutcome unsupported(std::string why)
    {
        return {false, std::move(why)};
    }
};

// Drives a fill function's per-role declarations and enforces deny-by-default.
//
// An initializer declares, for each role it knows, whether that input is FREE
// (fill from a numeric range), STRUCTURED (needs internal structure we cannot
// synthesize), or DERIVED (must satisfy a relation to another computation). After
// all declarations, finish() returns Filled only if EVERY owned leaf input was
// accounted for — any owned uid that no declaration claimed is itself a refusal
// (a role the initializer forgot, or a tensor it does not understand). This is the
// safety net that prevents a half-filled input map from reaching an executor.
//
// Absent optional inputs are passed as uid 0 (the flatbuffer default) or simply
// not present in `inputs`; such uids are ignored — only uids that are actually
// owned leaf inputs need accounting.
class RoleAccounting
{
public:
    RoleAccounting(const std::vector<int64_t>& ownedLeafInputUids, InputTensorMap& inputs)
        : _inputs(inputs)
        , _owned(ownedLeafInputUids.begin(), ownedLeafInputUids.end())
    {
    }

    // FREE role: if `uid` is an owned leaf input, fill it with uniform values in
    // [lo, hi] and mark it accounted. A uid of 0 or one not in the owned set is
    // ignored (an absent optional input). Uses the tensor's own dtype-aware random
    // fill, so no std::visit on dtype is needed here.
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

    // STRUCTURED role: declares that `uid`, if owned, cannot be synthesized
    // because it needs internal structure (sequence lengths, page tables, block
    // masks, dropout seeds, ...). Records a refusal reason.
    void markStructured(int64_t uid, const char* role)
    {
        if(!isOwned(uid))
        {
            return;
        }
        _accounted.insert(uid);
        _refusals.push_back(std::string(role) + " (structured input)");
    }

    // DERIVED role: declares that `uid`, if owned, cannot be synthesized standalone
    // because it must equal the output of another computation (e.g. SDPA-backward
    // consumes the forward's O and softmax stats). Records a refusal reason.
    void markDerived(int64_t uid, const char* role)
    {
        if(!isOwned(uid))
        {
            return;
        }
        _accounted.insert(uid);
        _refusals.push_back(std::string(role) + " (derived from another computation)");
    }

    // Filled iff every owned leaf input was accounted AND none were refused.
    // Otherwise Unsupported, listing the refused roles plus any owned uid no
    // declaration claimed (the deny-by-default catch).
    FillOutcome finish(const char* opName) const
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
            return FillOutcome::ok();
        }

        std::ostringstream os;
        os << opName << " inputs cannot be synthesized: ";
        for(size_t i = 0; i < reasons.size(); ++i)
        {
            os << (i == 0 ? "" : ", ") << reasons[i];
        }
        return FillOutcome::unsupported(os.str());
    }

private:
    bool isOwned(int64_t uid) const
    {
        return uid != 0 && _owned.count(uid) != 0;
    }

    InputTensorMap& _inputs;
    std::set<int64_t> _owned;
    std::set<int64_t> _accounted;
    std::vector<std::string> _refusals;
};

} // namespace hipdnn_integration_tests::golden
