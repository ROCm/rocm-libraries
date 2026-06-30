// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/input_init/SynthesisConfig.hpp"

namespace hipdnn_integration_tests
{

// Pre-allocated input tensors keyed by uid, handed to a fill function to populate.
using InputTensorMap
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

// Records what was actually used to fill each tensor during synthesis.
// Available after synthesize() via SynthesisResult::meta for serialization
// (e.g. --capture-bundles writes this to .meta.json for replay).
// Each TensorInit has its seed populated (non-nullopt).
class SynthesisMeta
{
public:
    void record(int64_t uid, TensorInit init, unsigned int seed)
    {
        init.seed = seed;
        _tensors[uid] = init;
    }

    std::vector<int64_t> tensorUids() const
    {
        std::vector<int64_t> uids;
        uids.reserve(_tensors.size());
        for(const auto& [uid, _] : _tensors)
        {
            uids.push_back(uid);
        }
        return uids;
    }

    const TensorInit& init(int64_t uid) const
    {
        return _tensors.at(uid);
    }
    unsigned int seed(int64_t uid) const
    {
        return *_tensors.at(uid).seed;
    }
    bool has(int64_t uid) const
    {
        return _tensors.count(uid) != 0;
    }

    const std::unordered_map<int64_t, TensorInit>& tensors() const
    {
        return _tensors;
    }

private:
    std::unordered_map<int64_t, TensorInit> _tensors;
};

// Result of a synthesis step — returned by per-node fill functions and by
// synthesizer.synthesize(). filled==true means synthesis can proceed; filled==false
// means at least one input could not be synthesized — reason says which and why.
struct SynthesisResult
{
    bool filled = false;
    std::string reason;
    SynthesisMeta meta;

    static SynthesisResult ok(SynthesisMeta m = {})
    {
        return {true, {}, std::move(m)};
    }
    static SynthesisResult unsupported(std::string why)
    {
        return {false, std::move(why), {}};
    }
};

// Tracks which leaf inputs of a bundle's graph have been accounted for by the
// per-node fill functions. A bundle contains a graph of one or more nodes — a
// single conv, or a fused chain like conv → bias_add → relu. One synthesizer is
// created for the entire graph's leaf inputs (non-virtual, non-output tensors),
// shared across all fill functions, and synthesize() is called once at the end.
//
// Graph structure (conv + bias + relu fused graph):
//
//   Data flows top-down. Roots are the leaf input tensors that the synthesizer
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
//   Roots  = leaf input tensors, owned by synthesizer: {1, 2, 4}
//   Virtual = inter-node edges, not owned → declare/markDerived skip them
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
//   DERIVED    — the value must come from another op's output, not from random
//                generation.
//
// Two-phase synthesis: declare exceptions, then fill all.
//
// All owned leaf uids are assumed FREE by default. Declaration functions
// only need to speak up for exceptions:
//   - range(uid, lo, hi) — op-specific range (e.g. variance in [0.5, 1.5])
//   - markStructured(uid) — can't be randomized
//   - markDerived(uid)    — must come from another op's output
//
// synthesize() resolves every owned uid and fills it. Uids marked structured
// or derived are skipped (and reported as refusals).
//
// Resolution priority per uid:
//   1. Code         → synthesis().range(uid, lo, hi) in the test   (config)
//   2. Metadata     → loaded from bundle JSON in setBundle()       (config)
//   3. Op default   → synthesizer.range(uid, lo, hi) from fill fns    (synthesizer)
//   4. Struct default → TensorInit{} (lo=-1, hi=1, kind=FREE)
//
// Config owns user/metadata layers. Tracker owns op defaults. Clean
// separation: user overrides can never be stomped by fill function defaults.
//
// Seeds are assigned at construction (deterministic per-tensor from rng).
class InputSynthesizer
{
public:
    InputSynthesizer(const std::vector<int64_t>& ownedLeafInputUids,
                     InputTensorMap& inputs,
                     SynthesisConfig* config = nullptr)
        : _inputs(inputs)
        , _owned(ownedLeafInputUids.begin(), ownedLeafInputUids.end())
        , _config(config)
    {
        std::mt19937 rng(config != nullptr ? config->getSeedEntropy()
                                           : SynthesisConfig::K_DEFAULT_SEED_ENTROPY);

        for(const int64_t uid : ownedLeafInputUids)
        {
            std::optional<unsigned int> s;
            if(config != nullptr)
            {
                s = config->resolveSeed(uid);
            }
            _seeds[uid] = s.has_value() ? *s : static_cast<unsigned int>(rng());
        }
    }

    // ── Exceptions only — everything else is FREE by default ─────────────

    // Registers an op-specific default range for `uid`.
    void range(int64_t uid, float lo, float hi)
    {
        if(!isOwned(uid))
        {
            return;
        }
        _opDefaults[uid].kind = TensorInit::Kind::FREE;
        _opDefaults[uid].lo = lo;
        _opDefaults[uid].hi = hi;
    }

    void range(flatbuffers::Optional<int64_t> uid, float lo, float hi)
    {
        if(uid.has_value())
        {
            range(*uid, lo, hi);
        }
    }

    // Marks `uid` as STRUCTURED — will refuse synthesis.
    void markStructured(int64_t uid, const char* role)
    {
        if(!isOwned(uid))
        {
            return;
        }
        _excluded.insert(uid);
        _refusals.push_back(std::string(role) + " (structured input)");
    }

    void markStructured(flatbuffers::Optional<int64_t> uid, const char* role)
    {
        if(uid.has_value())
        {
            markStructured(*uid, role);
        }
    }

    // Marks `uid` as DERIVED — will refuse synthesis.
    void markDerived(int64_t uid, const char* role)
    {
        if(!isOwned(uid))
        {
            return;
        }
        _excluded.insert(uid);
        _refusals.push_back(std::string(role) + " (derived from another computation)");
    }

    void markDerived(flatbuffers::Optional<int64_t> uid, const char* role)
    {
        if(uid.has_value())
        {
            markDerived(*uid, role);
        }
    }

    // ── Resolve + fill all owned uids ────────────────────────────────────

    // Walks every node in the graph, calls per-op declaration functions,
    // then resolves + fills all owned uids. Defined out-of-class in
    // SynthesizeInputs.hpp (breaks the include cycle).
    SynthesisResult synthesize(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph);

    // Resolves + fills all owned uids after declarations are done.
    // opName is used only in the error message if any uid was refused.
    SynthesisResult synthesize(const char* opName)
    {
        for(const int64_t uid : _owned)
        {
            if(_excluded.count(uid) != 0)
            {
                continue;
            }

            auto seedIt = _seeds.find(uid);
            if(seedIt == _seeds.end())
            {
                continue;
            }

            const TensorInit init = resolve(uid);
            const unsigned int seed = seedIt->second;

            switch(init.kind)
            {
            case TensorInit::Kind::FREE:
                _inputs.at(uid)->fillTensorWithRandomValues(init.lo, init.hi, seed);
                break;
            case TensorInit::Kind::FIXED:
                _inputs.at(uid)->fillTensorWithValue(init.value);
                break;
            case TensorInit::Kind::STRUCTURED:
                _refusals.push_back("uid=" + std::to_string(uid) + " (structured, config)");
                break;
            case TensorInit::Kind::DERIVED:
                _refusals.push_back("uid=" + std::to_string(uid) + " (derived, config)");
                break;
            default:
                break;
            }
            _meta.record(uid, init, seed);
        }

        if(_refusals.empty())
        {
            return SynthesisResult::ok(std::move(_meta));
        }

        std::ostringstream os;
        os << opName << " inputs cannot be synthesized: ";
        for(size_t i = 0; i < _refusals.size(); ++i)
        {
            os << (i == 0 ? "" : ", ") << _refusals[i];
        }
        return SynthesisResult::unsupported(os.str());
    }

    bool isOwned(int64_t uid) const
    {
        return _owned.count(uid) != 0;
    }

    TensorInit resolve(int64_t uid) const
    {
        if(_config != nullptr)
        {
            if(auto fromConfig = _config->resolve(uid))
            {
                return *fromConfig;
            }
        }
        if(auto it = _opDefaults.find(uid); it != _opDefaults.end())
        {
            return it->second;
        }
        return TensorInit{};
    }

private:
    InputTensorMap& _inputs;
    std::set<int64_t> _owned;
    std::set<int64_t> _excluded;
    std::vector<std::string> _refusals;
    SynthesisConfig* _config = nullptr;
    std::unordered_map<int64_t, unsigned int> _seeds;
    std::unordered_map<int64_t, TensorInit> _opDefaults;
    SynthesisMeta _meta;
};

} // namespace hipdnn_integration_tests
