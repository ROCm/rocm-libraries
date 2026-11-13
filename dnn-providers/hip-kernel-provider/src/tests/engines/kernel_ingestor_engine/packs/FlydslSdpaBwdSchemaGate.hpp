// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlanBuilder.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "core/ApplicabilityManager.hpp"
#include "core/ApplicabilityPool.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file FlydslSdpaBwdSchemaGate.hpp
 * @brief The shipped SDPA backward engines behind both schema gates, over the graphs a backend
 *        stamps with `min_required_engine_api_version`.
 *
 * GenericPlanBuilder::isApplicable declines a graph whose schema floor is above the engine's
 * `sdk_version` before any matcher runs, and a graph without the stamp reads as 1.0.0. The
 * provider's ApplicabilityManager declines it the same way for an engine registered below the
 * floor, and Container registers a descriptor-backed engine at its UED's `sdk_version`. The
 * matcher tests (FlydslSdpaBwdTestGraphs.hpp) leave graphs unstamped, so they never see either
 * gate; the cases here stamp what the backend stamps (ragged tensors and non-default
 * alignment 1.3.0, a ragged offset multiplier 1.4.0) and ask the builder and the manager over
 * the engine's loaded descriptor set, on the architecture the shard was packed for.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

namespace schema_gate
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

struct Settings
{
    hipdnn_plugin_sdk::ingestor::IngestorSettings ingestorSettings;
};

/// The minimum a GenericPlanBuilder asks of its context; isApplicable() never builds a plan.
class PlanContext
{
public:
    const Settings& executionSettings() const
    {
        return _settings;
    }

    void setPlan(std::unique_ptr<hipdnn_plugin_sdk::IPlan<Handle>> /*plan*/) {}

private:
    Settings _settings;
};

/// One fixed device of @p arch.
class FixedArchResolver : public hipdnn_plugin_sdk::ingestor::IDeviceResolver<Handle>
{
public:
    explicit FixedArchResolver(const std::string& arch)
    {
        _properties.gcnArchName = arch;
        _properties.warpSize = arch.rfind("gfx9", 0) == 0 ? 64 : 32;
    }

    hipdnn_plugin_sdk::ingestor::DeviceId deviceId(const Handle& /*handle*/) const override
    {
        return 0;
    }

    const hipdnn_plugin_sdk::ingestor::DeviceProperties&
        deviceProperties(hipdnn_plugin_sdk::ingestor::DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    hipdnn_plugin_sdk::ingestor::DeviceProperties _properties;
};

/// The engine @p set describes, with a builder over a state manager of its own and a device
/// of @p arch, and the same engine registered in an ApplicabilityManager as Container does,
/// at the `sdk_version` of @p set. A caller may lower @p sdkVersion to check that a case
/// notices each gate: the builder reads it from its descriptor, the manager from the
/// registration, and the engine itself keeps the descriptor of @p set.
class GatedEngine
{
public:
    using Builder = hipdnn_plugin_sdk::ingestor::GenericPlanBuilder<Handle, Settings, PlanContext>;

    GatedEngine(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                const std::string& arch,
                std::optional<hipdnn_data_sdk::utilities::Version> sdkVersion = std::nullopt)
        : _engine(set.engine)
        , _resolver(arch)
        , _engineId(hipdnn_data_sdk::utilities::engineNameToId(set.engine.name))
    {
        if(sdkVersion.has_value())
        {
            _engine.sdkVersion = *sdkVersion;
        }
        // The discovered state is immutable and shared; the state manager over it, with its
        // own caches, is this engine's alone.
        auto state = discoveredSharedState(set.engine.id);
        if(state == nullptr)
        {
            throw std::runtime_error("no shared state for engine '" + set.engine.name + "'");
        }
        _stateManager
            = std::make_unique<hipdnn_plugin_sdk::ingestor::KernelIngestorStateManager<Handle>>(
                state);
        _builder = std::make_unique<Builder>(_engine, *_stateManager, _resolver);
        _manager.addEngine(hipdnn_plugin_sdk::ingestor::makeEngine<Handle, ::Settings, ::Context>(
                               set, std::move(state), _resolver),
                           set.engine.name,
                           _engine.sdkVersion);
    }

    /// Whether the builder accepts @p spec's graph, stamped @p stamp (nullopt: unstamped).
    bool applicable(const SdpaBwdSpec& spec, std::optional<std::pair<uint32_t, uint32_t>> stamp)
    {
        return withGraph(spec, stamp, [&](const auto& graph) {
            const Handle handle;
            return _builder->isApplicable(handle, graph);
        });
    }

    /// Whether the ApplicabilityManager lists the engine for @p spec's graph, stamped
    /// @p stamp (nullopt: unstamped).
    bool providerApplicable(const SdpaBwdSpec& spec,
                            std::optional<std::pair<uint32_t, uint32_t>> stamp)
    {
        return withGraph(spec, stamp, [&](const auto& graph) {
            Handle handle;
            const auto ids = _manager.getApplicableEngineIds(handle, graph);
            return std::find(ids.begin(), ids.end(), _engineId) != ids.end();
        });
    }

private:
    template <typename Ask>
    static bool withGraph(const SdpaBwdSpec& spec,
                          std::optional<std::pair<uint32_t, uint32_t>> stamp,
                          Ask&& ask)
    {
        const auto stamped = spec.withEdit([stamp](SdpaBwdGraph& graph) {
            if(stamp.has_value())
            {
                graph.minRequiredEngineApi
                    = data_objects::EngineApiVersion(stamp->first, stamp->second, 0);
            }
        });
        const auto buffer = buildSdpaBwdGraph(stamped);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            buffer.GetBufferPointer(), buffer.GetSize());
        return ask(graph);
    }

    hipdnn_plugin_sdk::ingestor::EngineDescriptor _engine;
    FixedArchResolver _resolver;
    int64_t _engineId;
    core::ApplicabilityPool _pool;
    core::ApplicabilityManager _manager{_pool};
    std::unique_ptr<hipdnn_plugin_sdk::ingestor::KernelIngestorStateManager<Handle>> _stateManager;
    std::unique_ptr<Builder> _builder;
};

/// A padded graph whose tensors named in @p names are ragged, each with an offset tensor of
/// its own and the multiplier @p multiplier. A ragged Q brings ragged stats packed (H, T).
inline SdpaBwdSpec raggedSpec(const std::vector<std::string>& names, int64_t multiplier)
{
    return SdpaBwdSpec{}.withEdit([names, multiplier](SdpaBwdGraph& g) {
        addSeqLens(g);
        for(const auto& name : names)
        {
            addRagged(g, name, multiplier);
            if(name == "q")
            {
                addRagged(g, "stats");
                g.tensor("stats").strides = {int64_t{1} << 20, g.batch * g.seqQ, 1, 1};
            }
        }
    });
}

inline std::optional<std::pair<uint32_t, uint32_t>> stampOf(uint32_t major, uint32_t minor)
{
    return std::make_pair(major, minor);
}

/// One graph and the verdict the engine must give it once the gate lets it through.
struct Case
{
    std::string label;
    SdpaBwdSpec spec;
    /// The (major, minor) the backend stamps on the graph; nullopt: unstamped.
    std::optional<std::pair<uint32_t, uint32_t>> stamp;
    bool served;
};

/// What the backend stamps and the matcher decides for every field 1.3.0 and 1.4.0 let
/// through: ragged tensors with and without a multiplier, a non-default alignment, and one
/// graph per field the matcher declines (misaligned, ragged stats beside a dense Q, a ragged
/// bias, a multiplier below 1).
inline std::vector<Case> cases()
{
    std::vector<Case> all;
    all.push_back({"dense, unstamped", SdpaBwdSpec{}, std::nullopt, true});
    all.push_back(
        {"ragged k, dk, dv at 1.3.0", raggedSpec({"k", "dk", "dv"}, 1), stampOf(1, 3), true});
    all.push_back({"ragged q, k, v at 1.3.0, multiplier 2",
                   raggedSpec({"q", "k", "v"}, 2),
                   stampOf(1, 3),
                   true});
    all.push_back({"ragged q, k, v at 1.4.0, multiplier 2",
                   raggedSpec({"q", "k", "v"}, 2),
                   stampOf(1, 4),
                   true});
    all.push_back({"ragged q, o, do, dq at 1.4.0, multiplier 2",
                   raggedSpec({"q", "o", "do", "dq"}, 2),
                   stampOf(1, 4),
                   true});
    all.push_back({"alignment 32 at 1.3.0",
                   SdpaBwdSpec{}.withEdit([](SdpaBwdGraph& g) { g.tensor("q").alignment = 32; }),
                   stampOf(1, 3),
                   true});
    all.push_back({"alignment 8 at 1.3.0",
                   SdpaBwdSpec{}.withEdit([](SdpaBwdGraph& g) { g.tensor("q").alignment = 8; }),
                   stampOf(1, 3),
                   false});
    all.push_back({"ragged stats beside a dense Q at 1.4.0",
                   raggedSpec({"k"}, 2).withEdit([](SdpaBwdGraph& g) { addRagged(g, "stats"); }),
                   stampOf(1, 4),
                   false});
    all.push_back({"ragged bias at 1.4.0",
                   raggedSpec({"k"}, 2).withEdit([](SdpaBwdGraph& g) {
                       addBias(g, {g.batch, g.heads, g.seqQ, g.seqKv});
                       g.tensor("bias").raggedOffsetUid = sdpa_bwd_uid::RAGGED_OFFSET_K;
                   }),
                   stampOf(1, 4),
                   false});
    all.push_back({"multiplier 0 at 1.4.0", raggedSpec({"k"}, 0), stampOf(1, 4), false});
    return all;
}

inline constexpr const char* SCHEMA_GATE_LOG = "it understands graph schema";

/// Every case's disagreement with its verdict on @p set's engine, on a device of @p arch,
/// through the plan builder and through the provider's ApplicabilityManager, as "label: what
/// happened". A declined case also fails when a schema gate, not the matcher, declined it.
inline std::vector<std::string> failures(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set,
                                         const std::string& arch)
{
    std::vector<std::string> found;
    GatedEngine engine(set, arch);
    for(const auto& c : cases())
    {
        auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        const bool served = engine.applicable(c.spec, c.stamp);
        if(served != c.served)
        {
            found.push_back(c.label + ": " + (served ? "served" : "declined") + ", expected "
                            + (c.served ? "served" : "declined"));
        }
        const bool listed = engine.providerApplicable(c.spec, c.stamp);
        if(listed != c.served)
        {
            found.push_back(c.label + ": " + (listed ? "listed" : "not listed")
                            + " by the provider, expected " + (c.served ? "listed" : "not listed"));
        }
        if(recorder.hasLogContaining(HIPDNN_SEV_INFO, SCHEMA_GATE_LOG))
        {
            found.push_back(c.label + ": declined by the schema gate (engine sdk_version "
                            + set.engine.sdkVersion.str() + ")");
        }
    }
    return found;
}

/// A message when an engine declaring schema 1.2.0, the one the backward engines shipped with,
/// does not decline a ragged graph stamped 1.3.0 at a gate; nullopt when both decline it. The
/// plan builder is asked with the engine's descriptor lowered and the provider's manager with
/// the registration lowered, each on its own, so the cases above can be shown to see both.
inline std::optional<std::string>
    gateIsInvisible(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set, const std::string& arch)
{
    GatedEngine engine(set, arch, hipdnn_data_sdk::utilities::Version{std::string("1.2.0")});
    const auto spec = raggedSpec({"k"}, 1);
    {
        auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        const bool served = engine.applicable(spec, stampOf(1, 3));
        if(served || !recorder.hasLogContaining(HIPDNN_SEV_INFO, SCHEMA_GATE_LOG))
        {
            return std::string("a 1.2.0 engine ") + (served ? "served" : "declined")
                   + " a graph stamped 1.3.0 without the plan builder's schema gate declining it";
        }
    }
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const bool listed = engine.providerApplicable(spec, stampOf(1, 3));
    if(listed || !recorder.hasLogContaining(HIPDNN_SEV_INFO, SCHEMA_GATE_LOG))
    {
        return std::string("an engine registered at 1.2.0 was ")
               + (listed ? "listed" : "not listed")
               + " for a graph stamped 1.3.0 without the provider's schema gate declining it";
    }
    return std::nullopt;
}

} // namespace schema_gate

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
