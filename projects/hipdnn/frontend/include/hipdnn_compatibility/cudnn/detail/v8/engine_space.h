// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hipdnn_backend.h>

#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_compatibility/cudnn/detail/v8/notes.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/CreateBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

// A finalized hipDNN operation graph and its engine index space. The engine
// space is enumerated on first use (the hipDNN heuristics need a HIP device);
// a failed enumeration is retried on the next call. Modes A and B rerank the
// same engines through hipDNN's ModeA/ModeB policies, each queried at most
// once per graph.
class GraphCore final : public RecordOf<CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR>
{
public:
    GraphCore(hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor graph, std::string tag)
        : _graph(std::move(graph))
        , _tag(std::move(tag))
    {
    }

    GraphCore(const GraphCore&) = delete;
    GraphCore& operator=(const GraphCore&) = delete;
    GraphCore(GraphCore&&) = delete;
    GraphCore& operator=(GraphCore&&) = delete;
    ~GraphCore() = default;

    hipdnnBackendDescriptor_t hipdnnGraph() const noexcept
    {
        return _graph.get();
    }

    const std::string& tag() const noexcept
    {
        return _tag;
    }

    std::int64_t engineCount() const
    {
        return static_cast<std::int64_t>(engineSpace().byId.size());
    }

    // Engine IDs in the rank order of `mode`; every mode ranks the FALLBACK
    // engine set (OPENSOURCE ranks as FALLBACK).
    const std::vector<std::int64_t>&
        rankedEngineIds(hipdnn_frontend::HeuristicMode mode
                        = hipdnn_frontend::HeuristicMode::FALLBACK) const
    {
        switch(mode)
        {
        case hipdnn_frontend::HeuristicMode::A:
            return policyRanking(_modeA, mode);
        case hipdnn_frontend::HeuristicMode::B:
            return policyRanking(_modeB, mode);
        default:
            return engineSpace().rankedIds;
        }
    }

    // Dense index (engines ordered by signed ascending ID) <-> engine ID.
    std::optional<std::int64_t> engineIdAt(std::int64_t index) const
    {
        const std::vector<Engine>& engines = engineSpace().byId;
        if(index < 0 || index >= static_cast<std::int64_t>(engines.size()))
        {
            return std::nullopt;
        }
        return engines[static_cast<std::size_t>(index)].id;
    }

    std::optional<std::int64_t> engineIndexOf(std::int64_t id) const
    {
        const std::vector<Engine>& engines = engineSpace().byId;
        const auto found = findEngine(engines, id);
        if(found == engines.end())
        {
            return std::nullopt;
        }
        return static_cast<std::int64_t>(found - engines.begin());
    }

    // The finalized hipDNN ENGINECFG (default knobs) and ENGINE the heuristics
    // produced for `id`; null when `id` is not in the index space. Owned here.
    hipdnnBackendDescriptor_t defaultEngineConfig(std::int64_t id) const
    {
        const std::vector<Engine>& engines = engineSpace().byId;
        const auto found = findEngine(engines, id);
        return found == engines.end() ? nullptr : found->config.get();
    }

    hipdnnBackendDescriptor_t defaultEngine(std::int64_t id) const
    {
        const std::vector<Engine>& engines = engineSpace().byId;
        const auto found = findEngine(engines, id);
        return found == engines.end() ? nullptr : found->engine.get();
    }

private:
    struct Engine
    {
        std::int64_t id;
        hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor config;
        hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor engine;
    };

    struct EngineSpace
    {
        std::vector<std::int64_t> rankedIds;
        std::vector<Engine> byId;
    };

    struct PolicyRanking
    {
        std::once_flag once;
        std::vector<std::int64_t> rankedIds;
    };

    // The heuristic is kept alive until its result configs are finalized.
    struct HeuristicQuery
    {
        hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor heuristic;
        std::vector<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> configs;
    };

    static std::vector<Engine>::const_iterator findEngine(const std::vector<Engine>& engines,
                                                          std::int64_t id)
    {
        const auto found = std::lower_bound(
            engines.begin(), engines.end(), id, [](const Engine& engine, std::int64_t value) {
                return engine.id < value;
            });
        return found != engines.end() && found->id == id ? found : engines.end();
    }

    const EngineSpace& engineSpace() const
    {
        std::call_once(_engineOnce, [this] { _engines = enumerateEngines(hipdnnGraph()); });
        return _engines;
    }

    const std::vector<std::int64_t>& policyRanking(PolicyRanking& ranking,
                                                   hipdnn_frontend::HeuristicMode mode) const
    {
        const EngineSpace& space = engineSpace();
        std::call_once(ranking.once, [this, &ranking, &space, mode] {
            ranking.rankedIds = rankByPolicy(hipdnnGraph(), space, mode);
        });
        return ranking.rankedIds;
    }

    // A FALLBACK-mode ENGINEHEUR for `graph`, finalized, with its result
    // configs unfinalized in rank order. A non-empty `policyOrder` selects the
    // prediction policies, as the native frontend sends modes A and B.
    static HeuristicQuery queryHeuristic(hipdnnBackendDescriptor_t graph,
                                         const std::vector<std::int64_t>& policyOrder)
    {
        using hipdnn_frontend::detail::hipdnnBackend;

        HeuristicQuery query;
        query.heuristic = createHipdnnDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                                                 "creating the hipDNN engine heuristic");
        auto* const heuristic = query.heuristic.get();
        checkHipdnn(hipdnnBackend()->backendSetAttribute(heuristic,
                                                         HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         1,
                                                         static_cast<const void*>(&graph)),
                    "setting the hipDNN engine heuristic graph");
        const hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
        checkHipdnn(hipdnnBackend()->backendSetAttribute(heuristic,
                                                         HIPDNN_ATTR_ENGINEHEUR_MODE,
                                                         HIPDNN_TYPE_HEUR_MODE,
                                                         1,
                                                         static_cast<const void*>(&mode)),
                    "setting the hipDNN engine heuristic mode");
        if(!policyOrder.empty())
        {
            checkHipdnn(
                hipdnnBackend()->backendSetAttribute(heuristic,
                                                     HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT,
                                                     HIPDNN_TYPE_INT64,
                                                     static_cast<std::int64_t>(policyOrder.size()),
                                                     static_cast<const void*>(policyOrder.data())),
                "setting the hipDNN engine heuristic policy order");
        }
        checkHipdnn(hipdnnBackend()->backendFinalize(heuristic),
                    "finalizing the hipDNN engine heuristic");

        std::int64_t available = 0;
        checkHipdnn(hipdnnBackend()->backendGetAttribute(heuristic,
                                                         HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         0,
                                                         &available,
                                                         nullptr),
                    "counting the hipDNN engine heuristic results");
        if(available <= 0)
        {
            return query;
        }
        std::vector<hipdnnBackendDescriptor_t> rawConfigs;
        query.configs.reserve(static_cast<std::size_t>(available));
        rawConfigs.reserve(static_cast<std::size_t>(available));
        for(std::int64_t i = 0; i < available; ++i)
        {
            query.configs.push_back(createHipdnnDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR,
                                                           "creating a hipDNN engine config"));
            rawConfigs.push_back(query.configs.back().get());
        }
        std::int64_t returned = 0;
        checkHipdnn(hipdnnBackend()->backendGetAttribute(heuristic,
                                                         HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         available,
                                                         &returned,
                                                         static_cast<void*>(rawConfigs.data())),
                    "reading the hipDNN engine heuristic results");
        query.configs.resize(
            static_cast<std::size_t>(std::clamp<std::int64_t>(returned, 0, available)));
        return query;
    }

    // Every applicable engine with its default config. hipDNN reports an engine
    // ID only after its config is finalized, which runs the engine's workspace
    // query; an engine whose query fails is dropped instead of failing the
    // whole graph.
    static EngineSpace enumerateEngines(hipdnnBackendDescriptor_t graph)
    {
        using hipdnn_frontend::detail::hipdnnBackend;
        using hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor;

        HeuristicQuery query = queryHeuristic(graph, {});
        EngineSpace space;
        std::unordered_set<std::int64_t> seen;
        for(std::size_t i = 0; i < query.configs.size(); ++i)
        {
            if(hipdnnBackend()->backendFinalize(query.configs[i].get()) != HIPDNN_STATUS_SUCCESS)
            {
                // Drained even when the warning is filtered out.
                const std::string message = takeHipdnnMessage();
                HIPDNN_FE_LOG_WARN("[cudnn_frontend] Dropping heuristic result "
                                   << i << ": its default engine config failed to finalize: "
                                   << message);
                continue;
            }
            ScopedHipdnnBackendDescriptor engine = engineOfConfig(query.configs[i].get());
            const std::int64_t id = engineGlobalIndex(engine.get());
            if(!seen.insert(id).second)
            {
                continue;
            }
            space.rankedIds.push_back(id);
            space.byId.push_back(Engine{id, std::move(query.configs[i]), std::move(engine)});
        }
        std::sort(space.byId.begin(), space.byId.end(), [](const Engine& a, const Engine& b) {
            return a.id < b.id;
        });
        return space;
    }

    // The FALLBACK engine set in the order the `mode` policy ranks it: IDs
    // outside the set are dropped, and engines the policy did not return keep
    // their FALLBACK order after the ranked ones. Results use only the order,
    // not the policy's configs.
    static std::vector<std::int64_t> rankByPolicy(hipdnnBackendDescriptor_t graph,
                                                  const EngineSpace& space,
                                                  hipdnn_frontend::HeuristicMode mode)
    {
        using hipdnn_frontend::detail::hipdnnBackend;

        if(space.byId.size() <= 1)
        {
            return space.rankedIds;
        }
        const HeuristicQuery query
            = queryHeuristic(graph, hipdnn_frontend::detail::heuristicPolicyOrder({mode}));
        std::vector<std::int64_t> ranked;
        ranked.reserve(space.byId.size());
        std::unordered_set<std::int64_t> seen;
        for(const hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor& config : query.configs)
        {
            if(hipdnnBackend()->backendFinalize(config.get()) != HIPDNN_STATUS_SUCCESS)
            {
                // Unranked: its engine, if in the set, keeps its FALLBACK place below.
                static_cast<void>(takeHipdnnMessage());
                continue;
            }
            const std::int64_t id = engineGlobalIndex(engineOfConfig(config.get()).get());
            if(findEngine(space.byId, id) != space.byId.end() && seen.insert(id).second)
            {
                ranked.push_back(id);
            }
        }
        for(const std::int64_t id : space.rankedIds)
        {
            if(seen.count(id) == 0)
            {
                ranked.push_back(id);
            }
        }
        return ranked;
    }

    hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor _graph;
    std::string _tag;
    // Declared after _graph so the cached configs and engines are released
    // before the graph they refer to.
    mutable std::once_flag _engineOnce;
    mutable EngineSpace _engines;
    mutable PolicyRanking _modeA;
    mutable PolicyRanking _modeB;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
