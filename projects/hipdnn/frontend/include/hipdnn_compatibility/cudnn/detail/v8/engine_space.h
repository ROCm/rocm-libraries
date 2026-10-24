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
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

// A finalized hipDNN operation graph and its engine index space. The engine
// space is enumerated on first use (the hipDNN heuristics need a HIP device);
// a failed enumeration is retried on the next call.
class GraphCore final : public RecordOf<CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR>
{
public:
    GraphCore(hipdnnHandle_t handle,
              hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor graph,
              std::string tag)
        : _handle(handle)
        , _graph(std::move(graph))
        , _tag(std::move(tag))
    {
    }

    GraphCore(const GraphCore&) = delete;
    GraphCore& operator=(const GraphCore&) = delete;
    GraphCore(GraphCore&&) = delete;
    GraphCore& operator=(GraphCore&&) = delete;
    ~GraphCore() = default;

    // hipDNN has no getter for a graph's handle, so this is the only record of it.
    hipdnnHandle_t handle() const noexcept
    {
        return _handle;
    }

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

    // Engine IDs in heuristic rank order.
    const std::vector<std::int64_t>& rankedEngineIds() const
    {
        return engineSpace().rankedIds;
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

    // Every applicable engine with its default config. hipDNN reports an engine
    // ID only after its config is finalized, which runs the engine's workspace
    // query; an engine whose query fails is dropped instead of failing the
    // whole graph.
    static EngineSpace enumerateEngines(hipdnnBackendDescriptor_t graph)
    {
        using hipdnn_frontend::detail::hipdnnBackend;
        using hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor;

        const ScopedHipdnnBackendDescriptor heuristic = createHipdnnDescriptor(
            HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, "creating the hipDNN engine heuristic");
        checkHipdnn(hipdnnBackend()->backendSetAttribute(heuristic.get(),
                                                         HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         1,
                                                         static_cast<const void*>(&graph)),
                    "setting the hipDNN engine heuristic graph");
        const hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
        checkHipdnn(hipdnnBackend()->backendSetAttribute(heuristic.get(),
                                                         HIPDNN_ATTR_ENGINEHEUR_MODE,
                                                         HIPDNN_TYPE_HEUR_MODE,
                                                         1,
                                                         static_cast<const void*>(&mode)),
                    "setting the hipDNN engine heuristic mode");
        checkHipdnn(hipdnnBackend()->backendFinalize(heuristic.get()),
                    "finalizing the hipDNN engine heuristic");

        std::int64_t available = 0;
        checkHipdnn(hipdnnBackend()->backendGetAttribute(heuristic.get(),
                                                         HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         0,
                                                         &available,
                                                         nullptr),
                    "counting the hipDNN engine heuristic results");

        EngineSpace space;
        if(available <= 0)
        {
            return space;
        }
        std::vector<ScopedHipdnnBackendDescriptor> configs;
        std::vector<hipdnnBackendDescriptor_t> rawConfigs;
        configs.reserve(static_cast<std::size_t>(available));
        rawConfigs.reserve(static_cast<std::size_t>(available));
        for(std::int64_t i = 0; i < available; ++i)
        {
            configs.push_back(createHipdnnDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR,
                                                     "creating a hipDNN engine config"));
            rawConfigs.push_back(configs.back().get());
        }
        std::int64_t returned = 0;
        checkHipdnn(hipdnnBackend()->backendGetAttribute(heuristic.get(),
                                                         HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         available,
                                                         &returned,
                                                         static_cast<void*>(rawConfigs.data())),
                    "reading the hipDNN engine heuristic results");

        const auto results
            = static_cast<std::size_t>(std::clamp<std::int64_t>(returned, 0, available));
        std::unordered_set<std::int64_t> seen;
        for(std::size_t i = 0; i < results; ++i)
        {
            if(hipdnnBackend()->backendFinalize(rawConfigs[i]) != HIPDNN_STATUS_SUCCESS)
            {
                // Drained even when the warning is filtered out.
                const std::string message = takeHipdnnMessage();
                HIPDNN_FE_LOG_WARN("[cudnn_frontend] Dropping heuristic result "
                                   << i << ": its default engine config failed to finalize: "
                                   << message);
                continue;
            }
            hipdnnBackendDescriptor_t rawEngine = nullptr;
            checkHipdnn(hipdnnBackend()->backendGetAttribute(rawConfigs[i],
                                                             HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                             1,
                                                             nullptr,
                                                             static_cast<void*>(&rawEngine)),
                        "reading the engine of a hipDNN engine config");
            ScopedHipdnnBackendDescriptor engine(rawEngine);
            std::int64_t id = 0;
            checkHipdnn(hipdnnBackend()->backendGetAttribute(engine.get(),
                                                             HIPDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                                             HIPDNN_TYPE_INT64,
                                                             1,
                                                             nullptr,
                                                             static_cast<void*>(&id)),
                        "reading a hipDNN engine ID");
            if(!seen.insert(id).second)
            {
                continue;
            }
            space.rankedIds.push_back(id);
            space.byId.push_back(Engine{id, std::move(configs[i]), std::move(engine)});
        }
        std::sort(space.byId.begin(), space.byId.end(), [](const Engine& a, const Engine& b) {
            return a.id < b.id;
        });
        return space;
    }

    hipdnnHandle_t _handle;
    hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor _graph;
    std::string _tag;
    // Declared after _graph so the cached configs and engines are released
    // before the graph they refer to.
    mutable std::once_flag _engineOnce;
    mutable EngineSpace _engines;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
