// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/RockeClientDispatcher.hpp"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "RockeClientHandle.hpp"
#include "dispatcher/SdpaGraphAdapter.hpp"
#include "dispatcher/SelectionConstraints.hpp"

namespace rocke_client::dispatcher
{

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::flatbuffer_utilities;

// Bare gfx arch string (e.g. "gfx942") for the stream's device, or "" when no
// device is resolvable (e.g. host-only unit tests). Only ever called inside
// selectInstance's try/catch, so a std::bad_alloc from the small string build
// is handled there rather than escaping the noexcept selection path.
std::string deviceArch(hipStream_t stream)
{
    int device = 0;
    if(hipStreamGetDevice(stream, &device) != hipSuccess)
    {
        return {};
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
    {
        return {};
    }
    std::string arch = props.gcnArchName;
    const auto colon = arch.find(':'); // strip "gfx942:sramecc+:xnack-"
    if(colon != std::string::npos)
    {
        arch.resize(colon);
    }
    return arch;
}

// Emit a selection-failure warning without ever letting the log path throw: a
// std::bad_alloc from building the message must not escape the noexcept
// selection path and turn a graceful decline into std::terminate.
void logSelectionFailure(const char* reason) noexcept
{
    try
    {
        HIPDNN_PLUGIN_LOG_WARN("rocke-client dispatcher selection failed: " << reason);
    }
    // NOLINTNEXTLINE(bugprone-empty-catch) -- a failed log must never escape this noexcept path
    catch(...)
    {
    }
}

} // namespace

RockeClientDispatcher::RockeClientDispatcher(AotCatalog catalog)
    : _catalog(std::move(catalog))
{
}

std::optional<AotInstance> RockeClientDispatcher::select(const SdpaProblem& problem) const
{
    const auto candidates = _catalog.candidatesFor(problem.op, problem.arch);
    if(candidates.empty())
    {
        return std::nullopt;
    }

    // Build the runtime attribute view once and reuse it across candidates.
    const std::map<std::string, AttrValue> attributes = problem.attributes();

    // Collect ALL satisfying instances so a trained model can rank them; today,
    // absent a model + featurizer, the first (stable catalog order) still wins.
    std::vector<const AotInstance*> matches;
    for(const AotInstance& instance : candidates)
    {
        if(satisfies(instance, problem, attributes))
        {
            matches.push_back(&instance);
        }
    }
    if(matches.empty())
    {
        return std::nullopt;
    }

    // TODO(heuristics): model-scored tie-break. The registry is now generated
    // (rocke_model_registry.h: rocke_lookup_model(op, arch, dtype) ->
    // {score, num_features}). To wire it here:
    //   const RockeModelEntry* model = rocke_lookup_model(
    //       problem.op.c_str(), problem.arch.c_str(), problem.dtype.c_str());
    //   if(model && matches.size() > 1) {
    //       // DRIFT GUARD: refuse to score if the predictor's width disagrees
    //       // with the op featurizer's output -- fall through to first-match.
    //       if(model->num_features == fmhaFeatureCount()) {
    //           return *argmax(matches, [&](const AotInstance* inst) {
    //               auto f = featurize(problem, *inst); // <-- the remaining piece
    //               return model->score(f.data());
    //           });
    //       }
    //   }
    // The blocker is featurize(SdpaProblem, AotInstance) -> feature_spec.json-
    // ordered vector (the C++ half of the feature contract, part of #8866): the
    // AotInstance must carry the tiling knobs (tileSize/numWarps) the model ranks
    // on. Until it lands, first-match preserves Phase-1 behaviour.
    return *matches.front();
}

std::optional<AotInstance>
    RockeClientDispatcher::selectForArch(const std::string& arch,
                                         const fb::IGraph& graph) const noexcept
{
    try
    {
        std::optional<SdpaProblem> problem = translate(graph);
        if(!problem.has_value())
        {
            return std::nullopt;
        }
        problem->arch = arch;
        return select(*problem);
    }
    catch(const std::exception& e)
    {
        logSelectionFailure(e.what());
        return std::nullopt;
    }
    catch(...)
    {
        logSelectionFailure("unknown error");
        return std::nullopt;
    }
}

std::optional<AotInstance>
    RockeClientDispatcher::selectInstance(const RockeClientHandle& handle,
                                          const fb::IGraph& graph) const noexcept
{
    // deviceArch() builds a std::string and may throw std::bad_alloc; guard it so
    // nothing escapes this noexcept function (selectForArch is itself noexcept).
    try
    {
        return selectForArch(deviceArch(handle.getStream()), graph);
    }
    catch(...)
    {
        return std::nullopt;
    }
}

bool RockeClientDispatcher::isApplicable(const RockeClientHandle& handle,
                                         const fb::IGraph& graph) const noexcept
{
    return selectInstance(handle, graph).has_value();
}

} // namespace rocke_client::dispatcher
