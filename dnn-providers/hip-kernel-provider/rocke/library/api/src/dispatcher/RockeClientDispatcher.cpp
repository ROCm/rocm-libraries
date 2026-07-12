// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/RockeClientDispatcher.hpp"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <array>

#include "RockeClientHandle.hpp"
#include "dispatcher/HardwareProfile.hpp"
#include "dispatcher/SdpaGraphAdapter.hpp"
#include "dispatcher/SelectionConstraints.hpp"
#include "dispatcher/sdpa_fwd/FmhaFeaturizer.hpp"
#include "dispatcher/sdpa_fwd/rocke_model_registry.h"

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

// Build the exact FMHA feature vector for (problem, candidate instance). Problem
// fields come from SdpaProblem; the config/tiling knobs the model ranks on come
// from the instance's CompileSpec (blockSizeQ->tile_m0, tileSize->tile_n0,
// numWarps, headSize->tile_k0 group). Knobs not carried on the base instance
// keep the featurizer's own defaults today (arch-specific instance subclasses
// will supply them later; TiledSpecDefaults.hpp holds the per-arch values).
FmhaFeatures featurizeFmha(const SdpaProblem& problem, const AotInstance& inst)
{
    const CompileSpec& cs = inst.compileSpec;

    FmhaProblemInputs p;
    p.batch = static_cast<double>(problem.batch);
    p.sq    = static_cast<double>(problem.seqlenQ);
    p.sk    = static_cast<double>(problem.seqlenK);
    p.hq    = static_cast<double>(problem.numQueryHeads);
    p.hk    = static_cast<double>(problem.numKvHeads);
    p.dq    = static_cast<double>(problem.headSize);
    p.dv    = static_cast<double>(problem.headSize); // single head dim today
    p.dtype = problem.dtype;

    FmhaConfigInputs c;
    c.tm0       = static_cast<double>(cs.blockSizeQ); // block_m_per_warp
    c.tn0       = static_cast<double>(cs.tileSize);   // 2D tile width T
    c.num_warps = static_cast<double>(cs.numWarps);
    // tk0/tn1/tk1/tk0max default from head_size/tn0 inside the featurizer (0 ->
    // derived), matching extract()'s defaults. Variant flags (mask/bias/...) are
    // not carried per-instance yet; they keep featurizer defaults (mask=0 etc.),
    // consistent with the mask=none sweep the models are trained on.

    FmhaHwInputs hw;
    hw.num_cus        = static_cast<double>(problem.hw.num_cus);
    hw.simds_per_cu   = static_cast<double>(problem.hw.simds_per_cu);
    hw.total_simds    = static_cast<double>(problem.hw.total_simds());
    hw.shader_engines = static_cast<double>(problem.hw.shader_engines);
    hw.max_clock_mhz  = static_cast<double>(problem.hw.max_clock_mhz);
    hw.wavefront_size = static_cast<double>(problem.hw.wavefront_size);
    hw.lds_capacity   = static_cast<double>(problem.hw.lds_capacity);
    hw.num_xcd        = static_cast<double>(problem.hw.num_xcd);

    return fmha_featurize(p, c, hw);
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

    // Single satisfying instance: nothing to rank, skip the model entirely.
    if(matches.size() == 1)
    {
        return *matches.front();
    }

    // Model-scored tie-break. Look up the predictor for (op, arch, dtype); if
    // none is registered, keep Phase-1 first-match. The featurizer produces the
    // exact feature_spec vector the model trained on (bit-identical to the Python
    // engine -- see test_fmha_featurizer_roundtrip.py).
    const RockeModelEntry* model =
        rocke_lookup_model(problem.op.c_str(), problem.arch.c_str(), problem.dtype.c_str());
    if(model == nullptr || model->score == nullptr)
    {
        return *matches.front();
    }

    // DRIFT GUARD: a predictor built against a different feature count than the
    // featurizer produces must NOT be scored (it would read a wrong-width
    // vector). Fall back to first-match, loudly-safe rather than silently wrong.
    if(model->num_features != FmhaFeatures::kNumFeatures)
    {
        logSelectionFailure("model num_features != featurizer output; first-match");
        return *matches.front();
    }

    // Argmax the model score over the satisfying instances. Ties broken by stable
    // catalog order (>= keeps the earliest on equal score).
    const AotInstance* best = matches.front();
    double bestScore = -1e300;
    for(const AotInstance* inst : matches)
    {
        const FmhaFeatures feats = featurizeFmha(problem, *inst);
        const std::array<double, FmhaFeatures::kNumFeatures> arr = feats.to_array();
        const double s = model->score(arr.data());
        if(s > bestScore)
        {
            bestScore = s;
            best = inst;
        }
    }
    return *best;
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
        // Fill group-C hardware features from the live device (policy: no CU
        // counts in source). Zero profile on host-only test paths (no GPU) --
        // selection is unaffected since satisfies() ignores hw.
        problem->hw = HardwareProfile::fromDevice();
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
