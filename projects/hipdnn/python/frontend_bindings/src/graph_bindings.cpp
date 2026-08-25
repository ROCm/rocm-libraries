// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleBwdAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#ifdef HIPDNN_ENABLE_SDPA
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#endif
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <utility>

namespace nb = nanobind;
using namespace hipdnn_frontend;

namespace
{

// Unwraps the optional handle taken by the engine-name-reporting bindings. Python cannot
// overload, so where C++ offers a handle-taking form alongside a handle-free one those
// bindings take a single trailing handle defaulting to None. None yields a null handle,
// which Graph resolves from the built-in registry.
hipdnnHandle_t optionalRawHandle(const nb::object& handle)
{
    if(handle.is_none())
    {
        return nullptr;
    }

    auto handlePtr = handle.attr("get")();
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));
}

// Shared body for both autotune() bindings. The UID-keyed and tensor-keyed variant
// packs differ only in their key type, which Graph::autotune() itself overloads on.
// Pointers cross the boundary as integers, exactly as the execute() binding does.
template <typename KeyT>
std::vector<AutotuneResult> autotunePy(graph::Graph& g,
                                       const nb::object& handle,
                                       const std::unordered_map<KeyT, uintptr_t>& variantPack,
                                       uintptr_t workspace,
                                       std::optional<int64_t> workspaceSize,
                                       const AutotuneConfig& config,
                                       const AutotuneStorageConfig& storageConfig)
{
    auto handlePtr = handle.attr("get")();
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto rawHandle = reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));

    std::unordered_map<KeyT, void*> cppVariantPack;
    cppVariantPack.reserve(variantPack.size());
    for(const auto& [key, value] : variantPack)
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        cppVariantPack[key] = reinterpret_cast<void*>(value);
    }

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* workspacePtr = workspace ? reinterpret_cast<void*>(workspace) : nullptr;

    // Benchmarking touches no Python object: config/settings are native types, and
    // AutotuneConfig::rankingFn (the one field that could call back into Python) is
    // deliberately unbound, so nothing here needs to reacquire the GIL mid-loop. This
    // can run for minutes across many candidates, so release it for the duration:
    // other Python threads proceed, and a pending Ctrl-C is delivered promptly
    // instead of waiting for the whole benchmarking loop to finish.
    std::vector<AutotuneResult> results;
    Error err;
    {
        const nb::gil_scoped_release release;
        err = workspaceSize
                  ? g.autotune(rawHandle,
                               cppVariantPack,
                               workspacePtr,
                               *workspaceSize,
                               config,
                               storageConfig,
                               &results)
                  : g.autotune(
                        rawHandle, cppVariantPack, workspacePtr, config, storageConfig, &results);
    }
    if(err.is_bad())
    {
        throw std::runtime_error("Autotune failed: " + err.get_message());
    }
    return results;
}

// Shared body for the autotune_exhaustive_sweep() binding. Forces exhaustive mode and
// benchmarking, accepts no sweep/variant parameter, and returns both the per-engine
// results and the exact-match cache write outcome as a tuple, since nanobind does not
// expose C++ out-parameters to Python.
//
// Only a UID-keyed variant pack is bound, because Graph::autotuneExhaustiveSweep()
// itself takes only that form.
std::pair<std::vector<AutotuneResult>, AutotuneCacheWriteOutcome>
    autotuneExhaustiveSweepPy(graph::Graph& g,
                              const nb::object& handle,
                              const std::unordered_map<int64_t, uintptr_t>& variantPack,
                              uintptr_t workspace,
                              int64_t workspaceSize,
                              const AutotuneConfig& config,
                              const AutotuneStorageConfig& storageConfig)
{
    auto handlePtr = handle.attr("get")();
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto rawHandle = reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));

    std::unordered_map<int64_t, void*> cppVariantPack;
    cppVariantPack.reserve(variantPack.size());
    for(const auto& [key, value] : variantPack)
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        cppVariantPack[key] = reinterpret_cast<void*>(value);
    }

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* workspacePtr = workspace != 0U ? reinterpret_cast<void*>(workspace) : nullptr;

    // Same rationale as autotunePy: benchmarking touches no Python object, so the GIL
    // can be released for the duration of what may be a minutes-long call.
    std::vector<AutotuneResult> results;
    AutotuneCacheWriteOutcome outcome
        = AutotuneCacheWriteOutcome::NOT_ATTEMPTED_NO_SUCCESSFUL_ENGINE;
    Error err;
    {
        const nb::gil_scoped_release release;
        err = g.autotuneExhaustiveSweep(rawHandle,
                                        cppVariantPack,
                                        workspacePtr,
                                        workspaceSize,
                                        config,
                                        storageConfig,
                                        &results,
                                        &outcome);
    }
    if(err.is_bad())
    {
        throw std::runtime_error("autotune_exhaustive_sweep failed: " + err.get_message());
    }
    return {results, outcome};
}

} // namespace

void graphBindings(nb::module_& m)
{
    nb::class_<graph::Graph>(m, "Graph")
        .def(nb::init<>())
        .def("validate", &graph::Graph::validate)
        .def("checkNoDuplicateTensorIds", &graph::Graph::checkNoDuplicateTensorIds)
        .def("topologicallySortGraph", &graph::Graph::topologicallySortGraph)
        .def(
            "build_operation_graph",
            [](graph::Graph& g, const nb::object& handle) {
                auto handlePtr = handle.attr("get")();
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                auto rawHandle = reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));
                return g.build_operation_graph(rawHandle);
            },
            nb::arg("handle"),
            "Build the operation graph with the given handle")
        .def("create_execution_plans",
             &graph::Graph::create_execution_plans,
             nb::arg("modes") = std::vector<HeuristicMode>{HeuristicMode::FALLBACK},
             "Create execution plans with specified heuristic modes")
        .def("create_execution_plan_ext",
             &graph::Graph::create_execution_plan_ext,
             nb::arg("engine_id"),
             nb::arg("knob_settings") = std::vector<KnobSetting>{},
             "Hard-select an engine: create/select the execution plan descriptor for "
             "this exact engine id, optionally overriding its knobs (call build_plans() "
             "to finalize). Knobs the engine does not expose are ignored with a warning; "
             "a value that violates a knob's constraint, or an engine that is not "
             "valid/applicable, returns an Error whose is_bad() is set (no heuristic "
             "fallback).")
        .def(
            "get_execution_plan_engine_id",
            [](const graph::Graph& g) {
                int64_t engineId = 0;
                const auto err = g.get_execution_plan_engine_id(engineId);
                if(err.is_bad())
                {
                    throw std::runtime_error("Failed to get execution plan engine id: "
                                             + err.get_message());
                }
                return engineId;
            },
            "Engine id actually backing the built execution plan (ground truth; "
            "detects a silent soft-preference fallback).")
        .def(
            "get_ranked_engine_ids",
            [](graph::Graph& g, const std::vector<HeuristicMode>& modes) {
                std::vector<int64_t> ids;
                const auto err = g.get_ranked_engine_ids(ids, modes);
                if(err.is_bad())
                {
                    throw std::runtime_error("Failed to get ranked engine ids: "
                                             + err.get_message());
                }
                return ids;
            },
            nb::arg("modes") = std::vector<HeuristicMode>{HeuristicMode::FALLBACK},
            "Get ranked engine IDs for the built operation graph. Requires "
            "build_operation_graph() to have been called first.")
        .def(
            "get_behavior_notes_for_engine",
            [](graph::Graph& g, int64_t engineId) {
                std::vector<BehaviorNote> notes;
                auto err = g.get_behavior_notes_for_engine(engineId, notes);
                if(err.is_bad())
                {
                    throw std::runtime_error("Failed to get behavior notes for engine: "
                                             + err.get_message());
                }
                std::vector<BehaviorNote> knownNotes;
                knownNotes.reserve(notes.size());
                for(auto note : notes)
                {
                    if(isKnownBehaviorNote(note))
                    {
                        knownNotes.push_back(note);
                    }
                }
                return knownNotes;
            },
            nb::arg("engine_id"),
            "Get behavior notes for an engine applicable to the built operation graph.")
        .def("check_support", &graph::Graph::check_support)
        .def("build_plans",
             &graph::Graph::build_plans,
             nb::arg("policy") = BuildPlanPolicy::HEURISTICS_CHOICE,
             "Finalize execution plans. HEURISTICS_CHOICE compiles only the "
             "active plan; ALL compiles every plan.")
        .def(
            "get_workspace_size",
            [](const graph::Graph& g) {
                int64_t workspaceSize = 0;
                const auto result = g.get_workspace_size(workspaceSize);
                if(!result.is_good())
                {
                    throw std::runtime_error("Failed to get workspace size: "
                                             + result.get_message());
                }
                return workspaceSize;
            },
            "Get the workspace size required for graph execution")
        .def(
            "execute",
            [](const graph::Graph& g,
               const nb::object& handle,
               std::unordered_map<int64_t, uintptr_t>& variantPack,
               uintptr_t workspace) {
                auto handlePtr = handle.attr("get")();
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                auto rawHandle = reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));

                std::unordered_map<int64_t, void*> cppVariantPack;
                for(const auto& [key, value] : variantPack)
                {
                    // NOLINTNEXTLINE(performance-no-int-to-ptr)
                    cppVariantPack[key] = reinterpret_cast<void*>(value);
                }

                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                void* workspacePtr = workspace ? reinterpret_cast<void*>(workspace) : nullptr;

                return g.execute(rawHandle, cppVariantPack, workspacePtr);
            },
            nb::arg("handle"),
            nb::arg("variant_pack"),
            nb::arg("workspace") = 0,
            "Execute the graph with the given handle, variant pack, and optional workspace")
        .def("get_execution_plan_count",
             &graph::Graph::get_execution_plan_count,
             "Number of compiled plans, including ones that failed to compile. Use with "
             "build_plans(BuildPlanPolicy.ALL) to drive a manual per-plan tuning loop.")
        .def("get_autotune_workspace_size",
             &graph::Graph::get_autotune_workspace_size,
             "Largest workspace across the compiled, non-barred plans, or 0 when none "
             "compiled. This is the compiled-plan counterpart of "
             "get_estimated_max_workspace_size(), which covers the plan-spec path and "
             "errors when no add_engine_*() spec was added.")
        .def(
            "get_plan_name",
            [](const graph::Graph& g, const nb::object& handle) {
                std::string name;
                const auto err = g.get_plan_name(optionalRawHandle(handle), name);
                if(err.is_bad())
                {
                    throw std::runtime_error("Failed to get plan name: " + err.get_message());
                }
                return name;
            },
            nb::arg("handle") = nb::none(),
            "Engine name of the active plan, for example the autotune winner. Pass the "
            "handle the graph was built with to name plugin-supplied engines; without it "
            "only the built-in registry is consulted and such engines report a hex ID. "
            "Raises RuntimeError when no plan is active.")
        .def(
            "execute_plan_at_index",
            [](const graph::Graph& g,
               const nb::object& handle,
               const std::unordered_map<int64_t, uintptr_t>& variantPack,
               uintptr_t workspace,
               int64_t planIndex) {
                auto handlePtr = handle.attr("get")();
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                auto rawHandle = reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));

                std::unordered_map<int64_t, void*> cppVariantPack;
                cppVariantPack.reserve(variantPack.size());
                for(const auto& [key, value] : variantPack)
                {
                    // NOLINTNEXTLINE(performance-no-int-to-ptr)
                    cppVariantPack[key] = reinterpret_cast<void*>(value);
                }

                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                void* workspacePtr = workspace ? reinterpret_cast<void*>(workspace) : nullptr;

                return g.execute_plan_at_index(rawHandle, cppVariantPack, workspacePtr, planIndex);
            },
            nb::arg("handle"),
            nb::arg("variant_pack"),
            nb::arg("workspace"),
            nb::arg("plan_index"),
            "Execute one compiled plan without making it active. Returns an Error whose "
            "is_bad() is set for an out-of-bounds, barred or uncompiled plan, so a manual "
            "tuning loop can skip it and continue.")
        .def(
            "get_workspace_size_plan_at_index",
            [](const graph::Graph& g, int64_t planIndex) {
                int64_t size = 0;
                const auto err = g.get_workspace_size_plan_at_index(planIndex, size);
                if(err.is_bad())
                {
                    throw std::runtime_error("Failed to get workspace size for plan: "
                                             + err.get_message());
                }
                return size;
            },
            nb::arg("plan_index"),
            "Workspace size cached for one compiled plan. Raises RuntimeError for an "
            "out-of-bounds index; the C++ overload that reports -1 instead is not bound "
            "separately, since the exception already separates that case from a zero size.")
        .def(
            "get_plan_name_at_index",
            [](const graph::Graph& g, int64_t planIndex, const nb::object& handle) {
                std::string name;
                const auto err
                    = g.get_plan_name_at_index(optionalRawHandle(handle), planIndex, name);
                if(err.is_bad())
                {
                    throw std::runtime_error("Failed to get plan name: " + err.get_message());
                }
                return name;
            },
            nb::arg("plan_index"),
            nb::arg("handle") = nb::none(),
            "Engine name backing one compiled plan. Pass the handle the graph was built "
            "with to name plugin-supplied engines; without it only the built-in registry "
            "is consulted and such engines report a hex ID.")
        .def("build_plan_at_index",
             &graph::Graph::build_plan_at_index,
             nb::arg("plan_index"),
             "Compile if needed and make the plan at this index active, so the following "
             "execute() uses it. Returns an Error whose is_bad() is set for an "
             "out-of-bounds, barred or invalid plan.")
        .def(
            "get_engine_configs",
            [](graph::Graph& g,
               const std::vector<HeuristicMode>& modes,
               const nb::object& handle) {
                std::vector<EngineConfigInfo> configs;
                const auto err = g.get_engine_configs(optionalRawHandle(handle), configs, modes);
                if(err.is_bad())
                {
                    throw std::runtime_error("Failed to get engine configs: " + err.get_message());
                }
                return configs;
            },
            nb::arg("modes") = std::vector<HeuristicMode>{HeuristicMode::FALLBACK},
            nb::arg("handle") = nb::none(),
            "Discover engine metadata (id, name, knobs, workspace estimate) for the built "
            "operation graph. Requires build_operation_graph() to have been called first. "
            "Pass the handle the graph was built with to name plugin-supplied engines; "
            "without it engine_name falls back to the built-in registry and reports a hex "
            "ID for engines it does not carry.")
        .def(
            "get_knobs_for_engine",
            [](const graph::Graph& g, int64_t engineId) {
                std::vector<Knob> knobs;
                const auto err = g.get_knobs_for_engine(engineId, knobs);
                if(err.is_bad())
                {
                    throw std::runtime_error("Failed to get knobs for engine: "
                                             + err.get_message());
                }
                return knobs;
            },
            nb::arg("engine_id"),
            "Get the tunable knobs an engine exposes for the built operation graph.")
        .def(
            "get_knob_lookup_for_engine",
            [](const graph::Graph& g, int64_t engineId) {
                std::unordered_map<KnobType_t, Knob> knobs;
                const auto err = g.get_knob_lookup_for_engine(engineId, knobs);
                if(err.is_bad())
                {
                    throw std::runtime_error("Failed to get knob lookup for engine: "
                                             + err.get_message());
                }
                return knobs;
            },
            nb::arg("engine_id"),
            "Same knobs as get_knobs_for_engine(), keyed by knob id for direct lookup.")
        .def(
            "get_estimated_max_workspace_size",
            [](const graph::Graph& g) {
                int64_t maxSize = 0;
                const auto err = g.get_estimated_max_workspace_size(maxSize);
                if(err.is_bad())
                {
                    throw std::runtime_error("Failed to get estimated max workspace size: "
                                             + err.get_message());
                }
                return maxSize;
            },
            "Maximum workspace estimate across the plan specs added by add_engine_*(). "
            "Allocate this much workspace before calling autotune().")
        .def("add_engine_configs",
             &graph::Graph::add_engine_configs,
             nb::arg("configs"),
             "Add engine configs (from get_engine_configs()) as autotune plan specs.")
        .def("add_engine",
             &graph::Graph::add_engine,
             nb::arg("engine_id"),
             nb::arg("knob_settings") = std::vector<KnobSetting>{},
             "Add one engine with optional knob settings as an autotune plan spec.")
        .def("add_engines",
             &graph::Graph::add_engines,
             nb::arg("engine_ids"),
             "Add several engines, each with its default knob settings, as plan specs.")
        .def("add_engine_variants",
             &graph::Graph::add_engine_variants,
             nb::arg("variants"),
             "Add one plan spec per EngineVariant (engine id plus explicit knob settings).")
        .def("add_engine_sweep",
             &graph::Graph::add_engine_sweep,
             nb::arg("specs"),
             "Add the Cartesian product of each EngineSweepSpec's knob axes as plan specs.")
        .def("add_all_engines",
             &graph::Graph::add_all_engines,
             nb::arg("modes") = std::vector<HeuristicMode>{HeuristicMode::FALLBACK},
             "Add every engine applicable to the built operation graph as a plan spec.")
        .def("deselect_workspace_greater_than",
             &graph::Graph::deselect_workspace_greater_than,
             nb::arg("workspace"),
             nb::rv_policy::reference_internal,
             "Bar plans whose compiled workspace exceeds this many bytes.")
        .def("deselect_engines",
             nb::overload_cast<const std::vector<std::string>&>(&graph::Graph::deselect_engines),
             nb::arg("engine_names"),
             nb::rv_policy::reference_internal,
             "Bar plans belonging to these engine names.")
        .def("deselect_engines",
             nb::overload_cast<const std::vector<int64_t>&>(&graph::Graph::deselect_engines),
             nb::arg("engine_ids"),
             nb::rv_policy::reference_internal,
             "Bar plans belonging to these engine IDs.")
        .def("autotune",
             &autotunePy<int64_t>,
             nb::arg("handle"),
             nb::arg("variant_pack"),
             nb::arg("workspace") = 0,
             nb::arg("workspace_size") = nb::none(),
             nb::arg("config") = AutotuneConfig{},
             nb::arg("storage_config") = AutotuneStorageConfig{},
             "Benchmark autotune candidates and return one AutotuneResult per candidate.\n"
             "variant_pack maps either tensor UIDs or tensors to device pointers.\n"
             "Pass workspace_size for the plan-spec path added via add_engine_*(), sized "
             "with get_estimated_max_workspace_size(); omit it for the compiled-plan path "
             "built with build_plans(BuildPlanPolicy.ALL), sized with "
             "get_autotune_workspace_size(). The winning plan is left active, so a "
             "following execute() uses it, and get_plan_name() reports it. Raises "
             "RuntimeError if autotuning fails.\n"
             "The two C++ overloads taking a trailing userImpl pointer are not bound: that "
             "argument exists for cuDNN signature compatibility and is ignored, so calling "
             "this method covers them.")
        .def("autotune",
             &autotunePy<std::shared_ptr<graph::TensorAttributes>>,
             nb::arg("handle"),
             nb::arg("variant_pack"),
             nb::arg("workspace") = 0,
             nb::arg("workspace_size") = nb::none(),
             nb::arg("config") = AutotuneConfig{},
             nb::arg("storage_config") = AutotuneStorageConfig{},
             "autotune() overload taking a tensor-keyed variant pack.")
        .def("autotune_exhaustive_sweep",
             &autotuneExhaustiveSweepPy,
             nb::arg("handle"),
             nb::arg("variant_pack"),
             nb::arg("workspace"),
             nb::arg("workspace_size"),
             nb::arg("config") = AutotuneConfig{},
             nb::arg("storage_config") = AutotuneStorageConfig{},
             "Benchmarks every candidate engine for this graph and caches the measured "
             "ranking.\n"
             "Compiles and times every plan spec added via "
             "add_engine_configs()/add_all_engines()/add_engine(), ranks the engines by "
             "measured speed, and writes that order to the exact-match autotune cache. "
             "Takes no sweep/variant argument: populate candidates before calling.\n"
             "This is a FULL sweep, meant to run once so every later run of the same graph "
             "on the same device inherits the order. The cache record is keyed on the graph "
             "and device only -- not on knobs, filters, or the workspace budget -- and is "
             "consulted by every later run of that graph however it is configured. A "
             "ranking is therefore only persisted when the sweep covered every engine a "
             "later run could see, which means two things:\n"
             "  1. One candidate per engine, no knob variants. Use add_engine_configs(), "
             "add_all_engines(), or add_engine() with no knob settings. A record stores "
             "engine ids, so several plan specs for one engine (add_engine_sweep(), "
             "add_engine_variants(), or repeated add_engine() with different knobs) "
             "collapse to one id: the best-ranked variant wins and the winning knob "
             "settings are not recorded. Sweep knobs with autotune() instead, which keeps "
             "them on the active plan.\n"
             "  2. No filtering, and a workspace large enough for every candidate. Leave "
             "config.engine_id_filter empty, apply no deselect filter, and pass a "
             "workspace_size at least as large as get_autotune_workspace_size() reports. "
             "Each of these can hold an applicable engine out of the timing loop while "
             "leaving it a live candidate for a later unrestricted run -- an engine the "
             "ranking never measured. Rather than persist a record that is wrong or "
             "unusable, this call declines the write and reports "
             "AutotuneCacheWriteOutcome.NOT_ATTEMPTED_PARTIAL_SWEEP. The sweep still runs "
             "and still returns its results; only the cache write is skipped.\n"
             "Engines are ranked on AutotuneResult.robust_time_ms -- the mean of a "
             "candidate's iterations after discarding its slow tail -- so the order "
             "reflects what each engine usually costs. A candidate that is normally slower "
             "but occasionally lucky does not outrank a consistently faster one on the "
             "strength of a single good iteration.\n"
             "Two config fields are set by this call and ignored on input: mode "
             "(TuneMode.EXHAUSTIVE) and the priming-failure policy (benchmark unprimed). "
             "Every other field is honoured as given, including strategy.\n"
             "More iterations improve the odds that the genuinely fastest engine ranks "
             "first, at a proportional cost in sweep time. Engines whose true times differ "
             "by a wide margin separate reliably at any count; engines within a few percent "
             "of each other need substantially more iterations to order correctly, because "
             "each measurement carries run-to-run jitter of a similar size. Which dial "
             "applies depends on the strategy you set: config.timed_iterations is the exact "
             "count per candidate under AutotuneStrategy.FIXED_AVERAGE, and is unused under "
             "the default AutotuneStrategy.RUN_UNTIL_STABLE, which instead runs until the "
             "trailing-window variation falls below config.stability_threshold or "
             "config.max_iterations is reached. Set config.strategy = "
             "AutotuneStrategy.FIXED_AVERAGE before raising config.timed_iterations, or the "
             "raise has no effect.\n"
             "The cached order is keyed on this graph's serialized content and the "
             "handle's device. Returns a (results, cache_write_outcome) tuple: the "
             "per-engine AutotuneResult list and the AutotuneCacheWriteOutcome reporting "
             "whether that write happened. Raises RuntimeError if the sweep fails.")
        .def("set_name", &graph::Graph::set_name, nb::rv_policy::reference_internal)
        .def("set_compute_data_type",
             &graph::Graph::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("set_intermediate_data_type",
             &graph::Graph::set_intermediate_data_type,
             nb::rv_policy::reference_internal)
        .def("set_io_data_type", &graph::Graph::set_io_data_type, nb::rv_policy::reference_internal)
        .def("batchnorm", &graph::Graph::batchnorm)
        .def("batchnorm_backward", &graph::Graph::batchnorm_backward)
        .def("batchnorm_inference",
             &graph::Graph::batchnorm_inference,
             nb::arg("x"),
             nb::arg("mean"),
             nb::arg("inv_variance"),
             nb::arg("scale"),
             nb::arg("bias"),
             nb::arg("attributes"))
        .def(
            "pointwise",
            nb::overload_cast<std::shared_ptr<graph::TensorAttributes>, graph::PointwiseAttributes>(
                &graph::Graph::pointwise))
        .def("pointwise",
             nb::overload_cast<std::shared_ptr<graph::TensorAttributes>,
                               std::shared_ptr<graph::TensorAttributes>,
                               graph::PointwiseAttributes>(&graph::Graph::pointwise))
        .def("pointwise",
             nb::overload_cast<std::shared_ptr<graph::TensorAttributes>,
                               std::shared_ptr<graph::TensorAttributes>,
                               std::shared_ptr<graph::TensorAttributes>,
                               graph::PointwiseAttributes>(&graph::Graph::pointwise))
        .def("conv_fprop", &graph::Graph::conv_fprop)
        .def("matmul", &graph::Graph::matmul)
        .def("conv_dgrad", &graph::Graph::conv_dgrad)
        .def("conv_wgrad", &graph::Graph::conv_wgrad)
        .def("batchnorm_inference_variance_ext", &graph::Graph::batchnorm_inference_variance_ext)
        .def("layernorm",
             [](graph::Graph& graph,
                std::shared_ptr<graph::TensorAttributes> x,
                std::shared_ptr<graph::TensorAttributes> scale,
                std::shared_ptr<graph::TensorAttributes> bias,
                graph::LayernormAttributes attributes) {
                 const auto outputs = graph.layernorm(
                     std::move(x), std::move(scale), std::move(bias), std::move(attributes));
                 return nb::make_tuple(outputs[0], outputs[1], outputs[2]);
             })
        .def("layernorm_backward",
             [](graph::Graph& graph,
                std::shared_ptr<graph::TensorAttributes> dy,
                std::shared_ptr<graph::TensorAttributes> x,
                std::shared_ptr<graph::TensorAttributes> scale,
                graph::LayernormBackwardAttributes attributes) {
                 const auto outputs = graph.layernorm_backward(
                     std::move(dy), std::move(x), std::move(scale), std::move(attributes));
                 return nb::make_tuple(outputs[0], outputs[1], outputs[2]);
             })
        .def("rmsnorm",
             [](graph::Graph& graph,
                std::shared_ptr<graph::TensorAttributes> x,
                std::shared_ptr<graph::TensorAttributes> scale,
                graph::RMSNormAttributes attributes) {
                 const auto outputs
                     = graph.rmsnorm(std::move(x), std::move(scale), std::move(attributes));
                 return nb::make_tuple(outputs[0], outputs[1]);
             })
        .def("rmsnorm_backward",
             [](graph::Graph& graph,
                std::shared_ptr<graph::TensorAttributes> dy,
                std::shared_ptr<graph::TensorAttributes> x,
                std::shared_ptr<graph::TensorAttributes> scale,
                std::shared_ptr<graph::TensorAttributes> invRms,
                graph::RMSNormBackwardAttributes attributes) {
                 const auto outputs = graph.rmsnorm_backward(std::move(dy),
                                                             std::move(x),
                                                             std::move(scale),
                                                             std::move(invRms),
                                                             std::move(attributes));
                 return nb::make_tuple(outputs[0], outputs[1], outputs[2]);
             })
        .def("block_scale_dequantize", &graph::Graph::block_scale_dequantize)
        .def("block_scale_quantize",
             [](graph::Graph& graph,
                std::shared_ptr<graph::TensorAttributes> x,
                graph::BlockScaleQuantizeAttributes attributes) {
                 const auto outputs
                     = graph.block_scale_quantize(std::move(x), std::move(attributes));
                 return nb::make_tuple(outputs[0], outputs[1]);
             })
        .def(
            "reduction",
            nb::overload_cast<std::shared_ptr<graph::TensorAttributes>, graph::ReductionAttributes>(
                &graph::Graph::reduction))
        .def("reduction",
             nb::overload_cast<std::shared_ptr<graph::TensorAttributes>,
                               std::shared_ptr<graph::TensorAttributes>,
                               graph::ReductionAttributes>(&graph::Graph::reduction))
        .def("moe_grouped_matmul", &graph::Graph::moe_grouped_matmul)
        .def("custom_op", &graph::Graph::custom_op)
#ifdef HIPDNN_ENABLE_SDPA
        .def("sdpa",
             [](graph::Graph& graph,
                std::shared_ptr<graph::TensorAttributes> q,
                std::shared_ptr<graph::TensorAttributes> k,
                std::shared_ptr<graph::TensorAttributes> v,
                graph::SdpaAttributes attributes) {
                 const auto outputs
                     = graph.sdpa(std::move(q), std::move(k), std::move(v), std::move(attributes));
                 return nb::make_tuple(outputs[0], outputs[1]);
             })
        .def("sdpa_backward",
             [](graph::Graph& graph,
                std::shared_ptr<graph::TensorAttributes> q,
                std::shared_ptr<graph::TensorAttributes> k,
                std::shared_ptr<graph::TensorAttributes> v,
                std::shared_ptr<graph::TensorAttributes> o,
                std::shared_ptr<graph::TensorAttributes> dO,
                std::shared_ptr<graph::TensorAttributes> stats,
                graph::SdpaBackwardAttributes attributes) {
                 const auto outputs = graph.sdpa_backward(std::move(q),
                                                          std::move(k),
                                                          std::move(v),
                                                          std::move(o),
                                                          std::move(dO),
                                                          std::move(stats),
                                                          std::move(attributes));
                 return nb::make_tuple(outputs[0], outputs[1], outputs[2]);
             })
#endif
        .def("resample",
             [](graph::Graph& graph,
                std::shared_ptr<graph::TensorAttributes> x,
                graph::ResampleFwdAttributes attributes) {
                 const auto outputs = graph.resample(std::move(x), std::move(attributes));
                 return nb::make_tuple(outputs[0], outputs[1]);
             })
        .def("resample_fwd", &graph::Graph::resample_fwd)
        .def("resample_bwd",
             &graph::Graph::resample_bwd,
             nb::arg("dy"),
             nb::arg("attributes"),
             nb::arg("index") = nb::none())
        .def("set_preferred_engine_id_ext",
             nb::overload_cast<std::optional<int64_t>>(&graph::Graph::set_preferred_engine_id_ext),
             nb::arg("engineId") = std::nullopt,
             nb::rv_policy::reference_internal)
        .def("set_preferred_engine_id_ext",
             nb::overload_cast<const std::string&>(&graph::Graph::set_preferred_engine_id_ext),
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::Graph::get_name)
        .def("get_compute_data_type", &graph::Graph::get_compute_data_type)
        .def("get_intermediate_data_type", &graph::Graph::get_intermediate_data_type)
        .def("get_io_data_type", &graph::Graph::get_io_data_type)
        .def("get_preferred_engine_id_ext", &graph::Graph::get_preferred_engine_id_ext)
        .def_static("tensor",
                    static_cast<std::shared_ptr<graph::TensorAttributes> (*)(
                        const graph::TensorAttributes&)>(&graph::Graph::tensor),
                    nb::arg("tensor"),
                    nb::rv_policy::reference)
        .def_static(
            "tensor_like", &graph::Graph::tensor_like, nb::arg("tensor"), nb::arg("name") = "")
        .def(
            "to_json",
            [](graph::Graph& g) {
                auto [json, err] = g.to_json();
                if(err.is_bad())
                {
                    throw std::runtime_error(err.get_message());
                }
                return json;
            },
            "Serialize the graph to a JSON string")
        .def(
            "from_json",
            [](graph::Graph& g, const nb::object& handle, const std::string& jsonStr) {
                auto handlePtr = handle.attr("get")();
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                auto rawHandle = reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));
                return g.deserialize(rawHandle, jsonStr);
            },
            nb::arg("handle"),
            nb::arg("json_string"),
            "Deserialize and finalize graph from JSON with a backend handle.\n"
            "The graph is ready for create_execution_plans() after this call.")
        .def(
            "from_json",
            [](graph::Graph& g, const std::string& jsonStr) { return g.deserialize(jsonStr); },
            nb::arg("json_string"),
            "Deserialize graph structure from JSON without a handle.\n"
            "Only restores the graph topology and attributes (nodes, tensors, parameters).\n"
            "Call build_operation_graph(handle) after to finalize for execution.")
        .def(
            "to_binary",
            [](graph::Graph& g) {
                auto [data, err] = g.to_binary();
                if(err.is_bad())
                {
                    throw std::runtime_error(err.get_message());
                }
                return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
            },
            "Serialize the graph to binary bytes")
        .def(
            "from_binary",
            [](graph::Graph& g, const nb::object& handle, const nb::bytes& data) {
                auto handlePtr = handle.attr("get")();
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                auto rawHandle = reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));
                const auto* ptr = reinterpret_cast<const uint8_t*>(data.c_str());
                const std::vector<uint8_t> vec(ptr, ptr + data.size());
                return g.deserialize(rawHandle, vec);
            },
            nb::arg("handle"),
            nb::arg("data"),
            "Deserialize and finalize graph from binary with a backend handle.\n"
            "The graph is ready for create_execution_plans() after this call.")
        .def(
            "from_binary",
            [](graph::Graph& g, const nb::bytes& data) {
                const auto* ptr = reinterpret_cast<const uint8_t*>(data.c_str());
                const std::vector<uint8_t> vec(ptr, ptr + data.size());
                return g.deserialize(vec);
            },
            nb::arg("data"),
            "Deserialize graph structure from binary without a handle.\n"
            "Only restores the graph topology and attributes (nodes, tensors, parameters).\n"
            "Call build_operation_graph(handle) after to finalize for execution.");
}
