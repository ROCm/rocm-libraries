// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"
#include "cudnn_compat.hpp"
#include "dlpack_adapter.hpp"
#include "handle_wrapper.hpp"

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace hipdnn_frontend;
using hipdnn_python::throwIfBad;
using hipdnn_python::throwUnsupportedParameter;

namespace
{

using OverrideUids = std::optional<std::vector<int64_t>>;
using OverrideDims = std::optional<std::vector<std::vector<int64_t>>>;

struct PyGraph : graph::Graph
{
    // Supplied by the constructor, or created on the first handle-less call that needs one:
    // creating a handle loads the engine plugins, which graphs that never build should not pay.
    std::shared_ptr<HandleWrapper> handle;

    HandleWrapper& ensureHandle()
    {
        if(!handle)
        {
            handle = std::make_shared<HandleWrapper>();
        }
        return *handle;
    }

    // An explicit call handle wins, even one whose get() is null; only None falls back to the
    // graph's handle.
    hipdnnHandle_t resolveHandle(const nb::handle& callHandle)
    {
        if(!callHandle.is_none())
        {
            return hipdnn_python::rawHandleOrNull(callHandle);
        }
        return ensureHandle().get();
    }
};

template <typename T>
bool isUnset(const std::optional<std::vector<T>>& value)
{
    return !value || value->empty();
}

void rejectOverrides(const OverrideUids& uids,
                     const OverrideDims& shapes,
                     const OverrideDims& strides)
{
    if(!isUnset(uids))
    {
        throwUnsupportedParameter("override_uids", "None");
    }
    if(!isUnset(shapes))
    {
        throwUnsupportedParameter("override_shapes", "None");
    }
    if(!isUnset(strides))
    {
        throwUnsupportedParameter("override_strides", "None");
    }
}

void* workspacePointer(const nb::handle& workspace)
{
    return workspace.is_none() ? nullptr : hipdnn_python::toDataPointer(workspace, "workspace");
}

// cuDNN's execute skips None keys, so optional tensors a builder returned as None can stay in
// the dict.
std::unordered_map<int64_t, void*> cudnnVariantPack(const nb::dict& tensorToDeviceBuffer)
{
    bool hasNoneKey = false;
    for(const auto& item : tensorToDeviceBuffer)
    {
        if(item.first.is_none())
        {
            hasNoneKey = true;
            break;
        }
    }
    if(!hasNoneKey)
    {
        return hipdnn_python::toVariantPack(tensorToDeviceBuffer);
    }

    nb::dict filtered;
    for(const auto& [key, value] : tensorToDeviceBuffer)
    {
        if(!key.is_none())
        {
            filtered[key] = value;
        }
    }
    return hipdnn_python::toVariantPack(filtered);
}

void executeCudnn(const PyGraph& g,
                  hipdnnHandle_t handle,
                  std::unordered_map<int64_t, void*>& variantPack,
                  void* workspace,
                  const OverrideUids& overrideUids,
                  const OverrideDims& overrideShapes,
                  const OverrideDims& overrideStrides)
{
#ifdef HIPDNN_ENABLE_SDPA
    // Same dispatch as cuDNN: shapes and strides only matter once uids are given.
    if(!isUnset(overrideUids))
    {
        throwIfBad(g.execute(handle,
                             variantPack,
                             workspace,
                             *overrideUids,
                             overrideShapes.value_or(std::vector<std::vector<int64_t>>{}),
                             overrideStrides.value_or(std::vector<std::vector<int64_t>>{})));
        return;
    }
#else
    rejectOverrides(overrideUids, overrideShapes, overrideStrides);
#endif
    throwIfBad(g.execute(handle, variantPack, workspace));
}

void deserializePy(PyGraph& g, const nb::handle& handle, const nb::handle& pyobj)
{
    // Like cuDNN, the JSON form restores structure only and ignores the handle.
    if(nb::isinstance<nb::str>(pyobj))
    {
        throwIfBad(g.deserialize(nb::cast<std::string>(pyobj)));
        return;
    }

    std::vector<uint8_t> data;
    if(nb::isinstance<nb::bytes>(pyobj))
    {
        const auto bytes = nb::borrow<nb::bytes>(pyobj);
        const auto* ptr = reinterpret_cast<const uint8_t*>(bytes.c_str());
        data.assign(ptr, ptr + bytes.size());
    }
    else if(!nb::try_cast(pyobj, data))
    {
        throw nb::type_error("deserialize() expects a str (JSON) or bytes / list[int] (binary)");
    }
    throwIfBad(g.deserialize(g.resolveHandle(handle), data));
}

int64_t workspaceSizePlanAtIndex(const PyGraph& g, int64_t index)
{
    int64_t size = 0;
    throwIfBad(g.get_workspace_size_plan_at_index(index, size));
    return size;
}

void executePlanAtIndex(PyGraph& g,
                        const nb::object& handle,
                        const nb::dict& variantPack,
                        const nb::object& workspace,
                        int64_t index)
{
    auto cppVariantPack = hipdnn_python::toVariantPack(variantPack);
    throwIfBad(g.execute_plan_at_index(
        g.resolveHandle(handle), cppVariantPack, workspacePointer(workspace), index));
}

} // namespace

void pygraphBindings(nb::module_& m)
{
    nb::class_<PyGraph, graph::Graph>(
        m,
        "pygraph",
        "hipdnn.Graph with the semantics of cudnn.pygraph: it owns a handle (supplied or created "
        "on first use) for calls made without one, and methods raise instead of returning "
        "Error.")
        .def(
            "__init__",
            [](PyGraph* self,
               const std::string& name,
               DataType ioDataType,
               DataType intermediateDataType,
               DataType computeDataType,
               std::shared_ptr<HandleWrapper> handle,
               std::optional<int64_t> smCount,
               std::optional<int64_t> smVersion,
               const nb::object& kernelCache,
               const nb::object& deviceProperty,
               bool isDynamicShapeEnabled,
               bool isOverrideShapeEnabled) {
                // Checked before construction so a rejected call leaves no half-built graph.
                if(smCount)
                {
                    throwUnsupportedParameter("sm_count", "None");
                }
                if(smVersion)
                {
                    throwUnsupportedParameter("sm_version", "None");
                }
                if(!kernelCache.is_none())
                {
                    throwUnsupportedParameter("kernel_cache", "None");
                }
                if(!deviceProperty.is_none())
                {
                    throwUnsupportedParameter("device_property", "None");
                }
                if(isDynamicShapeEnabled)
                {
                    throwUnsupportedParameter("is_dynamic_shape_enabled", "False");
                }
#ifndef HIPDNN_ENABLE_SDPA
                if(isOverrideShapeEnabled)
                {
                    throwUnsupportedParameter("is_override_shape_enabled",
                                              "False in a build without SDPA support");
                }
#endif

                auto* g = new(self) PyGraph();
                g->handle = std::move(handle);
                if(!name.empty())
                {
                    g->set_name(name);
                }
                g->set_io_data_type(ioDataType)
                    .set_intermediate_data_type(intermediateDataType)
                    .set_compute_data_type(computeDataType);
#ifdef HIPDNN_ENABLE_SDPA
                if(isOverrideShapeEnabled)
                {
                    g->set_override_shape_enabled(true);
                }
#endif
            },
            nb::arg("name") = "",
            nb::arg("io_data_type") = DataType::NOT_SET,
            nb::arg("intermediate_data_type") = DataType::NOT_SET,
            nb::arg("compute_data_type") = DataType::NOT_SET,
            nb::arg("handle") = nb::none(),
            nb::arg("sm_count") = nb::none(),
            nb::arg("sm_version") = nb::none(),
            nb::arg("kernel_cache") = nb::none(),
            nb::arg("device_property") = nb::none(),
            nb::arg("is_dynamic_shape_enabled") = false,
            nb::arg("is_override_shape_enabled") = false,
            "Create a graph. handle is a hipdnn.Handle used by every call made without one; "
            "when omitted, one is created on first use. sm_count, sm_version, kernel_cache, "
            "device_property and is_dynamic_shape_enabled accept only their defaults.")
        .def(
            "validate", [](PyGraph& g) { throwIfBad(g.validate()); }, "Validate the graph.")
        .def("checkNoDuplicateTensorIds",
             [](PyGraph& g) { throwIfBad(g.checkNoDuplicateTensorIds()); })
        .def("topologicallySortGraph", [](PyGraph& g) { throwIfBad(g.topologicallySortGraph()); })
        .def(
            "build_operation_graph",
            [](PyGraph& g, const nb::object& handle) {
                throwIfBad(g.build_operation_graph(g.resolveHandle(handle)));
            },
            nb::arg("handle").none(),
            "Build the operation graph with the given handle; None uses the graph's handle.")
        .def(
            "build_operation_graph",
            [](PyGraph& g) { throwIfBad(g.build_operation_graph(g.resolveHandle(nb::none()))); },
            "Build the operation graph with the graph's handle.")
        .def(
            "create_execution_plans",
            [](PyGraph& g, const std::vector<HeuristicMode>& modes) {
                throwIfBad(g.create_execution_plans(modes));
            },
            nb::arg("modes") = std::vector<HeuristicMode>{HeuristicMode::FALLBACK},
            "Create execution plans with the given heuristic modes.")
        .def(
            "create_execution_plan_ext",
            [](PyGraph& g, int64_t engineId, const std::vector<KnobSetting>& knobSettings) {
                throwIfBad(g.create_execution_plan_ext(engineId, knobSettings));
            },
            nb::arg("engine_id"),
            nb::arg("knob_settings") = std::vector<KnobSetting>{},
            "Hard-select an engine, optionally overriding its knobs (call build_plans() to "
            "finalize).")
        .def("check_support", [](PyGraph& g) { throwIfBad(g.check_support()); })
        .def(
            "build_plans",
            [](PyGraph& g, BuildPlanPolicy policy) { throwIfBad(g.build_plans(policy)); },
            nb::arg("policy") = BuildPlanPolicy::HEURISTICS_CHOICE,
            "Finalize execution plans. HEURISTICS_CHOICE compiles only the active plan; ALL "
            "compiles every plan.")
        .def(
            "build_plan_at_index",
            [](PyGraph& g, int64_t planIndex) { throwIfBad(g.build_plan_at_index(planIndex)); },
            nb::arg("plan_index"),
            "Compile if needed and make the plan at this index active.")
        .def(
            "build_plan_at_index",
            [](PyGraph& g, int64_t index) { throwIfBad(g.build_plan_at_index(index)); },
            nb::arg("index"),
            "Compile if needed and make the plan at this index active.")
        .def(
            "build",
            [](PyGraph& g, const std::vector<HeuristicMode>& heurModes) {
                throwIfBad(g.validate());
                throwIfBad(g.build_operation_graph(g.resolveHandle(nb::none())));
                throwIfBad(g.create_execution_plans(heurModes));
                throwIfBad(g.check_support());
                throwIfBad(g.build_plans(BuildPlanPolicy::HEURISTICS_CHOICE));
            },
            nb::arg("heur_modes"),
            "validate(), build_operation_graph(), create_execution_plans(heur_modes), "
            "check_support() and build_plans(), using the graph's handle.")
        .def(
            "build",
            [](PyGraph& g) {
                throwIfBad(g.validate());
                throwIfBad(g.build_operation_graph(g.resolveHandle(nb::none())));
            },
            "validate() and build_operation_graph(), using the graph's handle.")
        .def(
            "get_workspace_size",
            [](const PyGraph& g) {
                int64_t size = 0;
                throwIfBad(g.get_workspace_size(size));
                return size;
            },
            "Workspace size the active plan needs.")
        .def(
            "get_workspace_size",
            [](const PyGraph& g,
               const nb::object& /*handle*/,
               const OverrideUids& overrideUids,
               const OverrideDims& overrideShapes,
               const OverrideDims& overrideStrides) {
                rejectOverrides(overrideUids, overrideShapes, overrideStrides);
                int64_t size = 0;
                throwIfBad(g.get_workspace_size(size));
                return size;
            },
            nb::arg("handle") = nb::none(),
            nb::arg("override_uids") = nb::none(),
            nb::arg("override_shapes") = nb::none(),
            nb::arg("override_strides") = nb::none(),
            "Workspace size the active plan needs. handle is accepted for cuDNN compatibility "
            "and unused; the override arguments accept only None.")
        .def(
            "get_workspace_size_plan_at_index",
            [](const PyGraph& g, int64_t planIndex) {
                return workspaceSizePlanAtIndex(g, planIndex);
            },
            nb::arg("plan_index"),
            "Workspace size cached for one compiled plan.")
        .def(
            "get_workspace_size_plan_at_index",
            [](const PyGraph& g, int64_t index) { return workspaceSizePlanAtIndex(g, index); },
            nb::arg("index"),
            "Workspace size cached for one compiled plan.")
        .def(
            "get_workspace_size_plan_at_index",
            [](const PyGraph& g,
               int64_t index,
               const nb::object& /*handle*/,
               const OverrideUids& overrideUids,
               const OverrideDims& overrideShapes,
               const OverrideDims& overrideStrides) {
                rejectOverrides(overrideUids, overrideShapes, overrideStrides);
                return workspaceSizePlanAtIndex(g, index);
            },
            nb::arg("index"),
            nb::arg("handle") = nb::none(),
            nb::arg("override_uids") = nb::none(),
            nb::arg("override_shapes") = nb::none(),
            nb::arg("override_strides") = nb::none(),
            "Workspace size cached for one compiled plan. handle is accepted for cuDNN "
            "compatibility and unused; the override arguments accept only None.")
        .def(
            "execute",
            [](PyGraph& g,
               const nb::object& handle,
               const nb::dict& variantPack,
               const nb::object& workspace) {
                auto cppVariantPack = hipdnn_python::toVariantPack(variantPack);
                throwIfBad(g.execute(
                    g.resolveHandle(handle), cppVariantPack, workspacePointer(workspace)));
            },
            nb::arg("handle").none(),
            nb::arg("variant_pack"),
            nb::arg("workspace") = 0,
            "Execute with the given handle (None uses the graph's handle). variant_pack and "
            "workspace accept the same keys and values as hipdnn.Graph.execute().")
        .def(
            "execute",
            [](PyGraph& g,
               const nb::dict& tensorToDeviceBuffer,
               const nb::object& workspace,
               const nb::object& handle,
               const OverrideUids& overrideUids,
               const OverrideDims& overrideShapes,
               const OverrideDims& overrideStrides) {
                auto variantPack = cudnnVariantPack(tensorToDeviceBuffer);
                executeCudnn(g,
                             g.resolveHandle(handle),
                             variantPack,
                             workspacePointer(workspace),
                             overrideUids,
                             overrideShapes,
                             overrideStrides);
            },
            nb::arg("tensor_to_device_buffer"),
            nb::arg("workspace"),
            nb::arg("handle") = nb::none(),
            nb::arg("override_uids") = nb::none(),
            nb::arg("override_shapes") = nb::none(),
            nb::arg("override_strides") = nb::none(),
            "Execute the active plan (cuDNN form). Keys are tensors or UIDs (None keys are "
            "skipped); values and workspace accept the same kinds as hipdnn.Graph.execute(). "
            "handle=None uses the graph's handle. The override arguments need a build with SDPA "
            "support; otherwise they accept only None.")
        .def("execute_plan_at_index",
             &executePlanAtIndex,
             nb::arg("handle").none(),
             nb::arg("variant_pack"),
             nb::arg("workspace"),
             nb::arg("plan_index"),
             "Execute one compiled plan without making it active.")
        .def("execute_plan_at_index",
             &executePlanAtIndex,
             nb::arg("handle").none(),
             nb::arg("variant_pack"),
             nb::arg("workspace"),
             nb::arg("index"),
             "Execute one compiled plan without making it active.")
        .def(
            "execute_plan_at_index",
            [](PyGraph& g,
               const nb::dict& tensorToDeviceBuffer,
               const nb::object& workspace,
               int64_t index,
               const nb::object& handle,
               const OverrideUids& overrideUids,
               const OverrideDims& overrideShapes,
               const OverrideDims& overrideStrides) {
                rejectOverrides(overrideUids, overrideShapes, overrideStrides);
                auto variantPack = cudnnVariantPack(tensorToDeviceBuffer);
                throwIfBad(g.execute_plan_at_index(
                    g.resolveHandle(handle), variantPack, workspacePointer(workspace), index));
            },
            nb::arg("tensor_to_device_buffer"),
            nb::arg("workspace"),
            nb::arg("index"),
            nb::arg("handle") = nb::none(),
            nb::arg("override_uids") = nb::none(),
            nb::arg("override_shapes") = nb::none(),
            nb::arg("override_strides") = nb::none(),
            "Execute one compiled plan without making it active (cuDNN form). handle=None uses "
            "the graph's handle; the override arguments accept only None.")
        .def(
            "add_engine_configs",
            [](PyGraph& g, const std::vector<EngineConfigInfo>& configs) {
                throwIfBad(g.add_engine_configs(configs));
            },
            nb::arg("configs"),
            "Add engine configs (from get_engine_configs()) as autotune plan specs.")
        .def(
            "add_engine",
            [](PyGraph& g, int64_t engineId, const std::vector<KnobSetting>& knobSettings) {
                throwIfBad(g.add_engine(engineId, knobSettings));
            },
            nb::arg("engine_id"),
            nb::arg("knob_settings") = std::vector<KnobSetting>{},
            "Add one engine with optional knob settings as an autotune plan spec.")
        .def(
            "add_engines",
            [](PyGraph& g, const std::vector<int64_t>& engineIds) {
                throwIfBad(g.add_engines(engineIds));
            },
            nb::arg("engine_ids"),
            "Add several engines, each with its default knob settings, as plan specs.")
        .def(
            "add_engine_variants",
            [](PyGraph& g, const std::vector<EngineVariant>& variants) {
                throwIfBad(g.add_engine_variants(variants));
            },
            nb::arg("variants"),
            "Add one plan spec per EngineVariant.")
        .def(
            "add_engine_sweep",
            [](PyGraph& g, const std::vector<EngineSweepSpec>& specs) {
                throwIfBad(g.add_engine_sweep(specs));
            },
            nb::arg("specs"),
            "Add the Cartesian product of each EngineSweepSpec's knob axes as plan specs.")
        .def(
            "add_all_engines",
            [](PyGraph& g, const std::vector<HeuristicMode>& modes) {
                throwIfBad(g.add_all_engines(modes));
            },
            nb::arg("modes") = std::vector<HeuristicMode>{HeuristicMode::FALLBACK},
            "Add every engine applicable to the built operation graph as a plan spec.")
        .def(
            "set_ranking_metric",
            [](PyGraph& g, std::string metric) {
                throwIfBad(g.set_ranking_metric(std::move(metric)));
            },
            nb::arg("metric"),
            "Choose the registered ranking metric ('tflops' or 'time').")
        .def(
            "from_json",
            [](PyGraph& g, const nb::object& handle, const std::string& jsonStr) {
                throwIfBad(g.deserialize(g.resolveHandle(handle), jsonStr));
            },
            nb::arg("handle").none(),
            nb::arg("json_string"),
            "Deserialize and finalize the graph from JSON with a backend handle (None uses "
            "the graph's handle).")
        .def(
            "from_json",
            [](PyGraph& g, const std::string& jsonStr) { throwIfBad(g.deserialize(jsonStr)); },
            nb::arg("json_string"),
            "Deserialize the graph structure from JSON without a handle.")
        .def(
            "from_binary",
            [](PyGraph& g, const nb::object& handle, const nb::bytes& data) {
                const auto* ptr = reinterpret_cast<const uint8_t*>(data.c_str());
                const std::vector<uint8_t> vec(ptr, ptr + data.size());
                throwIfBad(g.deserialize(g.resolveHandle(handle), vec));
            },
            nb::arg("handle").none(),
            nb::arg("data"),
            "Deserialize and finalize the graph from binary with a backend handle (None uses "
            "the graph's handle).")
        .def(
            "from_binary",
            [](PyGraph& g, const nb::bytes& data) {
                const auto* ptr = reinterpret_cast<const uint8_t*>(data.c_str());
                const std::vector<uint8_t> vec(ptr, ptr + data.size());
                throwIfBad(g.deserialize(vec));
            },
            nb::arg("data"),
            "Deserialize the graph structure from binary without a handle.")
        .def(
            "serialize",
            [](PyGraph& g) {
                std::vector<uint8_t> data;
                throwIfBad(g.serialize(data));
                return data;
            },
            "Serialize the graph to a list of bytes (not interchangeable with cuDNN blobs).")
        .def("deserialize",
             &deserializePy,
             nb::arg("handle_").none(),
             nb::arg("pyobj"),
             "Restore a graph from serialize() output (bytes or list[int]), finalized with "
             "handle_ (None uses the graph's handle), or the structure from a JSON str.")
        .def(
            "deserialize",
            [](PyGraph& g, const nb::handle& pyobj) { deserializePy(g, nb::none(), pyobj); },
            nb::arg("pyobj"),
            "Restore a graph from serialize() output with the graph's handle, or the "
            "structure from a JSON str.");
}
