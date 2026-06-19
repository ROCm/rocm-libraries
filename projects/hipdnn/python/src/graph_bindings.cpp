// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

namespace nb = nanobind;
using namespace hipdnn_frontend;

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
        .def(
            "create_execution_plan_ext",
            [](graph::Graph& g, int64_t engineId) {
                return g.create_execution_plan_ext(engineId, {});
            },
            nb::arg("engine_id"),
            "Hard-select an engine: create/select the execution plan descriptor for "
            "this exact engine id (call build_plans() to finalize). Returns an Error "
            "whose is_bad() is set if the engine is not valid/applicable (no "
            "heuristic fallback).")
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
        .def("build_plans", &graph::Graph::build_plans)
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
        .def("tensor", &graph::Graph::tensor, nb::rv_policy::reference)
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
