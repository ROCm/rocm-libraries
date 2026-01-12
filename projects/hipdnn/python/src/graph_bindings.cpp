// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace hipdnn_frontend;

void graph_bindings(nb::module_& m)
{
    nb::class_<graph::Graph>(m, "Graph")
        .def(nb::init<>())
        .def("validate", &graph::Graph::validate)
        .def("checkNoDuplicateTensorIds", &graph::Graph::checkNoDuplicateTensorIds)
        .def("topologicallySortGraph", &graph::Graph::topologicallySortGraph)
        .def("buildFlatbufferOperationGraph", &graph::Graph::buildFlatbufferOperationGraph)
        .def(
            "build_operation_graph",
            [](graph::Graph& g, nb::object handle) {
                // Extract handle pointer from Python Handle object
                auto handlePtr = handle.attr("get")();
                hipdnnHandle_t rawHandle
                    = reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));
                return g.build_operation_graph(rawHandle);
            },
            nb::arg("handle"),
            "Build the operation graph with the given handle")
        .def("create_execution_plans",
             &graph::Graph::create_execution_plans,
             nb::arg("modes") = std::vector<HeuristicMode>{HeuristicMode::FALLBACK},
             "Create execution plans with specified heuristic modes")
        .def("check_support", &graph::Graph::check_support)
        .def("build_plans", &graph::Graph::build_plans)
        .def(
            "get_workspace_size",
            [](const graph::Graph& g) {
                int64_t workspaceSize;
                auto result = g.get_workspace_size(workspaceSize);
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
               nb::object handle,
               std::unordered_map<int64_t, uintptr_t>& variantPack,
               uintptr_t workspace) {
                // Extract handle pointer from Python Handle object
                auto handlePtr = handle.attr("get")();
                hipdnnHandle_t rawHandle
                    = reinterpret_cast<hipdnnHandle_t>(nb::cast<uintptr_t>(handlePtr));

                // Convert Python integer pointers to void*
                std::unordered_map<int64_t, void*> cppVariantPack;
                for(const auto& [key, value] : variantPack)
                {
                    cppVariantPack[key] = reinterpret_cast<void*>(value);
                }

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
        .def("set_preferred_engine_id_ext", &graph::Graph::set_preferred_engine_id_ext)
        .def("tensor", &graph::Graph::tensor, nb::rv_policy::reference)
        .def_static(
            "tensor_like", &graph::Graph::tensor_like, nb::arg("tensor"), nb::arg("name") = "");
}
