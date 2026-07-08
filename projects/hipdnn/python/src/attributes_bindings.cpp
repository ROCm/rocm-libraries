// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace hipdnn_frontend;

void attributesBindings(nb::module_& m)
{
    // BatchnormAttributes
    nb::class_<graph::BatchnormAttributes>(m, "BatchnormAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::BatchnormAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::BatchnormAttributes::get_name)
        .def("set_compute_data_type",
             &graph::BatchnormAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::BatchnormAttributes::get_compute_data_type)
        .def(
            "set_epsilon",
            [](graph::BatchnormAttributes& self,
               const std::shared_ptr<graph::TensorAttributes>& epsilon)
                -> graph::BatchnormAttributes& { return self.set_epsilon(epsilon); },
            nb::rv_policy::reference_internal)
        .def("get_epsilon", &graph::BatchnormAttributes::get_epsilon)
        .def("get_momentum", &graph::BatchnormAttributes::get_momentum)
        .def("get_x", &graph::BatchnormAttributes::get_x)
        .def("get_scale", &graph::BatchnormAttributes::get_scale)
        .def("get_bias", &graph::BatchnormAttributes::get_bias)
        .def("get_y", &graph::BatchnormAttributes::get_y)
        .def("get_mean", &graph::BatchnormAttributes::get_mean)
        .def("get_inv_variance", &graph::BatchnormAttributes::get_inv_variance);

    // BatchnormBackwardAttributes
    nb::class_<graph::BatchnormBackwardAttributes>(m, "BatchnormBackwardAttributes")
        .def(nb::init<>())
        .def("set_name",
             &graph::BatchnormBackwardAttributes::set_name,
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::BatchnormBackwardAttributes::get_name)
        .def("set_compute_data_type",
             &graph::BatchnormBackwardAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::BatchnormBackwardAttributes::get_compute_data_type);

    // BatchnormInferenceAttributes
    nb::class_<graph::BatchnormInferenceAttributes>(m, "BatchnormInferenceAttributes")
        .def(nb::init<>())
        .def("set_name",
             &graph::BatchnormInferenceAttributes::set_name,
             nb::arg("name"),
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::BatchnormInferenceAttributes::get_name)
        .def("set_compute_data_type",
             &graph::BatchnormInferenceAttributes::set_compute_data_type,
             nb::arg("compute_type"),
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::BatchnormInferenceAttributes::get_compute_data_type)
        .def("get_x", &graph::BatchnormInferenceAttributes::get_x)
        .def("get_mean", &graph::BatchnormInferenceAttributes::get_mean)
        .def("get_inv_variance", &graph::BatchnormInferenceAttributes::get_inv_variance)
        .def("get_scale", &graph::BatchnormInferenceAttributes::get_scale)
        .def("get_bias", &graph::BatchnormInferenceAttributes::get_bias)
        .def("get_y", &graph::BatchnormInferenceAttributes::get_y);

    // ConvolutionDgradAttributes (using typedef)
    auto convDgradClass = nb::class_<graph::ConvDgradAttributes>(m, "ConvolutionDgradAttributes")
                              .def(nb::init<>())
                              .def(
                                  "set_name",
                                  [](graph::ConvDgradAttributes& self,
                                     const std::string& name) -> graph::ConvDgradAttributes& {
                                      return self.set_name(name);
                                  },
                                  nb::arg("name"),
                                  nb::rv_policy::reference_internal)
                              .def("set_padding",
                                   &graph::ConvDgradAttributes::set_padding,
                                   nb::rv_policy::reference_internal)
                              .def("set_pre_padding",
                                   static_cast<graph::ConvDgradAttributes& (
                                       graph::ConvDgradAttributes::*)(const std::vector<int64_t>&)>(
                                       &graph::ConvDgradAttributes::set_pre_padding),
                                   nb::rv_policy::reference_internal)
                              .def("set_post_padding",
                                   static_cast<graph::ConvDgradAttributes& (
                                       graph::ConvDgradAttributes::*)(const std::vector<int64_t>&)>(
                                       &graph::ConvDgradAttributes::set_post_padding),
                                   nb::rv_policy::reference_internal)
                              .def("set_stride",
                                   static_cast<graph::ConvDgradAttributes& (
                                       graph::ConvDgradAttributes::*)(const std::vector<int64_t>&)>(
                                       &graph::ConvDgradAttributes::set_stride),
                                   nb::rv_policy::reference_internal)
                              .def("set_dilation",
                                   static_cast<graph::ConvDgradAttributes& (
                                       graph::ConvDgradAttributes::*)(const std::vector<int64_t>&)>(
                                       &graph::ConvDgradAttributes::set_dilation),
                                   nb::rv_policy::reference_internal)
                              .def("get_name", &graph::ConvDgradAttributes::get_name);
    // Add alias for shorter name
    m.attr("ConvDgradAttributes") = convDgradClass;

    // ConvolutionFpropAttributes (using typedef)
    auto convFpropClass = nb::class_<graph::ConvFpropAttributes>(m, "ConvolutionFpropAttributes")
                              .def(nb::init<>())
                              .def(
                                  "set_name",
                                  [](graph::ConvFpropAttributes& self,
                                     const std::string& name) -> graph::ConvFpropAttributes& {
                                      return self.set_name(name);
                                  },
                                  nb::arg("name"),
                                  nb::rv_policy::reference_internal)
                              .def("set_padding",
                                   &graph::ConvFpropAttributes::set_padding,
                                   nb::rv_policy::reference_internal)
                              .def("set_stride",
                                   static_cast<graph::ConvFpropAttributes& (
                                       graph::ConvFpropAttributes::*)(const std::vector<int64_t>&)>(
                                       &graph::ConvFpropAttributes::set_stride),
                                   nb::rv_policy::reference_internal)
                              .def("set_dilation",
                                   static_cast<graph::ConvFpropAttributes& (
                                       graph::ConvFpropAttributes::*)(const std::vector<int64_t>&)>(
                                       &graph::ConvFpropAttributes::set_dilation),
                                   nb::rv_policy::reference_internal)
                              .def("get_name", &graph::ConvFpropAttributes::get_name);
    // Add alias for shorter name
    m.attr("ConvFpropAttributes") = convFpropClass;

    // ConvolutionWgradAttributes (using typedef)
    auto convWgradClass = nb::class_<graph::ConvWgradAttributes>(m, "ConvolutionWgradAttributes")
                              .def(nb::init<>())
                              .def(
                                  "set_name",
                                  [](graph::ConvWgradAttributes& self,
                                     const std::string& name) -> graph::ConvWgradAttributes& {
                                      return self.set_name(name);
                                  },
                                  nb::arg("name"),
                                  nb::rv_policy::reference_internal)
                              .def("set_padding",
                                   &graph::ConvWgradAttributes::set_padding,
                                   nb::rv_policy::reference_internal)
                              .def("set_pre_padding",
                                   static_cast<graph::ConvWgradAttributes& (
                                       graph::ConvWgradAttributes::*)(const std::vector<int64_t>&)>(
                                       &graph::ConvWgradAttributes::set_pre_padding),
                                   nb::rv_policy::reference_internal)
                              .def("set_post_padding",
                                   static_cast<graph::ConvWgradAttributes& (
                                       graph::ConvWgradAttributes::*)(const std::vector<int64_t>&)>(
                                       &graph::ConvWgradAttributes::set_post_padding),
                                   nb::rv_policy::reference_internal)
                              .def("set_stride",
                                   static_cast<graph::ConvWgradAttributes& (
                                       graph::ConvWgradAttributes::*)(const std::vector<int64_t>&)>(
                                       &graph::ConvWgradAttributes::set_stride),
                                   nb::rv_policy::reference_internal)
                              .def("set_dilation",
                                   static_cast<graph::ConvWgradAttributes& (
                                       graph::ConvWgradAttributes::*)(const std::vector<int64_t>&)>(
                                       &graph::ConvWgradAttributes::set_dilation),
                                   nb::rv_policy::reference_internal)
                              .def("get_name", &graph::ConvWgradAttributes::get_name);
    // Add alias for shorter name
    m.attr("ConvWgradAttributes") = convWgradClass;

    // MatmulAttributes
    nb::class_<graph::MatmulAttributes>(m, "MatmulAttributes")
        .def(nb::init<>())
        .def(
            "set_name",
            [](graph::MatmulAttributes& self, const std::string& name) -> graph::MatmulAttributes& {
                return self.set_name(name);
            },
            nb::arg("name"),
            nb::rv_policy::reference_internal)
        .def("get_name", &graph::MatmulAttributes::get_name)
        .def("set_compute_data_type",
             &graph::MatmulAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::MatmulAttributes::get_compute_data_type);

    // PointwiseAttributes
    nb::class_<graph::PointwiseAttributes>(m, "PointwiseAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::PointwiseAttributes::set_name, nb::rv_policy::reference_internal)
        .def("set_mode", &graph::PointwiseAttributes::set_mode, nb::rv_policy::reference_internal)
        .def("get_name", &graph::PointwiseAttributes::get_name)
        .def("get_mode", &graph::PointwiseAttributes::get_mode);
}
