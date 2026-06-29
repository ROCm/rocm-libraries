// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

void tensorBindings(nb::module_& m)
{
    nb::class_<TensorAttributes>(m, "Tensor")
        .def(nb::init<>())
        .def_static(
            "create",
            [](const std::vector<int64_t>& dims, DataType dataType) {
                auto tensor = std::make_shared<TensorAttributes>();
                tensor->set_dim(dims).set_data_type(dataType);
                tensor->set_stride(hipdnn_data_sdk::utilities::generateStrides(
                    dims, hipdnn_data_sdk::utilities::TensorLayout::NCHW.strideOrder));
                return tensor;
            },
            nb::arg("dims"),
            nb::arg("data_type"))
        .def("get_uid", &TensorAttributes::get_uid)
        .def("set_uid", &TensorAttributes::set_uid, nb::rv_policy::reference_internal)
        .def("get_name", &TensorAttributes::get_name)
        .def("set_name", &TensorAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_data_type", &TensorAttributes::get_data_type)
        .def("set_data_type", &TensorAttributes::set_data_type, nb::rv_policy::reference_internal)
        .def("get_stride", &TensorAttributes::get_stride)
        .def("set_stride", &TensorAttributes::set_stride, nb::rv_policy::reference_internal)
        .def("get_dim", &TensorAttributes::get_dim)
        .def("set_dim", &TensorAttributes::set_dim, nb::rv_policy::reference_internal)
        .def("get_is_virtual", &TensorAttributes::get_is_virtual)
        .def("set_is_virtual", &TensorAttributes::set_is_virtual, nb::rv_policy::reference_internal)
        .def("set_output", &TensorAttributes::set_output, nb::rv_policy::reference_internal)
        .def("set_value", &TensorAttributes::set_value<float>, nb::rv_policy::reference_internal)
        .def("get_volume", &TensorAttributes::get_volume)
        .def("has_uid", &TensorAttributes::has_uid)
        .def("clear_uid", &TensorAttributes::clear_uid, nb::rv_policy::reference_internal)
        .def("validate", &TensorAttributes::validate);
}
