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

#include <limits>
#include <optional>
#include <string>

namespace nb = nanobind;
using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

template <typename T>
T castScalar(nb::handle value)
{
    T out{};
    if(!nb::try_cast(value, out))
    {
        throw nb::type_error(("set_value(): cannot convert "
                              + std::string(nb::type_name(value.type()).c_str())
                              + " to the data type")
                                 .c_str());
    }
    return out;
}

DataType inferScalarType(nb::handle value)
{
    // bool is an int subclass in Python, so test it first.
    if(nb::isinstance<nb::bool_>(value))
    {
        return DataType::BOOLEAN;
    }
    if(nb::isinstance<nb::int_>(value))
    {
        return DataType::INT64;
    }
    if(nb::isinstance<nb::float_>(value))
    {
        return DataType::FLOAT;
    }
    throw nb::type_error(("set_value(): cannot infer a data type from "
                          + std::string(nb::type_name(value.type()).c_str()) + "; pass data_type")
                             .c_str());
}

template <typename T>
T castInteger(nb::handle value)
{
    const auto v = castScalar<int64_t>(value);
    if(v < static_cast<int64_t>(std::numeric_limits<T>::min())
       || v > static_cast<int64_t>(std::numeric_limits<T>::max()))
    {
        throw nb::value_error(
            ("set_value(): " + std::to_string(v) + " is out of range for the data type").c_str());
    }
    return static_cast<T>(v);
}

// Bakes a compile-time constant (plugin API 1.0.0). The data type comes from
// data_type, else the tensor's current type, else the Python value.
TensorAttributes&
    setValue(TensorAttributes& tensor, nb::handle value, std::optional<DataType> dataType)
{
    if(tensor.get_volume() != 1)
    {
        throw nb::value_error(
            "set_value(): the tensor must have one element; set_value resets dims to [1]");
    }
    DataType dt = dataType.value_or(tensor.get_data_type());
    if(dt == DataType::NOT_SET)
    {
        dt = inferScalarType(value);
    }
    switch(dt)
    {
    case DataType::FLOAT:
        return tensor.set_value(static_cast<float>(castScalar<double>(value)));
    case DataType::DOUBLE:
        return tensor.set_value(castScalar<double>(value));
    case DataType::HALF:
        return tensor.set_value(half(castScalar<double>(value)));
    case DataType::BFLOAT16:
        return tensor.set_value(bfloat16(castScalar<double>(value)));
    case DataType::UINT8:
        return tensor.set_value(castInteger<uint8_t>(value));
    case DataType::INT32:
        return tensor.set_value(castInteger<int32_t>(value));
    case DataType::INT64:
        return tensor.set_value(castScalar<int64_t>(value));
    case DataType::BOOLEAN:
        return tensor.set_value(castScalar<bool>(value));
    default:
        throw nb::value_error(
            "set_value(): data type must be FLOAT, DOUBLE, HALF, BFLOAT16, UINT8, INT32, "
            "INT64, or BOOLEAN");
    }
}

} // namespace

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
        .def("set_value",
             &setValue,
             nb::arg("value"),
             nb::arg("data_type") = nb::none(),
             nb::rv_policy::reference_internal,
             "Store a compile-time constant scalar (plugin API 1.0.0). The data type is "
             "data_type, else the tensor's current type, else inferred from value "
             "(bool -> BOOLEAN, int -> INT64, float -> FLOAT). Resets dims and strides to "
             "[1]; the tensor must have one element.")
        .def("get_is_pass_by_value", &TensorAttributes::get_is_pass_by_value)
        .def("get_is_runtime_pass_by_value", &TensorAttributes::get_is_runtime_pass_by_value)
        .def("set_is_pass_by_value",
             &TensorAttributes::set_is_pass_by_value,
             nb::rv_policy::reference_internal)
        .def("get_volume", &TensorAttributes::get_volume)
        .def("has_uid", &TensorAttributes::has_uid)
        .def("clear_uid", &TensorAttributes::clear_uid, nb::rv_policy::reference_internal)
        .def("validate", &TensorAttributes::validate);
}
