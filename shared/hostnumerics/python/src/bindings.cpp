// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <roc/hostnumerics/scalar_type.hpp>
#include <roc/hostnumerics/version.hpp>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

NB_MODULE(_hostnumerics, module) {
    module.attr("__version__") = version().data();
    nb::enum_<ScalarCategory>(module, "ScalarCategory")
        .value("Boolean", ScalarCategory::Boolean)
        .value("SignedInteger", ScalarCategory::SignedInteger)
        .value("UnsignedInteger", ScalarCategory::UnsignedInteger)
        .value("FloatingPoint", ScalarCategory::FloatingPoint)
        .value("Complex", ScalarCategory::Complex);

    nb::enum_<ScalarType>(module, "ScalarType")
        .value("Boolean", ScalarType::Boolean)
        .value("Int4", ScalarType::Int4)
        .value("Int8", ScalarType::Int8)
        .value("Int16", ScalarType::Int16)
        .value("Int32", ScalarType::Int32)
        .value("Int64", ScalarType::Int64)
        .value("UInt8", ScalarType::UInt8)
        .value("UInt16", ScalarType::UInt16)
        .value("UInt32", ScalarType::UInt32)
        .value("UInt64", ScalarType::UInt64)
        .value("Float4E2M1", ScalarType::Float4E2M1)
        .value("Float6E2M3", ScalarType::Float6E2M3)
        .value("Float6E3M2", ScalarType::Float6E3M2)
        .value("Float8E4M3", ScalarType::Float8E4M3)
        .value("Float8E5M2", ScalarType::Float8E5M2)
        .value("Float8E4M3Fnuz", ScalarType::Float8E4M3Fnuz)
        .value("Float8E5M2Fnuz", ScalarType::Float8E5M2Fnuz)
        .value("E4M3", ScalarType::E4M3)
        .value("E5M3", ScalarType::E5M3)
        .value("E8M0", ScalarType::E8M0)
        .value("E8M0Zero", ScalarType::E8M0Zero)
        .value("Float16", ScalarType::Float16)
        .value("BFloat16", ScalarType::BFloat16)
        .value("Float32", ScalarType::Float32)
        .value("Float64", ScalarType::Float64)
        .value("ComplexFloat32", ScalarType::ComplexFloat32)
        .value("ComplexFloat64", ScalarType::ComplexFloat64);

    nb::class_<ScalarTypeInfo>(module, "ScalarTypeInfo")
        .def_prop_ro("name", [](const ScalarTypeInfo& info) { return std::string(info.name); })
        .def_ro("category", &ScalarTypeInfo::category)
        .def_ro("storage_bits", &ScalarTypeInfo::storageBits)
        .def_ro("exponent_bits", &ScalarTypeInfo::exponentBits)
        .def_ro("mantissa_bits", &ScalarTypeInfo::mantissaBits)
        .def_ro("exponent_bias", &ScalarTypeInfo::exponentBias)
        .def_ro("is_signed", &ScalarTypeInfo::isSigned)
        .def_ro("supports_nan", &ScalarTypeInfo::supportsNaN)
        .def_ro("supports_infinity", &ScalarTypeInfo::supportsInfinity)
        .def("is_packed", &ScalarTypeInfo::isPacked);

    module.def("scalar_type_info", &scalarTypeInfo, "type"_a);
}
