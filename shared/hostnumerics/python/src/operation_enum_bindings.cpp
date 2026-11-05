// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include <roc/hostnumerics/validation.hpp>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerOperationEnumBindings(nb::module_& module) {
    nb::enum_<MathMode>(module, "MathMode")
        .value("Default", MathMode::Default)
        .value("XFloat32", MathMode::XFloat32);

    nb::enum_<AccumulationRounding>(module, "AccumulationRounding")
        .value("TypeDefault", AccumulationRounding::TypeDefault)
        .value("FullPrecision", AccumulationRounding::FullPrecision)
        .value("AfterProductAndSum", AccumulationRounding::AfterProductAndSum);

    nb::enum_<GemmBackend>(module, "GemmBackend")
        .value("Automatic", GemmBackend::Automatic)
        .value("Blocked", GemmBackend::Blocked);

    nb::enum_<OutputConversion>(module, "OutputConversion")
        .value("Default", OutputConversion::Default)
        .value("SaturatingInt8", OutputConversion::SaturatingInt8);

    nb::enum_<Activation>(module, "Activation")
        .value("None_", Activation::None)
        .value("Absolute", Activation::Absolute)
        .value("ClippedRelu", Activation::ClippedRelu)
        .value("Relu", Activation::Relu)
        .value("Gelu", Activation::Gelu)
        .value("GeluDerivative", Activation::GeluDerivative)
        .value("GeluScaling", Activation::GeluScaling)
        .value("LeakyRelu", Activation::LeakyRelu)
        .value("ReluDerivative", Activation::ReluDerivative)
        .value("Sigmoid", Activation::Sigmoid)
        .value("Tanh", Activation::Tanh)
        .value("Silu", Activation::Silu)
        .value("Swish", Activation::Swish)
        .value("Clamp", Activation::Clamp);

    nb::enum_<StructuredSparsitySelection>(module, "StructuredSparsitySelection")
        .value("Fixed", StructuredSparsitySelection::Fixed)
        .value("Random", StructuredSparsitySelection::Random);

    nb::enum_<ActivationApplication>(module, "ActivationApplication")
        .value("Forward", ActivationApplication::Forward)
        .value("Gradient", ActivationApplication::Gradient);

    nb::enum_<ReductionOperation>(module, "ReductionOperation")
        .value("Sum", ReductionOperation::Sum)
        .value("MaximumAbsolute", ReductionOperation::MaximumAbsolute);
}
}  // namespace roc::hostnumerics::python_bindings
