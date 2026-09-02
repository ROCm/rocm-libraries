// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <optional>
#include <roc/host_numerics/gemm.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace roc::host_numerics::python_bindings {
namespace {
void validatePythonGemmBackend(GemmBackend backend) {
    if (backend == GemmBackend::Blas)
        throw std::invalid_argument(
            "Python reference_gemm exposes Pointwise and Blocked backends.");
    if (backend == GemmBackend::Mixed)
        throw std::invalid_argument("Mixed is a reporting-only GEMM backend value.");
}

Tensor referenceGemmOwned(Tensor a, Tensor b, Tensor c, ScalarType outputType, GemmOptions options,
                          std::optional<Layout> outputLayout, GemmBackend backend) {
    validatePythonGemmBackend(backend);
    return referenceGemm(std::move(a), std::move(b), std::move(c), outputType, options,
                         std::move(outputLayout), backend);
}

GemmBackend referenceGemmIntoBound(Tensor a, Tensor b, Tensor c, Tensor d, GemmOptions options,
                                   GemmBackend backend) {
    validatePythonGemmBackend(backend);
    return referenceGemmInto(std::move(a), std::move(b), std::move(c), std::move(d), options,
                             backend);
}

}  // namespace

void registerGemmBindings(nb::module_& module) {
    nb::class_<GemmEpilogue>(module, "_GemmEpilogue")
        .def(nb::init<ScalarType>(), "coefficient_type"_a)
        .def_prop_rw(
            "alpha",
            [](const GemmEpilogue& epilogue) { return epilogue.alpha.as<std::complex<double>>(); },
            [](GemmEpilogue& epilogue, nb::object value) {
                epilogue.alpha = scalarFromPython(value);
            })
        .def_prop_rw(
            "beta",
            [](const GemmEpilogue& epilogue) { return epilogue.beta.as<std::complex<double>>(); },
            [](GemmEpilogue& epilogue, nb::object value) {
                epilogue.beta = scalarFromPython(value);
            })
        .def_prop_rw(
            "scale_c",
            [](const GemmEpilogue& epilogue) { return epilogue.scaleC.as<std::complex<double>>(); },
            [](GemmEpilogue& epilogue, nb::object value) {
                epilogue.scaleC = scalarFromPython(value);
            })
        .def_rw("bias", &GemmEpilogue::bias)
        .def_rw("scale_alpha", &GemmEpilogue::scaleAlpha)
        .def_rw("scale_a", &GemmEpilogue::scaleA)
        .def_rw("scale_b", &GemmEpilogue::scaleB)
        .def_prop_rw(
            "output_scale",
            [](const GemmEpilogue& epilogue) {
                return epilogue.outputScale.as<std::complex<double>>();
            },
            [](GemmEpilogue& epilogue, nb::object value) {
                epilogue.outputScale = scalarFromPython(value);
            })
        .def_rw("output_conversion", &GemmEpilogue::outputConversion)
        .def_rw("activation", &GemmEpilogue::activation)
        .def_prop_rw(
            "activation_parameter0",
            [](const GemmEpilogue& epilogue) { return epilogue.activationParameter0.as<double>(); },
            [](GemmEpilogue& epilogue, nb::object value) {
                epilogue.activationParameter0 = scalarFromPython(value);
            })
        .def_prop_rw(
            "activation_parameter1",
            [](const GemmEpilogue& epilogue) { return epilogue.activationParameter1.as<double>(); },
            [](GemmEpilogue& epilogue, nb::object value) {
                epilogue.activationParameter1 = scalarFromPython(value);
            });

    nb::class_<GemmOptions>(module, "_GemmOptions")
        .def(nb::init<ScalarType>(), "accumulator_type"_a = ScalarType::Float32)
        .def_rw("accumulator_type", &GemmOptions::accumulatorType)
        .def_rw("accumulation_rounding", &GemmOptions::accumulationRounding)
        .def_rw("math_mode", &GemmOptions::mathMode)
        .def_rw("compute_type_a", &GemmOptions::computeTypeA)
        .def_rw("compute_type_b", &GemmOptions::computeTypeB)
        .def_rw("pre_quantization_scales_a", &GemmOptions::preQuantizationScalesA)
        .def_rw("pre_quantization_scales_b", &GemmOptions::preQuantizationScalesB)
        .def_rw("block_scale_a", &GemmOptions::blockScaleA)
        .def_rw("block_scale_b", &GemmOptions::blockScaleB)
        .def_rw("block_size_a", &GemmOptions::blockSizeA)
        .def_rw("block_size_b", &GemmOptions::blockSizeB)
        .def_rw("conjugate_a", &GemmOptions::conjugateA)
        .def_rw("conjugate_b", &GemmOptions::conjugateB)
        .def_rw("epilogue", &GemmOptions::epilogue)
        .def_rw("output_selection", &GemmOptions::outputSelection);

    module.def("_reference_gemm", &referenceGemmOwned, "a"_a, "b"_a, "c"_a,
               "output_type"_a = ScalarType::Float32, "options"_a = GemmOptions{},
               "output_layout"_a = std::optional<Layout>{}, "backend"_a = GemmBackend::Pointwise);
    module.def("_reference_gemm_into", &referenceGemmIntoBound, "a"_a, "b"_a, "c"_a, "d"_a,
               "options"_a = GemmOptions{}, "backend"_a = GemmBackend::Pointwise);
}
}  // namespace roc::host_numerics::python_bindings
