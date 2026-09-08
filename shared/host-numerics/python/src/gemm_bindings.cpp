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
#include "detail/gemm_invocation.hpp"
#include "detail/reference_gemm.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace roc::host_numerics::python_bindings {
namespace {
void validatePythonGemmBackend(GemmBackend backend) {
    if (backend == GemmBackend::Blas)
        throw std::invalid_argument("Python reference_gemm exposes the Blocked backend.");
}

Tensor referenceGemmOwned(Tensor a, Tensor b, Tensor c, ScalarType outputType, GemmOptions options,
                          std::optional<Layout> outputLayout, GemmBackend backend) {
    validatePythonGemmBackend(backend);
    const GemmSpecification problem(std::move(a), std::move(b), std::move(c), outputType, options);
    detail::validateRuntimeGemmProblem(problem);
    const Shape outputShape{problem.a.shape()[0], problem.b.shape()[1]};
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    if (layout.shape() != outputShape)
        throw std::invalid_argument("Owning reference GEMM output layout shape mismatch.");
    Tensor destination(outputType, layout);
    (void)detail::executeGemm(GemmInvocation(problem, destination, options.outputSelection),
                              backend);
    return destination;
}

Tensor matmulOwned(Tensor a, Tensor b, ScalarType outputType, MatmulOptions options,
                   std::optional<Layout> outputLayout, GemmBackend backend) {
    validatePythonGemmBackend(backend);
    return matmul(std::move(a), std::move(b), outputType, options, std::move(outputLayout),
                  backend);
}

void matmulIntoBound(Tensor a, Tensor b, Tensor output, MatmulOptions options,
                     OutputSelection selection, GemmBackend backend) {
    validatePythonGemmBackend(backend);
    matmulInto(a, b, std::move(output), options, std::move(selection), backend);
}

void referenceGemmIntoBound(Tensor a, Tensor b, Tensor c, Tensor d, GemmOptions options,
                            GemmBackend backend) {
    validatePythonGemmBackend(backend);
    (void)detail::executeGemm(
        GemmInvocation(std::move(a), std::move(b), std::move(c), std::move(d), options), backend);
}

}  // namespace

void registerGemmBindings(nb::module_& module) {
    nb::class_<MatmulOptions>(module, "_MatmulOptions")
        .def(nb::init<ScalarType>(), "accumulator_type"_a = ScalarType::Float32)
        .def_rw("accumulator_type", &MatmulOptions::accumulatorType)
        .def_rw("accumulation_rounding", &MatmulOptions::accumulationRounding)
        .def_rw("math_mode", &MatmulOptions::mathMode)
        .def_rw("compute_type_a", &MatmulOptions::computeTypeA)
        .def_rw("compute_type_b", &MatmulOptions::computeTypeB)
        .def_rw("pre_quantization_scales_a", &MatmulOptions::preQuantizationScalesA)
        .def_rw("pre_quantization_scales_b", &MatmulOptions::preQuantizationScalesB)
        .def_rw("block_scale_a", &MatmulOptions::blockScaleA)
        .def_rw("block_scale_b", &MatmulOptions::blockScaleB)
        .def_rw("block_size_a", &MatmulOptions::blockSizeA)
        .def_rw("block_size_b", &MatmulOptions::blockSizeB)
        .def_rw("conjugate_a", &MatmulOptions::conjugateA)
        .def_rw("conjugate_b", &MatmulOptions::conjugateB);

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
        .def_prop_rw(
            "alpha",
            [](const GemmOptions& options) { return options.alpha.item<std::complex<double>>(); },
            [](GemmOptions& options, nb::object value) { options.alpha = scalarFromPython(value); })
        .def_prop_rw(
            "beta",
            [](const GemmOptions& options) { return options.beta.item<std::complex<double>>(); },
            [](GemmOptions& options, nb::object value) { options.beta = scalarFromPython(value); })
        .def_prop_rw(
            "scale_c",
            [](const GemmOptions& options) { return options.scaleC.item<std::complex<double>>(); },
            [](GemmOptions& options, nb::object value) {
                options.scaleC = scalarFromPython(value);
            })
        .def_rw("bias", &GemmOptions::bias)
        .def_rw("scale_alpha", &GemmOptions::scaleAlpha)
        .def_rw("scale_a", &GemmOptions::scaleA)
        .def_rw("scale_b", &GemmOptions::scaleB)
        .def_prop_rw(
            "output_scale",
            [](const GemmOptions& options) {
                return options.outputScale.item<std::complex<double>>();
            },
            [](GemmOptions& options, nb::object value) {
                options.outputScale = scalarFromPython(value);
            })
        .def_rw("output_conversion", &GemmOptions::outputConversion)
        .def_rw("activation", &GemmOptions::activation)
        .def_prop_rw(
            "activation_parameter0",
            [](const GemmOptions& options) { return options.activationParameter0.item<double>(); },
            [](GemmOptions& options, nb::object value) {
                options.activationParameter0 = scalarFromPython(value);
            })
        .def_prop_rw(
            "activation_parameter1",
            [](const GemmOptions& options) { return options.activationParameter1.item<double>(); },
            [](GemmOptions& options, nb::object value) {
                options.activationParameter1 = scalarFromPython(value);
            })
        .def_rw("output_selection", &GemmOptions::outputSelection);

    module.def("_reference_gemm", &referenceGemmOwned, "a"_a, "b"_a, "c"_a,
               "output_type"_a = ScalarType::Float32, "options"_a = GemmOptions{},
               "output_layout"_a = std::optional<Layout>{}, "backend"_a = GemmBackend::Automatic);
    module.def("_reference_gemm_into", &referenceGemmIntoBound, "a"_a, "b"_a, "c"_a, "d"_a,
               "options"_a = GemmOptions{}, "backend"_a = GemmBackend::Automatic);
    module.def("_matmul", &matmulOwned, "a"_a, "b"_a, "output_type"_a = ScalarType::Float32,
               "options"_a = MatmulOptions{}, "output_layout"_a = std::optional<Layout>{},
               "backend"_a = GemmBackend::Automatic);
    module.def("_matmul_into", &matmulIntoBound, "a"_a, "b"_a, "output"_a,
               "options"_a = MatmulOptions{}, "output_selection"_a = OutputSelection::all(),
               "backend"_a = GemmBackend::Automatic);
}
}  // namespace roc::host_numerics::python_bindings
