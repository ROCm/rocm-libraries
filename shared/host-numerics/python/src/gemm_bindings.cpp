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
        throw std::invalid_argument("Python reference_gemm exposes the Blocked backend.");
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

    module.def("_matmul", &matmulOwned, "a"_a, "b"_a, "output_type"_a = ScalarType::Float32,
               "options"_a = MatmulOptions{}, "output_layout"_a = std::optional<Layout>{},
               "backend"_a = GemmBackend::Automatic);
    module.def("_matmul_into", &matmulIntoBound, "a"_a, "b"_a, "output"_a,
               "options"_a = MatmulOptions{}, "output_selection"_a = OutputSelection::all(),
               "backend"_a = GemmBackend::Automatic);
}
}  // namespace roc::host_numerics::python_bindings
