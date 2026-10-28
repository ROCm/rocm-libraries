// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <optional>
#include <roc/hostnumerics/gemm.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings.hpp"
namespace nb = nanobind;
using namespace nb::literals;

namespace roc::hostnumerics::python_bindings {
namespace {
void validatePythonGemmBackend(GemmBackend backend) {
    if (backend == GemmBackend::Blas)
        throw std::runtime_error("Python reference_gemm exposes the Blocked backend.");
}

Tensor matmulOwned(Tensor a, Tensor b, ScalarType outputType, MatmulOptions options,
                   std::optional<Layout> outputLayout, GemmBackend backend) {
    validatePythonGemmBackend(backend);
    return matmul(std::move(a), std::move(b), outputType, options, std::move(outputLayout),
                  backend);
}

template <typename Indices, typename Order>
void matmulIntoBound(Tensor a, Tensor b, Tensor output, MatmulOptions options,
                     const Indices& indices, const Order& order, GemmBackend backend) {
    validatePythonGemmBackend(backend);
    matmulInto(a, b, std::move(output), options, indices, order, backend);
}

}  // namespace

void registerGemmBindings(nb::module_& module) {
    nb::class_<MatmulBlockScale>(module, "MatmulBlockScale")
        .def(nb::init<Tensor, size_t>(), "scales"_a, "block_size"_a)
        .def_rw("scales", &MatmulBlockScale::scales)
        .def_rw("block_size", &MatmulBlockScale::blockSize);

    nb::class_<MatmulOptions>(module, "_MatmulOptions")
        .def(nb::init<ScalarType>(), "accumulator_type"_a)
        .def_rw("accumulator_type", &MatmulOptions::accumulatorType)
        .def_rw("accumulation_rounding", &MatmulOptions::accumulationRounding)
        .def_rw("math_mode", &MatmulOptions::mathMode)
        .def_rw("compute_type_a", &MatmulOptions::computeTypeA)
        .def_rw("compute_type_b", &MatmulOptions::computeTypeB)
        .def_rw("pre_quantization_scales_a", &MatmulOptions::preQuantizationScalesA)
        .def_rw("pre_quantization_scales_b", &MatmulOptions::preQuantizationScalesB)
        .def_rw("block_scale_a", &MatmulOptions::blockScaleA)
        .def_rw("block_scale_b", &MatmulOptions::blockScaleB)
        .def_rw("conjugate_a", &MatmulOptions::conjugateA)
        .def_rw("conjugate_b", &MatmulOptions::conjugateB);

    module.def("_matmul", &matmulOwned, "a"_a, "b"_a, "output_type"_a, "options"_a,
               "output_layout"_a, "backend"_a);
    module.def("_matmul_into", &matmulIntoBound<IndexRange, std::vector<size_t>>, "a"_a, "b"_a,
               "output"_a, "options"_a, "indices"_a, "index_order"_a, "backend"_a);
    module.def("_matmul_into", &matmulIntoBound<IndexRange, LastDimensionFastest>, "a"_a, "b"_a,
               "output"_a, "options"_a, "indices"_a, "index_order"_a, "backend"_a);
    module.def("_matmul_into", &matmulIntoBound<std::set<size_t>, std::vector<size_t>>, "a"_a,
               "b"_a, "output"_a, "options"_a, "indices"_a, "index_order"_a, "backend"_a);
    module.def("_matmul_into", &matmulIntoBound<std::set<size_t>, LastDimensionFastest>, "a"_a,
               "b"_a, "output"_a, "options"_a, "indices"_a, "index_order"_a, "backend"_a);
}
}  // namespace roc::hostnumerics::python_bindings
