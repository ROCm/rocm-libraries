// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <optional>
#include <roc/host_validation/backends/blocked.hpp>
#include <roc/host_validation/gemm.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace roc::host_validation::python_bindings {
namespace {
const GemmBackendImplementation* pythonGemmBackendImplementation(GemmBackend backend) {
    if (backend == GemmBackend::Automatic || backend == GemmBackend::Blocked) {
        static const BlockedGemmBackend blocked;
        return &blocked;
    }
    if (backend != GemmBackend::Automatic && backend != GemmBackend::Pointwise)
        throw std::invalid_argument(
            "Python reference_gemm exposes Pointwise and Blocked backends.");
    return nullptr;
}

GemmResult referenceGemmRequestBound(const GemmRequest& request, const GemmExecution& execution) {
    GemmRunInfo runInfo =
        referenceGemm(request, execution, pythonGemmBackendImplementation(execution.backend));
    return {.output = request.d, .runInfo = std::move(runInfo)};
}

GemmResult referenceGemmProblemOwned(const GemmProblem& problem, const GemmOutputOptions& output,
                                     const GemmExecution& execution) {
    return referenceGemm(problem, output, execution,
                         pythonGemmBackendImplementation(execution.backend));
}

}  // namespace

void registerGemmBindings(nb::module_& module) {
    nb::class_<VectorBinding>(module, "VectorBinding",
                              "Owning row- or column-axis tensor binding used by GEMM requests.")
        .def(nb::init<Tensor, MatrixAxis>(), "values"_a, "axis"_a = MatrixAxis::Row)
        .def_rw("values", &VectorBinding::values)
        .def_rw("axis", &VectorBinding::axis);

    nb::class_<BlockScaleBinding>(
        module, "BlockScaleBinding",
        "Owning tensor and reduction-block size used for GEMM block scaling.")
        .def(nb::init<Tensor, size_t>(), "values"_a, "block_size"_a)
        .def_rw("values", &BlockScaleBinding::values)
        .def_rw("block_size", &BlockScaleBinding::blockSize);

    nb::class_<GemmOperand>(
        module, "GemmOperand",
        "Owning GEMM operand, including compute-input quantization and scaling metadata.")
        .def(nb::init<Tensor>(), "values"_a)
        .def_rw("values", &GemmOperand::values)
        .def_rw("compute_type", &GemmOperand::computeType)
        .def_rw("pre_quantization_scales", &GemmOperand::preQuantizationScales)
        .def_rw("block_scale", &GemmOperand::blockScale)
        .def_rw("conjugate", &GemmOperand::conjugate);

    nb::class_<GemmEpilogue>(
        module, "GemmEpilogue",
        "Owning GEMM alpha/beta, vector scaling, activation, and output-conversion settings.")
        .def(nb::init<ScalarType>(), "coefficient_type"_a)
        .def_prop_rw(
            "alpha",
            [](const GemmEpilogue& epilogue) { return epilogue.alpha.as<std::complex<double>>(); },
            [](GemmEpilogue& epilogue, std::complex<double> value) {
                epilogue.alpha = Scalar(value);
            })
        .def_prop_rw(
            "beta",
            [](const GemmEpilogue& epilogue) { return epilogue.beta.as<std::complex<double>>(); },
            [](GemmEpilogue& epilogue, std::complex<double> value) {
                epilogue.beta = Scalar(value);
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
            [](GemmEpilogue& epilogue, std::complex<double> value) {
                epilogue.outputScale = Scalar(value);
            })
        .def_rw("output_conversion", &GemmEpilogue::outputConversion)
        .def_rw("activation", &GemmEpilogue::activation)
        .def_prop_rw(
            "activation_parameter0",
            [](const GemmEpilogue& epilogue) { return epilogue.activationParameter0.as<double>(); },
            [](GemmEpilogue& epilogue, double value) {
                epilogue.activationParameter0 = Scalar(value);
            })
        .def_prop_rw(
            "activation_parameter1",
            [](const GemmEpilogue& epilogue) { return epilogue.activationParameter1.as<double>(); },
            [](GemmEpilogue& epilogue, double value) {
                epilogue.activationParameter1 = Scalar(value);
            });

    nb::class_<GemmExecution>(
        module, "GemmExecution",
        "Call-time GEMM backend selection. Python supplies the built-in blocked implementation.")
        .def(nb::init<>())
        .def(nb::init<GemmBackend, bool>(), "backend"_a, "require_requested_backend"_a = false)
        .def_rw("backend", &GemmExecution::backend)
        .def_rw("require_requested_backend", &GemmExecution::requireRequestedBackend);

    nb::class_<GemmOutputOptions>(module, "GemmOutputOptions",
                                  "Owning GEMM output layout and logical-coordinate selection.")
        .def(nb::init<>())
        .def_rw("layout", &GemmOutputOptions::layout)
        .def_rw("selection", &GemmOutputOptions::selection);

    nb::class_<GemmProblem>(module, "GemmProblem", "Reusable numerical GEMM descriptor.")
        .def(nb::init<GemmOperand, GemmOperand, Tensor, ScalarType, ScalarType>(), "a"_a, "b"_a,
             "c"_a, "output_type"_a = ScalarType::Float32,
             "accumulator_type"_a = ScalarType::Float32)
        .def_rw("a", &GemmProblem::a)
        .def_rw("b", &GemmProblem::b)
        .def_rw("c", &GemmProblem::c)
        .def_rw("output_type", &GemmProblem::outputType)
        .def_rw("accumulator_type", &GemmProblem::accumulatorType)
        .def_rw("accumulation_rounding", &GemmProblem::accumulationRounding)
        .def_rw("math_mode", &GemmProblem::mathMode)
        .def_rw("epilogue", &GemmProblem::epilogue);

    nb::class_<GemmRequest, GemmProblem>(
        module, "GemmRequest",
        "Caller-owned GEMM invocation. The result aliases the supplied D tensor.")
        .def(nb::init<GemmOperand, GemmOperand, Tensor, Tensor, ScalarType>(), "a"_a, "b"_a, "c"_a,
             "d"_a, "accumulator_type"_a = ScalarType::Float32)
        .def(nb::init<GemmProblem, Tensor, OutputSelection>(), "problem"_a, "d"_a,
             "output_selection"_a = OutputSelection::all())
        .def_rw("d", &GemmRequest::d)
        .def_rw("output_selection", &GemmRequest::outputSelection);

    nb::class_<GemmRunInfo>(module, "GemmRunInfo")
        .def_ro("backend_used", &GemmRunInfo::backendUsed)
        .def_ro("fallback_reason", &GemmRunInfo::fallbackReason)
        .def_ro("output_elements_written", &GemmRunInfo::outputElementsWritten)
        .def_ro("output_elements_covered", &GemmRunInfo::outputElementsCovered);

    nb::class_<GemmResult>(module, "GemmResult")
        .def_prop_ro(
            "output", [](const GemmResult& result) -> const Tensor& { return result.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const GemmResult& result) -> const GemmRunInfo& { return result.runInfo; },
            nb::rv_policy::reference_internal);

    module.def("reference_gemm_result", &referenceGemmRequestBound, "request"_a,
               "execution"_a = GemmExecution{.backend = GemmBackend::Pointwise});
    module.def("reference_gemm_result", &referenceGemmProblemOwned, "problem"_a,
               "output"_a = GemmOutputOptions{},
               "execution"_a = GemmExecution{.backend = GemmBackend::Pointwise});
}
}  // namespace roc::host_validation::python_bindings
