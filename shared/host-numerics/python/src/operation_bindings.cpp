// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <optional>
#include <roc/host_numerics/validation.hpp>
#include <utility>
#include <vector>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace roc::host_numerics::python_bindings {
namespace {
AxpbyResult referenceAxpbyOwned(std::optional<Tensor> x, std::optional<Tensor> y,
                                ScalarType outputType, ScalarType accumulatorType,
                                std::complex<double> alpha, std::complex<double> beta) {
    AxpbyProblem problem(std::move(x), std::move(y), outputType, accumulatorType);
    problem.alpha = alpha;
    problem.beta = beta;
    return referenceAxpby(problem);
}

SoftmaxResult referenceSoftmaxOwned(const Tensor& input, ScalarType outputType,
                                    ScalarType accumulatorType, size_t axis) {
    return referenceSoftmax(SoftmaxProblem(input, outputType, axis, accumulatorType));
}

LayerNormResult referenceLayerNormOwned(const Tensor& input, ScalarType outputType,
                                        ScalarType statisticsType, ScalarType accumulatorType,
                                        size_t axis, double epsilon, std::optional<Tensor> gamma,
                                        std::optional<Tensor> beta) {
    LayerNormProblem problem(input, outputType, axis, accumulatorType);
    problem.meanType = statisticsType;
    problem.inverseVarianceType = statisticsType;
    problem.gamma = std::move(gamma);
    problem.beta = std::move(beta);
    problem.epsilon = epsilon;
    return referenceLayerNorm(problem);
}

EpilogueResult referenceEpilogueOwned(
    const Tensor& input, ScalarType outputType, ScalarType computeType, std::optional<Tensor> bias,
    MatrixAxis biasAxis, Activation activation, ActivationApplication activationApplication,
    std::optional<Tensor> auxiliaryInput, std::optional<ScalarType> auxiliaryOutputType,
    std::optional<Tensor> gateResidual, std::complex<double> outputScale,
    std::complex<double> auxiliaryScale, double activationParameter0, double activationParameter1,
    OutputConversion outputConversion, bool includeRawOutput, bool includeAmax,
    OutputSelection outputSelection) {
    EpilogueProblem problem(input, outputType, computeType);
    if (includeRawOutput) problem.rawOutputType = computeType;
    problem.auxiliaryOutputType = auxiliaryOutputType;
    if (includeAmax) problem.amaxType = computeType;
    problem.auxiliaryInput = std::move(auxiliaryInput);
    problem.gateResidual = std::move(gateResidual);
    if (bias) problem.bias = VectorBinding{*bias, biasAxis};
    problem.outputScale = outputScale;
    problem.auxiliaryScale = auxiliaryScale;
    problem.outputConversion = outputConversion;
    problem.activation = activation;
    problem.activationApplication = activationApplication;
    problem.activationParameter0 = activationParameter0;
    problem.activationParameter1 = activationParameter1;
    problem.outputSelection = std::move(outputSelection);
    return referenceEpilogue(problem);
}

Tensor referenceSumOwned(const Tensor& input, ScalarType outputType, ScalarType accumulatorType,
                         std::vector<size_t> axes) {
    return referenceSum(ReductionProblem(input, outputType, accumulatorType, std::move(axes)))
        .output;
}

Tensor referenceMaximumAbsoluteOwned(const Tensor& input, ScalarType outputType,
                                     ScalarType accumulatorType) {
    return referenceMaximumAbsolute(input, outputType, accumulatorType).output;
}

StructuredSparsityResult applyStructuredSparsityOwned(const Tensor& input,
                                                      StructuredSparsityPattern pattern,
                                                      bool emitTwoOfFourMetadata) {
    StructuredSparsityProblem problem(
        input, std::move(pattern),
        {.retainedIndices = true, .twoOfFourMetadata = emitTwoOfFourMetadata});
    return applyStructuredSparsity(problem);
}

TwoOfFourMetadataResult encodeTwoOfFourMetadataOwned(const Tensor& retainedIndices, size_t axis) {
    return encodeTwoOfFourMetadata(TwoOfFourMetadataProblem(retainedIndices, axis));
}
}  // namespace

void registerOperationBindings(nb::module_& module) {
    nb::class_<AxpbyProblem>(module, "AxpbyProblem")
        .def(nb::init<std::optional<Tensor>, std::optional<Tensor>, ScalarType, ScalarType>(),
             "x"_a = std::optional<Tensor>{}, "y"_a = std::optional<Tensor>{},
             "output_type"_a = ScalarType::Float32, "accumulator_type"_a = ScalarType::Float32)
        .def_rw("x", &AxpbyProblem::x)
        .def_rw("y", &AxpbyProblem::y)
        .def_rw("output_type", &AxpbyProblem::outputType)
        .def_rw("accumulator_type", &AxpbyProblem::accumulatorType)
        .def_prop_rw(
            "alpha",
            [](const AxpbyProblem& problem) { return problem.alpha.as<std::complex<double>>(); },
            [](AxpbyProblem& problem, std::complex<double> value) {
                problem.alpha = Scalar(value);
            })
        .def_prop_rw(
            "beta",
            [](const AxpbyProblem& problem) { return problem.beta.as<std::complex<double>>(); },
            [](AxpbyProblem& problem, std::complex<double> value) {
                problem.beta = Scalar(value);
            });
    nb::class_<AxpbyRequest, AxpbyProblem>(module, "AxpbyRequest")
        .def(nb::init<std::optional<Tensor>, std::optional<Tensor>, Tensor, ScalarType>(), "x"_a,
             "y"_a, "output"_a, "accumulator_type"_a = ScalarType::Float32)
        .def(nb::init<AxpbyProblem, Tensor>(), "problem"_a, "output"_a)
        .def_rw("output", &AxpbyRequest::output);
    nb::class_<AxpbyRunInfo>(module, "AxpbyRunInfo")
        .def_ro("output_elements_written", &AxpbyRunInfo::outputElementsWritten);
    nb::class_<AxpbyResult>(module, "AxpbyResult")
        .def_prop_ro(
            "output", [](const AxpbyResult& result) -> const Tensor& { return result.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const AxpbyResult& result) -> const AxpbyRunInfo& { return result.runInfo; },
            nb::rv_policy::reference_internal);

    nb::class_<SoftmaxProblem>(module, "SoftmaxProblem")
        .def(nb::init<Tensor, ScalarType, size_t, ScalarType>(), "input"_a,
             "output_type"_a = ScalarType::Float32, "axis"_a = 0,
             "accumulator_type"_a = ScalarType::Float32)
        .def_rw("input", &SoftmaxProblem::input)
        .def_rw("output_type", &SoftmaxProblem::outputType)
        .def_rw("axis", &SoftmaxProblem::axis)
        .def_rw("accumulator_type", &SoftmaxProblem::accumulatorType);
    nb::class_<SoftmaxRequest, SoftmaxProblem>(module, "SoftmaxRequest")
        .def(nb::init<Tensor, Tensor, size_t, ScalarType>(), "input"_a, "output"_a, "axis"_a = 0,
             "accumulator_type"_a = ScalarType::Float32)
        .def(nb::init<SoftmaxProblem, Tensor>(), "problem"_a, "output"_a)
        .def_rw("output", &SoftmaxRequest::output);
    nb::class_<SoftmaxRunInfo>(module, "SoftmaxRunInfo")
        .def_ro("slices_processed", &SoftmaxRunInfo::slicesProcessed)
        .def_ro("output_elements_written", &SoftmaxRunInfo::outputElementsWritten);
    nb::class_<SoftmaxResult>(module, "SoftmaxResult")
        .def_prop_ro(
            "output", [](const SoftmaxResult& result) -> const Tensor& { return result.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const SoftmaxResult& result) -> const SoftmaxRunInfo& { return result.runInfo; },
            nb::rv_policy::reference_internal);

    nb::class_<LayerNormProblem>(module, "LayerNormProblem")
        .def(nb::init<Tensor, ScalarType, size_t, ScalarType>(), "input"_a,
             "output_type"_a = ScalarType::Float32, "axis"_a = 0,
             "accumulator_type"_a = ScalarType::Float32)
        .def_rw("input", &LayerNormProblem::input)
        .def_rw("output_type", &LayerNormProblem::outputType)
        .def_rw("mean_type", &LayerNormProblem::meanType)
        .def_rw("inverse_variance_type", &LayerNormProblem::inverseVarianceType)
        .def_rw("gamma", &LayerNormProblem::gamma)
        .def_rw("beta", &LayerNormProblem::beta)
        .def_rw("axis", &LayerNormProblem::axis)
        .def_rw("accumulator_type", &LayerNormProblem::accumulatorType)
        .def_rw("epsilon", &LayerNormProblem::epsilon);
    nb::class_<LayerNormRequest, LayerNormProblem>(module, "LayerNormRequest")
        .def(nb::init<LayerNormProblem, Tensor, std::optional<Tensor>, std::optional<Tensor>>(),
             "problem"_a, "output"_a, "mean"_a = std::optional<Tensor>{},
             "inverse_variance"_a = std::optional<Tensor>{})
        .def_rw("output", &LayerNormRequest::output)
        .def_rw("mean", &LayerNormRequest::mean)
        .def_rw("inverse_variance", &LayerNormRequest::inverseVariance);
    nb::class_<LayerNormRunInfo>(module, "LayerNormRunInfo")
        .def_ro("slices_processed", &LayerNormRunInfo::slicesProcessed)
        .def_ro("output_elements_written", &LayerNormRunInfo::outputElementsWritten)
        .def_ro("mean_elements_written", &LayerNormRunInfo::meanElementsWritten)
        .def_ro("inverse_variance_elements_written",
                &LayerNormRunInfo::inverseVarianceElementsWritten);
    nb::class_<LayerNormResult>(module, "LayerNormResult")
        .def_prop_ro(
            "output", [](const LayerNormResult& result) -> const Tensor& { return result.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "mean",
            [](const LayerNormResult& result) -> const std::optional<Tensor>& {
                return result.mean;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "inverse_variance",
            [](const LayerNormResult& result) -> const std::optional<Tensor>& {
                return result.inverseVariance;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const LayerNormResult& result) -> const LayerNormRunInfo& { return result.runInfo; },
            nb::rv_policy::reference_internal);

    nb::class_<ReductionProblem>(module, "ReductionProblem")
        .def(nb::init<Tensor, ScalarType, ScalarType, std::vector<size_t>, ReductionOperation>(),
             "input"_a, "output_type"_a, "accumulator_type"_a, "axes"_a,
             "operation"_a = ReductionOperation::Sum)
        .def_rw("input", &ReductionProblem::input)
        .def_rw("output_type", &ReductionProblem::outputType)
        .def_rw("accumulator_type", &ReductionProblem::accumulatorType)
        .def_rw("axes", &ReductionProblem::axes)
        .def_rw("operation", &ReductionProblem::operation);
    nb::class_<ReductionRequest, ReductionProblem>(module, "ReductionRequest")
        .def(nb::init<ReductionProblem, Tensor>(), "problem"_a, "output"_a)
        .def_rw("output", &ReductionRequest::output);
    nb::class_<ReductionRunInfo>(module, "ReductionRunInfo")
        .def_ro("output_elements_written", &ReductionRunInfo::outputElementsWritten)
        .def_ro("input_elements_read", &ReductionRunInfo::inputElementsRead);
    nb::class_<ReductionResult>(module, "ReductionResult")
        .def_prop_ro(
            "output", [](const ReductionResult& result) -> const Tensor& { return result.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const ReductionResult& result) -> const ReductionRunInfo& { return result.runInfo; },
            nb::rv_policy::reference_internal);

    nb::class_<EpilogueProblem>(module, "EpilogueProblem")
        .def(nb::init<Tensor, ScalarType, ScalarType>(), "input"_a, "output_type"_a,
             "compute_type"_a)
        .def_rw("input", &EpilogueProblem::input)
        .def_rw("output_type", &EpilogueProblem::outputType)
        .def_rw("compute_type", &EpilogueProblem::computeType)
        .def_rw("raw_output_type", &EpilogueProblem::rawOutputType)
        .def_rw("auxiliary_output_type", &EpilogueProblem::auxiliaryOutputType)
        .def_rw("amax_type", &EpilogueProblem::amaxType)
        .def_rw("auxiliary_input", &EpilogueProblem::auxiliaryInput)
        .def_rw("gate_residual", &EpilogueProblem::gateResidual)
        .def_rw("bias", &EpilogueProblem::bias)
        .def_prop_rw(
            "output_scale",
            [](const EpilogueProblem& problem) {
                return problem.outputScale.as<std::complex<double>>();
            },
            [](EpilogueProblem& problem, std::complex<double> value) {
                problem.outputScale = Scalar(value);
            })
        .def_prop_rw(
            "auxiliary_scale",
            [](const EpilogueProblem& problem) {
                return problem.auxiliaryScale.as<std::complex<double>>();
            },
            [](EpilogueProblem& problem, std::complex<double> value) {
                problem.auxiliaryScale = Scalar(value);
            })
        .def_rw("output_conversion", &EpilogueProblem::outputConversion)
        .def_rw("activation", &EpilogueProblem::activation)
        .def_rw("activation_application", &EpilogueProblem::activationApplication)
        .def_prop_rw(
            "activation_parameter0",
            [](const EpilogueProblem& problem) {
                return problem.activationParameter0.as<double>();
            },
            [](EpilogueProblem& problem, double value) {
                problem.activationParameter0 = Scalar(value);
            })
        .def_prop_rw(
            "activation_parameter1",
            [](const EpilogueProblem& problem) {
                return problem.activationParameter1.as<double>();
            },
            [](EpilogueProblem& problem, double value) {
                problem.activationParameter1 = Scalar(value);
            })
        .def_rw("output_selection", &EpilogueProblem::outputSelection);
    nb::class_<EpilogueRequest, EpilogueProblem>(module, "EpilogueRequest")
        .def(nb::init<EpilogueProblem, Tensor, std::optional<Tensor>, std::optional<Tensor>,
                      std::optional<Tensor>>(),
             "problem"_a, "output"_a, "raw_output"_a = std::optional<Tensor>{},
             "auxiliary_output"_a = std::optional<Tensor>{}, "amax"_a = std::optional<Tensor>{})
        .def_rw("output", &EpilogueRequest::output)
        .def_rw("raw_output", &EpilogueRequest::rawOutput)
        .def_rw("auxiliary_output", &EpilogueRequest::auxiliaryOutput)
        .def_rw("amax", &EpilogueRequest::amax)
        .def_rw("accumulate_amax", &EpilogueRequest::accumulateAmax);
    nb::class_<EpilogueRunInfo>(module, "EpilogueRunInfo")
        .def_ro("output_elements_written", &EpilogueRunInfo::outputElementsWritten)
        .def_ro("raw_output_elements_written", &EpilogueRunInfo::rawOutputElementsWritten)
        .def_ro("auxiliary_output_elements_written",
                &EpilogueRunInfo::auxiliaryOutputElementsWritten)
        .def_ro("amax_elements_written", &EpilogueRunInfo::amaxElementsWritten);
    nb::class_<EpilogueResult>(module, "EpilogueResult")
        .def_prop_ro(
            "output", [](const EpilogueResult& result) -> const Tensor& { return result.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "raw_output",
            [](const EpilogueResult& result) -> const std::optional<Tensor>& {
                return result.rawOutput;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "auxiliary_output",
            [](const EpilogueResult& result) -> const std::optional<Tensor>& {
                return result.auxiliaryOutput;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "amax",
            [](const EpilogueResult& result) -> const std::optional<Tensor>& {
                return result.amax;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const EpilogueResult& result) -> const EpilogueRunInfo& { return result.runInfo; },
            nb::rv_policy::reference_internal);

    nb::class_<StructuredSparsityPattern>(module, "StructuredSparsityPattern")
        .def(nb::init<>())
        .def_rw("axis", &StructuredSparsityPattern::axis)
        .def_rw("group_size", &StructuredSparsityPattern::groupSize)
        .def_rw("retained_elements", &StructuredSparsityPattern::retainedElements)
        .def_rw("selection", &StructuredSparsityPattern::selection)
        .def_rw("fixed_positions", &StructuredSparsityPattern::fixedPositions)
        .def_rw("seed", &StructuredSparsityPattern::seed)
        .def_rw("index_order", &StructuredSparsityPattern::indexOrder);
    nb::class_<StructuredSparsityOutputs>(module, "StructuredSparsityOutputs")
        .def(nb::init<>())
        .def_rw("retained_indices", &StructuredSparsityOutputs::retainedIndices)
        .def_rw("two_of_four_metadata", &StructuredSparsityOutputs::twoOfFourMetadata);
    nb::class_<StructuredSparsityProblem>(module, "StructuredSparsityProblem")
        .def(nb::init<Tensor, StructuredSparsityPattern, StructuredSparsityOutputs>(), "input"_a,
             "pattern"_a, "outputs"_a = StructuredSparsityOutputs{})
        .def_rw("input", &StructuredSparsityProblem::input)
        .def_rw("pattern", &StructuredSparsityProblem::pattern)
        .def_rw("outputs", &StructuredSparsityProblem::outputs);
    nb::class_<StructuredSparsityRequest, StructuredSparsityProblem>(module,
                                                                     "StructuredSparsityRequest")
        .def(nb::init<StructuredSparsityProblem, Tensor, Tensor, std::optional<Tensor>,
                      std::optional<Tensor>>(),
             "problem"_a, "pruned"_a, "compressed"_a,
             "retained_indices"_a = std::optional<Tensor>{},
             "two_of_four_metadata"_a = std::optional<Tensor>{})
        .def_rw("pruned", &StructuredSparsityRequest::pruned)
        .def_rw("compressed", &StructuredSparsityRequest::compressed)
        .def_rw("retained_indices", &StructuredSparsityRequest::retainedIndices)
        .def_rw("two_of_four_metadata", &StructuredSparsityRequest::twoOfFourMetadata);
    nb::class_<StructuredSparsitySliceRange>(module, "StructuredSparsitySliceRange")
        .def(nb::init<>())
        .def_rw("first_slice", &StructuredSparsitySliceRange::firstSlice)
        .def_rw("slice_count", &StructuredSparsitySliceRange::sliceCount);
    nb::class_<StructuredSparsityRunInfo>(module, "StructuredSparsityRunInfo")
        .def_ro("groups_processed", &StructuredSparsityRunInfo::groupsProcessed)
        .def_ro("input_elements_visited", &StructuredSparsityRunInfo::inputElementsVisited)
        .def_ro("pruned_elements_written", &StructuredSparsityRunInfo::prunedElementsWritten)
        .def_ro("compressed_elements_written",
                &StructuredSparsityRunInfo::compressedElementsWritten)
        .def_ro("retained_indices_written", &StructuredSparsityRunInfo::retainedIndicesWritten)
        .def_ro("metadata_bytes_written", &StructuredSparsityRunInfo::metadataBytesWritten);
    nb::class_<StructuredSparsityResult>(module, "StructuredSparsityResult")
        .def_prop_ro(
            "pruned",
            [](const StructuredSparsityResult& result) -> const Tensor& { return result.pruned; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "compressed",
            [](const StructuredSparsityResult& result) -> const Tensor& {
                return result.compressed;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "retained_indices",
            [](const StructuredSparsityResult& result) -> const std::optional<Tensor>& {
                return result.retainedIndices;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "two_of_four_metadata",
            [](const StructuredSparsityResult& result) -> const std::optional<Tensor>& {
                return result.twoOfFourMetadata;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const StructuredSparsityResult& result) -> const StructuredSparsityRunInfo& {
                return result.runInfo;
            },
            nb::rv_policy::reference_internal);

    nb::class_<TwoOfFourMetadataProblem>(module, "TwoOfFourMetadataProblem")
        .def(nb::init<Tensor, size_t>(), "retained_indices"_a, "axis"_a)
        .def_rw("retained_indices", &TwoOfFourMetadataProblem::retainedIndices)
        .def_rw("axis", &TwoOfFourMetadataProblem::axis);
    nb::class_<TwoOfFourMetadataRequest, TwoOfFourMetadataProblem>(module,
                                                                   "TwoOfFourMetadataRequest")
        .def(nb::init<TwoOfFourMetadataProblem, Tensor>(), "problem"_a, "metadata"_a)
        .def_rw("metadata", &TwoOfFourMetadataRequest::metadata);
    nb::class_<TwoOfFourMetadataRunInfo>(module, "TwoOfFourMetadataRunInfo")
        .def_ro("sparsity_groups_encoded", &TwoOfFourMetadataRunInfo::sparsityGroupsEncoded)
        .def_ro("metadata_bytes_written", &TwoOfFourMetadataRunInfo::metadataBytesWritten);
    nb::class_<TwoOfFourMetadataResult>(module, "TwoOfFourMetadataResult")
        .def_prop_ro(
            "metadata",
            [](const TwoOfFourMetadataResult& result) -> const Tensor& { return result.metadata; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const TwoOfFourMetadataResult& result) -> const TwoOfFourMetadataRunInfo& {
                return result.runInfo;
            },
            nb::rv_policy::reference_internal);

    module.def("reference_axpby", &referenceAxpbyOwned, "x"_a = std::optional<Tensor>{},
               "y"_a = std::optional<Tensor>{}, "output_type"_a = ScalarType::Float32,
               "accumulator_type"_a = ScalarType::Float32, "alpha"_a = 1.0, "beta"_a = 1.0);
    module.def("reference_axpby",
               static_cast<AxpbyRunInfo (*)(const AxpbyRequest&)>(&referenceAxpby), "request"_a);
    module.def("reference_axpby",
               static_cast<AxpbyResult (*)(const AxpbyProblem&)>(&referenceAxpby), "problem"_a);

    module.def("reference_softmax", &referenceSoftmaxOwned, "input"_a,
               "output_type"_a = ScalarType::Float32, "accumulator_type"_a = ScalarType::Float32,
               "axis"_a = 0);
    module.def("reference_softmax",
               static_cast<SoftmaxRunInfo (*)(const SoftmaxRequest&)>(&referenceSoftmax),
               "request"_a);
    module.def("reference_softmax",
               static_cast<SoftmaxResult (*)(const SoftmaxProblem&)>(&referenceSoftmax),
               "problem"_a);

    module.def("reference_layer_norm", &referenceLayerNormOwned, "input"_a,
               "output_type"_a = ScalarType::Float32, "statistics_type"_a = ScalarType::Float32,
               "accumulator_type"_a = ScalarType::Float32, "axis"_a = 0, "epsilon"_a = 1e-5,
               "gamma"_a = std::optional<Tensor>{}, "beta"_a = std::optional<Tensor>{});
    module.def("reference_layer_norm",
               static_cast<LayerNormRunInfo (*)(const LayerNormRequest&)>(&referenceLayerNorm),
               "request"_a);
    module.def("reference_layer_norm",
               static_cast<LayerNormResult (*)(const LayerNormProblem&)>(&referenceLayerNorm),
               "problem"_a);

    module.def("reference_reduce",
               static_cast<ReductionRunInfo (*)(const ReductionRequest&)>(&referenceReduce),
               "request"_a);
    module.def("reference_reduce",
               static_cast<ReductionResult (*)(const ReductionProblem&)>(&referenceReduce),
               "problem"_a);
    module.def("reference_sum", &referenceSumOwned, "input"_a, "output_type"_a,
               "accumulator_type"_a, "axes"_a);
    module.def("reference_sum",
               static_cast<ReductionRunInfo (*)(const ReductionRequest&)>(&referenceSum),
               "request"_a);
    module.def("reference_sum",
               static_cast<ReductionResult (*)(const ReductionProblem&)>(&referenceSum),
               "problem"_a);
    module.def("reference_maximum_absolute", &referenceMaximumAbsoluteOwned, "input"_a,
               "output_type"_a, "accumulator_type"_a);
    module.def(
        "reference_maximum_absolute_result",
        [](Tensor input, ScalarType outputType, ScalarType accumulatorType) {
            return referenceMaximumAbsolute(std::move(input), outputType, accumulatorType);
        },
        "input"_a, "output_type"_a, "accumulator_type"_a);
    module.def(
        "reference_maximum_absolute",
        static_cast<ReductionRunInfo (*)(const ReductionRequest&)>(&referenceMaximumAbsolute),
        "request"_a);

    module.def("reference_epilogue", &referenceEpilogueOwned, "input"_a, "output_type"_a,
               "compute_type"_a, "bias"_a = std::optional<Tensor>{},
               "bias_axis"_a = MatrixAxis::Row, "activation"_a = Activation::None,
               "activation_application"_a = ActivationApplication::Forward,
               "auxiliary_input"_a = std::optional<Tensor>{},
               "auxiliary_output_type"_a = std::optional<ScalarType>{},
               "gate_residual"_a = std::optional<Tensor>{},
               "output_scale"_a = std::complex<double>(1.0, 0.0),
               "auxiliary_scale"_a = std::complex<double>(1.0, 0.0),
               "activation_parameter0"_a = 0.0, "activation_parameter1"_a = 0.0,
               "output_conversion"_a = OutputConversion::Default, "include_raw_output"_a = false,
               "include_amax"_a = false, "output_selection"_a = OutputSelection::all());
    module.def("reference_epilogue",
               static_cast<EpilogueRunInfo (*)(const EpilogueRequest&)>(&referenceEpilogue),
               "request"_a);
    module.def("reference_epilogue",
               static_cast<EpilogueResult (*)(const EpilogueProblem&)>(&referenceEpilogue),
               "problem"_a);

    module.def("apply_structured_sparsity", &applyStructuredSparsityOwned, "input"_a, "pattern"_a,
               "emit_two_of_four_metadata"_a = false);
    module.def("apply_structured_sparsity",
               static_cast<StructuredSparsityRunInfo (*)(const StructuredSparsityRequest&,
                                                         StructuredSparsitySliceRange)>(
                   &applyStructuredSparsity),
               "request"_a, "slice_range"_a = StructuredSparsitySliceRange{});
    module.def("apply_structured_sparsity",
               static_cast<StructuredSparsityResult (*)(const StructuredSparsityProblem&)>(
                   &applyStructuredSparsity),
               "problem"_a);

    module.def("encode_two_of_four_metadata", &encodeTwoOfFourMetadataOwned, "retained_indices"_a,
               "axis"_a);
    module.def("encode_two_of_four_metadata",
               static_cast<TwoOfFourMetadataRunInfo (*)(const TwoOfFourMetadataRequest&)>(
                   &encodeTwoOfFourMetadata),
               "request"_a);
    module.def("encode_two_of_four_metadata",
               static_cast<TwoOfFourMetadataResult (*)(const TwoOfFourMetadataProblem&)>(
                   &encodeTwoOfFourMetadata),
               "problem"_a);
}
}  // namespace roc::host_numerics::python_bindings
