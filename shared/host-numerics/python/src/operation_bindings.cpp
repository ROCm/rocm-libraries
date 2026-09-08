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
Tensor tensorOperand(nb::handle value, ScalarType scalarType) {
    if (nb::isinstance<Tensor>(value)) return nb::cast<Tensor>(value);
    Tensor scalar = scalarFromPython(value);
    return Tensor::scalar(scalarType, scalar.item<std::complex<double>>());
}

Tensor addOwned(Tensor left, nb::object right, ScalarType outputType, ScalarType computeType) {
    Tensor rightTensor = tensorOperand(right, left.type());
    return add(std::move(left), std::move(rightTensor), outputType, computeType);
}

void addIntoBound(Tensor left, nb::object right, Tensor output, ScalarType computeType) {
    Tensor rightTensor = tensorOperand(right, left.type());
    addInto(std::move(left), std::move(rightTensor), std::move(output), computeType);
}

Tensor multiplyOwned(Tensor left, nb::object right, ScalarType outputType, ScalarType computeType) {
    Tensor rightTensor = tensorOperand(right, left.type());
    return multiply(std::move(left), std::move(rightTensor), outputType, computeType);
}

void multiplyIntoBound(Tensor left, nb::object right, Tensor output, ScalarType computeType) {
    Tensor rightTensor = tensorOperand(right, left.type());
    multiplyInto(std::move(left), std::move(rightTensor), std::move(output), computeType);
}

ActivationFunction activationFunction(Activation activation, double parameter0, double parameter1) {
    switch (activation) {
        case Activation::None:
            return IdentityActivation{};
        case Activation::Absolute:
            return AbsoluteActivation{};
        case Activation::ClippedRelu:
            return ClippedReluActivation{parameter0, parameter1};
        case Activation::Relu:
            return ReluActivation{};
        case Activation::Gelu:
            return GeluActivation{};
        case Activation::GeluDerivative:
            return GeluDerivativeActivation{};
        case Activation::GeluScaling:
            return GeluScalingActivation{parameter0};
        case Activation::LeakyRelu:
            return LeakyReluActivation{parameter0};
        case Activation::ReluDerivative:
            return ReluDerivativeActivation{};
        case Activation::Sigmoid:
            return SigmoidActivation{};
        case Activation::Tanh:
            return TanhActivation{parameter0, parameter1};
        case Activation::Silu:
            return SiluActivation{};
        case Activation::Swish:
            return SwishActivation{parameter0};
        case Activation::Clamp:
            return ClampActivation{parameter0, parameter1};
    }
    throw std::invalid_argument("Unsupported Python activation.");
}

Tensor referenceSoftmaxOwned(Tensor input, ScalarType outputType, ScalarType accumulatorType,
                             size_t axis) {
    return referenceSoftmax(std::move(input), axis, outputType, accumulatorType);
}

LayerNormOutputs referenceLayerNormOwned(Tensor input, ScalarType outputType,
                                         ScalarType statisticsType, ScalarType accumulatorType,
                                         size_t axis, double epsilon, std::optional<Tensor> gamma,
                                         std::optional<Tensor> beta) {
    LayerNormOptions options;
    options.axis = axis;
    options.accumulatorType = accumulatorType;
    options.gamma = std::move(gamma);
    options.beta = std::move(beta);
    options.epsilon = epsilon;
    return referenceLayerNorm(
        std::move(input),
        {.output = outputType, .mean = statisticsType, .inverseVariance = statisticsType}, options);
}

EpilogueOutputs referenceEpilogueOwned(Tensor input, ScalarType outputType, ScalarType computeType,
                                       std::optional<Tensor> bias, Activation activation,
                                       ActivationApplication activationApplication,
                                       std::optional<Tensor> auxiliaryInput,
                                       std::optional<ScalarType> auxiliaryOutputType,
                                       std::optional<Tensor> gateResidual, nb::object outputScale,
                                       nb::object auxiliaryScale, nb::object activationParameter0,
                                       nb::object activationParameter1,
                                       OutputConversion outputConversion, bool includeRawOutput,
                                       bool includeAmax, OutputSelection outputSelection) {
    EpilogueOptions options(computeType);
    options.auxiliaryInput = std::move(auxiliaryInput);
    options.gateResidual = std::move(gateResidual);
    options.bias = std::move(bias);
    options.outputScale = scalarFromPython(outputScale);
    options.auxiliaryScale = scalarFromPython(auxiliaryScale);
    options.outputConversion = outputConversion;
    options.activation = activationFunction(activation, nb::cast<double>(activationParameter0),
                                            nb::cast<double>(activationParameter1));
    options.activationApplication = activationApplication;
    options.outputSelection = std::move(outputSelection);
    return referenceEpilogue(
        std::move(input),
        {.output = outputType,
         .rawOutput = includeRawOutput ? std::optional(computeType) : std::nullopt,
         .auxiliaryOutput = auxiliaryOutputType,
         .amax = includeAmax ? std::optional(computeType) : std::nullopt},
        options);
}

Tensor referenceSumOwned(const Tensor& input, ScalarType outputType, ScalarType accumulatorType,
                         std::vector<size_t> axes) {
    return referenceSum(input, std::move(axes), outputType, accumulatorType);
}

Tensor referenceMaximumAbsoluteOwned(const Tensor& input, ScalarType outputType,
                                     ScalarType accumulatorType) {
    return referenceMaximumAbsolute(input, outputType, accumulatorType);
}

StructuredSparseTensor applyStructuredSparsityOwned(Tensor input, StructuredSparsityPattern pattern,
                                                    bool emitTwoOfFourMetadata) {
    return applyStructuredSparsity(
        std::move(input), std::move(pattern),
        {.retainedIndices = true, .twoOfFourMetadata = emitTwoOfFourMetadata});
}

}  // namespace

void registerOperationBindings(nb::module_& module) {
    nb::class_<LayerNormOutputs>(module, "LayerNormOutputs")
        .def_prop_ro(
            "output",
            [](const LayerNormOutputs& outputs) -> const Tensor& { return outputs.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "mean",
            [](const LayerNormOutputs& outputs) -> const std::optional<Tensor>& {
                return outputs.mean;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "inverse_variance",
            [](const LayerNormOutputs& outputs) -> const std::optional<Tensor>& {
                return outputs.inverseVariance;
            },
            nb::rv_policy::reference_internal);

    nb::class_<EpilogueOutputs>(module, "EpilogueOutputs")
        .def_prop_ro(
            "output",
            [](const EpilogueOutputs& outputs) -> const Tensor& { return outputs.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "raw_output",
            [](const EpilogueOutputs& outputs) -> const std::optional<Tensor>& {
                return outputs.rawOutput;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "auxiliary_output",
            [](const EpilogueOutputs& outputs) -> const std::optional<Tensor>& {
                return outputs.auxiliaryOutput;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "amax",
            [](const EpilogueOutputs& outputs) -> const std::optional<Tensor>& {
                return outputs.amax;
            },
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
    nb::class_<StructuredSparsitySliceRange>(module, "StructuredSparsitySliceRange")
        .def(nb::init<>())
        .def_rw("first_slice", &StructuredSparsitySliceRange::firstSlice)
        .def_rw("slice_count", &StructuredSparsitySliceRange::sliceCount);
    nb::class_<StructuredSparseTensor>(module, "StructuredSparseTensor")
        .def_prop_ro(
            "pruned",
            [](const StructuredSparseTensor& outputs) -> const Tensor& { return outputs.pruned; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "compressed",
            [](const StructuredSparseTensor& outputs) -> const Tensor& {
                return outputs.compressed;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "retained_indices",
            [](const StructuredSparseTensor& outputs) -> const std::optional<Tensor>& {
                return outputs.retainedIndices;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "two_of_four_metadata",
            [](const StructuredSparseTensor& outputs) -> const std::optional<Tensor>& {
                return outputs.twoOfFourMetadata;
            },
            nb::rv_policy::reference_internal);

    module.def("add", &addOwned, "left"_a, "right"_a, "output_type"_a = ScalarType::Float32,
               "compute_type"_a = ScalarType::Float32);
    module.def("add_into", &addIntoBound, "left"_a, "right"_a, "output"_a,
               "compute_type"_a = ScalarType::Float32);
    module.def("multiply", &multiplyOwned, "left"_a, "right"_a,
               "output_type"_a = ScalarType::Float32, "compute_type"_a = ScalarType::Float32);
    module.def("multiply_into", &multiplyIntoBound, "left"_a, "right"_a, "output"_a,
               "compute_type"_a = ScalarType::Float32);
    module.def("absolute", [](Tensor input) { return absolute(std::move(input)); }, "input"_a);
    module.def("relu", [](Tensor input) { return relu(std::move(input)); }, "input"_a);
    module.def("gelu", [](Tensor input) { return gelu(std::move(input)); }, "input"_a);
    module.def("sigmoid", [](Tensor input) { return sigmoid(std::move(input)); }, "input"_a);
    module.def(
        "tanh",
        [](Tensor input, double inputScale, double outputScale) {
            return tanh(std::move(input), inputScale, outputScale);
        },
        "input"_a, "input_scale"_a = 1.0, "output_scale"_a = 1.0);
    module.def("silu", [](Tensor input) { return silu(std::move(input)); }, "input"_a);
    module.def(
        "swish", [](Tensor input, double beta) { return swish(std::move(input), beta); }, "input"_a,
        "beta"_a = 1.0);
    module.def(
        "clip",
        [](Tensor input, double minimum, double maximum) {
            return clip(std::move(input), minimum, maximum);
        },
        "input"_a, "minimum"_a, "maximum"_a);

    module.def("reference_softmax", &referenceSoftmaxOwned, "input"_a,
               "output_type"_a = ScalarType::Float32, "accumulator_type"_a = ScalarType::Float32,
               "axis"_a = 0);
    module.def("reference_softmax_into", &referenceSoftmaxInto, "input"_a, "output"_a, "axis"_a = 0,
               "accumulator_type"_a = ScalarType::Float32);

    module.def("reference_layer_norm", &referenceLayerNormOwned, "input"_a,
               "output_type"_a = ScalarType::Float32, "statistics_type"_a = ScalarType::Float32,
               "accumulator_type"_a = ScalarType::Float32, "axis"_a = 0, "epsilon"_a = 1e-5,
               "gamma"_a = std::optional<Tensor>{}, "beta"_a = std::optional<Tensor>{});

    module.def("reference_reduce", &referenceReduce, "input"_a, "axes"_a, "operation"_a,
               "output_type"_a, "accumulator_type"_a);
    module.def("reference_reduce_into", &referenceReduceInto, "input"_a, "output"_a, "axes"_a,
               "operation"_a, "accumulator_type"_a);
    module.def("reference_sum", &referenceSumOwned, "input"_a, "output_type"_a,
               "accumulator_type"_a, "axes"_a);
    module.def("reference_sum_into", &referenceSumInto, "input"_a, "output"_a, "axes"_a,
               "accumulator_type"_a);
    module.def("reference_maximum_absolute", &referenceMaximumAbsoluteOwned, "input"_a,
               "output_type"_a, "accumulator_type"_a);
    module.def("reference_maximum_absolute_into", &referenceMaximumAbsoluteInto, "input"_a,
               "output"_a, "accumulator_type"_a);

    module.def("reference_epilogue", &referenceEpilogueOwned, "input"_a, "output_type"_a,
               "compute_type"_a, "bias"_a = std::optional<Tensor>{},
               "activation"_a = Activation::None,
               "activation_application"_a = ActivationApplication::Forward,
               "auxiliary_input"_a = std::optional<Tensor>{},
               "auxiliary_output_type"_a = std::optional<ScalarType>{},
               "gate_residual"_a = std::optional<Tensor>{},
               "output_scale"_a = std::complex<double>(1.0, 0.0),
               "auxiliary_scale"_a = std::complex<double>(1.0, 0.0),
               "activation_parameter0"_a = 0.0, "activation_parameter1"_a = 0.0,
               "output_conversion"_a = OutputConversion::Default, "include_raw_output"_a = false,
               "include_amax"_a = false, "output_selection"_a = OutputSelection::all());

    module.def("apply_structured_sparsity", &applyStructuredSparsityOwned, "input"_a, "pattern"_a,
               "emit_two_of_four_metadata"_a = false);
    module.def("encode_two_of_four_metadata", &encodeTwoOfFourMetadata, "retained_indices"_a,
               "axis"_a);
}
}  // namespace roc::host_numerics::python_bindings
