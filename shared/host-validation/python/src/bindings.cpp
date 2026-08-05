// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <cstring>
#include <optional>
#include <roc/host_validation/backends/tiled.hpp>
#include <roc/host_validation/validation.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::host_validation;

namespace {
struct PythonEpilogueResult {
    Tensor output;
    std::optional<Tensor> rawOutput;
    std::optional<Tensor> auxiliaryOutput;
    std::optional<Tensor> amax;
};

std::vector<size_t> dimensions(const Shape& shape) {
    return {shape.dimensions().begin(), shape.dimensions().end()};
}

std::vector<ptrdiff_t> strides(const Layout& layout) {
    return {layout.strides().begin(), layout.strides().end()};
}

template <typename Function>
void forEachIndex(const Shape& shape, Function&& function) {
    const size_t count = shape.elementCount();
    std::vector<size_t> indices(shape.rank(), 0);
    for (size_t linear = 0; linear < count; ++linear) {
        function(std::span<const size_t>(indices));
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            if (++indices[index] < shape[index]) break;
            indices[index] = 0;
        }
    }
}

template <typename Value, typename Tag>
Value loadTensorValue(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    return detail::decodeScalarKnown<Tag::type, Value>(storage, logicalOffset);
}

template <typename Value>
void appendTensorValues(nb::list& result, const Tensor& tensor) {
    using LoadFunction = Value (*)(std::span<const std::byte>, ptrdiff_t);
    const LoadFunction load = visitScalarType(
        tensor.type(), []<typename Tag>() -> LoadFunction { return &loadTensorValue<Value, Tag>; });
    const auto storage = tensor.storage();
    const auto& layout = tensor.layout();
    forEachIndex(tensor.shape(), [&](std::span<const size_t> indices) {
        result.append(load(storage, layout.elementOffset(indices)));
    });
}

nb::list tensorValues(const Tensor& tensor) {
    nb::list result;
    const ScalarCategory category = scalarTypeInfo(tensor.type()).category;
    switch (category) {
        case ScalarCategory::Boolean:
            appendTensorValues<bool>(result, tensor);
            break;
        case ScalarCategory::SignedInteger:
            appendTensorValues<int64_t>(result, tensor);
            break;
        case ScalarCategory::UnsignedInteger:
            appendTensorValues<uint64_t>(result, tensor);
            break;
        case ScalarCategory::Complex:
            appendTensorValues<std::complex<double>>(result, tensor);
            break;
        case ScalarCategory::FloatingPoint:
        case ScalarCategory::Scale:
            appendTensorValues<double>(result, tensor);
            break;
    }
    return result;
}

Tensor tensorFromStorage(ScalarType type, std::vector<size_t> dimensions, nb::bytes rawStorage,
                         std::optional<std::vector<ptrdiff_t>> tensorStrides, ptrdiff_t offset) {
    std::vector<std::byte> storage(rawStorage.size());
    std::memcpy(storage.data(), rawStorage.c_str(), rawStorage.size());
    Shape shape(std::move(dimensions));
    Layout layout = Layout::contiguous(shape);
    if (tensorStrides || offset != 0) {
        std::vector<ptrdiff_t> selectedStrides =
            tensorStrides ? std::move(*tensorStrides) : strides(layout);
        layout = Layout(std::move(shape), std::move(selectedStrides), offset);
    }
    return Tensor::fromStorage(type, std::move(layout), std::move(storage));
}

Tensor referenceGemmOwned(const Tensor& a, const Tensor& b, const Tensor& c, ScalarType outputType,
                          ScalarType accumulatorType, std::complex<double> alpha,
                          std::complex<double> beta, std::optional<ScalarType> computeTypeA,
                          std::optional<ScalarType> computeTypeB, MathMode mathMode,
                          Activation activation, double activationParameter0,
                          double activationParameter1, OutputSelection outputSelection,
                          GemmBackend backend) {
    if (a.shape().rank() != 2 || b.shape().rank() != 2)
        throw std::invalid_argument("Python reference_gemm requires rank-2 A and B tensors.");

    Tensor d(outputType, Shape{a.shape()[0], b.shape()[1]});
    GemmOperand operandA(a.view());
    GemmOperand operandB(b.view());
    operandA.computeType = computeTypeA;
    operandB.computeType = computeTypeB;
    GemmProblem problem(std::move(operandA), std::move(operandB), c.view(), d.mutableView(),
                        accumulatorType);
    problem.mathMode = mathMode;
    problem.epilogue.alpha = alpha;
    problem.epilogue.beta = beta;
    problem.epilogue.activation = activation;
    problem.epilogue.activationParameter0 = activationParameter0;
    problem.epilogue.activationParameter1 = activationParameter1;
    problem.outputSelection = std::move(outputSelection);
    if (backend == GemmBackend::Canonical) {
        referenceGemm(problem);
    } else if (backend == GemmBackend::Tiled) {
        static const TiledGemmBackend tiled;
        referenceGemm(problem, {
                                   .backend = GemmBackend::Tiled,
                                   .requireRequestedBackend = true,
                                   .backendImplementation = &tiled,
                               });
    } else {
        throw std::invalid_argument("Python reference_gemm exposes Canonical and Tiled backends.");
    }
    return d;
}

PythonEpilogueResult referenceEpilogueOwned(
    const Tensor& input, ScalarType outputType, ScalarType computeType, std::optional<Tensor> bias,
    MatrixAxis biasAxis, Activation activation, ActivationApplication activationApplication,
    std::optional<Tensor> auxiliaryInput, std::optional<ScalarType> auxiliaryOutputType,
    std::optional<Tensor> gateResidual, std::complex<double> outputScale,
    std::complex<double> auxiliaryScale, double activationParameter0, double activationParameter1,
    bool includeRawOutput, bool includeAmax, OutputSelection outputSelection) {
    Tensor output(outputType, input.shape());
    std::optional<Tensor> rawOutput;
    std::optional<Tensor> auxiliaryOutput;
    std::optional<Tensor> amax;
    if (includeRawOutput) rawOutput.emplace(computeType, input.shape());
    if (auxiliaryOutputType) auxiliaryOutput.emplace(*auxiliaryOutputType, input.shape());
    if (includeAmax) amax.emplace(computeType, Shape{1});

    EpilogueProblem problem(input.view(), output.mutableView(), computeType);
    if (rawOutput) problem.rawOutput = rawOutput->mutableView();
    if (auxiliaryOutput) problem.auxiliaryOutput = auxiliaryOutput->mutableView();
    if (auxiliaryInput) problem.auxiliaryInput = auxiliaryInput->view();
    if (gateResidual) problem.gateResidual = gateResidual->view();
    if (amax) problem.amax = amax->mutableView();
    if (bias) problem.bias = VectorBinding{bias->view(), biasAxis};
    problem.outputScale = outputScale;
    problem.auxiliaryScale = auxiliaryScale;
    problem.activation = activation;
    problem.activationApplication = activationApplication;
    problem.activationParameter0 = activationParameter0;
    problem.activationParameter1 = activationParameter1;
    problem.outputSelection = std::move(outputSelection);
    referenceEpilogue(problem);
    return {
        .output = std::move(output),
        .rawOutput = std::move(rawOutput),
        .auxiliaryOutput = std::move(auxiliaryOutput),
        .amax = std::move(amax),
    };
}

Tensor referenceSumOwned(const Tensor& input, ScalarType outputType, ScalarType accumulatorType,
                         std::vector<size_t> axes) {
    std::vector<bool> reduced(input.shape().rank(), false);
    for (const size_t axis : axes) {
        if (axis >= input.shape().rank())
            throw std::out_of_range("Python reference_sum axis exceeds input rank.");
        if (reduced[axis]) throw std::invalid_argument("Python reference_sum axes must be unique.");
        reduced[axis] = true;
    }

    std::vector<size_t> outputDimensions;
    for (size_t dimension = 0; dimension < input.shape().rank(); ++dimension) {
        if (!reduced[dimension]) outputDimensions.push_back(input.shape()[dimension]);
    }

    Tensor output(outputType, Shape(std::move(outputDimensions)));
    referenceSum(
        ReductionProblem(input.view(), output.mutableView(), accumulatorType, std::move(axes)));
    return output;
}

Tensor generateOwned(ScalarType type, std::vector<size_t> shape, const GenerationOptions& options) {
    Tensor output(type, Shape(std::move(shape)));
    generate(output.mutableView(), options);
    return output;
}
}  // namespace

NB_MODULE(_roc_host_validation, module) {
    nb::enum_<ScalarCategory>(module, "ScalarCategory")
        .value("Boolean", ScalarCategory::Boolean)
        .value("SignedInteger", ScalarCategory::SignedInteger)
        .value("UnsignedInteger", ScalarCategory::UnsignedInteger)
        .value("FloatingPoint", ScalarCategory::FloatingPoint)
        .value("Complex", ScalarCategory::Complex)
        .value("Scale", ScalarCategory::Scale);

    nb::enum_<ScalarType>(module, "ScalarType")
        .value("Boolean", ScalarType::Boolean)
        .value("UInt8", ScalarType::UInt8)
        .value("Int8", ScalarType::Int8)
        .value("UInt16", ScalarType::UInt16)
        .value("Int16", ScalarType::Int16)
        .value("UInt32", ScalarType::UInt32)
        .value("Int32", ScalarType::Int32)
        .value("UInt64", ScalarType::UInt64)
        .value("Int64", ScalarType::Int64)
        .value("Float16", ScalarType::Float16)
        .value("BFloat16", ScalarType::BFloat16)
        .value("Float32", ScalarType::Float32)
        .value("Float64", ScalarType::Float64)
        .value("ComplexFloat32", ScalarType::ComplexFloat32)
        .value("ComplexFloat64", ScalarType::ComplexFloat64)
        .value("Float8E4M3", ScalarType::Float8E4M3)
        .value("Float8E5M2", ScalarType::Float8E5M2)
        .value("Float8E4M3Fnuz", ScalarType::Float8E4M3Fnuz)
        .value("Float8E5M2Fnuz", ScalarType::Float8E5M2Fnuz)
        .value("Float6E2M3", ScalarType::Float6E2M3)
        .value("Float6E3M2", ScalarType::Float6E3M2)
        .value("Float4E2M1", ScalarType::Float4E2M1)
        .value("Int4", ScalarType::Int4)
        .value("Int12", ScalarType::Int12)
        .value("E8M0", ScalarType::E8M0)
        .value("E5M3", ScalarType::E5M3);

    nb::enum_<MathMode>(module, "MathMode")
        .value("Default", MathMode::Default)
        .value("XFloat32", MathMode::XFloat32);

    nb::enum_<GemmBackend>(module, "GemmBackend")
        .value("Canonical", GemmBackend::Canonical)
        .value("Tiled", GemmBackend::Tiled);

    nb::enum_<MatrixAxis>(module, "MatrixAxis")
        .value("Row", MatrixAxis::Row)
        .value("Column", MatrixAxis::Column);

    nb::enum_<Activation>(module, "Activation")
        .value("None_", Activation::None)
        .value("Relu", Activation::Relu)
        .value("Gelu", Activation::Gelu)
        .value("Silu", Activation::Silu)
        .value("Clamp", Activation::Clamp);

    nb::enum_<DataPattern>(module, "DataPattern")
        .value("Zero", DataPattern::Zero)
        .value("RandomInteger", DataPattern::RandomInteger)
        .value("UniformInteger", DataPattern::UniformInteger)
        .value("AlternatingRandomInteger", DataPattern::AlternatingRandomInteger)
        .value("UniformReal", DataPattern::UniformReal)
        .value("Sine", DataPattern::Sine)
        .value("Cosine", DataPattern::Cosine)
        .value("Constant", DataPattern::Constant);

    nb::enum_<GenerationPattern>(module, "GenerationPattern")
        .value("Zero", GenerationPattern::Zero)
        .value("Constant", GenerationPattern::Constant)
        .value("UniformInteger", GenerationPattern::UniformInteger)
        .value("UniformReal", GenerationPattern::UniformReal)
        .value("Normal", GenerationPattern::Normal)
        .value("Sine", GenerationPattern::Sine)
        .value("Cosine", GenerationPattern::Cosine)
        .value("AbsoluteSine", GenerationPattern::AbsoluteSine)
        .value("AbsoluteCosine", GenerationPattern::AbsoluteCosine)
        .value("SerialIndex", GenerationPattern::SerialIndex)
        .value("SerialDimension", GenerationPattern::SerialDimension)
        .value("Identity", GenerationPattern::Identity)
        .value("CheckerboardUniformInteger", GenerationPattern::CheckerboardUniformInteger);

    nb::enum_<LogicalIndexOrder>(module, "LogicalIndexOrder")
        .value("FirstDimensionFastest", LogicalIndexOrder::FirstDimensionFastest)
        .value("LastDimensionFastest", LogicalIndexOrder::LastDimensionFastest);

    nb::enum_<ActivationApplication>(module, "ActivationApplication")
        .value("Forward", ActivationApplication::Forward)
        .value("Gradient", ActivationApplication::Gradient);

    nb::enum_<OutputSelectionKind>(module, "OutputSelectionKind")
        .value("All", OutputSelectionKind::All)
        .value("Strided", OutputSelectionKind::Strided)
        .value("Explicit", OutputSelectionKind::Explicit);

    nb::class_<OutputSelection>(module, "OutputSelection")
        .def_static("all", &OutputSelection::all)
        .def_static("strided", &OutputSelection::strided, "first"_a, "stride"_a)
        .def_static("explicit_indices", &OutputSelection::explicitIndices)
        .def_static("prime_stride", &OutputSelection::primeStride, "logical_elements"_a,
                    "allocated_elements"_a, "requested_elements"_a)
        .def_prop_ro("kind", &OutputSelection::kind)
        .def_prop_ro("selects_all", &OutputSelection::selectsAll)
        .def("indices", &OutputSelection::indices, "logical_elements"_a);

    nb::class_<ScalarTypeInfo>(module, "ScalarTypeInfo")
        .def_prop_ro("name", [](const ScalarTypeInfo& info) { return std::string(info.name); })
        .def_ro("category", &ScalarTypeInfo::category)
        .def_ro("storage_bits", &ScalarTypeInfo::storageBits)
        .def_ro("exponent_bits", &ScalarTypeInfo::exponentBits)
        .def_ro("mantissa_bits", &ScalarTypeInfo::mantissaBits)
        .def_ro("exponent_bias", &ScalarTypeInfo::exponentBias)
        .def_ro("supports_nan", &ScalarTypeInfo::supportsNaN)
        .def_ro("supports_infinity", &ScalarTypeInfo::supportsInfinity)
        .def("is_packed", &ScalarTypeInfo::isPacked);

    nb::class_<Shape>(module, "Shape")
        .def(nb::init<std::vector<size_t>>())
        .def_prop_ro("rank", &Shape::rank)
        .def_prop_ro("dimensions", &dimensions)
        .def_prop_ro("element_count", &Shape::elementCount);

    nb::class_<Layout>(module, "Layout")
        .def(nb::init<Shape, std::vector<ptrdiff_t>, ptrdiff_t>(), "shape"_a, "strides"_a,
             "offset"_a = 0)
        .def_static("contiguous", &Layout::contiguous)
        .def_prop_ro("shape", &Layout::shape, nb::rv_policy::reference_internal)
        .def_prop_ro("strides", &strides)
        .def_prop_ro("offset", &Layout::offset);

    nb::class_<Tensor>(module, "Tensor")
        .def(nb::init<ScalarType, Shape>())
        .def_static(
            "from_values",
            [](ScalarType type, std::vector<size_t> shape, const std::vector<double>& values) {
                return Tensor::fromValues(type, Shape(std::move(shape)),
                                          std::span<const double>(values));
            })
        .def_static(
            "from_signed_values",
            [](ScalarType type, std::vector<size_t> shape, const std::vector<int64_t>& values) {
                return Tensor::fromValues(type, Shape(std::move(shape)),
                                          std::span<const int64_t>(values));
            })
        .def_static(
            "from_unsigned_values",
            [](ScalarType type, std::vector<size_t> shape, const std::vector<uint64_t>& values) {
                return Tensor::fromValues(type, Shape(std::move(shape)),
                                          std::span<const uint64_t>(values));
            })
        .def_static("from_complex_values",
                    [](ScalarType type, std::vector<size_t> shape,
                       const std::vector<std::complex<double>>& values) {
                        return Tensor::fromValues(type, Shape(std::move(shape)),
                                                  std::span<const std::complex<double>>(values));
                    })
        .def_static("from_storage", &tensorFromStorage, "type"_a, "shape"_a, "storage"_a,
                    "strides"_a = std::optional<std::vector<ptrdiff_t>>{}, "offset"_a = 0)
        .def_prop_ro("type", &Tensor::type)
        .def_prop_ro("shape", [](const Tensor& tensor) { return dimensions(tensor.shape()); })
        .def_prop_ro("strides", [](const Tensor& tensor) { return strides(tensor.layout()); })
        .def_prop_ro("offset", [](const Tensor& tensor) { return tensor.layout().offset(); })
        .def_prop_ro("size", &Tensor::size)
        .def_prop_ro("storage",
                     [](const Tensor& tensor) {
                         const auto storage = tensor.storage();
                         return nb::bytes(reinterpret_cast<const char*>(storage.data()),
                                          storage.size());
                     })
        .def_prop_ro("values", &tensorValues);

    nb::class_<ComparisonOptions>(module, "ComparisonOptions")
        .def(nb::init<>())
        .def_rw("absolute_tolerance", &ComparisonOptions::absoluteTolerance)
        .def_rw("relative_tolerance", &ComparisonOptions::relativeTolerance)
        .def_rw("symmetric_relative_tolerance", &ComparisonOptions::symmetricRelativeTolerance)
        .def_rw("max_reported_mismatches", &ComparisonOptions::maxReportedMismatches);

    nb::class_<Mismatch>(module, "Mismatch")
        .def_ro("index", &Mismatch::index)
        .def_ro("observed", &Mismatch::observed)
        .def_ro("expected", &Mismatch::expected)
        .def_ro("absolute_difference", &Mismatch::absoluteDifference);

    nb::class_<ComparisonResult>(module, "ComparisonResult")
        .def_ro("compared", &ComparisonResult::compared)
        .def_ro("mismatches", &ComparisonResult::mismatches)
        .def_ro("max_absolute_difference", &ComparisonResult::maxAbsoluteDifference)
        .def_ro("reported_mismatches", &ComparisonResult::reportedMismatches)
        .def_prop_ro("passed", &ComparisonResult::passed);

    nb::class_<GenerationPatternSpec>(module, "GenerationPatternSpec")
        .def(nb::init<>())
        .def_rw("pattern", &GenerationPatternSpec::pattern)
        .def_rw("parameter0", &GenerationPatternSpec::parameter0)
        .def_rw("parameter1", &GenerationPatternSpec::parameter1)
        .def_rw("stream", &GenerationPatternSpec::stream)
        .def_rw("dimension", &GenerationPatternSpec::dimension);

    nb::class_<GenerationOptions>(module, "GenerationOptions")
        .def(nb::init<>())
        .def_rw("seed", &GenerationOptions::seed)
        .def_rw("index_order", &GenerationOptions::indexOrder)
        .def_rw("real", &GenerationOptions::real)
        .def_rw("imaginary", &GenerationOptions::imaginary);

    nb::class_<PythonEpilogueResult>(module, "EpilogueResult")
        .def_prop_ro(
            "output",
            [](const PythonEpilogueResult& result) -> const Tensor& { return result.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "raw_output",
            [](const PythonEpilogueResult& result) -> const std::optional<Tensor>& {
                return result.rawOutput;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "auxiliary_output",
            [](const PythonEpilogueResult& result) -> const std::optional<Tensor>& {
                return result.auxiliaryOutput;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "amax",
            [](const PythonEpilogueResult& result) -> const std::optional<Tensor>& {
                return result.amax;
            },
            nb::rv_policy::reference_internal);

    module.def("scalar_type_info", [](ScalarType type) { return scalarTypeInfo(type); });
    module.def(
        "fill",
        [](Tensor& tensor, DataPattern pattern, uint32_t seed, double parameter0,
           double parameter1) -> Tensor& {
            RandomGenerator generator(seed);
            fill(tensor.mutableView(), pattern, generator, parameter0, parameter1);
            return tensor;
        },
        "tensor"_a, "pattern"_a, "seed"_a, "parameter0"_a = 0.0, "parameter1"_a = 0.0,
        nb::rv_policy::reference);
    module.def("generate_tensor", &generateOwned, "type"_a, "shape"_a, "options"_a);
    module.def(
        "compare",
        [](const Tensor& observed, const Tensor& expected, const ComparisonOptions& options) {
            return compare(observed.view(), expected.view(), options);
        },
        "observed"_a, "expected"_a, "options"_a = ComparisonOptions{});
    module.def("reference_gemm", &referenceGemmOwned, "a"_a, "b"_a, "c"_a, "output_type"_a,
               "accumulator_type"_a, "alpha"_a = 1.0, "beta"_a = 0.0,
               "compute_type_a"_a = std::optional<ScalarType>{},
               "compute_type_b"_a = std::optional<ScalarType>{}, "math_mode"_a = MathMode::Default,
               "activation"_a = Activation::None, "activation_parameter0"_a = 0.0,
               "activation_parameter1"_a = 0.0, "output_selection"_a = OutputSelection::all(),
               "backend"_a = GemmBackend::Canonical);
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
               "include_raw_output"_a = false, "include_amax"_a = false,
               "output_selection"_a = OutputSelection::all());
    module.def("reference_sum", &referenceSumOwned, "input"_a, "output_type"_a,
               "accumulator_type"_a, "axes"_a);
}
