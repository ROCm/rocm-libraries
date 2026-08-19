// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "bindings.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <roc/host_validation/backends/blocked.hpp>
#include <roc/host_validation/validation.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::host_validation;

namespace {
struct PythonGemmResult {
    Tensor output;
    GemmRunInfo runInfo;
};

struct PythonVectorBinding {
    PythonVectorBinding(Tensor tensor, MatrixAxis selectedAxis)
        : values(std::move(tensor)), axis(selectedAxis) {}

    Tensor values;
    MatrixAxis axis = MatrixAxis::Row;
};

struct PythonBlockScaleBinding {
    PythonBlockScaleBinding(Tensor tensor, size_t selectedBlockSize)
        : values(std::move(tensor)), blockSize(selectedBlockSize) {}

    Tensor values;
    size_t blockSize;
};

struct PythonGemmOperand {
    explicit PythonGemmOperand(Tensor tensor) : values(std::move(tensor)) {}

    Tensor values;
    std::optional<ScalarType> computeType;
    std::vector<PythonVectorBinding> preQuantizationScales;
    std::optional<PythonBlockScaleBinding> blockScale;
    bool conjugate = false;
};

struct PythonGemmEpilogue {
    std::complex<double> alpha = {1.0, 0.0};
    std::complex<double> beta = {0.0, 0.0};
    std::optional<PythonVectorBinding> bias;
    std::optional<PythonVectorBinding> scaleAlpha;
    std::optional<Tensor> scaleA;
    std::optional<Tensor> scaleB;
    std::complex<double> outputScale = {1.0, 0.0};
    OutputConversion outputConversion = OutputConversion::Default;
    Activation activation = Activation::None;
    double activationParameter0 = 0.0;
    double activationParameter1 = 0.0;
};

struct PythonGemmRequest {
    PythonGemmRequest(PythonGemmOperand operandA, PythonGemmOperand operandB,
                      std::optional<Tensor> tensorC, ScalarType selectedOutputType,
                      ScalarType selectedAccumulatorType,
                      std::optional<Layout> selectedOutputLayout)
        : a(std::move(operandA)),
          b(std::move(operandB)),
          c(std::move(tensorC)),
          outputType(selectedOutputType),
          outputLayout(std::move(selectedOutputLayout)),
          accumulatorType(selectedAccumulatorType) {}

    PythonGemmOperand a;
    PythonGemmOperand b;
    std::optional<Tensor> c;
    ScalarType outputType;
    std::optional<Layout> outputLayout;
    ScalarType accumulatorType;
    AccumulationRounding accumulationRounding = AccumulationRounding::TypeDefault;
    MathMode mathMode = MathMode::Default;
    PythonGemmEpilogue epilogue;
    OutputSelection outputSelection = OutputSelection::all();
};

struct PythonLayerNormResult {
    Tensor output;
    Tensor mean;
    Tensor inverseVariance;
    LayerNormRunInfo runInfo;
};

struct PythonEpilogueResult {
    Tensor output;
    std::optional<Tensor> rawOutput;
    std::optional<Tensor> auxiliaryOutput;
    std::optional<Tensor> amax;
};

struct PythonStructuredSparsityResult {
    Tensor pruned;
    Tensor compressed;
    Tensor retainedIndices;
    std::optional<Tensor> twoOfFourMetadata;
    StructuredSparsityRunInfo runInfo;
};

struct PythonTwoOfFourMetadataResult {
    Tensor metadata;
    TwoOfFourMetadataRunInfo runInfo;
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
void appendTensorValues(nb::list& result, Tensor tensor) {
    using LoadFunction = Value (*)(std::span<const std::byte>, ptrdiff_t);
    const LoadFunction load = visitScalarType(
        tensor.type(), []<typename Tag>() -> LoadFunction { return &loadTensorValue<Value, Tag>; });
    const auto storage = tensor.storage();
    const auto& layout = tensor.layout();
    forEachIndex(tensor.shape(), [&](std::span<const size_t> indices) {
        result.append(load(storage, layout.elementOffset(indices)));
    });
}

nb::list tensorValues(Tensor tensor) {
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

struct NumpyStorageType {
    ScalarType type;
    size_t itemSize;
};

class ReadOnlyPythonBuffer {
   public:
    explicit ReadOnlyPythonBuffer(nb::handle owner) {
        if (PyObject_GetBuffer(owner.ptr(), &m_view, PyBUF_RECORDS_RO) != 0)
            throw nb::python_error();
    }

    ReadOnlyPythonBuffer(const ReadOnlyPythonBuffer&) = delete;
    ReadOnlyPythonBuffer& operator=(const ReadOnlyPythonBuffer&) = delete;

    ~ReadOnlyPythonBuffer() {
        PyBuffer_Release(&m_view);
    }

    const Py_buffer& view() const {
        return m_view;
    }

   private:
    Py_buffer m_view{};
};

NumpyStorageType numpyStorageType(nb::handle array) {
    const nb::object dtype = array.attr("dtype");
    if (!nb::cast<bool>(dtype.attr("isnative")))
        throw nb::type_error("Tensor.from_numpy requires a native-endian NumPy dtype.");

    const std::string kind = nb::cast<std::string>(dtype.attr("kind"));
    const size_t itemSize = nb::cast<size_t>(dtype.attr("itemsize"));
    if (kind.size() == 1) {
        switch (kind[0]) {
            case 'b':
                if (itemSize == 1) return {ScalarType::Boolean, itemSize};
                break;
            case 'u':
                if (itemSize == 1) return {ScalarType::UInt8, itemSize};
                if (itemSize == 2) return {ScalarType::UInt16, itemSize};
                if (itemSize == 4) return {ScalarType::UInt32, itemSize};
                if (itemSize == 8) return {ScalarType::UInt64, itemSize};
                break;
            case 'i':
                if (itemSize == 1) return {ScalarType::Int8, itemSize};
                if (itemSize == 2) return {ScalarType::Int16, itemSize};
                if (itemSize == 4) return {ScalarType::Int32, itemSize};
                if (itemSize == 8) return {ScalarType::Int64, itemSize};
                break;
            case 'f':
                if (itemSize == 2) return {ScalarType::Float16, itemSize};
                if (itemSize == 4) return {ScalarType::Float32, itemSize};
                if (itemSize == 8) return {ScalarType::Float64, itemSize};
                break;
            case 'c':
                if (itemSize == 8) return {ScalarType::ComplexFloat32, itemSize};
                if (itemSize == 16) return {ScalarType::ComplexFloat64, itemSize};
                break;
        }
    }

    throw nb::type_error(
        "Tensor.from_numpy supports only exact native NumPy bool, integer, "
        "float16/32/64, and complex64/128 storage dtypes.");
}

ptrdiff_t checkedElementDelta(ptrdiff_t stride, size_t elementCount) {
    if (stride == 0 || elementCount == 0) return 0;
    if (!std::in_range<ptrdiff_t>(elementCount))
        throw std::overflow_error("NumPy Tensor stride extent exceeds ptrdiff_t.");

    const ptrdiff_t signedElementCount = static_cast<ptrdiff_t>(elementCount);
    if (stride > 0) {
        if (signedElementCount > std::numeric_limits<ptrdiff_t>::max() / stride)
            throw std::overflow_error("NumPy Tensor stride extent overflow.");
    } else if (stride == -1) {
        return -signedElementCount;
    } else if (signedElementCount > std::numeric_limits<ptrdiff_t>::min() / stride) {
        throw std::overflow_error("NumPy Tensor stride extent overflow.");
    }
    return stride * signedElementCount;
}

ptrdiff_t checkedElementOffsetAdd(ptrdiff_t left, ptrdiff_t right) {
    if ((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
        throw std::overflow_error("NumPy Tensor element offset overflow.");
    return left + right;
}

size_t checkedStorageBytes(size_t elementCount, size_t itemSize) {
    if (itemSize != 0 && elementCount > std::numeric_limits<size_t>::max() / itemSize)
        throw std::overflow_error("NumPy Tensor storage byte count overflow.");
    return elementCount * itemSize;
}

Tensor tensorFromNumpy(nb::object array, std::optional<ScalarType> requestedType) {
    const nb::object numpyArrayType = nb::module_::import_("numpy").attr("ndarray");
    if (!nb::isinstance(array, numpyArrayType))
        throw nb::type_error("Tensor.from_numpy requires a NumPy ndarray.");

    const NumpyStorageType storageType = numpyStorageType(array);
    if (requestedType && *requestedType != storageType.type)
        throw std::invalid_argument(
            "Tensor.from_numpy scalar_type must exactly match the NumPy storage dtype.");
    const ScalarType type = requestedType.value_or(storageType.type);

    const ReadOnlyPythonBuffer buffer(array);
    const Py_buffer& view = buffer.view();
    if (view.ndim < 0 || (view.ndim > 0 && (!view.shape || !view.strides)))
        throw std::invalid_argument("NumPy Tensor buffer geometry is invalid.");
    if (view.itemsize <= 0 || static_cast<size_t>(view.itemsize) != storageType.itemSize)
        throw std::invalid_argument("NumPy Tensor buffer item size does not match its dtype.");
    if (!std::in_range<ptrdiff_t>(storageType.itemSize))
        throw std::overflow_error("NumPy Tensor item size exceeds ptrdiff_t.");
    const ptrdiff_t signedItemSize = static_cast<ptrdiff_t>(storageType.itemSize);

    std::vector<size_t> tensorDimensions;
    std::vector<ptrdiff_t> tensorStrides;
    tensorDimensions.reserve(static_cast<size_t>(view.ndim));
    tensorStrides.reserve(static_cast<size_t>(view.ndim));
    bool empty = false;
    for (Py_ssize_t dimension = 0; dimension < view.ndim; ++dimension) {
        if (view.shape[dimension] < 0)
            throw std::invalid_argument("NumPy Tensor extent is negative.");
        if (!std::in_range<ptrdiff_t>(view.strides[dimension]))
            throw std::overflow_error("NumPy Tensor byte stride exceeds ptrdiff_t.");
        const size_t extent = static_cast<size_t>(view.shape[dimension]);
        const ptrdiff_t byteStride = static_cast<ptrdiff_t>(view.strides[dimension]);
        if (byteStride % signedItemSize != 0)
            throw std::invalid_argument(
                "NumPy Tensor byte strides must be exact multiples of item size.");
        tensorDimensions.push_back(extent);
        tensorStrides.push_back(byteStride / signedItemSize);
        empty = empty || extent == 0;
    }

    Shape shape(std::move(tensorDimensions));
    if (empty) {
        return Tensor(type, Layout(std::move(shape), std::move(tensorStrides)),
                      std::span<const std::byte>(reinterpret_cast<const std::byte*>(view.buf), 0));
    }
    if (!view.buf) throw std::invalid_argument("Nonempty NumPy Tensor has a null data pointer.");

    ptrdiff_t lowerOffset = 0;
    ptrdiff_t upperOffset = 0;
    for (size_t dimension = 0; dimension < tensorStrides.size(); ++dimension) {
        const ptrdiff_t delta = checkedElementDelta(tensorStrides[dimension], shape[dimension] - 1);
        if (delta < 0)
            lowerOffset = checkedElementOffsetAdd(lowerOffset, delta);
        else
            upperOffset = checkedElementOffsetAdd(upperOffset, delta);
    }

    if (lowerOffset == std::numeric_limits<ptrdiff_t>::min())
        throw std::overflow_error("NumPy Tensor base offset overflow.");
    const ptrdiff_t normalizedOffset = -lowerOffset;
    const ptrdiff_t normalizedUpper = checkedElementOffsetAdd(upperOffset, normalizedOffset);
    if (!std::in_range<size_t>(normalizedOffset) || !std::in_range<size_t>(normalizedUpper))
        throw std::overflow_error("NumPy Tensor addressed range exceeds size_t.");

    const size_t prefixBytes =
        checkedStorageBytes(static_cast<size_t>(normalizedOffset), storageType.itemSize);
    const size_t normalizedUpperSize = static_cast<size_t>(normalizedUpper);
    if (normalizedUpperSize == std::numeric_limits<size_t>::max())
        throw std::overflow_error("NumPy Tensor addressed range overflow.");
    const size_t storageBytes = checkedStorageBytes(normalizedUpperSize + 1, storageType.itemSize);

    const uintptr_t logicalAddress = reinterpret_cast<uintptr_t>(view.buf);
    if (prefixBytes > logicalAddress)
        throw std::overflow_error("NumPy Tensor base address underflow.");
    const uintptr_t storageAddress = logicalAddress - prefixBytes;
    if (storageBytes > std::numeric_limits<uintptr_t>::max() - storageAddress)
        throw std::overflow_error("NumPy Tensor storage address overflow.");

    return Tensor(type, Layout(std::move(shape), std::move(tensorStrides), normalizedOffset),
                  std::span<const std::byte>(reinterpret_cast<const std::byte*>(storageAddress),
                                             storageBytes));
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

GemmOperand gemmOperand(const PythonGemmOperand& operand) {
    GemmOperand result(operand.values);
    result.computeType = operand.computeType;
    result.conjugate = operand.conjugate;
    result.preQuantizationScales.reserve(operand.preQuantizationScales.size());
    for (const PythonVectorBinding& binding : operand.preQuantizationScales) {
        result.preQuantizationScales.push_back(VectorBinding{binding.values, binding.axis});
    }
    if (operand.blockScale) {
        result.blockScale =
            BlockScaleBinding{operand.blockScale->values, operand.blockScale->blockSize};
    }
    return result;
}

GemmEpilogue gemmEpilogue(const PythonGemmEpilogue& epilogue, ScalarType accumulatorType) {
    GemmEpilogue result(accumulatorType);
    result.alpha = epilogue.alpha;
    result.beta = epilogue.beta;
    if (epilogue.bias) result.bias = VectorBinding{epilogue.bias->values, epilogue.bias->axis};
    if (epilogue.scaleAlpha) {
        result.scaleAlpha = VectorBinding{epilogue.scaleAlpha->values, epilogue.scaleAlpha->axis};
    }
    if (epilogue.scaleA) result.scaleA = epilogue.scaleA;
    if (epilogue.scaleB) result.scaleB = epilogue.scaleB;
    result.outputScale = epilogue.outputScale;
    result.outputConversion = epilogue.outputConversion;
    result.activation = epilogue.activation;
    result.activationParameter0 = epilogue.activationParameter0;
    result.activationParameter1 = epilogue.activationParameter1;
    return result;
}

const GemmBackendImplementation* pythonGemmBackendImplementation(
    GemmBackend backend, bool useBlockedForAutomatic = false) {
    if (backend == GemmBackend::Blocked ||
        (backend == GemmBackend::Automatic && useBlockedForAutomatic)) {
        static const BlockedGemmBackend blocked;
        return &blocked;
    }
    if (backend != GemmBackend::Automatic && backend != GemmBackend::Pointwise)
        throw std::invalid_argument(
            "Python reference_gemm exposes Pointwise and Blocked backends.");
    return nullptr;
}

PythonGemmResult referenceGemmRequestOwned(const PythonGemmRequest& request,
                                           const GemmExecution& execution) {
    if (request.a.values.shape().rank() != 2 || request.b.values.shape().rank() != 2)
        throw std::invalid_argument("Python GemmRequest requires rank-2 A and B tensors.");
    if (!request.c && request.epilogue.beta != std::complex<double>{})
        throw std::invalid_argument("Python GemmRequest requires C when epilogue beta is nonzero.");

    const Shape outputShape{request.a.values.shape()[0], request.b.values.shape()[1]};
    const Layout outputLayout =
        request.outputLayout ? *request.outputLayout : Layout::contiguous(outputShape);
    Tensor output(request.outputType, outputLayout);
    std::optional<Tensor> zeroC;
    if (!request.c) zeroC.emplace(request.outputType, outputShape);
    const Tensor& c = request.c ? *request.c : *zeroC;

    GemmRequest nativeRequest(gemmOperand(request.a), gemmOperand(request.b), c, output,
                              request.accumulatorType);
    nativeRequest.accumulationRounding = request.accumulationRounding;
    nativeRequest.mathMode = request.mathMode;
    nativeRequest.epilogue = gemmEpilogue(request.epilogue, request.accumulatorType);
    nativeRequest.outputSelection = request.outputSelection;

    const GemmResult result =
        referenceGemm(nativeRequest, execution, pythonGemmBackendImplementation(execution.backend));
    return {.output = std::move(output), .runInfo = result.runInfo};
}

PythonGemmResult referenceGemmOwned(
    const Tensor& a, const Tensor& b, const Tensor& c, ScalarType outputType,
    ScalarType accumulatorType, std::complex<double> alpha, std::complex<double> beta,
    std::optional<ScalarType> computeTypeA, std::optional<ScalarType> computeTypeB,
    MathMode mathMode, Activation activation, double activationParameter0,
    double activationParameter1, OutputSelection outputSelection, GemmBackend backend,
    std::optional<Tensor> blockScaleA, std::optional<Tensor> blockScaleB, size_t blockSizeA,
    size_t blockSizeB, std::vector<Tensor> preQuantizationScalesA,
    std::vector<MatrixAxis> preQuantizationAxesA, std::vector<Tensor> preQuantizationScalesB,
    std::vector<MatrixAxis> preQuantizationAxesB, std::complex<double> outputScale,
    OutputConversion outputConversion, AccumulationRounding accumulationRounding) {
    if (a.shape().rank() != 2 || b.shape().rank() != 2)
        throw std::invalid_argument("Python reference_gemm requires rank-2 A and B tensors.");

    Tensor d(outputType, Shape{a.shape()[0], b.shape()[1]});
    GemmOperand operandA(a);
    GemmOperand operandB(b);
    operandA.computeType = computeTypeA;
    operandB.computeType = computeTypeB;
    auto addPreQuantizationScales = [](GemmOperand& operand, const std::vector<Tensor>& scales,
                                       const std::vector<MatrixAxis>& axes, MatrixAxis defaultAxis,
                                       const char* name) {
        if (!axes.empty() && axes.size() != scales.size())
            throw std::invalid_argument(std::string("Python reference_gemm ") + name +
                                        " scale/axis counts differ.");
        for (size_t index = 0; index < scales.size(); ++index)
            operand.preQuantizationScales.push_back(
                VectorBinding{scales[index], axes.empty() ? defaultAxis : axes[index]});
    };
    addPreQuantizationScales(operandA, preQuantizationScalesA, preQuantizationAxesA,
                             MatrixAxis::Row, "A pre-quantization");
    addPreQuantizationScales(operandB, preQuantizationScalesB, preQuantizationAxesB,
                             MatrixAxis::Column, "B pre-quantization");
    if (blockScaleA || blockScaleB) {
        if (!blockScaleA || !blockScaleB || blockSizeA == 0 || blockSizeB == 0)
            throw std::invalid_argument(
                "Python reference_gemm block scales require both tensors and nonzero sizes.");
        operandA.blockScale = BlockScaleBinding{*blockScaleA, blockSizeA};
        operandB.blockScale = BlockScaleBinding{*blockScaleB, blockSizeB};
    }
    GemmRequest request(std::move(operandA), std::move(operandB), c, d, accumulatorType);
    request.accumulationRounding = accumulationRounding;
    request.mathMode = mathMode;
    request.epilogue.alpha = alpha;
    request.epilogue.beta = beta;
    request.epilogue.outputScale = outputScale;
    request.epilogue.outputConversion = outputConversion;
    request.epilogue.activation = activation;
    request.epilogue.activationParameter0 = activationParameter0;
    request.epilogue.activationParameter1 = activationParameter1;
    request.outputSelection = std::move(outputSelection);
    const GemmExecution execution{
        .backend = backend,
        .requireRequestedBackend = backend == GemmBackend::Blocked,
    };
    GemmRunInfo runInfo =
        referenceGemm(request, execution, pythonGemmBackendImplementation(backend, true)).runInfo;
    return {.output = std::move(d), .runInfo = std::move(runInfo)};
}

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

PythonLayerNormResult referenceLayerNormOwned(const Tensor& input, ScalarType outputType,
                                              ScalarType statisticsType, ScalarType accumulatorType,
                                              size_t axis, double epsilon,
                                              std::optional<Tensor> gamma,
                                              std::optional<Tensor> beta) {
    if (axis >= input.shape().rank())
        throw std::out_of_range("Python reference_layer_norm axis exceeds input rank.");
    std::vector<size_t> statisticsDimensions;
    statisticsDimensions.reserve(input.shape().rank() - 1);
    for (size_t dimension = 0; dimension < input.shape().rank(); ++dimension) {
        if (dimension != axis) statisticsDimensions.push_back(input.shape()[dimension]);
    }

    Tensor output(outputType, input.layout());
    Tensor mean(statisticsType, Shape(statisticsDimensions));
    Tensor inverseVariance(statisticsType, Shape(std::move(statisticsDimensions)));
    LayerNormProblem problem(input, output, axis, accumulatorType);
    problem.mean = mean;
    problem.inverseVariance = inverseVariance;
    if (gamma) problem.gamma = gamma;
    if (beta) problem.beta = beta;
    problem.epsilon = epsilon;
    const LayerNormRunInfo runInfo = referenceLayerNorm(problem);
    return {
        .output = std::move(output),
        .mean = std::move(mean),
        .inverseVariance = std::move(inverseVariance),
        .runInfo = runInfo,
    };
}

PythonEpilogueResult referenceEpilogueOwned(
    const Tensor& input, ScalarType outputType, ScalarType computeType, std::optional<Tensor> bias,
    MatrixAxis biasAxis, Activation activation, ActivationApplication activationApplication,
    std::optional<Tensor> auxiliaryInput, std::optional<ScalarType> auxiliaryOutputType,
    std::optional<Tensor> gateResidual, std::complex<double> outputScale,
    std::complex<double> auxiliaryScale, double activationParameter0, double activationParameter1,
    OutputConversion outputConversion, bool includeRawOutput, bool includeAmax,
    OutputSelection outputSelection) {
    Tensor output(outputType, input.shape());
    std::optional<Tensor> rawOutput;
    std::optional<Tensor> auxiliaryOutput;
    std::optional<Tensor> amax;
    if (includeRawOutput) rawOutput.emplace(computeType, input.shape());
    if (auxiliaryOutputType) auxiliaryOutput.emplace(*auxiliaryOutputType, input.shape());
    if (includeAmax) amax.emplace(computeType, Shape{1});

    EpilogueProblem problem(input, output, computeType);
    if (rawOutput) problem.rawOutput = rawOutput;
    if (auxiliaryOutput) problem.auxiliaryOutput = auxiliaryOutput;
    if (auxiliaryInput) problem.auxiliaryInput = auxiliaryInput;
    if (gateResidual) problem.gateResidual = gateResidual;
    if (amax) problem.amax = amax;
    if (bias) problem.bias = VectorBinding{*bias, biasAxis};
    problem.outputScale = outputScale;
    problem.auxiliaryScale = auxiliaryScale;
    problem.outputConversion = outputConversion;
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
    referenceSum(ReductionProblem(input, output, accumulatorType, std::move(axes)));
    return output;
}

Tensor referenceMaximumAbsoluteOwned(const Tensor& input, ScalarType outputType,
                                     ScalarType accumulatorType) {
    Tensor output(outputType, Shape{});
    referenceMaximumAbsolute(input, output, accumulatorType);
    return output;
}

PythonStructuredSparsityResult applyStructuredSparsityOwned(const Tensor& input,
                                                            StructuredSparsityPattern pattern,
                                                            bool emitTwoOfFourMetadata) {
    if (pattern.axis >= input.shape().rank())
        throw std::out_of_range("Python structured sparsity axis exceeds tensor rank.");
    if (pattern.groupSize == 0)
        throw std::invalid_argument("Python structured sparsity group size must be nonzero.");
    if (input.shape()[pattern.axis] % pattern.groupSize != 0)
        throw std::invalid_argument(
            "Python structured sparsity axis extent must be divisible by group size.");

    std::vector<size_t> compressedDimensions(input.shape().dimensions().begin(),
                                             input.shape().dimensions().end());
    compressedDimensions[pattern.axis] =
        input.shape()[pattern.axis] / pattern.groupSize * pattern.retainedElements;
    const Shape compressedShape(std::move(compressedDimensions));

    Tensor pruned(input.type(), input.shape());
    Tensor compressed(input.type(), compressedShape);
    Tensor retainedIndices(ScalarType::UInt8, compressedShape);
    std::optional<Tensor> twoOfFourMetadata;
    StructuredSparsityProblem problem(input, pruned, compressed, retainedIndices, pattern);
    if (emitTwoOfFourMetadata) {
        if (pattern.groupSize != 4 || pattern.retainedElements != 2)
            throw std::invalid_argument(
                "Python two-of-four metadata output requires a two-of-four pattern.");
        std::vector<size_t> metadataDimensions(input.shape().dimensions().begin(),
                                               input.shape().dimensions().end());
        const size_t sparsityGroups = input.shape()[pattern.axis] / 4;
        metadataDimensions[pattern.axis] = (sparsityGroups + 1) / 2;
        twoOfFourMetadata.emplace(ScalarType::UInt8, Shape(std::move(metadataDimensions)));
        problem.twoOfFourMetadata = twoOfFourMetadata;
    }
    const StructuredSparsityRunInfo runInfo = applyStructuredSparsity(problem);
    return {
        .pruned = std::move(pruned),
        .compressed = std::move(compressed),
        .retainedIndices = std::move(retainedIndices),
        .twoOfFourMetadata = std::move(twoOfFourMetadata),
        .runInfo = runInfo,
    };
}

PythonTwoOfFourMetadataResult encodeTwoOfFourMetadataOwned(const Tensor& retainedIndices,
                                                           size_t axis) {
    if (axis >= retainedIndices.shape().rank())
        throw std::out_of_range("Python two-of-four metadata axis exceeds tensor rank.");
    if (retainedIndices.shape()[axis] % 2 != 0)
        throw std::invalid_argument(
            "Python two-of-four metadata requires two retained indices per sparsity group.");

    std::vector<size_t> metadataDimensions(retainedIndices.shape().dimensions().begin(),
                                           retainedIndices.shape().dimensions().end());
    const size_t sparsityGroups = retainedIndices.shape()[axis] / 2;
    metadataDimensions[axis] = (sparsityGroups + 1) / 2;
    Tensor metadata(ScalarType::UInt8, Shape(std::move(metadataDimensions)));
    const TwoOfFourMetadataRunInfo runInfo =
        encodeTwoOfFourMetadata(TwoOfFourMetadataProblem(retainedIndices, metadata, axis));
    return {.metadata = std::move(metadata), .runInfo = runInfo};
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
        .value("E5M3", ScalarType::E5M3)
        .value("E4M3", ScalarType::E4M3);

    nb::enum_<MathMode>(module, "MathMode")
        .value("Default", MathMode::Default)
        .value("XFloat32", MathMode::XFloat32);

    nb::enum_<AccumulationRounding>(module, "AccumulationRounding")
        .value("TypeDefault", AccumulationRounding::TypeDefault)
        .value("FullPrecision", AccumulationRounding::FullPrecision)
        .value("AfterProductAndSum", AccumulationRounding::AfterProductAndSum);

    nb::enum_<GemmBackend>(module, "GemmBackend")
        .value("Automatic", GemmBackend::Automatic)
        .value("Pointwise", GemmBackend::Pointwise)
        .value("Blocked", GemmBackend::Blocked);

    nb::enum_<OutputConversion>(module, "OutputConversion")
        .value("Default", OutputConversion::Default)
        .value("SaturatingInt8", OutputConversion::SaturatingInt8);

    nb::enum_<MatrixAxis>(module, "MatrixAxis")
        .value("Row", MatrixAxis::Row)
        .value("Column", MatrixAxis::Column);

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

    python_bindings::registerGenerationBindings(module);
    python_bindings::registerMxBindings(module);

    nb::enum_<StructuredSparsitySelection>(module, "StructuredSparsitySelection")
        .value("Fixed", StructuredSparsitySelection::Fixed)
        .value("Random", StructuredSparsitySelection::Random);

    nb::enum_<ActivationApplication>(module, "ActivationApplication")
        .value("Forward", ActivationApplication::Forward)
        .value("Gradient", ActivationApplication::Gradient);

    nb::enum_<ReductionOperation>(module, "ReductionOperation")
        .value("Sum", ReductionOperation::Sum)
        .value("MaximumAbsolute", ReductionOperation::MaximumAbsolute);

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
        .def_prop_ro("element_count", [](const Shape& shape) { return shape.elementCount(); });

    nb::class_<Layout>(module, "Layout")
        .def(nb::init<Shape, std::vector<ptrdiff_t>, ptrdiff_t>(), "shape"_a, "strides"_a,
             "offset"_a = 0)
        .def_static("contiguous", &Layout::contiguous)
        .def_prop_ro("shape", &Layout::shape, nb::rv_policy::reference_internal)
        .def_prop_ro("strides", &strides)
        .def_prop_ro("offset", &Layout::offset);

    nb::class_<Tensor>(module, "Tensor")
        .def(nb::init<ScalarType, Shape>())
        .def_static("from_numpy", &tensorFromNumpy, "array"_a,
                    "scalar_type"_a = std::optional<ScalarType>{})
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
        .def_prop_ro("values", [](const Tensor& tensor) { return tensorValues(tensor); })
        .def("clone", [](const Tensor& tensor) { return tensor.clone(); })
        .def("to", static_cast<Tensor (Tensor::*)(ScalarType) const>(&Tensor::to), "type"_a);

    nb::class_<PythonVectorBinding>(
        module, "VectorBinding", "Owning row- or column-axis tensor binding used by GEMM requests.")
        .def(nb::init<Tensor, MatrixAxis>(), "values"_a, "axis"_a = MatrixAxis::Row)
        .def_rw("values", &PythonVectorBinding::values)
        .def_rw("axis", &PythonVectorBinding::axis);

    nb::class_<PythonBlockScaleBinding>(
        module, "BlockScaleBinding",
        "Owning tensor and reduction-block size used for GEMM block scaling.")
        .def(nb::init<Tensor, size_t>(), "values"_a, "block_size"_a)
        .def_rw("values", &PythonBlockScaleBinding::values)
        .def_rw("block_size", &PythonBlockScaleBinding::blockSize);

    nb::class_<PythonGemmOperand>(
        module, "GemmOperand",
        "Owning GEMM operand, including compute-input quantization and scaling metadata.")
        .def(nb::init<Tensor>(), "values"_a)
        .def_rw("values", &PythonGemmOperand::values)
        .def_rw("compute_type", &PythonGemmOperand::computeType)
        .def_rw("pre_quantization_scales", &PythonGemmOperand::preQuantizationScales)
        .def_rw("block_scale", &PythonGemmOperand::blockScale)
        .def_rw("conjugate", &PythonGemmOperand::conjugate);

    nb::class_<PythonGemmEpilogue>(
        module, "GemmEpilogue",
        "Owning GEMM alpha/beta, vector scaling, activation, and output-conversion settings.")
        .def(nb::init<>())
        .def_rw("alpha", &PythonGemmEpilogue::alpha)
        .def_rw("beta", &PythonGemmEpilogue::beta)
        .def_rw("bias", &PythonGemmEpilogue::bias)
        .def_rw("scale_alpha", &PythonGemmEpilogue::scaleAlpha)
        .def_rw("scale_a", &PythonGemmEpilogue::scaleA)
        .def_rw("scale_b", &PythonGemmEpilogue::scaleB)
        .def_rw("output_scale", &PythonGemmEpilogue::outputScale)
        .def_rw("output_conversion", &PythonGemmEpilogue::outputConversion)
        .def_rw("activation", &PythonGemmEpilogue::activation)
        .def_rw("activation_parameter0", &PythonGemmEpilogue::activationParameter0)
        .def_rw("activation_parameter1", &PythonGemmEpilogue::activationParameter1);

    nb::class_<GemmExecution>(
        module, "GemmExecution",
        "Call-time GEMM backend selection. Python supplies the built-in blocked implementation.")
        .def(nb::init<>())
        .def(nb::init<GemmBackend, bool>(), "backend"_a, "require_requested_backend"_a = false)
        .def_rw("backend", &GemmExecution::backend)
        .def_rw("require_requested_backend", &GemmExecution::requireRequestedBackend);

    nb::class_<PythonGemmRequest>(
        module, "GemmRequest",
        "Owning GEMM numerical request. Each call allocates a fresh output tensor.")
        .def(nb::init<PythonGemmOperand, PythonGemmOperand, std::optional<Tensor>, ScalarType,
                      ScalarType, std::optional<Layout>>(),
             "a"_a, "b"_a, "c"_a = std::optional<Tensor>{}, "output_type"_a = ScalarType::Float32,
             "accumulator_type"_a = ScalarType::Float32,
             "output_layout"_a = std::optional<Layout>{})
        .def_rw("a", &PythonGemmRequest::a)
        .def_rw("b", &PythonGemmRequest::b)
        .def_rw("c", &PythonGemmRequest::c)
        .def_rw("output_type", &PythonGemmRequest::outputType)
        .def_rw("output_layout", &PythonGemmRequest::outputLayout)
        .def_rw("accumulator_type", &PythonGemmRequest::accumulatorType)
        .def_rw("accumulation_rounding", &PythonGemmRequest::accumulationRounding)
        .def_rw("math_mode", &PythonGemmRequest::mathMode)
        .def_rw("epilogue", &PythonGemmRequest::epilogue)
        .def_rw("output_selection", &PythonGemmRequest::outputSelection);

    nb::enum_<UlpComparisonMode>(module, "UlpComparisonMode")
        .value("RelativeSpacing", UlpComparisonMode::RelativeSpacing)
        .value("EncodedDistance", UlpComparisonMode::EncodedDistance);

    nb::enum_<ComplexPointwiseMode>(module, "ComplexPointwiseMode")
        .value("Componentwise", ComplexPointwiseMode::Componentwise)
        .value("Magnitude", ComplexPointwiseMode::Magnitude);

    nb::class_<ComparisonSelection>(module, "ComparisonSelection")
        .def(nb::init<>())
        .def_rw("first", &ComparisonSelection::first)
        .def_rw("stride", &ComparisonSelection::stride)
        .def_rw("max_elements", &ComparisonSelection::maxElements)
        .def_rw("index_order", &ComparisonSelection::indexOrder);

    nb::class_<ComparisonOptions>(module, "ComparisonOptions")
        .def(nb::init<>())
        .def_rw("pointwise", &ComparisonOptions::pointwise)
        .def_rw("absolute_tolerance", &ComparisonOptions::absoluteTolerance)
        .def_rw("relative_tolerance", &ComparisonOptions::relativeTolerance)
        .def_rw("symmetric_relative_tolerance", &ComparisonOptions::symmetricRelativeTolerance)
        .def_rw("strict_tolerance", &ComparisonOptions::strictTolerance)
        .def_rw("equal_nans", &ComparisonOptions::equalNaNs)
        .def_rw("equal_signed_zero", &ComparisonOptions::equalSignedZero)
        .def_rw("complex_pointwise_mode", &ComparisonOptions::complexPointwiseMode)
        .def_rw("compute_pointwise_statistics", &ComparisonOptions::computePointwiseStatistics)
        .def_rw("compute_frobenius", &ComparisonOptions::computeFrobenius)
        .def_rw("compute_ulp", &ComparisonOptions::computeUlp)
        .def_rw("ulp_type", &ComparisonOptions::ulpType)
        .def_rw("ulp_mode", &ComparisonOptions::ulpMode)
        .def_rw("relative_frobenius_tolerance", &ComparisonOptions::relativeFrobeniusTolerance)
        .def_rw("maximum_ulp_tolerance", &ComparisonOptions::maximumUlpTolerance)
        .def_rw("report_matching_elements", &ComparisonOptions::reportMatchingElements)
        .def_rw("max_reported_mismatches", &ComparisonOptions::maxReportedMismatches)
        .def_rw("selection", &ComparisonOptions::selection);

    nb::class_<ComparisonValue>(module, "ComparisonValue")
        .def_ro("real", &ComparisonValue::real)
        .def_ro("imaginary", &ComparisonValue::imaginary)
        .def_ro("complex", &ComparisonValue::complex);

    nb::class_<Mismatch>(module, "Mismatch")
        .def_ro("index", &Mismatch::index)
        .def_ro("coordinates", &Mismatch::coordinates)
        .def_ro("observed_offset", &Mismatch::observedOffset)
        .def_ro("expected_offset", &Mismatch::expectedOffset)
        .def_ro("observed", &Mismatch::observed)
        .def_ro("expected", &Mismatch::expected)
        .def_ro("observed_imaginary", &Mismatch::observedImaginary)
        .def_ro("expected_imaginary", &Mismatch::expectedImaginary)
        .def_ro("absolute_difference", &Mismatch::absoluteDifference)
        .def_ro("tolerance", &Mismatch::tolerance)
        .def_ro("matched", &Mismatch::matched);

    nb::class_<ComparisonResult>(module, "ComparisonResult")
        .def_ro("compared", &ComparisonResult::compared)
        .def_ro("mismatches", &ComparisonResult::mismatches)
        .def_ro("matched_nans", &ComparisonResult::matchedNaNs)
        .def_ro("matched_infinities", &ComparisonResult::matchedInfinities)
        .def_ro("non_finite_mismatches", &ComparisonResult::nonFiniteMismatches)
        .def_ro("signed_zero_mismatches", &ComparisonResult::signedZeroMismatches)
        .def_ro("max_absolute_difference", &ComparisonResult::maxAbsoluteDifference)
        .def_ro("max_relative_difference", &ComparisonResult::maxRelativeDifference)
        .def_ro("max_symmetric_relative_difference",
                &ComparisonResult::maxSymmetricRelativeDifference)
        .def_ro("maximum_observed_magnitude", &ComparisonResult::maximumObservedMagnitude)
        .def_ro("maximum_expected_magnitude", &ComparisonResult::maximumExpectedMagnitude)
        .def_ro("frobenius_difference", &ComparisonResult::frobeniusDifference)
        .def_ro("frobenius_observed", &ComparisonResult::frobeniusObserved)
        .def_ro("frobenius_expected", &ComparisonResult::frobeniusExpected)
        .def_ro("relative_frobenius_error", &ComparisonResult::relativeFrobeniusError)
        .def_ro("maximum_ulp", &ComparisonResult::maximumUlp)
        .def_ro("sum_ulp", &ComparisonResult::sumUlp)
        .def_ro("average_ulp", &ComparisonResult::averageUlp)
        .def_ro("ulp_compared", &ComparisonResult::ulpCompared)
        .def_ro("pointwise_passed", &ComparisonResult::pointwisePassed)
        .def_ro("frobenius_passed", &ComparisonResult::frobeniusPassed)
        .def_ro("ulp_passed", &ComparisonResult::ulpPassed)
        .def_ro("reported_mismatches", &ComparisonResult::reportedMismatches)
        .def_ro("reported_comparisons", &ComparisonResult::reportedComparisons)
        .def_prop_ro("passed", &ComparisonResult::passed);

    nb::class_<ComparisonTolerance>(module, "ComparisonTolerance")
        .def_ro("absolute", &ComparisonTolerance::absolute)
        .def_ro("relative", &ComparisonTolerance::relative);

    nb::enum_<SentinelRegion>(module, "SentinelRegion")
        .value("Unspecified", SentinelRegion::Unspecified)
        .value("Before", SentinelRegion::Before)
        .value("Inside", SentinelRegion::Inside)
        .value("After", SentinelRegion::After);

    nb::class_<SentinelMismatch>(module, "SentinelMismatch")
        .def_ro("region", &SentinelMismatch::region)
        .def_ro("index", &SentinelMismatch::index)
        .def_ro("observed", &SentinelMismatch::observed);

    nb::class_<SentinelResult>(module, "SentinelResult")
        .def_ro("checked", &SentinelResult::checked)
        .def_ro("mismatches", &SentinelResult::mismatches)
        .def_ro("reported_mismatches", &SentinelResult::reportedMismatches)
        .def_prop_ro("passed", &SentinelResult::passed);

    module.attr("ComparisonPlan") = module.attr("ComparisonOptions");
    module.attr("ComparisonReport") = module.attr("ComparisonResult");

    nb::class_<StructuredSparsityPattern>(module, "StructuredSparsityPattern")
        .def(nb::init<>())
        .def_rw("axis", &StructuredSparsityPattern::axis)
        .def_rw("group_size", &StructuredSparsityPattern::groupSize)
        .def_rw("retained_elements", &StructuredSparsityPattern::retainedElements)
        .def_rw("selection", &StructuredSparsityPattern::selection)
        .def_rw("fixed_positions", &StructuredSparsityPattern::fixedPositions)
        .def_rw("seed", &StructuredSparsityPattern::seed)
        .def_rw("index_order", &StructuredSparsityPattern::indexOrder);

    nb::class_<StructuredSparsityRunInfo>(module, "StructuredSparsityRunInfo")
        .def_ro("groups_processed", &StructuredSparsityRunInfo::groupsProcessed)
        .def_ro("input_elements_visited", &StructuredSparsityRunInfo::inputElementsVisited)
        .def_ro("pruned_elements_written", &StructuredSparsityRunInfo::prunedElementsWritten)
        .def_ro("compressed_elements_written",
                &StructuredSparsityRunInfo::compressedElementsWritten)
        .def_ro("retained_indices_written", &StructuredSparsityRunInfo::retainedIndicesWritten)
        .def_ro("metadata_bytes_written", &StructuredSparsityRunInfo::metadataBytesWritten);

    nb::class_<PythonStructuredSparsityResult>(module, "StructuredSparsityResult")
        .def_prop_ro(
            "pruned",
            [](const PythonStructuredSparsityResult& result) -> const Tensor& {
                return result.pruned;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "compressed",
            [](const PythonStructuredSparsityResult& result) -> const Tensor& {
                return result.compressed;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "retained_indices",
            [](const PythonStructuredSparsityResult& result) -> const Tensor& {
                return result.retainedIndices;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "two_of_four_metadata",
            [](const PythonStructuredSparsityResult& result) -> const std::optional<Tensor>& {
                return result.twoOfFourMetadata;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const PythonStructuredSparsityResult& result) -> const StructuredSparsityRunInfo& {
                return result.runInfo;
            },
            nb::rv_policy::reference_internal);

    nb::class_<TwoOfFourMetadataRunInfo>(module, "TwoOfFourMetadataRunInfo")
        .def_ro("sparsity_groups_encoded", &TwoOfFourMetadataRunInfo::sparsityGroupsEncoded)
        .def_ro("metadata_bytes_written", &TwoOfFourMetadataRunInfo::metadataBytesWritten);

    nb::class_<PythonTwoOfFourMetadataResult>(module, "TwoOfFourMetadataResult")
        .def_prop_ro(
            "metadata",
            [](const PythonTwoOfFourMetadataResult& result) -> const Tensor& {
                return result.metadata;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const PythonTwoOfFourMetadataResult& result) -> const TwoOfFourMetadataRunInfo& {
                return result.runInfo;
            },
            nb::rv_policy::reference_internal);

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

    nb::class_<LayerNormRunInfo>(module, "LayerNormRunInfo")
        .def_ro("slices_processed", &LayerNormRunInfo::slicesProcessed)
        .def_ro("output_elements_written", &LayerNormRunInfo::outputElementsWritten)
        .def_ro("mean_elements_written", &LayerNormRunInfo::meanElementsWritten)
        .def_ro("inverse_variance_elements_written",
                &LayerNormRunInfo::inverseVarianceElementsWritten);

    nb::class_<PythonLayerNormResult>(module, "LayerNormResult")
        .def_prop_ro(
            "output",
            [](const PythonLayerNormResult& result) -> const Tensor& { return result.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "mean",
            [](const PythonLayerNormResult& result) -> const Tensor& { return result.mean; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "inverse_variance",
            [](const PythonLayerNormResult& result) -> const Tensor& {
                return result.inverseVariance;
            },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const PythonLayerNormResult& result) -> const LayerNormRunInfo& {
                return result.runInfo;
            },
            nb::rv_policy::reference_internal);

    nb::class_<GemmRunInfo>(module, "GemmRunInfo")
        .def_ro("backend_used", &GemmRunInfo::backendUsed)
        .def_ro("fallback_reason", &GemmRunInfo::fallbackReason)
        .def_ro("output_elements_written", &GemmRunInfo::outputElementsWritten)
        .def_ro("output_elements_covered", &GemmRunInfo::outputElementsCovered);

    nb::class_<PythonGemmResult>(module, "GemmResult")
        .def_prop_ro(
            "output", [](const PythonGemmResult& result) -> const Tensor& { return result.output; },
            nb::rv_policy::reference_internal)
        .def_prop_ro(
            "run_info",
            [](const PythonGemmResult& result) -> const GemmRunInfo& { return result.runInfo; },
            nb::rv_policy::reference_internal);

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
    module.def("apply_structured_sparsity", &applyStructuredSparsityOwned, "input"_a, "pattern"_a,
               "emit_two_of_four_metadata"_a = false);
    module.def("encode_two_of_four_metadata", &encodeTwoOfFourMetadataOwned, "retained_indices"_a,
               "axis"_a);
    module.def(
        "compare",
        [](const Tensor& observed, const Tensor& expected, const ComparisonOptions& options) {
            return compare(observed, expected, options);
        },
        "observed"_a, "expected"_a, "options"_a = ComparisonOptions{});
    module.def(
        "compare",
        [](Tensor observed, Tensor expected, const ComparisonOptions& options) {
            return compare(observed, expected, options);
        },
        "observed"_a, "expected"_a, "options"_a = ComparisonOptions{});
    module.def("default_comparison_options", &defaultComparisonOptions, "type"_a,
               "symmetric_relative_tolerance"_a = std::optional<double>{});
    module.def("near_comparison_options", &nearComparisonOptions, "absolute_tolerance"_a);
    module.def("allclose_comparison_options", &allCloseComparisonOptions,
               "absolute_tolerance"_a = 1e-8, "relative_tolerance"_a = 1e-5,
               "equal_nans"_a = false);
    module.def("ulp_mantissa_bits", &ulpMantissaBits, "type"_a);
    module.def("ulp_distance", &ulpDistance, "exact"_a, "approximation"_a, "mantissa_bits"_a);
    module.def("encoded_ulp_distance", &encodedUlpDistance, "exact"_a, "approximation"_a, "type"_a);
    module.def(
        "find_allclose_tolerance",
        [](const Tensor& observed, const Tensor& expected,
           const std::vector<double>& absoluteCandidates,
           const std::vector<double>& relativeCandidates, const ComparisonOptions& options) {
            return findAllCloseTolerance(observed, expected,
                                         std::span<const double>(absoluteCandidates),
                                         std::span<const double>(relativeCandidates), options);
        },
        "observed"_a, "expected"_a, "absolute_candidates"_a, "relative_candidates"_a,
        "options"_a = allCloseComparisonOptions());
    module.def(
        "find_allclose_tolerance",
        [](Tensor observed, Tensor expected, const std::vector<double>& absoluteCandidates,
           const std::vector<double>& relativeCandidates, const ComparisonOptions& options) {
            return findAllCloseTolerance(observed, expected,
                                         std::span<const double>(absoluteCandidates),
                                         std::span<const double>(relativeCandidates), options);
        },
        "observed"_a, "expected"_a, "absolute_candidates"_a, "relative_candidates"_a,
        "options"_a = allCloseComparisonOptions());
    module.def(
        "check_unwritten_sentinel",
        [](const Tensor& tensor, SentinelRegion region, size_t maxReportedMismatches) {
            return checkUnwrittenSentinel(tensor.type(), tensor.storage(), 0,
                                          tensor.shape().elementCount(), region,
                                          maxReportedMismatches);
        },
        "tensor"_a, "region"_a = SentinelRegion::Unspecified, "max_reported_mismatches"_a = 10);
    module.def(
        "check_unused_tensor_storage",
        [](const Tensor& tensor, size_t allocatedElements, SentinelRegion region,
           size_t maxReportedMismatches) {
            return checkUnusedTensorStorage(tensor, allocatedElements, region,
                                            maxReportedMismatches);
        },
        "tensor"_a, "allocated_elements"_a, "region"_a = SentinelRegion::Inside,
        "max_reported_mismatches"_a = 10);
    module.def(
        "check_unused_tensor_storage",
        [](Tensor tensor, size_t allocatedElements, SentinelRegion region,
           size_t maxReportedMismatches) {
            return checkUnusedTensorStorage(tensor, allocatedElements, region,
                                            maxReportedMismatches);
        },
        "tensor"_a, "allocated_elements"_a, "region"_a = SentinelRegion::Inside,
        "max_reported_mismatches"_a = 10);
    module.def("reference_axpby", &referenceAxpbyOwned, "x"_a = std::optional<Tensor>{},
               "y"_a = std::optional<Tensor>{}, "output_type"_a = ScalarType::Float32,
               "accumulator_type"_a = ScalarType::Float32, "alpha"_a = 1.0, "beta"_a = 1.0);
    module.def("reference_softmax", &referenceSoftmaxOwned, "input"_a,
               "output_type"_a = ScalarType::Float32, "accumulator_type"_a = ScalarType::Float32,
               "axis"_a = 0);
    module.def("reference_layer_norm", &referenceLayerNormOwned, "input"_a,
               "output_type"_a = ScalarType::Float32, "statistics_type"_a = ScalarType::Float32,
               "accumulator_type"_a = ScalarType::Float32, "axis"_a = 0, "epsilon"_a = 1e-5,
               "gamma"_a = std::optional<Tensor>{}, "beta"_a = std::optional<Tensor>{});
    module.def("reference_gemm_result", &referenceGemmOwned, "a"_a, "b"_a, "c"_a, "output_type"_a,
               "accumulator_type"_a, "alpha"_a = 1.0, "beta"_a = 0.0,
               "compute_type_a"_a = std::optional<ScalarType>{},
               "compute_type_b"_a = std::optional<ScalarType>{}, "math_mode"_a = MathMode::Default,
               "activation"_a = Activation::None, "activation_parameter0"_a = 0.0,
               "activation_parameter1"_a = 0.0, "output_selection"_a = OutputSelection::all(),
               "backend"_a = GemmBackend::Pointwise, "block_scale_a"_a = std::optional<Tensor>{},
               "block_scale_b"_a = std::optional<Tensor>{}, "block_size_a"_a = 0,
               "block_size_b"_a = 0, "pre_quantization_scales_a"_a = std::vector<Tensor>{},
               "pre_quantization_axes_a"_a = std::vector<MatrixAxis>{},
               "pre_quantization_scales_b"_a = std::vector<Tensor>{},
               "pre_quantization_axes_b"_a = std::vector<MatrixAxis>{},
               "output_scale"_a = std::complex<double>(1.0, 0.0),
               "output_conversion"_a = OutputConversion::Default,
               "accumulation_rounding"_a = AccumulationRounding::TypeDefault);
    module.def("reference_gemm_result", &referenceGemmRequestOwned, "request"_a,
               "execution"_a = GemmExecution{});
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
    module.def("reference_sum", &referenceSumOwned, "input"_a, "output_type"_a,
               "accumulator_type"_a, "axes"_a);
    module.def("reference_maximum_absolute", &referenceMaximumAbsoluteOwned, "input"_a,
               "output_type"_a, "accumulator_type"_a);
}
