// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
#include <roc/host_validation/backends/tiled.hpp>
#ifdef HOST_VALIDATION_PYTHON_HAS_MX
#include <roc/host_validation/mx.hpp>
#endif
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
void appendTensorValues(nb::list& result, TensorView tensor) {
    using LoadFunction = Value (*)(std::span<const std::byte>, ptrdiff_t);
    const LoadFunction load = visitScalarType(
        tensor.type(), []<typename Tag>() -> LoadFunction { return &loadTensorValue<Value, Tag>; });
    const auto storage = tensor.storage();
    const auto& layout = tensor.layout();
    forEachIndex(tensor.shape(), [&](std::span<const size_t> indices) {
        result.append(load(storage, layout.elementOffset(indices)));
    });
}

nb::list tensorValues(TensorView tensor) {
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

nb::list tensorValues(const Tensor& tensor) {
    return tensorValues(tensor.view());
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
        throw nb::type_error("TensorView.from_numpy requires a native-endian NumPy dtype.");

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
        "TensorView.from_numpy supports only exact native NumPy bool, integer, "
        "float16/32/64, and complex64/128 storage dtypes.");
}

ptrdiff_t checkedElementDelta(ptrdiff_t stride, size_t elementCount) {
    if (stride == 0 || elementCount == 0) return 0;
    if (!std::in_range<ptrdiff_t>(elementCount))
        throw std::overflow_error("NumPy TensorView stride extent exceeds ptrdiff_t.");

    const ptrdiff_t signedElementCount = static_cast<ptrdiff_t>(elementCount);
    if (stride > 0) {
        if (signedElementCount > std::numeric_limits<ptrdiff_t>::max() / stride)
            throw std::overflow_error("NumPy TensorView stride extent overflow.");
    } else if (stride == -1) {
        return -signedElementCount;
    } else if (signedElementCount > std::numeric_limits<ptrdiff_t>::min() / stride) {
        throw std::overflow_error("NumPy TensorView stride extent overflow.");
    }
    return stride * signedElementCount;
}

ptrdiff_t checkedElementOffsetAdd(ptrdiff_t left, ptrdiff_t right) {
    if ((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
        throw std::overflow_error("NumPy TensorView element offset overflow.");
    return left + right;
}

size_t checkedStorageBytes(size_t elementCount, size_t itemSize) {
    if (itemSize != 0 && elementCount > std::numeric_limits<size_t>::max() / itemSize)
        throw std::overflow_error("NumPy TensorView storage byte count overflow.");
    return elementCount * itemSize;
}

TensorView tensorViewFromNumpy(nb::object array, std::optional<ScalarType> requestedType) {
    const nb::object numpyArrayType = nb::module_::import_("numpy").attr("ndarray");
    if (!nb::isinstance(array, numpyArrayType))
        throw nb::type_error("TensorView.from_numpy requires a NumPy ndarray.");

    const NumpyStorageType storageType = numpyStorageType(array);
    if (requestedType && *requestedType != storageType.type)
        throw std::invalid_argument(
            "TensorView.from_numpy scalar_type must exactly match the NumPy storage dtype.");
    const ScalarType type = requestedType.value_or(storageType.type);

    const ReadOnlyPythonBuffer buffer(array);
    const Py_buffer& view = buffer.view();
    if (view.ndim < 0 || (view.ndim > 0 && (!view.shape || !view.strides)))
        throw std::invalid_argument("NumPy TensorView buffer geometry is invalid.");
    if (view.itemsize <= 0 || static_cast<size_t>(view.itemsize) != storageType.itemSize)
        throw std::invalid_argument("NumPy TensorView buffer item size does not match its dtype.");
    if (!std::in_range<ptrdiff_t>(storageType.itemSize))
        throw std::overflow_error("NumPy TensorView item size exceeds ptrdiff_t.");
    const ptrdiff_t signedItemSize = static_cast<ptrdiff_t>(storageType.itemSize);

    std::vector<size_t> tensorDimensions;
    std::vector<ptrdiff_t> tensorStrides;
    tensorDimensions.reserve(static_cast<size_t>(view.ndim));
    tensorStrides.reserve(static_cast<size_t>(view.ndim));
    bool empty = false;
    for (Py_ssize_t dimension = 0; dimension < view.ndim; ++dimension) {
        if (view.shape[dimension] < 0)
            throw std::invalid_argument("NumPy TensorView extent is negative.");
        if (!std::in_range<ptrdiff_t>(view.strides[dimension]))
            throw std::overflow_error("NumPy TensorView byte stride exceeds ptrdiff_t.");
        const size_t extent = static_cast<size_t>(view.shape[dimension]);
        const ptrdiff_t byteStride = static_cast<ptrdiff_t>(view.strides[dimension]);
        if (byteStride % signedItemSize != 0)
            throw std::invalid_argument(
                "NumPy TensorView byte strides must be exact multiples of item size.");
        tensorDimensions.push_back(extent);
        tensorStrides.push_back(byteStride / signedItemSize);
        empty = empty || extent == 0;
    }

    Shape shape(std::move(tensorDimensions));
    if (empty) {
        return TensorView(
            type, Layout(std::move(shape), std::move(tensorStrides)),
            std::span<const std::byte>(reinterpret_cast<const std::byte*>(view.buf), 0));
    }
    if (!view.buf)
        throw std::invalid_argument("Nonempty NumPy TensorView has a null data pointer.");

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
        throw std::overflow_error("NumPy TensorView base offset overflow.");
    const ptrdiff_t normalizedOffset = -lowerOffset;
    const ptrdiff_t normalizedUpper = checkedElementOffsetAdd(upperOffset, normalizedOffset);
    if (!std::in_range<size_t>(normalizedOffset) || !std::in_range<size_t>(normalizedUpper))
        throw std::overflow_error("NumPy TensorView addressed range exceeds size_t.");

    const size_t prefixBytes =
        checkedStorageBytes(static_cast<size_t>(normalizedOffset), storageType.itemSize);
    const size_t normalizedUpperSize = static_cast<size_t>(normalizedUpper);
    if (normalizedUpperSize == std::numeric_limits<size_t>::max())
        throw std::overflow_error("NumPy TensorView addressed range overflow.");
    const size_t storageBytes = checkedStorageBytes(normalizedUpperSize + 1, storageType.itemSize);

    const uintptr_t logicalAddress = reinterpret_cast<uintptr_t>(view.buf);
    if (prefixBytes > logicalAddress)
        throw std::overflow_error("NumPy TensorView base address underflow.");
    const uintptr_t storageAddress = logicalAddress - prefixBytes;
    if (storageBytes > std::numeric_limits<uintptr_t>::max() - storageAddress)
        throw std::overflow_error("NumPy TensorView storage address overflow.");

    return TensorView(type, Layout(std::move(shape), std::move(tensorStrides), normalizedOffset),
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
    GemmOutputConversion outputConversion, AccumulationRounding accumulationRounding) {
    if (a.shape().rank() != 2 || b.shape().rank() != 2)
        throw std::invalid_argument("Python reference_gemm requires rank-2 A and B tensors.");

    Tensor d(outputType, Shape{a.shape()[0], b.shape()[1]});
    GemmOperand operandA(a.view());
    GemmOperand operandB(b.view());
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
                VectorBinding{scales[index].view(), axes.empty() ? defaultAxis : axes[index]});
    };
    addPreQuantizationScales(operandA, preQuantizationScalesA, preQuantizationAxesA,
                             MatrixAxis::Row, "A pre-quantization");
    addPreQuantizationScales(operandB, preQuantizationScalesB, preQuantizationAxesB,
                             MatrixAxis::Column, "B pre-quantization");
    if (blockScaleA || blockScaleB) {
        if (!blockScaleA || !blockScaleB || blockSizeA == 0 || blockSizeB == 0)
            throw std::invalid_argument(
                "Python reference_gemm block scales require both tensors and nonzero sizes.");
        operandA.blockScale = BlockScaleBinding{blockScaleA->view(), blockSizeA};
        operandB.blockScale = BlockScaleBinding{blockScaleB->view(), blockSizeB};
    }
    GemmProblem problem(std::move(operandA), std::move(operandB), c.view(), d.mutableView(),
                        accumulatorType);
    problem.accumulationRounding = accumulationRounding;
    problem.mathMode = mathMode;
    problem.epilogue.alpha = alpha;
    problem.epilogue.beta = beta;
    problem.epilogue.outputScale = outputScale;
    problem.epilogue.outputConversion = outputConversion;
    problem.epilogue.activation = activation;
    problem.epilogue.activationParameter0 = activationParameter0;
    problem.epilogue.activationParameter1 = activationParameter1;
    problem.outputSelection = std::move(outputSelection);
    GemmInvocation invocation(std::move(problem));
    if (backend == GemmBackend::Automatic || backend == GemmBackend::Tiled) {
        static const TiledGemmBackend tiled;
        invocation.execution = {
            .backend = backend,
            .requireRequestedBackend = backend == GemmBackend::Tiled,
            .backendImplementation = &tiled,
        };
    } else if (backend != GemmBackend::Canonical) {
        throw std::invalid_argument("Python reference_gemm exposes Canonical and Tiled backends.");
    }
    GemmRunInfo runInfo = referenceGemm(invocation);
    return {.output = std::move(d), .runInfo = std::move(runInfo)};
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

Tensor referenceMaximumAbsoluteOwned(const Tensor& input, ScalarType outputType,
                                     ScalarType accumulatorType) {
    Tensor output(outputType, Shape{});
    referenceMaximumAbsolute(input.view(), output.mutableView(), accumulatorType);
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
    StructuredSparsityProblem problem(input.view(), pruned.mutableView(), compressed.mutableView(),
                                      retainedIndices.mutableView(), pattern);
    if (emitTwoOfFourMetadata) {
        if (pattern.groupSize != 4 || pattern.retainedElements != 2)
            throw std::invalid_argument(
                "Python two-of-four metadata output requires a two-of-four pattern.");
        std::vector<size_t> metadataDimensions(input.shape().dimensions().begin(),
                                               input.shape().dimensions().end());
        const size_t sparsityGroups = input.shape()[pattern.axis] / 4;
        metadataDimensions[pattern.axis] = (sparsityGroups + 1) / 2;
        twoOfFourMetadata.emplace(ScalarType::UInt8, Shape(std::move(metadataDimensions)));
        problem.twoOfFourMetadata = twoOfFourMetadata->mutableView();
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
    const TwoOfFourMetadataRunInfo runInfo = encodeTwoOfFourMetadata(
        TwoOfFourMetadataProblem(retainedIndices.view(), metadata.mutableView(), axis));
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
        .value("E5M3", ScalarType::E5M3);

    nb::enum_<MathMode>(module, "MathMode")
        .value("Default", MathMode::Default)
        .value("XFloat32", MathMode::XFloat32);

    nb::enum_<AccumulationRounding>(module, "AccumulationRounding")
        .value("TypeDefault", AccumulationRounding::TypeDefault)
        .value("FullPrecision", AccumulationRounding::FullPrecision)
        .value("AfterProductAndSum", AccumulationRounding::AfterProductAndSum);

    nb::enum_<GemmBackend>(module, "GemmBackend")
        .value("Automatic", GemmBackend::Automatic)
        .value("Canonical", GemmBackend::Canonical)
        .value("Tiled", GemmBackend::Tiled);

    nb::enum_<GemmOutputConversion>(module, "GemmOutputConversion")
        .value("Default", GemmOutputConversion::Default)
        .value("SaturatingInt8", GemmOutputConversion::SaturatingInt8);

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
        .value("AbsoluteUniformInteger", GenerationPattern::AbsoluteUniformInteger)
        .value("UniformReal", GenerationPattern::UniformReal)
        .value("Normal", GenerationPattern::Normal)
        .value("Sine", GenerationPattern::Sine)
        .value("Cosine", GenerationPattern::Cosine)
        .value("AbsoluteSine", GenerationPattern::AbsoluteSine)
        .value("AbsoluteCosine", GenerationPattern::AbsoluteCosine)
        .value("SerialIndex", GenerationPattern::SerialIndex)
        .value("SerialDimension", GenerationPattern::SerialDimension)
        .value("Identity", GenerationPattern::Identity)
        .value("CheckerboardUniformInteger", GenerationPattern::CheckerboardUniformInteger)
        .value("TypeMaximum", GenerationPattern::TypeMaximum)
        .value("TypeLowest", GenerationPattern::TypeLowest)
        .value("TypeDenormalMinimum", GenerationPattern::TypeDenormalMinimum)
        .value("TypeDenormalMaximum", GenerationPattern::TypeDenormalMaximum)
        .value("TypeNaN", GenerationPattern::TypeNaN)
        .value("TypeInfinity", GenerationPattern::TypeInfinity)
        .value("TypeNegativeInfinity", GenerationPattern::TypeNegativeInfinity)
        .value("TypeNegativeZero", GenerationPattern::TypeNegativeZero)
        .value("UniformTypeRange", GenerationPattern::UniformTypeRange)
        .value("RandomEncodedExponent", GenerationPattern::RandomEncodedExponent)
        .value("RawConstant", GenerationPattern::RawConstant)
        .value("UniformRawInteger", GenerationPattern::UniformRawInteger)
        .value("RandomRawBits", GenerationPattern::RandomRawBits)
        .value("RawSerialDimension", GenerationPattern::RawSerialDimension);

    nb::enum_<LogicalIndexOrder>(module, "LogicalIndexOrder")
        .value("FirstDimensionFastest", LogicalIndexOrder::FirstDimensionFastest)
        .value("LastDimensionFastest", LogicalIndexOrder::LastDimensionFastest);

    nb::enum_<GenerationTransform>(module, "GenerationTransform")
        .value("Identity", GenerationTransform::None)
        .value("Absolute", GenerationTransform::Absolute)
        .value("Sine", GenerationTransform::Sine)
        .value("Cosine", GenerationTransform::Cosine);

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
        .def_prop_ro("element_count", &Shape::elementCount);

    nb::class_<Layout>(module, "Layout")
        .def(nb::init<Shape, std::vector<ptrdiff_t>, ptrdiff_t>(), "shape"_a, "strides"_a,
             "offset"_a = 0)
        .def_static("contiguous", &Layout::contiguous)
        .def_prop_ro("shape", &Layout::shape, nb::rv_policy::reference_internal)
        .def_prop_ro("strides", &strides)
        .def_prop_ro("offset", &Layout::offset);

    nb::class_<TensorView>(module, "TensorView")
        .def_static("from_numpy", &tensorViewFromNumpy, "array"_a,
                    "scalar_type"_a = std::optional<ScalarType>{}, nb::keep_alive<0, 1>())
        .def_prop_ro("type", &TensorView::type)
        .def_prop_ro("shape", [](const TensorView& tensor) { return dimensions(tensor.shape()); })
        .def_prop_ro("strides", [](const TensorView& tensor) { return strides(tensor.layout()); })
        .def_prop_ro("offset", [](const TensorView& tensor) { return tensor.layout().offset(); })
        .def_prop_ro("size", [](const TensorView& tensor) { return tensor.shape().elementCount(); })
        .def_prop_ro("storage",
                     [](const TensorView& tensor) {
                         const auto storage = tensor.storage();
                         return nb::bytes(reinterpret_cast<const char*>(storage.data()),
                                          storage.size());
                     })
        .def_prop_ro("values", [](const TensorView& tensor) { return tensorValues(tensor); });

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
        .def_prop_ro("values", [](const Tensor& tensor) { return tensorValues(tensor); })
        .def("view", &Tensor::view, nb::keep_alive<0, 1>());

    nb::enum_<ComparisonIndexOrder>(module, "ComparisonIndexOrder")
        .value("FirstDimensionFastest", ComparisonIndexOrder::FirstDimensionFastest)
        .value("LastDimensionFastest", ComparisonIndexOrder::LastDimensionFastest);

    nb::enum_<UlpComparisonMode>(module, "UlpComparisonMode")
        .value("RelativeSpacing", UlpComparisonMode::RelativeSpacing)
        .value("EncodedDistance", UlpComparisonMode::EncodedDistance);

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

    nb::class_<GenerationPatternSpec>(module, "GenerationPatternSpec")
        .def(nb::init<>())
        .def_rw("pattern", &GenerationPatternSpec::pattern)
        .def_rw("parameter0", &GenerationPatternSpec::parameter0)
        .def_rw("parameter1", &GenerationPatternSpec::parameter1)
        .def_rw("value_scale", &GenerationPatternSpec::valueScale)
        .def_rw("value_offset", &GenerationPatternSpec::valueOffset)
        .def_rw("stream", &GenerationPatternSpec::stream)
        .def_rw("dimension", &GenerationPatternSpec::dimension)
        .def_rw("source_type", &GenerationPatternSpec::sourceType)
        .def_rw("transform", &GenerationPatternSpec::transform)
        .def_rw("alternating_dimensions", &GenerationPatternSpec::alternatingDimensions)
        .def_rw("negative_parity", &GenerationPatternSpec::negativeParity);

    nb::class_<GenerationOptions>(module, "GenerationOptions")
        .def(nb::init<>())
        .def_rw("seed", &GenerationOptions::seed)
        .def_rw("index_order", &GenerationOptions::indexOrder)
        .def_rw("real", &GenerationOptions::real)
        .def_rw("imaginary", &GenerationOptions::imaginary);

#ifdef HOST_VALIDATION_PYTHON_HAS_MX
    nb::enum_<MxGenerationMode>(module, "MxGenerationMode")
        .value("Bounded", MxGenerationMode::Bounded)
        .value("BoundedAlternatingSign", MxGenerationMode::BoundedAlternatingSign)
        .value("Unbounded", MxGenerationMode::Unbounded)
        .value("Identity", MxGenerationMode::Identity)
        .value("Ones", MxGenerationMode::Ones)
        .value("Zeros", MxGenerationMode::Zeros)
        .value("Sequential", MxGenerationMode::Sequential)
        .value("RowIndex", MxGenerationMode::RowIndex)
        .value("ColumnIndex", MxGenerationMode::ColumnIndex)
        .value("Checkerboard", MxGenerationMode::Checkerboard)
        .value("ScaledDiagonal", MxGenerationMode::ScaledDiagonal)
        .value("Twos", MxGenerationMode::Twos)
        .value("NegativeOnes", MxGenerationMode::NegativeOnes)
        .value("Maximum", MxGenerationMode::Maximum)
        .value("DenormalMinimum", MxGenerationMode::DenormalMinimum)
        .value("DenormalMaximum", MxGenerationMode::DenormalMaximum)
        .value("NaN", MxGenerationMode::NaN)
        .value("Infinity", MxGenerationMode::Infinity)
        .value("Trigonometric", MxGenerationMode::Trigonometric)
        .value("Normal", MxGenerationMode::Normal)
        .value("UniformInteger", MxGenerationMode::UniformInteger);

    nb::class_<MxGenerationRecipe>(module, "MxGenerationRecipe")
        .def(nb::init<>())
        .def_rw("mode", &MxGenerationRecipe::mode)
        .def_rw("parameter0", &MxGenerationRecipe::parameter0)
        .def_rw("parameter1", &MxGenerationRecipe::parameter1);

    nb::class_<MxGenerationProblem>(module, "MxGenerationProblem")
        .def(nb::init<>())
        .def_rw("data_type", &MxGenerationProblem::dataType)
        .def_rw("scale_type", &MxGenerationProblem::scaleType)
        .def_rw("shape", &MxGenerationProblem::shape)
        .def_rw("leading_dimension", &MxGenerationProblem::leadingDimension)
        .def_rw("block_axis", &MxGenerationProblem::blockAxis)
        .def_rw("block_size", &MxGenerationProblem::blockSize)
        .def_rw("data", &MxGenerationProblem::data)
        .def_rw("scale", &MxGenerationProblem::scale)
        .def_rw("seed", &MxGenerationProblem::seed);

    nb::class_<MxGenerationResult>(module, "MxGenerationResult")
        .def_ro("data", &MxGenerationResult::data)
        .def_ro("scales", &MxGenerationResult::scales)
        .def_ro("scale_indices", &MxGenerationResult::scaleIndices)
        .def_ro("reference", &MxGenerationResult::reference);

    module.def("generate_mx", &generateMx, "problem"_a);
#endif

    nb::class_<StructuredSparsityPattern>(module, "StructuredSparsityPattern")
        .def(nb::init<>())
        .def_rw("axis", &StructuredSparsityPattern::axis)
        .def_rw("group_size", &StructuredSparsityPattern::groupSize)
        .def_rw("retained_elements", &StructuredSparsityPattern::retainedElements)
        .def_rw("selection", &StructuredSparsityPattern::selection)
        .def_rw("fixed_positions", &StructuredSparsityPattern::fixedPositions)
        .def_rw("seed", &StructuredSparsityPattern::seed)
        .def_rw("stream", &StructuredSparsityPattern::stream)
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

    nb::class_<GemmRunInfo>(module, "GemmRunInfo")
        .def_ro("backend_used", &GemmRunInfo::backendUsed)
        .def_ro("fallback_reason", &GemmRunInfo::fallbackReason)
        .def_ro("output_elements_computed", &GemmRunInfo::outputElementsComputed);

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
    module.def("apply_structured_sparsity", &applyStructuredSparsityOwned, "input"_a, "pattern"_a,
               "emit_two_of_four_metadata"_a = false);
    module.def("encode_two_of_four_metadata", &encodeTwoOfFourMetadataOwned, "retained_indices"_a,
               "axis"_a);
    module.def(
        "compare",
        [](const Tensor& observed, const Tensor& expected, const ComparisonOptions& options) {
            return compare(observed.view(), expected.view(), options);
        },
        "observed"_a, "expected"_a, "options"_a = ComparisonOptions{});
    module.def(
        "compare",
        [](TensorView observed, TensorView expected, const ComparisonOptions& options) {
            return compare(observed, expected, options);
        },
        "observed"_a, "expected"_a, "options"_a = ComparisonOptions{});
    module.def("default_comparison_options", &defaultComparisonOptions, "type"_a,
               "symmetric_relative_tolerance"_a = std::optional<double>{});
    module.def("near_comparison_options", &nearComparisonOptions, "absolute_tolerance"_a);
    module.def("allclose_comparison_options", &allCloseComparisonOptions, "absolute_tolerance"_a,
               "relative_tolerance"_a, "equal_nans"_a = false);
    module.def("ulp_mantissa_bits", &ulpMantissaBits, "type"_a);
    module.def("ulp_distance", &ulpDistance, "exact"_a, "approximation"_a, "mantissa_bits"_a);
    module.def("encoded_ulp_distance", &encodedUlpDistance, "exact"_a, "approximation"_a, "type"_a);
    module.def(
        "find_allclose_tolerance",
        [](const Tensor& observed, const Tensor& expected,
           const std::vector<double>& absoluteCandidates,
           const std::vector<double>& relativeCandidates, const ComparisonOptions& options) {
            return findAllCloseTolerance(observed.view(), expected.view(),
                                         std::span<const double>(absoluteCandidates),
                                         std::span<const double>(relativeCandidates), options);
        },
        "observed"_a, "expected"_a, "absolute_candidates"_a, "relative_candidates"_a,
        "options"_a = ComparisonOptions{});
    module.def(
        "find_allclose_tolerance",
        [](TensorView observed, TensorView expected, const std::vector<double>& absoluteCandidates,
           const std::vector<double>& relativeCandidates, const ComparisonOptions& options) {
            return findAllCloseTolerance(observed, expected,
                                         std::span<const double>(absoluteCandidates),
                                         std::span<const double>(relativeCandidates), options);
        },
        "observed"_a, "expected"_a, "absolute_candidates"_a, "relative_candidates"_a,
        "options"_a = ComparisonOptions{});
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
            return checkUnusedTensorStorage(tensor.view(), allocatedElements, region,
                                            maxReportedMismatches);
        },
        "tensor"_a, "allocated_elements"_a, "region"_a = SentinelRegion::Inside,
        "max_reported_mismatches"_a = 10);
    module.def(
        "check_unused_tensor_storage",
        [](TensorView tensor, size_t allocatedElements, SentinelRegion region,
           size_t maxReportedMismatches) {
            return checkUnusedTensorStorage(tensor, allocatedElements, region,
                                            maxReportedMismatches);
        },
        "tensor"_a, "allocated_elements"_a, "region"_a = SentinelRegion::Inside,
        "max_reported_mismatches"_a = 10);
    module.def("reference_gemm_result", &referenceGemmOwned, "a"_a, "b"_a, "c"_a, "output_type"_a,
               "accumulator_type"_a, "alpha"_a = 1.0, "beta"_a = 0.0,
               "compute_type_a"_a = std::optional<ScalarType>{},
               "compute_type_b"_a = std::optional<ScalarType>{}, "math_mode"_a = MathMode::Default,
               "activation"_a = Activation::None, "activation_parameter0"_a = 0.0,
               "activation_parameter1"_a = 0.0, "output_selection"_a = OutputSelection::all(),
               "backend"_a = GemmBackend::Canonical, "block_scale_a"_a = std::optional<Tensor>{},
               "block_scale_b"_a = std::optional<Tensor>{}, "block_size_a"_a = 0,
               "block_size_b"_a = 0, "pre_quantization_scales_a"_a = std::vector<Tensor>{},
               "pre_quantization_axes_a"_a = std::vector<MatrixAxis>{},
               "pre_quantization_scales_b"_a = std::vector<Tensor>{},
               "pre_quantization_axes_b"_a = std::vector<MatrixAxis>{},
               "output_scale"_a = std::complex<double>(1.0, 0.0),
               "output_conversion"_a = GemmOutputConversion::Default,
               "accumulation_rounding"_a = AccumulationRounding::TypeDefault);
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
    module.def("reference_maximum_absolute", &referenceMaximumAbsoluteOwned, "input"_a,
               "output_type"_a, "accumulator_type"_a);
}
